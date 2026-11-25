"""Bounded optional scene export; rendering/storage never owns the control loop."""
from pathlib import Path
import queue
import threading
import time


def drain_worker(worker, inbox, *, timeout=20.):
    """Share the shutdown deadline between enqueueing the sentinel and joining."""
    deadline = time.monotonic() + timeout
    while worker.is_alive():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            inbox.put(None, timeout=min(.1, remaining))
            break
        except queue.Full:
            continue
    worker.join(timeout=max(0., deadline - time.monotonic()))
    if worker.is_alive():
        raise RuntimeError(f'{worker.name} failed to drain within {timeout:g}s')


def materialize_scene(snapshot, scene, *, consumer, metrics=None):
    """Complete the archive with device layers from the same sealed map."""
    host = scene.geometry
    if any(host.metadata.get(key) != value for key, value in snapshot.metadata.items()):
        raise ValueError('Scene export/device observation mismatch')
    fields = [name for name in snapshot._layers if name not in host.layers]
    if not fields:
        return scene
    additional = snapshot.materialize(consumer=consumer, fields=fields, metrics=metrics)
    return scene.with_additional_geometry_layers(additional.layers)


class SceneExporter:
    def __init__(self,directory,settings,*,metrics=None,prior_directory=None):
        self.directory=Path(directory);self.settings=settings;self.metrics=metrics
        self.prior_directory=prior_directory;self.queue=queue.Queue(2)
        self.error=None;self.dropped=0;self.written=0
        self.worker=threading.Thread(target=self._work,name='scene-export',daemon=True);self.worker.start()

    def submit(self,snapshot,scene,*,rgb=None,name=None):
        if self.error:raise RuntimeError('Scene export failed: '+self.error)
        item=(snapshot,scene,rgb,name or snapshot.owner)
        try:self.queue.put_nowait(item);return True
        except queue.Full:self.dropped+=1;return False

    def _work(self):
        try:
            import torch
            from perception.environment.prior import StaticPriorView
            prior=None if self.prior_directory is None else StaticPriorView(self.prior_directory)
            stream=None
            while True:
                item=self.queue.get()
                if item is None:return
                snapshot,scene,rgb,name=item
                if rgb is not None:scene=scene.with_rgb(rgb,self.settings)
                if stream is None:stream=torch.cuda.Stream(device=next(iter(snapshot._layers.values())).device,priority=0)
                with torch.cuda.stream(stream):
                    bundle=materialize_scene(snapshot,scene,consumer='explicit_scene_export',metrics=self.metrics)
                if prior is not None:bundle=prior.attach(bundle,self.settings)
                bundle.save(self.directory/name);self.written+=1
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        drain_worker(self.worker, self.queue)
        if self.error:raise RuntimeError('Scene export failed: '+self.error)
        return dict(written=self.written,dropped=self.dropped,complete=self.dropped==0)
