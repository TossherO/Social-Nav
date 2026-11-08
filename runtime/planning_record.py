"""Optional bounded planning evidence; retains sealed observations, never waits on disk."""
import json
from pathlib import Path
import queue
import threading


class PlanningRecorder:
    def __init__(self,directory,*,metrics=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.metrics=metrics;self.queue=queue.Queue(8);self.error=None
        self.written=0;self.dropped=0;self.scenes=set();self.references=set();self.jobs=set();self.replies=set()
        self.next_capture=float('-inf');self.worker=threading.Thread(target=self._work,name='planning-record',daemon=True)
        self.worker.start()

    def submit(self,packet,scene,device,reference,trajectory,job,reply,*,dt,timing,capture_required=False):
        if self.error:return False
        periodic=scene is not None and packet['stamp_s']>=self.next_capture
        capture=scene is not None and (periodic or capture_required)
        item=(packet,scene,device if capture else None,reference,trajectory,job,reply,dt,dict(timing),capture)
        try:self.queue.put_nowait(item)
        except queue.Full:self.dropped+=1;return False
        if periodic:self.next_capture=packet['stamp_s']+.5
        return True

    def _save_scene(self,scene):
        name=scene.record['input_fingerprint']
        if name not in self.scenes:
            scene.save(self.directory/'scenes'/name);self.scenes.add(name)
        return name

    def _json(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(value,stream,allow_nan=False,separators=(',',':'))

    def _work(self):
        stream=None
        try:
            with (self.directory/'cycles.jsonl').open('x') as output:
                while True:
                    item=self.queue.get()
                    if item is None:return
                    packet,scene,device,reference,trajectory,job,reply,dt,timing,capture=item
                    row=dict(sequence_id=packet['sequence_id'],stamp_s=packet['stamp_s'],dt=dt,
                        packet_binding=packet['packet_binding'],scene_binding=packet.get('scene_binding'),
                        reference_binding=None if reference is None else reference['binding'],
                        trajectory=trajectory,timing=timing,saved_scene=None)
                    if reference is not None and reference['binding'] not in self.references:
                        self._json(self.directory/'references'/(reference['binding']+'.json'),reference)
                        self.references.add(reference['binding'])
                    if capture:
                        name=self._save_scene(scene);row['saved_scene']=name
                        if device is not None:
                            if device.metadata['snapshot_id']!=scene.geometry.metadata['snapshot_id']:
                                raise ValueError('Planning record device/scene mismatch')
                            path=self.directory/'scenes'/name/'display_layers.npz'
                            if not path.exists():
                                import torch
                                if stream is None:stream=torch.cuda.Stream(device=next(iter(device._layers.values())).device)
                                with torch.cuda.stream(stream):
                                    grid=device.materialize(consumer='planning_record',metrics=self.metrics)
                                grid.save(path)
                    if job is not None and job['job_id'] not in self.jobs:
                        name=self._save_scene(job['scene'])
                        fields={k:job[k] for k in ('job_id','epoch','request','now','clock_rate','sent_mono')}
                        self._json(self.directory/'queries'/(str(job['job_id'])+'.json'),dict(fields,scene_binding=name))
                        self.jobs.add(job['job_id'])
                    if reply is not None and reply[0]['job_id'] not in self.replies:
                        original,response=reply;name=self._save_scene(original['scene'])
                        self._json(self.directory/'replies'/(str(original['job_id'])+'.json'),dict(response,scene_binding=name))
                        self.replies.add(original['job_id'])
                    output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');self.written+=1
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        while self.worker.is_alive():
            try:self.queue.put(None,timeout=.1);break
            except queue.Full:continue
        self.worker.join(timeout=20.)
        if self.worker.is_alive():raise RuntimeError('Planning recorder failed to drain')
        return dict(written=self.written,dropped=self.dropped,scenes=len(self.scenes),queries=len(self.jobs),
            replies=len(self.replies),error=self.error,complete=self.error is None and self.dropped==0)
