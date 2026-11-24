"""Bounded reference worker shared by direct runtime and standalone planning."""
import multiprocessing as mp
from multiprocessing.reduction import DupFd
import signal
import time
import zlib
import numpy as np
from interfaces.shared import SharedArrayReader
from perception.environment.shared_scene import ScenePool,read_scene
from .continuity import goal_binding,reference_envelope


def query_time(job,mono=None):
    """Advance a frozen source-clock anchor at its own replay rate."""
    rate=job.get('clock_rate',1.)
    if not np.isfinite(rate) or rate<=0:raise ValueError('Invalid planning source clock rate')
    return job['now']+((time.monotonic() if mono is None else mono)-job['sent_mono'])*rate


def encode_display_map(grid,local_from_map,job_id,envelope):
    """Pack one mapped search raster independently of its reference envelope."""
    metadata=grid.metadata
    try:
        cells=np.asarray(grid.layers['occupancy'],dtype=np.int8,order='C')
        conditional=grid.layers.get('conditional_obstacle')
        compressed_cells=zlib.compress(cells,level=1)
        compressed_conditional=None if conditional is None else zlib.compress(np.packbits(conditional),level=1)
    except (MemoryError,zlib.error):
        return None
    result=envelope['result']
    return dict(job_id=job_id,stamp_s=envelope['as_of'],
        frame_id=metadata['frame_id'],shape=cells.shape,resolution=float(metadata['resolution']),
        origin_xy=tuple(map(float,metadata['origin_xy'])),snapshot_id=metadata['snapshot_id'],
        status=result['status'],reason=result['reason'],result_binding=result['result_binding'],
        reference_binding=envelope['binding'],goal_binding=goal_binding(result['request']),
        local_from_map=tuple(tuple(row) for row in local_from_map.tolist()),
        cells=compressed_cells,conditional=compressed_conditional)


def decode_display_map(snapshot):
    """Read the occupancy and optional conditional layer from a reference display packet."""
    shape=snapshot['shape']
    cells=np.frombuffer(zlib.decompress(snapshot['cells']),dtype=np.int8).reshape(shape)
    packed=snapshot['conditional']
    if packed is None:
        return cells,None
    bits=np.frombuffer(zlib.decompress(packed),dtype=np.uint8)
    conditional=np.unpackbits(bits,count=cells.size).view(np.bool_).reshape(shape)
    conditional.flags.writeable=False
    return cells,conditional


class ReferenceQuery:
    """Own reference-search state independently of the worker's transport."""
    def __init__(self,planning,geometry):
        from .planner import Planner
        self.planner=Planner(planning,geometry,require_actual_start=True)
        self.epoch=None
        self.display_map=None

    def query(self,job,scene,asset):
        self.display_map=None
        if job['epoch']!=self.epoch:
            self.planner.reset();self.epoch=job['epoch']
        captured=[]
        def capture(grid,local_from_map):
            captured.append((grid,local_from_map))
        result=self.planner.query(job['request'],scene=scene,asset=asset,
            now=query_time(job),clock_domain=scene.record['clock_domain'],
            reference_obstacles=job.get('reference_obstacles',()),
            on_mapped_grid=capture if job.get('display_enabled') else None)
        envelope=reference_envelope(result,job['request'],scene,asset)
        if captured:
            grid,local_from_map=captured[0]
            self.display_map=encode_display_map(grid,local_from_map,job['job_id'],envelope)
        return envelope

    def release_scene(self):
        """Drop planner views before the caller releases shared scene storage."""
        self.planner._pose_grid=None;self.planner._pose_binding=None
        self.display_map=None


def _worker(connection,handle,owner,slots,slot_bytes,planning,geometry):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    from perception.environment.storage import StaticEnvironmentMap
    reader=SharedArrayReader(handle.detach(),owner,slots,slot_bytes)
    query=ReferenceQuery(planning,geometry);asset=None;scene=None
    connection.send(dict(ready=True))
    try:
        while True:
            job=connection.recv()
            if job is None:break
            began=time.monotonic();cpu_start=time.process_time();scene=None
            try:
                scene=read_scene(reader,job.pop('shared_scene'))
                packet=job['asset']
                if packet is not None:
                    if (asset is None or str(asset.directory)!=packet['directory'] or
                            asset.manifest!=packet['manifest'] or asset.navigation_window!=packet['navigation_window']):
                        asset=StaticEnvironmentMap(packet['directory'],navigation_window=packet['navigation_window'])
                    if asset.manifest!=packet['manifest']:raise ValueError('Static asset metadata changed')
                else:asset=None
                result=query.query(job,scene,asset)
                response=dict(job_id=job['job_id'],result=result,error=None,compute_s=time.monotonic()-began)
                if query.display_map is not None:response['display_map']=query.display_map
            except Exception as error:
                response=dict(job_id=job['job_id'],result=None,error=type(error).__name__+': '+str(error),compute_s=time.monotonic()-began)
            # The planner must release its view before the parent reuses the slot.
            query.release_scene();scene=None
            response['cpu_s']=time.process_time()-cpu_start
            connection.send(response)
    except (EOFError,BrokenPipeError):pass
    finally:
        query=None;scene=None;reader.close();connection.close()


class Worker:
    def __init__(self,planning,geometry,*,scene_factory=None,metrics=None):
        self.planning,self.geometry=planning,geometry;self.job=None;self.description=None
        self.scene_factory=scene_factory;self.metrics=metrics
        self.launch()

    def launch(self):
        self.scenes=ScenePool();pool=self.scenes.pool
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=_worker,args=(child,DupFd(pool.fd),pool.owner,pool.slots,pool.slot_bytes,self.planning,self.geometry),daemon=True)
        self.process.start();child.close();self.ready=False;self.started=time.monotonic()

    def send(self,job):
        if not self.ready or self.job is not None:raise RuntimeError('Reference worker is not idle')
        if self.scene_factory is not None:job=dict(job,scene=self.scene_factory(job['scene']))
        wire={k:v for k,v in job.items() if k!='scene'};asset=job['asset']
        if asset is not None:
            wire['asset']=dict(directory=str(asset.directory),manifest=asset.manifest,
                               navigation_window=asset.navigation_window)
        self.description=self.scenes.publish(job['scene'],'reference');wire['shared_scene']=self.description
        try:self.connection.send(wire)
        except Exception:
            self.scenes.release(self.description,'reference');self.description=None;raise
        self.job=job

    def poll(self):
        if not self.connection.poll():return None
        result=self.connection.recv()
        if result.get('ready'):self.ready=True;return None
        job=self.job;self.job=None
        if job is None or result['job_id']!=job['job_id']:raise RuntimeError('Worker result identity mismatch')
        self.scenes.release(self.description,'reference');self.description=None
        if self.metrics is not None:self.metrics.add('reference_query_cpu',result['cpu_s'])
        return job,result

    def stop(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=.5)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=.5)
        self.connection.close();self.job=None
        if self.description is not None:self.scenes.release(self.description,'reference');self.description=None
        self.scenes.close()

    def restart(self):
        self.stop();self.launch()
