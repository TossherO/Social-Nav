"""Bounded reference worker shared by direct runtime and standalone planning."""
import multiprocessing as mp
from multiprocessing.reduction import DupFd
import signal
import time
import numpy as np
from interfaces.shared import SharedArrayReader
from perception.environment.shared_scene import ScenePool,read_scene
from .continuity import reference_envelope


def query_time(job,mono=None):
    """Advance a frozen source-clock anchor at its own replay rate."""
    rate=job.get('clock_rate',1.)
    if not np.isfinite(rate) or rate<=0:raise ValueError('Invalid planning source clock rate')
    return job['now']+((time.monotonic() if mono is None else mono)-job['sent_mono'])*rate


class ReferenceQuery:
    """Own reference-search state independently of the worker's transport."""
    def __init__(self,planning,geometry):
        from .planner import Planner
        self.planner=Planner(planning,geometry,require_actual_start=True)
        self.epoch=None

    def query(self,job,scene,asset):
        if job['epoch']!=self.epoch:
            self.planner.reset();self.epoch=job['epoch']
        result=self.planner.query(job['request'],scene=scene,asset=asset,
            now=query_time(job),clock_domain=scene.record['clock_domain'])
        return reference_envelope(result,job['request'],scene,asset)

    def release_scene(self):
        """Drop planner views before the caller releases shared scene storage."""
        self.planner._pose_grid=None;self.planner._pose_binding=None


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
                    if asset is None or str(asset.directory)!=packet['directory']:
                        asset=StaticEnvironmentMap(packet['directory'])
                    if asset.manifest!=packet['manifest']:raise ValueError('Static asset metadata changed')
                else:asset=None
                result=query.query(job,scene,asset)
                response=dict(job_id=job['job_id'],result=result,error=None,compute_s=time.monotonic()-began)
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
        if asset is not None:wire['asset']=dict(directory=str(asset.directory),manifest=asset.manifest)
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
