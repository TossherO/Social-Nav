"""Single CPU reference worker using bounded sealed arrays instead of scene pickles."""
import multiprocessing as mp
from multiprocessing.reduction import DupFd
import time
import signal
from navigation.planning.runtime import Worker,query_time
from .buffers import SharedArrayReader
from .scenes import ScenePool,read_scene


def _run(connection,handle,owner,slots,slot_bytes,planning,geometry):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    from navigation.planning.planner import Planner
    from navigation.planning.continuity import reference_envelope
    from perception.environment.storage import StaticEnvironmentMap
    reader=SharedArrayReader(handle.detach(),owner,slots,slot_bytes)
    planner=Planner(planning,geometry,require_actual_start=True);asset=None;epoch=None;scene=None
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
                if job['epoch']!=epoch:planner.reset();epoch=job['epoch']
                now=query_time(job)
                result=planner.query(job['request'],scene=scene,asset=asset,now=now,clock_domain=scene.record['clock_domain'])
                result=reference_envelope(result,job['request'],scene,asset)
                response=dict(job_id=job['job_id'],result=result,error=None,compute_s=time.monotonic()-began)
            except Exception as error:
                response=dict(job_id=job['job_id'],result=None,error=type(error).__name__+': '+str(error),compute_s=time.monotonic()-began)
            # PoseGrid retains a snapshot; release it before acknowledging the
            # shared lease. The persistent native backend owns its cache arrays.
            planner._pose_grid=None;planner._pose_binding=None;scene=None
            response['cpu_s']=time.process_time()-cpu_start
            connection.send(response)
    except (EOFError,BrokenPipeError):pass
    finally:
        planner=None;scene=None;reader.close();connection.close()


class SharedReferenceWorker(Worker):
    def __init__(self,role,planning,motion,geometry,*,scene_factory=None,metrics=None):
        if role!='reference':raise ValueError('Shared worker supports the single reference role')
        self.planning,self.geometry=planning,geometry;self.role=role;self.job=None;self.description=None;self.scene_factory=scene_factory;self.metrics=metrics
        self.launch()

    def launch(self):
        self.scenes=ScenePool();pool=self.scenes.pool
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=_run,args=(child,DupFd(pool.fd),pool.owner,pool.slots,pool.slot_bytes,self.planning,self.geometry),daemon=True)
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
        result=super().poll()
        if result is not None:
            self.scenes.release(self.description,'reference');self.description=None
            if self.metrics is not None:self.metrics.add('reference_query_cpu',result[1]['cpu_s'])
        return result

    def stop(self):
        super().stop()
        if self.description is not None:self.scenes.release(self.description,'reference');self.description=None
        self.scenes.close()
