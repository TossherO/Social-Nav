"""Owned, bounded Hybrid A* reference worker; no robot commands."""
import multiprocessing as mp
import time
import numpy as np
from .continuity import reference_envelope


def query_time(job,mono=None):
    """Advance a frozen source-clock anchor at its own replay rate."""
    rate=job.get('clock_rate',1.)
    if not np.isfinite(rate) or rate<=0:raise ValueError('Invalid planning source clock rate')
    return job['now']+((time.monotonic() if mono is None else mono)-job['sent_mono'])*rate


def _worker(connection,role,planning,motion,geometry):
    if role!='reference':raise ValueError('Only the reference worker is supported')
    from .planner import Planner
    from perception.environment.storage import StaticEnvironmentMap
    planner=Planner(planning,geometry,require_actual_start=True)
    epoch=None;cached_asset=None
    connection.send(dict(ready=True))
    try:
        while True:
            job=connection.recv()
            if job is None:return
            began=time.monotonic()
            try:
                packet=job['asset']
                if packet is not None:
                    if (cached_asset is None or str(cached_asset.directory)!=packet['directory'] or
                            cached_asset.manifest['layers_sha256']!=packet['manifest']['layers_sha256']):
                        cached_asset=StaticEnvironmentMap(packet['directory'])
                    if cached_asset.manifest!=packet['manifest']:raise ValueError('Static asset metadata changed')
                    job['asset']=cached_asset
                if epoch!=job['epoch']:planner.reset();epoch=job['epoch']
                result=planner.query(job['request'],scene=job['scene'],asset=job['asset'],
                    now=query_time(job),clock_domain=job['scene'].record['clock_domain'])
                result=reference_envelope(result,job['request'],job['scene'],job['asset'])
                response=dict(job_id=job['job_id'],result=result,error=None,compute_s=time.monotonic()-began)
            except Exception as error:
                response=dict(job_id=job['job_id'],result=None,error=type(error).__name__+': '+str(error),
                    compute_s=time.monotonic()-began)
            connection.send(response)
    except (EOFError,BrokenPipeError):pass
    finally:connection.close()


class Worker:
    def __init__(self,role,planning,motion,geometry):
        self.role=role;self.arguments=(role,planning,motion,geometry);self.job=None
        self.launch()

    def launch(self):
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=_worker,args=(child,*self.arguments),daemon=True)
        self.process.start();child.close();self.ready=False;self.started=time.monotonic()

    def send(self,job):
        if not self.ready or self.job is not None:raise RuntimeError('Worker is not idle')
        wire=dict(job);asset=job['asset']
        if asset is not None:wire['asset']=dict(directory=str(asset.directory),manifest=asset.manifest,metadata=asset.snapshot.metadata)
        self.connection.send(wire);self.job=job

    def poll(self):
        if not self.connection.poll():return None
        result=self.connection.recv()
        if result.get('ready'):self.ready=True;return None
        job=self.job;self.job=None
        if job is None or result['job_id']!=job['job_id']:raise RuntimeError('Worker result identity mismatch')
        return job,result

    def stop(self):
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=.5)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=.5)
        self.connection.close();self.job=None

    def restart(self):
        self.stop();self.launch()
