"""Synthetic worker and observation fixtures for current reference-runtime tests."""
from perception.environment.scene import assemble_scene
from perception.environment.layers import load_config as environment_config
from tests.test_navigation_planning import bundle


class FakeWorker:
    def __init__(self,*args):self.ready=True;self.job=None;self.started=0.;self.response=None;self.restarts=0
    def send(self,job):
        if self.job is not None:raise AssertionError('Unbounded pending queue')
        self.job=job
    def poll(self):
        if self.response is None:return None
        value=self.job,self.response;self.job=self.response=None;return value
    def stop(self):self.job=None
    def restart(self):self.restarts+=1;self.job=self.response=None
    def reply(self,result):
        self.response=dict(job_id=self.job['job_id'],result=result,error=None,compute_s=.01)


def scene_at(stamp,*,associated=True,generation=0):
    original=bundle(generation=generation,associated=associated);grid=original.geometry
    grid.metadata['stamp_s']=stamp;grid.metadata['association'].update(stamp_s=stamp,measurement_stamp_s=stamp-.1)
    return assemble_scene(grid,clock_domain='synthetic',settings=environment_config()['scene'])
