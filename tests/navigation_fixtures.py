"""Shared synthetic worker and observation fixtures; not a test entry point."""
import hashlib
from types import SimpleNamespace
import numpy as np
from perception.environment.scene import assemble_scene
from perception.environment.layers import GridSnapshot, load_config as environment_config


def grid(cells=None, *, generation=0):
    cells = np.zeros((60, 60), np.int8) if cells is None else cells
    transform = np.eye(4); transform[:2, 3] = [2., 2.]
    return GridSnapshot(dict(shape=list(cells.shape), resolution=.1, origin_xy=[0., 0.], frame_id='odom',
        snapshot_id='local:'+str(generation), stamp_s=10., session_id='test', generation=generation,
        world_from_lidar=transform.tolist(), geometry_status='OBSERVED', sensor_frame='lidar', localization_source='synthetic'),
        dict(occupancy=cells))


def bundle(cells=None, *, generation=0, associated=True):
    current = grid(cells, generation=generation)
    current.metadata['association'] = dict(valid=associated, map_id='map-a', map_frame='map', local_frame='odom',
        local_session_id='test', local_generation=generation, stamp_s=10., measurement_stamp_s=9.9,
        map_from_local=np.eye(4).tolist(), association_session='association', association_generation=0)
    return assemble_scene(current, clock_domain='synthetic', settings=environment_config()['scene'])


def asset(cells=None):
    current = grid(cells)
    current.metadata.update(frame_id='map', map_id='map-a', snapshot_id='static-a')
    return SimpleNamespace(snapshot=current, manifest=dict(snapshot_id='static-a',
        layers_sha256=hashlib.sha256(current.layers['occupancy'].tobytes()).hexdigest()))


def request(mode='static', **changes):
    values = dict(request_id='r', task_id='t', goal_revision=0, mode=mode, frame_id='map' if mode != 'local' else 'odom',
                  target_reference='base_link', goal=[4., 4.], map_snapshot_id='static-a')
    if mode == 'static': values['start'] = [2., 2., 0.]
    values.update(changes)
    return values


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
