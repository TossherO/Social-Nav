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


def square_vehicle_geometry():
    from navigation.planning.footprint import VehicleGeometry

    return VehicleGeometry([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]],
                           np.eye(4), source='synthetic')


def dwa_inputs():
    """Provide the planning, motion and feedback inputs shared by DWA tests."""
    from navigation.planning.inputs import load_config
    from navigation.planning.motion import load_motion_config

    cfg = load_config()
    cfg['algorithm'] = 'hybrid_astar'
    cfg['local_replanning']['enabled'] = False
    motion = load_motion_config()
    geometry = square_vehicle_geometry()
    req = request('local', goal=[4., 2., 0.], goal_behavior='stop')
    feedback = dict(status='valid', source='synthetic_actual_plant', pose=[2., 2., 0.],
                    velocity=[0., 0.], stamp_s=10., clock_domain='synthetic',
                    local_identity=['test', 0, 'odom'])
    return SimpleNamespace(cfg=cfg, motion=motion, geometry=geometry, req=req, feedback=feedback)


def dwa_fixture():
    """Plan a straight reference through the synthetic flat local scene."""
    from navigation.control.dwa import DwaController
    from navigation.planning.continuity import reference_envelope
    from navigation.planning.planner import Planner

    fixture = dwa_inputs()
    fixture.scene = bundle()
    result = Planner(fixture.cfg, fixture.geometry).query(
        fixture.req, scene=fixture.scene, now=10., clock_domain='synthetic')
    if result['status'] != 'path_found':
        raise AssertionError(result)
    fixture.reference = reference_envelope(result, fixture.req, fixture.scene, None)
    fixture.controller = DwaController(fixture.cfg, fixture.motion, fixture.geometry)
    return fixture


def shadow_inputs():
    """Build a flat scene and measured stationary poses for shadow control."""
    from navigation.planning.inputs import load_config
    from navigation.planning.motion import load_motion_config
    from navigation.planning.reference import planning_config

    rows = [dict(stamp_s=stamp, health=0, session_id='test', generation=0, frame_id='odom',
                 position=[2., 2., 0.], quaternion_xyzw=[0., 0., 0., 1.])
            for stamp in (9.7, 9.8, 9.9, 10.)]
    return SimpleNamespace(config=planning_config(load_config()), motion=load_motion_config(),
                           geometry=square_vehicle_geometry(), scene=bundle(),
                           req=request('local', goal=[4., 2.]), rows=rows)


def shadow_packet_fixture():
    """Generate a shadow proposal from the flat synthetic scene."""
    from navigation.control.shadow import DwaShadowController
    from navigation.planning.continuity import reference_envelope
    from navigation.planning.planner import Planner

    fixture = shadow_inputs()
    planner = Planner(fixture.config, fixture.geometry, require_actual_start=True)
    result = planner.query(fixture.req, scene=fixture.scene, now=10., clock_domain='synthetic')
    fixture.reference = reference_envelope(result, fixture.req, fixture.scene, None)
    fixture.controller = DwaShadowController(fixture.config, fixture.motion, fixture.geometry)
    fixture.packet = fixture.controller.step(fixture.req, fixture.scene, fixture.rows, fixture.reference,
                                             now=10., clock_domain='synthetic', mono=0.)
    return fixture


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
