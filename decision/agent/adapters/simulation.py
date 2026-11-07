"""Agent navigation adapter over the project's planner, DWA, safety and plant.

No ROS imports or robot output. All feedback explicitly belongs to simulation.
"""
import copy
import numpy as np
from .. import ROOT


class SimulationNavigation:
    scope = 'simulation'
    clock_domain = 'simulation'

    def __init__(self):
        from navigation.simulation import PlanarWorld
        from navigation.planning.inputs import load_config
        from navigation.planning.motion import load_motion_config
        from navigation.planning.reference import planning_config
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.planner import Planner
        from navigation.control.runtime import DwaNavigationControl
        self.motion = load_motion_config()
        self.motion['collision_backend'] = 'cpu'
        self.planning = planning_config(load_config())
        # Offline development only: allow a bounded CPU search through U-turns.
        # This never modifies the online planning configuration.
        self.planning['pose_search']['max_search_s'] = 2.0
        self.planning['motion_constraints'] = {k: self.motion[k] for k in ('min_turning_radius_m', 'allow_reverse', 'allow_in_place_rotation')}
        self.geometry = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
        self.world = PlanarWorld('agent')
        self.planner = Planner(self.planning, self.geometry, require_actual_start=True)
        self.control = DwaNavigationControl(self.planning, self.motion, self.geometry, simulation=True)
        self.pose = np.array([4., 4., 0.])
        self.velocity = np.zeros(2)
        self.t = 0.
        self.dt = .1
        self.queue = [np.zeros(2)]
        self.request = None
        self.handle = None
        self.reference = None
        self.last_reference = -1e9
        self.stopping = False
        self.status = 'idle'
        self.collision = False
        self.goal = None
        self.no_progress_s = 0.
        self.best_distance = None
        self.progress_pose = self.pose.copy()
        self.stop_reason = None
        self.last_command_reason = None

    def snapshot(self):
        return dict(scope=self.scope, clock_domain=self.clock_domain, stamp_s=10+self.t,
                    frame_id='sim_odom', session_id='agent', generation=0, source='synthetic_differential_drive_plant',
                    pose=self.pose.tolist(), velocity=self.velocity.tolist(), handle=copy.deepcopy(self.handle),
                    origin_pose=self.pose.tolist(), origin_frame_id='sim_odom',
                    status=self.status, collision=self.collision, goal=copy.deepcopy(self.goal), executable=False, execution_feedback=True,
                    stop_reason=self.stop_reason,
                    command_reason=self.last_command_reason,
                    reference_status=None if self.reference is None else self.reference['result']['status'],
                    reference_reason=None if self.reference is None else self.reference['result']['reason'],
                    planning_search_budget_s=self.planning['pose_search']['max_search_s'],
                    stopped=bool(np.max(np.abs(self.velocity)) < 1e-8 and all(np.max(abs(x)) < 1e-8 for x in self.queue)))

    def spatial_inputs(self):
        from perception.environment.layers import GridSnapshot
        scene = self.world.scene(self.t,self.pose,self.geometry)
        metadata = dict(scene.geometry.metadata,snapshot_id='synthetic:agent:static')
        global_grid = GridSnapshot(metadata,dict(occupancy=self.world.static.copy()))
        return dict(global_grid=global_grid,scene=scene,pose=self.pose.tolist(),local_pose=self.pose.tolist(),
                    stamp_s=10+self.t,clock_domain=self.clock_domain,geometry=self.geometry)

    def reference_grid(self):
        return self.spatial_inputs()['global_grid']

    def check_route(self, place):
        scene = self.world.scene(self.t, self.pose, self.geometry)
        request = dict(request_id='preflight', task_id='preflight', goal_revision=1, mode='local',
                       frame_id='sim_odom', target_reference='base_link', goal=place['pose'], goal_behavior='stop')
        result = self.planner.query(request, scene=scene, now=10+self.t, clock_domain='simulation')
        return dict(status=result['status'], reason=result['reason'], executable=False,
                    scope=self.scope, reference_found=result['status'] == 'path_found')

    def start(self, place, handle):
        if self.status in ('running', 'stopping') or not self.snapshot()['stopped']:
            raise RuntimeError('Only one active goal, and replacement requires a confirmed stop')
        self.handle = copy.deepcopy(handle)
        self.request = dict(request_id=handle['call_id'], task_id=handle['task_id'], goal_revision=handle['goal_revision'],
                            mode='local', frame_id='sim_odom', target_reference='base_link', goal=place['pose'], goal_behavior='stop')
        self.status, self.stopping = 'running', False
        self.reference, self.goal = None, None
        self.best_distance, self.no_progress_s = None, 0.
        self.progress_pose = self.pose.copy()
        self.stop_reason = None
        return dict(accepted=True, execution_feedback=False, handle=copy.deepcopy(handle), scope=self.scope)

    def cancel(self, handle):
        if self.handle != handle:
            raise ValueError('Cancellation belongs to a different active call')
        self.stopping = True
        self.status = 'stopping'
        self.control.controller.clear()

    def tick(self):
        from navigation.kinematics import advance, velocity_step
        from navigation.planning.continuity import reference_envelope
        from navigation.planning.motion import interpolate_se2
        from navigation.planning.footprint import swept_check
        from perception.environment.layers import GridSnapshot
        if self.request is None or (self.status in ('arrived', 'stopped', 'failed') and self.snapshot()['stopped']):
            self.t = round(self.t+self.dt, 8)
            return self.snapshot()
        scene = self.world.scene(self.t, self.pose, self.geometry)
        now = 10+self.t
        feedback = dict(status='valid', source='synthetic_differential_drive_plant', pose=self.pose.tolist(),
                        velocity=self.velocity.tolist(), stamp_s=now, local_identity=['agent', 0, 'sim_odom'], clock_domain='simulation')
        active = self.status == 'running' and not self.stopping
        if active and (self.reference is None or self.t-self.last_reference >= 3):
            result = self.planner.query(self.request, scene=scene, now=now, clock_domain='simulation')
            self.reference = reference_envelope(result, self.request, scene, None)
            self.last_reference = self.t
        packet = self.control.step(self.request, scene, feedback, now=now, clock_domain='simulation', dt=self.dt,
                                   result=self.reference if active else None, active=active)
        self.goal = packet['goal']
        self.last_command_reason = packet['command']['reason']
        self.queue.append(np.array(packet['command']['velocity']))
        target = self.queue.pop(0)
        after = velocity_step(self.velocity, target, self.motion, self.dt)
        previous = self.pose.copy()
        self.pose = advance(self.pose, (self.velocity+after)/2, self.dt)
        self.velocity = after
        truth = GridSnapshot(scene.geometry.metadata, dict(occupancy=self.world.occupancy(self.t)))
        dense, error = interpolate_se2(np.array([previous, self.pose]), self.world.resolution, 2000)
        self.collision |= swept_check(truth, self.geometry, dense, error, 2000, prefix=False)['status'] != 'free'
        self.t = round(self.t+self.dt, 8)
        if self.collision:
            self.stopping, self.status = True, 'stopping'
        if active and packet['goal'].get('task_completed') and self.snapshot()['stopped']:
            self.status = 'arrived'
        if self.stopping and self.snapshot()['stopped']:
            self.status = 'failed' if self.collision else 'stopped'
        if active:
            distance = float(np.linalg.norm(self.pose[:2]-self.request['goal'][:2]))
            # A curvature-constrained U-turn may first increase straight-line
            # goal distance. Actual displacement also counts as progress;
            # Total simulation time is bounded separately by the Agent CLI.
            if (self.best_distance is None or self.best_distance-distance > .03 or
                    np.linalg.norm(self.pose[:2]-self.progress_pose[:2]) > .1):
                self.best_distance, self.no_progress_s = distance, 0.
                self.progress_pose = self.pose.copy()
            else:
                self.no_progress_s += self.dt
            if self.no_progress_s > 15 and self.status == 'running':
                self.status, self.stopping = 'stopping', True
                self.stop_reason = 'no_physical_progress'
        return self.snapshot()
