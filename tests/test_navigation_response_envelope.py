"""Synthetic response counterexamples and conservative execution enclosure."""
import unittest
import json

import numpy as np

from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry, footprint_status
from navigation.safety.guard import stopping_trajectory
from navigation.safety.response import execution_response_enclosure, execution_enclosure_geometry, _rotation_footprint
from navigation.safety.guard import LocalSafety
from navigation.planning.inputs import load_config as planning_config
from perception.environment.context import fingerprint
from tests.test_navigation_planning import grid, bundle, request
from tests.test_navigation_prediction import tracked_scene, track


def _travel_to_target(initial, target, acceleration, duration):
    """Exact signed distance under a bounded constant-rate velocity change."""
    change = target - initial
    elapsed = min(abs(change) / acceleration, duration)
    end = initial + np.sign(change) * acceleration * elapsed
    return (initial + end) * elapsed / 2 + end * (duration - elapsed), end


def _independent_stop(initial, pending, desired, response_acceleration, braking, delay, reaction):
    distance, v = _travel_to_target(initial, pending, response_acceleration, delay)
    segment, v = _travel_to_target(v, desired, response_acceleration, reaction)
    if v < 0:
        raise ValueError('These counterexamples are forward-only')
    return distance + segment + v * v / (2 * braking)


class ResponseEnvelopeCounterexamples(unittest.TestCase):
    def setUp(self):
        self.motion = load_motion_config('navigation/planning/configs/motion_trial.yaml')
        self.settings = dict(actuator_delay_s=.8, reaction_s=1.)
        self.dt = .1
        self.motion['execution_response_bounds'] = dict(max_planar_speed_mps=.10,max_stop_time_s=1.4)
        self.motion['execution_max_angular_rps'] = .15
        self.geometry = VehicleGeometry([[-.28,-.33],[.82,-.33],[.82,.33],[-.28,.33]],
            np.eye(4), source='synthetic_geometry_values')

    def predicted_distance(self, initial, pending, desired):
        feedback = dict(pose=np.zeros(3), velocity=np.array([initial, 0.]))
        path = stopping_trajectory(feedback, np.array([desired, 0.]), self.motion,
            self.settings, self.dt, pending=np.array([pending, 0.]))
        return float(path[-1, 1])

    def test_fast_response_can_outtravel_one_nominal_prediction(self):
        # Candidate target .01 is one .1-second planning step from zero.
        # Existing execution-profile validation permits a vehicle acceleration
        # upper bound above the planner's nominal acceleration.
        predicted = self.predicted_distance(0., .1, .01)
        physical = _independent_stop(0., .1, .01, .3, .075, .8, 1.1)
        self.assertLess(predicted, physical)
        envelope=execution_response_enclosure(dict(pose=[1.,1.,0.]),self.geometry,self.motion,self.settings,self.dt,0.)
        self.assertGreaterEqual(envelope['translation_bound_m'], physical)

    def test_slow_deceleration_can_outtravel_one_nominal_prediction(self):
        # Even when the maximum acceleration equals the model's .1, the
        # permitted .075 minimum braking does not enforce .1 deceleration.
        predicted = self.predicted_distance(.1, .1, 0.)
        physical = _independent_stop(.1, .1, 0., .075, .075, .8, 1.1)
        self.assertLess(predicted, physical)
        envelope=execution_response_enclosure(dict(pose=[1.,1.,0.]),self.geometry,self.motion,self.settings,self.dt,0.)
        self.assertGreaterEqual(envelope['translation_bound_m'], physical)

    def test_nominal_free_does_not_enclose_a_faster_response_footprint(self):
        # Synthetic finely gridded static scene; this is not the live map.
        cells = np.zeros((900, 900), dtype=np.int8)
        cells[400, 589] = 100
        scene = grid(cells)
        scene.metadata['resolution'] = .005
        predicted = self.predicted_distance(0., .1, .01)
        physical = _independent_stop(0., .1, .01, .3, .075, .8, 1.1)
        self.assertEqual(footprint_status(scene, self.geometry, [2+predicted, 2, 0], .05), 'free')
        self.assertEqual(footprint_status(scene, self.geometry, [2+physical, 2, 0], .05), 'occupied')
        envelope=execution_response_enclosure(dict(pose=[2.,2.,0.]),self.geometry,self.motion,self.settings,self.dt,0.)
        outer=execution_enclosure_geometry(self.geometry,envelope)
        self.assertEqual(footprint_status(scene,outer,[2.,2.,0.],.05+envelope['additional_margin_m']), 'occupied')

    def test_enclosure_covers_reverse_turning_and_stop_time(self):
        envelope=execution_response_enclosure(dict(pose=[1.,1.,0.]),self.geometry,self.motion,self.settings,self.dt,.1)
        self.assertAlmostEqual(envelope['duration_s'],2.6)
        self.assertAlmostEqual(envelope['translation_bound_m'],.26)
        theta=.15*envelope['duration_s']
        maximum_rotation=2*self.geometry.radius*np.sin(theta/2)
        self.assertAlmostEqual(envelope['rotation_margin_m'],maximum_rotation)
        self.assertAlmostEqual(envelope['additional_margin_m'],.26)
        footprint=np.asarray(envelope['footprint_xy_m'])
        self.assertGreater(footprint[:,1].max(),self.geometry.footprint[:,1].max())

    def test_bounded_planar_and_yaw_paths_stay_in_expanded_footprint(self):
        envelope=execution_response_enclosure(dict(pose=[0.,0.,0.]),self.geometry,self.motion,self.settings,self.dt,0.)
        footprint=np.asarray(envelope['footprint_xy_m'])
        lower=footprint.min(axis=0)-envelope['additional_margin_m']
        upper=footprint.max(axis=0)+envelope['additional_margin_m']
        rng=np.random.default_rng(2026)
        steps=200;step=envelope['duration_s']/steps
        for _ in range(30):
            xy=np.zeros(2);yaw=0.
            for __ in range(steps):
                heading=rng.uniform(-np.pi,np.pi);speed=rng.uniform(0.,.1)
                xy+=speed*step*np.array([np.cos(heading),np.sin(heading)])
                yaw+=rng.uniform(-.15,.15)*step
                c,s=np.cos(yaw),np.sin(yaw)
                rotated=self.geometry.footprint@np.array([[c,s],[-s,c]])
                actual=rotated+xy
                self.assertTrue(np.all(actual>=lower-1e-12) and np.all(actual<=upper+1e-12))

    def test_rotation_rectangle_encloses_vertices_across_angle_interval(self):
        for limit in (.001,.3,1.2,np.pi):
            box=np.asarray(_rotation_footprint(self.geometry.footprint,limit))
            lower,upper=box.min(axis=0),box.max(axis=0)
            for angle in np.linspace(-limit,limit,501):
                c,s=np.cos(angle),np.sin(angle)
                actual=self.geometry.footprint@np.array([[c,s],[-s,c]])
                self.assertTrue(np.all(actual>=lower-1e-12) and np.all(actual<=upper+1e-12))


class ExecutionEnvelopeGateTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config('navigation/planning/configs/motion_trial.yaml')
        self.motion.update(parameters_assumed=False,collision_backend='cpu',collision_library=None,
            execution_response_bounds=dict(max_planar_speed_mps=.10,max_stop_time_s=.5),
            execution_max_angular_rps=.15)
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),
            source='synthetic_only',assumed=False)
        self.req=request('local')
        self.feedback=dict(status='valid',source='synthetic',pose=[2.,2.,0.],velocity=[0.,0.],
            stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])

    def check(self,scene):
        guard=LocalSafety(self.geometry,self.motion,.05,people_settings=planning_config()['people'],execution=True)
        guard.last_target=np.array([.1,0.])
        proposal=dict(status='candidate',control_mode='dwa',velocity=[.01,0.],
            request_binding=fingerprint(self.req),scene_binding=scene.record['input_fingerprint'],
            trajectory_as_of=10.,trajectory_binding='synthetic')
        return guard.gate(proposal,self.req,scene,self.feedback,now=10.,clock_domain='synthetic',dt=.1)

    def test_clear_environment_allows_original_candidate(self):
        result=self.check(bundle())
        self.assertEqual(result['state'],'allowed',result)
        self.assertEqual(result['velocity'],[.01,0.])
        self.assertGreater(result['response_envelope']['additional_margin_m'],0.)
        json.dumps(result['response_envelope'],allow_nan=False)

    def test_static_obstacle_outside_nominal_path_is_rejected(self):
        cells=np.zeros((60,60),np.int8);cells[20,23]=100
        result=self.check(bundle(cells))
        self.assertEqual(result['check']['status'],'free')
        self.assertEqual(result['reason'],'response_envelope_occupied')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_dynamic_person_outside_nominal_path_is_rejected(self):
        scene=tracked_scene([track(2.66,2.,(-.03,0.,0.))])
        result=self.check(scene)
        self.assertTrue(result['people_check']['valid'],result)
        self.assertEqual(result['reason'],'predicted_people_collision')
        self.assertFalse(result['response_envelope']['people_check']['valid'])
        self.assertEqual(result['velocity'],[0.,0.])

    def test_feedback_outside_declared_physical_bound_is_rejected(self):
        self.feedback['velocity']=[.11,0.]
        result=self.check(bundle())
        self.assertEqual(result['reason'],'feedback_exceeds_execution_response_bound')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_people_prediction_horizon_cannot_be_silently_shortened(self):
        self.motion['execution_response_bounds']['max_stop_time_s']=6.
        result=self.check(tracked_scene([track(4.5,2.,(0.,0.,0.))]))
        self.assertEqual(result['reason'],'people_prediction_horizon_exceeded')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_execution_constructor_requires_angular_and_response_bounds(self):
        for key in ('execution_response_bounds','execution_max_angular_rps'):
            motion=dict(self.motion);motion.pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):
                LocalSafety(self.geometry,motion,.05,people_settings=planning_config()['people'],execution=True)


if __name__ == '__main__':
    unittest.main()
