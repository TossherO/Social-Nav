"""Synthetic feedback and independent stopping/timeout boundaries; no hardware."""
import unittest
from unittest.mock import patch
import numpy as np
from perception.environment.context import fingerprint
from navigation.control.tracking import goal_feedback,GoalMonitor,load_config as goal_config
from navigation.safety.guard import LocalSafety
from navigation.kinematics import braking_path,velocity_step,turning_residual
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry
from tests.navigation_fixtures import bundle,request


class ControlTests(unittest.TestCase):
    def test_delayed_weaker_braking_with_pose_noise_does_not_overshoot_into_a_loop(self):
        from navigation.simulation import run_case
        geometry=VehicleGeometry.historical(self.config['geometry_reference'])
        result=run_case('straight',self.config,self.motion,geometry,max_s=20.,plant_delay_s=.2,
            plant_acceleration_scale=.75,feedback_noise_m=.02,feedback_noise_rad=.02)
        self.assertTrue(result['completed']);self.assertFalse(result['collision'])
        self.assertLess(result['simulated_s'],20.)
        self.assertLess(np.max(abs(np.array([h['pose'][2] for h in result['history']]))),.2)
        self.assertLess(np.linalg.norm(np.array(result['final_pose'])[:2]-[8.,2.]),.12)

    def setUp(self):
        self.motion=load_motion_config();self.config=load_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.goal_settings=goal_config()
        self.guard=LocalSafety(self.geometry,self.motion,.05,simulation=True)
        self.scene=bundle();self.req=request('local',goal=[4.,2.],goal_behavior='stop')
        self.feedback=dict(status='valid',source='synthetic_plant',clock_domain='synthetic',local_identity=['test',0,'odom'],
            stamp_s=10.,pose=[2.,2.,0.],velocity=[0.,0.])

    def proposal(self,velocity):
        return dict(status='candidate',control_mode='dwa',velocity=velocity,request_binding=fingerprint(self.req),
            scene_binding=self.scene.record['input_fingerprint'],trajectory_as_of=10.,trajectory_binding='synthetic_checked_candidate')

    def gate(self,proposal,feedback=None,**kw):
        return self.guard.gate(proposal,self.req,self.scene,feedback or self.feedback,now=10.,clock_domain='synthetic',dt=.1,**kw)

    def test_retired_control_proposal_cannot_reach_the_gate(self):
        proposal=self.proposal([.5,0.]);proposal.pop('control_mode')
        self.assertEqual(self.gate(proposal)['reason'],'unsupported_control_mode')

    def test_model_braking_distance_matches_independent_energy_formula(self):
        path=braking_path(np.zeros(3),np.array([.8,0.]),self.motion,.2)
        expected=.8*.2+.8**2/(2*self.motion['max_acceleration_mps2'])
        self.assertAlmostEqual(path[-1,0],expected,places=8)

    def test_braking_sweep_blocks_obstacle_beyond_the_next_command_step(self):
        cells=np.zeros((60,60),np.int8);cells[18:23,30]=100;self.scene=bundle(cells)
        r=self.gate(self.proposal([.8,0.]),dict(self.feedback,velocity=[.8,0.]))
        self.assertEqual(r['reason'],'stopping_sweep_occupied')
        self.assertEqual(r['velocity'],[0.,0.]);self.assertEqual(r['state'],'stop_request')

    def test_unknown_cannot_be_cleared_by_control(self):
        cells=np.zeros((60,60),np.int8);cells[:,30:]=-1;self.scene=bundle(cells)
        r=self.gate(self.proposal([.8,0.]),dict(self.feedback,velocity=[.8,0.]))
        self.assertEqual(r['reason'],'stopping_sweep_unknown');self.assertEqual(cells[20,30],-1)

    def test_independent_gate_rejects_when_its_own_people_check_finishes_late(self):
        model=self.guard._people(self.scene);original=model.check;clock=[0.]
        def late(*args,**kwargs):
            result=original(*args,**kwargs);clock[0]+=.11;return result
        with patch('navigation.safety.guard.time.perf_counter',side_effect=lambda:clock[0]), \
                patch.object(model,'check',side_effect=late):
            result=self.gate(self.proposal([.05,0.]))
        self.assertEqual(result['reason'],'stopping_certification_time_budget_exceeded')
        self.assertEqual(result['state'],'stop_request');self.assertEqual(result['velocity'],[0.,0.])
        np.testing.assert_array_equal(self.guard.last_target,[0.,0.])

    def test_cancel_unavailable_emergency_and_takeover_request_actual_braking(self):
        f=dict(self.feedback,velocity=[.6,.1]);p=self.proposal([.8,.2])
        for proposal,kw,reason in [(None,{},'planner_unavailable'),(p,dict(emergency_stop=True),'emergency_stop'),
                                   (p,dict(manual_override=True),'manual_override')]:
            r=self.gate(proposal,f,**kw);self.assertEqual(r['reason'],reason)
            self.assertLess(r['velocity'][0],.6);self.assertFalse(r['executable'])

    def test_feedback_and_proposal_lifecycle_and_age_are_independent_gates(self):
        for feedback in [dict(self.feedback,stamp_s=8.),dict(self.feedback,local_identity=['other',0,'odom']),
                         dict(self.feedback,pose=['invalid',0,0]),dict(self.feedback,velocity=['1',0]),
                         dict(self.feedback,velocity=[float('nan'),0])]:
            self.assertEqual(self.gate(self.proposal([.5,0.]),feedback)['reason'],'feedback_invalid')
        r=self.gate(dict(self.proposal([.5,0.]),request_binding='old_task'))
        self.assertEqual(r['reason'],'proposal_binding_or_age_invalid')
        r=self.gate(self.proposal([.5,0.]),dict(self.feedback,velocity=[100.,0.]))
        self.assertEqual(r['reason'],'feedback_outside_motion_model')

    def test_corrupted_scene_requests_stop_without_trusting_controller_output(self):
        self.scene.geometry.layers['occupancy'][30,30]=100
        r=self.gate(self.proposal([.5,0.]))
        self.assertEqual(r['reason'],'scene_binding_invalid');self.assertEqual(r['velocity'],[0.,0.])

    def test_arrival_requires_active_fresh_actual_feedback_and_requested_stop(self):
        f=dict(self.feedback,pose=[4.,2.,0.],velocity=[.3,0.])
        def outcome(f,active=True):return goal_feedback(self.req,self.scene,f,self.goal_settings,
            now=10.,clock_domain='synthetic',active=active)
        self.assertFalse(outcome(f)['task_completed']);f=dict(f,velocity=[0.,0.])
        self.assertTrue(outcome(f)['criteria_met']);self.assertFalse(outcome(f,False)['task_completed'])
        self.assertFalse(outcome(dict(f,stamp_s=8.))['task_completed'])
        oriented=dict(self.req,goal=[4.,2.,np.pi/2])
        r=goal_feedback(oriented,self.scene,f,self.goal_settings,now=10.,clock_domain='synthetic',active=True)
        self.assertFalse(r['criteria_met']);self.assertFalse(r['yaw_aligned'])

    def test_goal_monitor_needs_sustained_actual_feedback_and_clears_on_cancel(self):
        m=GoalMonitor();f=dict(self.feedback,pose=[4.,2.,0.])
        def update(t,active=True):return m.update(self.req,self.scene,dict(f,stamp_s=t),now=t,
            clock_domain='synthetic',active=active)
        self.assertFalse(update(10.)['task_completed']);self.assertFalse(update(10.3)['task_completed'])
        self.assertTrue(update(10.5)['task_completed']);self.assertFalse(update(10.6,False)['task_completed'])
        self.assertFalse(m.update(self.req,self.scene,dict(f,stamp_s='invalid'),now=10.6,
            clock_domain='synthetic',active=True)['task_completed'])
        self.assertFalse(update(10.7)['task_completed'])


    def test_simulation_adapters_cannot_be_implicitly_used_as_real_chassis_adapters(self):
        with self.assertRaises(ValueError):LocalSafety(self.geometry,self.motion,.05)

    def test_safety_independently_rejects_spin_and_near_spin_proposals(self):
        self.motion['allow_in_place_rotation']=False
        self.guard=LocalSafety(self.geometry,self.motion,.05,simulation=True)
        for v in (0.,.001):
            r=self.gate(self.proposal([v,.2]))
            self.assertEqual(r['reason'],'turning_radius_violation')
            self.assertEqual(r['velocity'],[0.,0.])

    def test_turning_brake_respects_cone_acceleration_and_has_no_residual_spin(self):
        v=np.array([.25,.22]);trace=[v.copy()]
        for _ in range(30):
            after=velocity_step(v,[0.,0.],self.motion,.1)
            self.assertGreaterEqual(turning_residual(after,self.motion),-1e-9)
            self.assertLessEqual(abs(after[0]-v[0]),self.motion['max_acceleration_mps2']*.1+1e-9)
            self.assertLessEqual(abs(after[1]-v[1]),self.motion['max_angular_acceleration_rps2']*.1+1e-9)
            trace.append(after.copy());v=after
        np.testing.assert_allclose(v,[0.,0.],atol=1e-12)
        from navigation.planning.motion import twists
        path=braking_path(np.zeros(3),trace[0],self.motion,0.)
        intervals=twists(path,np.ones(len(path)-1))
        self.assertTrue(np.all(abs(intervals[:,2])<=abs(intervals[:,0])/self.motion['min_turning_radius_m']+1e-9))


if __name__=='__main__':unittest.main()
