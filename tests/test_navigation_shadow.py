"""Shadow transport, independent observations and wall-clock stop semantics."""
import copy
import unittest
from unittest.mock import Mock,patch
import numpy as np
from perception.environment.context import fingerprint
from navigation.control.shadow import DwaShadowController,ClockWatch,observed_feedback,publisher
from navigation.safety.shadow import ShadowSafety
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.planner import Planner
from navigation.planning.reference import planning_config
from navigation.planning.continuity import reference_envelope
from tests.navigation_fixtures import bundle,request,grid


class ChassisAuthorizationTests(unittest.TestCase):
    def test_authorization_requires_fresh_exact_goal_and_respects_expiry(self):
        from navigation.safety.shadow import chassis_authorization
        goal={'task_id':'t','goal_revision':1}
        result=dict(state='allowed',completion_freshness_checked=True,wire_velocity_units=[100,20],
            velocity=[.1,.02],request=goal,decision_complete_mono=10.,remaining_validity_s=.12)
        args=dict(arm={'enabled':True,'revision':2},emergency=False,manual=False,
            active_goal=goal,clock_domain='runtime_live')
        with patch('navigation.safety.shadow.time.monotonic',return_value=10.01):
            packet=chassis_authorization(result,**args)
            self.assertTrue(packet['allowed']);self.assertAlmostEqual(packet['expires_mono'],10.12)
            self.assertEqual(packet['units'],[100,20]);self.assertEqual(packet['arm_revision'],2)
            for change in (dict(emergency=True),dict(manual=True),dict(active_goal={'task_id':'other'}),dict(active_goal=None)):
                self.assertFalse(chassis_authorization(result,**dict(args,**change))['allowed'])
            for change in (dict(completion_freshness_checked=False),dict(wire_velocity_units=[99,20]),dict(state='stop')):
                self.assertFalse(chassis_authorization(dict(result,**change),**args)['allowed'])
            missing=chassis_authorization(None,**args)
            self.assertFalse(missing['allowed']);self.assertEqual(missing['expires_mono'],missing['mono'])


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.config=planning_config(load_config());self.motion=load_motion_config();self.scene=bundle();self.req=request('local',goal=[4.,2.])
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.rows=[dict(stamp_s=t,health=0,session_id='test',generation=0,frame_id='odom',position=[2.,2.,0.],quaternion_xyzw=[0.,0.,0.,1.]) for t in [9.7,9.8,9.9,10.]]
        planner=Planner(self.config,self.geometry,require_actual_start=True)
        result=planner.query(self.req,scene=self.scene,now=10.,clock_domain='synthetic')
        self.reference=reference_envelope(result,self.req,self.scene,None)
        self.controller=DwaShadowController(self.config,self.motion,self.geometry)
        self.packet=self.controller.step(self.req,self.scene,self.rows,self.reference,now=10.,clock_domain='synthetic',mono=0.)
        self.result=self.controller.controller.result
        self.safety=ShadowSafety(self.config,self.motion,self.geometry,clock_domain='synthetic')
        self.safety.command('goal',self.req);self.safety.push_grid(self.scene.geometry,mono=0.);self.safety.push_record(self.scene.record)
        for row in self.rows:self.safety.push_state(row)
        self.safety.push_packet(self.packet,mono=0.)

    def test_full_shadow_chain_never_reports_execution_or_task_success(self):
        r=self.safety.tick(10.,mono=0.)
        self.assertEqual(r['state'],'allowed',r);self.assertGreater(r['velocity'][0],0.)
        self.assertEqual(r['trajectory_source_scene_binding'],self.result['input_binding']['scene'])
        self.assertFalse(r['executable']);self.assertFalse(r['task_completed']);self.assertFalse(r['execution_feedback'])
        self.assertFalse(self.packet['feedback']['execution_feedback'])


    def test_independent_gate_derives_its_own_motion_not_packet_feedback(self):
        packet=copy.deepcopy(self.packet);packet['sequence_id']+=1;packet['feedback']['velocity']=[100.,10.]
        packet['packet_binding']=fingerprint({k:v for k,v in packet.items() if k!='packet_binding'})
        self.safety.push_packet(packet,mono=0.);r=self.safety.tick(10.,mono=0.)
        self.assertEqual(r['feedback']['velocity'],[0.,0.]);self.assertEqual(r['state'],'allowed')

    def test_pause_uses_wall_clock_and_no_packet_watchdog_is_independent(self):
        self.safety.tick(10.,mono=0.)
        self.assertEqual(self.safety.tick(10.,mono=.31)['reason'],'clock_stalled')
        r=self.safety.tick(10.01,mono=.32)
        self.assertEqual(r['reason'],'proposal_timeout');self.assertEqual(r['velocity'],[0.,0.])

    def test_cancel_override_and_stale_revisions_cannot_be_overridden_by_old_proposal(self):
        for field,reason in [('manual_override','manual_override'),('emergency_stop','emergency_stop')]:
            value=dict(manual_override=False,emergency_stop=False);value[field]=True
            self.safety.command('override',value);self.assertEqual(self.safety.tick(10.,mono=0.)['reason'],reason)
        self.safety.command('override',dict(manual_override=False,emergency_stop=False))
        self.safety.command('cancel',dict(task_id=self.req['task_id'],goal_revision=self.req['goal_revision']))
        self.assertEqual(self.safety.tick(10.,mono=0.)['reason'],'no_active_goal')
        with self.assertRaises(ValueError):self.safety.command('goal',self.req)
        self.safety.command('goal',dict(self.req,goal_revision=1))
        self.assertEqual(self.safety.tick(10.,mono=0.)['reason'],'active_goal_mismatch')

    def test_cancel_newer_revision_covers_an_older_active_goal_after_lost_ack(self):
        revision=self.req['goal_revision']+1
        self.safety.command('cancel',dict(task_id='different-task',goal_revision=revision))
        self.assertEqual(self.safety.tick(10.,mono=0.)['state'],'allowed')
        self.safety.command('cancel',dict(task_id=self.req['task_id'],goal_revision=revision))
        self.assertEqual(self.safety.tick(10.,mono=.01)['reason'],'no_active_goal')
        for old in (self.req['goal_revision'],revision):
            with self.assertRaises(ValueError):self.safety.command('goal',dict(self.req,goal_revision=old))

    def test_new_grid_is_independently_joined_and_checked_without_waiting_for_planner(self):
        self.safety.tick(10.,mono=0.)
        repeated=copy.deepcopy(self.scene.geometry);repeated.layers['occupancy'][20,22]=100
        self.safety.push_grid(repeated,mono=.001)
        self.assertEqual(self.safety.tick(10.,mono=.001)['reason'],'motion_sweep_occupied')
        newer=grid();newer.metadata.update(snapshot_id='new',stamp_s=10.01)
        self.safety.push_grid(newer,mono=.01)
        r=self.safety.tick(10.01,mono=.01)
        self.assertEqual(r['state'],'allowed',r);self.assertTrue(r['current_observation_revalidated'])
        self.assertNotEqual(r['scene_binding'],r['proposal_scene_binding'])
        blocked=grid();blocked.metadata.update(snapshot_id='blocked',stamp_s=10.02);blocked.layers['occupancy'][20,22]=100
        self.safety.push_grid(blocked,mono=.02);r=self.safety.tick(10.02,mono=.02)
        self.assertEqual(r['reason'],'motion_sweep_occupied');self.assertEqual(r['velocity'],[0.,0.])

    def test_corruption_late_packets_and_future_source_are_rejected(self):
        packet=copy.deepcopy(self.packet);packet['proposal']['velocity']=[1.,1.]
        with self.assertRaises(ValueError):self.safety.push_packet(packet,mono=.01)
        with self.assertRaises(ValueError):self.safety.push_packet(self.packet,mono=.01)
        packet=copy.deepcopy(self.packet);packet.update(sequence_id=2,stamp_s=10.1)
        packet['packet_binding']=fingerprint({k:v for k,v in packet.items() if k!='packet_binding'})
        self.safety.push_packet(packet,mono=0.)
        self.assertEqual(self.safety.tick(10.,mono=0.)['reason'],'proposal_clock_or_age_invalid')

    def test_feedback_has_causal_pose_provenance_and_does_not_fill_missing_velocity(self):
        future=dict(self.rows[-1],stamp_s=10.1,position=[100.,0.,0.])
        f=observed_feedback(self.rows+[future],self.scene,self.geometry,self.motion,now=10.)
        self.assertEqual(f['velocity'],[0.,0.]);self.assertEqual(f['stamp_s'],10.);self.assertIn('source_binding',f)
        self.assertEqual(observed_feedback(self.rows[-1:],self.scene,self.geometry,self.motion,now=10.)['status'],'unavailable')

    def test_output_cannot_be_remapped_to_a_chassis_topic(self):
        ros=Mock();ros.resolve_name.return_value='/cmd_vel'
        with self.assertRaises(ValueError):publisher(ros,'/navigation/shadow/velocity',object)
        ros.Publisher.assert_not_called()

    def test_map_reassociation_invalidates_a_proposal_even_with_fresh_local_geometry(self):
        from tests.navigation_fixtures import asset
        current=asset();req=request('mapped',goal=[4.,2.])
        planner=Planner(self.config,self.geometry,require_actual_start=True)
        result=reference_envelope(planner.query(req,scene=self.scene,asset=current,now=10.,clock_domain='synthetic'),req,self.scene,current)
        controller=DwaShadowController(self.config,self.motion,self.geometry,asset=current)
        packet=controller.step(req,self.scene,self.rows,result,now=10.,clock_domain='synthetic',mono=0.)
        gate=ShadowSafety(self.config,self.motion,self.geometry,clock_domain='synthetic',asset=current)
        gate.command('goal',req);gate.push_grid(self.scene.geometry,mono=0.);gate.push_record(self.scene.record)
        for row in self.rows:gate.push_state(row)
        gate.push_packet(packet,mono=0.)
        newer=copy.deepcopy(self.scene.geometry);newer.metadata.update(snapshot_id='reassociated',stamp_s=10.01)
        newer.metadata['association'].update(stamp_s=10.01,measurement_stamp_s=10.01,association_generation=1)
        gate.push_grid(newer,mono=.01)
        self.assertEqual(gate.tick(10.01,mono=.01)['reason'],'shadow_context_changed')

    def test_near_stationary_noise_keeps_raw_estimate_and_inflates_stopping_envelope(self):
        rows=[dict(r,position=[2.-.003*(r['stamp_s']-9.7),2.,0.]) for r in self.rows]
        feedback=observed_feedback(rows,self.scene,self.geometry,self.motion,now=10.)
        self.assertEqual(feedback['velocity'],[0.,0.]);self.assertLess(feedback['velocity_raw'][0],0.)
        self.assertGreater(feedback['stationary_evidence']['window_s'],.2)
        self.safety.rows.clear()
        for row in rows:self.safety.push_state(row)
        result=self.safety.tick(10.,mono=0.)
        self.assertEqual(result['state'],'allowed');self.assertGreater(result['motion_uncertainty_margin_m'],0.)
        from navigation.planning.motion import motion_uncertainty_margin
        self.assertGreater(motion_uncertainty_margin(feedback,self.geometry,1.),.01)
        reverse=[dict(r,position=[2.-.3*(r['stamp_s']-9.7),2.,0.]) for r in self.rows]
        estimate=observed_feedback(reverse,self.scene,self.geometry,self.motion,now=10.)
        self.assertNotIn('stationary_evidence',estimate);self.assertLess(estimate['velocity'][0],-.2)
        slow=[dict(r,position=[2.+.025*(r['stamp_s']-9.7),2.,0.]) for r in self.rows]
        slow_feedback=observed_feedback(slow,self.scene,self.geometry,self.motion,now=10.)
        self.assertEqual(slow_feedback['velocity'],[0.,0.])
        self.assertGreater(slow_feedback['stationary_evidence']['linear_bound_mps'],.024)
        self.assertGreater(motion_uncertainty_margin(slow_feedback,self.geometry,1.),.025)
        moving=[dict(r,position=[2.+.035*(r['stamp_s']-9.7),2.,0.]) for r in self.rows]
        self.assertNotIn('stationary_evidence',observed_feedback(moving,self.scene,self.geometry,self.motion,now=10.))
        rotating=[]
        for r in self.rows:
            angle=.03*(r['stamp_s']-9.7)
            rotating.append(dict(r,quaternion_xyzw=[0.,0.,float(np.sin(angle/2)),float(np.cos(angle/2))]))
        feedback=observed_feedback(rotating,self.scene,self.geometry,self.motion,now=10.)
        self.assertEqual(feedback['velocity'],[0.,0.])
        self.assertGreater(feedback['stationary_evidence']['yaw_span_rad'],.006)
        self.assertGreater(feedback['stationary_evidence']['angular_bound_rps'],.029)

    def test_grid_change_during_allowed_gate_is_rechecked_before_publication(self):
        original=self.safety.guard.gate;calls=[]
        def updating(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(result)
            if len(calls)==1:
                blocked=grid();blocked.metadata.update(snapshot_id='arrived_during_check',stamp_s=10.)
                blocked.layers['occupancy'][20,22]=100;self.safety.push_grid(blocked,mono=0.)
            return result
        self.safety.guard.gate=updating
        result=self.safety.tick(10.,mono=0.)
        self.assertEqual(len(calls),2);self.assertEqual(result['revalidation_attempts'],2)
        self.assertEqual(result['velocity'],[0.,0.]);self.assertEqual(result['reason'],'motion_sweep_occupied')

    def test_unavailable_proposal_does_not_assemble_or_verify_full_maps(self):
        packet=copy.deepcopy(self.packet);packet['sequence_id']+=1
        packet['proposal'].update(status='unavailable',reason='trajectory_unavailable')
        packet['packet_binding']=fingerprint({k:v for k,v in packet.items() if k!='packet_binding'})
        self.safety.push_packet(packet,mono=0.);self.safety._current_scene=Mock(side_effect=AssertionError('Should not assemble'))
        self.assertEqual(self.safety.tick(10.,mono=0.)['reason'],'planner_unavailable')

    def test_lease_expiry_during_gate_cannot_publish_an_old_positive_suggestion(self):
        clock=iter([10.,10.51]);result=self.safety.tick(10.,mono=0.,clock_now=lambda:next(clock))
        self.assertEqual(result['reason'],'proposal_expired_during_gate');self.assertEqual(result['velocity'],[0.,0.])

    def test_follow_request_deadline_limits_authorization_and_is_rechecked(self):
        req=dict(self.req,valid_until_s=10.05)
        packet=self.controller.step(req,self.scene,self.rows,self.reference,now=10.,clock_domain='synthetic',mono=0.)
        self.safety.command('goal',req);self.safety.push_packet(packet,mono=0.)
        result=self.safety.tick(10.,mono=0.,clock_now=lambda:10.02)
        self.assertEqual(result['state'],'allowed',result)
        self.assertAlmostEqual(result['remaining_request_validity_s'],.03)
        self.assertLessEqual(result['remaining_validity_s'],.03+1e-9)
        clock=iter([10.,10.06]);result=self.safety.tick(10.,mono=.01,clock_now=lambda:next(clock))
        self.assertEqual(result['reason'],'request_expired_during_gate')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_new_valid_grid_during_check_can_be_recertified_without_unconditional_stop(self):
        original=self.safety.guard.gate;calls=[]
        def updating(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(result)
            if len(calls)==1:
                newer=grid();newer.metadata.update(snapshot_id='new_free',stamp_s=10.)
                self.safety.push_grid(newer,mono=0.)
            return result
        self.safety.guard.gate=updating
        result=self.safety.tick(10.,mono=0.)
        self.assertEqual(result['state'],'allowed');self.assertEqual(result['revalidation_attempts'],2)

    def test_localization_failure_during_check_overrides_an_allowed_proposal(self):
        original=self.safety.guard.gate
        def invalidating(*args,**kwargs):
            result=original(*args,**kwargs);self.safety.push_state(dict(self.rows[-1],health=2));return result
        self.safety.guard.gate=invalidating
        result=self.safety.tick(10.,mono=0.)
        self.assertEqual(result['reason'],'localization_changed_during_gate');self.assertEqual(result['velocity'],[0.,0.])

    def test_identical_heartbeat_refreshes_watchdog_without_duplicate_event_tick(self):
        self.safety.wake.clear();packet=copy.deepcopy(self.packet);packet['sequence_id']+=1
        packet['packet_binding']=fingerprint({k:v for k,v in packet.items() if k!='packet_binding'})
        self.safety.push_packet(packet,mono=.1)
        self.assertFalse(self.safety.wake.is_set());self.assertEqual(self.safety.packet_received,.1)

    def test_live_pose_change_wakes_recheck_without_refreshing_source_time(self):
        self.safety.execution=True  # Exercise live wake scheduling without hardware.
        self.safety.wake.clear()
        self.safety.push_state(copy.deepcopy(self.rows[-1]))
        self.assertFalse(self.safety.wake.is_set())
        fresh=dict(self.rows[-1],stamp_s=10.1)
        self.safety.push_state(fresh)
        self.assertTrue(self.safety.wake.is_set())
        self.assertEqual(self.safety.rows[-1]['stamp_s'],10.1)
        self.safety.wake.clear()
        self.safety.push_state(dict(fresh,health=2))
        self.assertTrue(self.safety.wake.is_set())
        self.safety.wake.clear()
        self.safety.push_state(dict(fresh,health=2,has_map_pose=True))
        self.assertTrue(self.safety.wake.is_set())
        self.safety.wake.clear()
        self.safety.push_state(dict(fresh,stamp_s=9.))
        self.assertTrue(self.safety.wake.is_set())
        self.assertEqual(self.safety.barrier,'localization_reset')

    def test_live_pose_rechecks_cover_nominal_cadence_but_not_stale_feedback(self):
        from perception.environment.scene import assemble_scene
        self.geometry.assumed=False
        motion=dict(self.motion,parameters_assumed=False,
            execution_max_longitudinal_speed_mps=self.motion['max_forward_mps'],
            execution_max_angular_rps=self.motion['max_angular_rps'])
        scene=assemble_scene(self.scene.geometry,clock_domain='runtime_live',settings=self.safety.scene_settings)
        planner=Planner(self.config,self.geometry,require_actual_start=True)
        reference=reference_envelope(planner.query(self.req,scene=scene,now=10.,clock_domain='runtime_live'),self.req,scene,None)
        controller=DwaShadowController(self.config,motion,self.geometry)
        packet=controller.step(self.req,scene,self.rows,reference,now=10.,clock_domain='runtime_live',mono=0.)
        safety=ShadowSafety(self.config,motion,self.geometry,clock_domain='runtime_live',execution=True)
        safety.command('goal',self.req);safety.push_grid(scene.geometry,mono=0.);safety.push_record(scene.record)
        for row in self.rows:safety.push_state(row)
        for i in range(4):
            stamp=10.+i*.1;mono=i*.1
            if i:
                safety.wake.clear()
                safety.push_state(dict(self.rows[-1],stamp_s=stamp))
                self.assertTrue(safety.wake.is_set())
            current=copy.deepcopy(packet);current.update(sequence_id=packet['sequence_id']+i,stamp_s=stamp)
            current['packet_binding']=fingerprint({k:v for k,v in current.items() if k!='packet_binding'})
            safety.push_packet(current,mono=mono)
            now=stamp+.02
            result=safety.tick(now,mono=mono+.02,clock_now=lambda:now)
            self.assertEqual(result['state'],'allowed',result)
            self.assertGreater(result['remaining_validity_s'],.1)
            self.assertAlmostEqual(result['feedback']['stamp_s'],stamp)
        # No new source observation: its age, rather than the wake or packet,
        # must prevent further authorization.
        result=safety.tick(10.47,mono=.47,clock_now=lambda:10.47)
        self.assertEqual(result['state'],'stop_request',result)
        self.assertEqual(result['velocity'],[0.,0.])

    def test_feedback_cannot_age_past_its_limit_while_the_proposal_is_still_valid(self):
        clock=iter([10.,10.16]);result=self.safety.tick(10.,mono=0.,clock_now=lambda:next(clock))
        self.assertEqual(result['reason'],'feedback_expired_during_gate');self.assertEqual(result['velocity'],[0.,0.])

    def test_association_lifetime_limits_authorization_and_expiry_stops(self):
        with patch('navigation.safety.shadow._association_remaining',return_value=.02):
            result=self.safety.tick(10.,mono=0.,clock_now=lambda:10.)
        self.assertEqual(result['state'],'allowed',result)
        self.assertAlmostEqual(result['remaining_association_validity_s'],.02)
        self.assertAlmostEqual(result['remaining_validity_s'],.02)
        with patch('navigation.safety.shadow._association_remaining',side_effect=ValueError('association_expired_during_gate')):
            result=self.safety.tick(10.,mono=.01,clock_now=lambda:10.)
        self.assertEqual(result['reason'],'association_expired_during_gate')
        self.assertEqual(result['velocity'],[0.,0.])

    def test_scene_join_does_not_block_pose_callbacks_or_hide_new_observations(self):
        import threading
        from navigation.safety import shadow
        original=shadow.planning_scene;threads=[]
        def joining(*args,**kwargs):
            received=threading.Event()
            def receive():
                self.safety.push_state(dict(self.rows[-1],stamp_s=10.01))
                blocked=grid();blocked.metadata.update(snapshot_id='during_join',stamp_s=10.01)
                blocked.layers['occupancy'][20,22]=100
                self.safety.push_grid(blocked,mono=.01);received.set()
            if not threads:
                thread=threading.Thread(target=receive);threads.append(thread);thread.start()
                self.assertTrue(received.wait(1.),'Scene assembly held the pose/grid receive lock')
            return original(*args,**kwargs)
        try:
            with patch.object(shadow,'planning_scene',side_effect=joining):
                result=self.safety.tick(10.01,mono=.01)
            self.assertEqual(result['reason'],'motion_sweep_occupied')
            self.assertEqual(result['velocity'],[0.,0.]);self.assertEqual(result['revalidation_attempts'],2)
        finally:
            for thread in threads:thread.join(1.)

    def test_bound_source_verification_does_not_hold_pose_receive_lock(self):
        import threading
        from perception.environment.scene import SceneInputs
        original=SceneInputs.verify;received=threading.Event();threads=[]
        def verify(scene):
            if not threads:
                def receive():self.safety.push_state(dict(self.rows[-1],stamp_s=10.01));received.set()
                thread=threading.Thread(target=receive);threads.append(thread);thread.start()
                self.assertTrue(received.wait(1.),'Source verification held the pose receive lock')
            return original(scene)
        try:
            with patch.object(SceneInputs,'verify',verify):
                result=self.safety.tick(10.01,mono=.01)
            self.assertEqual(result['state'],'allowed',result)
        finally:
            for thread in threads:thread.join(1.)


if __name__=='__main__':unittest.main()
