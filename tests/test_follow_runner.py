"""Runtime follow orchestration uses fresh tracks and a stable owner."""
import unittest
from unittest.mock import patch
from collections import deque
import threading
from types import SimpleNamespace
import numpy as np

from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from perception.environment.context import fingerprint
from runtime.follow import FollowRunner
from tests.test_follow_continuity import reference,scene


class FollowRunnerTests(unittest.TestCase):
    def setUp(self):
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.runner=FollowRunner(load_config(),geometry)
        self.handle=dict(task_id='f',plan_revision=1,goal_id='g',call_id='call')
        self.initial=scene(10.);self.ref=reference(self.initial)
        self.ref.update(task_revision=1,goal_id='g')
        self.ref['reference_id']=fingerprint({k:v for k,v in self.ref.items() if k!='reference_id'})
        self.runner.observe(self.initial)

    def test_updates_keep_handle_but_advance_internal_goal_revision(self):
        first=self.runner.start(self.handle,self.ref,self.initial,now=10.)
        self.assertEqual(first['request']['task_id'],'follow:call')
        self.assertEqual(first['request']['request_id'],'call')
        later=scene(10.1,x=4.4);self.runner.observe(later)
        second=self.runner.update(later,now=10.1)
        self.assertGreater(second['request']['goal_revision'],first['request']['goal_revision'])
        self.assertEqual(second['request']['request_id'],'call')
        self.assertEqual(self.runner.update(None,now=10.2)['operation'],'stop')
        self.runner.clear();self.assertIsNone(self.runner.update(later,now=10.2))

    def test_wrong_task_or_revision_never_starts(self):
        for handle in (dict(self.handle,plan_revision=2),dict(self.handle,goal_id='other')):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                self.runner.start(handle,self.ref,self.initial,now=10.)
        self.assertIsNone(self.runner.policy)

    def test_visual_reconfirmation_preserves_the_measured_corner(self):
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        for i,(x,y) in enumerate([(4.5,2.),(5.,2.),(5.5,2.),(5.5,2.5),(5.5,3.)],1):
            current=scene(10.+i*.2,x,y=y)
            self.runner.observe(current)
            ref=reference(current)
            ref.update(task_revision=1,goal_id='g')
            ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
            self.runner.retarget(ref,current,now=current.record['as_of'])
            action=self.runner.update(current,now=current.record['as_of'])
        # The last metre is northbound. The remaining half metre is on the eastbound leg.
        np.testing.assert_allclose(action['request']['goal'],[5.,2.])

    def test_visual_occlusion_has_a_fixed_deadline_and_visible_can_clear_it(self):
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        def review(value,state):
            return dict(state=state,identity=['tracker_session',7,'a'],
                        scene_binding=value.record['input_fingerprint'],rgb_stamp_s=value.record['as_of']-.02)
        first=scene(10.1);self.runner.observe(first)
        self.runner.visual(review(first,'occluded'),first,now=10.1)
        self.runner.update(first,now=10.1)
        for i in range(2,62):
            current=scene(10.+i*.1);self.runner.observe(current)
            if i==20:self.runner.visual(review(current,'occluded'),current,now=12.)
            result=self.runner.update(current,now=10.+i*.1)
        self.assertEqual(result['reason'],'follow_visual_occlusion_expired')
        self.runner.clear();self.runner.history.rows.clear();self.runner.observe(self.initial)
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        first=scene(10.1);self.runner.observe(first)
        self.runner.visual(review(first,'occluded'),first,now=10.1)
        current=scene(10.2);self.runner.observe(current)
        self.runner.visual(review(current,'visible'),current,now=10.2)
        self.assertIsNone(self.runner.occluded_since)
        self.assertEqual(self.runner.policy.last_visual,10.18)
        with self.assertRaisesRegex(ValueError,'out_of_order'):
            self.runner.visual(review(first,'occluded'),current,now=10.2)

    def test_visual_review_cannot_change_identity_or_use_future_input(self):
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        value=dict(state='visible',identity=['tracker_session',7,'other'],
                   scene_binding=self.initial.record['input_fingerprint'],rgb_stamp_s=9.98)
        with self.assertRaisesRegex(ValueError,'identity_mismatch'):
            self.runner.visual(value,self.initial,now=10.)
        value.update(identity=['tracker_session',7,'a'],rgb_stamp_s=10.01)
        with self.assertRaisesRegex(ValueError,'source_unavailable'):
            self.runner.visual(value,self.initial,now=10.)

    def test_retarget_rejects_stale_or_changed_source_without_mutating_target(self):
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        current=scene(10.1);self.runner.observe(current)
        ref=reference(current);ref.update(task_revision=1,goal_id='g')
        ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
        altered=dict(ref,stamp_s=10.2)
        with self.assertRaisesRegex(ValueError,'reference_invalid'):
            self.runner.retarget(altered,current,now=10.1)
        self.assertEqual(self.runner.policy.target,('tracker_session',7,'a'))
        with self.assertRaisesRegex(ValueError,'expired'):
            self.runner.retarget(ref,current,now=12.2)
        self.runner.retarget(ref,current,now=10.1)
        self.assertEqual(self.runner.policy.last_visual,10.08)
        self.assertEqual(self.runner.update(current,now=10.1)['operation'],'plan')

    def test_delayed_occlusion_does_not_override_newer_observed_track(self):
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        covered=scene(10.1,state='predicted',support=10.);self.runner.observe(covered)
        returned=scene(10.2);self.runner.observe(returned)
        result=self.runner.visual(dict(state='occluded',identity=['tracker_session',7,'a'],
            scene_binding=covered.record['input_fingerprint'],rgb_stamp_s=10.08),returned,now=10.2)
        self.assertEqual(result,'superseded')
        self.assertIsNone(self.runner.occluded_since)
        self.assertEqual(self.runner.policy.last_visual,9.98)
        self.assertEqual(self.runner.update(returned,now=10.2)['operation'],'plan')

    def test_retarget_switches_to_distinct_continuous_track(self):
        from tests.test_navigation_prediction import track, tracked_scene
        self.runner.start(self.handle,self.ref,self.initial,now=10.)
        def two(stamp):
            values=[]
            for name,x in [('a',4.),('b',3.)]:
                value=track(x,2.,stamp=stamp,id=name)
                value.update(state='observed',lifecycle_generation=7)
                values.append(value)
            return tracked_scene(values,stamp=stamp)
        source=two(10.1);self.runner.observe(source)
        record=source.record['people']['data']
        ref=reference(source);ref['track']['identity']=[record['session_id'],record['generation'],'b']
        ref.update(task_revision=1,goal_id='g')
        ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
        current=two(10.2);self.runner.observe(current)
        self.runner.retarget(ref,current,now=10.2)
        switched=self.runner.update(current,now=10.2)
        self.assertEqual(switched['target_identity'],['tracker_session',7,'b'])
        np.testing.assert_allclose(switched['request']['goal'],[2.,2.])

    def test_moving_person_drives_planner_control_and_synthetic_plant(self):
        from navigation.control.runtime import DwaNavigationControl
        from navigation.kinematics import advance,velocity_step
        from navigation.planning.continuity import goal_binding,reference_envelope
        from navigation.planning.motion import load_motion_config
        from navigation.planning.planner import Planner
        from navigation.planning.reference import planning_config
        from navigation.simulation import PlanarWorld
        cfg=planning_config(load_config());motion=load_motion_config();motion['collision_backend']='cpu'
        geometry=self.runner.geometry;world=PlanarWorld('follow');pose=np.array([2.,2.,0.]);velocity=np.zeros(2)
        first=world.scene(.2,pose,geometry);ref=reference(first);ref.update(task_revision=1,goal_id='g')
        ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
        runner=FollowRunner(cfg,geometry);runner.observe(first)
        planner=Planner(cfg,geometry,require_actual_start=True)
        controller=DwaNavigationControl(cfg,motion,geometry,simulation=True)
        previous=None;envelope=None;allowed=0
        for i in range(15):
            t=round(.2+i*.1,3);now=10.+t;current=world.scene(t,pose,geometry)
            if i:runner.observe(current);action=runner.update(current,now=now)
            else:action=runner.start(self.handle,ref,current,now=now)
            self.assertEqual(action['operation'],'plan')
            request=action['request'];binding=goal_binding(request)
            if binding!=previous or i%10==0:
                result=planner.query(request,scene=current,now=now,clock_domain='simulation')
                envelope=reference_envelope(result,request,current,None);previous=binding
            feedback=dict(status='valid',source='synthetic_differential_drive_plant',pose=pose.tolist(),
                velocity=velocity.tolist(),stamp_s=now,clock_domain='simulation',local_identity=['follow',0,'sim_odom'])
            packet=controller.step(request,current,feedback,now=now,clock_domain='simulation',dt=.1,result=envelope)
            allowed+=packet['command']['state']=='allowed'
            after=velocity_step(velocity,np.asarray(packet['command']['velocity']),motion,.1)
            pose=advance(pose,(velocity+after)/2,.1);velocity=after
        self.assertGreater(allowed,0)
        self.assertGreater(pose[0],2.)
        self.assertEqual(runner.update(None,now=11.7)['operation'],'stop')


class FollowOwnershipTests(unittest.TestCase):
    def setUp(self):
        from runtime.command import CommandInbox
        self.events=[];self.chassis=dict(state='armed',status_fresh=True,arm_revision=1)
        self.control=dict(request=None,map_binding=dict(valid=True))
        self.inbox=object.__new__(CommandInbox)
        self.inbox.follow_enabled=True;self.inbox.agent_execution=True;self.inbox.agent_owner=None
        self.inbox.lock=threading.Lock();self.inbox.pending=deque();self.inbox.route_preempted=False
        self.inbox.session=SimpleNamespace(execute=True,status=lambda:dict(chassis=self.chassis),note_command=lambda *_:None)
        self.inbox.safety=SimpleNamespace(command=lambda *args:self.events.append(args))
        self.handle=dict(task_id='f',goal_id='g',plan_revision=1,goal_revision=1,call_id='call')
        from tests.test_follow_recovery import candidate_observation
        from decision.vlm.t2.target import reference as visual_reference
        import base64
        obs=candidate_observation(10.,mapped=True)
        self.ref,_,_=visual_reference(obs,0,'red jacket',task_id='f',revision=1,goal_id='g')
        self.control.update(stamp_s=10.,clock_domain=obs.record['clock_domain'],map_binding=obs.record['map_binding'])
        self.inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(control=self.control,received_age_s=.01),
            follow_candidate=lambda *_:(obs.record,base64.b64encode(obs.image).decode()),
            clear_follow_observations=lambda **_:None,reserve_follow_observations=lambda *_:None)
        self.req=dict(task_id='follow:call',request_id='call',goal_revision=1,frame_id='odom',mode='local',
                      target_reference='base_link',goal=[2.5,2.],valid_until_s=10.5)

    def start(self):
        return self.inbox.agent_follow('start',dict(handle=self.handle,intent=dict(target_description='red jacket',reference=self.ref)),live=True)

    def test_follow_finish_blocks_later_goal_renewal_and_preserves_owner(self):
        self.assertTrue(self.start()['accepted'])
        self.assertTrue(self.inbox.follow_goal(self.handle,self.req))
        self.inbox.agent_follow('finish',dict(call_id='call'),live=True)
        self.assertEqual(self.events[-1],('goal_hold',dict(task_id='follow:call',goal_revision=1)))
        self.assertFalse(self.inbox.follow_goal(self.handle,dict(self.req,goal_revision=2)))
        self.assertTrue(self.inbox.agent_owner['held'])

    def test_follow_finish_requires_observed_stop_before_next_goal(self):
        self.start();self.inbox.follow_goal(self.handle,self.req)
        self.inbox.agent_follow('finish',dict(call_id='call'),live=True)
        next_goal=dict(request_id='next',task_id='f',goal_revision=2,frame_id='map',mode='mapped',
                       target_reference='base_link',goal=[3.,2.,0.],map_snapshot_id='static')
        with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
            self.inbox.agent_live('goal',next_goal)
        self.control.update(request=None,stamp_s=10.,cancelled_request=dict(task_id='follow:call',goal_revision=1),
            localization=dict(health=0,session_id='s',generation=0,frame_id='odom'),
            idle_feedback=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
                local_identity=['s',0,'odom'],stamp_s=10.,velocity=[0.,0.]))
        self.assertTrue(self.inbox.agent_live('goal',next_goal)['accepted'])
        self.assertEqual(self.events[-1],('goal',next_goal))

    def test_cancel_after_follow_hold_still_disarms(self):
        self.start();self.inbox.follow_goal(self.handle,self.req)
        self.inbox.agent_follow('finish',dict(call_id='call'),live=True)
        self.inbox.agent_follow('cancel',dict(call_id='call'),live=True)
        self.assertEqual(self.events[-1],('cancel',dict(task_id='follow:call',goal_revision=1)))
        self.assertIsNone(self.inbox.agent_owner)

    def test_follow_goal_expires_if_agent_stops_polling(self):
        with patch('runtime.command.time.monotonic',return_value=10.):
            self.start();self.assertTrue(self.inbox.follow_goal(self.handle,self.req))
        with patch('runtime.command.time.monotonic',return_value=13.1):
            self.assertFalse(self.inbox.follow_goal(self.handle,self.req))
            self.inbox.follow_failed(self.handle)
        self.assertEqual(self.events[-1][0],'cancel')

    def test_manual_preemption_and_disarm_never_allow_a_late_follow_goal(self):
        self.start();self.assertTrue(self.inbox.follow_goal(self.handle,self.req))
        manual=dict(self.req,request_id='manual',task_id='operator')
        self.inbox.offer('goal',manual)
        self.assertFalse(self.inbox.follow_goal(self.handle,dict(self.req,goal_revision=2)))
        self.assertEqual(self.events[-1],('goal',manual))
        self.assertIsNone(self.inbox.follow_failed(self.handle))
        self.control['request']=None;self.start();self.chassis['state']='disarmed'
        self.assertFalse(self.inbox.follow_goal(self.handle,self.req))
        self.inbox.follow_failed(self.handle)
        self.assertEqual(self.events[-1][0],'cancel')

    def test_start_requires_arm_and_valid_description(self):
        self.chassis['state']='disarmed'
        with self.assertRaisesRegex(ValueError,'manual_arming_required'):self.start()
        self.chassis['state']='armed'
        with self.assertRaisesRegex(ValueError,'intent_invalid'):
            self.inbox.agent_follow('start',dict(handle=self.handle,intent=dict(target_description='')),live=True)
        self.assertEqual(self.events,[])



if __name__=='__main__':unittest.main()
