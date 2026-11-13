"""Follow selection must remain tied to continuously supported 3D tracks."""
import copy
import unittest

import numpy as np

from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.tasks import FollowTrackHistory, TaskPolicy
from perception.environment.context import fingerprint
from tests.test_navigation_prediction import track, tracked_scene


def scene(stamp, x=4., *, support=None, state='observed', generation=7, present=True, velocity=(.5,0.,0.)):
    value=track(x,2.,velocity,stamp=stamp if support is None else support)
    value.update(state=state,lifecycle_generation=generation)
    return tracked_scene([value] if present else [],stamp=stamp)


def reference(value, task_id='f'):
    row=value.record['people']['data'];raw=row['tracks'][0]
    result=dict(task_id=task_id,scene_binding=value.record['input_fingerprint'],stamp_s=value.record['as_of'],
        rgb_stamp_s=value.record['as_of']-.02,clock_domain=value.record['clock_domain'],
        local_identity=[value.record[k] for k in ('session_id','generation','frame_id')],
        track=dict(identity=[row['session_id'],row['generation'],raw['track_id']],
                   lifecycle_generation=raw['lifecycle_generation'],people_fingerprint=row['input_fingerprint'],
                   people_stamp_s=row['stamp_s'],support_stamp_s=raw['last_observed_stamp_s']))
    result['reference_id']=fingerprint(result)
    return result


class FollowContinuityTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.history=FollowTrackHistory(self.config)

    def test_delayed_selection_and_continuous_updates_do_not_refresh_image_time(self):
        initial=scene(10.);ref=reference(initial,'follow');self.history.add(initial)
        for i in range(1,51):
            current=scene(10.+i*.1,4.+i*.02);self.history.add(current)
        policy=TaskPolicy('follow','follow',self.config,self.geometry)
        result=policy.start_continuous_follow(ref,current,self.history,now=15.)
        self.assertEqual(result['operation'],'plan')
        self.assertEqual(policy.last_visual,9.98)
        self.assertEqual(policy.response_stamp,10.)
        self.assertEqual(result['request']['completion_policy'],'continuous')
        self.assertLessEqual(result['request']['valid_until_s'],15.5)
        ticket=policy.observe(current,pid_to_track={0:['tracker_session',7,'a']})
        with self.assertRaisesRegex(ValueError,'separate_visual_revalidation'):
            policy.respond(ticket,dict(id=0,state='visible'),scene=current,now=15.)
        later=scene(15.1,5.4);self.history.add(later);updated=policy.update(later,now=15.1,history=self.history)
        self.assertEqual(updated['operation'],'plan')
        self.assertGreater(updated['request']['goal_revision'],result['request']['goal_revision'])
        self.assertFalse(updated['task_completed'])

    def test_person_waits_for_selection_then_starts_walking(self):
        initial=scene(10.,velocity=(0.,0.,0.));ref=reference(initial);self.history.add(initial)
        for i in range(1,51):
            current=scene(10.+i*.1,velocity=(0.,0.,0.));self.history.add(current)
        policy=TaskPolicy('f','follow',self.config,self.geometry)
        waiting=policy.start_continuous_follow(ref,current,self.history,now=15.)
        self.assertEqual(waiting['operation'],'plan')
        moving=scene(15.1,x=4.4,velocity=(.5,0.,0.));self.history.add(moving)
        following=policy.update(moving,now=15.1,history=self.history)
        self.assertEqual(following['operation'],'plan')
        self.assertGreater(following['request']['goal_revision'],waiting['request']['goal_revision'])
        self.assertEqual(policy.last_visual,9.98)

    def test_predicted_track_expires_and_cannot_resume_automatically(self):
        initial=scene(10.);self.history.add(initial);policy=TaskPolicy('f','follow',self.config,self.geometry)
        policy.start_continuous_follow(reference(initial),initial,self.history,now=10.)
        brief=scene(10.3,support=10.,state='predicted')
        self.history.add(brief)
        self.assertEqual(policy.update(brief,now=10.3,history=self.history)['operation'],'plan')
        late=scene(10.9,support=10.,state='predicted')
        self.history.add(late)
        self.assertEqual(policy.update(late,now=10.9,history=self.history)['operation'],'stop')
        self.assertEqual(policy.update(scene(11.),now=11.,history=self.history)['operation'],'stop')

    def test_gap_identity_reuse_and_missing_source_refuse_start(self):
        initial=scene(10.);ref=reference(initial);self.history.add(initial)
        changed=scene(10.1,generation=8);self.history.add(changed)
        with self.assertRaisesRegex(ValueError,'continuity'):
            TaskPolicy('f','follow',self.config,self.geometry).start_continuous_follow(ref,changed,self.history,now=10.1)
        history=FollowTrackHistory(self.config);history.add(initial);later=scene(11.);history.add(later)
        with self.assertRaisesRegex(ValueError,'continuity'):
            history.validate(ref,later,now=11.)
        history=FollowTrackHistory(self.config);history.add(later)
        with self.assertRaisesRegex(ValueError,'history_unavailable'):
            history.validate(ref,later,now=11.)
        modified=copy.deepcopy(ref);modified['stamp_s']+=1
        with self.assertRaisesRegex(ValueError,'reference_invalid'):
            history.validate(modified,later,now=11.)

    def test_intermediate_track_loss_is_not_hidden_by_later_reappearance(self):
        initial=scene(10.);self.history.add(initial);policy=TaskPolicy('f','follow',self.config,self.geometry)
        policy.start_continuous_follow(reference(initial),initial,self.history,now=10.)
        missing=scene(10.1,present=False);self.history.add(missing)
        returned=scene(10.2);self.history.add(returned)
        result=policy.update(returned,now=10.2,history=self.history)
        self.assertEqual(result['reason'],'follow_track_continuity_lost')
        self.assertEqual(policy.update(returned,now=10.2,history=self.history)['operation'],'stop')

    def test_delayed_selection_does_not_reuse_a_lost_track(self):
        initial=scene(10.);self.history.add(initial);ref=reference(initial)
        for i in range(1,40):
            stamp=10.+i*.1
            current=scene(stamp,present=i<12,state='predicted' if i in (10,11) else 'observed',
                          support=10.9 if i in (10,11) else None)
            self.history.add(current)
        with self.assertRaisesRegex(ValueError,'continuity'):
            TaskPolicy('f','follow',self.config,self.geometry).start_continuous_follow(ref,current,self.history,now=13.9)

    def test_turning_cancellation_and_expiry(self):
        from navigation.planning.inputs import request_status
        initial=scene(10.);self.history.add(initial);policy=TaskPolicy('f','follow',self.config,self.geometry)
        result=policy.start_continuous_follow(reference(initial),initial,self.history,now=10.)
        self.assertIsNone(request_status(result['request'],initial,10.,'synthetic'))
        self.assertEqual(request_status(result['request'],initial,10.51,'synthetic'),'request_expired')
        turning=scene(10.1,velocity=(0.,.5,0.));self.history.add(turning)
        result=policy.update(turning,now=10.1,history=self.history)
        np.testing.assert_allclose(result['request']['goal'],[4.,.5])
        self.assertFalse(policy.cancel()['task_completed'])
        self.assertEqual(policy.update(turning,now=10.1,history=self.history)['reason'],'task_cancelled')

    def test_continuous_request_uses_existing_planner_and_safety(self):
        from navigation.control.runtime import DwaNavigationControl
        from navigation.planning.continuity import reference_envelope
        from navigation.planning.motion import load_motion_config
        from navigation.planning.planner import Planner
        from navigation.planning.reference import planning_config
        initial=scene(10.);self.history.add(initial);policy=TaskPolicy('f','follow',self.config,self.geometry)
        request=policy.start_continuous_follow(reference(initial),initial,self.history,now=10.)['request']
        config=planning_config(self.config)
        result=Planner(config,self.geometry,require_actual_start=True).query(request,scene=initial,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result.get('reason'))
        envelope=reference_envelope(result,request,initial,None)
        feedback=dict(status='valid',source='synthetic',stamp_s=10.,clock_domain='synthetic',
                      local_identity=['test',0,'odom'],pose=[2,2,0],velocity=[0,0])
        control=DwaNavigationControl(config,load_motion_config(),self.geometry,simulation=True)
        outcome=control.step(request,initial,feedback,now=10.,clock_domain='synthetic',dt=.1,result=envelope)
        self.assertEqual(outcome['command']['state'],'allowed')
        self.assertFalse(outcome['goal']['task_completed'])


if __name__=='__main__':unittest.main()
