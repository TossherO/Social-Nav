"""Continuous seating updates retain task ownership and stop on invalid evidence."""
import copy
import threading
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from decision.agent.core.seating import poll_updates
from runtime.command import CommandInbox


class SeatingUpdateTests(unittest.TestCase):
    def test_manual_retarget_updates_preview_and_clears_arrival_evidence(self):
        from decision.agent.adapters.manual import ManualNavigation
        nav=object.__new__(ManualNavigation)
        old=dict(call_id='call',goal_revision=1);new=dict(call_id='call',goal_revision=2)
        nav.handle=copy.deepcopy(old);nav.status='running';nav.follow_active=False
        nav.request=dict(pose=[1.,2.,0.],require_yaw=True)
        nav.confirmation={'old':True};nav.automatic_evidence={'old':True}
        nav.arrival_since=nav.arrival_last_stamp=5.
        nav.retarget(dict(pose=[2.,2.,1.]),old,new)
        self.assertEqual(nav.request['pose'],[2.,2.,1.])
        self.assertEqual(nav.handle,new)
        self.assertIsNone(nav.confirmation);self.assertIsNone(nav.automatic_evidence)
        with self.assertRaisesRegex(ValueError,'no longer active'):
            nav.retarget(dict(pose=[3.,2.,1.]),old,new)

    def test_live_retarget_requires_same_owner_and_fresh_armed_state(self):
        inbox=object.__new__(CommandInbox)
        inbox.agent_execution=True;inbox.lock=threading.Lock();inbox.pending=deque()
        inbox.agent_owner=dict(request_id='call',task_id='task',goal_revision=1,arm_revision=4)
        goal=dict(request_id='call',task_id='task',goal_revision=1,mode='mapped',frame_id='map',
                  target_reference='base_link',goal=[1.,2.,0.],map_snapshot_id='map-a')
        control=dict(request=copy.deepcopy(goal),map_binding=dict(valid=True))
        inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(control=control,received_age_s=.1))
        inbox.session=SimpleNamespace(status=lambda:dict(chassis=dict(state='armed',status_fresh=True,arm_revision=4)),
                                      note_command=Mock())
        events=[];inbox.safety=SimpleNamespace(command=lambda *v:events.append(v))
        next_goal=dict(goal,goal_revision=2,goal=[1.2,2.,0.])
        with self.assertRaisesRegex(ValueError,'ownership_or_state'):
            inbox.agent_live_retarget(dict(previous_revision=1,goal=dict(next_goal,task_id='other')))
        self.assertEqual(events,[])
        self.assertTrue(inbox.agent_live_retarget(dict(previous_revision=1,goal=next_goal))['accepted'])
        self.assertEqual(events[-1],('goal',next_goal))
        with self.assertRaisesRegex(ValueError,'ownership_or_state'):
            inbox.agent_live_retarget(dict(previous_revision=1,goal=next_goal))
        inbox.agent_live_cancel_owned(dict(task_id='task',call_id='call'))
        self.assertEqual(events[-1],('cancel',dict(task_id='task',goal_revision=2)))

    def test_update_applies_new_pose_only_for_current_stage(self):
        context=dict(clock_domain='live',local_identity=['session',1,'odom'],map_binding=dict(valid=True,map_id='m',map_frame='map'))
        handle=dict(task_id='task',goal_id='seat',call_id='call',goal_revision=1)
        active=dict(selection_id='selection',goal_id='seat',revision=1,context=context,
                    observation_binding='original',sequence=1,handle=handle)
        result=dict(selection_id='selection',sequence=2,status='selected',observation_binding='original',
            local_identity=context['local_identity'],clock_domain='live',map_binding=context['map_binding'],
            geometry_stamp_s=10.,pose=[2.,1.,0.],mapped_edge={})
        feedback=dict(stamp_s=10.,clock_domain='live',session_id='session',generation=1,frame_id='odom',map_binding=context['map_binding'])
        goal=dict(binding='binding',requirement={},selected=dict(pose=[1.,1.,0.],handle=handle))
        skill=SimpleNamespace(record=dict(handle=handle,status='running'),
            retarget=Mock(return_value=dict(status='accepted',handle=dict(handle,goal_revision=2))))
        nav=SimpleNamespace(value={'seating':result},snapshot=lambda:feedback,client=SimpleNamespace(exchange=Mock()))
        session=SimpleNamespace(seating_active=active,seating_pending=None,state=dict(status='waiting_navigation',revision=1),
            skill=skill,navigation=nav,task_goals=SimpleNamespace(current=lambda _:goal),catalog=SimpleNamespace(value={'frame_id':'map'}),
            spatial_view=SimpleNamespace(record={'view_id':'view'}),prepare_current_images=lambda:None,
            check_candidate=lambda *_:dict(status='available',static_footprint=dict(status='free'),local_footprint=dict(status='free')),
            observation=None,_save=Mock(),stop=Mock())
        poll_updates(session)
        self.assertEqual(skill.retarget.call_count,1)
        self.assertEqual(goal['selected']['pose'],[2.,1.,0.])
        self.assertEqual(session.seating_active['handle']['goal_revision'],2)
        # A delayed response from an old task cannot replace the active goal.
        result['sequence']=3;result['local_identity']=['other',1,'odom']
        skill.record['handle']=session.seating_active['handle']
        poll_updates(session)
        self.assertEqual(skill.retarget.call_count,1)
        session.stop.assert_called_once_with('needs_clarification','seating_proposal_expired_or_context_changed')

    def test_hold_confirmation_retains_result_then_rechecks_after_stop(self):
        context=dict(clock_domain='live',local_identity=['session',1,'odom'],map_binding=dict(valid=True,map_id='m',map_frame='map'))
        handle=dict(task_id='task',goal_id='seat',call_id='call',goal_revision=1)
        active=dict(selection_id='selection',goal_id='seat',revision=1,context=context,
                    observation_binding='original',sequence=1,handle=handle)
        result=dict(selection_id='selection',sequence=2,status='selected',observation_binding='original',
            local_identity=context['local_identity'],clock_domain='live',map_binding=context['map_binding'],
            geometry_stamp_s=10.,pose=[2.,1.,0.],mapped_edge={})
        feedback=dict(stamp_s=10.,clock_domain='live',session_id='session',generation=1,frame_id='odom',map_binding=context['map_binding'])
        goal=dict(binding='binding',requirement={},selected=dict(pose=[1.,1.,0.],handle=handle))
        skill=SimpleNamespace(record=dict(handle=handle,status='holding',completed=[]),
            start=Mock(return_value=dict(status='accepted',handle=dict(handle,call_id='new',goal_revision=2))))
        nav=SimpleNamespace(scope='live',value={'seating':result},snapshot=lambda:feedback,client=SimpleNamespace(exchange=Mock()))
        session=SimpleNamespace(seating_active=active,seating_pending=None,state=dict(status='waiting_navigation',revision=1),
            skill=skill,navigation=nav,task_goals=SimpleNamespace(current=lambda _:goal),catalog=SimpleNamespace(value={'frame_id':'map'}),
            spatial_view=SimpleNamespace(record={'view_id':'view'}),prepare_current_images=lambda:None,
            check_candidate=lambda *_:dict(status='available',static_footprint=dict(status='free'),local_footprint=dict(status='free')),
            observation=None,_save=Mock(),stop=Mock())
        poll_updates(session)
        self.assertEqual(active['sequence'],1)
        skill.start.assert_not_called()
        skill.record['status']='held'
        poll_updates(session)
        self.assertEqual(active['sequence'],2)
        skill.start.assert_called_once()
        self.assertEqual(goal['selected']['handle']['call_id'],'new')


if __name__=='__main__':unittest.main()
