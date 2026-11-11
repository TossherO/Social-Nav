"""Live intent ownership and arrival evidence; no physical hardware is used."""
from collections import deque
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from runtime.command import CommandInbox
from decision.agent.tools.navigation import NavigationSkill


class LiveIntentTests(unittest.TestCase):
    def setUp(self):
        self.sent=[]
        self.chassis=dict(state='armed',status_fresh=True,arm_revision=1)
        self.control=dict(request=None,map_binding=dict(valid=True),goal_observation={})
        self.inbox=object.__new__(CommandInbox)
        self.inbox.agent_execution=True
        self.inbox.agent_owner=None
        self.inbox.lock=threading.Lock()
        self.inbox.pending=deque()
        self.inbox.route_preempted=False
        self.inbox.session=SimpleNamespace(status=lambda:dict(chassis=self.chassis),note_command=lambda *a:None)
        self.inbox.safety=SimpleNamespace(command=lambda *a:self.sent.append(a))
        self.inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(control=self.control,received_age_s=.01))
        self.goal=dict(request_id='one',task_id='task',goal_revision=1,mode='mapped',frame_id='map',
                       target_reference='base_link',goal=[1.,2.,0.],map_snapshot_id='static')

    def test_requires_manual_arm_and_current_map(self):
        self.chassis['state']='disarmed'
        with self.assertRaisesRegex(ValueError,'manual_arming_required'):self.inbox.agent_live('goal',self.goal)
        self.chassis['state']='armed';self.control['map_binding']['valid']=False
        with self.assertRaisesRegex(ValueError,'live_navigation_state_unavailable'):self.inbox.agent_live('goal',self.goal)
        self.assertEqual(self.sent,[])

    def test_goals_do_not_override_another_owner_or_unfinished_goal(self):
        self.control['request']=dict(request_id='manual')
        with self.assertRaisesRegex(ValueError,'another_goal_is_active'):self.inbox.agent_live('goal',self.goal)
        self.control['request']=None
        self.inbox.agent_live('goal',self.goal)
        self.control['request']=copy.deepcopy(self.goal)
        second=dict(self.goal,request_id='two',goal_revision=2)
        with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):self.inbox.agent_live('goal',second)
        self.control['goal_observation']=dict(status='criteria_held',task_id='task',goal_revision=1)
        self.inbox.agent_live('goal',second)
        self.assertEqual([item[1]['request_id'] for item in self.sent],['one','two'])

    def test_manual_goal_revokes_agent_and_old_cancel_cannot_preempt_it(self):
        self.inbox.agent_live('goal',self.goal)
        manual=dict(self.goal,request_id='manual',task_id='operator')
        self.inbox.offer('goal',manual)
        with self.assertRaisesRegex(ValueError,'agent_goal_ownership_lost'):
            self.inbox.agent_live('cancel',dict(task_id='task',goal_revision=1))
        self.assertEqual(self.sent[-1],('goal',manual))

    def test_hold_requires_owned_current_arrival_and_next_goal_requires_fresh_stop(self):
        self.inbox.agent_live('goal',self.goal)
        self.control['request']=copy.deepcopy(self.goal)
        target=dict(task_id='task',goal_revision=1)
        with self.assertRaisesRegex(ValueError,'agent_goal_not_ready_for_hold'):
            self.inbox.agent_live('hold',target)
        self.control['goal_observation']=dict(status='criteria_held',task_id='task',goal_revision=1)
        self.inbox.agent_live('hold',target)
        self.assertEqual(self.sent[-1],('goal_hold',target))
        second=dict(self.goal,request_id='two',goal_revision=2)
        with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
            self.inbox.agent_live('goal',second)
        self.control.update(request=None,cancelled_request=target,stamp_s=10.,
            localization=dict(health=0,session_id='s',generation=0,frame_id='odom'))
        with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
            self.inbox.agent_live('goal',second)
        self.control['idle_feedback']=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
            local_identity=['s',0,'odom'],stamp_s=10.,velocity=[0.,0.])
        self.control['idle_feedback']['velocity']=[1.,0.]
        with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
            self.inbox.agent_live('goal',second)
        self.control['idle_feedback']['velocity']=[0.,0.]
        self.inbox.agent_live('goal',second)
        self.assertEqual(self.sent[-1],('goal',second))

    def test_cancel_during_hold_still_disarms_and_blocks_old_owner(self):
        self.inbox.agent_live('goal',self.goal)
        self.control.update(request=copy.deepcopy(self.goal),
            goal_observation=dict(status='criteria_held',task_id='task',goal_revision=1))
        target=dict(task_id='task',goal_revision=1)
        self.inbox.agent_live('hold',target)
        self.inbox.agent_live('cancel',target)
        self.assertEqual(self.sent[-1],('cancel',target))
        self.assertIsNone(self.inbox.agent_owner)
        with self.assertRaisesRegex(ValueError,'agent_goal_ownership_lost'):
            self.inbox.agent_live('hold',target)

    def test_safety_hold_clears_goal_but_manual_cancel_and_override_disarm(self):
        from runtime.safety import _apply_command
        events=[];engine=SimpleNamespace(command=lambda *args:events.append(args))
        arm=dict(enabled=True,revision=3);target=dict(task_id='task',goal_revision=1)
        _apply_command(engine,arm,object(),'goal_hold',target)
        self.assertEqual(events[-1],('cancel',target))
        self.assertEqual(arm,dict(enabled=True,revision=3))
        _apply_command(engine,arm,object(),'cancel',target)
        self.assertEqual(arm,dict(enabled=False,revision=4))
        arm['enabled']=True
        _apply_command(engine,arm,object(),'override',dict(manual_override=True,emergency_stop=False))
        self.assertEqual(arm,dict(enabled=False,revision=5))


class LiveCompletionTests(unittest.TestCase):
    def test_exit_cancels_held_goal_but_does_not_repeat_pending_cancel(self):
        from decision.agent.workflows.live import cleanup_goal
        calls=[]
        nav=SimpleNamespace(handle=dict(task_id='task'),cancel_requested=True,
            cancel=lambda handle:calls.append(handle),tick=lambda:dict(stop_observation=True))
        result=cleanup_goal(nav,SimpleNamespace(record=dict(status='held')))
        self.assertTrue(result['requested']);self.assertTrue(result['stop_observed'])
        self.assertEqual(len(calls),1)
        result=cleanup_goal(nav,SimpleNamespace(record=dict(status='stopping')))
        self.assertFalse(result['requested']);self.assertEqual(len(calls),1)

    def test_two_goals_progress_with_held_first_goal_and_explicit_cancel(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.core.session import AgentSession
        from decision.agent.core.store import Store
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback,navigate,point_goal
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value)
        value['scope']='development_prior'
        for place in value['places']:place['status']='manual_annotation'
        nav=ScriptedFeedback();nav.scope='live'
        nav.state.update(scope='live',origin_pose=[4.,4.,0.],origin_frame_id='sim_odom',origin_stamp_s=10.,
            arrival_observation=False,stop_observation=False,chassis_state='armed',chassis_status_fresh=True)
        original_tick=nav.tick
        def tick():
            feedback=original_tick()
            nav.state['origin_stamp_s']=feedback['stamp_s']
            return nav.snapshot()
        nav.tick=tick
        holds=[]
        def hold(handle):
            holds.append(copy.deepcopy(handle))
            nav.state.update(status='stopped',stopped=True,arrival_observation=False,stop_observation=True)
        nav.hold=hold
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('two stops',Catalog(value),nav,store)
                session.call('record_task',goals=[point_goal('elevator'),point_goal('lobby_entrance')])
                self.assertEqual(navigate(session,'elevator')['status'],'accepted')
                nav.arrive();nav.state.update(arrival_observation=True,origin_pose=copy.deepcopy(nav.target_pose))
                session.tick()
                self.assertEqual(session.skill.record['status'],'holding')
                session.tick()
                self.assertEqual(session.skill.record['status'],'held')
                self.assertEqual(session.call('check_goal',goal_id='elevator')['status'],'completed')
                nav.state.update(stop_observation=False,arrival_observation=False)
                self.assertEqual(navigate(session,'lobby_entrance')['status'],'accepted')
                self.assertEqual(len(holds),1)
                self.assertEqual(session.task_goals.goals[1]['status'],'pending')
                session.event('cancel')
                self.assertEqual(nav.state['status'],'stopping')
            finally:store.close()

    def test_cancelled_goal_needs_fresh_idle_localization_not_command_ack(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.tools.places import Catalog
        from decision.agent.adapters.shadow import RuntimeLiveNavigation
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value)
        value.update(scope='development_prior',map_snapshot_id='static',localization_map_id='map-id')
        catalog=Catalog(value)
        binding=dict(valid=True,map_id='map-id',map_frame='sim_odom')
        handle=dict(task_id='task',goal_revision=1,call_id='call')
        feedback=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
                      local_identity=['s',0,'odom'],stamp_s=10.,pose=[1.,2.,0.],velocity=[0.,0.])
        control=dict(clock_domain='runtime_live',stamp_s=10.,request=dict(request_id='call'),feedback=feedback,
                     localization=dict(session_id='s',generation=0,frame_id='odom',health=0),map_binding=binding,
                     goal_observation=dict(status='criteria_held',task_id='task',goal_revision=1,source=feedback['source']))
        obs=dict(local_identity=['s',0,'odom'],clock_domain='runtime_live',map_binding=binding,
                 map_base_pose=[1.,2.,0.],stamp_s=10.)
        state=dict(control=control,observation=obs,received_age_s=.01,
                   chassis=dict(state='armed',status_fresh=True,arm_revision=3),
                   agent_owner=dict(request_id='call',arm_revision=3))
        client=SimpleNamespace(snapshot=lambda **kwargs:state,exchange=lambda *args:dict(accepted=True))
        nav=RuntimeLiveNavigation(catalog,client);nav.handle=handle
        self.assertTrue(nav.snapshot()['arrival_observation'])
        control['request']=None
        self.assertEqual(nav.snapshot()['status'],'pending')
        control['request']=dict(request_id='call')
        nav.cancel(handle)
        control.update(request=None,feedback={},cancelled_request=dict(task_id='task',goal_revision=1),idle_feedback=None)
        state['agent_owner']=None
        self.assertFalse(nav.snapshot()['stop_observation'])
        control['idle_feedback']=feedback
        self.assertTrue(nav.snapshot()['stop_observation'])
        control['stamp_s']=11.
        self.assertFalse(nav.snapshot()['stop_observation'])

    def test_arrival_requires_hold_then_fresh_stop(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value)
        value['scope']='development_prior'
        for place in value['places']:place['status']='manual_annotation'
        catalog=Catalog(value)
        nav=ScriptedFeedback();nav.scope='live';nav.state['scope']='live'
        nav.state.update(origin_pose=[4.,4.,0.],origin_frame_id='sim_odom',arrival_observation=False,stop_observation=False,
                         chassis_state='armed',chassis_status_fresh=True)
        holds=[]
        nav.hold=lambda handle:holds.append(copy.deepcopy(handle))
        skill=NavigationSkill(nav,catalog,'task',lambda *args:None)
        requirement=dict(kind='point',target_id='elevator')
        destination=dict(goal_id='goal',place_id='elevator',frame_id='sim_odom',pose=[5.,4.,0.],
                         view_id='view',task_goal_binding='binding',requirement=requirement)
        result=skill.start(destination,1)
        self.assertEqual(result['status'],'accepted')
        handle=skill.record['handle']
        nav.state.update(status='arrived',arrival_observation=True,stopped=True,origin_pose=[5.,4.,0.],
                         goal=dict(task_id='task',goal_revision=handle['goal_revision']))
        skill.tick()
        self.assertEqual(skill.record['status'],'holding')
        self.assertEqual(skill.record['completed'],[])
        self.assertEqual(holds,[handle])
        nav.state.update(status='stopped',arrival_observation=False,stop_observation=True)
        skill.tick()
        self.assertEqual(skill.record['status'],'held')
        self.assertEqual(skill.record['handle'],handle)
        proof=skill.record['completed'][0]
        self.assertEqual(proof['evidence_kind'],'localization_estimate')
        self.assertFalse(proof['execution_feedback'])
        self.assertFalse(proof['physical_acceptance'])
        skill.cancel(reason='operator_cancel')
        self.assertEqual(skill.record['status'],'stopping')
        self.assertEqual(nav.state['status'],'stopping')

    def test_manual_disarm_invalidates_agent_context_before_another_goal(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value);value['scope']='development_prior'
        for place in value['places']:place['status']='manual_annotation'
        nav=ScriptedFeedback();nav.scope='live';nav.state.update(scope='live',chassis_state='armed',chassis_status_fresh=True)
        skill=NavigationSkill(nav,Catalog(value),'task',lambda *args:None)
        destination=dict(goal_id='goal',place_id='elevator',frame_id='sim_odom',pose=[5.,4.,0.],
            view_id='view',task_goal_binding='binding',requirement=dict(kind='point',target_id='elevator'))
        self.assertEqual(skill.start(destination,1)['status'],'accepted')
        nav.state['chassis_state']='disarmed'
        skill.tick()
        self.assertTrue(skill.record['context_lost'])
        self.assertEqual(skill.record['status'],'stopping')
        self.assertEqual(skill.start(destination,1)['reason'],'navigation_context_unavailable')

    def test_explicit_rejection_is_not_treated_as_an_uncertain_submission(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.tools.navigation import IntentRejected
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value);value['scope']='development_prior'
        for place in value['places']:place['status']='manual_annotation'
        nav=ScriptedFeedback();nav.scope='live';nav.state.update(scope='live',chassis_state='armed',chassis_status_fresh=True)
        nav.start=lambda *_: (_ for _ in ()).throw(IntentRejected('manual_arming_required'))
        nav.cancel=lambda *_: self.fail('An explicitly rejected goal must not be cancelled as accepted')
        skill=NavigationSkill(nav,Catalog(value),'task',lambda *args:None)
        destination=dict(goal_id='goal',place_id='elevator',frame_id='sim_odom',pose=[5.,4.,0.],
            view_id='view',task_goal_binding='binding',requirement=dict(kind='point',target_id='elevator'))
        result=skill.start(destination,1)
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['reason'],'manual_arming_required')
        self.assertIsNone(skill.record['handle'])

    def test_rejected_live_submission_restores_previous_adapter_handle(self):
        from decision.agent.adapters.shadow import RuntimeLiveNavigation
        from decision.agent.tools.navigation import IntentRejected
        nav=object.__new__(RuntimeLiveNavigation)
        nav.catalog=SimpleNamespace(value=dict(frame_id='map',map_snapshot_id='map-id'))
        nav.handle=dict(call_id='old');nav.request=dict(request_id='old');nav.cancel_requested=True
        nav.client=SimpleNamespace(exchange=lambda *_:(_ for _ in ()).throw(IntentRejected('manual_arming_required')))
        with self.assertRaises(IntentRejected):
            nav.start(dict(pose=[1.,2.,0.]),dict(call_id='new',task_id='task',goal_revision=2))
        self.assertEqual(nav.handle,dict(call_id='old'))
        self.assertEqual(nav.request,dict(request_id='old'))
        self.assertTrue(nav.cancel_requested)

    def test_blocked_planner_notifies_once_without_issuing_an_extra_command(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value);value['scope']='development_prior'
        for place in value['places']:place['status']='manual_annotation'
        nav=ScriptedFeedback();nav.scope='live';nav.state.update(scope='live',chassis_state='armed',chassis_status_fresh=True)
        events=[];skill=NavigationSkill(nav,Catalog(value),'task',lambda kind,_:events.append(kind))
        destination=dict(goal_id='goal',place_id='elevator',frame_id='sim_odom',pose=[5.,4.,0.],
            view_id='view',task_goal_binding='binding',requirement=dict(kind='point',target_id='elevator'))
        self.assertEqual(skill.start(destination,1)['status'],'accepted')
        nav.state['planning_reason']='no_admissible_velocity'
        self.assertFalse(skill.tick())
        nav.t=10.2
        self.assertTrue(skill.tick())
        self.assertEqual(skill.record['status'],'blocked')
        self.assertIsNotNone(skill.record['handle'])
        self.assertFalse(skill.tick())
        self.assertEqual(events.count('navigation_blocked'),1)
        nav.state['planning_reason']=None
        self.assertTrue(skill.tick())
        self.assertEqual(skill.record['status'],'running')

    def test_manual_disarm_stops_agent_loop_without_another_model_call(self):
        from decision.agent import CONFIG_DIR
        from decision.agent.core.loop import AgentLoop
        from decision.agent.core.session import AgentSession
        from decision.agent.core.store import Store
        from decision.agent.tools.places import Catalog
        from tests.test_agent import ScriptedFeedback,reply
        value=copy.deepcopy(Catalog.load(CONFIG_DIR/'simulation.json').value);value['scope']='development_prior'
        nav=ScriptedFeedback();nav.scope='live';nav.state.update(scope='live',chassis_state='armed',chassis_status_fresh=True)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('test',Catalog(value),nav,store)
                nav.state['chassis_state']='disarmed'
                actor=AgentLoop(session,lambda *_:reply('get_navigation_state'))
                try:
                    actor.tick()
                    self.assertEqual(session.state['status'],'needs_clarification')
                    self.assertEqual(session.state['reason'],'chassis_disarmed_or_status_unavailable')
                    self.assertEqual(actor.requests,0)
                finally:actor.close()
            finally:store.close()
