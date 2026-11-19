"""Task monitor integration and causal progress evidence."""
import copy
import tempfile
import threading
import unittest
from pathlib import Path

from decision.agent import ROOT
from decision.agent.core.loop import AgentLoop
from decision.agent.core.monitor import TaskMonitor
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback, navigate, point_goal, query_map, reply


class AgentMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'session.sqlite')
        self.catalog = Catalog.load(ROOT/'decision/agent/configs/simulation.json')
        self.nav = ScriptedFeedback()
        self.session = AgentSession('到电梯并留意入口', self.catalog, self.nav, self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def start(self, **settings):
        self.session.call('record_task', goals=[point_goal('elevator')])
        result = self.session.call('set_navigation_monitor', goal_id='elevator', focus='观察入口', **settings)
        self.assertEqual(result['status'], 'configured')
        self.assertEqual(navigate(self.session, 'elevator')['status'], 'accepted')
        self.session.call('wait_for_event', event='navigation')
        self.session.tick()

    def test_periodic_review_keeps_the_active_goal_and_coalesces(self):
        self.start(interval_s=5)
        handle = copy.deepcopy(self.nav.state['handle'])
        self.nav.t = 5.1
        self.session.tick()
        self.assertEqual(self.session.state['status'], 'ready')
        event = self.session.state['history'][-1]['value']
        self.assertEqual(event['handle'], handle)
        self.assertEqual(event['reasons'][0]['kind'], 'semantic_review_due')
        self.assertEqual(self.nav.state['status'], 'running')
        self.assertEqual(self.session.call('wait_for_event', event='navigation')['status'], 'waiting')
        self.session.tick()
        self.assertEqual(self.session.state['status'], 'waiting_navigation')
        self.assertEqual(self.nav.state['handle'], handle)

    def test_region_transition_is_reported_once(self):
        self.start(region_ids=['lobby_area'])
        self.nav.state['pose'] = [7., 4., 0.]
        self.session.tick()
        event = self.session.state['history'][-1]['value']
        self.assertEqual(event['reasons'][0]['region_id'], 'lobby_area')
        self.session.call('wait_for_event', event='navigation')
        self.session.tick()
        self.assertEqual(self.session.state['status'], 'waiting_navigation')

    def test_new_monitor_events_do_not_invalidate_pending_model(self):
        self.start(interval_s=5)
        entered, gate = threading.Event(), threading.Event()
        def work(*_):
            entered.set()
            gate.wait(2)
            return reply('wait_for_event', event='navigation')
        actor = AgentLoop(self.session, work)
        try:
            self.nav.t = 5.1
            actor.tick()
            self.assertTrue(entered.wait(1))
            self.nav.t = 10.5
            actor.tick()
            self.assertTrue(actor.model.busy)
            gate.set()
            actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(self.session.state['status'], 'waiting_navigation')
            self.assertEqual(self.nav.state['status'], 'running')
            actor.tick()
            self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='model_result_discarded'").fetchone()[0], 0)
            self.assertEqual(self.nav.state['status'], 'running')
        finally:
            gate.set()
            if actor.model.thread is not None:
                actor.model.thread.join(1)
            actor.close()

    def test_unsupported_stage_cannot_start_navigation(self):
        goal = dict(goal_id='follow', kind='skill', target_id='', stage_type='follow',
                    description='跟随蓝衣引导员到电梯')
        self.assertEqual(self.session.call('record_task', goals=[goal, point_goal('elevator')])['status'], 'recorded')
        result = self.session.call('navigate', goal_id='follow', pose=[5., 4., 0.],
                                   view_id='unused', orientation_source='autonomous')
        self.assertEqual(result['reason'], 'current_stage_does_not_support_navigation')
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertEqual(self.session.task_goals.goals[0]['status'], 'pending')
        self.assertEqual(len(self.session.task_goals.goals), 2)

    def test_rejected_goal_during_review_does_not_overwrite_active_selection(self):
        self.start(interval_s=5)
        selected = copy.deepcopy(self.session.task_goals.goals[0]['selected'])
        self.nav.t = 5.1
        self.session.tick()
        view = query_map(self.session)
        result = self.session.call('navigate', goal_id='elevator', pose=[5., 4., 0.],
                                   view_id=view, orientation_source='autonomous')
        self.assertEqual(result['reason'], 'cancel_and_confirm_existing_call_first')
        self.assertEqual(self.session.task_goals.goals[0]['selected'], selected)
        self.nav.arrive()
        self.session.tick()
        self.assertEqual(self.session.call('check_goal', goal_id='elevator')['status'], 'completed')
        self.assertIsNone(self.session.monitor.snapshot())

    def test_replacement_waits_for_stop_and_clears_monitor(self):
        self.start(interval_s=5)
        self.session.event('replace', '改去大厅')
        self.assertIsNotNone(self.session.skill.record['handle'])
        self.assertEqual(self.session.state['revision'], 1)
        self.nav.stopped()
        self.session.tick()
        self.assertEqual(self.session.state['revision'], 2)
        self.assertIsNone(self.session.monitor.snapshot())

    def test_progress_uses_continuous_pose_and_does_not_call_waiting_a_failure(self):
        monitor = TaskMonitor(self.catalog)
        monitor.configure('g', '检查进展', progress_interval_s=10)
        navigation = dict(handle={'call_id':'c'}, status='running', goal_id='g', context_lost=False, connected=True)
        def update(stamp, pose, **extra):
            value = dict(stamp_s=stamp, frame_id='sim_odom', pose=pose, localization_status='valid', handle={'call_id':'c'})
            value.update(extra)
            monitor.update(navigation, value)
        for stamp in range(11):
            update(stamp, [0., 0., 0.], planning_reason='waiting_for_pedestrian')
        event = monitor.take()
        self.assertEqual(event['reasons'][0]['kind'], 'no_measured_progress')
        self.assertEqual(event['reasons'][0]['planning_reason'], 'waiting_for_pedestrian')
        monitor.reset_runtime()
        for stamp in range(11):
            update(stamp, [0., 0., stamp*.2])
        self.assertIsNone(monitor.take())
        monitor.reset_runtime()
        update(0, [0., 0., 0.])
        update(11, [0., 0., 0.])
        self.assertIsNone(monitor.take())
        monitor.reset_runtime()
        for stamp in range(11):
            update(stamp, [0., 0., 0.])
        update(11, [.2, 0., 0.])
        self.assertIsNone(monitor.take())
        monitor.reset_runtime()
        update(0, [0., 0., 0.], handle={'call_id':'other'})
        self.assertIsNone(monitor.take())

    def test_region_requires_current_pose_and_matching_handle(self):
        monitor = TaskMonitor(self.catalog)
        monitor.configure('g', '关注大厅', region_ids=['lobby_area'])
        nav = dict(handle={'call_id':'c'}, status='running', goal_id='g', context_lost=False, connected=True)
        value = dict(stamp_s=20., origin_stamp_s=19., origin_frame_id='sim_odom',
                     origin_pose=[7., 4., 0.], handle={'call_id':'c'}, localization_status='valid')
        monitor.update(nav, value)
        self.assertIsNone(monitor.take())
        value.update(stamp_s=20.1, origin_stamp_s=20.1, handle={'call_id':'old'})
        monitor.update(nav, value)
        self.assertIsNone(monitor.take())
        value['handle'] = {'call_id':'c'}
        monitor.update(nav, value)
        self.assertEqual(monitor.take()['reasons'][0]['kind'], 'region_entered')

    def test_conditional_review_serializes_new_observations_and_resets_on_exit(self):
        monitor=TaskMonitor(self.catalog)
        monitor.configure('g','观察举手',continuous_region_id='lobby_area',review_mode='after_result')
        nav=dict(handle={'call_id':'c'},status='running',goal_id='g',context_lost=False,connected=True)
        feedback=dict(handle={'call_id':'c'},stamp_s=20.,origin_stamp_s=20.,origin_frame_id='sim_odom',
                      origin_pose=[0.,0.,0.],localization_status='valid')
        monitor.update(nav,feedback)
        self.assertEqual(monitor.snapshot()['conditional_review']['region_state'],'outside')
        self.assertIsNone(monitor.take())
        feedback.update(stamp_s=21.,origin_stamp_s=21.,origin_pose=[7.,4.,0.])
        monitor.update(nav,feedback)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'conditional_review_due')
        monitor.observe('frame-2',21.)
        self.assertIsNone(monitor.take())  # no second review while one is being judged
        self.assertTrue(monitor.await_next_observation('frame-1'))
        monitor.observe('frame-1',21.)
        self.assertIsNone(monitor.take())
        monitor.observe('frame-2',21.1)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'conditional_review_due')
        epoch=monitor.region_epoch
        feedback.update(stamp_s=22.,origin_stamp_s=22.,origin_pose=[0.,0.,0.])
        monitor.update(nav,feedback)
        self.assertGreater(monitor.region_epoch,epoch)
        self.assertEqual(monitor.snapshot()['conditional_review']['region_state'],'outside')
        self.assertFalse(monitor.await_next_observation('frame-2'))
        monitor.observe('frame-3',22.)
        self.assertIsNone(monitor.take())
        feedback.update(stamp_s=23.,origin_stamp_s=23.,origin_pose=[7.,4.,0.])
        monitor.update(nav,feedback)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'conditional_review_due')
        feedback.update(stamp_s=24.,input_stale=True)
        monitor.update(nav,feedback)
        self.assertEqual(monitor.snapshot()['conditional_review']['region_state'],'unknown')

    def test_conditional_review_registration_checks_regions(self):
        goal=dict(goal_id='person',stage='follow',description='进入大厅后观察',
            conditions=dict(robot_inside_region_id='lobby_area'),
            monitor=dict(continuous_region_id='lobby_area',review_mode='after_result',focus='观察目标'))
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        req=self.session.task_goals.goals[0]['requirement']
        self.assertEqual(req['robot_inside_region_id'],'lobby_area')
        self.assertEqual(self.session.monitor.settings['continuous_region_id'],'lobby_area')

    def test_conditional_monitor_rejects_incomplete_or_unknown_region(self):
        base=dict(goal_id='g',stage='follow',description='跟随')
        for monitor in (dict(continuous_region_id='lobby_area'),
                        dict(continuous_region_id='missing',review_mode='after_result')):
            result=self.session.call('record_task',goals=[dict(base,monitor=monitor)])
            self.assertNotEqual(result['status'],'recorded')
        result=self.session.call('record_task',goals=[dict(base,conditions=dict(robot_inside_region_id='missing'))])
        self.assertNotEqual(result['status'],'recorded')

    def test_inflight_review_is_discarded_after_region_exit(self):
        self.start(continuous_region_id='lobby_area',review_mode='after_result')
        entered,gate=threading.Event(),threading.Event()
        def work(*_):
            entered.set();gate.wait(2.)
            return reply('wait_for_event',event='monitor')
        actor=AgentLoop(self.session,work)
        try:
            self.nav.state['pose']=[7.,4.,0.]
            actor.tick();self.assertTrue(entered.wait(1))
            self.nav.state['pose']=[4.,4.,0.]
            actor.tick()
            self.assertTrue(actor.model.busy)
            gate.set();actor.model.thread.join(1);actor.tick()
            self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM events WHERE kind='model_result_discarded'").fetchone()[0],1)
            self.assertEqual(self.nav.state['status'],'running')
            self.assertEqual(self.session.state['status'],'waiting_monitor')
            self.assertEqual(self.session.monitor.snapshot()['conditional_review']['region_state'],'outside')
            self.nav.state['pose']=[7.,4.,0.]
            actor.tick()
            self.assertEqual(actor.requests,2)
            self.assertEqual(self.session.monitor.snapshot()['conditional_review']['region_state'],'inside')
        finally:
            gate.set()
            if actor.model.thread is not None:actor.model.thread.join(1)
            actor.close()

    def test_idle_observation_can_use_conditional_region_review(self):
        monitor=TaskMonitor(self.catalog)
        monitor.configure('g','观察',continuous_region_id='lobby_area',review_mode='after_result')
        feedback=dict(stamp_s=10.,frame_id='sim_odom',pose=[7.,4.,0.],localization_status='valid')
        monitor.update_idle('g',feedback)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'conditional_review_due')

    def test_agent_review_is_required_and_cannot_override_execution_evidence(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        navigate(self.session,'elevator')
        self.nav.arrive();self.session.tick()
        context=self.session.context()
        self.assertEqual(self.session.call('check_goal',_context=context,goal_id='elevator')['status'],
                         'needs_task_review')
        context=self.session.context()
        result=self.session.call('check_goal',_context=context,goal_id='elevator',
                                 review_result='unknown',review_evidence='当前依据不足')
        self.assertEqual(result['status'],'unknown')
        self.assertEqual(self.session.task_goals.goals[0]['status'],'pending')
        context=self.session.context()
        result=self.session.call('check_goal',_context=context,goal_id='elevator',
                                 review_result='satisfied',review_evidence='定位与目标一致且已停稳')
        self.assertEqual(result['status'],'completed')
        review=self.session.task_goals.goals[0]['completion']['semantic']['agent_task_review']
        self.assertEqual(review['result'],'satisfied')
        self.assertEqual(review['evidence'],'定位与目标一致且已停稳')

    def test_satisfied_model_review_cannot_create_arrival(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        navigate(self.session,'elevator')
        result=self.session.call('check_goal',_context=self.session.context(),goal_id='elevator',
            review_result='satisfied',review_evidence='认为已经到达')
        self.assertEqual(result['status'],'unproven')
        self.assertEqual(self.session.task_goals.goals[0]['status'],'pending')

    def test_model_loop_applies_review_after_navigation_completion(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        navigate(self.session,'elevator');self.nav.arrive();self.session.tick()
        actor=AgentLoop(self.session,lambda *_:reply('complete_subtask',goal_id='elevator',
            evidence='当前位置与任务目标一致，且已停稳'))
        try:
            actor.tick()
            self.assertIsNotNone(actor.model.thread)
            actor.model.thread.join(3.)
            actor.tick()
            self.assertEqual(self.session.task_goals.goals[0]['status'],'completed')
            self.assertEqual(self.session.task_goals.goals[0]['completion']['semantic']
                             ['agent_task_review']['result'],'satisfied')
            self.assertEqual(self.session.state['status'],'succeeded')
            self.assertEqual(actor.requests,1)
        finally:actor.close()

    def test_completion_refuses_movement_after_offer(self):
        self.session.record_task([point_goal('elevator')])
        navigate(self.session,'elevator');self.nav.arrive();self.session.tick()
        facts=self.session.prompt()['current_state']['completion_evidence']
        self.assertEqual(facts['missing'],[])
        context=self.session.context()
        self.nav.state.update(stopped=False,pose=[1.,1.,0.])
        result=self.session.call('complete_subtask',_context=context,goal_id='elevator',evidence='先前已到达')
        self.assertEqual(result['status'],'unproven')
        self.assertFalse(self.session.task_goals.finished())

    def test_stopped_trigger_restarts_after_missing_measurements(self):
        monitor=TaskMonitor(self.catalog);monitor.configure('g','停稳复核',stopped_s=1.)
        nav=dict(handle={'call_id':'c'},status='running',goal_id='g',context_lost=False,connected=True)
        def update(stamp):
            monitor.update(nav,dict(stamp_s=stamp,origin_stamp_s=stamp,origin_frame_id='sim_odom',
                origin_pose=[0.,0.,0.],localization_status='valid',handle={'call_id':'c'},stopped=True))
        update(0.);update(1.1)
        self.assertIsNone(monitor.take())
        for stamp in (1.35,1.6,1.85,2.1):update(stamp)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'stopped_duration')

    def test_combined_motion_stop_and_timeout_triggers_use_new_pose_times(self):
        monitor=TaskMonitor(self.catalog)
        monitor.configure('g','复核',distance_m=.5,rotation_rad=.5,stopped_s=1.,timeout_s=3.)
        nav=dict(handle={'call_id':'c'},status='running',goal_id='g',context_lost=False,connected=True)
        def update(stamp,pose,stopped=False):
            monitor.update(nav,dict(stamp_s=stamp,origin_stamp_s=stamp,origin_frame_id='sim_odom',
                origin_pose=pose,localization_status='valid',handle={'call_id':'c'},stopped=stopped))
        update(0.,[0.,0.,0.]);update(1.,[.6,0.,.6],True)
        self.assertEqual({x['kind'] for x in monitor.take()['reasons']},{'distance_progress','rotation_progress'})
        for stamp in (1.25,1.5,1.75,2.):update(stamp,[.6,0.,.6],True)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'stopped_duration')
        update(3.,[.6,0.,.6],True)
        self.assertEqual(monitor.take()['reasons'][0]['kind'],'monitor_timeout')
        update(4.,[.6,0.,.6],True)
        self.assertIsNone(monitor.take())

    def test_observation_stage_monitor_wait_and_operator_review(self):
        goal=dict(goal_id='inspect',kind='skill',target_id='',stage_type='observation',description='查看现场')
        self.session.call('record_task',goals=[goal])
        self.assertEqual(self.session.call('set_navigation_monitor',goal_id='inspect',focus='查看现场',
                                           interval_s=5)['status'],'configured')
        self.assertEqual(self.session.call('wait_for_event',event='monitor')['status'],'waiting')
        self.session.tick();self.nav.t=5.1;self.session.tick()
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.monitor.snapshot()['last_trigger']['reasons'][0]['kind'],'semantic_review_due')
        self.session.call('wait_for_event',event='monitor')
        self.session.event('review')
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.monitor.snapshot()['last_trigger']['reasons'][0]['kind'],'operator_review_requested')


if __name__ == '__main__':
    unittest.main()
