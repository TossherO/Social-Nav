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
        self.assertEqual(result['reason'], 'stage_skill_not_available')
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


if __name__ == '__main__':
    unittest.main()
