"""Hosted model and effort selection through Agent workflow decisions."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decision.agent import CONFIG_DIR
from decision.agent.cli import main
from decision.agent.core.model import ModelTemporarilyUnavailable
from decision.agent.core.model_settings import DEFAULT_MODEL, GPT_MODEL
from decision.agent.core.reuse import import_registered_task
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from decision.agent.workflows.common import make_loop, model_options
from tests import test_agent_composition, test_agent_goals
from tests.test_agent import ScriptedFeedback, navigate, point_goal, reply


class ModelPhaseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'session.sqlite'
        self.store = Store(self.path)
        self.addCleanup(self.store.close)
        self.nav = ScriptedFeedback()
        catalog = Catalog.load(CONFIG_DIR / 'simulation.json')
        self.session = AgentSession('到电梯口', catalog, self.nav, self.store)
        api = patch('decision.agent.core.model.api_work')
        self.api = api.start()
        self.addCleanup(api.stop)

    def make_actor(self):
        args = SimpleNamespace(actions=None, api_config='unused', cache_mode='implicit', max_requests=8,
                               model=GPT_MODEL, planning_thinking_mode='medium', execution_thinking_mode='off',
                               monitor_thinking_mode='low')
        actor = make_loop(args, self.session, self.path)
        self.addCleanup(actor.close)
        return actor

    def turn(self, actor, response, phase, effort):
        self.api.reset_mock()
        self.api.side_effect = response if isinstance(response, Exception) else None
        self.api.return_value = response
        limit = actor.requests + 1
        actor.before_request = lambda current: current.requests < limit
        actor.tick()
        self.assertEqual(actor.requests, limit)
        actor.model.thread.join(3)
        self.assertFalse(actor.model.thread.is_alive())
        actor.tick()
        self.api.assert_called_once()
        kwargs = self.api.call_args.kwargs
        self.assertEqual((kwargs['model_name'], kwargs['decision_phase'], kwargs['thinking_mode']),
                         (GPT_MODEL, phase, effort))

    def test_registration_and_ordinary_execution_use_their_own_efforts(self):
        actor = self.make_actor()
        self.turn(actor, reply('record_task', goals=[point_goal('elevator')]), 'registration', 'medium')
        self.assertIsNotNone(self.session.monitor.settings)
        self.turn(actor, reply('ask_user', question='现在出发吗？'), 'execution', 'off')

    def test_imported_registration_begins_in_execution(self):
        plan = dict(goals=[point_goal('elevator')], clarifications=[], instruction=self.session.state['instruction'],
                    source_task_id='previous', source_revision=1, source_event_seq=2, digest='test')
        import_registered_task(self.session, plan, {}, self.path)
        self.turn(self.make_actor(), reply('ask_user', question='现在出发吗？'), 'execution', 'off')

    def test_operator_review_retains_effort_while_collecting_evidence_and_retrying(self):
        self.session.call('record_task', goals=[point_goal('elevator')])
        actor = self.make_actor()
        actor.event('review')
        self.turn(actor, reply('observe_scene'), 'monitor', 'low')
        self.turn(actor, ModelTemporarilyUnavailable(), 'monitor', 'low')
        actor.model_retry_at = 0.
        self.turn(actor, reply('set_navigation_monitor', goal_id='elevator', focus='电梯口', interval_s=5),
                  'monitor', 'low')
        self.assertIsNotNone(self.session.monitor.last_trigger)
        self.turn(actor, reply('ask_user', question='现在出发吗？'), 'execution', 'off')

    def test_timer_and_confirmed_arrival_are_monitor_requests(self):
        self.session.call('record_task', goals=[point_goal('elevator')])
        self.session.call('set_navigation_monitor', goal_id='elevator', focus='电梯口', interval_s=5)
        self.assertEqual(navigate(self.session, 'elevator')['status'], 'accepted')
        self.session.call('wait_for_event', event='navigation')
        self.session.tick()
        actor = self.make_actor()
        self.nav.t = 5.1
        self.turn(actor, reply('wait_for_event', event='navigation'), 'monitor', 'low')
        self.nav.arrive()
        self.turn(actor, reply('check_goal', goal_id='elevator'), 'monitor', 'low')

    def test_conditional_review_is_active_until_leaving_the_region(self):
        self.session.call('record_task', goals=[point_goal('elevator')])
        self.session.call('set_navigation_monitor', goal_id='elevator', focus='观察大厅',
                          continuous_region_id='lobby_area', review_mode='after_result')
        self.assertEqual(navigate(self.session, 'elevator')['status'], 'accepted')
        self.session.call('wait_for_event', event='navigation')
        self.session.tick()
        self.nav.state['pose'] = [7., 4., 0.]
        self.session.tick()
        self.assertEqual(self.session.take_model_phase(self.session.prompt()), 'monitor')
        self.assertEqual(self.session.take_model_phase(self.session.prompt()), 'monitor')
        self.nav.state['pose'] = [4., 4., 0.]
        self.session.tick()
        self.assertEqual(self.session.take_model_phase(self.session.prompt()), 'execution')

    def test_visual_failure_returns_to_execution_for_the_next_navigation_choice(self):
        goal = dict(point_goal('elevator'), semantic_conditions=['看清电梯门'])
        self.session.call('record_task', goals=[goal])
        self.assertEqual(navigate(self.session, 'elevator')['status'], 'accepted')
        self.nav.arrive()
        self.session.tick()
        self.session.tick()
        self.session.observation = test_agent_goals.IndoorGoalTests.rgb(self)
        self.assertEqual(self.session.take_model_phase(self.session.prompt()), 'monitor')
        actor = self.make_actor()
        self.turn(actor, reply('record_monitor_review', goal_id='elevator', result='not_satisfied',
                               evidence='当前无法看清电梯门'), 'monitor', 'low')
        self.turn(actor, reply('ask_user', question='可以选择附近的位置吗？'), 'execution', 'off')

    def test_registered_elevator_uses_monitor_only_during_the_independent_request(self):
        self.session.call('record_task', goals=[point_goal('elevator')])
        self.session.elevator = SimpleNamespace(transition=SimpleNamespace(status='monitoring_ride'))
        actor = self.make_actor()
        for phase in ('monitoring_ride', 'awaiting_destination_localization'):
            self.session.elevator.transition.status = phase
            payload = dict(current_state=dict(requirements_registered=True, ride=dict(status=phase)),
                           ride_inputs=[('current_elevator_rgb', {}, b'', 'JPEG')])
            payload['decision_phase'] = self.session.take_model_phase(payload)
            actor.work(payload, None, None)
            self.assertEqual(self.api.call_args.kwargs['decision_phase'], 'monitor')
        self.turn(actor, reply('ask_user', question='现在出发吗？'), 'execution', 'off')

    def test_follow_starts_in_execution_and_monitors_only_matching_live_feedback(self):
        fixture = test_agent_composition.CompositionTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        session = fixture.session
        session.call('record_task', goals=[test_agent_composition.follow_goal('person')])
        ref = fixture.select_follow()
        self.assertEqual(session.take_model_phase(session.prompt()), 'execution')
        self.assertEqual(session.call('start_local_task', goal_id='person', task_type='follow',
                                      reference_id=ref)['status'], 'accepted')
        self.assertEqual(session.take_model_phase(session.prompt()), 'execution')
        session.tick()
        self.assertEqual(session.take_model_phase(session.prompt()), 'monitor')
        fixture.nav.state['follow_status']['handle'] = None
        self.assertEqual(session.take_model_phase(session.prompt()), 'execution')

    def test_model_defaults_and_global_override(self):
        self.assertEqual(model_options(SimpleNamespace()),
                         (DEFAULT_MODEL, dict(registration='low', execution='off', monitor='off')))
        self.assertEqual(model_options(SimpleNamespace(model=GPT_MODEL)),
                         (GPT_MODEL, dict(registration='low', execution='low', monitor='low')))
        self.assertEqual(model_options(SimpleNamespace(model=GPT_MODEL, thinking_mode='xhigh')),
                         (GPT_MODEL, dict(registration='xhigh', execution='xhigh', monitor='xhigh')))

    def test_invalid_cli_effort_fails_before_dispatch(self):
        cli = ['agent', 'execute', '--instruction', '到电梯口', '--output', 'unused',
               '--runtime-socket', 'unused', '--model', DEFAULT_MODEL, '--monitor-thinking-mode', 'high']
        with patch('sys.argv', cli), patch('decision.agent.workflows.live.run') as run, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_openai_explicit_cache_is_rejected_before_a_live_connection(self):
        from decision.agent.workflows.live import run

        args = SimpleNamespace(actions=None, model=GPT_MODEL, cache_mode='explicit')
        with patch('decision.agent.workflows.live.LiveRuntimeClient') as connect:
            with self.assertRaisesRegex(ValueError, 'requires implicit caching'):
                run(args)
        connect.assert_not_called()

    def test_explicit_actions_do_not_create_model_transport(self):
        fixture = self.path.parent / 'actions.json'
        fixture.write_text(json.dumps([dict(tool='record_task', arguments=dict(goals=[point_goal('elevator')]))]))
        args = SimpleNamespace(actions=fixture, model=GPT_MODEL, max_requests=4)
        with patch('decision.agent.core.model.SessionTransport') as transport:
            actor = make_loop(args, self.session, self.path)
            self.addCleanup(actor.close)
            result = actor.work(self.session.prompt(), None, None)
        self.assertEqual(result.action['tool'], 'record_task')
        transport.assert_not_called()
        self.api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
