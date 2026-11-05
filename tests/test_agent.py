"""Tool protocol, real-feedback requirements and hosted-client failure behavior."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import httpx
import jsonschema
from decision.agent import ROOT
from decision.agent.tools.places import Catalog
from decision.agent.core.model import ApiError, Model
from decision.agent.core.store import Budget, Store
from decision.agent.core.session import AgentSession
from decision.agent.tools.framework import validate_call


def action(tool, **arguments):
    return dict(tool=tool, arguments=arguments)


class ScriptedFeedback:
    """Explicit unit test double; never used by the CLI or physics evaluation."""
    scope = clock_domain = 'simulation'

    def __init__(self):
        self.t = 0.
        self.state = dict(status='idle', handle=None, stamp_s=10., scope=self.scope, clock_domain=self.clock_domain,
                          session_id='unit', generation=0, frame_id='sim_odom', stopped=True, collision=False,
                          source='synthetic_differential_drive_plant', execution_feedback=True)

    def snapshot(self):
        return copy.deepcopy(self.state)

    def tick(self):
        self.t += .1
        self.state['stamp_s'] = 10+self.t
        return self.snapshot()

    def check_route(self, place):
        return dict(reference_found=True, status='path_found', reason=None)

    def start(self, place, handle):
        self.state.update(status='running', handle=copy.deepcopy(handle), stopped=False)
        return dict(accepted=True, handle=copy.deepcopy(handle), execution_feedback=False)

    def cancel(self, handle):
        if handle != self.state['handle']:
            raise ValueError('wrong call')
        self.state['status'] = 'stopping'

    def arrive(self):
        handle = self.state['handle']
        self.state.update(status='arrived', stopped=True, goal=dict(task_completed=True,
                          task_id=handle['task_id'], goal_revision=handle['goal_revision']))

    def stopped(self):
        self.state.update(status='stopped', stopped=True)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root/'session.sqlite')
        self.catalog = Catalog.load(ROOT/'decision/agent/configs/simulation.json')
        self.nav = ScriptedFeedback()
        self.tools = AgentSession('先去电梯口，再回起点', self.catalog, self.nav, self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_call_arguments_are_checked_without_repair(self):
        for value in (action('navigate'), action('observe_scene', place_id='elevator'),
                      action('override'), [action('observe_scene')]):
            with self.assertRaises(jsonschema.ValidationError):
                validate_call(value)
        self.assertEqual(validate_call(action('lookup_places', name='')), action('lookup_places', name=''))

    def test_ack_is_not_arrival_and_cannot_finish(self):
        ack = self.tools.call('navigate', place_id='elevator')
        self.assertEqual(ack['status'], 'accepted')
        self.tools.tick()
        self.assertEqual(self.tools.skill.record['completed'], [])
        self.assertEqual(self.tools.call('finish_task', status='succeeded', evidence='ACK received')['status'], 'refused')

    def test_navigation_wait_and_explicit_finish_require_bound_feedback(self):
        self.tools.call('navigate', place_id='elevator')
        self.tools.call('wait_for_event', event='navigation')
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'waiting_navigation')
        with self.assertRaises(ValueError):
            self.tools.call('navigate', place_id='lobby_center')
        self.nav.arrive()
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'ready')
        self.assertEqual(self.tools.skill.record['completed'][0]['place_id'], 'elevator')
        self.assertEqual(self.tools.call('finish_task', status='succeeded', evidence='到达停稳')['status'], 'succeeded')

    def test_wrong_task_and_old_call_cannot_supply_arrival(self):
        self.tools.call('navigate', place_id='elevator')
        real = copy.deepcopy(self.nav.state['handle'])
        self.nav.arrive()
        self.nav.state['goal']['task_id'] = 'foreign'
        self.tools.tick()
        self.assertEqual(self.tools.skill.record['completed'], [])
        self.nav.state['handle'] = dict(real, call_id='old')
        self.nav.state['goal']['task_id'] = real['task_id']
        self.tools.tick()
        self.assertEqual(self.tools.skill.record['completed'], [])

    def test_localization_health_failure_requests_stop_without_reset(self):
        for published in (True, False):
            with self.subTest(goal_published_in_feedback=published):
                nav = ScriptedFeedback()
                tools = AgentSession('health test', self.catalog, nav, self.store)
                tools.call('navigate', place_id='elevator')
                tools.call('wait_for_event', event='navigation')
                nav.state['localization_status'] = 'invalid'
                if not published:
                    original_tick = nav.tick
                    nav.tick = lambda: dict(original_tick(), handle=None)
                tools.tick()
                self.assertEqual(tools.skill.record['status'], 'stopping')
                self.assertEqual(tools.skill.record['reason'], 'navigation_state_unavailable')
                self.assertEqual(tools.skill.record['completed'], [])

    def test_cancel_and_replacement_wait_for_stop(self):
        original = self.tools.state['instruction']
        self.tools.call('navigate', place_id='elevator')
        self.tools.event('replace', '改去大厅中央')
        self.tools.tick()
        self.assertEqual((self.tools.state['status'], self.tools.state['instruction']), ('stopping', original))
        self.nav.stopped()
        self.tools.tick()
        self.assertEqual(self.tools.state['instruction'], '改去大厅中央')
        self.assertEqual(self.tools.state['revision'], 2)
        self.assertIsNone(self.tools.skill.record['handle'])

    def test_pause_resume_new_call_and_cancel(self):
        self.tools.call('navigate', place_id='elevator')
        previous = copy.deepcopy(self.tools.skill.record['handle'])
        self.tools.event('pause')
        self.nav.stopped()
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'paused')
        self.tools.event('resume')
        self.tools.call('navigate', place_id='elevator')
        self.assertGreater(self.tools.skill.record['handle']['goal_revision'], previous['goal_revision'])
        self.tools.event('cancel')
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'stopping')
        self.nav.stopped()
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'cancelled')

    def test_user_continue_is_real_event_and_answer_keeps_instruction(self):
        self.tools.call('wait_for_event', event='user_continue')
        self.tools.tick()
        self.assertEqual(self.tools.state['status'], 'waiting_user')
        self.tools.event('continue')
        self.tools.call('ask_user', question='哪一个入口？')
        old = self.tools.state['instruction']
        self.tools.event('answer', '大厅玻璃门内侧')
        self.assertEqual(self.tools.state['instruction'], old)
        self.assertEqual(self.tools.prompt()['clarifications'][0]['answer'], '大厅玻璃门内侧')

    def test_failed_or_unknown_goal_is_not_erased_by_earlier_success(self):
        self.tools.call('navigate', place_id='elevator')
        self.nav.arrive(); self.tools.tick()
        self.tools.call('navigate', place_id='unknown')
        result = self.tools.call('finish_task', status='succeeded', evidence='只完成一个站点')
        self.assertEqual(result['status'], 'refused')
        self.assertIn('unknown', result['uncompleted_places'])

    def test_shadow_cannot_finish_with_recorded_arrival(self):
        value = copy.deepcopy(self.catalog.value)
        value['scope'] = 'development_prior'
        for p in value['places']:
            p.update(allow_shadow=True)
        self.nav.scope = 'shadow'
        self.nav.state['scope'] = 'shadow'
        tools = AgentSession('shadow', Catalog(value), self.nav, self.store)
        tools.call('navigate', place_id='elevator')
        self.nav.arrive(); tools.tick()
        self.assertEqual(tools.skill.record['completed'], [])
        self.assertEqual(tools.call('finish_task', status='succeeded', evidence='recorded pose')['status'], 'refused')

    def test_real_catalog_scope_and_missing_rgb(self):
        catalog = Catalog.load(ROOT/'decision/agent/configs/floor1.json')
        with self.assertRaises(ValueError):
            catalog.destination('elevator', 'simulation')
        self.assertEqual(self.tools.call('observe_scene')['status'], 'unavailable')

    def test_budget_shared_across_sessions_and_failures(self):
        path = self.root/'budget.sqlite'
        budget = Budget(path, limit=.1)
        budget.reserve(50000, 1024)
        budget.close()
        budget = Budget(path, limit=.1)
        with self.assertRaises(RuntimeError):
            budget.reserve(50000, 1024)
        self.assertEqual(budget.summary()['calls'], 1)
        budget.close()

    def client(self, reply):
        config = self.root/'private.json'
        config.write_text(json.dumps(dict(base_url='https://example.invalid/v1', api_key='test-secret')))
        budget = Budget(self.root/'budget.sqlite')
        return Model(config, budget, self.store, transport=httpx.MockTransport(reply)), budget

    def test_api_error_is_redacted_not_retried_and_keeps_reservation(self):
        requests = []
        def reply(request):
            requests.append(request)
            return httpx.Response(401, text='private test-secret service details')
        client, budget = self.client(reply)
        try:
            with self.assertRaises(ApiError) as caught:
                client.decide(self.tools.prompt())
            self.assertEqual(len(requests), 1)
            self.assertNotIn('test-secret', str(caught.exception))
            self.assertNotIn('test-secret', str(self.store.db.execute('SELECT value FROM events').fetchall()))
            self.assertGreater(budget.summary()['accounted_rmb'], 0)
        finally:
            budget.close()

    def test_streamed_tool_call_preserves_content_and_records_usage(self):
        output = action('lookup_places', name='')
        chunks = [dict(model='qwen3.8-flash', choices=[dict(delta=dict(content=json.dumps(output)), finish_reason='stop')]),
                  dict(choices=[], usage=dict(prompt_tokens=100, completion_tokens=50))]
        wire = ''.join('data: '+json.dumps(v)+'\n\n' for v in chunks)+'data: [DONE]\n\n'
        client, budget = self.client(lambda request: httpx.Response(200, text=wire))
        try:
            self.assertEqual(client.decide(self.tools.prompt()), output)
            self.assertAlmostEqual(budget.summary()['accounted_rmb'], .00032)
        finally:
            budget.close()

    def test_bad_array_or_tool_arguments_remain_raw_without_unwrapping(self):
        for output in ([action('observe_scene')], action('navigate', question='wrong argument')):
            with self.subTest(output=output):
                chunk = dict(model='qwen3.8-flash', choices=[dict(delta=dict(content=json.dumps(output)), finish_reason='stop')],
                             usage=dict(prompt_tokens=10, completion_tokens=10))
                client, budget = self.client(lambda request: httpx.Response(200,text='data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n'))
                try:
                    with self.assertRaises(ApiError):
                        client.decide(self.tools.prompt())
                    saved = json.loads(self.store.db.execute("SELECT value FROM events WHERE kind='api_response' ORDER BY seq DESC LIMIT 1").fetchone()[0])
                    self.assertEqual(json.loads(saved['content']), output)
                finally:
                    budget.close()


if __name__ == '__main__':
    unittest.main()
