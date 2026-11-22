"""Session HTTP lifetime and decision timing with a simulated hosted service."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from decision.agent import CONFIG_DIR
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Budget, Store
from decision.agent.tools.places import Catalog
from decision.agent.workflows.common import make_loop
from tests.test_agent import ScriptedFeedback, reply
from tests.test_agent_conversation import wire


class TransportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.budget = root/'test-budget.sqlite'
        self.session_path = root/'session.sqlite'
        self.store = Store(self.session_path)
        self.addCleanup(lambda:self.store.close())
        self.session = AgentSession('到电梯口',Catalog.load(CONFIG_DIR/'simulation.json'),
                                    ScriptedFeedback(),self.store)
        config = root/'private.json'
        config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
        self.args = SimpleNamespace(actions=None,api_config=config,cache_mode='implicit',
                                    thinking_mode='planning',planning_thinking_mode='medium',max_requests=4)
        budget_patch = patch('decision.agent.core.model.shared_budget',side_effect=lambda:Budget(self.budget))
        budget_patch.start()
        self.addCleanup(budget_patch.stop)

    def make_actor(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler),trust_env=False)
        self.addCleanup(client.close)
        client_patch = patch('decision.agent.core.model.http_client',return_value=client)
        factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        actor = make_loop(self.args,self.session,self.session_path)
        self.addCleanup(actor.close)
        return actor,client,factory

    def events(self, kind):
        return [json.loads(row[0]) for row in self.store.db.execute(
            'SELECT value FROM events WHERE kind=? ORDER BY seq',(kind,))]

    def test_uses_one_client_across_phase_change_and_correlates_tool_timing(self):
        registration = reply('record_task',goals=[dict(goal_id='g1',stage='point',
            target=dict(id='elevator'),description='到电梯口')])
        requests = []

        def handle(request):
            requests.append(request)
            choice = registration if len(requests)==1 else reply('ask_user',question='现在出发吗？')
            return httpx.Response(200,headers={'Set-Cookie':'ignored=1'},text=wire(choice.message['tool_calls']))

        actor,client,factory = self.make_actor(handle)
        factory.assert_not_called()
        actor.tick()
        actor.model.thread.join(5)
        actor.tick()
        self.assertTrue(self.session.prompt()['current_state']['requirements_registered'])
        actor.model.thread.join(5)
        actor.tick()

        factory.assert_called_once()
        self.assertEqual(len(requests),2)
        planning,execution = (json.loads(request.content) for request in requests)
        self.assertEqual(planning['reasoning_effort'],'medium')
        self.assertTrue(planning['enable_thinking'])
        self.assertFalse(execution['enable_thinking'])
        self.assertEqual(execution['tool_choice'],'required')
        self.assertNotIn('Cookie',requests[1].headers)

        responses = self.events('api_response')
        consumed = self.events('model_result_consumed')
        tools = self.events('model_tool_processed')
        self.assertEqual((len(responses),len(consumed),len(tools)),(2,2,2))
        for index,(response,result,tool) in enumerate(zip(responses,consumed,tools),1):
            self.assertEqual(result['request_index'],index)
            self.assertEqual(tool['request_index'],index)
            self.assertEqual(response['call_id'],result['timing']['call_id'])
            self.assertEqual(response['call_id'],tool['timing']['call_id'])
            phases = ('prepare_started_s','submitted_s','worker_started_s','prepare_completed_s',
                      'transport_started_s','transport_first_delta_s','transport_complete_s',
                      'http_body_complete_s','worker_finished_s','consumed_s','tool_processing_started_s',
                      'tool_invoked_s','tool_returned_s','tool_processing_finished_s')
            timing = dict(result['timing'],**tool['timing'])
            times = [timing[phase] for phase in phases]
            self.assertEqual(times,sorted(times))
            self.assertGreaterEqual(response['first_token_s'],
                                    timing['transport_first_delta_s']-timing['transport_started_s'])
            self.assertGreaterEqual(response['full_return_s'],
                                    timing['transport_complete_s']-timing['transport_started_s'])
        self.assertFalse(client.is_closed)
        actor.close()
        self.assertTrue(client.is_closed)
        with self.assertRaises(RuntimeError):
            actor.model.submit('tool',{},lambda:None)

    def test_closing_inflight_work_defers_client_close_and_discards_late_reply(self):
        entered,release = threading.Event(),threading.Event()

        def handle(request):
            entered.set()
            release.wait(5)
            return httpx.Response(200,text=wire(reply('ask_user',question='过期问题').message['tool_calls']))

        actor,client,factory = self.make_actor(handle)
        actor.tick()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(actor.close()['request_unresolved'])
            self.assertFalse(client.is_closed)
        finally:
            self.store.close()
            release.set()
            actor.model.thread.join(5)
            self.session.store = self.store = Store(self.session_path)
        self.assertTrue(client.is_closed)
        self.assertIsNone(actor.model.poll())
        self.assertFalse(self.events('api_response'))
        self.assertFalse(self.events('model_result_consumed'))
        failure, = self.events('api_failure')
        self.assertIsNone(failure['full_return_s'])
        self.assertNotIn('transport_complete_s',failure['timing'])
        budget = Budget(self.budget)
        self.addCleanup(budget.close)
        row = budget.db.execute('SELECT status FROM calls WHERE id=?',(failure['call_id'],)).fetchone()
        self.assertEqual(row,('usage_unknown',))
        factory.assert_called_once()

    def test_done_keeps_the_completed_reply_without_reading_an_unfinished_body(self):
        read_after_done = []

        class UnfinishedBody(httpx.SyncByteStream):
            def __iter__(self):
                yield wire(reply('ask_user',question='继续吗？').message['tool_calls']).encode()
                read_after_done.append(True)
                yield b': heartbeat\n\n'

        actor,_,_ = self.make_actor(lambda _:httpx.Response(200,headers={'Transfer-Encoding':'chunked'},
                                                            stream=UnfinishedBody()))
        actor.tick()
        actor.model.thread.join(5)
        actor.tick()
        response, = self.events('api_response')
        self.assertIn('transport_complete_s',response['timing'])
        self.assertNotIn('http_body_complete_s',response['timing'])
        self.assertFalse(read_after_done)
        self.assertFalse(self.events('api_failure'))
        self.assertFalse(self.events('model_retry_scheduled'))

    def test_open_http1_tail_is_interrupted_after_done_without_replaying(self):
        interrupted = threading.Event()
        requests = []

        class OpenBody(httpx.SyncByteStream):
            def __iter__(self):
                yield wire(reply('ask_user',question='继续吗？').message['tool_calls']).encode()
                if not interrupted.wait(2):
                    raise RuntimeError('Expected the completed SSE body to be interrupted')
                raise httpx.ReadError('tail interrupted')

        connection = SimpleNamespace(shutdown=lambda _direction:interrupted.set())
        stream = SimpleNamespace(get_extra_info=lambda info:connection if info=='socket' else None)

        def handle(request):
            requests.append(request)
            return httpx.Response(200,headers={'Transfer-Encoding':'chunked'},stream=OpenBody(),
                                  extensions={'network_stream':stream})

        actor,_,_ = self.make_actor(handle)
        actor.tick()
        actor.model.thread.join(3)
        actor.tick()

        self.assertTrue(interrupted.is_set())
        self.assertEqual(len(requests),1)
        self.assertEqual(actor.state['status'],'needs_clarification')
        response, = self.events('api_response')
        self.assertIn('transport_complete_s',response['timing'])
        self.assertNotIn('http_body_complete_s',response['timing'])
        self.assertFalse(self.events('api_failure'))
        self.assertFalse(self.events('model_retry_scheduled'))
