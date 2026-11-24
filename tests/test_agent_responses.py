"""Offline coverage for the Agent's OpenAI Responses transport."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from decision.agent.core.media import png_part
from decision.agent.core.model import ApiError, Model, ModelTemporarilyUnavailable, SessionTransport
from decision.agent.core.model_settings import GPT_MODEL
from decision.agent.core.store import Budget, Store


def wire(*events):
    return ''.join('data: '+json.dumps(event)+'\n\n' for event in events)


def completed(*output, model=GPT_MODEL, status='completed'):
    return dict(type='response.completed', response=dict(model=model, status=status,
        output=list(output), usage=dict(input_tokens=17, output_tokens=11, total_tokens=28,
            input_tokens_details=dict(cached_tokens=8), output_tokens_details=dict(reasoning_tokens=4))))


def call(arguments='{"question":"继续吗？"}'):
    return dict(type='function_call', status='completed', id='fc_response_item', call_id='call_original',
                name='ask_user', arguments=arguments)


class ResponsesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.config = root/'private.json'
        self.store = Store(root/'session.sqlite')
        self.addCleanup(self.store.close)
        self.budget = Budget(root/'budget.sqlite')
        self.addCleanup(self.budget.close)
        self.key = patch.dict('os.environ', {'OPENAI_API_KEY':'unit-fixture-key'})
        self.key.start()
        self.addCleanup(self.key.stop)

    def model(self, handler=None, **kwargs):
        transport = None if handler is None else httpx.MockTransport(handler)
        return Model(self.config, self.budget, self.store, model_name=GPT_MODEL, transport=transport, **kwargs)

    def events(self, kind):
        return [json.loads(row[0]) for row in self.store.db.execute(
            'SELECT value FROM events WHERE kind=? ORDER BY seq',(kind,))]

    def test_histories_convert_tool_ids_and_images_without_duplicate_storage(self):
        picture = io.BytesIO()
        Image.new('RGB', (8,8)).save(picture, format='PNG')
        image = png_part(picture.getvalue())
        text = dict(type='message', role='assistant', status='completed',
                    content=[dict(type='output_text', text='我需要确认。', annotations=[])])
        delta = dict(type='response.output_text.delta', delta='我需要确认。')
        response = completed(text, call())
        requests = []

        def send(request):
            requests.append(request)
            return httpx.Response(200, text=wire(delta, response))

        messages = [dict(role='assistant',content='先了解位置。',tool_calls=[dict(id='call_previous',type='function',
            function=dict(name='lookup_places',arguments='{}'))]),
            dict(role='tool',tool_call_id='call_previous',content='{"places":[]}'),
            dict(role='user',content=[dict(type='text',text='现在的位置'),image])]
        reply = self.model(send, thinking_mode='xhigh', decision_phase='execution').request('系统',messages)
        self.assertEqual(reply.message['content'], '我需要确认。')
        self.assertEqual(reply.message['tool_calls'][0]['id'], 'call_original')
        self.assertEqual(reply.action, dict(tool='ask_user',arguments=dict(question='继续吗？')))
        request, = requests
        self.assertEqual(str(request.url),'https://api.openai.com/v1/responses')
        self.assertEqual(request.headers['Authorization'],'Bearer unit-fixture-key')
        body = json.loads(request.content)
        self.assertEqual(body['reasoning'],dict(effort='xhigh'))
        self.assertEqual(body['max_output_tokens'],6144)
        self.assertFalse(body['store'])
        self.assertFalse(body['parallel_tool_calls'])
        self.assertTrue(all(tool['type']=='function' and tool['strict'] is False for tool in body['tools']))
        self.assertEqual(body['input'], [dict(role='system',content='系统'),
            dict(role='assistant',content='先了解位置。'),
            dict(type='function_call',call_id='call_previous',name='lookup_places',arguments='{}'),
            dict(type='function_call_output',call_id='call_previous',output='{"places":[]}'),
            dict(role='user',content=[dict(type='input_text',text='现在的位置'),
                dict(type='input_image',detail='auto',image_url=image['image_url']['url'])])])
        saved_request, = self.events('api_request')
        for key,value in dict(provider='openai',api_format='responses',decision_phase='execution',thinking_mode='xhigh').items():
            self.assertEqual(saved_request[key],value)
        self.assertNotIn('input',saved_request['request'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM request_images').fetchone(),(1,))
        self.assertNotIn('unit-fixture-key',json.dumps(saved_request))
        saved_response, = self.events('api_response')
        self.assertEqual(saved_response['raw_chunks'],[delta,response])
        self.assertEqual(saved_response['usage']['completion_tokens_details']['reasoning_tokens'],4)
        self.assertEqual(saved_response['cache']['cached_tokens'],8)
        self.assertEqual(saved_response['decision_phase'],'execution')

    def test_requires_completed_matching_result_and_classifies_interruption(self):
        cases = (
            (dict(type='response.output_text.delta',delta='partial'), ModelTemporarilyUnavailable),
            (dict(type='response.failed',response=dict(error=dict(code='server_error'))), ModelTemporarilyUnavailable),
            (dict(type='error',code='rate_limit_exceeded'), ModelTemporarilyUnavailable),
            (dict(type='response.failed',response=dict(error=dict(code='invalid_request_error'))), ApiError),
            (dict(type='response.incomplete',response=dict(incomplete_details=dict(reason='max_output_tokens'))), ApiError),
            (completed(call(),model='wrong-model'), ApiError),
            (completed(call(),status='incomplete'), ApiError),
        )
        for event,error in cases:
            with self.subTest(event=event):
                model=self.model(lambda _:httpx.Response(200,text=wire(event),headers={'Retry-After':'30'}))
                with self.assertRaises(error) as caught:
                    model.request('系统', [])
                self.assertIs(type(caught.exception),error)
                if error is ModelTemporarilyUnavailable:
                    self.assertEqual(caught.exception.retry_after_s,30.)
        self.assertFalse(self.events('api_response'))
        self.assertEqual(len(self.events('api_failure')),len(cases))

    def test_transient_completion_keeps_only_metadata_and_uses_pooled_proxy_client(self):
        bodies, sink = [], []

        def send(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200,text=wire(completed(call())))

        client = httpx.Client(transport=httpx.MockTransport(send))
        self.addCleanup(client.close)
        transport = SessionTransport(model_name=GPT_MODEL)
        self.addCleanup(transport.close)
        with patch('decision.agent.core.model.http_client',return_value=client) as factory:
            model = self.model(session_transport=transport, transient_sink=sink.append)
            for _ in range(2):
                model.request('私有乘梯提示', [], transient=True)
            factory.assert_called_once_with(trust_env=True)
        self.assertEqual([body['reasoning']['effort'] for body in bodies], ['none','none'])
        self.assertEqual([item['kind'] for item in sink],['request','response','request','response'])
        for event in self.events('api_request')+self.events('api_response'):
            self.assertEqual(event['decision_phase'],'monitor')
            self.assertNotIn('request',event)
            self.assertNotIn('raw_chunks',event)
            self.assertNotIn('message',event)
            self.assertNotIn('私有乘梯提示',json.dumps(event,ensure_ascii=False))

    def test_invalid_tool_arguments_are_reported_and_kept_as_returned(self):
        args = '{"question": "one", "question": "two"}'
        model = self.model(lambda _:httpx.Response(200,text=wire(completed(call(args)))))
        reply = model.request('系统', [])
        self.assertIsNone(reply.action)
        self.assertEqual(reply.argument_error['reason'],'invalid_tool_arguments')
        self.assertEqual(reply.message['tool_calls'][0]['function']['arguments'],args)
