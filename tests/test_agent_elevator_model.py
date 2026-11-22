"""Elevator judgments keep history only in the transient model request."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from decision.agent.core.model import ApiError,ModelTemporarilyUnavailable,Model,retry_after_seconds
from decision.agent.core.store import Budget,Store
from decision.agent.tools.framework import Reply
from decision.agent.workflows.elevator_camera import image_record
from tests.test_agent_conversation import wire


class ElevatorModelTests(unittest.TestCase):
    def test_temporary_network_errors_are_classified_for_all_agent_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/'private.json'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            store=Store(root/'session.sqlite')
            budget=Budget(root/'budget.sqlite')
            def timeout(request):
                raise httpx.ConnectTimeout('test timeout',request=request)
            try:
                model=Model(config,budget,store,transport=httpx.MockTransport(timeout))
                for transient in (False,True):
                    with self.assertRaises(ModelTemporarilyUnavailable):
                        model.request('test',[],transient=transient)
                model.transport=httpx.MockTransport(lambda request:httpx.Response(429,headers={'Retry-After':'75'}))
                with self.assertRaises(ModelTemporarilyUnavailable) as temporary:
                    model.request('test',[])
                self.assertEqual(temporary.exception.retry_after_s,75.)
                model.transport=httpx.MockTransport(lambda request:httpx.Response(401))
                with self.assertRaises(ApiError) as failure:
                    model.request('test',[],transient=True)
                self.assertNotIsInstance(failure.exception,ModelTemporarilyUnavailable)
            finally:
                store.close()
                budget.close()

    def test_retry_after_http_date_and_malformed_header(self):
        with patch('decision.agent.core.model.time.time',return_value=784111770.):
            self.assertEqual(retry_after_seconds('Sun, 06 Nov 1994 08:49:37 GMT'),7.)
            self.assertIsNone(retry_after_seconds('invalid'))

    def test_standard_reference_images_and_prompt_are_in_the_actual_request(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);config=root/'private.json'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            record=image_record(np.zeros((10,10,3),dtype=np.uint8),stamp_s=10.,frame_id='camera',sequence=0)
            jpeg=base64.b64decode(record.pop('image_base64'))
            buffer=io.BytesIO();Image.new('RGB',(10,10)).save(buffer,format='PNG');png=buffer.getvalue()
            native=Reply.fixture(dict(tool='record_elevator_observation',arguments=dict(
                judgment='unknown',evidence='尚不能确认',retain_observation=False))).message['tool_calls']
            captured=[]
            def handler(request):
                captured.append(json.loads(request.content));return httpx.Response(200,text=wire(native))
            store=Store(root/'session.sqlite');budget=Budget(root/'budget.sqlite')
            model=Model(config,budget,store,transport=httpx.MockTransport(handler))
            payload=dict(messages=[dict(role='user',content='乘梯')],
                current_state=dict(requirements_registered=True,ride=dict(status='monitoring_ride',
                    standard_start=dict(floor_id='floor1')),navigation=dict(feedback=dict(stamp_s=10.,clock_domain='ros_time'))),
                ride_inputs=[('current_elevator_rgb',record,jpeg,'JPEG'),('current_local_lidar',record,png,'PNG'),
                    ('standard_start_reference_rgb',record,jpeg,'JPEG'),('standard_start_reference_lidar',record,png,'PNG')])
            try:self.assertEqual(model.decide(payload).action['arguments']['judgment'],'unknown')
            finally:store.close();budget.close()
            body=json.dumps(captured[0],ensure_ascii=False)
            self.assertIn('standard_start_reference_rgb',body)
            self.assertIn('standard_start_reference_lidar',body)
            self.assertIn('固定初值启动定位',body)
            self.assertEqual(body.count('data:image/'),4)

    def test_transient_request_uses_history_without_persisting_images_or_judgments(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);config=root/'private.json';session=root/'session.sqlite'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            record=image_record(np.zeros((10,10,3),dtype=np.uint8),stamp_s=10.,frame_id='camera',sequence=0)
            image=base64.b64decode(record.pop('image_base64'))
            native=Reply.fixture(dict(tool='record_elevator_observation',arguments=dict(judgment='not_completed',evidence='private-evidence',retain_observation=True))).message['tool_calls']
            captured=[]
            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200,text=wire(native))
            store=Store(session);budget=Budget(root/'budget.sqlite')
            visible=[]
            model=Model(config,budget,store,transport=httpx.MockTransport(handler),transient_sink=visible.append)
            reference_id = 'a'*64
            payload=dict(messages=[dict(role='user',content='乘梯')],
                current_state=dict(requirements_registered=True,ride=dict(status='monitoring_ride',
                                   standard_start=dict(floor_id='floor1')),
                                   follow_reference=dict(reference_id=reference_id),
                                   navigation=dict(feedback=dict(stamp_s=10.,clock_domain='ros_time'))),
                ride_inputs=[('current_elevator_rgb',record,image,'JPEG')])
            try:
                result=model.decide(payload)
                model.decide(dict(payload,ride_inputs=None))
            finally:store.close();budget.close()
            self.assertEqual(result.action['arguments']['judgment'],'not_completed')
            ride_tools={tool['function']['name'] for tool in captured[0]['tools']}
            self.assertEqual(captured[0]['tool_choice'],'required')
            normal_tools={tool['function']['name'] for tool in captured[1]['tools']}
            self.assertIn('record_elevator_observation',ride_tools)
            self.assertNotIn('pause_map_localization',ride_tools)
            self.assertNotIn('restart_map_localization',ride_tools)
            self.assertIn('pause_map_localization',normal_tools)
            state=json.loads(captured[0]['messages'][-1]['content'][0]['text'])['state']
            self.assertEqual(state['ride']['status'],'monitoring_ride')
            self.assertEqual(state['ride']['standard_start']['floor_id'],'floor1')
            self.assertFalse(state['visual_inputs']['maps']['available'])
            self.assertEqual(state['follow_reference']['reference_id'],'r1')
            self.assertEqual(captured[0]['messages'][0],captured[1]['messages'][0])
            self.assertIn('data:image/jpeg;base64,',json.dumps(captured[0]))
            self.assertEqual([item['kind'] for item in visible],['request','response'])
            self.assertIn('data:image/jpeg;base64,',json.dumps(visible[0]['request']))
            self.assertIn('private-evidence',json.dumps(visible[1]['message']))
            store=Store(session)
            try:
                events=str(store.db.execute('select value from events order by seq limit 2').fetchall())
                normal_request=json.loads(store.db.execute("SELECT value FROM events WHERE kind='api_request' ORDER BY seq DESC").fetchone()[0])
            finally:store.close()
            self.assertNotIn('model_references',events)
            self.assertNotIn(reference_id,events)
            self.assertNotIn('model_references',visible[0])
            self.assertEqual(normal_request['model_references']['r1'],['ref',reference_id])
            budget=Budget(root/'budget.sqlite')
            try:records=str(budget.db.execute('select record from calls').fetchall())
            finally:budget.close()
            for output in (events,records):
                self.assertNotIn('data:image/',output)
                self.assertNotIn('private-evidence',output)

    def test_destination_phase_offers_localization_and_rejects_stale_pause(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);config=root/'private.json'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            record=image_record(np.zeros((10,10,3),dtype=np.uint8),stamp_s=10.,frame_id='camera',sequence=0)
            image=base64.b64decode(record.pop('image_base64'))
            captured=[]
            response=Reply.fixture(dict(tool='restart_map_localization',arguments=dict(pose=[1.,2.,0.],evidence='墙角'))).message['tool_calls']
            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200,text=wire(response))
            store=Store(root/'session.sqlite');budget=Budget(root/'budget.sqlite')
            model=Model(config,budget,store,transport=httpx.MockTransport(handler))
            payload=dict(messages=[dict(role='user',content='乘梯')],
                current_state=dict(requirements_registered=True,ride=dict(status='awaiting_destination_localization'),
                                   navigation=dict(feedback=dict(stamp_s=10.,clock_domain='ros_time'))),
                ride_inputs=[('current_elevator_rgb',record,image,'JPEG')])
            try:
                self.assertEqual(model.decide(payload).action['tool'],'restart_map_localization')
                offered={tool['function']['name'] for tool in captured[-1]['tools']}
                self.assertIn('restart_map_localization',offered)
                self.assertNotIn('pause_map_localization',offered)
                self.assertNotIn('record_elevator_observation',offered)
                state=json.loads(captured[-1]['messages'][-1]['content'][0]['text'])['state']
                self.assertEqual(state['ride']['status'],'awaiting_destination_localization')
                response[:]=Reply.fixture(dict(tool='pause_map_localization',arguments=dict(goal_id='lift'))).message['tool_calls']
                with self.assertRaises(ApiError):model.decide(payload)
                payload['current_state']['ride']['status']='unknown'
                with self.assertRaisesRegex(ValueError,'Unknown elevator phase'):model.decide(payload)
            finally:
                store.close();budget.close()
