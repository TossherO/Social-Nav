"""Elevator judgments keep history only in the transient model request."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

import httpx
import numpy as np
from decision.agent.core.model import Model
from decision.agent.core.store import Budget,Store
from decision.agent.tools.framework import Reply
from decision.agent.workflows.elevator_camera import image_record
from tests.test_agent_conversation import wire


class ElevatorModelTests(unittest.TestCase):
    def test_transient_request_uses_history_without_persisting_images_or_judgments(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);config=root/'private.json';session=root/'session.sqlite'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            record=image_record(np.zeros((10,10,3),dtype=np.uint8),stamp_s=10.,frame_id='camera',sequence=0)
            image=base64.b64decode(record.pop('image_base64'))
            native=Reply.fixture(dict(tool='record_elevator_observation',arguments=dict(judgment='inside',evidence='private-evidence',retain_observation=True))).message['tool_calls']
            captured=[]
            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200,text=wire(native))
            store=Store(session);budget=Budget(root/'budget.sqlite')
            model=Model(config,budget,store,transport=httpx.MockTransport(handler))
            payload=dict(messages=[dict(role='user',content='乘梯')],
                current_state=dict(requirements_registered=True,navigation=dict(feedback=dict(stamp_s=10.,clock_domain='ros_time'))),
                ride_inputs=[('current_elevator_rgb',record,image,'JPEG')])
            try:
                result=model.decide(payload)
                model.decide(dict(payload,ride_inputs=None))
            finally:store.close();budget.close()
            self.assertEqual(result.action['arguments']['judgment'],'inside')
            self.assertEqual(captured[0]['tools'],captured[1]['tools'])
            self.assertEqual(captured[0]['messages'][0],captured[1]['messages'][0])
            self.assertIn('data:image/jpeg;base64,',json.dumps(captured[0]))
            store=Store(session)
            try:events=str(store.db.execute('select value from events order by seq limit 2').fetchall())
            finally:store.close()
            budget=Budget(root/'budget.sqlite')
            try:records=str(budget.db.execute('select record from calls').fetchall())
            finally:budget.close()
            for output in (events,records):
                self.assertNotIn('data:image/',output)
                self.assertNotIn('private-evidence',output)
