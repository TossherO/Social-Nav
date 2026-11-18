"""Elevator judgments keep history only in the transient model request."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from decision.agent.core.elevator_model import decide_elevator
from decision.agent.core.store import Budget,Store
from decision.agent.tools.framework import Reply
from decision.agent.workflows.elevator_camera import image_record
from tests.test_agent_conversation import wire


class ElevatorModelTests(unittest.TestCase):
    def test_transient_request_uses_history_without_persisting_images_or_judgments(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);config=root/'private.json';session=root/'session.sqlite'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            initial=Store(session);initial.close()
            record=image_record(np.zeros((10,10,3),dtype=np.uint8),stamp_s=10.,frame_id='camera',sequence=0)
            image=base64.b64decode(record.pop('image_base64'))
            frame=dict(record=record,image=image,status='available')
            native=Reply.fixture(dict(tool='record_elevator_observation',arguments=dict(judgment='inside',evidence='private-evidence',retain_observation=True))).message['tool_calls']
            captured=[]
            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200,text=wire(native))
            with patch('decision.agent.core.elevator_model.shared_budget',side_effect=lambda:Budget(root/'budget.sqlite')):
                result=decide_elevator(config,session,dict(status='awaiting_entry'),[],frame,transport=httpx.MockTransport(handler))
            self.assertEqual(result['judgment'],'inside')
            self.assertEqual(captured[0]['tools'][0]['function']['name'],'record_elevator_observation')
            self.assertIn('data:image/jpeg;base64,',json.dumps(captured[0]))
            store=Store(session)
            try:events=str(store.db.execute('select value from events').fetchall())
            finally:store.close()
            budget=Budget(root/'budget.sqlite')
            try:records=str(budget.db.execute('select record from calls').fetchall())
            finally:budget.close()
            for output in (events,records):
                self.assertNotIn('data:image/',output)
                self.assertNotIn('private-evidence',output)
