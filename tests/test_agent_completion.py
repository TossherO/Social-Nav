"""Current completion protocol, manual follow isolation and T3 sensor proposals."""
from collections import deque
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from tests.navigation_fixtures import bundle
from decision.agent.tools.observations import Observation
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from runtime.command import CommandInbox
from runtime.seating import SeatingWorker


class LocalTaskTests(unittest.TestCase):
    def test_manual_follow_only_updates_preview_even_on_cancel(self):
        inbox=object.__new__(CommandInbox)
        inbox.session=SimpleNamespace(live=True,execute=False)
        inbox.lock=threading.Lock()
        inbox.manual_preview_owner=inbox.agent_owner=None
        inbox.pending=deque()
        inbox.follow_enabled=True
        inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(received_age_s=.1,control=dict(request=None)))
        inbox.safety=SimpleNamespace(command=lambda *_:self.fail('Manual preview called the safety command channel'))
        handle=dict(task_id='mission',goal_id='follow',call_id='turn',plan_revision=1,goal_revision=1)
        inbox.agent_follow('start',dict(handle=handle,intent=dict(target_description='红衣')),live=False,manual=True)
        goal=dict(request_id='turn',task_id='follow:turn',goal_revision=1,mode='mapped',frame_id='map',
                  target_reference='base_link',goal=[1,2],map_snapshot_id='map-a')
        self.assertFalse(inbox.follow_goal(handle,goal))
        inbox.follow_preview(handle,goal)
        self.assertEqual(inbox.manual_preview_snapshot()['request'],goal)
        with self.assertRaises(ValueError):
            inbox.agent_follow('cancel',dict(call_id='turn'),live=False)
        inbox.agent_follow('finish',dict(call_id='turn'),live=False,manual=True)
        self.assertIsNone(inbox.manual_preview_snapshot())
        self.assertIsNone(inbox.agent_owner)

    def test_current_table_scan_through_model_selection_and_path_check(self):
        scene=bundle()
        geometry=VehicleGeometry([[-.15,-.15],[.2,-.15],[.2,.15],[-.15,.15]],np.eye(4),source='synthetic',assumed=False)
        worker=SeatingWorker(geometry,load_config(),load_motion_config())
        worker.calibration=dict(matrix=[[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        rgb=np.full((96,128,3),170,np.uint8)
        obs=Observation.from_scene(scene,geometry=geometry,rgb=dict(pixels=rgb,stamp_s=10.,frame_id='camera'),
            sensors=dict(camera=dict(view=dict(status='approximate_pinhole'))))
        ys=np.linspace(-.8,.8,60)
        scan=np.column_stack([np.full_like(ys,1.5),ys,np.full_like(ys,.55)])
        camera=dict(K=[60.,0,64.,0,60.,48.,0,0,1],D=[0.]*5)
        selection=dict(selection_id='stage1',instruction='在桌前停靠',roi=[0.,0.,1.,.6],
                       observation_binding=obs.record['input_fingerprint'])
        feedback_time=time.monotonic()
        with patch('runtime.seating.time.monotonic',return_value=feedback_time), \
             patch('decision.vlm.online.request_dspark',return_value=dict(status='complete',raw_content='{"p":1,"s":"v"}')):
            try:
                worker.observe(scene,obs,rgb,scan,camera,time.monotonic())
                worker.update_scene(scene,feedback_time)
                worker.select(selection)
                worker.thread.join(6)
                result=worker.snapshot()
                self.assertEqual(result['status'],'selected',result)
                self.assertEqual(result['candidate_id'],1)
                self.assertEqual(result['path_status'],'path_found')
                self.assertEqual(result['source_stamp_s'],10.)
                self.assertEqual(result['geometry_stamp_s'],10.)
                self.assertTrue(np.isfinite(result['pose']).all())
                with self.assertRaisesRegex(ValueError,'expired'):
                    worker.select(dict(selection,observation_binding='wrong-frame'))
            finally:worker.close()
