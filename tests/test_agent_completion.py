"""Current completion protocol, manual follow isolation and T3 sensor proposals."""
from collections import deque
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.navigation_fixtures import bundle
from decision.agent.tools.observations import Observation
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from runtime.command import CommandInbox
from runtime.seating import SeatingWorker


class LocalTaskTests(unittest.TestCase):
    def test_round_table_model_proposes_and_revises_observation_pose(self):
        import json
        scene = bundle()
        geometry = VehicleGeometry([[-.15,-.15],[.2,-.15],[.2,.15],[-.15,.15]],
                                   np.eye(4), source='synthetic', assumed=False)
        worker = SeatingWorker(geometry,load_config(),load_motion_config())
        circle = dict(id='p-03-1',map_id=scene.record['prior']['map_id'],
                      center_xy_m=[1.5,0.],radius_m=.35)
        rgb = np.full((32,48,3),170,np.uint8)
        obs = Observation.from_scene(scene,geometry=geometry,rgb=dict(pixels=rgb,stamp_s=10.,frame_id='camera'))
        clock = (10.,1_000_000.)
        sent = []
        def infer(body, **_):
            fields = json.loads(body['messages'][1]['content'][-1]['text'])
            sent.append(fields)
            pose = [99.,0.,0.] if len(sent)==1 else [0.,.65,-.45]
            return dict(status='complete',raw_content=json.dumps(dict(s='o',pose=pose,focus='See the side of the table')))
        with patch('runtime.seating.time.monotonic',return_value=clock[1]), \
             patch.object(worker,'_route',return_value=(None,False)) as route, \
             patch('decision.vlm.online.request_dspark',side_effect=infer):
            try:
                worker.observe(scene,obs,rgb,np.empty((0,3)),{},clock)
                worker.update_scene(scene,clock)
                worker.select(dict(selection_id='view',instruction='Park beside the round table',
                                   round_table=circle,observation_binding=obs.record['input_fingerprint'],
                                   manual_observation=dict(after_s=0.,visited_xy_m=[])))
                worker.thread.join(6.)
                result = worker.snapshot()
                self.assertEqual(result['status'],'observation',result)
                self.assertEqual(result['observation_focus'],'See the side of the table')
                self.assertEqual(sent[1]['view_search']['rejected_views'][0]['reason'],
                                 'seating_view_outside_displayed_area')
                self.assertEqual(len(result['view_checks']),2)
                route.assert_called_once()
            finally:
                worker.close()

    def test_round_table_reselects_after_path_rejection(self):
        from decision.vlm.t3.geometry import round_candidates, round_clearance
        scene = bundle()
        geometry = VehicleGeometry([[-.15,-.15],[.2,-.15],[.2,.15],[-.15,.15]],
                                   np.eye(4), source='synthetic', assumed=False)
        worker = SeatingWorker(geometry, load_config(), load_motion_config())
        table = dict(id='p-03-1', map_id=scene.record['prior']['map_id'],
                     center_xy_m=[1.5, 0.0], radius_m=.4)
        gap = worker.seating['table_clearance_m']
        self.assertEqual(worker.planning['safety_margin_m'],0.)
        self.assertEqual(gap,.10)
        for candidate in round_candidates(table, geometry.footprint, gap):
            self.assertAlmostEqual(round_clearance(table, geometry.polygon(candidate['pose'])), gap)
        self.assertLess(round_clearance(table, geometry.polygon([1.5, 0., 0.])), 0.)
        rgb = np.full((32, 48, 3), 170, np.uint8)
        obs = Observation.from_scene(scene, geometry=geometry,
                                     rgb=dict(pixels=rgb, stamp_s=10., frame_id='camera'))
        selection = dict(selection_id='round-table', instruction='Park beside the round table',
                         round_table=table, observation_binding=obs.record['input_fingerprint'])
        clock = (10., 1_000_000.)
        denied = dict(status='no_path', current_path_valid=False, reason='synthetic_obstruction')
        allowed = dict(status='path_found', current_path_valid=True)
        replies = [dict(status='complete', raw_content='{"p":0,"s":"v"}'),
                   dict(status='complete', raw_content='{"p":1,"s":"v"}')]
        with patch('runtime.seating.time.monotonic', return_value=clock[1]), \
             patch('navigation.planning.footprint.footprint_status', return_value='free'), \
             patch('navigation.planning.planner.Planner.query', side_effect=[denied, allowed]), \
             patch('decision.vlm.online.request_dspark', side_effect=replies):
            try:
                worker.observe(scene, obs, rgb, np.empty((0, 3)), {}, clock)
                worker.update_scene(scene, clock)
                worker.select(selection)
                worker.thread.join(6.)
                result = worker.snapshot()
                self.assertEqual(result['status'], 'selected', result)
                self.assertEqual(result['candidate_id'], 1)
                self.assertEqual(result['mapped_round_table']['center_xy_m'], table['center_xy_m'])
                first, second = result['attempts']
                self.assertEqual(first['rejection'], 'synthetic_obstruction')
                self.assertEqual(second['candidates'][0]['geometry'], 'blocked')
            finally:
                worker.close()

    def test_manual_follow_only_updates_preview_even_on_cancel(self):
        inbox=object.__new__(CommandInbox)
        inbox.session=SimpleNamespace(live=True,execute=False)
        inbox.lock=threading.Lock()
        inbox.manual_preview_owner=inbox.agent_owner=None
        inbox.pending=deque()
        inbox.follow_enabled=True
        from tests.test_follow_recovery import candidate_observation
        from decision.vlm.t2.target import reference
        import base64
        obs=candidate_observation(10.)
        control=dict(request=None,stamp_s=10.,clock_domain=obs.record['clock_domain'],map_binding=obs.record['map_binding'])
        inbox.agent_state=SimpleNamespace(snapshot=lambda:dict(received_age_s=.1,control=control),
            follow_candidate=lambda *_:(obs.record,base64.b64encode(obs.image).decode()),
            clear_follow_observations=Mock(), reserve_follow_observations=Mock())
        inbox.safety=SimpleNamespace(command=lambda *_:self.fail('Manual preview called the safety command channel'))
        handle=dict(task_id='mission',goal_id='follow',call_id='turn',plan_revision=1,goal_revision=1)
        ref,_,_=reference(obs,0,'红衣',task_id='mission',revision=1,goal_id='follow')
        inbox.agent_follow('start',dict(handle=handle,intent=dict(target_description='红衣',reference=ref)),live=False,manual=True)
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
        inbox.agent_state.reserve_follow_observations.assert_called_once_with(handle)
        inbox.agent_state.clear_follow_observations.assert_called_once_with(stopped_call_id='turn')

    def test_current_table_scan_through_model_selection_and_path_check(self):
        scene=bundle()
        geometry=VehicleGeometry([[-.15,-.15],[.2,-.15],[.2,.15],[-.15,.15]],np.eye(4),source='synthetic',assumed=False)
        worker=SeatingWorker(geometry,load_config(),load_motion_config(),clock_rate=2.)
        worker.calibration=dict(matrix=[[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        rgb=np.full((96,128,3),170,np.uint8)
        obs=Observation.from_scene(scene,geometry=geometry,rgb=dict(pixels=rgb,stamp_s=10.,frame_id='camera'),
            sensors=dict(camera=dict(view=dict(status='approximate_pinhole'))))
        ys=np.linspace(-.8,.8,60)
        scan=np.column_stack([np.full_like(ys,1.5),ys,np.full_like(ys,.55)])
        camera=dict(K=[60.,0,64.,0,60.,48.,0,0,1],D=[0.]*5)
        selection=dict(selection_id='stage1',instruction='在桌前停靠',roi=[0.,0.,1.,.6],
                       observation_binding=obs.record['input_fingerprint'])
        feedback_time=1_000_000.
        with patch('runtime.seating.time.monotonic',return_value=feedback_time) as mono, \
             patch('decision.vlm.online.request_dspark',return_value=dict(status='complete',raw_content='{"p":1,"s":"v"}')) as infer:
            try:
                clock=(10.,feedback_time)
                worker.observe(scene,obs,rgb,scan,camera,clock)
                worker.update_scene(scene,clock)
                worker.select(selection)
                worker.thread.join(6)
                result=worker.snapshot()
                self.assertEqual(result['status'],'selected',result)
                self.assertEqual(result['candidate_id'],1)
                self.assertEqual(result['path_status'],'path_found')
                self.assertEqual(result['source_stamp_s'],10.)
                self.assertEqual(result['geometry_stamp_s'],10.)
                self.assertTrue(np.isfinite(result['pose']).all())
                from tests.navigation_fixtures import scene_at
                next_scene=scene_at(10.1)
                next_obs=Observation.from_scene(next_scene,geometry=geometry,rgb=dict(pixels=rgb,stamp_s=10.1,frame_id='camera'),
                    sensors=dict(camera=dict(view=dict(status='approximate_pinhole'))))
                clock=(10.1,feedback_time)
                worker.update_scene(next_scene,clock)
                worker.observe(next_scene,next_obs,rgb,scan,camera,clock)
                worker.thread.join(6)
                updated=worker.snapshot()
                self.assertEqual(updated['status'],'selected',updated)
                self.assertEqual(updated['source_stamp_s'],10.1)
                self.assertEqual(updated['sequence'],2)
                self.assertEqual(infer.call_count,2)
                worker.cancel(selection['selection_id'])
                gate=threading.Event();started=threading.Event()
                def delayed(*_args,**_kwargs):
                    started.set();gate.wait(3.)
                    return dict(status='complete',raw_content='{"p":1,"s":"v"}')
                infer.side_effect=delayed
                worker.select(dict(selection,selection_id='late'))
                self.assertTrue(started.wait(3.))
                worker.cancel('late');gate.set();worker.thread.join(6)
                self.assertEqual(worker.snapshot()['status'],'cancelled')
                with self.assertRaisesRegex(ValueError,'expired'):
                    worker.select(dict(selection,observation_binding='wrong-frame'))

                def late_response(*_args,**_kwargs):
                    mono.return_value=feedback_time+1.1  # 2.2 seconds on the source clock.
                    return dict(status='complete',raw_content='{"p":1,"s":"v"}')
                infer.side_effect=late_response
                worker.select(selection)
                worker.thread.join(6)
                self.assertEqual(worker.snapshot()['reason'],'Decision response deadline exceeded')
                infer.reset_mock()
                worker.select(selection)
                worker.thread.join(6)
                self.assertEqual(worker.snapshot()['reason'],'Request observation is already expired')
                infer.assert_not_called()
            finally:worker.close()
