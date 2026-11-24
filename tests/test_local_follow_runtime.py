"""Exercise single-frame ReID through the normal T2 worker and control consumer."""
import copy
import hashlib
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from decision.agent.adapters.runtime import follow_observation
from decision.osnet.reid import FollowReid
from decision.vlm.t2.target import reference
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.search import follow_candidates
from perception.environment.context import fingerprint
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from runtime.control import ControlLoop
from runtime.decision import DecisionBuffer
from runtime.follow import FollowRunner
from runtime.follow_decision import FollowDecisionWorker
from runtime.metrics import Metrics
from tests.decision_fixtures import follow_search, observation
from tests.navigation_fixtures import asset


def frame(stamp, *, ambiguous=False, corrupt=False, lost=False, associated=False,
          box=None, position=None, vehicle=None, occupied=None, predicted=False):
    people, arrays, association, scene = observation(stamp)
    pixels = arrays['rgb']
    pixels[:] = (245, 10, 10)
    if corrupt:
        pixels[:] = (10, 245, 10)
    else:
        association['image_input']['raw_rgb_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
    association['image_input']['camera_frame'] = 'camera'
    box = [20., 5., 40., 45.] if box is None else box
    association.update(detector_identity='synthetic', detections=[dict(box=box, score=.9, class_id=0)])
    association['result']['tracks']['a/g7'].update(detection_index=0, rgb_box=box)
    if ambiguous:
        association['detections'].append(dict(box=[40., 5., 60., 45.], score=.9, class_id=0))
    if lost:
        association['detections'].clear()
        association['result']['tracks']['a/g7'] = dict(status='unmatched')
    if associated or position is not None or predicted:
        grid = scene.geometry
        if vehicle is not None:
            pose = np.eye(4)
            pose[:2, 3] = vehicle
            grid.metadata['world_from_lidar'] = pose.tolist()
            people['world_from_lidar'] = pose.tolist()
        if occupied is not None:
            cells = grid.layers['occupancy'].copy()
            cells[occupied] = 100
            grid.layers['occupancy'] = cells
        if associated:
            grid.metadata['association'] = dict(valid=True, map_id='map-a', map_frame='map', local_frame='odom',
                local_session_id='test', local_generation=0, stamp_s=stamp, measurement_stamp_s=stamp,
                map_from_local=np.eye(4).tolist(), association_session='association', association_generation=0)
        if position is not None:
            people['tracks'][0]['box'][:2] = position
        if predicted:
            people['tracks'][0].update(state='predicted',last_observed_stamp_s=stamp-.1)
        people['input_fingerprint'] = fingerprint({key: value for key, value in people.items()
                                                    if key != 'input_fingerprint'})
        association['perception_input_fingerprint'] = people['input_fingerprint']
        scene = assemble_scene(grid, clock_domain='synthetic', settings=environment_config()['scene'], people=people)
    return people, arrays, association, scene


def encode(crops):
    return [np.asarray(crop).mean(axis=(0, 1)) for crop in crops]


class LocalFollowRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.states = []
        self.pauses = []
        self.frames = []
        self.handle = dict(task_id='mission', goal_id='person', plan_revision=1, goal_revision=1, call_id='call')
        _, arrays, association, source = frame(10.)
        observation = follow_observation(source, arrays['rgb'], association, None, None)
        self.reference, _, self.crop = reference(observation, 0, 'red shirt', task_id='mission', revision=1, goal_id='person')
        self.worker = FollowDecisionWorker(follow_search, backend='reid', reid_factory=self.matcher,
            buffer=DecisionBuffer(max_frames=16, slot_bytes=32768), publish_agent_frame=self.publish)
        self.addCleanup(self.worker.close)
        self.addCleanup(self.worker.stop)
        self.begin()
        geometry = VehicleGeometry([[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]], np.eye(4), source='test')
        self.control = control = object.__new__(ControlLoop)
        control.metrics = Metrics()
        control.lock = threading.Lock()
        control.follow = FollowRunner(load_config(), geometry)
        control.follow.handle = copy.deepcopy(self.handle)
        control.follow.reid_reference_id = self.reference['reference_id']
        control.follow_pending = dict(handle=self.handle, intent=dict(reference=self.reference))
        control.follow_decision = self.worker
        control.follow_search_anchor = control.follow_search_choice = None
        control.request = control.cancelled_request = control.asset = None
        control.clock_domain = 'synthetic'
        control.record = lambda name, event: self.events.append(event)
        control.reference = SimpleNamespace(planning=load_config(), invalidate=lambda *_: None, cancel=lambda *_: None)
        control.controller = SimpleNamespace(controller=SimpleNamespace(geometry=geometry, clear=lambda: None))
        control.commands = SimpleNamespace(follow_goal=lambda *_: True, pause_follow=self.pause,
            follow_owner_valid=lambda _: True, follow_failed=lambda *args, **kwargs: dict(target=self.pause(self.handle)),
            agent_state=SimpleNamespace(publish_follow=lambda state: self.states.append(copy.deepcopy(state))))

    def begin(self):
        self.worker.begin(self.handle, 'red shirt', reference=self.reference, crop=self.crop)

    def pause(self, handle):
        self.pauses.append(handle)
        return dict(task_id='follow:call', goal_revision=0)

    def matcher(self, crop, record):
        return FollowReid(encode, crop, record)

    def publish(self, handle, scene, pixels, association, camera_info, *, bound_candidates=None):
        result = follow_observation(scene, pixels, association, None, camera_info,bound_candidates=bound_candidates)
        self.frames.append((handle, result))

    def drain(self):
        deadline = time.monotonic()+3.
        while self.worker.snapshot()['inflight'] and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(self.worker.snapshot()['inflight'])

    def feed(self, stamp, *, apply=True, **options):
        people, arrays, association, scene = frame(stamp, **options)
        self.control.follow.observe(scene)
        self.control.scene = scene
        self.assertTrue(self.worker.submit(people, arrays, scene=scene, image_metadata=None,
                                           association=association, context=self.control.follow_context()))
        self.drain()
        if apply:
            self.control._update_follow(stamp)
        return scene

    def test_single_frame_selection_without_vlm_and_ordinary_completion_evidence(self):
        with patch('runtime.follow_decision.render_window', side_effect=AssertionError('rendered VLM window')), \
                patch('runtime.follow_decision.FollowRequest', side_effect=AssertionError('called VLM')):
            self.feed(10.1)
            self.assertIsNone(self.control.request)
            self.feed(10.2)
        self.assertEqual(self.states[-1]['status'], 'running')
        self.assertEqual(self.states[-1]['target_identity'], ['tracker_session', 7, 'a'])
        self.assertEqual(self.control.request['request_id'], self.handle['call_id'])
        published_handle, agent_frame = self.frames[-1]
        self.assertEqual(published_handle, self.handle)
        self.assertEqual(agent_frame.record['follow_candidates']['binding'], self.events[-1]['candidate_binding'])
        self.assertEqual(self.control.follow.last_evidence['scene_binding'], agent_frame.record['scene_binding'])

    def test_loaded_model_provenance_reaches_control_and_timing(self):
        self.worker.reid_factory = None
        self.worker.reid_weights = '/tmp/osnet.engine'
        self.worker.reid_device = 'cuda:0'
        self.worker.metrics = Metrics(trace_path='/tmp/reid-provenance-test.jsonl')
        model = dict(backend='tensorrt_cuda', path=self.worker.reid_weights, sha256='1' * 64,
                     onnx_sha256='2' * 64, device='cuda:0', tensorrt_version='10.13.3')
        encoder = Mock(side_effect=encode, **model)
        encoder.path = Path(encoder.path)
        self.assertIsNone(self.worker.snapshot()['reid_model'])
        with patch('decision.osnet.reid.OsnetEmbedder', return_value=encoder) as loader:
            self.feed(10.1)
            self.feed(10.2)
        loader.assert_called_once_with(self.worker.reid_weights, device='cuda:0')
        self.assertEqual(self.worker.snapshot()['reid_model'], model)
        self.assertEqual(self.events[-1]['reid_model'], model)
        self.assertEqual(self.events[-1]['reid_model_sha256'], model['sha256'])
        loaded, = [row for row in self.worker.metrics.trace_rows if row['event'] == 't2_reid_model_loaded']
        self.assertEqual(loaded['reid_model'], model)

    def test_engine_load_failure_stops_follow_without_cpu_fallback(self):
        self.worker.reid_factory = None
        self.worker.reid_weights = '/tmp/osnet.engine'
        with patch('decision.osnet.reid.OsnetEmbedder', side_effect=ValueError('engine mismatch')) as loader:
            self.feed(10.1)
        loader.assert_called_once_with(self.worker.reid_weights, device=None)
        self.assertIsNone(self.worker.snapshot()['reid_model'])
        self.assertIsNone(self.control.request)
        self.assertEqual(self.states[-1]['status'], 'failed')
        self.assertEqual(self.states[-1]['reason'], 'follow_t2_engine mismatch')

    def test_ambiguous_and_missing_pixels_pause_and_require_new_confirmations(self):
        self.feed(10.1)
        self.feed(10.2)
        self.feed(10.3, ambiguous=True)
        self.assertIsNone(self.control.request)
        self.assertEqual(self.states[-1]['reason'], 'target_region_ambiguous')
        self.assertTrue(self.pauses)
        self.feed(10.4)
        self.assertIsNone(self.control.request)
        self.feed(10.5)
        self.assertEqual(self.states[-1]['status'], 'running')
        self.feed(10.6, corrupt=True)
        self.assertIsNone(self.control.request)
        self.assertEqual(self.events[-1]['event'], 'follow_t2_input_unavailable')
        self.feed(10.7)
        self.assertIsNone(self.control.request)

    def test_stale_and_cancelled_results_cannot_acquire_owner(self):
        self.feed(10.1)
        self.feed(10.2, apply=False)
        self.control._update_follow(11.)
        self.assertIsNone(self.control.request)
        self.assertIn('expired', self.events[-1]['reason'])
        self.feed(11.1)
        self.assertIsNone(self.control.request)
        self.feed(11.2, apply=False)
        self.worker.stop()
        self.assertIsNone(self.worker.poll())

    def test_idle_preview_uses_one_frame_without_creating_the_matcher(self):
        self.worker.stop()
        self.feed(10.1, apply=False)
        self.assertIsNone(self.frames[-1][0])
        self.assertIsNone(self.worker.reid)
        self.assertIsNone(self.worker.poll())

    def test_bound_result_checks_original_reference_and_lifecycle(self):
        self.feed(10.1)
        scene = self.feed(10.2, apply=False)
        evidence = self.worker.poll()['local']
        for field in ('reference_id', 'lifecycle_generation'):
            with self.subTest(field=field):
                altered = copy.deepcopy(evidence)
                if field == 'reference_id':
                    altered[field] = 'wrong'
                else:
                    altered['decision'][field] += 1
                altered['binding'] = fingerprint({k: v for k, v in altered.items() if k != 'binding'})
                with self.assertRaises(ValueError):
                    self.control.follow.apply_reid(self.handle, altered, scene, now=10.2)
                self.assertIsNone(self.control.follow.policy)

    def test_waiting_for_new_visual_result_keeps_issued_goal_fixed(self):
        self.feed(10.1)
        self.feed(10.2)
        issued = copy.deepcopy(self.control.request)
        _, _, _, newer = frame(10.3, position=[4.5, 2.])
        self.control.follow.observe(newer)
        self.control.scene = newer
        self.control._update_follow(10.3)
        self.assertEqual(self.control.request['goal'], issued['goal'])
        self.assertEqual(self.control.request['goal_revision'], issued['goal_revision'])
        self.assertGreater(self.control.request['valid_until_s'], issued['valid_until_s'])
        self.feed(10.4, lost=True, position=[4.6, 2.])
        self.assertEqual(self.control.request['goal'], issued['goal'])
        self.feed(10.5, lost=True, predicted=True)
        self.assertIsNone(self.control.request)

    def test_last_position_uses_visual_source_even_when_3d_is_newer(self):
        self.feed(10.1)
        self.feed(10.2, apply=False)
        _, _, _, newer = frame(10.3, position=[4.5, 2.])
        self.control.follow.observe(newer)
        self.control.scene = newer
        self.control._update_follow(10.3)
        self.assertEqual(self.control.follow.reid_last_evidence['stamp_s'], 10.2)
        self.assertEqual(self.control.follow.reid_last_evidence['position_xy_m'], [4., 2.])
        self.assertEqual(self.control.follow.last_evidence['position_xy_m'], [4.5, 2.])

    def test_manual_preview_uses_the_same_fixed_recovery_goal_and_withdraws_it_on_ambiguity(self):
        self.control.manual_follow = True
        previews = []
        self.control.commands.follow_preview = lambda handle, request: previews.append(copy.deepcopy(request))
        self.start_recovery()
        self.assertIsNone(self.control.request)
        first = previews[-1]
        self.assertEqual(first['goal'], [4.5,2.])
        self.assertEqual(first['mode'], 'mapped')
        self.feed(11.8, lost=True, associated=True)
        self.assertEqual(previews[-1]['goal_revision'],first['goal_revision'])
        self.assertGreater(previews[-1]['valid_until_s'],first['valid_until_s'])
        pauses = len(self.pauses)
        self.feed(12., ambiguous=True, associated=True)
        self.assertEqual(len(self.pauses),pauses+1)
        self.assertIsNone(self.control.follow_search_intent)

    def start_recovery(self):
        self.control.asset = asset()
        self.worker.search_factory = lambda scene, context: follow_candidates(
            scene, self.control.asset, self.control.follow.geometry, context['last'],
            planning=self.control.reference.planning, local_reid=True, loss=context['loss'],
            continuation=context['continuation'])
        self.feed(10.1, associated=True)
        self.feed(10.2, associated=True)
        self.feed(10.4, associated=True, position=[4.5, 2.], box=[0., 5., 22., 45.])
        for i in range(6):
            self.feed(10.6+.2*i, lost=True, associated=True)
            if i < 5:
                self.assertIsNone(self.control.request)

    def test_fresh_loss_renews_same_safe_search_and_silence_expires_it(self):
        invalidations = []
        self.control.reference.invalidate = invalidations.append
        self.start_recovery()
        first = copy.deepcopy(self.control.request)
        self.assertEqual(first['goal'], [4.5, 2.])
        self.assertEqual(first['completion_policy'], 'goal')
        for i in range(26):
            stamp=round(11.8+.2*i, 2)
            self.feed(stamp, lost=True, associated=True, vehicle=[2.+i*.08, 2.])
            renewed = self.control.request
            self.assertEqual(first['goal_revision'], renewed['goal_revision'])
            self.assertEqual(first['goal'], renewed['goal'])
        self.assertGreater(renewed['valid_until_s'], first['valid_until_s']+4.)
        self.assertEqual(invalidations, ['follow_search_goal'])
        self.control._update_follow(renewed['valid_until_s']+.01)
        self.assertIsNone(self.control.request)
        self.assertEqual(self.states[-1]['reason'], 'follow_reid_search_observation_expired')

    def test_recovery_pauses_after_new_endpoint_occupancy(self):
        self.start_recovery()
        self.feed(11.8, lost=True, associated=True, occupied=(20, 45), apply=False)
        with patch.object(self.worker, 'poll', return_value=None):
            self.control._update_follow(11.8)
        self.assertIsNone(self.control.request)
        self.assertEqual(self.states[-1]['reason'], 'follow_last_position_currently_occupied')

    def test_recovery_arrival_waits_and_new_visual_confirmation_resumes_follow(self):
        self.start_recovery()
        self.feed(11.8, lost=True, associated=True, vehicle=[4.48, 2.])
        self.assertIsNone(self.control.request)
        self.feed(12., associated=True, position=[4.8, 2.], vehicle=[4.48, 2.])
        self.feed(12.2, associated=True, position=[5., 2.], vehicle=[4.48, 2.])
        self.assertEqual(self.states[-1]['status'], 'running')
        self.assertEqual(self.control.request['completion_policy'], 'continuous')

    def test_recovery_turn_uses_observed_exit_direction_then_renews_in_place(self):
        self.control.asset = asset()
        self.worker.search_factory = lambda scene, context: follow_candidates(
            scene, self.control.asset, self.control.follow.geometry, context['last'],
            planning=self.control.reference.planning, local_reid=True, loss=context['loss'],
            continuation=context['continuation'])
        self.feed(10.1, associated=True)
        self.feed(10.2, associated=True)
        self.feed(10.4, associated=True, position=[4.25, 2.])
        self.feed(10.6, associated=True, position=[4.25, 2.22], box=[31., 5., 51., 45.])
        self.feed(10.8, associated=True, position=[4.25, 2.45], box=[39., 5., 59., 45.])
        for i in range(6):
            self.feed(round(11.+.2*i,2), lost=True, associated=True)
        first = copy.deepcopy(self.control.request)
        np.testing.assert_allclose(first['goal'],[4.25,2.45,np.pi/2])
        self.feed(12.2, lost=True, associated=True, vehicle=[4.25,2.45])
        self.assertEqual(first['goal'],self.control.request['goal'])
        self.assertEqual(first['goal_revision'],self.control.request['goal_revision'])
        self.assertGreater(self.control.request['valid_until_s'],first['valid_until_s'])
