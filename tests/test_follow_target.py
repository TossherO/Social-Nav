"""First follow selection, source binding and immutable visual references."""
import copy
import hashlib
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.observations import Observation
from decision.agent.tools.places import Catalog
from decision.vlm.follow_target import annotated, candidates, reference, verify_candidates
from tests.test_agent import ScriptedFeedback
from tests.test_decision_runtime import observation


class FollowTargetTests(unittest.TestCase):
    def setUp(self):
        self.row, self.arrays, _, self.scene = observation(10.)
        self.detections = [dict(box=[20.,5.,35.,40.],score=.9,class_id=0),
                           dict(box=[3.,5.,15.,40.],score=.8,class_id=0)]
        self.association = dict(status='complete',tracks={'a/g7':dict(status='matched',
            rgb_box=[20.,5.,35.,40.],detection_index=0)})
        self.value = candidates(self.scene,self.arrays['rgb'],9.98,self.detections,self.association,
                                detector=dict(backend='test',model_sha256='fixture'))
        self.obs = Observation.from_scene(self.scene, rgb=dict(pixels=self.arrays['rgb'],stamp_s=9.98,frame_id='camera'),
                                          follow_candidates=self.value)
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'session.sqlite')
        self.catalog = Catalog.load(ROOT/'decision/agent/configs/simulation.json')
        self.nav = ScriptedFeedback()
        self.nav.clock_domain = self.scene.record['clock_domain']
        self.nav.state.update(clock_domain=self.nav.clock_domain,session_id=self.scene.record['session_id'],
                              generation=self.scene.record['generation'],frame_id=self.scene.record['frame_id'])
        self.nav.observe = lambda:self.obs
        self.session = AgentSession('跟随穿红衣的人',self.catalog,self.nav,self.store)
        goal = dict(goal_id='follow',kind='skill',target_id='',stage_type='follow',description='跟随穿红衣的人')
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        self.assertEqual(self.session.call('observe_scene')['status'],'available')

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_reference_binds_image_box_and_track_without_motion(self):
        record,image,crop=reference(self.obs,0,'穿红衣的人',task_id='task',revision=1,goal_id='follow')
        self.assertEqual(record['track']['identity'],['tracker_session',7,'a'])
        self.assertEqual(image,self.obs.image)
        self.assertEqual(hashlib.sha256(crop).hexdigest(),record['crop_sha256'])
        self.assertNotEqual(annotated(self.obs),self.obs.image)
        modified=copy.deepcopy(self.value)
        modified['candidates'][0]['box'][0]+=1
        with self.assertRaises(ValueError):verify_candidates(modified,self.obs)

    def test_old_agent_selection_is_unavailable(self):
        self.assertNotIn('select_follow_target',self.session.registry)
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertIsNone(self.store.read_follow_reference(self.session.state['task_id'],1,'follow'))

    def test_invalid_association_and_future_image_are_rejected(self):
        wrong=copy.deepcopy(self.association)
        wrong['tracks']['a/g7']['rgb_box']=[1.,1.,3.,3.]
        with self.assertRaises(ValueError):
            candidates(self.scene,self.arrays['rgb'],9.98,self.detections,wrong,detector={})
        with self.assertRaises(ValueError):
            candidates(self.scene,self.arrays['rgb'],10.01,self.detections,self.association,detector={})
        metadata,_,_=reference(self.obs,1,'目标',task_id='x',revision=1,goal_id='g')
        self.assertIsNone(metadata['track'])

    def test_legacy_observed_track_support_time_is_explicitly_derived(self):
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config
        row=copy.deepcopy(self.row)
        del row['tracks'][0]['last_observed_stamp_s']
        scene=assemble_scene(self.scene.geometry,clock_domain='synthetic',people=row,settings=load_config()['scene'])
        value=candidates(scene,self.arrays['rgb'],9.98,self.detections,self.association,detector={})
        track=value['candidates'][0]['track']
        self.assertEqual(track['support_stamp_s'],row['stamp_s'])
        self.assertEqual(track['support_time_source'],'observed_state_at_people_stamp')
        row['tracks'][0]['state']='predicted'
        scene=assemble_scene(self.scene.geometry,clock_domain='synthetic',people=row,settings=load_config()['scene'])
        self.assertIsNone(candidates(scene,self.arrays['rgb'],9.98,self.detections,self.association,detector={})['candidates'][0]['track'])

    def test_missing_3d_identity_keeps_only_visual_candidates(self):
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config
        row=copy.deepcopy(self.row)
        del row['tracks'][0]['lifecycle_generation']
        scene=assemble_scene(self.scene.geometry,clock_domain='synthetic',people=row,settings=load_config()['scene'])
        result=candidates(scene,self.arrays['rgb'],9.98,self.detections,self.association,detector={})
        self.assertEqual(result['association_status'],'track_identity_unavailable')
        self.assertIsNone(result['candidates'][0]['track'])

    def test_runtime_preparer_restores_scene_health_before_association(self):
        from decision.vlm.image_inputs import FollowImagePreparer
        preparer=FollowImagePreparer.__new__(FollowImagePreparer)
        preparer.stream=None
        preparer.calls=0
        preparer.calibration=dict(source_frame=self.row['detection_frame_id'],target_frame='camera',sha256='cal')
        preparer.transform=__import__('numpy').eye(4)
        class Detector:
            metadata=dict(backend='test',sha256='model')
            def detect(inner, pixels):return self.detections,0.
        class Matcher:
            def step(inner, people, *args, **kwargs):
                self.assertEqual(people[0]['kind'],'track')
                return self.association
        preparer.detector=Detector()
        preparer.matcher=Matcher()
        camera=dict(stamp_ns=9_950_000_000,frame_id='camera',width=64,height=48)
        with patch('torch.cuda.stream',return_value=nullcontext()):
            result=preparer.prepare_scene(self.scene,self.arrays['rgb'],9.98,'camera',camera,9_980_000_000)
            camera['stamp_ns']=10_000_000_000
            no_match=preparer.prepare_scene(self.scene,self.arrays['rgb'],9.98,'camera',camera,9_980_000_000)
            camera['stamp_ns']=9_950_000_000
            class BadMatcher:
                def step(inner,*args,**kwargs):raise ValueError('invalid camera')
            preparer.matcher=BadMatcher()
            invalid=preparer.prepare_scene(self.scene,self.arrays['rgb'],9.98,'camera',camera,9_980_000_000)
        self.assertEqual(result['candidates'][0]['track']['identity'],['tracker_session',7,'a'])
        self.assertIsNone(no_match['candidates'][0]['track'])
        self.assertIsNone(no_match['detector']['camera_info'])
        self.assertEqual(invalid['association_status'],'association_error_ValueError')
        self.assertIsNone(invalid['candidates'][0]['track'])

    def test_worker_failure_preserves_regular_observation(self):
        from decision.agent.adapters.runtime import ObservationWorker
        published=[]
        def fail():raise RuntimeError('detector unavailable')
        worker=ObservationWorker(lambda record,image:published.append(record),follow_candidates=True,follow_factory=fail)
        self.assertTrue(worker.submit(self.scene,self.arrays['rgb'],9.98,'camera'))
        result=worker.close()
        self.assertEqual(result['follow_candidates_error'],'RuntimeError')
        self.assertEqual(published[0]['follow_candidates_error'],'RuntimeError')
        self.assertNotIn('follow_candidates',published[0])

    def test_worker_preserves_exact_rgb_nanoseconds_with_candidates(self):
        from decision.agent.adapters.runtime import ObservationWorker
        published=[]
        class Preparer:
            def prepare_scene(inner,scene,image,stamp,frame,camera,stamp_ns):
                self.assertEqual(stamp_ns,9_980_000_000)
                return candidates(scene,image,stamp,self.detections,self.association,detector={},rgb_stamp_ns=stamp_ns)
        worker=ObservationWorker(lambda record,image:published.append((record,image)),
                                 follow_candidates=True,follow_factory=Preparer)
        self.assertTrue(worker.submit(self.scene,self.arrays['rgb'],9.98,'camera',None,9_980_000_000))
        result=worker.close()
        self.assertIsNone(result['follow_candidates_error'])
        record,image=published[0]
        Observation(record,image).verify()
        self.assertEqual(record['rgb']['stamp_ns'],9_980_000_000)
        self.assertEqual(record['follow_candidates']['rgb_stamp_ns'],9_980_000_000)

    def test_worker_recovers_after_one_invalid_candidate_frame(self):
        import threading
        from decision.agent.adapters.runtime import ObservationWorker
        published = []
        ready = threading.Event()
        class Preparer:
            def __init__(inner):
                inner.calls = 0
            def prepare_scene(inner, scene, image, stamp, frame, camera, stamp_ns):
                inner.calls += 1
                detections = self.detections * (33 if inner.calls == 1 else 1)
                return candidates(scene, image, stamp, detections, self.association, detector={})
        preparer = Preparer()
        def publish(record, image):
            published.append(record)
            ready.set()
        worker = ObservationWorker(publish, follow_candidates=True, follow_factory=lambda: preparer)
        try:
            self.assertTrue(worker.submit(self.scene, self.arrays['rgb'], 9.98, 'camera'))
            self.assertTrue(ready.wait(3))
            ready.clear()
            _, arrays, _, scene = observation(11.)
            self.assertTrue(worker.submit(scene, arrays['rgb'], 10.98, 'camera'))
            self.assertTrue(ready.wait(3))
        finally:
            result = worker.close()
        self.assertEqual(published[0]['follow_candidates_error'], 'ValueError')
        self.assertNotIn('follow_candidates', published[0])
        self.assertNotIn('follow_candidates_error', published[1])
        self.assertEqual(len(published[1]['follow_candidates']['candidates']), 2)
        self.assertEqual(preparer.calls, 2)
        self.assertIsNone(result['follow_candidates_error'])

    def test_agent_model_input_receives_numbered_image_and_candidate_binding(self):
        from decision.agent.core.model import Model
        model=Model.__new__(Model)
        captured={}
        def request(system,messages,**kwargs):
            captured['messages']=messages
            return 'fixture'
        model.request=request
        payload=self.session.prompt()
        self.assertEqual(model.decide(payload,observation=self.session.observation),'fixture')
        content=captured['messages'][-1]['content']
        self.assertTrue(any('黄色数字' in part.get('text','') for part in content))
        self.assertTrue(any(part.get('type')=='image_url' for part in content))
        visible=self.session.observation.prompt()['follow_candidates']
        self.assertEqual(visible['binding'],self.value['binding'])
        self.assertNotIn('detector',visible)


if __name__ == '__main__':
    unittest.main()
