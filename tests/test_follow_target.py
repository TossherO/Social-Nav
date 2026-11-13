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

    def select(self, match, candidate_id):
        return self.session.call('select_follow_target',_context=self.session.context(),goal_id='follow',
            observation_id=self.obs.record['observation_id'],candidate_binding=self.value['binding'],
            target_description='穿红衣的人',match=match,candidate_id=candidate_id)

    def test_match_binds_original_image_box_and_track_without_motion(self):
        result = self.select('unique',0)
        self.assertEqual(result['status'],'selected')
        self.assertFalse(result['executable'])
        self.assertEqual(result['reference']['track']['identity'],['tracker_session',7,'a'])
        saved,image,crop=self.store.read_follow_reference(self.session.state['task_id'],1,'follow')
        self.assertEqual(image,self.obs.image)
        self.assertEqual(hashlib.sha256(crop).hexdigest(),saved['crop_sha256'])
        self.assertEqual(self.select('unique',1)['reason'],'follow_reference_already_fixed')
        self.assertEqual(self.session.task_goals.goals[0]['status'],'pending')
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertNotEqual(annotated(self.obs),self.obs.image)

    def test_ambiguous_then_visual_only_remains_nonexecutable(self):
        self.assertEqual(self.select('ambiguous',-1)['status'],'ambiguous')
        self.assertIsNone(self.store.read_follow_reference(self.session.state['task_id'],1,'follow'))
        result=self.select('unique',1)
        self.assertEqual(result['status'],'visual_only')
        self.assertIsNone(result['reference']['track'])
        self.assertFalse(result['executable'])

    def test_observation_and_candidate_changes_cannot_be_substituted(self):
        modified=copy.deepcopy(self.value)
        modified['candidates'][0]['box'][0]+=1
        with self.assertRaises(ValueError):verify_candidates(modified,self.obs)
        context=self.session.context()
        self.nav.state['stamp_s']=41.
        result=self.session.call('select_follow_target',_context=context,goal_id='follow',
            observation_id=self.obs.record['observation_id'],candidate_binding=self.value['binding'],
            target_description='穿红衣的人',match='unique',candidate_id=0)
        self.assertEqual(result['reason'],'selection_requires_current_consumed_observation')

    def test_delayed_result_keeps_only_historical_reference(self):
        context=self.session.context()
        self.nav.state['stamp_s']=14.
        result=self.session.call('select_follow_target',_context=context,goal_id='follow',
            observation_id=self.obs.record['observation_id'],candidate_binding=self.value['binding'],
            target_description='穿红衣的人',match='unique',candidate_id=0)
        self.assertEqual(result['status'],'selected_historical')
        self.assertEqual(result['source_age_s'],4.)
        self.assertFalse(result['current_track_valid'])
        self.assertFalse(result['executable'])

    def test_follow_stage_requires_running_then_finish_and_observed_stop(self):
        self.nav.scope='live'
        self.nav.state.update(scope='live',chassis_state='armed',chassis_status_fresh=True,
                              localization_status='valid',input_stale=False,stop_observation=False)
        self.nav.check_route=lambda _:dict(proposal_allowed=True,status='available')
        def start(ref,handle):
            self.nav.state.update(status='running',handle=copy.deepcopy(handle),stopped=False,
                                  follow_status=dict(status='running',handle=copy.deepcopy(handle),
                                                     target_identity=ref['track']['identity']))
            return dict(accepted=True,handle=copy.deepcopy(handle),execution_feedback=False)
        def finish(handle):
            self.nav.state.update(status='stopping',follow_status=dict(status='stopped',handle=copy.deepcopy(handle)))
        self.nav.start_follow=start;self.nav.finish_follow=finish
        self.assertEqual(self.select('unique',0)['status'],'selected')
        result=self.session.call('start_follow',goal_id='follow')
        self.assertEqual(result['status'],'accepted')
        self.session.tick()
        self.assertTrue(self.session.skill.record['follow_ready'])
        self.assertEqual(self.session.call('check_goal',goal_id='follow')['status'],'unproven')
        self.assertEqual(self.session.call('observe_scene')['status'],'available')
        reviews=[]
        self.nav.review_follow=lambda handle,value:(reviews.append((handle,value)) or dict(status='submitted'))
        reviewed=self.session.call('review_follow_target',_context=self.session.context(),goal_id='follow',
            observation_id=self.obs.record['observation_id'],state='visible',candidate_id=0)
        self.assertEqual(reviewed['status'],'submitted')
        self.assertEqual(reviews[-1][1]['identity'],['tracker_session',7,'a'])
        from decision.agent.core.model import Model
        client=object.__new__(Model)
        client.request=lambda system,messages,**kwargs:messages
        payload=self.session.prompt()
        messages=client.decide(payload,observation=self.session.observation)
        self.assertTrue(any('历史目标参考裁剪' in part.get('text','') for part in messages[-1]['content']))
        finish_result=self.session.call('finish_follow',_context=self.session.context(),goal_id='follow',
            observation_id=self.obs.record['observation_id'],evidence='已在电梯入口旁')
        self.assertEqual(finish_result['status'],'stopping')
        self.assertEqual(self.session.call('check_goal',goal_id='follow')['status'],'unproven')
        self.nav.state.update(status='stopped',stopped=True,stop_observation=True)
        self.session.tick()
        self.assertEqual(self.session.skill.record['status'],'held')
        self.assertEqual(self.session.call('check_goal',goal_id='follow')['status'],'needs_semantic_evidence')
        _,arrays,_,after=observation(10.4)
        self.nav.state['stamp_s']=10.4
        self.nav.observe=lambda:Observation.from_scene(after,rgb=dict(pixels=arrays['rgb'],stamp_s=10.38,frame_id='camera'))
        self.assertEqual(self.session.call('observe_scene')['status'],'available')
        self.nav.state['pose']=copy.deepcopy(self.session.observation.record['base_pose'])
        result=self.session.call('check_goal',_context=self.session.context(),goal_id='follow',
            observation_id=self.session.observation.record['observation_id'],semantic_result='satisfied',evidence='停在电梯入口旁')
        self.assertEqual(result['status'],'completed')

    def test_visual_review_stops_when_selected_candidate_has_no_matching_3d_track(self):
        self.select('unique',0)
        handle=dict(task_id=self.session.state['task_id'],goal_id='follow',goal_revision=1,call_id='call')
        self.session.skill.record.update(stage_type='follow',goal_id='follow',handle=handle,status='running')
        self.nav.state.update(handle=copy.deepcopy(handle),status='running',
            follow_status=dict(status='running',target_identity=['tracker_session',7,'a'],handle=copy.deepcopy(handle)))
        result=self.session.call('review_follow_target',_context=self.session.context(),goal_id='follow',
            observation_id=self.obs.record['observation_id'],state='visible',candidate_id=1)
        self.assertEqual(result['status'],'stopping')
        self.assertEqual(self.session.skill.record['reason'],'follow_candidate_not_current_target_track')
        self.assertEqual(self.nav.state['status'],'stopping')
        self.assertEqual(self.session.skill.record['completed'],[])

    def test_delayed_agent_start_uses_runtime_revalidation_not_old_image_as_current(self):
        from decision.agent.core.loop import AgentLoop
        from decision.agent.tools.framework import Reply
        import threading
        self.select('unique',0)
        self.nav.check_route=lambda _:dict(proposal_allowed=True,status='available')
        calls=[]
        def start(ref,handle):
            calls.append((copy.deepcopy(ref),copy.deepcopy(handle)))
            self.nav.state.update(handle=copy.deepcopy(handle),status='running',
                                  follow_status=dict(status='starting',handle=copy.deepcopy(handle)))
            return dict(accepted=True,handle=copy.deepcopy(handle),execution_feedback=False)
        self.nav.start_follow=start
        gate=threading.Event()
        def work(*_):
            gate.wait(2)
            return Reply.fixture(dict(tool='start_follow',arguments=dict(goal_id='follow')))
        actor=AgentLoop(self.session,work)
        try:
            actor.tick();self.nav.t=4.;gate.set();actor.model.thread.join(1);actor.tick()
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][0]['stamp_s'],10.)
            self.assertEqual(self.session.state['status'],'waiting_navigation')
        finally:
            gate.set()
            if actor.model.thread is not None:actor.model.thread.join(1)
            actor.close()

    def test_delayed_agent_reply_can_save_historical_reference_without_motion(self):
        from decision.agent.core.loop import AgentLoop
        from decision.agent.tools.framework import Reply
        import threading
        gate=threading.Event()
        def work(*_):
            gate.wait(1)
            return Reply.fixture(dict(tool='select_follow_target',arguments=dict(goal_id='follow',
                observation_id=self.obs.record['observation_id'],candidate_binding=self.value['binding'],
                target_description='穿红衣的人',match='unique',candidate_id=0)))
        actor=AgentLoop(self.session,work)
        try:
            actor.tick()
            self.nav.t=4.
            gate.set()
            actor.model.thread.join(1)
            actor.tick()
            self.assertEqual(self.session.state['history'][-1]['value']['result']['status'],'selected_historical')
            self.assertIsNone(self.session.skill.record['handle'])
        finally:
            gate.set()
            if actor.model.thread is not None:actor.model.thread.join(1)
            actor.close()

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
