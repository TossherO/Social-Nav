"""First follow selection, source binding and immutable visual references."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.observations import Observation
from decision.agent.tools.places import Catalog
from decision.vlm.t2.target import annotated, candidates, reference, verify_candidates
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

    def test_candidate_view_is_only_selected_for_follow_stage(self):
        from decision.agent.adapters.runtime import FollowAgentFrame
        from runtime.command import AgentState
        _, arrays, association, scene = observation(10.)
        pixels=arrays['rgb'].copy()
        association['image_input']['raw_rgb_sha256']=hashlib.sha256(pixels.tobytes()).hexdigest()
        association['image_input']['camera_frame']='camera'
        association.update(detector_identity='test',detections=[self.detections[0]])
        association['result']['tracks']['a/g7']['detection_index']=0
        frame=FollowAgentFrame(scene,pixels,association,None,None)
        state=AgentState()
        state.control=dict(clock_domain=scene.record['clock_domain'],stamp_s=10.)
        state.publish_observation({'kind':'ordinary'},b'ordinary')
        state.offer_follow_frame(frame,None)
        self.assertEqual(state.snapshot()['observation']['kind'],'ordinary')
        pending=state.snapshot(with_follow_candidates=True)
        self.assertIsNone(frame.encoded)
        self.assertEqual(pending['observation']['kind'],'ordinary')
        self.assertEqual(pending['follow_frame']['observation_id'],scene.record['observation_id'])
        full=state.snapshot(with_image=True,with_follow_candidates=True)
        self.assertEqual(full['observation']['observation_id'],pending['follow_frame']['observation_id'])
        self.assertEqual(full['observation']['scene_binding'],pending['follow_frame']['scene_binding'])
        self.assertEqual(state.snapshot(with_follow_candidates=True)['observation'],full['observation'])
        candidate=state.follow_candidate(full['observation']['observation_id'],full['observation']['follow_candidates']['binding'])
        self.assertEqual(candidate,(full['observation'],full['image_base64']))

    def test_raw_follow_observation_does_not_claim_a_visual_identity(self):
        from decision.agent.adapters.runtime import follow_observation
        obs=follow_observation(self.scene,self.arrays['rgb'],None,None,None,
            image_metadata=dict(rgb_stamp_ns=9980000000,rgb_frame='camera'))
        obs.verify()
        self.assertEqual(obs.record['follow_candidates']['mode'],'raw_rgb')
        self.assertEqual(obs.record['follow_candidates']['candidates'],[])
        self.assertEqual(annotated(obs),obs.image)

    def test_agent_selects_reference_without_starting_motion(self):
        context=self.session.context()
        selected=self.session.call('select_follow_target',_context=context,goal_id='follow',
            observation_id=self.obs.record['observation_id'],candidate_id=0,target_description='穿红衣的人')
        self.assertEqual(selected['status'],'selected')
        self.assertIsNone(self.session.skill.record['handle'])
        record,image,crop=self.store.read_follow_reference(self.session.state['task_id'],1,'follow',selected['reference_id'])
        self.assertEqual(record['candidate_id'],0)
        self.assertEqual(hashlib.sha256(crop).hexdigest(),record['crop_sha256'])
        self.assertEqual(image,self.obs.image)
        self.assertEqual(self.session.call('start_local_task',goal_id='follow',task_type='follow',
            target_description='穿红衣的人')['reason'],'agent_confirmed_follow_reference_required')

    def test_altered_saved_reference_is_not_used_to_start_follow(self):
        selected=self.session.call('select_follow_target',_context=self.session.context(),goal_id='follow',
            observation_id=self.obs.record['observation_id'],candidate_id=0,target_description='穿红衣的人')
        with self.store.db:
            self.store.db.execute('UPDATE follow_reference_versions SET crop=? WHERE reference_id=?',
                                  (b'changed',selected['reference_id']))
        result=self.session.call('start_local_task',goal_id='follow',task_type='follow',
                                 reference_id=selected['reference_id'])
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['reason'],'Follow visual reference changed')
        self.assertIsNone(self.session.skill.record['handle'])

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


    def test_agent_model_input_receives_numbered_image_and_candidate_binding(self):
        from decision.agent.core.model import Model
        model=Model.__new__(Model)
        model.thinking_mode='off'
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
