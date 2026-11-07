"""Indoor goal contracts and spatial input; synthetic fixtures, no live model."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import httpx
import numpy as np
from PIL import Image
from decision.agent import CONFIG_DIR
from decision.agent.core.model import Model
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Budget, Store
from decision.agent.core.tasks import inside
from decision.agent.tools.observations import Observation
from decision.agent.tools.places import Catalog
from decision.agent.tools.framework import Reply
from perception.environment.context import fingerprint
from tests.test_agent import ScriptedFeedback, action, point_goal, navigate, query_map


class IndoorGoalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root/'session.sqlite')
        self.catalog = Catalog.load(CONFIG_DIR/'simulation.json')
        self.nav = ScriptedFeedback()
        self.session = AgentSession('Synthetic indoor instruction',self.catalog,self.nav,self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def select(self, goal, pose, source='semantic'):
        self.session.call('record_task',goals=[goal])
        view_id = query_map(self.session)
        return self.session.call('navigate',goal_id=goal['goal_id'],pose=pose,view_id=view_id,orientation_source=source)

    def rgb(self, stamp=None):
        stamp = self.nav.state['stamp_s'] if stamp is None else stamp
        pixels = np.zeros((12,16,3),np.uint8)
        buffer = io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG')
        image = buffer.getvalue()
        r = dict(observation_id='unit-rgb-'+str(stamp),stamp_s=stamp,clock_domain=self.nav.clock_domain,
            local_identity=[self.nav.state[k] for k in ('session_id','generation','frame_id')],
            scene_binding='unit-scene',localization_status='valid',base_pose=self.nav.state['pose'],map_base_pose=None,
            pose_reference='base_link',geometry_reference='synthetic_fixture',map_binding=None,
            geometry=dict(status='valid',stamp_s=stamp),people=dict(status='missing',count=None),
            scope='synthetic_visual_input_not_accuracy_evidence',
            rgb=dict(status='valid',stamp_s=stamp-.05,age_s=.05,encoded_sha256=hashlib.sha256(image).hexdigest(),
                     pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),shape=list(pixels.shape)),source={'kind':'unit_fixture'})
        r['input_fingerprint'] = fingerprint(r)
        return Observation(r,image).verify()

    def arrived(self):
        self.nav.arrive()
        self.session.tick()

    def test_point_heading_is_chosen_by_model_without_implicit_annotation_yaw(self):
        result = self.select(point_goal('elevator'),[5,4,.71],'autonomous')
        self.assertEqual(result['status'],'accepted')
        selected = self.session.task_goals.goals[0]['selected']
        self.assertEqual(selected['pose'],[5,4,.71])
        self.assertEqual(selected['orientation_source'],'autonomous')
        self.assertNotIn('required_yaw',selected['requirement'])

    def test_region_accepts_continuous_pose_without_candidates_or_snapping(self):
        pose = [6.837,4.173,.037]
        result = self.select(dict(goal_id='r',kind='region',target_id='lobby_area',description='Choose a place inside'),pose)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.nav.target_pose,pose)
        self.assertEqual(self.session.task_goals.goals[0]['selected']['pose'],pose)
        self.arrived()
        self.assertEqual(self.session.call('check_goal',goal_id='r')['status'],'completed')

    def test_required_yaw_is_checked_and_never_repaired(self):
        goal = dict(point_goal('elevator'),required_yaw=.4)
        result = self.select(goal,[5,4,1.4])
        self.assertEqual(result['status'],'refused')
        self.assertFalse(result['checks']['required_yaw_met'])
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertIsNone(self.session.task_goals.goals[0]['selected'])

    def test_facing_landmark_checks_model_yaw_without_generating_one(self):
        result = self.select(dict(point_goal('elevator'),facing='front_window'),[5,4,1.57])
        self.assertEqual(result['status'],'refused')
        view = self.session.spatial_view
        result = self.session.call('navigate',goal_id='elevator',pose=[5,4,.05],
            view_id=view.record['view_id'],orientation_source='semantic')
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.nav.target_pose,[5,4,.05])

    def test_region_geometry_supports_concavity_and_boundary(self):
        polygon = [[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]]
        self.assertTrue(inside([.5,1.5],polygon))
        self.assertTrue(inside([1,1],polygon))
        self.assertFalse(inside([1.5,1.5],polygon))

    def test_task_order_and_all_goal_completion_are_required(self):
        self.session.call('record_task',goals=[point_goal('elevator'),point_goal('lobby_entrance')])
        result = navigate(self.session,'lobby_entrance')
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(self.session.skill.record['attempts'],[])
        navigate(self.session,'elevator');self.arrived()
        self.assertEqual(self.session.call('check_goal',goal_id='elevator')['status'],'completed')
        result = self.session.call('finish_task',status='succeeded',evidence='Only first goal done')
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['uncompleted_goals'],['lobby_entrance'])
        self.assertEqual(self.session.call('record_task',goals=[point_goal('elevator')])['status'],'unavailable')

    def test_semantic_completion_requires_consumed_post_arrival_rgb(self):
        self.select(dict(point_goal('elevator'),semantic_conditions=['Entrance is visible']),[5,4,0])
        self.arrived()
        self.session.observation = self.rgb()
        args=dict(goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
                  semantic_result='satisfied',evidence='Synthetic fixture judgement only')
        self.assertEqual(self.session.call('check_goal',**args)['status'],'needs_semantic_evidence')
        result = self.session.call('check_goal',_context=self.session.context(),**args)
        self.assertEqual(result['status'],'completed')
        self.assertIn('not_independent_ground_truth',result['semantic']['source'])

    def test_old_image_and_unknown_semantics_cannot_complete_goal(self):
        self.select(dict(point_goal('elevator'),semantic_conditions=['View is clear']),[5,4,0])
        self.arrived()
        self.session.observation = self.rgb(self.nav.state['stamp_s']-.01)
        args=dict(goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
                  semantic_result='satisfied',evidence='Old image')
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'needs_semantic_evidence')
        self.session.observation = self.rgb()
        args.update(observation_id=self.session.observation.record['observation_id'],semantic_result='unknown',evidence='Insufficient evidence')
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'unknown')
        self.assertFalse(self.session.task_goals.finished())

    def test_spatial_view_tampering_or_age_does_not_move_target(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        view_id = query_map(self.session)
        self.session.spatial_view.record['local_bounds'][0] += 1
        result = self.session.call('navigate',goal_id='elevator',pose=[5,4,0],view_id=view_id,orientation_source='autonomous')
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(self.session.skill.record['handle'])
        query_map(self.session)
        self.nav.state['stamp_s'] += 4
        result = self.session.call('navigate',goal_id='elevator',pose=[5,4,0],
            view_id=self.session.spatial_view.record['view_id'],orientation_source='autonomous')
        self.assertEqual(result['status'],'refused')

    def test_future_map_conditions_do_not_expose_complete_annotations_or_execute(self):
        for mode in ('simple','none'):
            catalog = Catalog.load(CONFIG_DIR/'simulation.json',condition=mode)
            self.assertEqual(catalog.annotations()['points'],[])
            self.assertEqual(catalog.lookup(''),[])
            session = AgentSession('Future mode',catalog,self.nav,self.store)
            result = session.call('record_task',goals=[point_goal('elevator')])
            self.assertEqual(result['status'],'unavailable')
            self.assertIn('not_implemented',result['reason'])

    def test_map_view_never_projects_a_pose_from_a_different_map(self):
        from types import SimpleNamespace
        from decision.agent.adapters.shadow import SavedShadowNavigation, RuntimeShadowNavigation
        catalog = Catalog(dict(self.catalog.value,map_snapshot_id='expected_map'))
        record = dict(localization_status='valid',map_base_pose=[5,4,0],stamp_s=10.,
                      local_identity=['local',1,'odom'],clock_domain='bag',
                      map_binding=dict(valid=True,map_id='other_map',map_frame='sim_odom'))
        saved = object.__new__(SavedShadowNavigation)
        saved.catalog,saved.current = catalog,SimpleNamespace(record=record)
        with self.assertRaisesRegex(ValueError,'not localized'):
            saved.spatial_inputs()
        runtime = object.__new__(RuntimeShadowNavigation)
        runtime.catalog,runtime.clock_domain = catalog,'bag'
        runtime._asset = SimpleNamespace(snapshot=None)
        runtime.value = dict(observation=record)
        runtime.snapshot = lambda:dict(stamp_s=10.,input_stale=False,localization_status='valid',
            session_id='local',generation=1,frame_id='odom',map_binding=record['map_binding'])
        inputs = runtime.spatial_inputs()
        self.assertIsNone(inputs['pose'])
        self.assertIsNone(inputs['stamp_s'])

    def test_three_source_bound_images_reach_same_model_request(self):
        query_map(self.session)
        view = self.session.spatial_view
        rows = self.store.db.execute('SELECT role,sha256,png FROM map_images WHERE view_id=? ORDER BY role',(view.record['view_id'],)).fetchall()
        self.assertEqual(len(rows),2)
        self.assertTrue(all(hashlib.sha256(bytes(row[2])).hexdigest()==row[1] for row in rows))
        observation = self.rgb()
        config = self.root/'private.json'
        config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='unit-key')))
        budget = Budget(self.root/'budget.sqlite')
        seen=[]
        def handle(request):
            body=json.loads(request.content);seen.append(body)
            native=Reply.fixture(action('navigate',goal_id='a',pose=[5,4,.2],view_id=view.record['view_id'],orientation_source='semantic')).message['tool_calls'][0]
            parts=[dict(model='qwen3.8-flash',choices=[dict(delta=dict(tool_calls=[dict(native,index=0)]),finish_reason='tool_calls')]),
                   dict(model='qwen3.8-flash',choices=[],usage=dict(prompt_tokens=20,completion_tokens=10))]
            return httpx.Response(200,text=''.join('data: '+json.dumps(p)+'\n\n' for p in parts)+'data: [DONE]\n\n')
        try:
            result=Model(config,budget,self.store,transport=httpx.MockTransport(handle)).decide(
                self.session.prompt(),observation=observation,spatial_view=view)
            self.assertEqual(result.action['arguments']['pose'],[5,4,.2])
            images=[v for v in seen[0]['messages'][-1]['content'] if v['type']=='image_url']
            self.assertEqual(len(images),3)
            self.assertTrue(images[1]['image_url']['url'].startswith('data:image/png;'))
        finally:
            budget.close()

    def test_annotation_overlay_preserves_map_asset_and_adds_regions(self):
        asset = self.root/'asset.json'
        base = copy.deepcopy(self.catalog.value)
        base.pop('regions');base.pop('landmarks')
        asset.write_text(json.dumps(base))
        overlay = dict(self.catalog.value,asset=str(asset))
        path = self.root/'catalog.json';path.write_text(json.dumps(overlay))
        loaded = Catalog.load(path)
        self.assertIn('lobby_area',loaded.regions)
        self.assertIn('front_window',loaded.landmarks)
        self.assertNotIn('regions',json.loads(asset.read_text()))


if __name__=='__main__':
    unittest.main()
