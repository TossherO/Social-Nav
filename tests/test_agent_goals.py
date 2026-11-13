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

    def test_semantic_reference_allows_a_free_pose_but_needs_post_arrival_evidence(self):
        goal=dict(goal_id='near',kind='reference',target_id='front_window',description='Stop by the window',
                  semantic_conditions=['The chosen stopping place is by the window'])
        pose=[5.,4.,0.]
        result=self.select(goal,pose)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.nav.target_pose,pose)
        self.arrived()
        self.assertEqual(self.session.call('check_goal',goal_id='near')['status'],'needs_semantic_evidence')
        self.session.tick()
        self.session.observation=self.rgb()
        checked=self.session.call('check_goal',goal_id='near',observation_id=self.session.observation.record['observation_id'],
            semantic_result='satisfied',evidence='Synthetic window visible',_context=self.session.context())
        self.assertEqual(checked['status'],'completed')
        self.assertFalse(checked['real_robot_success'])

    def test_reference_without_semantic_completion_condition_is_rejected(self):
        result=self.session.call('record_task',goals=[dict(goal_id='near',kind='reference',target_id='front_window',description='By window')])
        self.assertEqual(result['status'],'unavailable')
        self.assertIn('observable original task condition',result['reason'])

    def test_reference_does_not_complete_when_actual_pose_misses_selected_point(self):
        goal=dict(goal_id='near',kind='reference',target_id='front_window',description='Stop by window',
                  semantic_conditions=['Near the window'])
        self.assertEqual(self.select(goal,[5.,4.,0.])['status'],'accepted')
        self.arrived();self.nav.state['pose']=[6.,4.,0.]
        result=self.session.call('check_goal',goal_id='near')
        self.assertEqual(result['status'],'not_met')
        self.assertFalse(result['checks']['selected_position_met'])

    def test_explicit_point_id_and_yaw_resolve_without_borrowing_annotation_yaw(self):
        self.session.call('record_task',goals=[point_goal('elevator')])
        view_id=query_map(self.session)
        result=self.session.call('navigate',goal_id='elevator',point_id='elevator',yaw=.71,
            view_id=view_id,orientation_source='autonomous')
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.nav.target_pose,[5,4,.71])
        self.assertEqual(self.session.task_goals.goals[0]['selected']['selected_point_id'],'elevator')

    def test_navigate_rejects_ambiguous_pose_forms(self):
        from decision.agent.tools.framework import validate_call
        import jsonschema
        common=dict(goal_id='g',view_id='v',orientation_source='autonomous')
        for extra in ({},{'point_id':'elevator'},{'pose':[5,4,0],'yaw':0},
                      {'pose':[5,4,0],'point_id':'elevator','yaw':0}):
            with self.assertRaises(jsonschema.ValidationError):
                validate_call(dict(tool='navigate',arguments=dict(common,**extra)))

    def test_required_yaw_is_checked_and_never_repaired(self):
        goal = dict(point_goal('elevator'),required_yaw=.4)
        result = self.select(goal,[5,4,1.4])
        self.assertEqual(result['status'],'refused')
        self.assertFalse(result['checks']['required_yaw_met'])
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertIsNone(self.session.task_goals.goals[0]['selected'])

    def test_facing_named_point_allows_semantic_heading_and_needs_current_judgement(self):
        result = self.select(dict(point_goal('elevator'),facing='front_window'),[5,4,1.57])
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(self.nav.target_pose,[5,4,1.57])
        self.arrived()
        self.assertEqual(self.session.call('check_goal',goal_id='elevator')['reason'],
                         'read_a_fresh_post_arrival_map_then_judge_facing')
        original=self.nav.spatial_inputs
        self.nav.spatial_inputs=lambda:dict(original(),pose=self.nav.state['pose'],stamp_s=self.nav.state['stamp_s'])
        query_map(self.session)
        result=self.session.call('check_goal',goal_id='elevator',facing_result='satisfied',
            facing_evidence='Synthetic map indicates the relevant corridor direction',_context=self.session.context())
        self.assertEqual(result['status'],'completed')
        self.assertIn('not_independent_ground_truth',result['semantic']['facing']['source'])

    def test_facing_region_can_follow_corridor_instead_of_region_centroid(self):
        goal=dict(point_goal('elevator'),facing='lobby_area')
        self.assertEqual(self.select(goal,[5,4,1.57])['status'],'accepted')
        self.arrived()
        original=self.nav.spatial_inputs
        self.nav.spatial_inputs=lambda:dict(original(),pose=self.nav.state['pose'],stamp_s=self.nav.state['stamp_s'])
        query_map(self.session)
        args=dict(goal_id='elevator',facing_result='unknown',facing_evidence='No corridor evidence')
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'unknown')
        args.update(facing_result='satisfied',facing_evidence='Synthetic connected corridor lies to the left')
        self.nav.state['pose']=[5,4,0.]
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['reason'],
                         'actual_heading_differs_from_selected_heading')
        self.nav.state['pose']=[5,4,1.57]
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'completed')

    def test_facing_judgement_requires_a_post_arrival_view_used_by_model(self):
        self.assertEqual(self.select(dict(point_goal('elevator'),facing='lobby_area'),[5,4,1.57])['status'],'accepted')
        self.arrived()
        original=self.nav.spatial_inputs
        self.nav.spatial_inputs=lambda:dict(original(),pose=self.nav.state['pose'],stamp_s=self.nav.state['stamp_s']-.01)
        query_map(self.session)
        args=dict(goal_id='elevator',facing_result='satisfied',facing_evidence='Synthetic corridor')
        result=self.session.call('check_goal',_context=self.session.context(),**args)
        self.assertEqual(result['reason'],'read_a_fresh_post_arrival_map_then_judge_facing')
        self.nav.spatial_inputs=lambda:dict(original(),pose=self.nav.state['pose'],stamp_s=self.nav.state['stamp_s'])
        query_map(self.session)
        self.assertEqual(self.session.call('check_goal',**args)['status'],'needs_semantic_evidence')
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'completed')

    def test_facing_rejects_semantic_view_with_different_pose(self):
        self.assertEqual(self.select(dict(point_goal('elevator'),facing='lobby_area'),[5,4,1.57])['status'],'accepted')
        self.arrived()
        original=self.nav.spatial_inputs
        self.nav.spatial_inputs=lambda:dict(original(),pose=[5,4,0.],stamp_s=self.nav.state['stamp_s'])
        query_map(self.session)
        result=self.session.call('check_goal',goal_id='elevator',facing_result='satisfied',
            facing_evidence='Synthetic map',_context=self.session.context())
        self.assertEqual(result['reason'],'map_view_pose_differs_from_current_pose')

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
        self.session.tick()
        self.session.observation = self.rgb()
        args=dict(goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
                  semantic_result='satisfied',evidence='Synthetic fixture judgement only')
        self.assertEqual(self.session.call('check_goal',**args)['status'],'needs_semantic_evidence')
        result = self.session.call('check_goal',_context=self.session.context(),**args)
        self.assertEqual(result['status'],'completed')
        self.assertIn('not_independent_ground_truth',result['semantic']['source'])

    def test_visual_preference_without_post_arrival_rgb_is_explicitly_unassessed(self):
        goal=dict(goal_id='near',kind='reference',target_id='front_window',description='Near window; prefer a view',
                  max_distance_m=5.,visual_preferences=['Prefer to see all targets'])
        self.assertEqual(self.select(goal,[5.,4.,0.])['status'],'accepted')
        self.arrived()
        result=self.session.call('check_goal',goal_id='near')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['semantic']['visual_preferences']['status'],'not_assessed')
        self.assertFalse(result['real_robot_success'])

    def test_visual_preference_judgement_needs_current_post_arrival_rgb(self):
        self.select(dict(point_goal('elevator'),visual_preferences=['Prefer to see all targets']),[5,4,0])
        self.arrived()
        self.session.observation=self.rgb(self.nav.state['stamp_s']-.01)
        args=dict(goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
                  preference_result='partly_satisfied',preference_evidence='Two targets visible in fixture')
        result=self.session.call('check_goal',_context=self.session.context(),**args)
        self.assertEqual(result['status'],'needs_preference_evidence')
        self.session.tick();self.session.observation=self.rgb()
        args['observation_id']=self.session.observation.record['observation_id']
        result=self.session.call('check_goal',_context=self.session.context(),**args)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['semantic']['visual_preferences']['status'],'partly_satisfied')
        self.assertIn('not_independent_ground_truth',result['semantic']['visual_preferences']['source'])

    def test_reference_can_register_visual_preference_without_invented_radius(self):
        goal=dict(goal_id='view',kind='reference',target_id='elevator',description='尽量看见电梯口',
                  visual_preferences=['尽量看见电梯口'])
        result=self.session.call('record_task',goals=[goal])
        self.assertEqual(result['status'],'recorded')
        registered=self.session.task_goals.current('view')['requirement']
        self.assertNotIn('max_distance_m',registered)
        self.assertNotIn('semantic_conditions',registered)

    def test_soft_preference_does_not_weaken_hard_visual_conditions(self):
        self.select(dict(point_goal('elevator'),semantic_conditions=['Entrance must be visible'],
                         visual_preferences=['Prefer to see all targets']),[5,4,0])
        self.arrived()
        result=self.session.call('check_goal',goal_id='elevator')
        self.assertEqual(result['status'],'needs_semantic_evidence')
        self.assertFalse(self.session.task_goals.finished())

    def test_scene_after_arrival_cannot_make_pre_arrival_rgb_current(self):
        self.select(dict(point_goal('elevator'),semantic_conditions=['Entrance is visible']),[5,4,0])
        self.arrived()
        self.nav.state['stamp_s'] += .02
        self.session.observation = self.rgb()
        result=self.session.call('check_goal',goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
            semantic_result='satisfied',evidence='Fixture',_context=self.session.context())
        self.assertEqual(result['status'],'needs_semantic_evidence')

    def test_old_image_and_unknown_semantics_cannot_complete_goal(self):
        self.select(dict(point_goal('elevator'),semantic_conditions=['View is clear']),[5,4,0])
        self.arrived()
        self.session.observation = self.rgb(self.nav.state['stamp_s']-.01)
        args=dict(goal_id='elevator',observation_id=self.session.observation.record['observation_id'],
                  semantic_result='satisfied',evidence='Old image')
        self.assertEqual(self.session.call('check_goal',_context=self.session.context(),**args)['status'],'needs_semantic_evidence')
        self.session.tick()
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
        self.session.call('record_task',goals=[point_goal('elevator')])
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
        base.pop('regions')
        asset.write_text(json.dumps(base))
        overlay = dict(self.catalog.value,asset=str(asset))
        path = self.root/'catalog.json';path.write_text(json.dumps(overlay))
        loaded = Catalog.load(path)
        self.assertIn('lobby_area',loaded.regions)
        self.assertIn('front_window',loaded.places)
        self.assertNotIn('regions',json.loads(asset.read_text()))


if __name__=='__main__':
    unittest.main()
