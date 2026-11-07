"""Grounded context, nonduplicated annotations and persistent task information."""
import copy
import io
import json
import unittest
import numpy as np
from PIL import Image
from decision.agent import CONFIG_DIR
from decision.agent.core.context import project_dialogue, robot_context, view_context
from decision.agent.core.session import AgentSession
from decision.agent.tools.framework import native_tools
from decision.agent.tools.places import Catalog
from decision.agent.tools.spatial import render
from navigation.planning.footprint import VehicleGeometry
from tests.test_agent import reply
from tests import test_agent_visual_context as fixtures


class AgentContextTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.VisualContextTests()
        self.fixture.setUp()
        self.session = self.fixture.session

    def tearDown(self):
        self.fixture.tearDown()

    def test_fixed_directory_has_one_definition_and_raw_records_keep_original_bindings(self):
        self.session.prompt()
        response = reply('query_map')
        result = self.session.call('query_map')
        self.session.conversation.commit(response,result)
        original = copy.deepcopy(self.session.spatial_view.record)
        body = self.fixture.request()
        reference = json.loads(body['messages'][1]['content'][0]['text'])['reference']
        initial = next(json.loads(m['content']) for m in body['messages'] if isinstance(m.get('content'),str)
            and m['content'].startswith('{') and json.loads(m['content']).get('source')=='initial_map_catalog')['catalog']
        self.assertNotIn('annotations',initial)
        self.assertEqual(initial['access'],self.session.catalog.prompt()['access'])
        self.assertEqual(initial['annotation_reference']['reference_map_id'],reference['reference_map_id'])
        origin = initial['annotation_reference']['task_points'][0]
        self.assertEqual(origin['place_id'],'mission_start');self.assertEqual(origin['display_id'],'P5')
        historical = next(json.loads(m['content'])['view'] for m in body['messages'] if m['role']=='tool')
        self.assertNotIn('annotations',historical)
        self.assertEqual(historical['view_id'],original['view_id'])
        self.assertEqual(historical['record_representation'],'projection_of_original_view_id')
        self.assertEqual(self.session.spatial_view.record,original)
        recorded = json.loads(self.fixture.store.db.execute("SELECT value FROM events WHERE kind='tool_result' ORDER BY seq DESC LIMIT 1").fetchone()[0])
        self.assertEqual(recorded['value']['view']['annotations'],original['annotations'])

    def test_incompatible_or_absent_reference_keeps_full_annotations(self):
        self.session.prompt()
        original = copy.deepcopy(self.session.spatial_view.record)
        changed = copy.deepcopy(original);changed['annotations']['regions'][0]['name']='different annotation'
        self.assertEqual(view_context(changed,self.session.reference_map),changed)
        self.assertEqual(view_context(original,None),original)
        messages = self.session.conversation.messages()
        self.assertEqual(project_dialogue(messages,None),messages)
        self.assertIn('annotations',messages[1]['content'])

    def test_lookup_covers_points_regions_and_landmarks_without_leaking_future_modes(self):
        all_items = self.session.call('lookup_places',name='')
        self.assertTrue(all_items['places']);self.assertTrue(all_items['regions']);self.assertTrue(all_items['landmarks'])
        landmark = all_items['landmarks'][0]
        self.assertEqual(self.session.call('lookup_places',name=landmark['landmark_id'])['landmarks'],[landmark])
        future = AgentSession('future',Catalog.load(CONFIG_DIR/'simulation.json',condition='simple'),
                              self.fixture.nav,self.fixture.store)
        result = future.call('lookup_places',name='')
        self.assertEqual((result['places'],result['regions'],result['landmarks']),([],[],[]))

    def test_requested_crop_survives_different_control_time_and_refreshes_same_intent(self):
        inputs = self.fixture.nav.spatial_inputs()
        self.fixture.nav.spatial_inputs = lambda:copy.copy(inputs)
        self.fixture.nav.state['stamp_s']=10.4
        bounds=[3.,2.,9.,8.]
        self.session.call('query_map',bounds=bounds)
        original = self.session.spatial_view.record['view_id']
        self.session.prompt()
        self.assertEqual(self.session.spatial_view.record['local_bounds'],bounds)
        self.assertEqual(self.session.spatial_view.record['view_id'],original)
        inputs['stamp_s']=10.2;inputs['pose']=[4.1,4.,0.]
        self.session.prompt()
        self.assertEqual(self.session.spatial_view.record['local_bounds'],bounds)
        self.assertNotEqual(self.session.spatial_view.record['view_id'],original)
        self.assertEqual(self.session.spatial_view.record['source_stamp_s'],10.2)

    def test_accepted_clarifications_survive_history_trimming_and_overflow_is_explicit(self):
        self.session.conversation.max_exchanges=1
        for index in range(10):
            self.session.call('ask_user',question='入口条件'+str(index))
            self.session.event('answer','约束'+str(index))
        state=self.session.prompt()['current_state']
        self.assertEqual(len(state['clarifications']),10)
        self.assertEqual(state['clarifications'][0]['answer'],'约束0')
        self.assertFalse(state['requirements_registered'])
        self.session.call('ask_user',question='更多条件')
        with self.assertRaisesRegex(ValueError,'16384'):
            self.session.event('answer','超长'*9000)
        self.assertEqual(len(self.session.state['clarifications']),10)
        self.assertEqual(self.session.state['status'],'needs_clarification')

    def test_robot_parameters_come_from_adapter_and_missing_limits_remain_unknown(self):
        unknown=robot_context(self.fixture.nav)
        self.assertIsNone(unknown['motion']['max_forward_mps'])
        self.fixture.nav.geometry=VehicleGeometry([[-.2,-.4],[.9,-.4],[.9,.4],[-.2,.4]],np.eye(4),source={'status':'unit_assumption'})
        self.fixture.nav.motion=dict(parameters_assumed=True,max_forward_mps=.37,allow_reverse=False,allow_in_place_rotation=True)
        actual=robot_context(self.fixture.nav)
        self.assertAlmostEqual(actual['body']['length_m'],1.1)
        self.assertAlmostEqual(actual['body']['width_m'],.8)
        self.assertEqual(actual['motion']['max_forward_mps'],.37)
        self.assertFalse(actual['motion']['allow_reverse'])
        self.assertFalse(actual['physical_platform_performance_verified'])

    def test_current_rgb_metadata_and_missing_inputs_are_explicit(self):
        self.session.prompt()
        observation=self.session.observation
        rgb=observation.prompt()['rgb']
        self.assertEqual(rgb['stamp_s'],observation.record['rgb']['stamp_s'])
        self.assertEqual(rgb['shape'],observation.record['rgb']['shape'])
        self.assertNotIn('encoded_sha256',rgb)
        self.fixture.nav.observe=lambda:None
        self.fixture.nav.state['localization_status']='invalid'
        body=self.fixture.request()
        state=json.loads(body['messages'][-1]['content'][0]['text'])['state']
        self.assertFalse(state['visual_inputs']['rgb']['available'])
        self.assertEqual(state['visual_inputs']['rgb']['reason'],'no_current_rgb')
        self.assertFalse(state['visual_inputs']['maps']['available'])

    def test_registration_and_latest_tool_rejection_survive_history_eviction(self):
        goal=dict(goal_id='first',kind='point',target_id='elevator',description='到电梯口')
        self.session.call('record_task',goals=[goal])
        self.session.call('record_task',goals=[goal])
        self.session.conversation.max_exchanges=1
        self.session.conversation.notice('history_trimmed_fixture')
        state=self.session.prompt()['current_state']
        self.assertTrue(state['requirements_registered'])
        self.assertEqual(state['next_goal_id'],'first')
        self.assertEqual(state['last_tool_feedback']['tool'],'record_task')
        self.assertIn('already recorded',state['last_tool_feedback']['reason'])

    def test_tool_parameter_semantics_are_in_the_native_schema(self):
        for tool in native_tools():
            for name,schema in tool['function']['parameters']['properties'].items():
                self.assertIn('description',schema,(tool['function']['name'],name))

    def test_unrequested_heading_cannot_be_labelled_as_user_requirement(self):
        self.session.call('record_task',goals=[dict(goal_id='first',kind='point',target_id='elevator',description='到电梯口')])
        state=self.session.prompt()['current_state']
        self.assertEqual(state['selected_goal']['allowed_orientation_sources'],['semantic','autonomous'])
        result=self.session.call('navigate',goal_id='first',pose=[5.,4.,0.],
            view_id=self.session.spatial_view.record['view_id'],orientation_source='user')
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['reason'],'orientation_source_disagrees_with_registered_requirement')
        self.assertEqual(self.session.skill.record['attempts'],[])
        self.assertIsNone(self.session.task_goals.goals[0]['selected'])

    def test_heading_arrow_remains_visible_on_coarse_global_overview(self):
        image=render(self.session.catalog,self.fixture.grid,[0.,0.,140.,100.],[40.,40.,0.],size=(768,640))
        pixels=np.asarray(Image.open(io.BytesIO(image)))
        yy,xx=np.nonzero(np.all(pixels==[204,102,0],axis=2))
        self.assertGreater(xx.max()-xx.min(),14)
        self.assertGreater(yy.max()-yy.min(),3)
