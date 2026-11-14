"""Grounded context, nonduplicated annotations and persistent task information."""
import copy
import io
import json
import unittest
import numpy as np
from PIL import Image
from decision.agent import CONFIG_DIR
from decision.agent.core.context import project_dialogue, robot_context, view_context
from decision.agent.core.model_projection import ModelProjection
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
        self.assertEqual(historical['view_id'],ModelProjection().reference('view',original['view_id']))
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

    def test_lookup_covers_points_and_regions_without_leaking_future_modes(self):
        all_items = self.session.call('lookup_places',name='')
        self.assertTrue(all_items['places']);self.assertTrue(all_items['regions'])
        window = next(p for p in all_items['places'] if p['place_id']=='front_window')
        self.assertEqual(self.session.call('lookup_places',name='front_window')['places'],[window])
        future = AgentSession('future',Catalog.load(CONFIG_DIR/'simulation.json',condition='simple'),
                              self.fixture.nav,self.fixture.store)
        result = future.call('lookup_places',name='')
        self.assertEqual((result['places'],result['regions']),([],[]))

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

    def test_delayed_point_goal_rechecks_new_view_without_changing_model_call(self):
        self.session.call('record_task',goals=[dict(goal_id='target',kind='point',target_id='elevator',description='到电梯')])
        self.session.prompt();context=self.session.context()
        old=self.session.spatial_view.record['view_id']
        call=dict(tool='navigate',arguments=dict(goal_id='target',pose=[5.,4.,0.],view_id=old,orientation_source='autonomous'))
        self.fixture.nav.state['stamp_s']=14.
        self.assertTrue(self.session.valid_expired_context(context))
        result=self.session.revalidate_delayed_navigation(call,context)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(call['arguments']['view_id'],old)
        self.assertEqual(result['revalidation']['model_view_id'],old)
        self.assertNotEqual(result['revalidation']['checked_view_id'],old)
        self.assertEqual(result['revalidation']['model_source_stamp_s'],10.)
        self.assertEqual(result['revalidation']['checked_source_stamp_s'],14.)

    def test_delayed_target_refused_after_robot_moves(self):
        self.session.call('record_task',goals=[dict(goal_id='target',kind='point',target_id='elevator',description='到电梯')])
        self.session.prompt();context=self.session.context()
        old=self.session.spatial_view.record['view_id']
        self.fixture.nav.state.update(stamp_s=14.,pose=[5.,4.,0.])
        result=self.session.revalidate_delayed_navigation(dict(tool='navigate',arguments=dict(
            goal_id='target',pose=[5.,4.,0.],view_id=old,orientation_source='autonomous')),context)
        self.assertEqual(result['reason'],'robot_pose_changed_since_model_input')
        self.assertEqual(self.session.skill.record['attempts'],[])

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

    def test_candidate_checks_static_footprint_and_candidate_camera_pose(self):
        geometry=VehicleGeometry([[-.2,-.3],[.8,-.3],[.8,.3],[-.2,.3]],np.eye(4),source='fixture',assumed=False)
        sensors=dict(status='configured',camera=dict(position_m=[.5,0.,0.],view=dict(
            status='approximate_pinhole',edge_yaw_rad=[.5,-.5])))
        inputs=dict(global_grid=self.fixture.grid,scene=None,pose=[4.,4.,0.],stamp_s=10.,
                    clock_domain=self.fixture.nav.clock_domain,geometry=geometry,sensors=sensors)
        self.fixture.nav.spatial_inputs=lambda:inputs
        self.session.prompt()
        view=self.session.spatial_view.record['view_id']
        point=self.session.catalog.places['elevator']['pose']
        yaw=float(np.arctan2(point[1]-4.,point[0]-4.))
        result=self.session.call('check_candidate',pose=[4.,4.,yaw],view_id=view,point_ids=['elevator'])
        self.assertEqual(result['static_footprint']['status'],'free')
        self.assertTrue(result['camera_view']['points'][0]['in_horizontal_fov'])
        self.assertFalse(result['camera_view']['actual_visibility_verified'])
        self.assertFalse(result['static_footprint']['route_checked'])
        result=self.session.call('check_candidate',pose=[4.,4.,yaw+np.pi],view_id=view,point_ids=['elevator'])
        self.assertFalse(result['camera_view']['points'][0]['in_horizontal_fov'])
        result=self.session.call('check_candidate',pose=[0.,0.,0.],view_id=view)
        self.assertEqual(result['static_footprint']['status'],'outside')
        inputs['stamp_s']=10.1
        self.assertEqual(self.session.call('check_candidate',pose=[4.,4.,0.],view_id=view)['reason'],'spatial_source_changed')

    def test_candidate_summary_and_preview_survive_trimming_and_expire_with_source(self):
        geometry=VehicleGeometry([[-.2,-.3],[.8,-.3],[.8,.3],[-.2,.3]],np.eye(4),source='fixture',assumed=False)
        sensors=dict(status='configured',camera=dict(position_m=[.5,0.,0.],view=dict(
            status='approximate_pinhole',edge_yaw_rad=[.5,-.5])))
        inputs=dict(global_grid=self.fixture.grid,scene=None,pose=[4.,4.,0.],stamp_s=10.,
                    clock_domain=self.fixture.nav.clock_domain,geometry=geometry,sensors=sensors)
        self.fixture.nav.spatial_inputs=lambda:inputs
        self.session.prompt();original=self.session.spatial_view.record['view_id']
        for _ in range(2):
            result=self.session.call('check_candidate',pose=[4.,4.,0.],view_id=original,point_ids=['elevator'])
            self.assertEqual(result['status'],'available')
        self.session.conversation.max_exchanges=1
        self.session.conversation.notice('trim_fixture',number=1)
        self.session.conversation.notice('trim_fixture',number=2)
        payload=self.session.prompt();entries=payload['current_state']['recent_candidate_checks']
        self.assertEqual(len(entries),1);self.assertEqual(entries[0]['checks'],2)
        self.assertEqual(entries[0]['source_stamp_s'],10.)
        self.assertEqual(entries[0]['static_footprint'],'free')
        self.assertFalse(entries[0]['occlusion_checked'])
        preview=self.session.spatial_view.record['candidate_preview']
        self.assertEqual(preview,entries[0]);self.assertNotEqual(self.session.spatial_view.record['view_id'],original)
        pixels=np.asarray(Image.open(io.BytesIO(self.session.spatial_view.images[1])))
        self.assertTrue(np.any(np.all(pixels==[180,39,181],axis=2)))
        view=self.session.spatial_view.record['view_id']
        result=self.session.call('check_candidate',pose=[8.,4.,0.],view_id=view,point_ids=['elevator'])
        self.assertEqual(result['status'],'available')
        self.session.prompt()
        self.assertTrue(self.session.spatial_view.record['candidate_display_expanded'])
        self.assertLess(self.session.spatial_view.record['requested_local_bounds'][2],8.)
        self.assertGreater(self.session.spatial_view.record['local_bounds'][2],8.)
        inputs['stamp_s']=10.1;self.fixture.nav.state['stamp_s']=10.1
        payload=self.session.prompt()
        history=payload['current_state']['recent_candidate_checks']
        self.assertEqual(len(history),2)
        self.assertTrue(all(entry['historical_only'] for entry in history))
        self.assertTrue(all(entry['source_stamp_s']==10. for entry in history))
        self.assertIsNone(self.session.spatial_view.record['candidate_preview'])
        view=self.session.spatial_view.record['view_id']
        self.session.call('check_candidate',pose=[8.,4.,0.],view_id=view,point_ids=['elevator'])
        payload=self.session.prompt();history=payload['current_state']['recent_candidate_checks']
        self.assertFalse(history[-1]['historical_only']);self.assertEqual(history[-1]['source_stamp_s'],10.1)
        self.fixture.nav.state['generation']+=1
        payload=self.session.prompt()
        self.assertEqual(payload['current_state']['recent_candidate_checks'],[])

    def test_candidate_preserves_occupied_unknown_and_missing_camera(self):
        from perception.environment.layers import GridSnapshot
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='fixture')
        for cell,expected in ((100,'occupied'),(-1,'unknown')):
            cells=np.zeros((100,140),np.int8);cells[40,40]=cell
            grid=GridSnapshot(dict(self.fixture.grid.metadata),dict(occupancy=cells))
            self.fixture.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,pose=[4.,4.,0.],stamp_s=10.,
                clock_domain=self.fixture.nav.clock_domain,geometry=geometry)
            self.session.prompt()
            result=self.session.call('check_candidate',pose=[4.,4.,0.],view_id=self.session.spatial_view.record['view_id'])
            self.assertEqual(result['static_footprint']['status'],expected)
            self.assertEqual(result['camera_view']['status'],'unavailable')

    def test_candidate_transforms_bound_local_grid_and_reports_coverage(self):
        from perception.environment.layers import GridSnapshot
        geometry=VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='fixture')
        cells=np.zeros((20,20),np.int8);cells[10,10]=100
        local=GridSnapshot(dict(frame_id='odom',shape=[20,20],resolution=.1,origin_xy=[0.,0.],snapshot_id='local'),
                           dict(occupancy=cells))
        transform=np.eye(4);transform[:2,3]=[3.,3.]
        inputs=dict(global_grid=self.fixture.grid,scene=None,local_grid=local,local_grid_observation_id='source',
                    local_grid_map_from_local=transform.tolist(),pose=[4.,4.,0.],stamp_s=10.,
                    clock_domain=self.fixture.nav.clock_domain,geometry=geometry)
        self.fixture.nav.spatial_inputs=lambda:inputs
        self.session.prompt();view=self.session.spatial_view.record['view_id']
        result=self.session.call('check_candidate',pose=[4.,4.,0.],view_id=view)
        self.assertEqual(result['static_footprint']['status'],'free')
        self.assertEqual(result['local_footprint']['status'],'occupied')
        result=self.session.call('check_candidate',pose=[8.,8.,0.],view_id=view)
        self.assertEqual(result['local_footprint']['status'],'not_covered')

    def test_query_map_static_crop_does_not_hide_prior_or_weaken_local_candidate_check(self):
        from perception.environment.layers import GridSnapshot
        geometry=VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]],np.eye(4),source='fixture')
        cells=np.full((100,140),-1,np.int8);cells[40,40]=100
        local=GridSnapshot(dict(self.fixture.grid.metadata,frame_id='odom',snapshot_id='local'),dict(occupancy=cells))
        inputs=dict(global_grid=self.fixture.grid,scene=None,local_grid=local,local_grid_observation_id='obs',
                    local_grid_map_from_local=np.eye(4).tolist(),pose=[4.,4.,0.],stamp_s=10.,
                    clock_domain=self.fixture.nav.clock_domain,geometry=geometry)
        self.fixture.nav.spatial_inputs=lambda:inputs
        self.session.prompt()
        default=self.session.spatial_view
        self.assertEqual(default.record['local_source'],'static_prior_only')
        self.assertTrue(default.record['local_grid_present'])
        pixels=np.asarray(Image.open(io.BytesIO(default.images[1])))
        self.assertFalse(np.any(np.all(pixels==[130,151,168],axis=2)))
        self.assertTrue(np.any(np.all(pixels==[246,246,239],axis=2)))
        default_candidate=self.session.call('check_candidate',pose=[4.,4.,0.],view_id=default.record['view_id'])
        self.assertEqual(default_candidate['local_footprint']['status'],'occupied')
        self.assertEqual(default_candidate['static_footprint']['status'],'free')
        result=self.session.call('query_map',bounds=[2.,2.,6.,6.])
        view=self.session.spatial_view
        self.assertEqual(result['view']['local_source'],'static_prior_only')
        self.assertIsNotNone(view.record['local_binding'])
        self.assertNotEqual(view.record['view_id'],default.record['view_id'])
        pixels=np.asarray(Image.open(io.BytesIO(view.images[1])))
        self.assertFalse(np.any(np.all(pixels==[130,151,168],axis=2)))
        self.assertTrue(np.any(np.all(pixels==[246,246,239],axis=2)))
        candidate=self.session.call('check_candidate',pose=[4.,4.,0.],view_id=view.record['view_id'])
        self.assertEqual(candidate['static_footprint']['status'],'free')
        self.assertEqual(candidate['local_footprint']['status'],'occupied')
        self.fixture.nav.state['stamp_s']=10.2;inputs['stamp_s']=10.2
        self.session.prompt()
        self.assertEqual(self.session.spatial_view.record['local_bounds'],[2.,2.,6.,6.])
        self.assertEqual(self.session.spatial_view.record['local_source'],'static_prior_only')

    def test_navigation_checks_final_orientation_not_only_model_candidate(self):
        from perception.environment.layers import GridSnapshot
        geometry=VehicleGeometry([[-.2,-.3],[.8,-.3],[.8,.3],[-.2,.3]],np.eye(4),source='fixture')
        cells=np.zeros((100,140),np.int8);cells[40,46]=100
        grid=GridSnapshot(dict(self.fixture.grid.metadata),dict(occupancy=cells))
        self.fixture.nav.geometry=geometry
        self.fixture.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,pose=[4.,4.,0.],stamp_s=10.,
            clock_domain=self.fixture.nav.clock_domain,geometry=geometry)
        self.session.prompt();view=self.session.spatial_view.record['view_id']
        self.session.call('record_task',goals=[dict(goal_id='target',kind='point',target_id='',
            position=[4.,4.],description='test')])
        prior=self.session.call('check_candidate',pose=[4.,4.,float(np.pi)],view_id=view)
        self.assertEqual(prior['static_footprint']['status'],'free')
        result=self.session.call('navigate',goal_id='target',pose=[4.,4.,0.],view_id=view,orientation_source='autonomous')
        self.assertEqual(result['reason'],'candidate_footprint_occupied')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_navigation_respects_current_local_occupied_and_cleared_static(self):
        from perception.environment.layers import GridSnapshot
        geometry=VehicleGeometry([[-.2,-.3],[.8,-.3],[.8,.3],[-.2,.3]],np.eye(4),source='fixture')
        self.fixture.nav.geometry=geometry
        for static_cell,local_cell,expected in ((0,100,'refused'),(100,0,'accepted')):
            static=np.zeros((100,140),np.int8);static[40,46]=static_cell
            local=np.zeros((100,140),np.int8);local[40,46]=local_cell
            grid=GridSnapshot(dict(self.fixture.grid.metadata),dict(occupancy=static))
            local_grid=GridSnapshot(dict(self.fixture.grid.metadata,snapshot_id='local'),dict(occupancy=local))
            self.fixture.nav.spatial_inputs=lambda:dict(global_grid=grid,scene=None,local_grid=local_grid,
                local_grid_observation_id='obs',local_grid_map_from_local=np.eye(4).tolist(),pose=[4.,4.,0.],
                stamp_s=10.,clock_domain=self.fixture.nav.clock_domain,geometry=geometry)
            session=AgentSession('test',self.session.catalog,self.fixture.nav,self.fixture.store)
            session.prompt();view=session.spatial_view.record['view_id']
            session.call('record_task',goals=[dict(goal_id='target',kind='point',target_id='',position=[4.,4.],description='test')])
            result=session.call('navigate',goal_id='target',pose=[4.,4.,0.],view_id=view,orientation_source='autonomous')
            self.assertEqual(result['status'],expected)

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
