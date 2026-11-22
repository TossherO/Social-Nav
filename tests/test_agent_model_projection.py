"""Model-facing short references preserve the original identities for execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from decision.agent.core.model import Model
from decision.agent.core.model_projection import ModelProjection, ModelReferenceNames
from decision.agent.core.store import Budget, Store
from decision.agent.tools.framework import EXECUTION_TOOLS, native_tools
from tests.test_agent_conversation import wire


class ModelProjectionTests(unittest.TestCase):
    @staticmethod
    def state(*, goal_id='g1', view='original-view', rgb='original-rgb', **selected):
        return dict(
            next_goal_id=goal_id, selected_goal=dict(goal_id=goal_id, **selected),
            visual_inputs=dict(maps=dict(available=True, view_id=view),
                               rgb=dict(available=True, observation_id=rgb)),
        )

    def test_model_can_use_registered_first_point_without_retyping_pose_or_ids(self):
        captured = []
        message = [dict(id='compact-navigation', type='function',
                        function=dict(name='navigate', arguments='{}'))]

        def handle(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, text=wire(message))

        projection = ModelProjection()
        projection.allow(self.state(selection=None, initial_pose=[3, 4, 1.2], initial_point_id='elevator',
                                    heading_required=True, allowed_orientation_sources=['user']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'fixture.json'
            config.write_text(json.dumps(dict(base_url='https://unit.invalid/v1', api_key='fixture')))
            store = Store(root/'session.sqlite')
            budget = Budget(root/'budget.sqlite')
            try:
                reply = Model(config, budget, store, transport=httpx.MockTransport(handle)).request(
                    'test', [], tool_names=('navigate',), projection=projection)
            finally:
                budget.close()
                store.close()

        schema = captured[0]['tools'][0]['function']['parameters']
        self.assertEqual(schema['required'], [])
        self.assertFalse({'goal_id', 'view_id', 'orientation_source'} & schema['properties'].keys())
        self.assertEqual(reply.message['tool_calls'], message)
        self.assertEqual(reply.action, dict(tool='navigate', arguments=dict(
            goal_id='g1', view_id='original-view', point_id='elevator', yaw=1.2, orientation_source='user')))

    def test_first_stop_defaults_leave_new_candidate_pose_and_source_validation_in_place(self):
        projection = ModelProjection()
        projection.allow(self.state(initial_pose=[3, 4], heading_required=False,
                                    allowed_orientation_sources=['semantic', 'autonomous']))
        self.assertEqual(projection.resolve_action(projection.validate_action(
            dict(tool='navigate', arguments={})))['arguments'],
            dict(goal_id='g1', view_id='original-view', pose=[3, 4]))
        action = dict(tool='navigate', arguments=dict(pose=[3.1, 4.2, 1.7]))
        self.assertEqual(projection.resolve_action(projection.validate_action(action))['arguments'],
                         dict(goal_id='g1', view_id='original-view', pose=[3.1, 4.2, 1.7],
                              orientation_source='autonomous'))
        self.assertEqual(action['arguments'], dict(pose=[3.1, 4.2, 1.7]))

        state = self.state(selection=dict(pose=[3, 4, 1.2]), initial_pose=[3, 4],
                           heading_required=True, allowed_orientation_sources=['semantic', 'autonomous'])
        next_turn = ModelProjection()
        next_turn.allow(state)
        self.assertEqual(next_turn.validate_action(action)['arguments']['orientation_source'], 'semantic')
        with self.assertRaisesRegex(ValueError, 'goal_id'):
            next_turn.resolve_action(next_turn.validate_action(dict(tool='navigate', arguments=dict(
                pose=[3.1, 4.2], goal_id='earlier-stage'))))

    def test_source_binding_does_not_replace_explicit_stale_references(self):
        names = ModelReferenceNames()
        first = ModelProjection(names)
        first.allow(self.state())
        old_view = first.reference('view', 'original-view')
        old_rgb = first.reference('obs', 'original-rgb')
        current = ModelProjection(names)
        current.allow(self.state(view='second-view', rgb='second-rgb'))
        for tool, arguments in (
            ('check_candidate', dict(pose=[1, 2], view_id=old_view)),
            ('save_keyframe', dict(reason='重要线索', observation_id=old_rgb)),
        ):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                current.resolve_action(current.validate_action(dict(tool=tool, arguments=arguments)))
        expanded = current.resolve_action(current.validate_action(dict(tool='check_candidate', arguments=dict(pose=[1, 2]))))
        self.assertEqual(expanded['arguments']['view_id'], 'second-view')
        schemas = {tool['function']['name']: tool['function']['parameters']['properties']
                   for tool in current.tools(native_tools(('read_keyframe', 'start_local_task', 'save_keyframe')))}
        self.assertIn('keyframe_id', schemas['read_keyframe'])
        self.assertIn('reference_id', schemas['start_local_task'])
        self.assertNotIn('observation_id', schemas['save_keyframe'])

    def test_navigation_and_visual_review_availability_tracks_the_current_stop(self):
        names = ('navigate', 'record_monitor_review', 'wait_for_event', 'complete_subtask')
        state = self.state(visual_adjustment=dict(review=None))
        before_arrival = ModelProjection()
        before_arrival.allow(state)
        self.assertEqual(before_arrival.available_tools(names), ('navigate', 'wait_for_event', 'complete_subtask'))

        state.update(completion_evidence=dict(arrival_stamp_s=10.))
        arrived = ModelProjection()
        arrived.allow(state)
        self.assertEqual(arrived.available_tools(names), ('record_monitor_review', 'wait_for_event', 'complete_subtask'))
        review = dict(tool='record_monitor_review', arguments=dict(result='partly_satisfied', evidence='看见门，另一侧被挡'))
        self.assertEqual(arrived.resolve_action(arrived.validate_action(review))['arguments']['observation_id'],
                         'original-rgb')

        state['selected_goal']['visual_adjustment']['review'] = dict(result='partly_satisfied')
        state.update(task_monitor=dict(continuous_region_id='lobby', conditional_review=dict(phase='reviewing')))
        reviewing = ModelProjection()
        reviewing.allow(state)
        self.assertEqual(reviewing.available_tools(names), names)
        self.assertNotIn('observation_id', reviewing.validate_action(review)['arguments'])
        review['arguments']['visual_priority'] = 'required'
        self.assertEqual(reviewing.resolve_action(reviewing.validate_action(review))['arguments']['observation_id'],
                         'original-rgb')

        state['task_monitor'] = {}
        state['completion_evidence']['missing'] = ['current_rgb']
        waiting = ModelProjection()
        waiting.allow(state)
        self.assertNotIn('record_monitor_review',waiting.available_tools(names))
        self.assertIn('wait_for_event',waiting.available_tools(names))

    def test_known_execution_stage_keeps_its_actions_and_common_controls(self):
        actions = {'navigate', 'navigate_route', 'select_follow_target', 'start_local_task',
                   'pause_map_localization', 'restart_map_localization', 'record_elevator_observation'}
        controls = {'wait_for_event', 'cancel_navigation', 'lookup_places', 'ask_user', 'fail_task',
                    'complete_subtask', 'observe_scene', 'check_candidate', 'set_navigation_monitor'}
        for stage, kind, expected in (
            ('navigation', 'point', {'navigate'}),
            ('navigation', 'region', {'navigate'}),
            ('navigation', 'reference', {'navigate'}),
            ('navigation', 'route', {'navigate_route'}),
            ('navigation', 'sequence', {'navigate_route'}),
            ('follow', 'skill', {'select_follow_target', 'start_local_task'}),
            ('seating', 'skill', {'start_local_task'}),
            ('observation', 'skill', set()),
            ('elevator', 'skill', {'pause_map_localization'}),
        ):
            with self.subTest(stage=stage, kind=kind):
                state = self.state()
                state['completion_evidence'] = dict(requirement=dict(goal_id='g1', stage_type=stage, kind=kind))
                projection = ModelProjection()
                projection.allow(state)
                offered = set(projection.available_tools(EXECUTION_TOOLS))
                self.assertEqual(offered & actions, expected)
                self.assertTrue(controls <= offered)
                self.assertNotIn('read_keyframe', offered)
                self.assertNotIn('record_monitor_review', offered)

        state.update(keyframes=[dict(keyframe_id='stored-keyframe')],
                     task_monitor=dict(continuous_region_id='lobby', conditional_review=dict(phase='reviewing')))
        projection = ModelProjection()
        projection.allow(state)
        self.assertTrue({'read_keyframe', 'record_monitor_review'} <= set(projection.available_tools(EXECUTION_TOOLS)))
        del state['completion_evidence']
        uncertain = ModelProjection()
        uncertain.allow(state)
        self.assertEqual(uncertain.available_tools(EXECUTION_TOOLS), EXECUTION_TOOLS)

    def test_single_visual_priority_is_bound_only_for_unambiguous_post_stop_review(self):
        state = self.state(visual_adjustment=dict(review=None))
        requirement = dict(goal_id='g1', kind='point', semantic_conditions=['入口可见'])
        state['completion_evidence'] = dict(requirement=requirement, arrival_stamp_s=10.)
        review = dict(tool='record_monitor_review', arguments=dict(result='partly_satisfied', evidence='看见门框'))
        for both, conditional in ((False, False), (True, False), (False, True)):
            with self.subTest(both=both, conditional=conditional):
                requirement['visual_preferences'] = ['尽量看到标牌'] if both else []
                state['task_monitor'] = dict(continuous_region_id='lobby') if conditional else None
                projection = ModelProjection()
                projection.allow(state)
                tool = projection.tools(native_tools(('record_monitor_review',)))[0]['function']
                self.assertEqual('visual_priority' in tool['parameters']['properties'], both or conditional)
                action = projection.resolve_action(projection.validate_action(review))
                self.assertEqual(action['arguments'].get('visual_priority'),
                                 None if both or conditional else 'required')

    def test_only_visual_navigation_schema_offers_combined_sightline_check(self):
        for visual in (False, True):
            with self.subTest(visual=visual):
                state = self.state()
                requirement = dict(goal_id='g1', kind='point',
                                   semantic_conditions=['入口可见'] if visual else [])
                state['completion_evidence'] = dict(requirement=requirement)
                projection = ModelProjection()
                projection.allow(state)
                tools = {tool['function']['name']: tool['function']
                         for tool in projection.tools(native_tools(('navigate', 'check_candidate')))}
                self.assertEqual('point_ids' in tools['navigate']['parameters']['properties'], visual)
                self.assertIn('point_ids', tools['check_candidate']['parameters']['properties'])
                self.assertIn('pose=[x,y,yaw]', tools['navigate']['description'])
                self.assertIn('point_id 加单独的 yaw', tools['navigate']['description'])

    def test_registered_xy_offers_only_one_way_to_express_a_new_heading(self):
        state = self.state()
        state['completion_evidence'] = dict(requirement=dict(goal_id='g1',kind='point',target_id='',
                                                             position=[-18.6,-9.0],semantic_conditions=['同框']))
        projection = ModelProjection()
        projection.allow(state)
        tool = projection.tools(native_tools(('navigate',)))[0]['function']
        self.assertNotIn('yaw',tool['parameters']['properties'])
        self.assertNotIn('point_id',tool['parameters']['properties'])
        self.assertIn('pose=[x,y,yaw]',tool['description'])

    def test_projected_references_resolve_without_changing_source_or_native_protocol(self):
        raw_view='a'*64;raw_observation='scene:'+'b'*64;raw_candidate='c'*64
        native_id='tool_call_'+'d'*32
        content=dict(source='runtime_current_state',state=dict(task_id='e'*32,
            map=dict(valid=True,catalog_fingerprint='f'*64),
            visual_inputs=dict(maps=dict(view_id=raw_view),rgb=dict(observation_id=raw_observation)),
            follow_candidates=dict(binding=raw_candidate),image_sha256='1'*64))
        messages=[dict(role='user',content=json.dumps(content)),
                  dict(role='assistant',content=None,tool_calls=[dict(id=native_id,type='function',
                       function=dict(name='check_candidate',arguments=json.dumps(dict(view_id=raw_view,pose=[0,0,0]))))]),
                  dict(role='tool',tool_call_id=native_id,content=json.dumps(dict(status='available',view_id=raw_view)))]
        original=copy.deepcopy(messages);projection=ModelProjection();projection.allow(content['state'])
        projected=projection.messages(messages)
        result=json.loads(projected[0]['content'])['state']
        self.assertNotIn('task_id',result);self.assertNotIn('image_sha256',result)
        self.assertNotIn('catalog_fingerprint',result['map']);self.assertTrue(result['map']['valid'])
        view=result['visual_inputs']['maps']['view_id'];obs=result['visual_inputs']['rgb']['observation_id']
        self.assertNotIn('binding',result['follow_candidates'])
        self.assertEqual((view,obs),('v1','o1'))
        self.assertEqual(json.loads(projected[1]['tool_calls'][0]['function']['arguments'])['view_id'],view)
        self.assertEqual(json.loads(projected[2]['content'])['view_id'],view)
        self.assertEqual(projected[1]['tool_calls'][0]['id'],native_id)
        self.assertEqual(projected[2]['tool_call_id'],native_id)
        action=dict(tool='test',arguments=dict(view_id=view,observation_id=obs))
        self.assertEqual(projection.resolve_action(action)['arguments'],dict(view_id=raw_view,
            observation_id=raw_observation))
        self.assertEqual(messages,original)
        self.assertEqual(ModelProjection(projection.names).reference('view',raw_view),view)

    def test_unknown_or_wrong_kind_reference_is_never_guessed(self):
        projection=ModelProjection();projection.allow(dict(visual_inputs=dict(maps=dict(view_id='a'*64))))
        view=projection.reference('view','a'*64)
        for value in ('o99','a'*64,view):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    projection.resolve_action(dict(tool='x',arguments=dict(observation_id=value)))

    def test_user_instruction_and_original_assistant_text_are_preserved(self):
        instruction=json.dumps(dict(source='user',task_id='a'*32))
        messages=[dict(role='user',content=instruction),dict(role='assistant',content=instruction)]
        self.assertEqual(ModelProjection().messages(messages,protected_indices=(0,)),messages)

    def test_current_follow_reference_and_keyframe_lookup(self):
        projection=ModelProjection()
        track=['tracking-session-'+'a'*32,2,'43']
        state=dict(visual_inputs={},keyframes=[dict(keyframe_id='b'*64)],visible_keyframe_ids=['b'*64],
                   follow_reference=dict(reference_id='c'*64,track=dict(identity=track)),
                   navigation=dict(feedback=dict(target_identity=track)),
                   map_binding=dict(valid=True,map_id='d'*64))
        projection.allow(state)
        result=projection.project(state)
        self.assertTrue(result['map_binding']['valid']);self.assertNotIn('map_id',result['map_binding'])
        self.assertEqual(result['follow_reference']['track']['identity'],'p1')
        self.assertEqual(result['navigation']['feedback']['target_identity'],'p1')
        self.assertEqual(projection.resolve_action(dict(tool='read_keyframe',
            arguments=dict(keyframe_id=result['visible_keyframe_ids'][0])))['arguments']['keyframe_id'],'b'*64)
        self.assertEqual(projection.resolve_action(dict(tool='start_local_task',
            arguments=dict(reference_id=result['follow_reference']['reference_id'])))['arguments']['reference_id'],'c'*64)

    def test_short_shaped_source_identity_is_not_mistaken_for_an_existing_alias(self):
        projection=ModelProjection()
        projection.reference('view','a'*64)
        original='v1'
        alias=projection.project(dict(view_id=original))['view_id']
        projection.allow(dict(visual_inputs=dict(maps=dict(view_id=original))))
        self.assertNotEqual(alias,original)
        self.assertEqual(projection.resolve_action(dict(tool='check_candidate',arguments=dict(view_id=alias)))['arguments']['view_id'],original)

    def test_history_keeps_source_aliases_without_granting_access_after_a_source_change(self):
        first_state = dict(visual_inputs=dict(maps=dict(view_id='a'*64),rgb=dict(observation_id='b'*64)),
                           follow_reference=dict(reference_id='c'*64),keyframes=[dict(keyframe_id='d'*64)])
        names=ModelReferenceNames()
        first = ModelProjection(names)
        first.allow(first_state)
        old = first.project(first_state)
        args = dict(goal_id='g1', task_type='follow', reference_id=old['follow_reference']['reference_id'])
        protocol = 'tool_call_'+'1'*32
        history = [
            dict(role='assistant',content=None,tool_calls=[dict(id=protocol,type='function',
                 function=dict(name='start_local_task',arguments=json.dumps(args)))]),
            dict(role='tool',tool_call_id=protocol,content=json.dumps(dict(status='accepted',evidence=first_state))),
        ]
        second = ModelProjection(names)
        second.allow(dict(visual_inputs=dict(maps=dict(view_id='e'*64),rgb=dict(observation_id='f'*64)),
                          follow_reference=dict(reference_id='2'*64),keyframes=[dict(keyframe_id='3'*64)]))
        projected = second.messages(history)
        self.assertEqual(projected[0],history[0])
        self.assertEqual(json.loads(projected[1]['content'])['evidence'],old)
        for key, reference in (
            ('view_id', old['visual_inputs']['maps']['view_id']),
            ('observation_id', old['visual_inputs']['rgb']['observation_id']),
            ('reference_id', old['follow_reference']['reference_id']),
            ('keyframe_id', old['keyframes'][0]['keyframe_id']),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                second.resolve_action(dict(tool='test',arguments={key:reference}))
        self.assertEqual(second.resolve_action(dict(tool='start_local_task',
            arguments=dict(reference_id=second.reference('ref','2'*64))))['arguments']['reference_id'],'2'*64)
        self.assertEqual(second.reference('ref','2'*64),'r2')

    def test_legacy_tool_argument_matches_projected_source_when_that_source_is_retained(self):
        original='a'*64
        legacy='view-'+hashlib.sha256(original.encode()).hexdigest()[:16]
        native='native-id'
        messages=[
            dict(role='assistant',content=None,tool_calls=[dict(id=native,type='function',
                 function=dict(name='check_candidate',arguments=json.dumps(dict(view_id=legacy,pose=[0,0,0]))))]),
            dict(role='tool',tool_call_id=native,content=json.dumps(dict(status='available',view_id=original))),
        ]
        projection=ModelProjection()
        result=projection.messages(messages)
        self.assertEqual(json.loads(result[0]['tool_calls'][0]['function']['arguments'])['view_id'],'v1')
        self.assertEqual(json.loads(result[1]['content'])['view_id'],'v1')
        self.assertIn(legacy,ModelProjection().messages(messages[:1])[0]['tool_calls'][0]['function']['arguments'])
        with self.assertRaises(ValueError):
            projection.resolve_action(dict(tool='check_candidate',arguments=dict(view_id='v1')))

    def test_map_labels_feedback_and_rejected_arguments_preserve_semantic_evidence(self):
        original_view='a'*64
        original_map='b'*64
        rejected_value='c'*64
        protocol='tool_call_'+'d'*32
        content = [dict(type='text',text=json.dumps(dict(source='current_spatial_view',
                        view=dict(view_id=original_view,annotation_reference=dict(reference_map_id=original_map))))),
                   dict(type='text',text='current map | view_id='+original_view),
                   dict(type='image_url',image_url=dict(url='data:image/png;base64,unit'))]
        args = dict(pose=[1,2,3], view_id=rejected_value, point_ids=['elevator'])
        rejected = dict(status='refused',reason='invalid_tool_arguments',tool='check_candidate',path=['view_id'],
                        detail=f'{rejected_value!r} is too long for view_id')
        original = [
            dict(role='assistant',content=None,tool_calls=[dict(id=protocol,type='function',
                 function=dict(name='check_candidate',arguments=json.dumps(args)))]),
            dict(role='tool',tool_call_id=protocol,content=json.dumps(rejected)),
            dict(role='user',content=content),
        ]
        projection=ModelProjection()
        projection.allow(dict(visual_inputs=dict(maps=dict(view_id=original_view))))
        projected=projection.messages(original)
        metadata=json.loads(projected[-1]['content'][0]['text'])
        view=metadata['view']['view_id']
        self.assertEqual(projected[-1]['content'][1]['text'],'current map | view_id='+view)
        self.assertEqual(projected[-1]['content'][2],content[2])
        self.assertEqual(metadata['view']['annotation_reference']['reference_map_id'],projection.reference('map',original_map))
        self.assertEqual(projected[0]['tool_calls'][0]['id'],projected[1]['tool_call_id'])
        cleaned=json.loads(projected[0]['tool_calls'][0]['function']['arguments'])
        self.assertEqual(cleaned,dict(args,view_id='[internal ID]'))
        feedback=json.loads(projected[1]['content'])
        self.assertEqual(feedback,dict(rejected,detail="'[internal ID]' is too long for view_id"))
        self.assertIn(rejected_value,original[0]['tool_calls'][0]['function']['arguments'])
        self.assertIn(rejected_value,original[1]['content'])


if __name__=='__main__':unittest.main()
