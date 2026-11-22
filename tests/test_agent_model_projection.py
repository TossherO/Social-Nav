"""Model-facing short references preserve the original identities for execution."""
import copy
import hashlib
import json
import unittest

from decision.agent.core.model_projection import ModelProjection, ModelReferenceNames


class ModelProjectionTests(unittest.TestCase):
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
