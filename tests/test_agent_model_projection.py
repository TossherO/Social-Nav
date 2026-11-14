"""Model-facing short references preserve the original identities for execution."""
import copy
import json
import unittest

from decision.agent.core.model_projection import ModelProjection


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
        original=copy.deepcopy(messages);projection=ModelProjection();projected=projection.messages(messages)
        result=json.loads(projected[0]['content'])['state']
        self.assertNotIn('task_id',result);self.assertNotIn('image_sha256',result)
        self.assertNotIn('catalog_fingerprint',result['map']);self.assertTrue(result['map']['valid'])
        view=result['visual_inputs']['maps']['view_id'];obs=result['visual_inputs']['rgb']['observation_id']
        candidate=result['follow_candidates']['binding']
        self.assertRegex(view,r'^view-[0-9a-f]{16}$');self.assertRegex(obs,r'^obs-[0-9a-f]{16}$')
        self.assertEqual(json.loads(projected[1]['tool_calls'][0]['function']['arguments'])['view_id'],view)
        self.assertEqual(json.loads(projected[2]['content'])['view_id'],view)
        self.assertEqual(projected[1]['tool_calls'][0]['id'],native_id)
        self.assertEqual(projected[2]['tool_call_id'],native_id)
        action=dict(tool='select_follow_target',arguments=dict(view_id=view,observation_id=obs,candidate_binding=candidate))
        self.assertEqual(projection.resolve_action(action)['arguments'],dict(view_id=raw_view,
            observation_id=raw_observation,candidate_binding=raw_candidate))
        self.assertEqual(messages,original)
        self.assertEqual(ModelProjection().reference('view',raw_view),view)

    def test_unknown_or_wrong_kind_reference_is_never_guessed(self):
        projection=ModelProjection();view=projection.reference('view','a'*64)
        for value in ('view-'+'0'*16,'a'*64,view):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    projection.resolve_action(dict(tool='x',arguments=dict(observation_id=value)))

    def test_user_instruction_and_original_assistant_text_are_preserved(self):
        instruction=json.dumps(dict(source='user',task_id='a'*32))
        messages=[dict(role='user',content=instruction),dict(role='assistant',content=instruction)]
        self.assertEqual(ModelProjection().messages(messages,protected_indices=(0,)),messages)

    def test_follow_search_and_historical_reference_lists(self):
        projection=ModelProjection()
        state=dict(candidate_set=dict(binding='a'*64),visible_keyframe_ids=['b'*64],
                   follow_reference=dict(reference_id='c'*64),map_binding=dict(valid=True,map_id='d'*64))
        result=projection.project(state)
        self.assertTrue(result['map_binding']['valid']);self.assertNotIn('map_id',result['map_binding'])
        self.assertEqual(projection.resolve_action(dict(tool='execute_follow_search',
            arguments=dict(candidate_binding=result['candidate_set']['binding'])))['arguments']['candidate_binding'],'a'*64)
        self.assertEqual(projection.resolve_action(dict(tool='read_keyframe',
            arguments=dict(keyframe_id=result['visible_keyframe_ids'][0])))['arguments']['keyframe_id'],'b'*64)
        self.assertEqual(projection.resolve_action(dict(tool='request_follow_recovery',
            arguments=dict(reference_id=result['follow_reference']['reference_id'])))['arguments']['reference_id'],'c'*64)

    def test_short_shaped_source_identity_is_not_mistaken_for_an_existing_alias(self):
        projection=ModelProjection();original='view-123456abcdef1234'
        alias=projection.project(dict(view_id=original))['view_id']
        self.assertNotEqual(alias,original)
        self.assertEqual(projection.resolve_action(dict(tool='check_candidate',arguments=dict(view_id=alias)))['arguments']['view_id'],original)


if __name__=='__main__':unittest.main()
