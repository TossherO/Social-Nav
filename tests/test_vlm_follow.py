"""T2 request isolation, causal descriptions and strict appearance decisions."""
import unittest

from jsonschema import Draft202012Validator, ValidationError

from decision.vlm.follow import (appearance_schema, appearance_fields, causal_description,
                                 validate_decision, follow_schema, FollowHistory)


class FollowInputTests(unittest.TestCase):
    def test_reference_cannot_come_from_future_or_another_recording(self):
        current = dict(source_id='bag', stamp_ns=100)
        causal_description(dict(source_id='bag', stamp_ns=100), current)
        for reference in [dict(source_id='bag', stamp_ns=101), dict(source_id='other', stamp_ns=90)]:
            with self.assertRaises(ValueError):
                causal_description(reference, current)

    def test_input_uses_description_and_geometry_without_hidden_answer(self):
        fields = dict(v=[1], dt=[-1, -.5, 0], ts={}, e=[-5, 15, -10, 10], o=[], d=[])
        window = dict(fields=fields, true_key='hidden_person', label={'target_pid': 1})
        model = appearance_fields(window, 'black jacket')
        self.assertEqual(set(model), set(fields) | {'target_description'})
        self.assertNotIn('hidden_person', str(model))
        model['v'].append(3)
        self.assertEqual(fields['v'], [1])
        with self.assertRaises(ValueError):
            appearance_fields(dict(fields=dict(fields, answer=1)), 'black jacket')

    def test_unknown_pids_and_refusal_with_a_claimed_identity_are_rejected(self):
        validator = Draft202012Validator(appearance_schema([2, 5]))
        validator.validate({'id': 5, 'match': 'unique'})
        for result in [{'id': 3, 'match': 'unique'}, {'id': -1, 'match': 'unique'},
                       {'id': 2, 'match': 'ambiguous'}, {'id': -1, 'match': 'lost'},
                       {'id': True, 'match': 'unique'}, {'id': 2, 'match': 'unique', 'reason': 'guess'}]:
            with self.assertRaises(ValidationError):
                validator.validate(result)

    def test_no_candidates_allows_refusal_and_never_creates_an_identity(self):
        validator = Draft202012Validator(appearance_schema([]))
        for state in ['ambiguous', 'none', 'insufficient']:
            validator.validate({'id': -1, 'match': state})
        with self.assertRaises(ValidationError):
            validator.validate({'id': 0, 'match': 'unique'})

    def test_numeric_pid_is_not_silently_coerced(self):
        schema = appearance_schema([2])
        for result in [{'id': 2.0, 'match': 'unique'}, {'id': -1.0, 'match': 'none'}]:
            with self.assertRaises(ValueError):
                validate_decision(result, schema)


class FollowHistoryTests(unittest.TestCase):
    def observation(self, seconds, pid=3, key='track/g0'):
        stamp = round(seconds * 1e9)
        points = [[i/10, 1, 0, True] for i in range(-30, 1)]
        return (dict(stamp_ns=stamp, source='bag'),
                dict(mapping={key:pid}, frames=['current'], trajectories={
                    'current':dict(status='available',ego=[points],people={key:[dict(points=points)]})}))

    def test_pid_is_remapped_and_occlusion_never_refreshes_visual_age(self):
        history = FollowHistory(hold_age_s=2.5)
        item,window = self.observation(10)
        history.accept(dict(id=3,state='visible'), item, window)
        item,window = self.observation(12, pid=1)
        context = history.context(item,window)
        self.assertEqual(context['current_pid'],1)
        self.assertTrue(context['hold_allowed'])
        history.accept(dict(id=1,state='occluded'),item,window)
        item,window = self.observation(13,pid=2)
        context=history.context(item,window)
        self.assertEqual(context['last_visual_age_s'],3)
        self.assertFalse(context['hold_allowed'])

    def test_identity_reset_or_uncertainty_revokes_holding(self):
        history=FollowHistory()
        item,window=self.observation(10)
        history.accept(dict(id=3,state='visible'),item,window)
        item,window=self.observation(12,key='track/g1')
        self.assertFalse(history.context(item,window)['hold_allowed'])
        item,window=self.observation(12)
        history.accept(dict(id=-1,state='uncertain'),item,window)
        self.assertFalse(history.context(item,window)['hold_allowed'])

    def test_invalid_pose_or_time_gap_cannot_be_used_as_hold_evidence(self):
        history=FollowHistory();item,window=self.observation(10)
        history.accept(dict(id=3,state='visible'),item,window)
        item,window=self.observation(12)
        window['trajectories']['current']['people']['track/g0'][0]['points']=[[0,1,0,True]]
        self.assertFalse(history.context(item,window)['hold_allowed'])
        window['trajectories']['current']['status']='unavailable'
        self.assertFalse(history.context(item,window)['hold_allowed'])

    def test_short_reset_is_not_hidden_by_flattening_separate_histories(self):
        history=FollowHistory();item,window=self.observation(10)
        history.accept(dict(id=3,state='visible'),item,window)
        item,window=self.observation(12)
        trace=window['trajectories']['current']
        points=trace['ego'][0]
        trace['ego']=[points[:20],points[21:]]
        self.assertFalse(history.context(item,window)['hold_allowed'])
        trace['ego']=[points]
        trace['people']['track/g0']=[dict(points=points[:20]),dict(points=points[21:])]
        self.assertFalse(history.context(item,window)['hold_allowed'])

    def test_schema_exposes_only_history_supported_states_and_ids(self):
        schema=follow_schema([2,5],dict(hold_allowed=False,confirmed_before=False))
        for result in [dict(id=2,state='occluded'),dict(id=-1,state='lost')]:
            with self.assertRaises(ValidationError):validate_decision(result,schema)
        schema=follow_schema([2,5],dict(hold_allowed=True,current_pid=5,confirmed_before=True))
        validate_decision(dict(id=5,state='occluded'),schema)
        for result in [dict(id=2,state='occluded'),dict(id=-1,state='not_found')]:
            with self.assertRaises(ValidationError):validate_decision(result,schema)
        with self.assertRaises(ValueError):
            follow_schema([2],dict(hold_allowed=True,current_pid=5,confirmed_before=True))


if __name__ == '__main__':
    unittest.main()
