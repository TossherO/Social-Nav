"""Current wheelchair task semantics, independent of model or service availability."""
import unittest
from jsonschema import Draft202012Validator,ValidationError
from decision.vlm.vlm_api_common import groups_schema,validate_groups
from decision.vlm.follow import appearance_schema,follow_schema,validate_decision


class PromptSchemaTests(unittest.TestCase):
    def test_groups_are_dynamic_and_cross_group_overlap_is_rejected(self):
        schema=groups_schema([0,1,2,3]);Draft202012Validator.check_schema(schema)
        validate_groups({'walking':[[0,1]],'talking':[[2,3]]},[0,1,2,3],schema)
        for raw in [{'walking':[[0,1]],'talking':[[1,2]]},{'walking':[[0,9]]},
                    {'walking':[[False,1]]},{'g':[],'u':[]}]:
            with self.subTest(raw=raw),self.assertRaises((ValueError,ValidationError)):
                validate_groups(raw,[0,1,2,3],schema)

    def test_less_than_two_people_allows_only_empty_groups(self):
        for pids in [[],[0]]:
            validate_groups({},pids)
            with self.assertRaises(ValidationError):validate_groups({'walking':[[0,1]]},pids)

    def test_appearance_and_history_states_cannot_be_interchanged(self):
        with self.assertRaises(ValidationError):validate_decision({'id':0,'state':'visible'},appearance_schema([0]))
        schema=follow_schema([0],dict(confirmed_before=False,hold_allowed=False))
        with self.assertRaises(ValidationError):validate_decision({'id':0,'state':'occluded'},schema)


if __name__=='__main__':unittest.main()
