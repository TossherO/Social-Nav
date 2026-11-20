"""The model-facing task contract preserves requirements and execution semantics."""
import copy
import unittest

from decision.agent import CONFIG_DIR
from decision.agent.core.tasks import TaskGoals
from decision.agent.tools.framework import native_tools, validate_call
from decision.agent.tools.places import Catalog


class RegistrationContractTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.load(CONFIG_DIR/'simulation.json')

    def test_advertised_schema_groups_target_and_conditions(self):
        tool = next(t for t in native_tools() if t['function']['name']=='record_task')
        goal = tool['function']['parameters']['properties']['goals']['items']
        self.assertEqual(goal['required'], ['goal_id','stage','description'])
        self.assertIn('target',goal['properties'])
        self.assertIn('conditions',goal['properties'])
        self.assertNotIn('kind',goal['properties'])
        self.assertNotIn('circuit',goal['properties'])

    def test_structured_requirements_keep_numeric_and_visual_constraints(self):
        goal=dict(goal_id='g1',stage='reference',description='到窗户附近，必须看到门，尽量看到电梯',
            target=dict(id='front_window'),conditions=dict(max_distance_m=2.,facing='elevator',
                visual=[dict(description='看到门',priority='required'),
                        dict(description='看到电梯',priority='preferred')]),
            monitor=dict(interval_s=15.),execution_notes='核对实际观察')
        validate_call(dict(tool='record_task',arguments=dict(goals=[goal])))
        task=TaskGoals(self.catalog);task.record([goal])
        req=task.goals[0]['requirement']
        self.assertEqual((req['kind'],req['target_id']),('reference','front_window'))
        self.assertEqual(req['max_distance_m'],2.)
        self.assertEqual(req['facing'],'elevator')
        self.assertEqual(req['semantic_conditions'],['看到门'])
        self.assertEqual(req['visual_preferences'],['看到电梯'])
        self.assertEqual(req['execution_notes'],'核对实际观察')
        self.assertEqual(task.monitor_for(req),dict(interval_s=15.))

    def test_cross_floor_and_monitor_defaults_are_preserved(self):
        second=copy.deepcopy(self.catalog.value);second['floor_id']='synthetic_floor2'
        other=Catalog(second)
        task=TaskGoals(self.catalog,floor_catalogs={self.catalog.value['floor_id']:self.catalog,
                                                    other.value['floor_id']:other})
        goals=[dict(goal_id='a',stage='region',target=dict(id='lobby_area'),
                    floor_id='synthetic_floor1',description='进入等候区'),
               dict(goal_id='b',stage='elevator',floor_id='synthetic_floor1',
                    to_floor='synthetic_floor2',description='乘梯到二楼'),
               dict(goal_id='c',stage='point',target=dict(id='elevator'),
                    floor_id='synthetic_floor2',description='到二楼电梯口')]
        validate_call(dict(tool='record_task',arguments=dict(goals=goals)))
        task.record(goals)
        self.assertEqual([x['requirement']['kind'] for x in task.goals],['region','skill','point'])
        self.assertEqual(task.goals[1]['requirement']['stage_type'],'elevator')
        self.assertEqual(task.monitor_for(task.goals[1]['requirement']),dict(interval_s=5.))
        self.assertEqual(task.monitor_for(dict(stage_type='elevator',monitor={})),{})

    def test_bad_target_is_rejected_and_task_remains_unregistered(self):
        task=TaskGoals(self.catalog)
        goal=dict(goal_id='g',stage='point',description='去电梯',target=dict(id='elevator',position=[5.,4.]))
        with self.assertRaisesRegex(ValueError,'exactly one'):
            task.record([goal])
        self.assertEqual(task.goals,[])

    def test_wait_clause_must_come_from_user_and_is_preserved_in_execution_plan(self):
        task=TaskGoals(self.catalog)
        goal=dict(goal_id='g',stage='point',description='到电梯口',target=dict(id='elevator'),
                  conditions=dict(wait_after='等我通知再乘梯'))
        with self.assertRaisesRegex(ValueError,'wait_after'):
            task.record([goal],user_messages=['到电梯口，再乘梯'])
        self.assertFalse(task.goals)
        task.record([goal],user_messages=['到电梯口，等我通知再乘梯'])
        self.assertEqual(task.execution_plan()[0]['wait_after'],'等我通知再乘梯')

    def test_legacy_fixture_remains_readable(self):
        goal=dict(goal_id='old',kind='point',target_id='elevator',description='到电梯')
        validate_call(dict(tool='record_task',arguments=dict(goals=[goal])))
        task=TaskGoals(self.catalog);task.record([goal])
        self.assertEqual(task.goals[0]['requirement'],goal)

    def test_local_task_plan_includes_required_follow_selection_only_for_follow(self):
        task=TaskGoals(self.catalog)
        task.record([dict(goal_id='f',stage='follow',description='跟随指定人物'),
                     dict(goal_id='s',stage='seating',description='到桌边停靠')])
        follow,seating=task.execution_plan()
        self.assertEqual(follow['execution_tools'],['select_follow_target','start_local_task','complete_subtask'])
        self.assertEqual(seating['execution_tools'],['start_local_task','complete_subtask'])


if __name__=='__main__':
    unittest.main()
