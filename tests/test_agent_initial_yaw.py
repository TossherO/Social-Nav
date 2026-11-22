"""Point navigation preserves the first-stop and final heading requirements."""
from pathlib import Path
import tempfile
import unittest

from decision.agent import CONFIG_DIR
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests import test_agent_goals
from tests.test_agent import ScriptedFeedback, query_map


class InitialYawTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = Store(Path(temp.name)/'session.sqlite')
        self.addCleanup(store.close)
        self.nav = ScriptedFeedback()
        self.session = AgentSession('电梯口首次停靠朝向 0，按登记要求检查视野',
                                    Catalog.load(CONFIG_DIR/'simulation.json'),self.nav,store)

    def register(self, *, visual=True, **conditions):
        if visual:
            conditions['visual'] = [dict(description='看清门口',priority='required')]
        goal = dict(goal_id='stop',stage='point',description='电梯口停车',
                    target=dict(id='elevator'),conditions=conditions)
        return self.session.call('record_task',goals=[goal])

    def navigate(self, yaw, source='user'):
        args = dict(goal_id='stop',view_id=query_map(self.session),point_id='elevator')
        if yaw is not None:
            args.update(yaw=yaw,orientation_source=source)
        return self.session.call('navigate',**args)

    def arrive(self):
        self.nav.arrive()
        self.session.tick()
        self.session.tick()
        self.session.observation = test_agent_goals.IndoorGoalTests.rgb(self)

    def review(self, result='not_satisfied'):
        return self.session.call('record_monitor_review',goal_id='stop',visual_priority='required',
                                 result=result,evidence='当前画面中门口情况',
                                 observation_id=self.session.observation.record['observation_id'],
                                 _context=self.session.context())

    def current(self):
        return self.session.prompt()['current_state']

    def test_initial_yaw_is_enforced_at_selection_and_measured_arrival(self):
        self.assertEqual(self.register(initial_yaw=0.,visual=False)['status'],'recorded')
        selected = self.current()['selected_goal']
        self.assertEqual(selected['initial_pose'],[5,4,0.])
        self.assertEqual(selected['initial_point_id'],'elevator')
        self.assertEqual(selected['allowed_orientation_sources'],['user'])
        self.assertEqual(self.navigate(None)['reason'],'heading_required_for_current_goal')
        self.assertFalse(self.navigate(.8)['checks']['initial_yaw_met'])
        self.assertEqual(self.navigate(0.,'semantic')['reason'],
                         'orientation_source_disagrees_with_registered_requirement')
        self.assertEqual(self.navigate(0.)['status'],'accepted')
        self.arrive()
        self.nav.state['pose']=[5,4,.8]
        facts = self.current()['completion_evidence']
        self.assertFalse(facts['checks']['initial_yaw_met'])
        self.assertIn('navigation_geometry',facts['missing'])
        self.nav.state['pose']=[5,4,0.]
        self.assertFalse(self.current()['completion_evidence']['missing'])

    def test_only_reviewed_visual_adjustment_releases_initial_yaw(self):
        self.register(initial_yaw=0.)
        first = self.navigate(0.)
        self.arrive()
        self.assertEqual(self.navigate(.8,'semantic')['reason'],'post_arrival_visual_review_required')
        self.assertEqual(self.review()['status'],'recorded')
        self.assertEqual(self.current()['selected_goal']['allowed_orientation_sources'],['semantic','autonomous'])
        self.assertEqual(self.navigate(.8)['reason'],'orientation_source_disagrees_with_registered_requirement')
        second = self.navigate(.8,'semantic')
        self.assertEqual(second['status'],'accepted')
        self.assertEqual(first['handle']['task_goal_binding'],second['handle']['task_goal_binding'])
        self.arrive()
        self.assertEqual(self.current()['completion_evidence']['status'],'available')

    def test_unknown_review_leaves_the_initial_yaw_in_scope(self):
        self.register(initial_yaw=0.)
        first = self.navigate(0.)
        self.arrive()
        self.assertEqual(self.review('unknown')['status'],'recorded')
        self.assertEqual(self.current()['selected_goal']['allowed_orientation_sources'],['user'])
        self.assertEqual(self.navigate(.8,'semantic')['status'],'refused')
        self.assertEqual(self.session.task_goals.current('stop')['selected']['handle'],first['handle'])

    def test_hard_final_yaw_stays_enforced_after_visual_shortfall(self):
        self.register(required_yaw=.8)
        self.assertEqual(self.current()['selected_goal']['initial_pose'],[5,4,.8])
        self.assertEqual(self.navigate(.8)['status'],'accepted')
        self.arrive()
        self.assertEqual(self.review()['status'],'recorded')
        self.assertEqual(self.current()['selected_goal']['allowed_orientation_sources'],['user'])
        self.assertFalse(self.navigate(1.5)['checks']['required_yaw_met'])

    def test_distinct_final_yaw_blocks_completion_but_allows_first_stop_review(self):
        self.register(initial_yaw=0.,required_yaw=.8)
        self.assertEqual(self.navigate(0.)['status'],'accepted')
        self.arrive()
        facts = self.current()['completion_evidence']
        self.assertTrue(facts['checks']['initial_yaw_met'])
        self.assertFalse(facts['checks']['required_yaw_met'])
        self.assertIn('required_yaw',facts['missing'])
        result = self.session.call('complete_subtask',goal_id='stop',evidence='已到达电梯口',
                                   _context=self.session.context())
        self.assertEqual(result['status'],'unproven')
        self.assertEqual(self.session.call('check_goal',goal_id='stop')['status'],'not_met')
        self.assertEqual(self.review()['status'],'recorded')
        self.assertEqual(self.navigate(.8,'semantic')['reason'],
                         'orientation_source_disagrees_with_registered_requirement')
        self.assertEqual(self.navigate(.8)['status'],'accepted')
        self.arrive()
        self.assertTrue(self.current()['completion_evidence']['checks']['required_yaw_met'])

    def test_distinct_first_and_final_yaws_without_visual_adjustment_need_two_stages(self):
        self.assertEqual(self.register(initial_yaw=0.,required_yaw=.8,visual=False)['status'],'unavailable')
        self.assertFalse(self.session.task_goals.goals)


if __name__ == '__main__':
    unittest.main()
