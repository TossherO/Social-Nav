"""Parking substitutions and post-arrival visual recovery in the agent session."""
import json
from pathlib import Path
import tempfile
import unittest

from decision.agent import CONFIG_DIR
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests import test_agent_goals
from tests.test_agent import ScriptedFeedback, query_map


class PositionRecoveryTests(unittest.TestCase):
    permission = '原点被挡时可以在原点附近停车'
    visual_permission = '视觉不足时也可以在原点附近调整停车位置'

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = Store(Path(temp.name)/'session.sqlite')
        self.addCleanup(store.close)
        self.nav = ScriptedFeedback()
        self.session = AgentSession('在电梯口停车。'+self.permission+'，'+self.visual_permission,
                                    Catalog.load(CONFIG_DIR/'simulation.json'),self.nav,store)

    def register(self, *, nearby=False, visual=(), unnamed=False, visual_reselection=False):
        conditions = {}
        if nearby:
            conditions['nearby_stop'] = dict(instruction_quote=self.permission,max_distance_m=.65)
            if visual_reselection:
                conditions['nearby_stop'].update(instruction_quote=self.permission+'，'+self.visual_permission,
                                                 allow_visual_reselection=True)
        if visual:
            conditions['visual'] = [dict(description='看清标注门口',priority=priority) for priority in visual]
        goal = dict(goal_id='stop',stage='point',description='停车',
                    target=dict(position=[5,4]) if unnamed else dict(id='elevator'),conditions=conditions)
        return self.session.call('record_task',goals=[goal])

    def move(self, pose):
        return self.session.call('navigate',goal_id='stop',view_id=query_map(self.session),
                                 pose=pose,orientation_source='semantic')

    def arrive(self):
        self.nav.arrive()
        self.session.tick()
        self.session.tick()
        self.session.observation = test_agent_goals.IndoorGoalTests.rgb(self)

    def review(self, priority='required', result='not_satisfied', observation_id=None):
        return self.session.call('record_monitor_review',goal_id='stop',visual_priority=priority,result=result,
                                 evidence='当前图像只显示部分门口',
                                 observation_id=observation_id or self.session.observation.record['observation_id'],
                                 _context=self.session.context())

    def set_occupancy(self, original_status, candidate_status='free'):
        self.nav.geometry = object()
        self.session.check_candidate = lambda pose,view_id: dict(status='available',
            static_footprint=dict(status=original_status if pose[:2]==[5,4] else candidate_status),
            local_footprint=dict(status='unavailable'))

    def test_original_stop_takes_priority_when_free_or_unknown(self):
        self.assertEqual(self.register(nearby=True)['status'],'recorded')
        for status in ('free','unknown'):
            self.set_occupancy(status)
            self.assertEqual(self.move([5.4,4,0])['reason'],'original_stop_has_priority')
        self.assertIsNone(self.session.task_goals.current('stop')['selected'])
        self.set_occupancy('free')
        self.assertEqual(self.move([5,4,0])['status'],'accepted')

    def test_nearby_stop_requires_permission_and_uses_its_own_arrival_pose(self):
        self.assertEqual(self.register(nearby=True)['status'],'recorded')
        self.set_occupancy('occupied')
        self.assertEqual(self.move([5.7,4,0])['reason'],'outside_authorized_nearby_stop')
        self.set_occupancy('occupied','occupied')
        self.assertEqual(self.move([5.4,4,0])['reason'],'candidate_footprint_occupied')
        self.set_occupancy('occupied','unknown')
        self.assertEqual(self.move([5.4,4,0])['reason'],'replacement_footprint_not_supported_by_map')
        self.set_occupancy('occupied')
        self.assertEqual(self.move([5.4,4,0])['status'],'accepted')
        del self.nav.geometry
        self.arrive()
        self.nav.state['pose']=[5.1,4,0]
        facts = self.session.prompt()['current_state']['completion_evidence']
        self.assertFalse(facts['checks']['selected_position_met'])
        self.nav.state['pose']=[5.4,4,0]
        facts = self.session.prompt()['current_state']['completion_evidence']
        self.assertFalse(facts['missing'])

    def test_exact_unnamed_point_cannot_move_within_arrival_tolerance(self):
        self.assertEqual(self.register(unnamed=True)['status'],'recorded')
        self.set_occupancy('occupied')
        self.assertEqual(self.move([5.1,4,0])['reason'],'use_registered_exact_position')

    def test_parking_permission_must_quote_the_user(self):
        goal = dict(goal_id='stop',stage='point',description='停车',target=dict(id='elevator'),
                    conditions=dict(nearby_stop=dict(instruction_quote='允许任意选择停车位置')))
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'unavailable')
        self.assertFalse(self.session.task_goals.goals)

    def test_visual_adjustments_are_deduplicated_bounded_and_keep_stage(self):
        self.register(visual=('required',))
        first = self.move([5,4,0])
        self.arrive()
        self.nav.t+=5.
        self.nav.state['stamp_s']+=5.  # The model may take time to interpret a valid post-stop frame.
        self.assertEqual(self.move([5,4,.4])['reason'],'post_arrival_visual_review_required')
        self.assertEqual(self.review(observation_id='wrong')['status'],'unproven')
        self.assertEqual(self.review()['remaining'],2)
        self.nav.state['stopped']=False
        self.assertEqual(self.move([5,4,.4])['reason'],'visual_adjustment_requires_current_stop')
        self.nav.state['stopped']=True
        self.nav.state['environment_epoch']=1
        self.assertEqual(self.move([5,4,.4])['reason'],'visual_adjustment_requires_current_stop')
        self.nav.state.pop('environment_epoch')
        self.assertEqual(self.move([5,4,.02])['reason'],'visual_adjustment_pose_already_tried')
        self.assertEqual(self.move([5.4,4,.4])['reason'],'use_registered_annotation_point')
        old_rgb = self.session.observation
        second = self.move([5,4,.4])
        self.assertNotEqual(first['handle'],second['handle'])
        self.assertEqual(first['handle']['task_goal_binding'],second['handle']['task_goal_binding'])
        self.arrive()
        fresh_rgb = self.session.observation
        self.session.observation = old_rgb
        self.assertEqual(self.review()['reason'],'current_post_stop_rgb_required')
        self.session.observation = fresh_rgb
        self.assertEqual(self.review()['remaining'],1)
        self.assertEqual(self.move([5,4,0])['reason'],'visual_adjustment_pose_already_tried')
        self.assertEqual(self.move([5,4,.8])['status'],'accepted')
        self.arrive()
        self.assertEqual(self.review()['remaining'],0)
        self.assertEqual(self.move([5,4,1.2])['reason'],'visual_adjustment_limit_reached')
        current = self.session.prompt()['current_state']
        self.assertEqual(current['next_goal_id'],'stop')
        self.assertEqual(current['selected_goal']['visual_adjustment']['remaining'],0)
        result = self.session.call('complete_subtask',goal_id='stop',evidence='仍然看不到门',
                                   _context=self.session.context())
        self.assertEqual(result['status'],'unproven')

    def test_visual_preference_can_report_partial_result(self):
        self.register(nearby=True,visual=('preferred',),visual_reselection=True)
        self.assertEqual(self.move([5,4,0])['status'],'accepted')
        self.arrive()
        self.assertEqual(self.review('preferred','partly_satisfied')['status'],'recorded')
        self.assertEqual(self.move([5.4,4,.4])['status'],'accepted')
        self.arrive()
        self.assertEqual(self.review('preferred','partly_satisfied')['status'],'recorded')
        self.session.prompt()
        result = self.session.call('complete_subtask',goal_id='stop',evidence='看见部分门口，满足程度有限',
                                   _context=self.session.context())
        self.assertEqual(result['status'],'completed')

    def test_required_partial_is_logged_and_allows_adjustment_but_needs_new_rgb_for_completion(self):
        self.register(visual=('required',))
        self.assertFalse(self.session.prompt()['current_state']['selected_goal']['visual_adjustment']['requires_review'])
        self.assertEqual(self.move([5,4,0])['status'],'accepted')
        self.arrive()
        self.assertTrue(self.session.prompt()['current_state']['selected_goal']['visual_adjustment']['requires_review'])
        outcome = self.session.call('record_monitor_review',goal_id='stop',result='partly_satisfied',
            evidence='看见左门，右门被挡住',_context=self.session.context())
        self.assertEqual(outcome['status'],'recorded')
        self.assertEqual(outcome['visual_priority'],'required')
        self.assertFalse(outcome['condition_met'])
        self.assertFalse(outcome['execution_completed'])
        self.assertIn('navigate',outcome['allowed_next_actions'])
        self.assertNotIn('complete_subtask',outcome['allowed_next_actions'])
        event = self.session.store.db.execute("SELECT value FROM events WHERE kind='visual_review'").fetchone()
        self.assertEqual(json.loads(event[0])['result'],'partly_satisfied')
        current = self.session.prompt()['current_state']
        self.assertIn('new_rgb_after_unsatisfied_visual_review',current['completion_evidence']['missing'])
        completed = self.session.call('complete_subtask',goal_id='stop',evidence='两扇门都可见',
                                      _context=self.session.context())
        self.assertEqual(completed['status'],'unproven')
        scripted = self.session.call('check_goal',goal_id='stop',semantic_result='satisfied',evidence='两扇门都可见',
            review_result='satisfied',review_evidence='两扇门都可见',
            observation_id=self.session.observation.record['observation_id'],_context=self.session.context())
        self.assertEqual(scripted['status'],'needs_semantic_evidence')
        self.assertEqual(self.session.monitor.last_review['result'],'partly_satisfied')
        self.assertEqual(self.move([5,4,.4])['status'],'accepted')
        self.arrive()
        self.session.prompt()
        completed = self.session.call('complete_subtask',goal_id='stop',evidence='两扇门都可见',
                                      _context=self.session.context())
        self.assertEqual(completed['status'],'completed')

    def test_visual_review_checks_priority_and_ignores_superseded_rgb(self):
        self.register(visual=('required','preferred'))
        self.assertEqual(self.move([5,4,0])['status'],'accepted')
        self.arrive()
        ambiguous=self.session.call('record_monitor_review',goal_id='stop',result='partly_satisfied',
            evidence='当前图像只显示部分门口',_context=self.session.context())
        self.assertEqual(ambiguous['reason'],'registered_visual_priority_required')
        self.assertEqual(ambiguous['allowed_next_actions'],['record_monitor_review'])
        old=self.session.observation
        old_context=self.session.context()
        self.session.tick()
        self.session.observation=test_agent_goals.IndoorGoalTests.rgb(self)
        self.assertEqual(self.review('required','partly_satisfied')['status'],'recorded')
        late=self.session.call('record_monitor_review',goal_id='stop',result='unknown',visual_priority='required',
            evidence='迟到判断',observation_id=old.record['observation_id'],_context=old_context)
        self.assertEqual(late['reason'],'current_post_stop_rgb_required')
        self.session.observation=old
        self.assertEqual(self.review('required','unknown')['reason'],'visual_review_rgb_superseded')
        stage=self.session.task_goals.current('stop')
        self.assertEqual(stage['required_visual_review']['result'],'partly_satisfied')

    def test_occupancy_only_permission_preserves_location_on_visual_shortfall(self):
        self.register(nearby=True,visual=('required',))
        self.assertEqual(self.move([5,4,0])['status'],'accepted')
        self.arrive()
        self.assertEqual(self.review()['status'],'recorded')
        self.assertEqual(self.move([5.4,4,.4])['reason'],
                         'nearby_stop_permission_does_not_allow_visual_reselection')
        self.assertEqual(self.move([5,4,.4])['status'],'accepted')

    def test_refused_visual_move_preserves_arrival_and_attempt_budget(self):
        self.register(visual=('required',))
        first = self.move([5,4,0])
        self.arrive()
        self.assertEqual(self.review()['status'],'recorded')
        self.nav.check_route = lambda place: dict(reference_found=False,status='no_path',reason='blocked')
        self.assertEqual(self.move([5,4,.4])['status'],'refused')
        stage = self.session.task_goals.current('stop')
        self.assertEqual(stage['selected']['handle'],first['handle'])
        self.assertEqual(self.session.task_goals.visual_adjustment_state(stage)['remaining'],2)


if __name__ == '__main__':
    unittest.main()
