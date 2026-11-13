"""Ordered navigation and follow stages using explicit simulated live feedback."""
import copy
import tempfile
import unittest
from pathlib import Path

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback, navigate, point_goal
from tests.test_follow_recovery import candidate_observation


def follow_goal(goal_id):
    return dict(goal_id=goal_id, kind='skill', stage_type='follow', target_id='',
                description='跟随目标直到指定地点')


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'session.sqlite')
        catalog = copy.deepcopy(Catalog.load(ROOT/'decision/agent/configs/simulation.json').value)
        catalog['scope'] = 'development_prior'
        for place in catalog['places']:
            place['status'] = 'manual_annotation'
        self.nav = ScriptedFeedback()
        self.nav.scope = 'live'
        self.nav.clock_domain = 'synthetic'
        self.nav.state.update(scope='live', clock_domain='synthetic', session_id='test', generation=0,
            frame_id='odom', localization_status='valid', input_stale=False,
            origin_pose=[4.,4.,0.], origin_frame_id=catalog['frame_id'], origin_stamp_s=10.,
            arrival_observation=False, stop_observation=False, chassis_state='armed', chassis_status_fresh=True)
        original_tick = self.nav.tick
        def tick():
            feedback = original_tick()
            self.nav.state['origin_stamp_s'] = feedback['stamp_s']
            return self.nav.snapshot()
        self.nav.tick = tick
        self.nav.check_route = lambda _: dict(reference_found=True, proposal_allowed=True, status='available')
        def hold(handle):
            self.nav.state.update(status='stopped', stopped=True, arrival_observation=False, stop_observation=True)
        self.nav.hold = hold
        def start_follow(reference, handle):
            self.nav.state.update(status='running', handle=copy.deepcopy(handle), stopped=False, stop_observation=False,
                follow_status=dict(status='running', handle=copy.deepcopy(handle), target_identity=reference['track']['identity']))
            return dict(accepted=True, handle=copy.deepcopy(handle))
        def finish_follow(handle):
            self.nav.state.update(status='stopping', follow_status=dict(status='stopped', handle=copy.deepcopy(handle)))
        self.nav.start_follow = start_follow
        self.nav.finish_follow = finish_follow
        self.session = AgentSession('先导航，再跟随两人，最后导航', Catalog(catalog), self.nav, self.store)

    def tearDown(self):
        self.session.follow_recovery_worker.close()
        self.store.close()
        self.temp.cleanup()

    def observation(self, track_id):
        obs = candidate_observation(self.nav.state['stamp_s'], track_id)
        self.nav.observe = lambda: obs
        self.assertEqual(self.session.call('observe_scene')['status'], 'available')
        return obs

    def finish_navigation(self, goal_id):
        self.assertEqual(navigate(self.session, goal_id)['status'], 'accepted')
        self.nav.arrive()
        self.nav.state.update(arrival_observation=True, origin_pose=copy.deepcopy(self.nav.target_pose))
        self.session.tick(); self.session.tick()
        self.assertEqual(self.session.skill.record['status'], 'held')
        return self.session.call('check_goal', goal_id=goal_id)

    def follow(self, goal_id, track_id):
        obs = self.observation(track_id)
        chosen = self.session.call('select_follow_target', _context=self.session.context(), goal_id=goal_id,
            observation_id=obs.record['observation_id'], candidate_binding=obs.record['follow_candidates']['binding'],
            target_description='当前跟随目标', match='unique', candidate_id=0)
        self.assertEqual(chosen['status'], 'selected')
        self.assertEqual(chosen['reference']['track']['identity'][-1], track_id)
        self.assertEqual(self.session.call('start_follow', goal_id=goal_id)['status'], 'accepted')
        self.session.tick()
        obs = self.observation(track_id)
        self.assertEqual(self.session.call('finish_follow', _context=self.session.context(), goal_id=goal_id,
            observation_id=obs.record['observation_id'], evidence='目标已到达约定地点')['status'], 'stopping')
        self.nav.state.update(status='stopped', stopped=True, stop_observation=True)
        self.session.tick()
        self.nav.tick()
        obs = self.observation(track_id)
        self.nav.state['pose'] = copy.deepcopy(obs.record['base_pose'])
        result = self.session.call('check_goal', _context=self.session.context(), goal_id=goal_id,
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='停稳后位置符合要求')
        self.assertEqual(result['status'], 'completed')
        return chosen['reference']

    def test_navigation_follow_follow_navigation_preserves_separate_evidence(self):
        goals = [point_goal('elevator'), follow_goal('person_a'), follow_goal('person_b'), point_goal('lobby_entrance')]
        self.session.call('record_task', goals=goals)
        self.assertEqual(self.session.call('start_follow', goal_id='person_a')['status'], 'unavailable')
        before = self.session.context()
        self.assertEqual(self.finish_navigation('elevator')['status'], 'completed')
        self.assertNotEqual(before['current_stage'], self.session.context()['current_stage'])
        self.assertFalse(self.session.valid_context(before))
        a = self.follow('person_a', 'a')
        self.assertIsNone(self.session.follow_reference)
        self.assertIsNone(self.session.follow_recovery_reference)
        b = self.follow('person_b', 'b')
        self.assertNotEqual(a['reference_id'], b['reference_id'])
        for goal_id, ref in [('person_a', a), ('person_b', b)]:
            self.assertEqual(self.store.read_follow_reference(self.session.state['task_id'], 1, goal_id)[0], ref)
        self.assertEqual(self.finish_navigation('lobby_entrance')['status'], 'completed')
        self.assertEqual(self.session.call('finish_task', status='succeeded', evidence='四个阶段已核对')['status'], 'succeeded')

    def test_future_skill_cannot_complete_or_be_relabelled_navigation(self):
        goal = dict(goal_id='seat', kind='skill', stage_type='seating', target_id='', description='桌边停靠')
        self.session.call('record_task', goals=[goal, point_goal('elevator')])
        self.assertEqual(self.session.call('check_goal', goal_id='seat')['reason'], 'stage_skill_not_available')
        self.assertEqual(self.session.call('finish_task', status='succeeded', evidence='尚未完成')['status'], 'refused')
        other = AgentSession('invalid', self.session.reference_catalog, self.nav, self.store)
        result = other.call('record_task', goals=[dict(point_goal('elevator'), stage_type='follow')])
        self.assertEqual(result['status'], 'unavailable')
        other.follow_recovery_worker.close()

    def test_clarification_after_stop_and_not_while_moving(self):
        self.session.call('record_task', goals=[point_goal('elevator'), follow_goal('person')])
        self.assertEqual(navigate(self.session, 'elevator')['status'], 'accepted')
        self.assertEqual(self.session.call('ask_user', question='跟随哪位？')['reason'],
                         'cancel_and_confirm_navigation_first')
        self.nav.arrive()
        self.nav.state.update(arrival_observation=True, origin_pose=copy.deepcopy(self.nav.target_pose))
        self.session.tick(); self.session.tick()
        self.assertEqual(self.session.call('check_goal', goal_id='elevator')['status'], 'completed')
        self.nav.state['chassis_status_fresh'] = False
        self.assertEqual(self.session.call('ask_user', question='跟随哪位？')['reason'],
                         'cancel_and_confirm_navigation_first')
        self.nav.state['chassis_status_fresh'] = True
        self.assertEqual(self.session.call('ask_user', question='跟随哪位？')['status'], 'waiting_user_answer')
        self.session.event('answer', '穿红衣的人')
        self.assertEqual(self.session.state['status'], 'ready')
        self.assertEqual(self.session.skill.record['status'], 'held')
        self.assertEqual(self.session.call('wait_for_event', event='user_continue')['status'], 'waiting')
        self.session.event('continue')
        self.assertEqual(self.session.state['status'], 'ready')

    def test_observation_stage_needs_new_image_and_stop_before_next_follow(self):
        observation = dict(goal_id='inspect', kind='skill', stage_type='observation', target_id='',
                           description='观察门是否打开')
        self.session.call('record_task', goals=[point_goal('elevator'), observation, follow_goal('person')])
        self.assertEqual(self.finish_navigation('elevator')['status'], 'completed')
        old = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(old.record['base_pose'])
        result = self.session.call('check_goal', _context=self.session.context(), goal_id='inspect',
            observation_id=old.record['observation_id'], semantic_result='satisfied', evidence='门打开')
        self.assertEqual(result['reason'], 'read_current_rgb_after_stage_entry')
        self.nav.tick()
        current = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(current.record['base_pose'])
        kwargs = dict(goal_id='inspect', observation_id=current.record['observation_id'], evidence='看到门')
        self.assertEqual(self.session.call('check_goal', _context=self.session.context(),
                         semantic_result='unknown', **kwargs)['status'], 'unknown')
        self.assertEqual(self.session.call('check_goal', _context=self.session.context(),
                         semantic_result='satisfied', **kwargs)['status'], 'completed')
        self.assertIsNone(self.session.observation)
        self.assertEqual(self.session.prompt()['current_state']['next_goal_id'], 'person')
        self.assertIsNotNone(self.session.task_goals.goals[1]['completion']['semantic']['observation_binding'])

    def test_initial_observation_cannot_use_unconfirmed_live_stop(self):
        goal = dict(goal_id='inspect', kind='skill', stage_type='observation', target_id='', description='观察门')
        self.session.call('record_task', goals=[goal])
        self.assertEqual(self.session.call('check_goal', goal_id='inspect')['reason'],
                         'observation_requires_confirmed_stop')
        self.nav.state['idle_stop_observation']=True
        self.nav.tick()
        obs=self.observation('a')
        self.nav.state['pose']=copy.deepcopy(obs.record['base_pose'])
        self.assertEqual(self.session.call('check_goal', _context=self.session.context(), goal_id='inspect',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='看到门')['status'],'completed')
        self.assertEqual(self.session.call('finish_task',status='succeeded',evidence='已完成观察')['status'],'succeeded')

    def test_observation_stage_rejects_old_rgb_then_accepts_new_judgement(self):
        goal = dict(goal_id='inspect', kind='skill', stage_type='observation', target_id='', description='观察门')
        self.session.call('record_task', goals=[goal])
        self.nav.state['idle_stop_observation'] = True
        self.nav.tick()
        obs = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(obs.record['base_pose'])
        context = self.session.context()
        # The scene is still younger than 3 seconds, but the RGB itself is not.
        self.nav.state['stamp_s'] = obs.record['rgb']['stamp_s'] + 3.01
        self.assertTrue(self.session.valid_context(context))
        result = self.session.call('check_goal', _context=context, goal_id='inspect',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='门打开')
        self.assertEqual(result['status'], 'needs_semantic_evidence')
        self.assertFalse(self.session.task_goals.finished())
        self.nav.state['stamp_s'] = obs.record['stamp_s'] + 4
        self.assertTrue(self.session.valid_expired_context(context))
        result = self.session.call('check_goal', _context=context, goal_id='inspect',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='门打开')
        self.assertEqual(result['status'], 'needs_semantic_evidence')
        fresh = self.observation('a')
        self.assertEqual(self.session.call('check_goal', _context=self.session.context(), goal_id='inspect',
            observation_id=fresh.record['observation_id'], semantic_result='satisfied', evidence='新图中门打开')['status'],
            'completed')

    def test_follow_finish_and_completion_need_fresh_visual_evidence(self):
        self.session.call('record_task', goals=[follow_goal('person')])
        obs = self.observation('a')
        chosen = self.session.call('select_follow_target', _context=self.session.context(), goal_id='person',
            observation_id=obs.record['observation_id'], candidate_binding=obs.record['follow_candidates']['binding'],
            target_description='目标', match='unique', candidate_id=0)
        self.assertEqual(chosen['status'], 'selected')
        self.assertEqual(self.session.call('start_follow', goal_id='person')['status'], 'accepted')
        self.session.tick()
        obs = self.observation('a')
        context = self.session.context()
        self.nav.t += 4
        self.nav.state['stamp_s'] = 10 + self.nav.t
        self.assertTrue(self.session.valid_expired_context(context))
        self.assertEqual(self.session.call('finish_follow', _context=context, goal_id='person',
            observation_id=obs.record['observation_id'], evidence='目标已到达')['status'], 'refused')
        fresh = self.observation('a')
        self.assertEqual(self.session.call('finish_follow', _context=self.session.context(), goal_id='person',
            observation_id=fresh.record['observation_id'], evidence='目标已到达')['status'], 'stopping')
        self.nav.state.update(status='stopped', stopped=True, stop_observation=True)
        self.session.tick()
        self.nav.tick()
        obs = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(obs.record['base_pose'])
        context = self.session.context()
        self.nav.t += 4
        self.nav.state['stamp_s'] = 10 + self.nav.t
        self.assertTrue(self.session.valid_expired_context(context))
        self.assertEqual(self.session.call('check_goal', _context=context, goal_id='person',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='位置符合要求')['status'],
            'needs_semantic_evidence')
        self.assertFalse(self.session.task_goals.finished())
        fresh = self.observation('a')
        self.assertEqual(self.session.call('check_goal', _context=self.session.context(), goal_id='person',
            observation_id=fresh.record['observation_id'], semantic_result='satisfied', evidence='新图确认位置')['status'],
            'completed')

    def test_agent_tool_loop_completes_idle_observation_from_new_image(self):
        from decision.agent.core.loop import AgentLoop
        from tests.test_agent import reply
        self.nav.state.update(idle_stop_observation=True, stopped=True, pose=[1.33,1.67,0.])
        self.nav.observe=lambda:candidate_observation(self.nav.state['stamp_s'], 'a')
        goal=dict(goal_id='inspect',kind='skill',stage_type='observation',target_id='',description='观察门')
        def work(payload, observation, spatial_view):
            current=payload['current_state']
            if not current['requirements_registered']:
                return reply('record_task',goals=[goal])
            if current['next_goal_id'] is not None:
                if observation is None:
                    return reply('observe_scene')
                return reply('check_goal',goal_id='inspect',observation_id=observation.record['observation_id'],
                             semantic_result='satisfied',evidence='看到门')
            return reply('finish_task',status='succeeded',evidence='观察已完成')
        actor=AgentLoop(self.session,work)
        try:
            for _ in range(16):
                actor.tick()
                if actor.model.thread is not None:
                    actor.model.thread.join(2)
                if self.session.state['status']=='succeeded':break
            self.assertEqual(self.session.state['status'],'succeeded')
            self.assertTrue(self.session.task_goals.finished())
        finally:actor.close()


if __name__ == '__main__':
    unittest.main()
