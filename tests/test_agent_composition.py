"""Ordered navigation and follow stages using explicit simulated live feedback."""
import copy
import tempfile
import unittest
from pathlib import Path

from decision.agent import ROOT
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback, navigate, point_goal, reply
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
        def start_follow(intent, handle):
            self.nav.state.update(status='running', handle=copy.deepcopy(handle), stopped=False, stop_observation=False,
                follow_status=dict(status='running', handle=copy.deepcopy(handle), target_identity=['tracker_session',7,'a'],
                                   target_generation=7))
            return dict(accepted=True, handle=copy.deepcopy(handle))
        def finish_follow(handle):
            self.nav.state.update(status='stopping', follow_status=dict(status='stopped', handle=copy.deepcopy(handle)))
        self.nav.start_follow = start_follow
        self.nav.finish_follow = finish_follow
        self.session = AgentSession('先导航，再跟随两人，最后导航', Catalog(catalog), self.nav, self.store)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def observation(self, track_id):
        obs = candidate_observation(self.nav.state['stamp_s'], track_id)
        self.nav.observe = lambda: obs
        self.assertEqual(self.session.call('observe_scene')['status'], 'available')
        return obs

    def select_follow(self,description='红衣人物'):
        obs=self.observation('a')
        result=self.session.call('select_follow_target',_context=self.session.context(),goal_id='person',
            observation_id=obs.record['observation_id'],candidate_id=0,target_description=description)
        self.assertEqual(result['status'],'selected',result)
        return result['reference_id']

    def finish_navigation(self, goal_id):
        self.assertEqual(navigate(self.session, goal_id)['status'], 'accepted')
        self.nav.arrive()
        self.nav.state.update(arrival_observation=True, origin_pose=copy.deepcopy(self.nav.target_pose))
        self.session.tick(); self.session.tick()
        self.assertEqual(self.session.skill.record['status'], 'held')
        return self.session.call('check_goal', goal_id=goal_id)

    def test_follow_stage_uses_confirmed_reference_without_repeating_description(self):
        self.session.call('record_task', goals=[point_goal('elevator'), follow_goal('person')])
        self.assertEqual(self.finish_navigation('elevator')['status'], 'completed')
        ref=self.select_follow()
        sent=[]
        start=self.nav.start_follow
        self.nav.start_follow=lambda intent,handle: (sent.append(copy.deepcopy(intent)),start(intent,handle))[1]
        result=self.session.call('start_local_task',goal_id='person',task_type='follow',reference_id=ref)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(sent[0]['target_description'],'红衣人物')
        self.assertEqual(self.session.monitor.settings['interval_s'],15.)
        self.assertEqual(self.session.skill.record['handle'],result['handle'])
        self.assertEqual(self.session.call('get_navigation_state')['stage_type'],'follow')
        self.assertFalse(self.session.task_goals.finished())

    def test_follow_ignores_rephrased_description_and_uses_confirmed_person(self):
        self.session.call('record_task',goals=[follow_goal('person')])
        ref=self.select_follow('背对相机、穿红衣的人')
        sent=[]
        start=self.nav.start_follow
        self.nav.start_follow=lambda intent,handle: (sent.append(copy.deepcopy(intent)),start(intent,handle))[1]
        result=self.session.call('start_local_task',goal_id='person',task_type='follow',
            target_description='红衣行人',reference_id=ref)
        self.assertEqual(result['status'],'accepted')
        self.assertEqual(sent[0]['target_description'],'背对相机、穿红衣的人')
        self.assertEqual(sent[0]['reference']['reference_id'],ref)

    def test_model_cannot_restart_active_follow_during_search_or_after_recovery(self):
        self.session.call('record_task',goals=[follow_goal('person')])
        ref=self.select_follow()
        started=self.session.call('start_local_task',goal_id='person',task_type='follow',reference_id=ref)
        for phase in ('searching','running'):
            self.session.skill.follow_phase=phase
            self.assertEqual(self.session.navigation_summary()['follow_phase'],phase)
            result=self.session.call('cancel_navigation',_context=self.session.context())
            self.assertEqual(result['reason'],'local_follow_owns_recovery')
            result=self.session.call('start_local_task',_context=self.session.context(),
                                     goal_id='person',task_type='follow',reference_id=ref)
            self.assertEqual(result['reason'],'local_follow_owns_recovery')
            self.assertEqual(self.session.skill.record['handle'],started['handle'])
        self.assertEqual(self.session.call('cancel_navigation')['status'],'stopping')

    def test_follow_rejects_wrong_or_modified_reference(self):
        self.session.call('record_task',goals=[follow_goal('person')])
        ref=self.select_follow()
        for kwargs in ({}, {'reference_id':'wrong'}):
            result=self.session.call('start_local_task',goal_id='person',task_type='follow',**kwargs)
            self.assertEqual(result['reason'],'agent_confirmed_follow_reference_required')
        self.session.follow_reference['description']='被修改的描述'
        result=self.session.call('start_local_task',goal_id='person',task_type='follow',reference_id=ref)
        self.assertEqual(result['reason'],'agent_confirmed_follow_reference_required')
        self.assertIsNone(self.session.skill.record['handle'])

    def test_follow_region_review_waits_for_new_observation_and_rechecks_completion(self):
        from decision.agent.tools.navigation import spatial_context
        self.session.catalog.value['frame_id']='map'
        first=candidate_observation(10.,mapped=True)
        self.nav.state.update(origin_frame_id='map',map_binding=first.record['map_binding'])
        self.session.skill.context=spatial_context(self.nav.state)
        goal=dict(follow_goal('person'),robot_inside_region_id='lobby_area',
            monitor=dict(continuous_region_id='lobby_area',review_mode='after_result',focus='查看举手'))
        self.session.call('record_task',goals=[goal])
        self.nav.observe=lambda:first
        self.assertEqual(self.session.call('observe_scene')['status'],'available')
        selected=self.session.call('select_follow_target',_context=self.session.context(),goal_id='person',
            observation_id=first.record['observation_id'],candidate_id=0,target_description='红衣')
        self.assertEqual(selected['status'],'selected')
        result=self.session.start_local_task('person','follow','红衣',reference_id=selected['reference_id'])
        self.assertEqual(result['status'],'accepted')
        self.session.tick()
        self.session.wait_for_event('navigation')
        self.nav.state['origin_pose']=[7.,4.,0.]
        current=candidate_observation(self.nav.state['stamp_s'],mapped=True)
        self.nav.observe=lambda:current
        self.session.tick()
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.monitor.review_phase,'reviewing')
        self.session.prompt()
        review=self.session.call('record_monitor_review',_context=self.session.context(),goal_id='person',
            result='not_satisfied',evidence='尚未举手')
        self.assertEqual(review['status'],'recorded',review)
        self.assertEqual(self.session.state['status'],'waiting_monitor')
        self.session.tick()
        self.assertEqual(self.session.state['status'],'waiting_monitor')
        current=candidate_observation(self.nav.state['stamp_s'],mapped=True)
        self.nav.observe=lambda:current
        self.session.tick()
        self.assertEqual(self.session.state['status'],'ready')
        facts=self.session.prompt()['current_state']['completion_evidence']
        self.assertEqual(facts['completion_region']['status'],'inside')
        self.assertEqual(facts['missing'],[])
        context=self.session.context()
        self.nav.state['origin_pose']=[4.,4.,0.]
        # Completion rechecks current position even before the next monitor tick.
        outcome=self.session.call('complete_subtask',_context=context,goal_id='person',evidence='行人举手')
        self.assertEqual(outcome['status'],'unproven')
        self.assertIn('completion_region_outside',outcome['missing'])
        self.assertEqual(self.session.skill.record['status'],'running')
        self.session.tick()
        self.assertFalse(self.session.valid_context(context))
        self.session.wait_for_event('navigation')
        self.nav.state['origin_pose']=[7.,4.,0.]
        self.session.tick()
        current=candidate_observation(self.nav.state['stamp_s'],mapped=True)
        self.nav.observe=lambda:current
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.session.prompt()['current_state']['completion_evidence']['missing'],[])
        outcome=self.session.call('complete_subtask',_context=self.session.context(),goal_id='person',evidence='目标停下举手')
        self.assertEqual(outcome['status'],'stopping')
        self.nav.state.update(status='stopped',stopped=True,stop_observation=True)
        self.session.tick()
        self.assertEqual(self.session.state['status'],'succeeded')
        proof=self.session.task_goals.goals[0]['completion']['semantic']
        self.assertEqual(proof['completion_region']['status'],'inside')

    def test_local_task_reuses_cancellation_and_waits_for_stop(self):
        self.session.call('record_task',goals=[follow_goal('person')])
        ref=self.select_follow()
        result=self.session.call('start_local_task',goal_id='person',task_type='follow',target_description='红衣人物',reference_id=ref)
        self.assertEqual(result['status'],'accepted')
        cancelled=self.session.call('cancel_navigation')
        self.assertEqual(cancelled['status'],'stopping')
        self.assertEqual(self.nav.state['status'],'stopping')
        self.assertFalse(self.session.task_goals.finished())
        self.assertIsNotNone(self.session.skill.record['handle'])

    def test_agent_loop_waits_after_local_task_is_accepted(self):
        from decision.agent.core.loop import AgentLoop
        self.session.call('record_task',goals=[follow_goal('person')])
        ref=self.select_follow()
        actor=AgentLoop(self.session,lambda *_:reply('start_local_task',goal_id='person',
                                                     task_type='follow',target_description='红衣人物',reference_id=ref))
        try:
            actor.tick()
            actor.model.thread.join(2)
            actor.tick()
            self.assertEqual(self.session.state['status'],'waiting_navigation')
            self.assertIsNotNone(self.session.skill.record['handle'])
            self.assertFalse(self.session.task_goals.finished())
        finally:actor.close()

    def test_table_selection_arrival_and_single_agent_completion(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from decision.agent.tools.observations import Observation
        from decision.agent.tools.navigation import spatial_context
        from navigation.planning.footprint import VehicleGeometry
        from perception.environment.layers import load_config
        from perception.environment.scene import assemble_scene
        from tests.navigation_fixtures import scene_at

        frame = self.session.catalog.value['frame_id']
        self.nav.geometry = VehicleGeometry([[-.15,-.15],[.2,-.15],[.2,.15],[-.15,.15]],
                                           np.eye(4), source='synthetic', assumed=False)
        def observe_table():
            stamp = self.nav.state['stamp_s']
            grid = scene_at(stamp).geometry
            grid.metadata['association']['map_frame'] = frame
            scene = assemble_scene(grid, clock_domain='synthetic', settings=load_config()['scene'])
            return Observation.from_scene(scene, geometry=self.nav.geometry, rgb=dict(
                pixels=np.full((30,40,3), 180, np.uint8), stamp_s=stamp, frame_id='camera'))
        self.nav.observe = observe_table
        binding = observe_table().record['map_binding']
        self.nav.state.update(map_binding=binding, idle_stop_observation=True)
        self.nav.value = {}
        self.nav.client = SimpleNamespace(exchange=lambda _, request:
                                          dict(accepted=True, selection_id=request['selection_id']))
        self.session.skill.context = spatial_context(self.nav.state)
        self.session.record_task([dict(goal_id='seat', kind='skill', stage_type='seating', target_id='',
                                       description='面向桌沿停靠')])
        self.nav.tick()
        obs = self.nav.observe()
        self.assertEqual(self.session.observe_scene()['status'], 'available')
        self.nav.state['pose'] = obs.record['base_pose']
        context = self.session.context()
        result = self.session.call('start_local_task', _context=context, goal_id='seat', task_type='seating',
                                   target_description='面前桌沿', table_roi=[.1,.2,.8,.6])
        self.assertEqual(result['status'], 'selecting', result)
        self.nav.value['seating'] = dict(status='selected', selection_id=result['selection_id'],
            observation_binding=context['observation_binding'], local_identity=['test',0,'odom'],
            clock_domain='synthetic', map_binding=binding, geometry_stamp_s=self.nav.state['stamp_s'],
            candidate_id=1, candidate_binding='table_view', source_stamp_s=obs.record['rgb']['stamp_s'],
            pose=[4.,4.,0.], mapped_edge=dict(endpoints_xy_m=[[4.4,3.],[4.4,5.]], interior_yaw_rad=0.))
        with patch.object(self.session, 'check_candidate', return_value=dict(status='available',
                          static_footprint=dict(status='free'), local_footprint=dict(status='free'))):
            self.session.tick()
        self.assertEqual(self.session.state['status'], 'waiting_navigation')
        self.assertFalse(self.session.task_goals.finished())
        self.nav.arrive()
        self.nav.state.update(arrival_observation=True, origin_pose=self.nav.target_pose)
        self.session.tick()
        self.session.tick()
        self.assertEqual(self.session.skill.record['status'], 'held')
        self.nav.tick()
        stopped = self.nav.observe()
        self.assertEqual(self.session.observe_scene()['status'], 'available')
        self.nav.state['pose'] = stopped.record['base_pose']
        facts = self.session.prompt()['current_state']['completion_evidence']
        self.assertEqual(facts['missing'], [])
        self.assertAlmostEqual(facts['table_selection']['measured_front_gap_m'], .2)
        self.session.call('complete_subtask', _context=self.session.context(), goal_id='seat',
                          evidence='实测车头距桌沿二十厘米，图像中仍为目标桌')
        self.assertEqual(self.session.state['status'], 'succeeded')

    def test_unavailable_skill_cannot_complete_or_be_relabelled_navigation(self):
        goal = dict(goal_id='seat', kind='skill', stage_type='seating', target_id='', description='桌边停靠')
        self.session.call('record_task', goals=[goal, point_goal('elevator')])
        self.assertEqual(self.session.call('start_local_task',goal_id='seat',task_type='seating')['reason'],
                         'seating_description_required')
        self.assertEqual(self.session.call('start_local_task',goal_id='seat',task_type='seating',
            target_description='桌边')['reason'],'stage_skill_not_available')
        self.assertEqual(self.session.call('start_local_task',goal_id='seat',task_type='follow',
            target_description='人物')['reason'],'local_task_type_does_not_match_stage')
        self.assertEqual(self.session.call('finish_local_task',goal_id='seat',observation_id='obs',
            evidence='桌边')['reason'],'stage_skill_not_available')
        self.assertIsNone(self.session.skill.record['handle'])
        self.assertEqual(self.session.call('check_goal', goal_id='seat')['reason'], 'stage_skill_not_available')
        self.assertEqual(self.session.call('finish_task', status='succeeded', evidence='尚未完成')['status'], 'refused')
        other = AgentSession('invalid', self.session.reference_catalog, self.nav, self.store)
        result = other.call('record_task', goals=[dict(point_goal('elevator'), stage_type='follow')])
        self.assertEqual(result['status'], 'unavailable')

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
        self.session.event('answer', '穿红衣的人，先等我通知')
        self.assertEqual(self.session.state['status'], 'ready')
        self.assertEqual(self.session.skill.record['status'], 'held')
        self.assertEqual(self.session.call('wait_for_event', event='user_continue',instruction_quote='先等我通知')['status'], 'waiting')
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

    def test_observation_stage_preserves_source_time_for_delayed_judgement(self):
        goal = dict(goal_id='inspect', kind='skill', stage_type='observation', target_id='', description='观察门')
        self.session.call('record_task', goals=[goal])
        self.nav.state['idle_stop_observation'] = True
        self.nav.tick()
        obs = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(obs.record['base_pose'])
        context = self.session.context()
        self.nav.state['stamp_s'] = obs.record['rgb']['stamp_s'] + 90.
        self.assertTrue(self.session.valid_context(context))
        result = self.session.call('check_goal', _context=context, goal_id='inspect',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='门打开')
        self.assertEqual(result['status'], 'completed')
        semantic=self.session.task_goals.goals[0]['completion']['semantic']
        self.assertAlmostEqual(semantic['observation_age_s'],90.)
        self.assertFalse(semantic['current_scene_verified'])
        self.assertEqual(semantic['source_stamp_s'],obs.record['rgb']['stamp_s'])



    def test_follow_finish_and_completion_keep_delayed_visual_source(self):
        self.session.call('record_task', goals=[follow_goal('person')])
        obs = self.observation('a')
        from decision.vlm.follow_target import reference
        ref,_,_=reference(obs,0,'目标',task_id=self.session.state['task_id'],revision=1,goal_id='person')
        self.session.follow_reference=ref
        requirement=self.session.task_goals.current('person')['requirement']
        started=self.session.skill.start_follow('person',ref,1,requirement)
        self.assertEqual(started['status'],'accepted')
        self.session.task_goals.current('person')['selected']=dict(handle=copy.deepcopy(started['handle']),
            follow_reference_id=ref['reference_id'],pose=None,orientation_source=None,view_id=None)
        self.session.tick()
        other=self.observation('unrelated')
        self.assertEqual(self.session.call('finish_local_task',_context=self.session.context(),goal_id='person',
            observation_id=other.record['observation_id'],evidence='other person')['reason'],
            'follow_target_not_identified_in_observed_rgb')
        obs = self.observation('a')
        self.nav.state['follow_status']['target_generation']=8
        self.assertEqual(self.session.call('finish_local_task',_context=self.session.context(),goal_id='person',
            observation_id=obs.record['observation_id'],evidence='reused track ID')['reason'],
            'follow_target_not_identified_in_observed_rgb')
        self.nav.state['follow_status']['target_generation']=7
        self.assertEqual(self.session.call('finish_local_task',_context=self.session.context(),goal_id='person')['status'],'refused')
        context = self.session.context()
        self.nav.t += 4
        self.nav.state['stamp_s'] = 10 + self.nav.t
        self.assertTrue(self.session.valid_context(context))
        self.assertEqual(self.session.call('finish_local_task', _context=context, goal_id='person',
            observation_id=obs.record['observation_id'], evidence='目标已到达')['status'], 'stopping')
        self.nav.state.update(status='stopped', stopped=True, stop_observation=True)
        self.session.tick()
        self.nav.tick()
        obs = self.observation('a')
        self.nav.state['pose'] = copy.deepcopy(obs.record['base_pose'])
        context = self.session.context()
        self.nav.t += 4
        self.nav.state['stamp_s'] = 10 + self.nav.t
        self.assertTrue(self.session.valid_context(context))
        result=self.session.call('check_goal', _context=context, goal_id='person',
            observation_id=obs.record['observation_id'], semantic_result='satisfied', evidence='位置符合要求')
        self.assertEqual(result['status'],'completed')
        semantic=self.session.task_goals.goals[0]['completion']['semantic']['after_stop']
        self.assertGreater(semantic['observation_age_s'],3.)
        self.assertFalse(semantic['current_scene_verified'])

    def test_single_follow_verdict_survives_disappearance_until_stop(self):
        self.session.record_task([follow_goal('person')])
        ref=self.select_follow('红衣')
        self.session.start_local_task('person','follow','红衣',reference_id=ref)
        self.session.tick()
        obs=self.observation('a')
        state=self.session.prompt()['current_state']
        self.assertEqual(state['completion_evidence']['missing'],[])
        result=self.session.call('complete_subtask',_context=self.session.context(),goal_id='person',evidence='目标举手')
        self.assertEqual(result['status'],'stopping')
        self.assertFalse(self.session.task_goals.finished())
        self.nav.observe=lambda:None
        self.nav.state.update(status='stopped',stopped=True,stop_observation=True)
        self.session.tick()
        self.assertEqual(self.session.state['status'],'succeeded')
        proof=self.session.task_goals.goals[0]['completion']['proof']
        self.assertEqual(proof['semantic']['observation_id'],obs.record['observation_id'])
        self.assertGreater(proof['stop_stamp_s'],proof['semantic']['source_stamp_s'])

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
                return reply('complete_subtask',goal_id='inspect',evidence='看到门')
            self.fail('Unexpected second completion request')
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
