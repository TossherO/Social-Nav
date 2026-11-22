"""Script the model's two route choices while exercising the real Agent tool lifecycle."""
import tempfile
from threading import Event
import unittest
from pathlib import Path
from unittest.mock import patch

from decision.agent import CONFIG_DIR
from decision.agent.core.completion import stage_evidence
from decision.agent.core.loop import AgentLoop
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from decision.agent.ui_data import goal_requirements
from tests.test_agent import query_map, reply
from tests.test_agent_route import RouteFeedback


class ExceptionSequenceTests(unittest.TestCase):
    nearby_permission='原点被挡时允许在附近停车'

    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store=Store(Path(temp.name)/'session.sqlite')
        self.addCleanup(store.close)
        self.nav=RouteFeedback()
        self.nav.state['localization_status']='valid'
        self.session=AgentSession('经一个允许的入口去终点。'+self.nearby_permission,
                                  Catalog.load(CONFIG_DIR/'simulation.json'),self.nav,store)
        self.goal=dict(goal_id='trip',stage='sequence',description='经任一允许的入口到指定终点',
                       target=dict(checkpoints=[dict(any_of=['elevator','lobby_entrance']),'corridor_wait']))

    def submit(self,entrance,end='corridor_wait'):
        points=[self.session.catalog.places[name]['pose'][:2] for name in (entrance,end)]
        return self.session.call('navigate_route',goal_id='trip',orientation_source='autonomous',
                                 waypoints=points,view_id=query_map(self.session))

    def planning_sample(self,stamp,status, *, kind='route_blocked', position=None):
        self.nav.t=stamp-10.
        self.nav.state.update(stamp_s=stamp,route_index=0,planning_observation=dict(
            source='reference',status=status,kind=kind if status=='blocked' else None,
            goal_binding='first-leg',reason='goal_occupied' if status=='blocked' else None,
            source_stamp_s=stamp,valid_until_s=stamp+5,scene_binding=str(stamp),
            blocked_position=position or self.session.catalog.places['elevator']['pose'][:2]))
        self.session.tick()

    def confirmed_route_blockage(self):
        self.session.call('record_task',goals=[self.goal])
        first=self.submit('elevator')
        self.assertEqual(first['status'],'accepted')
        self.planning_sample(11.,'blocked')
        self.planning_sample(14.1,'blocked')
        return first['handle']

    def delayed_reply(self, tool='cancel_navigation', *, operator_pause=False, **arguments):
        started, release = Event(), Event()
        self.addCleanup(release.set)

        def work(*_):
            started.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release the model reply')
            return reply(tool, **arguments)

        actor=AgentLoop(self.session,work)
        actor.tick()
        self.assertTrue(started.wait(5))
        actor.before_request=lambda _:False

        def consume():
            if operator_pause:
                actor.event('pause')
            release.set()
            actor.model.thread.join(5)
            self.assertFalse(actor.model.thread.is_alive())
            actor.tick()

        return consume

    def test_delayed_cancel_coalesces_unknown_and_reconfirmation_of_the_same_blockage(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            handle=self.confirmed_route_blockage()
            consume=self.delayed_reply()
            revision=self.session.conversation.revision
            self.planning_sample(14.2,'unknown')
            self.planning_sample(14.3,'blocked')
            self.planning_sample(17.4,'blocked')
            self.assertEqual(self.session.conversation.revision,revision)
            consume()
            self.assertEqual(self.session.skill.record['status'],'stopping')
            stage=self.session.task_goals.current('trip')
            self.assertEqual(stage['recovery_blockage']['handle'],handle)
            self.assertTrue(stage['recovery_blockage']['current'])
            self.assertEqual(stage['recovery_blockage']['source_stamp_s'],17.4)

    def test_confirmed_route_blockage_survives_pending_reference_until_withdrawal(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            handle=self.confirmed_route_blockage()
            consume=self.delayed_reply()
            self.planning_sample(14.2,'unknown')
            consume()
            stage=self.session.task_goals.current('trip')
            self.assertEqual(self.session.skill.record['status'],'stopping')
            evidence=stage['recovery_blockage']
            self.assertFalse(evidence['current'])
            self.assertEqual(evidence['source_stamp_s'],14.1)
            self.assertEqual(evidence['first_source_stamp_s'],11.)
            self.assertEqual(evidence['observation_count'],2)
            self.nav.state.update(status='stopped',stopped=True,stamp_s=14.3,stop_measurement_stamp_s=14.2)
            self.session.tick()
            self.assertEqual(self.session.navigation_replacement_gate(stage)['reason'],
                             'awaiting_confirmed_stop_after_blockage')
            self.nav.state.update(stamp_s=14.4,stop_measurement_stamp_s=14.4)
            self.session.tick()
            self.assertEqual(self.session.skill.record['status'],'cancelled')
            self.assertEqual(stage['route_attempts'][0]['blocked_position'],evidence['blocked_position'])
            self.assertEqual(self.submit('elevator')['reason'],'route_replacement_repeats_blocked_segment')
            self.assertEqual(self.submit('lobby_entrance',end='elevator')['reason'],'route_replacement_changed_final_position')
            with patch.dict(self.nav.state,environment_epoch=1):
                self.assertEqual(self.submit('lobby_entrance')['reason'],'recovery_navigation_context_changed')
            detour=self.submit('lobby_entrance')
            self.assertEqual(detour['status'],'accepted',detour)
            self.assertNotEqual(detour['handle'],handle)

    def test_pending_reference_without_sustained_blockage_cannot_authorize_detour(self):
        self.session.call('record_task',goals=[self.goal])
        self.assertEqual(self.submit('elevator')['status'],'accepted')
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.planning_sample(11.,'blocked')
            self.planning_sample(11.1,'unknown')
            self.session.cancel_navigation()
            self.nav.state.update(status='stopped',stopped=True,stamp_s=11.2,stop_measurement_stamp_s=11.2)
            self.session.tick()
            self.assertEqual(self.submit('lobby_entrance')['reason'],'route_replacement_requires_confirmed_blockage')

    def test_delayed_cancel_is_discarded_after_clear_and_new_blockage(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            old=self.confirmed_route_blockage()
            consume=self.delayed_reply()
            self.planning_sample(14.2,'clear')
            self.planning_sample(14.3,'blocked')
            self.planning_sample(17.4,'blocked')
            consume()
            self.assertEqual(self.nav.state['handle'],old)
            self.assertEqual(self.nav.state['status'],'running')
            self.assertNotIn('recovery_blockage',self.session.task_goals.current('trip'))

    def test_delayed_cancel_is_discarded_when_blocked_position_changes(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.confirmed_route_blockage()
            consume=self.delayed_reply()
            self.planning_sample(14.2,'blocked',position=[5.2,4.])
            consume()
            self.assertEqual(self.nav.state['status'],'running')
            self.assertNotIn('recovery_blockage',self.session.task_goals.current('trip'))

    def test_operator_pause_discards_delayed_model_cancel_without_an_extra_withdrawal(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.confirmed_route_blockage()
            consume=self.delayed_reply(operator_pause=True)
            self.planning_sample(14.2,'unknown')
            with patch.object(self.nav,'cancel',wraps=self.nav.cancel) as cancel:
                consume()
                self.assertEqual(cancel.call_count,1)
            self.assertEqual(self.session.skill.record['reason'],'pause')
            self.assertNotIn('recovery_blockage',self.session.task_goals.current('trip'))

    def test_delayed_cancel_checks_the_active_handle_segment_and_map_in_latest_feedback(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            handle=self.confirmed_route_blockage()
            context=self.session.context()
            self.assertTrue(self.session.valid_context(context,tool='cancel_navigation'))
            for key, value in (('handle',dict(handle,call_id='replacement')),('route_index',1),
                               ('map_binding',dict(valid=True,map_id='different-map'))):
                with self.subTest(key=key), patch.dict(self.nav.state,{key:value}):
                    self.assertFalse(self.session.valid_context(context,tool='cancel_navigation'))

    def test_reply_requested_during_unknown_does_not_survive_reconfirmation(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.confirmed_route_blockage()
            self.planning_sample(14.2,'unknown')
            context=self.session.context()
            self.planning_sample(14.3,'blocked')
            self.planning_sample(17.4,'blocked')
            self.assertFalse(self.session.valid_context(context,tool='cancel_navigation'))

    def test_delayed_non_cancel_remains_invalid_while_evidence_is_unknown(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.confirmed_route_blockage()
            consume=self.delayed_reply('wait_for_event',event='navigation')
            self.planning_sample(14.2,'unknown')
            consume()
            self.assertNotEqual(self.session.state['status'],'waiting_navigation')
            self.assertEqual(self.nav.state['status'],'running')

    def test_waiting_for_navigation_wakes_once_when_the_same_blockage_is_reconfirmed(self):
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.confirmed_route_blockage()
            self.planning_sample(14.2,'unknown')
            self.session.call('wait_for_event',event='navigation')
            self.planning_sample(14.3,'blocked')
            self.assertEqual(self.session.state['status'],'waiting_navigation')
            self.planning_sample(17.4,'blocked')
            self.assertEqual(self.session.state['status'],'ready')
            revision=self.session.conversation.revision
            self.planning_sample(17.5,'blocked')
            self.assertEqual(self.session.conversation.revision,revision)

    def test_delayed_cancel_with_unknown_destination_requires_stop_and_new_occupancy_proof(self):
        goal=dict(goal_id='stop',stage='point',description='在电梯口停车',target=dict(id='elevator'),
                  conditions=dict(nearby_stop=dict(instruction_quote=self.nearby_permission,max_distance_m=.65)))
        self.assertEqual(self.session.call('record_task',goals=[goal])['status'],'recorded')
        original=self.session.catalog.places['elevator']['pose'][:2]
        nearby=[original[0]+.4,original[1]]

        def move(pose):
            return self.session.call('navigate',goal_id='stop',pose=pose,view_id=query_map(self.session))

        with patch.object(self.nav,'tick',self.nav.snapshot):
            old=move(original)['handle']
            self.planning_sample(11.,'blocked',kind='destination_occupied')
            self.planning_sample(14.1,'blocked',kind='destination_occupied')
            pre_cancel_view=query_map(self.session)
            consume=self.delayed_reply()
            self.planning_sample(14.2,'unknown')
            consume()
            stage=self.session.task_goals.current('stop')
            self.assertEqual(stage['recovery_blockage']['handle'],old)
            self.assertFalse(stage['recovery_blockage']['current'])
            self.nav.state.update(status='stopped',stopped=True,stamp_s=14.3,stop_measurement_stamp_s=14.2)
            self.session.tick()
            self.assertEqual(self.session.navigation_replacement_gate(stage)['reason'],
                             'awaiting_confirmed_stop_after_blockage')
            self.nav.state.update(stamp_s=14.4,stop_measurement_stamp_s=14.4)
            self.session.tick()
        self.assertEqual(self.session.skill.record['recovery_stop']['handle'],old)
        self.nav.geometry=object()
        local_at_original='unknown'

        def candidate(pose, _):
            return dict(status='available',static_footprint=dict(status='free'),
                        local_footprint=dict(status=local_at_original if pose[:2]==original else 'free'))

        with patch.object(self.session,'check_candidate',candidate):
            local_at_original='occupied'
            refusal=self.session.call('navigate',goal_id='stop',pose=nearby,view_id=pre_cancel_view)
            self.assertEqual(refusal['reason'],'fresh_occupancy_evidence_required_after_cancel')
            local_at_original='unknown'
            self.assertEqual(move(nearby)['reason'],'original_stop_has_priority')
            local_at_original='free'
            self.assertEqual(move(nearby)['reason'],'original_stop_has_priority')
            local_at_original='occupied'
            result=move(nearby)
            self.assertEqual(result['status'],'accepted')
            self.assertNotEqual(result['handle'],old)

    def test_one_accepted_route_then_a_new_route_after_withdrawal_and_measured_stop(self):
        self.assertEqual(self.session.call('record_task',goals=[self.goal])['status'],'recorded')
        stage=self.session.task_goals.current('trip')
        self.assertIn('门1 或 门2',goal_requirements(stage,{'elevator':'门1','lobby_entrance':'门2'}))
        first=self.submit('elevator')
        self.assertEqual(first['status'],'accepted')
        self.assertEqual(len(self.session.skill.record['attempts']),1)
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.planning_sample(11.,'blocked')
            self.session.call('wait_for_event',event='observation')
            self.planning_sample(14.1,'blocked')
            self.assertEqual(self.session.state['status'],'ready')
            self.assertEqual(self.session.navigation_summary()['blockage']['kind'],'route_blocked')
            self.assertEqual(self.session.cancel_navigation()['status'],'stopping')
            self.assertEqual(self.session.navigation_replacement_gate(stage)['reason'],'awaiting_confirmed_stop_after_blockage')
            self.nav.state.update(status='stopped',stopped=True,stamp_s=14.4,stop_measurement_stamp_s=14.4)
            self.nav.t=4.4
            self.session.tick()
        self.assertEqual(self.session.state['status'],'ready')
        self.assertEqual(self.submit('lobby_entrance',end='elevator')['reason'],'route_replacement_changed_final_position')
        self.assertEqual(self.submit('elevator')['reason'],'route_replacement_repeats_blocked_segment')
        second=self.submit('lobby_entrance')
        self.assertEqual(second['status'],'accepted')
        self.assertNotEqual(first['handle'],second['handle'])
        self.assertEqual(stage['route_attempts'][0]['handle'],first['handle'])
        self.nav.state.pop('planning_observation')
        for _ in range(3):
            self.session.tick()
        facts,proof=stage_evidence(self.session,stage)
        self.assertFalse(facts['missing'],facts)
        self.assertTrue(all(visit['handle']==second['handle'] for visit in proof['route_visits']))
        self.assertEqual(stage['status'],'pending')  # Evidence is available for the Agent's decision.
        self.session.prompt()
        result=self.session.call('complete_subtask',goal_id='trip',evidence='已实测经过另一入口，在原终点停稳。',
                                 _context=self.session.context())
        self.assertEqual(result['task_status'],'succeeded')

    def test_cleared_obstruction_invalidates_inflight_decision_and_cannot_authorize_detour(self):
        self.session.call('record_task',goals=[self.goal])
        self.submit('elevator')
        with patch.object(self.nav,'tick',self.nav.snapshot):
            self.planning_sample(11.,'blocked')
            self.planning_sample(14.1,'blocked')
            context=self.session.context()
            self.planning_sample(14.2,'clear')
            self.assertFalse(self.session.valid_context(context))
            self.session.cancel_navigation()
        self.assertNotIn('recovery_blockage',self.session.task_goals.current('trip'))

    def test_final_checkpoint_must_be_single_and_all_choices_must_exist(self):
        for checkpoints in (['corridor_wait',{'any_of':['elevator','lobby_entrance']}],
                            [{'any_of':['elevator','missing_entrance']},'corridor_wait']):
            self.goal['target']['checkpoints']=checkpoints
            self.assertNotEqual(self.session.call('record_task',goals=[self.goal])['status'],'recorded')
