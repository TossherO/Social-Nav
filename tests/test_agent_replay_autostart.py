"""Replay bootstrap retains source, task and navigation checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from decision.agent.workflows.shadow import wait_for_bag_navigation,hold_after_t2_failure,start_unique_bag_follow,replay_t2_result


class ReplayAutostartTests(unittest.TestCase):
    def test_replay_displays_local_decision_and_remote_choice(self):
        local = dict(backend='reid', selection=dict(status='same', reason='confirmed', candidate_id=0))
        self.assertEqual(replay_t2_result(local)['selection'], '人物 0')
        local.update(selection=dict(status='unconfirmed', reason='no_detections'), search_choice=dict(kind='rotate'))
        self.assertEqual(replay_t2_result(local)['selection'], 'rotate')
        remote = dict(backend='dspark', selection='P0', choices={'P0': {'kind': 'person'}})
        self.assertEqual(replay_t2_result(remote)['choice'], remote['choices']['P0'])

    def test_local_start_requires_one_associated_candidate_and_uses_current_observation(self):
        candidates=[]
        observation=SimpleNamespace(record=dict(observation_id='frame',follow_candidates=dict(candidates=candidates)))
        navigation=SimpleNamespace(snapshot=lambda:dict(localization_status='valid',map_binding=dict(valid=True)),
                                   observation_record=lambda:observation.record,value={})
        session=SimpleNamespace(navigation=navigation,observation=observation,context=lambda:'context',
            call=Mock(side_effect=[{'status':'available'},{'status':'selected','reference_id':'ref'},{'status':'accepted'}]),
            wait_for_event=Mock(),_save=Mock())
        self.assertEqual(start_unique_bag_follow(session,'g'),'waiting')
        candidates.extend([dict(id=0,track={'track_key':'a/g1'}),dict(id=1,track={'track_key':'b/g1'})])
        self.assertEqual(start_unique_bag_follow(session,'g'),'agent_selection')
        session.call.assert_not_called()
        candidates[1]['track']=None
        self.assertEqual(start_unique_bag_follow(session,'g'),'agent_selection')
        candidates.pop(0)
        self.assertEqual(start_unique_bag_follow(session,'g'),'waiting')
        session.call.assert_not_called()
        candidates[0]=dict(id=0,track={'track_key':'a/g1'})
        self.assertEqual(start_unique_bag_follow(session,'g'),'started')
        self.assertEqual([c.args[0] for c in session.call.call_args_list],
                         ['observe_scene','select_follow_target','start_local_task'])
        self.assertEqual(session.call.call_args_list[1].kwargs['candidate_id'],0)
        self.assertEqual(session.call.call_args_list[1].kwargs['_context'],'context')
        session.wait_for_event.assert_called_once_with('navigation')

    def test_wait_requires_nonexecuting_bag_and_fresh_mapped_pose(self):
        catalog=SimpleNamespace(value={'frame_id':'map'})
        invalid=dict(input_stale=True,localization_status='valid',origin_pose=[0,0,0],
                     origin_frame_id='map',map_binding={'valid':True})
        valid=dict(invalid,input_stale=False)
        nav=SimpleNamespace(clock_domain='runtime_bag',tick=Mock(),snapshot=Mock(side_effect=[invalid,valid]))
        with patch('pathlib.Path.exists',return_value=True), \
             patch('runtime.transport.exchange_request',return_value={'live':False,'hardware_enabled':False}), \
             patch('decision.agent.workflows.shadow.RuntimeShadowNavigation',return_value=nav), \
             patch('decision.agent.workflows.shadow.time.sleep'):
            self.assertIs(wait_for_bag_navigation(catalog,'socket',timeout_s=2,emit_status=lambda _:None),nav)
        nav.tick.assert_called_once()
        with patch('pathlib.Path.exists',return_value=True), \
             patch('runtime.transport.exchange_request',return_value={'live':True,'hardware_enabled':False}):
            with self.assertRaisesRegex(ValueError,'nonexecuting bag'):
                wait_for_bag_navigation(catalog,'socket',timeout_s=2,emit_status=lambda _:None)

    def test_t2_transport_failure_waits_for_operator_instead_of_retrying(self):
        session=SimpleNamespace(skill=SimpleNamespace(record={'stage_type':'follow','status':'failed',
            'reason':'follow_t2_hosted_WriteTimeout'}),state={'status':'ready','reason':None},stop=Mock(),_save=Mock())
        actor=SimpleNamespace(model=SimpleNamespace(invalidate=Mock()))
        self.assertTrue(hold_after_t2_failure(session,actor))
        session.stop.assert_called_once_with('needs_clarification','follow_t2_hosted_WriteTimeout')
        actor.model.invalidate.assert_called_once()

    def test_failure_hook_runs_before_new_outer_model_request(self):
        from decision.agent.core.loop import AgentLoop
        state={'status':'ready','reason':None}
        def stop(target,reason):state.update(status=target,reason=reason)
        session=SimpleNamespace(state=state,tick=lambda:state,stop=stop,_save=Mock(),
            navigation=SimpleNamespace(snapshot=lambda:{}),
            skill=SimpleNamespace(record={'stage_type':'follow','status':'failed','connected':True,
                'context_lost':False,'association_waiting':False,'reason':'follow_t2_hosted_WriteTimeout'}))
        actor=AgentLoop(session,Mock())
        actor.before_model=lambda a:hold_after_t2_failure(session,a)
        try:
            actor.tick()
            self.assertEqual(state['status'],'needs_clarification')
            self.assertEqual(actor.requests,0)
            self.assertFalse(actor.model.busy)
        finally:actor.model.close()


if __name__=='__main__':unittest.main()
