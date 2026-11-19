"""Replay bootstrap retains source, task and navigation checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from decision.agent.workflows.shadow import wait_for_bag_navigation,hold_after_t2_failure


class ReplayAutostartTests(unittest.TestCase):
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
            skill=SimpleNamespace(record={'stage_type':'follow','status':'failed','connected':True,
                'context_lost':False,'reason':'follow_t2_hosted_WriteTimeout'}))
        actor=AgentLoop(session,Mock())
        actor.before_model=lambda a:hold_after_t2_failure(session,a)
        try:
            actor.tick()
            self.assertEqual(state['status'],'needs_clarification')
            self.assertEqual(actor.requests,0)
            self.assertFalse(actor.model.busy)
        finally:actor.model.close()


if __name__=='__main__':unittest.main()
