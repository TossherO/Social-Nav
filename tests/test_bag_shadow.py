"""Automatic bag routes use trusted map poses and the existing shadow DWA path."""
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from perception.environment.layers import load_config
from perception.environment.scene import assemble_scene
from runtime.bag_shadow import BagShadow, RecordedRoute, SCOPE
from runtime.command import CommandInbox
from runtime.interactive import cancel_operator_goal
from runtime.session import InteractiveSession
from tests.navigation_fixtures import asset, shadow_inputs, square_vehicle_geometry


def mapped_row(stamp, x, *, healthy=True):
    return dict(stamp_s=stamp, source='roll_fast_lio', status='PRIOR_ASSOCIATED' if healthy else 'LOCAL_ONLY',
                health=0, has_map_pose=healthy, map_frame='map', session_id='test', generation=0, frame_id='odom',
                map_revision=0, map_position=[x, 2., 0.], map_quaternion_xyzw=[0., 0., 0., 1.])


class BagShadowTests(unittest.TestCase):
    def test_route_breaks_on_loss_and_does_not_bridge_untrusted_samples(self):
        track=RecordedRoute([mapped_row(10.,2.),mapped_row(10.2,2.4),mapped_row(10.4,2.8),
                             mapped_row(10.6,3.2,healthy=False),mapped_row(10.8,3.6),
                             mapped_row(11.,4.),mapped_row(11.2,4.4)],square_vehicle_geometry(),'map')
        first, reason=track.target(10.,[2.,2.,0.],3.)
        self.assertIsNone(reason)
        self.assertAlmostEqual(first[1][0],2.8)
        self.assertEqual(track.target(10.6,[3.2,2.,0.],3.),(None,'outside_recorded_route_section'))
        later,reason=track.target(10.8,[3.6,2.,0.],3.)
        self.assertIsNone(reason)
        self.assertGreater(later[0],first[0])

    def test_current_goal_reaches_frozen_and_continuous_dwa_with_no_goal_feedback(self):
        from navigation.control.shadow import DwaShadowController
        from navigation.planning.frozen import preview
        from navigation.safety.shadow import ShadowSafety

        fixture=shadow_inputs()
        scene=assemble_scene(fixture.scene.geometry,clock_domain='runtime_bag',settings=load_config()['scene'])
        prior=asset()
        geometry=fixture.geometry
        controller=DwaShadowController(fixture.config,fixture.motion,geometry,asset=prior)
        settings=dict(route_replay=dict(lookahead_m=2.,min_update_s=2.,goal_shift_m=1.),input_python='/usr/bin/python3')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            shadow=BagShadow(None,None,settings,geometry,prior,start=0.,duration=None,cache=output,route_run=None,output=output)
            shadow.route=RecordedRoute([mapped_row(10.+k*.2,2.+k*.2) for k in range(11)],geometry,'map')
            safety=Mock()
            session=InteractiveSession(live=False,safety=safety,output=output,bag_shadow=shadow)
            self.addCleanup(session.close)
            self.assertTrue(session.handle('shadow_navigation',{'enabled':True})['accepted'])
            row=mapped_row(10.,2.)
            goal=shadow.select(scene,row,now=10.,clock_domain='runtime_bag',planning=fixture.config)
            self.assertEqual(goal['source'],SCOPE)
            self.assertEqual(goal['goal'][:2],[4.,2.])
            self.assertIs(shadow.select(scene,row,now=10.,clock_domain='runtime_bag',planning=fixture.config),goal)

            inbox=CommandInbox.__new__(CommandInbox)
            inbox.lock,inbox.safety,inbox.session=threading.Lock(),safety,session
            self.assertEqual(inbox.bag_shadow_goal(None,goal),goal)
            frozen=preview(goal,scene,fixture.rows,controller.controller,fixture.config,asset=prior,scope=SCOPE)
            self.assertTrue(frozen['reference_available'],frozen['reason'])
            self.assertTrue(frozen['generated'],frozen['reason'])
            self.assertEqual(frozen['scope'],SCOPE)
            packet=controller.step(goal,scene,fixture.rows,frozen['reference'],now=10.,clock_domain='runtime_bag',
                                   mono=.01,goal_diagnostic=False)
            self.assertEqual(packet['proposal']['status'],'candidate',packet['proposal'].get('reason'))
            self.assertIsNone(packet['recorded_goal_diagnostic'])
            self.assertFalse(packet['execution_feedback'] or packet['executable'] or packet['task_completed'])
            guard=ShadowSafety(fixture.config,fixture.motion,geometry,asset=prior,clock_domain='runtime_bag')
            guard.command('goal',goal)
            guard.push_grid(scene.geometry,mono=.01);guard.push_record(scene.record)
            for state in fixture.rows:guard.push_state(state)
            guard.push_packet(packet,mono=.01)
            outcome=guard.tick(10.,mono=.01)
            self.assertEqual(outcome['state'],'allowed',outcome)
            self.assertFalse(outcome['execution_feedback'] or outcome['executable'] or outcome['task_completed'])

            session.note_command('goal',{})
            self.assertEqual(shadow.status()['reason'],'manual_override')
            self.assertIsNone(inbox.bag_shadow_goal(goal,goal))
            self.assertEqual(safety.command.call_args_list,[call('goal',goal),call('cancel',{
                'task_id':goal['task_id'],'goal_revision':goal['goal_revision']})])
            self.assertIsNone(session.status()['shadow_navigation']['goal'])

    def test_live_session_cannot_enable_route_review(self):
        shadow=Mock()
        session=InteractiveSession(live=True,safety=Mock(),output=Path('/tmp'),bag_shadow=shadow)
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError,'requires a bag'):
            session.handle('shadow_navigation',{'enabled':True})
        shadow.toggle.assert_not_called()

    def test_operator_cancel_stops_route_preparation_and_repeat_preserves_other_preview(self):
        settings=dict(route_replay={},input_python='/usr/bin/python3')
        shadow=BagShadow(None,None,settings,None,asset(),start=0.,duration=None,
                         cache=Path('/tmp'),route_run=None,output=Path('/tmp'))
        session=InteractiveSession(live=False,safety=Mock(),output=Path('/tmp'),bag_shadow=shadow)
        self.addCleanup(session.close)
        self.addCleanup(shadow.close)
        stopped=threading.Event()

        def prepare(stop):
            if stop.wait(2.):stopped.set()
            raise InterruptedError('Preparation cancelled')

        shadow._load_route=prepare
        opened=session.handle('shadow_navigation',{'enabled':True})
        self.assertEqual(opened['state'],'preparing')
        bridge=SimpleNamespace(last_submitted=None)
        result=cancel_operator_goal(bridge,session.handle,bag_ready=True,execute=False)
        self.assertTrue(result['changed']);self.assertFalse(result['enabled'])
        self.assertTrue(stopped.wait(1.))

        session.preview={'request':{'request_id':'unrelated'}}
        session.goal_intent={'request_id':'unrelated'}
        generation=session.command_generation
        result=cancel_operator_goal(bridge,session.handle,bag_ready=True,execute=False)
        self.assertTrue(result['accepted']);self.assertFalse(result['changed'])
        self.assertEqual(generation,session.command_generation)
        self.assertEqual(session.preview['request']['request_id'],'unrelated')
        self.assertEqual(session.goal_intent['request_id'],'unrelated')

    def test_operator_cancel_preserves_manual_revision_and_live_disarming(self):
        bridge=Mock(last_submitted={'goal_revision':1})
        send=Mock()
        self.assertIs(cancel_operator_goal(bridge,send,bag_ready=True,execute=False),bridge.cancel.return_value)
        bridge.cancel.assert_called_once_with()
        send.assert_not_called()

        bridge.cancel.side_effect=RuntimeError('manual cancel failed')
        with self.assertRaisesRegex(RuntimeError,'manual cancel failed'):
            cancel_operator_goal(bridge,send,bag_ready=False,execute=True)
        send.assert_called_once_with('arm',{'enabled':False})
