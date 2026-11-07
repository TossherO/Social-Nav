"""Batch sequencing checks use generated results; no device is opened."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from navigation.control.ground_probe import probe_profile
from navigation.control.response_probe import main,run_batch,selected_plan


class ResponseBatchTests(unittest.TestCase):
    def fake_segment(self,command,console,*,fail=False,responded=True):
        kind=command[command.index('--maneuver')+1]
        speed=float(command[command.index('--speed')+1]) if '--speed' in command else None
        output=Path(command[command.index('--output')+1]);output.mkdir()
        profile=probe_profile(maneuver=kind,speed=speed,pulse_s=2.)
        result=dict(profile=profile,completed=not fail,stop_feedback_observed=True,stop_confirmation_timed_out=False,
            nonzero_candidate_feedback_observed=responded,error='ProbeAbort: joystick_takeover' if fail else None)
        (output/'result.json').write_text(json.dumps(result))
        return 1 if fail or not responded else 0

    def test_fixed_plan_uses_distinct_units_and_no_default_execution(self):
        plan=selected_plan();self.assertEqual(len(plan['profiles']),6)
        self.assertEqual([x['linear_mps'] for x in plan['profiles']],[.01,.03,.06,.05,0.,0.])
        self.assertEqual([x['angular_rps'] for x in plan['profiles']],[0.,0.,0.,0.,.1,.1])
        self.assertFalse(plan['execution_requested'])
        with patch('navigation.control.response_probe.run_batch') as execute,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]),0);execute.assert_not_called()

    def test_expected_low_speed_no_response_does_not_adapt_the_selected_plan(self):
        calls=[]
        def fake(command,console):
            calls.append(command)
            return self.fake_segment(command,console,responded=len(calls)>1)
        with tempfile.TemporaryDirectory() as folder,patch('navigation.control.response_probe._run_segment',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
            result=run_batch('/not/a/device',Path(folder)/'batch')
        self.assertEqual(len(calls),6);self.assertTrue(result['all_command_sequences_completed'])
        self.assertFalse(result['stages'][0]['candidate_response_observed'])
        self.assertFalse(result['motion_verified']);self.assertFalse(result['vehicle_profile_updated'])
        self.assertEqual([c[c.index('--speed')+1] for c in calls if '--speed' in c],['0.01','0.03','0.06'])

    def test_safety_abort_ends_batch_without_a_later_maneuver(self):
        calls=[]
        def fake(command,console):
            calls.append(command);return self.fake_segment(command,console,fail=len(calls)==2)
        with tempfile.TemporaryDirectory() as folder,patch('navigation.control.response_probe._run_segment',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
            result=run_batch('/not/a/device',Path(folder)/'batch')
            saved=json.loads((Path(folder)/'batch/summary.json').read_text())
        self.assertEqual(len(calls),2);self.assertFalse(result['all_command_sequences_completed'])
        self.assertFalse(result['stages'][-1]['safe_to_continue']);self.assertEqual(result,saved)

    def test_cancelled_child_is_recorded_and_no_later_stage_starts(self):
        with tempfile.TemporaryDirectory() as folder,patch('navigation.control.response_probe._run_segment',side_effect=KeyboardInterrupt),contextlib.redirect_stdout(io.StringIO()):
            result=run_batch('/not/a/device',Path(folder)/'batch')
        self.assertTrue(result['cancelled']);self.assertEqual(len(result['stages']),1)
        self.assertFalse(result['all_command_sequences_completed'])

    def test_missing_explicit_hardware_arguments_never_creates_output(self):
        with patch('navigation.control.response_probe.run_batch') as execute,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--execute'])
        execute.assert_not_called()


if __name__=='__main__':unittest.main()
