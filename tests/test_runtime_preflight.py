"""Deployment failures must stay explicit and never acquire a real chassis."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from runtime import preflight
from navigation.planning.motion import load_motion_config
from navigation.control.vehicle import load_vehicle_profile


class TrialConfigTests(unittest.TestCase):
    def test_trial_is_explicit_assumed_and_cannot_confirm_a_vehicle(self):
        normal=load_motion_config()
        trial=load_motion_config('navigation/planning/configs/motion_trial.yaml')
        self.assertTrue(trial['parameters_assumed'])
        self.assertEqual(trial['max_forward_mps'],.1)
        self.assertEqual(trial['max_reverse_mps'],.05)
        for key in ('max_forward_mps','max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2'):
            self.assertLess(trial[key],normal[key])
        self.assertEqual(trial['state_max_age_s'],normal['state_max_age_s'])
        with self.assertRaises(ValueError):
            load_vehicle_profile('navigation/control/vehicle.example.yaml',trial,require_execution=True)
        self.assertEqual(load_motion_config(),normal)


class PreflightTests(unittest.TestCase):
    def test_interactive_forwards_the_same_selected_trial_and_runtime_configs(self):
        from runtime.interactive import parser, runtime_command
        args=parser().parse_args(['--live','--output','unused','--motion-config','navigation/planning/configs/motion_trial.yaml',
                                 '--runtime-config','configs/runtime.yaml'])
        command=runtime_command(args,dict(map='map',static_map='static',coverage='coverage',motion_config='normal'),[0.]*6)
        self.assertEqual(Path(command[command.index('--motion-config')+1]).name,'motion_trial.yaml')
        self.assertEqual(Path(command[command.index('--runtime-config')+1]).name,'runtime.yaml')
        self.assertNotIn('--execute',command)

    def test_serial_query_does_not_open_device_and_rejects_busy_or_unknown(self):
        with patch('runtime.preflight.shutil.which',return_value='/usr/bin/fuser'), \
                patch('runtime.preflight.os.open',side_effect=AssertionError('must not open device')), \
                patch('runtime.preflight.subprocess.run') as run:
            run.return_value=SimpleNamespace(returncode=1,stdout='',stderr='')
            self.assertFalse(preflight.serial_access('/dev/null')['opened'])
            for result in (SimpleNamespace(returncode=0,stdout='123',stderr=''),
                           SimpleNamespace(returncode=1,stdout='',stderr='Permission denied')):
                run.return_value=result
                with self.assertRaises(RuntimeError):preflight.serial_access('/dev/null')

    def test_missing_inputs_report_multiple_errors_without_execution(self):
        with patch('runtime.preflight.probe',return_value='synthetic import result'), \
                patch('runtime.preflight.shutil.which',return_value=None), \
                patch('runtime.preflight.os.open',side_effect=AssertionError('no device access')):
            report=preflight.inspect('/does-not-exist/interactive.yaml',execution=True)
        errors={row['name'] for row in report['checks'] if row['status']=='error'}
        self.assertTrue({'interactive_config','motion_config_valid','measured_vehicle_profile','serial_access'}<=errors)
        self.assertFalse(report['software_checks_passed'])
        self.assertFalse(report['hardware_opened'])
        self.assertFalse(report['execution_enabled'])
        self.assertFalse(report['physical_acceptance_verified'])

    def test_complete_marker_does_not_hide_missing_keyframe_clouds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);maps=root/'map';maps.mkdir();(maps/'keyframes').mkdir()
            (maps/'manifest.yaml').write_text(yaml.safe_dump(dict(format='roll_keyframes_v1',keyframes=1)))
            config=root/'interactive.yaml'
            config.write_text(yaml.safe_dump(dict(map=str(maps),static_map=str(maps),coverage='missing.yaml',
                initial_pose_hint='missing.json',motion_config='navigation/planning/configs/motion_trial.yaml')))
            with patch('runtime.interactive.session_inputs',return_value=({},SimpleNamespace(manifest={'frame_id':'map'}),{})), \
                    patch('runtime.preflight.probe',return_value='synthetic import result'), \
                    patch('runtime.preflight.shutil.which',return_value=None):
                report=preflight.inspect(config)
            check=next(row for row in report['checks'] if row['name']=='map_pair_and_coverage')
            self.assertEqual(check['status'],'error')
            self.assertIn('corner0.pcd',check['detail'])

    def test_probe_is_bounded_and_inherits_explicit_gpu_without_loading_weights(self):
        with patch('runtime.preflight.subprocess.run') as run:
            run.return_value=SimpleNamespace(returncode=0,stdout='ok',stderr='')
            self.assertEqual(preflight.probe('/usr/bin/python3','print("ok")',gpu='2'),'ok')
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'2')
            self.assertEqual(run.call_args.kwargs['timeout'],30)
            self.assertIn('-B',run.call_args.args[0])
            run.return_value=SimpleNamespace(returncode=1,stdout='',stderr='incompatible library')
            with self.assertRaisesRegex(RuntimeError,'incompatible library'):
                preflight.probe('/usr/bin/python3','pass')

    def test_cli_returns_failure_and_never_translates_check_into_execute(self):
        report=dict(software_checks_passed=False,execution_enabled=False)
        with patch('runtime.preflight.inspect',return_value=report) as inspect, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code=preflight.main(['--execution-check','--vehicle-profile','unconfirmed.yaml','--chassis-port','/dev/not-real'])
        self.assertEqual(code,1)
        self.assertFalse(json.loads(output.getvalue())['execution_enabled'])
        self.assertTrue(inspect.call_args.kwargs['execution'])
