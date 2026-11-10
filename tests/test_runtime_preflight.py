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
    def test_motion_profiles_select_explicit_limits_and_keep_shared_settings(self):
        normal=load_motion_config(profile='normal');trial=load_motion_config(profile='trial')
        aggressive=load_motion_config(profile='aggressive')
        names=('max_forward_mps','max_reverse_mps','max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2')
        for value,expected in ((normal,(.8,.2,.4,.36,.3)),(trial,(.8,.2,.4,.36,.3)),
                               (aggressive,(1.3,.2,1.,.8,1.))):
            self.assertEqual(tuple(value[k] for k in names),expected)
            self.assertEqual(value['stationary_min_window_s'],.2)
            self.assertEqual(value['stationary_position_span_m'],.01)
            self.assertEqual(value['stationary_yaw_span_rad'],.01)
            self.assertTrue({'collision_backend','collision_library'}.isdisjoint(value))
            self.assertTrue({'performance_reference','performance_profile','parameter_provenance'}.isdisjoint(value))
        self.assertEqual(load_motion_config(),normal)
        with self.assertRaisesRegex(ValueError,'Unknown motion profile'):load_motion_config(profile='missing')

    def test_flat_custom_config_is_supported_without_implicit_profile(self):
        trial=load_motion_config(profile='trial')
        for key in ('stationary_min_window_s','stationary_position_span_m','stationary_yaw_span_rad',
                    'stationary_max_linear_mps','stationary_max_angular_rps'):
            trial.pop(key)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'custom.yaml';path.write_text(yaml.safe_dump(trial))
            self.assertEqual(load_motion_config(path),load_motion_config(profile='trial'))
            with self.assertRaisesRegex(ValueError,'does not define profiles'):load_motion_config(path,profile='trial')

    def test_collision_defaults_and_explicit_custom_settings(self):
        from navigation.planning.collision import DEFAULT_LIBRARY, SweepChecker
        motion=load_motion_config()
        checker=SweepChecker(motion.get('collision_backend','cpu'),motion.get('collision_library'))
        self.assertEqual(checker.backend,'cpu')
        self.assertEqual(checker.library,str(DEFAULT_LIBRARY))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'motion.yaml'
            path.write_text(yaml.safe_dump(dict(motion,collision_backend='cuda',collision_library='custom/sweep.so')))
            custom=load_motion_config(path)
        self.assertEqual(custom['collision_backend'],'cuda')
        self.assertEqual(custom['collision_library'],str(Path(__file__).resolve().parents[1]/'custom/sweep.so'))

    def test_legacy_flat_config_can_still_restore_historical_limits(self):
        legacy=load_motion_config(profile='normal')
        names=('max_forward_mps','max_reverse_mps','max_angular_rps','max_acceleration_mps2','max_angular_acceleration_rps2')
        for key in names:legacy.pop(key)
        legacy.update(performance_reference='configs/wheelchair_motion_reference.json',performance_profile='teb')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'legacy.yaml';path.write_text(yaml.safe_dump(legacy))
            loaded=load_motion_config(path)
        self.assertEqual(tuple(loaded[k] for k in names),(.8,.2,.32,.24,.24))
        self.assertEqual(loaded['parameter_provenance']['profile'],'teb')

    def test_profile_requires_all_limits_and_cannot_override_shared_fields(self):
        source=Path('navigation/planning/configs/motion.yaml')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'motion.yaml'
            for change in ('missing_limit','common_override'):
                config=yaml.safe_load(source.read_text())
                if change=='missing_limit':del config['profiles']['trial']['max_forward_mps']
                else:config['profiles']['trial']['state_window_s']=.1
                path.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError,'must declare all limits'):load_motion_config(path,profile='trial')

    def test_trial_is_explicit_assumed_and_cannot_confirm_a_vehicle(self):
        normal=load_motion_config()
        trial=load_motion_config(profile='trial')
        self.assertTrue(trial['parameters_assumed'])
        self.assertEqual(trial['max_forward_mps'],.8)
        self.assertEqual(trial['max_reverse_mps'],.2)
        self.assertEqual(trial['max_angular_rps'],.4)
        self.assertEqual(trial['max_acceleration_mps2'],.36)
        self.assertEqual(trial['max_angular_acceleration_rps2'],.3)
        self.assertNotIn('brake_deceleration_scale',trial)
        self.assertEqual(trial['max_forward_mps'],normal['max_forward_mps'])
        self.assertEqual(trial['max_angular_rps'],normal['max_angular_rps'])
        self.assertEqual(trial['state_max_age_s'],normal['state_max_age_s'])
        with self.assertRaises(ValueError):
            load_vehicle_profile('navigation/control/vehicle.example.yaml',trial,require_execution=True)
        self.assertEqual(load_motion_config(),normal)


class PreflightTests(unittest.TestCase):
    def test_interactive_profile_precedence_does_not_apply_to_custom_flat_files(self):
        from runtime.interactive import motion_selection
        config=dict(motion_config='navigation/planning/configs/motion.yaml',motion_profile='trial')
        self.assertEqual(motion_selection(config)[1],'trial')
        self.assertEqual(motion_selection(config,profile='aggressive')[1],'aggressive')
        self.assertIsNone(motion_selection(config,path='custom.yaml')[1])
        self.assertEqual(motion_selection(config,path='custom.yaml',profile='normal')[1],'normal')

    def test_interactive_forwards_the_same_selected_trial_and_runtime_configs(self):
        from runtime.interactive import parser, runtime_command
        args=parser().parse_args(['--live','--output','unused','--motion-profile','trial',
                                 '--runtime-config','configs/runtime.yaml','--environment-update-hz','7.5'])
        command=runtime_command(args,dict(map='map',static_map='static',coverage='coverage',motion_config='normal'),[0.]*6)
        self.assertEqual(command[command.index('--motion-profile')+1],'trial')
        self.assertEqual(Path(command[command.index('--runtime-config')+1]).name,'runtime.yaml')
        self.assertEqual(command[command.index('--environment-update-hz')+1], '7.5')
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
                initial_pose_hint='missing.json',motion_config='navigation/planning/configs/motion.yaml',motion_profile='trial')))
            with patch('runtime.interactive.session_inputs',return_value=({},SimpleNamespace(manifest={'frame_id':'map'}),{})), \
                    patch('runtime.preflight.probe',return_value='synthetic import result'), \
                    patch('runtime.preflight.shutil.which',return_value=None):
                report=preflight.inspect(config)
            check=next(row for row in report['checks'] if row['name']=='map_pair_and_coverage')
            self.assertEqual(check['status'],'error')
            self.assertIn('corner0.pcd',check['detail'])
            motion=next(row for row in report['checks'] if row['name']=='motion_config_valid')
            self.assertEqual(motion['status'],'ok')
            self.assertEqual(motion['detail']['max_forward_mps'],.8)

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
