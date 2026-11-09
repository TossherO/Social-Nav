"""Clock applicability and provenance without devices, ROS master or model loads."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import yaml

from perception.sensors.clock_session import record_bag_session, session_path, time_calibration_report, check_time_calibration
from perception.mapping_localization.configuration import algorithm_config, ROOT


class ClockSessionTests(unittest.TestCase):
    def test_partial_new_calibration_cannot_inherit_old_session_offset(self):
        source=self.root/'source';source.mkdir()
        (source/'run.json').write_text(json.dumps(dict(algorithm_parameters=self.config)))
        partial=copy.deepcopy(self.config);partial['common'].pop('imu_time_offset')
        partial['calibration']['deployment_clock_group']='driver_B'
        self.calibration.write_text(yaml.safe_dump(partial))
        config,_=algorithm_config(mode='odometry',source_run=source,calibration=self.calibration)
        self.assertNotIn('imu_time_offset',config['common'])
        report=time_calibration_report(config,group='driver_B')
        self.assertFalse(report['offset_configured'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bag = self.root/'capture.bag'
        self.bag.write_bytes(b'SYNTHETIC_FINGERPRINT_INPUT')
        self.config = dict(common=dict(imu_time_offset=-.04480758383899385, time_sync_en=False),
            mapping=dict(extrinsic_R=[1.,0.,0.,0.,1.,0.,0.,0.,1.], extrinsic_T=[0.,0.,0.]),
            calibration=dict(deployment_clock_group='driver_A', status='candidate_pending_replay'))
        self.calibration = self.root/'calibration.yaml'
        self.calibration.write_text(yaml.safe_dump(self.config))

    def check(self, config=None, **options):
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            result = check_time_calibration(config or self.config, **options)
        return result, log.getvalue()

    def test_unknown_input_does_not_inherit_calibration_group_or_claim_validation(self):
        result, log = self.check(bag=self.bag, source_file=self.calibration)
        self.assertIsNone(result['input_clock_group'])
        self.assertIsNone(result['clock_group_match'])
        self.assertEqual(result['offset_s'], -.04480758383899385)
        self.assertFalse(result['validation_performed'])
        self.assertIn('WARNING', log)
        self.assertIn('unknown', log)

    def test_zero_default_needs_no_clock_declaration_and_claims_no_calibration(self):
        config,_=algorithm_config(mode='odometry')
        result,log=self.check(config,bag=self.bag)
        self.assertEqual(result['offset_s'],0.)
        self.assertEqual(result['offset_policy'],'sensor_timestamps')
        self.assertIsNone(result['calibration_clock_group'])
        self.assertIsNone(result['input_clock_group'])
        self.assertFalse(result['validation_performed'])
        self.assertNotIn('WARNING',log)

    def test_matching_declarations_do_not_validate_a_candidate_offset(self):
        result, log = self.check(bag=self.bag, group='driver_A')
        self.assertIs(result['clock_group_match'], True)
        self.assertEqual(result['declared_calibration_status'], 'candidate_pending_replay')
        self.assertFalse(result['validation_performed'])
        self.assertIn('not_performed_by_startup_check', log)

    def test_known_mismatch_is_rejected_without_changing_values(self):
        before = copy.deepcopy(self.config)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, 'clock group mismatch'):
            check_time_calibration(self.config, bag=self.bag, group='driver_B')
        self.assertEqual(self.config, before)

    def test_multiple_bags_can_share_one_declared_driver_session(self):
        original = self.bag.read_bytes()
        other = self.root/'second.bag'
        other.write_bytes(b'OTHER_CAPTURE')
        for bag in (self.bag, other):
            record_bag_session(bag, 'driver_A')
            result = time_calibration_report(self.config, bag=bag)
            self.assertEqual(result['input_clock_group_source'], 'bag_session_record')
            self.assertIs(result['clock_group_match'], True)
        self.assertEqual(self.bag.read_bytes(), original)

    def test_session_file_is_bound_to_content_and_can_follow_a_bag_copy(self):
        record_bag_session(self.bag, 'driver_A')
        relocated = self.root/'relocated.bag'
        relocated.write_bytes(self.bag.read_bytes())
        result = time_calibration_report(self.config, bag=relocated, session_file=session_path(self.bag))
        self.assertIs(result['clock_group_match'], True)
        relocated.write_bytes(b'DIFFERENT_INPUT')
        with self.assertRaisesRegex(ValueError, 'does not identify this bag'):
            time_calibration_report(self.config, bag=relocated, session_file=session_path(self.bag))

    def test_argument_cannot_silently_override_a_recorded_session(self):
        record_bag_session(self.bag, 'driver_A')
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            time_calibration_report(self.config, bag=self.bag, group='driver_B')

    def test_bad_sidecar_is_not_ignored_even_with_an_explicit_group(self):
        session_path(self.bag).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'does not identify this bag'):
            time_calibration_report(self.config, bag=self.bag, group='driver_A')

    def test_session_writer_never_overwrites_existing_files_or_links(self):
        record_bag_session(self.bag, 'driver_A')
        saved = session_path(self.bag).read_bytes()
        with self.assertRaises(FileExistsError):
            record_bag_session(self.bag, 'driver_B')
        self.assertEqual(session_path(self.bag).read_bytes(), saved)
        link = self.root/'dangling.json'
        link.symlink_to(self.root/'absent')
        with self.assertRaises(FileExistsError):
            record_bag_session(self.bag, 'driver_B', link)

    def test_recording_cli_supports_a_separate_metadata_directory(self):
        from perception.sensors.clock_session import main
        target = self.root/'session.json'
        argv = ['clock_session', '--bag', str(self.bag), '--clock-group', 'driver_A', '--out', str(target)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertFalse(session_path(self.bag).exists())
        result = time_calibration_report(self.config, bag=self.bag, session_file=target)
        self.assertIs(result['clock_group_match'], True)

    def test_invalid_groups_and_nonfinite_offsets_are_rejected(self):
        for group in ('', 'driver A', 'a\nb', 12, False):
            with self.subTest(group=group), self.assertRaises(ValueError):
                time_calibration_report(self.config, group=group)
        for value in (None, True, '0.02', float('nan'), float('inf')):
            config = copy.deepcopy(self.config)
            config['common']['imu_time_offset'] = value
            with self.subTest(offset=value), self.assertRaises(ValueError):
                time_calibration_report(config)

    def test_absent_metadata_and_zero_default_are_explicitly_unconfirmed(self):
        result, log = self.check(dict(common={}), group='driver_A')
        self.assertEqual(result['offset_s'], 0.)
        self.assertFalse(result['offset_configured'])
        self.assertIsNone(result['clock_group_match'])
        self.assertIn('zero default', log)

    def test_heuristic_request_is_not_mistaken_for_a_verified_fixed_offset(self):
        config = copy.deepcopy(self.config)
        config['common']['time_sync_en'] = True
        result, log = self.check(config, group='driver_A')
        self.assertIs(result['heuristic_sync_requested'], True)
        self.assertIn('Heuristic', log)
        self.assertFalse(result['validation_performed'])
        config['common']['time_sync_en'] = 'false'
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            time_calibration_report(config)

    def test_source_run_identity_is_reused_only_for_the_same_bag(self):
        previous = time_calibration_report(self.config, bag=self.bag, group='driver_A')
        result = time_calibration_report(self.config, bag=self.bag, previous=previous)
        self.assertEqual(result['input_clock_group_source'], 'source_run')
        other = self.root/'new.bag'
        other.write_bytes(b'NEW_SESSION_DATA')
        self.assertIsNone(time_calibration_report(self.config, bag=other, previous=previous)['input_clock_group'])
        self.assertIsNone(time_calibration_report(self.config, previous=previous)['input_clock_group'])

    def test_bag_record_cannot_be_applied_to_live_input(self):
        record_bag_session(self.bag, 'driver_A')
        with self.assertRaisesRegex(ValueError, 'live input'):
            time_calibration_report(self.config, session_file=session_path(self.bag))

    def test_provenance_hashes_identify_the_files_actually_read(self):
        record_bag_session(self.bag, 'driver_A')
        result = time_calibration_report(self.config, bag=self.bag, source_file=self.calibration)
        self.assertEqual(result['calibration_source']['sha256'], hashlib.sha256(self.calibration.read_bytes()).hexdigest())
        self.assertEqual(result['session_record']['sha256'], hashlib.sha256(session_path(self.bag).read_bytes()).hexdigest())
        json.dumps(result, allow_nan=False)

    def test_explicit_calibration_override_drops_old_scope_and_validation(self):
        source = self.root/'previous'
        source.mkdir()
        manifest = source/'run.json'
        manifest.write_text(json.dumps(dict(algorithm_parameters=self.config)))
        original = manifest.read_bytes()
        override = self.root/'offset.yaml'
        override.write_text('common:\n  imu_time_offset: 0.02\n')
        config, _ = algorithm_config(mode='mapping', source_run=source, calibration=override)
        self.assertEqual(config['common']['imu_time_offset'], .02)
        self.assertEqual(config['mapping'], self.config['mapping'])
        self.assertEqual(config['calibration'], {})
        self.assertEqual(manifest.read_bytes(), original)

    def test_direct_cli_rejects_mismatch_before_cuda_or_output_creation(self):
        from runtime.__main__ import main
        output = self.root/'must_not_exist'
        argv = ['runtime', '--bag', str(self.bag), '--mode', 'odometry', '--output', str(output),
                '--lidar-imu-calibration', str(self.calibration), '--sensor-clock-group', 'driver_B']
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch': None}), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, 'clock group mismatch'):
            main()
        self.assertFalse(output.exists())

    def test_short_bag_and_live_default_reach_device_check_without_clock_initialization(self):
        from runtime.__main__ import main
        for source in (['--bag',str(self.bag)],['--live']):
            output=self.root/'device_unavailable'
            argv=['runtime',*source,'--mode','odometry','--output',str(output),'--duration','2']
            no_device=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
            with self.subTest(source=source),patch.object(sys,'argv',argv), \
                    patch.dict(sys.modules,{'torch':no_device}),contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError,'Selected CUDA device unavailable'):
                main()
            self.assertFalse(output.exists())

    def test_ros_diagnostic_checks_before_launch_and_preserves_report(self):
        from perception.mapping_localization.__main__ import arguments
        argv = ['diagnostic', '--mode', 'odometry', '--bag', str(self.bag), '--out', str(self.root/'diagnostic'),
                '--lidar-imu-calibration', str(self.calibration), '--sensor-clock-group', 'driver_B']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), \
                patch('perception.mapping_localization.__main__.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'clock group mismatch'):
                arguments()
            argv[-1] = 'driver_A'
            args = arguments()
            self.assertIs(args.imu_time_calibration['clock_group_match'], True)
            launch.assert_not_called()
        self.assertFalse((self.root/'diagnostic').exists())

    def test_ros_zero_default_precedes_the_selected_calibration(self):
        launch = ET.parse(ROOT/'perception/mapping_localization/launch/native.launch').getroot()
        items = list(launch)
        reset = next(i for i, x in enumerate(items) if x.tag == 'param' and x.get('name') == 'common/imu_time_offset')
        selected = next(i for i, x in enumerate(items) if x.tag == 'rosparam' and x.get('file') == '$(arg lidar_imu_calibration)')
        self.assertLess(reset, selected)
        self.assertEqual(float(items[reset].get('value')), 0.)
        sync = next(i for i, x in enumerate(items) if x.tag == 'param' and x.get('name') == 'common/time_sync_en')
        self.assertLess(sync, selected)
        self.assertEqual(items[sync].get('value'), 'false')


if __name__ == '__main__':
    unittest.main()
