"""Offline mapping orchestration with synthetic native results; no devices."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from perception.mapping_localization import build_map
from perception.environment.layers import load_config
from perception.environment.poses import PoseTimeline
from perception.environment.scans import sealed_scan_summary


class FakeSession:
    """Synthetic estimator for lifecycle tests, never a SLAM accuracy result."""
    def __init__(self, config_path, **kwargs):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.jobs = SimpleNamespace(qsize=lambda: 0)
        self.error = None
        self.calls = []
        self.valid = True
        self.failure_at = None

    def step(self, batch):
        self.calls.append('step')
        if self.failure_at == self.calls.count('step'):
            raise RuntimeError('synthetic estimator failure')
        stamp = batch['end']
        state = dict(stamp_s=stamp, stamp_ns=round(stamp*1e9), frame_id='odom', child_frame_id='rslidar',
                     session_id='synthetic', generation=0, position=[0., 0., 0.],
                     quaternion_xyzw=[0., 0., 0., 1.], health=0 if self.valid else 2)
        cloud = np.array([[1., 0., 0., 1., -50.], [2., 0., 0., 1., 0.]], np.float32)
        front = dict(valid=self.valid, reason=0 if self.valid else 1, residual=.01, gyro_bias=np.zeros(3),
                     acc_bias=np.zeros(3), processing_ms=2., matching_ms=1., map_update_ms=.5, effective_points=2)
        return state, {}, cloud, front, None

    def save(self, path):
        self.calls.append('save')
        root = Path(path)
        (root/'keyframes').mkdir(parents=True)
        (root/'manifest.yaml').write_text('format: roll_keyframes_v1\nframe: map\nkeyframes: 1\n')
        (root/'keyframes/poses.txt').write_text('10 0 0 0 0 0 0 0 1\n')
        (root/'GlobalMap.pcd').write_text('SYNTHETIC TEST CLOUD, NOT A REAL SLAM MAP\n')
        (root/'COMPLETE').write_text('synthetic\n')

    def close(self):
        self.calls.append('close')


class BuildMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bag = self.root/'input.bag'
        self.bag.write_bytes(b'SYNTHETIC INPUT; SENSOR READER IS MOCKED')
        self.original = self.bag.read_bytes()
        self.out = self.root/'run'
        self.session = None
        self.bag_count = 3
        self.raw_count = 3
        self.drop_last = False

    def args(self, *extra):
        return build_map.parser().parse_args(['--bag', str(self.bag), '--out', str(self.out), *extra])

    def make_session(self, *args, **kwargs):
        self.session = FakeSession(*args, **kwargs)
        return self.session

    def raw(self, *args, **kwargs):
        return iter([('/rslidar_points', index, None) for index in range(self.raw_count)])

    def observations(self, path, config, *, messages):
        for topic, index, unused in messages:
            if self.drop_last and index == self.raw_count-1:
                continue
            yield dict(end=10.+index*.1, sensor_frame='rslidar')

    def run_fake(self, *extra, stop=None, session_factory=None):
        with patch.object(build_map, 'bag_metadata', side_effect=lambda *args:
                dict(lidar_scans=self.bag_count, start_s=10., end_s=11.)), \
                patch('perception.sensors.observations.selected_messages', side_effect=self.raw), \
                patch('perception.sensors.observations.bag_observations', side_effect=self.observations), \
                patch('perception.mapping_localization.runtime.LocalizationSession', side_effect=session_factory or self.make_session), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return build_map.build(self.args(*extra), stop=stop)

    def test_success_seals_source_bound_scans_and_final_pose_consumer_can_read_them(self):
        summary = self.run_fake()
        self.assertTrue(summary['success'])
        self.assertEqual(summary['counts']['selected_lidar_scans'], 3)
        self.assertEqual(summary['counts']['scans'], 3)
        self.assertEqual(summary['counts']['journal_scans'], 3)
        self.assertEqual(self.session.calls, ['step', 'step', 'step', 'save', 'close'])
        run = json.loads((self.out/'run.json').read_text())
        self.assertIn('environment_scans_fingerprint', run)
        self.assertIn('map_fingerprint', run['playback'])
        self.assertEqual(sealed_scan_summary(self.out/'environment/scans.sqlite')['records'], 3)
        rows = [json.loads(line) for line in (self.out/'localization.jsonl').read_text().splitlines()]
        timeline = PoseTimeline(rows, load_config()['poses'], np.loadtxt(self.out/'map/keyframes/poses.txt', ndmin=2))
        self.assertEqual(timeline.bound_keyframes, 1)
        self.assertIsNotNone(timeline.at(10.1, exact_ns=10100000000))
        self.assertEqual(self.bag.read_bytes(), self.original)

    def test_backend_window_waits_instead_of_dropping_scans(self):
        sizes = iter([4, 4, 0])
        def session_factory(*args, **kwargs):
            session = self.make_session(*args, **kwargs)
            session.jobs.qsize = lambda: next(sizes, 0)
            return session
        summary = self.run_fake(session_factory=session_factory)
        self.assertEqual(summary['counts']['scans'], self.raw_count)
        self.assertGreater(summary['backpressure_wait_s'], 0.)

    def test_estimator_failure_never_seals_or_exports_partial_input(self):
        def session_factory(*args, **kwargs):
            session = self.make_session(*args, **kwargs)
            session.failure_at = 2
            return session
        with self.assertRaisesRegex(RuntimeError, 'synthetic estimator failure'):
            self.run_fake(session_factory=session_factory)
        run = json.loads((self.out/'run.json').read_text())
        self.assertFalse(run['success'])
        self.assertNotIn('environment_scans_fingerprint', run)
        self.assertNotIn('save', self.session.calls)
        self.assertEqual(self.session.calls[-1], 'close')
        with self.assertRaises(ValueError):
            sealed_scan_summary(self.out/'environment/scans.sqlite')

    def test_selected_tail_without_imu_coverage_is_an_error_even_for_short_runs(self):
        self.drop_last = True
        with self.assertRaisesRegex(RuntimeError, 'Not all selected scans'):
            self.run_fake('--duration', '1')
        self.assertFalse(json.loads((self.out/'summary.json').read_text())['success'])
        self.assertNotIn('save', self.session.calls)

    def test_full_bag_count_mismatch_is_not_reported_as_complete(self):
        self.bag_count = 4
        with self.assertRaisesRegex(RuntimeError, 'Not all bag scans'):
            self.run_fake()

    def test_selected_short_interval_does_not_claim_full_bag_coverage(self):
        self.bag_count = 8
        summary = self.run_fake('--start', '2', '--duration', '1')
        self.assertTrue(summary['success'])
        self.assertFalse(summary['full_bag_selected'])
        self.assertEqual(json.loads((self.out/'run.json').read_text())['start'], 2.)

    def test_interruption_preserves_diagnostics_and_marks_failure(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            self.run_fake(stop=stop)
        run = json.loads((self.out/'run.json').read_text())
        self.assertFalse(run['success'])
        self.assertEqual(run['termination'], 'cancelled')
        self.assertNotIn('save', self.session.calls)

    def test_interruption_during_export_does_not_mark_run_successful(self):
        stop = threading.Event()
        def session_factory(*args, **kwargs):
            session = self.make_session(*args, **kwargs)
            original = session.save
            def save(path):
                original(path)
                stop.set()
            session.save = save
            return session
        with self.assertRaisesRegex(InterruptedError, 'during map export'):
            self.run_fake(stop=stop, session_factory=session_factory)
        run = json.loads((self.out/'run.json').read_text())
        self.assertFalse(run['success'])
        self.assertEqual(run['termination'], 'cancelled')
        self.assertNotIn('environment_scans_fingerprint', run)

    def test_no_scan_option_is_explicit_and_backend_overrides_are_recorded(self):
        summary = self.run_fake('--no-record-scans', '--lio-backend', 'cpu', '--roll-backend', 'cpu', '--deskew-backend', 'cpu')
        self.assertEqual(summary['counts']['journal_scans'], 0)
        self.assertFalse((self.out/'environment').exists())
        self.assertEqual(self.session.config['matching_backend'], 'cpu')
        self.assertEqual(self.session.config['roll']['matchingBackend'], 'cpu')
        self.assertEqual(self.session.config['deskew_backend'], 'cpu')

    def test_no_valid_frontend_does_not_produce_a_map(self):
        def session_factory(*args, **kwargs):
            session = self.make_session(*args, **kwargs)
            session.valid = False
            return session
        with self.assertRaisesRegex(RuntimeError, 'No valid frontend'):
            self.run_fake(session_factory=session_factory)
        self.assertNotIn('save', self.session.calls)

    def test_bad_paths_and_intervals_are_rejected_before_output_creation(self):
        for arguments in (('--start', '-1'), ('--duration', 'nan'), ('--backend-timeout', '0')):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_map.validate(self.args(*arguments))
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            build_map.validate(self.args())
        self.out.rmdir()
        self.out.symlink_to(self.root/'does-not-exist')
        with self.assertRaises(FileExistsError):
            build_map.validate(self.args())

    def test_explicit_clock_mismatch_is_rejected_before_native_constructor(self):
        from perception.mapping_localization.configuration import DEFAULT_CALIBRATION
        config = yaml.safe_load(DEFAULT_CALIBRATION.read_text())
        config['common']['imu_time_offset'] = -.04
        config['calibration']['deployment_clock_group'] = 'synthetic_A'
        calibration = self.root/'calibration.yaml'
        calibration.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, 'clock group mismatch'):
            self.run_fake('--lidar-imu-calibration', str(calibration), '--sensor-clock-group', 'synthetic_B')
        self.assertIsNone(self.session)
        self.assertFalse(self.out.exists())

    def test_backend_worker_error_is_reported_before_accepting_more_input(self):
        session = SimpleNamespace(error='synthetic worker error', jobs=SimpleNamespace(qsize=lambda: 4))
        with self.assertRaisesRegex(RuntimeError, 'synthetic worker error'):
            build_map.wait_for_backend(session, threading.Event(), 1.)

    def test_localization_can_wait_with_one_pending_scan(self):
        queue = SimpleNamespace(pending=1)
        queue.qsize = lambda: queue.pending
        class Stop:
            def is_set(self): return False
            def wait(self, _): queue.pending = 0
        build_map.wait_for_backend(SimpleNamespace(error=None,jobs=queue),Stop(),1.,max_pending=1)
        self.assertEqual(queue.pending,0)


if __name__ == '__main__':
    unittest.main()
