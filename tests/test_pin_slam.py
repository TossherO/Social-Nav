"""PIN input alignment and complete, restricted map I/O; no ROS master or GPU."""
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import threading

import numpy as np
import torch

from perception.mapping_localization.backends.pin_slam.inputs import decode_scan
from perception.mapping_localization.backends.pin_slam.map_io import digest, load_map, save_map
from perception.mapping_localization.__main__ import wait_for_scan_ack


class ScanAcknowledgementTests(unittest.TestCase):
    def test_stale_or_wrong_scan_cannot_complete_the_next_input(self):
        condition = threading.Condition()
        for receipt in ({'count':1,'stamp_ns':10}, {'count':2,'stamp_ns':9}):
            with self.assertRaisesRegex(RuntimeError, 'per-frame budget'):
                wait_for_scan_ack(condition, receipt, 10, 1, lambda:True, timeout=.002)
        wait_for_scan_ack(condition, {'count':2,'stamp_ns':10}, 10, 1, lambda:True, timeout=.002)

    def test_worker_exit_ends_the_wait_without_inventing_a_result(self):
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            wait_for_scan_ack(threading.Condition(), {'count':0,'stamp_ns':None}, 10, 0, lambda:False)


def cloud(big_endian=False):
    endian = '>' if big_endian else '<'
    dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'timestamp'],
                      'formats': [endian+'f4']*4 + [endian+'f8'],
                      'offsets': [0, 4, 8, 12, 18], 'itemsize': 26})
    data = bytearray(3*(16*26+8))
    rows = np.ndarray((3, 16), dtype=dtype, buffer=data, strides=(16*26+8, 26))
    ids = np.arange(48).reshape(3, 16)
    rows['x'], rows['y'], rows['z'], rows['intensity'] = ids+1, 1, 2, ids
    rows['timestamp'] = 1791630335.7 + ids/470
    rows['x'][0, 0] = np.nan  # Earliest acquisition still defines the scan start.
    rows['intensity'][1, 0] = np.inf
    rows['timestamp'][2, 0] = np.nan
    fields = [SimpleNamespace(name=name, offset=dtype.fields[name][1], count=1,
                              datatype=8 if name == 'timestamp' else 7) for name in dtype.names]
    return SimpleNamespace(width=16, height=3, point_step=26, row_step=16*26+8,
                           is_bigendian=big_endian, data=bytes(data), fields=fields,
                           header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:1791630335.8)))


class PinInputTests(unittest.TestCase):
    def test_shared_filter_preserves_double_time_endian_and_row_padding(self):
        for endian in (False, True):
            scan = decode_scan(cloud(endian))
            self.assertEqual(len(scan.points), 45)
            np.testing.assert_allclose(scan.times, scan.points[:, 3]/47, atol=3e-6)
            self.assertGreater(scan.times[0], 0)  # No post-filter renormalization.
            self.assertAlmostEqual(scan.times[-1], 1)
            self.assertAlmostEqual(scan.duration_s, .1, places=5)

    def test_missing_or_wrong_clock_is_rejected_instead_of_inventing_times(self):
        msg = cloud()
        msg.fields = msg.fields[:-1]
        with self.assertRaisesRegex(ValueError, 'Missing'):
            decode_scan(msg)
        self.assertIsNone(decode_scan(msg, deskew=False).times)
        msg = cloud()
        msg.header.stamp.to_sec = lambda: 10
        with self.assertRaisesRegex(ValueError, 'absolute scan'):
            decode_scan(msg)


class PinMapTests(unittest.TestCase):
    def fixture(self):
        neural = SimpleNamespace(neural_points=torch.randn(5, 3),
            point_orientations=torch.tensor([[1., 0, 0, 0]]*5), geo_features=torch.randn(6, 8),
            point_ts_create=torch.arange(5, dtype=torch.int32), point_ts_update=torch.arange(5, dtype=torch.int32),
            point_certainties=torch.ones(5), travel_dist=torch.arange(5, dtype=torch.float32),
            cur_ts=4, max_ts=4, reboot_ts=0, count=lambda:5, assign_local_to_global=lambda:None,
            after_pgo=False,
            position_encoder_geo=torch.nn.Identity(), position_encoder_color=torch.nn.Identity())
        decoder = torch.nn.Linear(11, 1)
        cfg = SimpleNamespace(feature_dim=8, color_on=False, semantic_on=False, device='cuda')
        return neural, decoder, cfg

    def test_roundtrip_preserves_full_tensors_and_refuses_overwrite(self):
        neural, decoder, cfg = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map'
            save_map(path, neural, decoder, cfg, {'initial_pose':[0.]*6})
            before = digest(path/'map.pt')
            restored, metadata = load_map(path)
            for name in ('neural_points','point_orientations','geo_features','point_ts_create',
                         'point_ts_update','point_certainties','travel_dist'):
                torch.testing.assert_close(restored[name], getattr(neural, name))
            torch.testing.assert_close(restored['decoder']['weight'], decoder.weight)
            self.assertEqual(metadata['reference'], 'scan_end')
            self.assertFalse(restored['after_pgo'])
            self.assertEqual(digest(path/'map.pt'), before)
            with self.assertRaises(FileExistsError):
                save_map(path, neural, decoder, cfg, {})

    def test_pgo_point_frames_and_legacy_state_survive_restore_without_rewriting_map(self):
        neural, decoder, cfg = self.fixture()
        neural.after_pgo = True
        neural.point_orientations[:] = torch.tensor([.9238795, 0., 0., .3826834])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map'
            save_map(path, neural, decoder, cfg, {'map_revision':2})
            restored, metadata = load_map(path)
            self.assertTrue(restored['after_pgo'])
            torch.testing.assert_close(restored['point_orientations'], neural.point_orientations)
            # Emulate the earlier tensor-only project format, whose PGO revision
            # is available but whose point-query frame switch was not saved.
            del restored['after_pgo']
            del metadata['after_pgo']
            torch.save(restored, path/'map.pt')
            checksum = digest(path/'map.pt')
            metadata['tensor_sha256'] = checksum
            (path/'manifest.json').write_text(json.dumps(metadata))
            (path/'COMPLETE').write_text(checksum+'\n')
            restored, _ = load_map(path)
            self.assertTrue(restored['after_pgo'])
            self.assertEqual(digest(path/'map.pt'), checksum)
            # A missing correction history cannot silently disable rotated frames.
            metadata['map_revision'] = 0
            (path/'manifest.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'recorded PGO'):
                load_map(path)

    def test_restricted_load_rejects_object_pickle_even_with_matching_binding(self):
        neural, decoder, cfg = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'map'
            save_map(path, neural, decoder, cfg, {})
            payload, _ = load_map(path)
            payload['neural_points'] = SimpleNamespace(unknown='object format is unsupported')
            torch.save(payload, path/'map.pt')
            checksum = digest(path/'map.pt')
            metadata = json.loads((path/'manifest.json').read_text())
            metadata['tensor_sha256'] = checksum
            (path/'manifest.json').write_text(json.dumps(metadata))
            (path/'COMPLETE').write_text(checksum+'\n')
            with self.assertRaises(Exception) as rejected:
                load_map(path)
            self.assertIn('Weights only load failed', str(rejected.exception))


@unittest.skipUnless(importlib.util.find_spec('open3d'), 'PIN independent environment required')
class PinRegistrationTests(unittest.TestCase):
    def test_final_surface_fitting_preserves_decoder_geometry_and_observation_state(self):
        from perception.mapping_localization.backends.pin_slam.core import Config
        from perception.mapping_localization.backends.pin_slam.map_refinement import refine_global_features
        cfg = Config()
        cfg.device, cfg.weighted_first = 'cpu', True
        cfg.setup_dtype()
        cfg.final_map_refine_iters, cfg.final_map_refine_bs, cfg.final_map_refine_offset_m = 40, 64, .08
        x,y = torch.meshgrid(torch.arange(10)*.2, torch.arange(10)*.2, indexing='ij')
        points = torch.stack((x.flatten(),y.flatten(),torch.zeros(100)),dim=1)
        neural = SimpleNamespace(neural_points=points, point_certainties=torch.full((100,),10.),
            point_ts_update=torch.arange(100), buffer_pt_index=torch.arange(100), max_ts=99,
            geo_features=torch.full((100,1),.3), assign_local_to_global=Mock(),
            recreate_hash=Mock(), reset_local_map=Mock(), requires_grad_=Mock())
        def query(coordinates, **_kwargs):
            nearest = torch.cdist(coordinates.detach(),points).argmin(dim=1)
            feature = coordinates[:,2,None] + neural.geo_features[nearest]
            return feature, None, None, torch.full((len(coordinates),),6), torch.ones(len(coordinates))
        neural.query_feature = query
        decoder = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            decoder.weight.fill_(1.)
        decoder.sdf = lambda features:decoder(features).squeeze(1)
        before = points.clone(), decoder.weight.clone(), neural.point_certainties.clone(), neural.point_ts_update.clone()
        result = refine_global_features(neural,decoder,cfg)
        self.assertLess(result['final_loss'],result['initial_loss'])
        torch.testing.assert_close(points,before[0])
        torch.testing.assert_close(decoder.weight,before[1])
        torch.testing.assert_close(neural.point_certainties,before[2])
        torch.testing.assert_close(neural.point_ts_update,before[3])
        self.assertFalse(decoder.weight.requires_grad)
        self.assertFalse(neural.geo_features.requires_grad)
        self.assertTrue(cfg.decoder_freezed)
        neural.reset_local_map.assert_called_once()

    def test_rejection_metrics_preserve_the_original_acceptance_rules(self):
        from perception.mapping_localization.backends.pin_slam.core import Config, Tracker
        cfg = Config()
        cfg.device = 'cpu'
        cfg.setup_dtype()
        tracker = Tracker(cfg, Mock(), {'sdf':Mock(), 'semantic':None, 'color':None})
        points = torch.zeros((40, 3))
        tracker.registration_step = Mock(return_value=(torch.eye(4, dtype=torch.float64),
            None, None, None, points[:4], 20., None))
        _, _, _, accepted = tracker.tracking(points)
        self.assertFalse(accepted)
        self.assertEqual(tracker.last_diagnostics['iterations'], 1)
        self.assertEqual(tracker.last_diagnostics['valid_ratio'], .1)
        self.assertEqual(tracker.last_diagnostics['reasons'],
                         ['INSUFFICIENT_VALID_POINTS', 'FINAL_SDF_RESIDUAL'])
        tracker.registration_step.return_value = (torch.eye(4, dtype=torch.float64),
            None, None, None, points, 2., None)
        _, _, _, accepted = tracker.tracking(points)
        self.assertTrue(accepted)
        self.assertEqual(tracker.last_diagnostics['reasons'], [])
        self.assertEqual(tracker.last_diagnostics['valid_points'], 40)


@unittest.skipUnless(importlib.util.find_spec('open3d'), 'PIN independent environment required')
class PinLifecycleTests(unittest.TestCase):
    def fixture(self, accepted=True):
        from perception.mapping_localization.backends.pin_slam.core import PinSlam
        from perception.mapping_localization.backends.pin_slam.inputs import Scan
        core = PinSlam.__new__(PinSlam)
        core.config = SimpleNamespace(device='cpu', dtype=torch.float32, deskew=False,
            min_range=.8, max_range=35, min_z=-3, max_z=5, vox_down_m=.15,
            source_vox_down_m=.5, surface_sample_range_m=.25, stop_frame_thre=20, pgo_on=False,
            motion_prediction_max_gap_s=.5)
        core.mode, core.frames, core.valid_frames, core.lost_frames, core.map_revision = 'localization', 0, 0, 0, 0
        core.map_pose, core.local_pose, core.delta = np.eye(4), np.eye(4), np.eye(4)
        core.last_valid_stamp = core.last_input_stamp = core.delta_dt = None
        core.clock_fault = False
        core.dataset = SimpleNamespace(travel_dist=np.zeros(20), odom_poses=np.tile(np.eye(4),(20,1,1)),
                                       lose_track=False, last_odom_tran=np.eye(4), stop_count=0)
        refined = torch.eye(4, dtype=torch.float64)
        refined[0, 3] = .03
        core.tracker = SimpleNamespace(tracking=Mock(return_value=(refined, None, None, accepted)))
        core.neural = SimpleNamespace(geo_features=torch.randn(6,8))
        core.mapper = None
        points = np.column_stack([np.arange(40)*.6+1, np.ones(40), np.ones(40), np.ones(40)]).astype(np.float32)
        return core, Scan(points, None, 10., 0.)

    def test_first_localization_scan_is_registered_and_map_features_remain_fixed(self):
        core, scan = self.fixture()
        original = core.neural.geo_features.clone()
        result = core.process(scan)
        core.tracker.tracking.assert_called_once()
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.map_pose[0, 3], .03)
        torch.testing.assert_close(core.neural.geo_features, original)
        self.assertIsNone(core.mapper)

    def test_failed_match_and_clock_rollback_never_publish_previous_valid_pose(self):
        core, scan = self.fixture(accepted=False)
        result = core.process(scan)
        self.assertFalse(result.valid)
        self.assertIsNone(result.local_pose)
        self.assertEqual(result.reason, 'PIN_REGISTRATION_REJECTED')
        scan.stamp_s = 9.
        rollback = core.process(scan)
        scan.stamp_s = 11.
        later = core.process(scan)
        self.assertEqual(rollback.reason, 'NONMONOTONIC_SCAN_TIME_RESTART_REQUIRED')
        self.assertFalse(later.valid)
        self.assertIsNone(later.map_pose)
        self.assertEqual(core.tracker.tracking.call_count, 1)

    def test_map_correction_is_separate_from_continuous_local_motion(self):
        core, scan = self.fixture()
        first = core.process(scan)
        # Simulate a backend PGO correction, then the next actual registration.
        core.map_pose[0, 3] += 5
        refined = torch.as_tensor(core.map_pose.copy())
        refined[0, 3] += .1
        core.tracker.tracking.return_value = (refined, None, None, True)
        scan.stamp_s += .1
        result = core.process(scan)
        self.assertAlmostEqual(result.local_pose[0,3]-first.local_pose[0,3], .1)
        self.assertAlmostEqual(result.map_pose[0,3]-result.local_pose[0,3], 5)

    def test_short_rejection_retains_a_bounded_initial_guess_without_publishing_it(self):
        core, scan = self.fixture()
        core.process(scan)
        pose = torch.eye(4, dtype=torch.float64)
        pose[0, 3] = .06
        core.tracker.tracking.return_value = (pose, None, None, True)
        scan.stamp_s = 10.1
        core.process(scan)
        core.tracker.tracking.return_value = (pose, None, None, False)
        for stamp in (10.2, 10.3, 11.):
            scan.stamp_s = stamp
            result = core.process(scan)
            self.assertFalse(result.valid)
            self.assertIsNone(result.map_pose)
            guess = core.tracker.tracking.call_args.args[1]
            self.assertAlmostEqual(guess[0, 3].item(), .06 if stamp == 11. else .06+.3*(stamp-10.1))


if __name__ == '__main__':
    unittest.main()
