"""Association policy and immutable result replay, without models or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from perception.detection_tracking.inference.config import DEFAULT
from perception.detection_tracking.inference.association import Association, bev_diou
from perception.detection_tracking.inference.retrack import retrack_rows, replay, read_people
from perception.detection_tracking.inference.tracker import SimpleTrack

CONFIG = json.loads(DEFAULT.read_text())['tracking']


def box(x=0, y=0, size=.6, z=0, height=1.7, score=.9):
    return np.array([x, y, z, size, size, height, 0, score])


def record(stamp, generation=0, health='valid'):
    return dict(observation_id=str(stamp), sequence_id='bag', session_id='original', generation=generation,
        stamp_s=stamp, clock_domain='ros_sim_time', timestamp_source='sensor', frame_id='odom',
        detection_frame_id='lidar', health=health, reason=None, pose_source='localization',
        pose_health='valid' if health == 'valid' else 'invalid', pose_stamp_s=stamp,
        world_from_lidar=np.eye(4).tolist() if health == 'valid' else None,
        input_fingerprint='a'*64, detections=[box().tolist()], detection_input={'used_sweeps': 3},
        localization_binding=dict(session_id='lio', generation=generation, frame_id='odom'))


class AssociationTests(unittest.TestCase):
    def test_xy_motion_wins_over_spurious_size_agreement(self):
        candidates = [box(.05, size=.2), box(.25, size=1.2)]
        chosen = {}
        for metric in ('giou', 'diou', 'xy_diou'):
            tracker = SimpleTrack(dict(CONFIG, association=metric, association_backend='cpu'))
            tracker.step([box()], 1.)
            chosen[metric] = tracker.step(candidates, 1.1)[0]['detection_index']
        self.assertEqual(chosen['xy_diou'], 0)
        self.assertEqual(chosen['giou'], 1)

    def test_xy_cost_ignores_vertical_box_jitter_and_gates_motion(self):
        settings = dict(CONFIG, association='xy_diou', association_backend='cpu')
        engine = Association(settings)
        predicted = np.array([box()]);covariance = np.array([np.eye(10)*.02])
        first = engine.costs(np.array([box(.1)]), predicted, covariance)
        second = engine.costs(np.array([box(.1, z=3, height=.2)]), predicted, covariance)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(engine.costs(np.array([box(1.)]), predicted, covariance)[0, 0], 1e6)
        self.assertEqual(engine.costs(np.array([box(5.)]), predicted, covariance)[0, 0], 1e6)

    def test_bev_diou_center_distance_is_xy_not_size(self):
        self.assertAlmostEqual(bev_diou(box(), box(z=2, height=.1)), 1.)
        self.assertAlmostEqual(bev_diou(box(), box(size=.3)), .25)
        self.assertLess(bev_diou(box(), box(2)), 0)

    def test_matching_can_leave_invalid_pairs_unmatched(self):
        settings = dict(CONFIG, association='xy_diou', association_backend='cpu')
        tracker = SimpleTrack(settings)
        tracker.step([box(), box(3)], 1.)
        result = tracker.step([box(.05), box(30)], 1.1)
        self.assertEqual(result[0]['detection_index'], 0)
        self.assertIsNone(result[1]['detection_index'])
        self.assertEqual(result[2]['detection_index'], 1)

    def test_low_support_keeps_prediction_and_uncertainty(self):
        tracker = SimpleTrack(dict(CONFIG, association='xy_diou', association_backend='cpu'))
        first = tracker.step([box()], 1.)[0]
        second = tracker.step([box(.02, score=.2)], 1.1)[0]
        self.assertEqual(second['state'], 'predicted')
        self.assertIsNone(second['detection_index'])
        self.assertEqual(second['last_observed_stamp_s'], first['last_observed_stamp_s'])
        self.assertGreater(np.trace(np.array(second['covariance']).reshape(10, 10)),
                           np.trace(np.array(first['covariance']).reshape(10, 10)))

    def test_explicit_backend_and_parameter_validation(self):
        for override in [dict(association_backend='auto'), dict(diou_weight=2), dict(motion_gate_chi2=0)]:
            with self.assertRaises(ValueError):
                SimpleTrack(dict(CONFIG, **override))


class ReplayTests(unittest.TestCase):
    def test_growing_source_cannot_receive_a_complete_provenance_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.jsonl'
            source.write_text(json.dumps(record(1.))+'\n')
            def changing(path):
                yield from read_people(path)
                with source.open('a') as stream:
                    stream.write(json.dumps(record(1.1))+'\n')
            output = Path(directory)/'retrack'
            with patch('perception.detection_tracking.inference.retrack.read_people', changing):
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    replay(source, output, json.loads(DEFAULT.read_text()))
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'failed')

    def test_replay_writes_bound_new_output_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.jsonl'
            original = json.dumps(record(1.))+'\n'
            source.write_text(original)
            output = Path(directory)/'retrack'
            summary = replay(source, output, json.loads(DEFAULT.read_text()))
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(source.read_text(), original)
            self.assertEqual(summary['frames'], 1)
            self.assertEqual(manifest['status'], 'complete')
            self.assertFalse(manifest['model_loaded'])
            self.assertFalse(manifest['scan_content_revalidated'])
            with self.assertRaises(FileExistsError):
                replay(source, output, json.loads(DEFAULT.read_text()))

    def test_replay_preserves_input_and_isolates_source_lifecycles(self):
        rows = [record(1.), record(1.1), record(1.2, generation=1)]
        frozen = copy.deepcopy(rows)
        result = list(retrack_rows(rows, CONFIG, session_id='replay'))
        self.assertEqual(rows, frozen)
        for source, target in zip(rows, result):
            for name in ('detections', 'input_fingerprint', 'detection_input', 'world_from_lidar', 'localization_binding'):
                self.assertEqual(source[name], target[name])
            self.assertEqual(target['detection_source'], 'recorded_detections')
        self.assertEqual(result[0]['tracks'][0]['track_id'], result[1]['tracks'][0]['track_id'])
        self.assertNotEqual(result[1]['tracks'][0]['track_id'], result[2]['tracks'][0]['track_id'])
        self.assertTrue(result[0]['tracks'][0]['track_id'].startswith('replay:'))

    def test_invalid_pose_and_failed_observations_never_reuse_tracks(self):
        rows = [record(1.), record(1.1, health='degraded'), record(1.2), record(1.3, health='invalid'), record(1.4)]
        result = list(retrack_rows(rows, CONFIG))
        self.assertEqual(result[1]['tracks'], [])
        self.assertEqual(result[3]['health'], 'invalid')
        self.assertEqual(result[3]['tracks'], [])
        self.assertEqual(len({result[i]['tracks'][0]['track_id'] for i in (0, 2, 4)}), 3)

    def test_future_or_mismatched_pose_and_bad_fingerprint_are_rejected(self):
        for field, value in [('pose_stamp_s', 2.), ('input_fingerprint', 'unbound'),
                             ('localization_binding', dict(frame_id='other'))]:
            row = record(1.);row[field] = value
            with self.assertRaises(ValueError):
                list(retrack_rows([row], CONFIG))
        with self.assertRaises(ValueError):
            list(retrack_rows([record(1.), record(1.)], CONFIG))


if __name__ == '__main__':
    unittest.main()
