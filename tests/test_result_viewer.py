"""Result fidelity, read-only map binding and deterministic playback; no GUI/ROS/GPU."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from perception.mapping_localization.results import (
    Result, Timeline, file_fingerprint, load_result, map_fingerprint, read_pcd,
)
from perception.visualization.people import RecordedPeople
from perception.visualization.recording import merge_people, sensor_frame, display_transform
from perception.visualization.scans import rotation_matrix
from tests.test_people_visualization import observation


def sample(stamp, health=0, generation=0, revision=0):
    return dict(stamp_s=stamp, health=health, source='synthetic', session_id='session',
                generation=generation, map_revision=revision, status='TRACKING', reason='',
                frame_id='odom', child_frame_id='rslidar', position=[stamp, 0, 0],
                quaternion_xyzw=[0, 0, 0, 1], has_map_pose=True, map_frame='map',
                map_position=[stamp, 0, 0], map_quaternion_xyzw=[0, 0, 0, 1])


def pcd(path, binary=True):
    header = ('FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              'WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ' + ('binary' if binary else 'ascii') + '\n')
    points = np.array([[1, 2, 3, 9], [4, 5, 6, 8], [7, 8, 9, 7]], dtype='<f4')
    path.write_bytes(header.encode() + (points.tobytes() if binary else b'1 2 3 9\n4 5 6 8\n7 8 9 7\n'))


def complete_map(path):
    (path / 'keyframes').mkdir(parents=True)
    (path / 'COMPLETE').write_text('SYNTHETIC_MAP')
    (path / 'manifest.yaml').write_text('format: roll_keyframes_v1\nframe: map\n')
    (path / 'keyframes/poses.txt').write_text('10 0 0 0 0 0 0 0 1\n12 1 0 0 0 0 0 1 1\n')
    pcd(path / 'GlobalMap.pcd')


class ResultTests(unittest.TestCase):
    def test_failed_run_with_only_sensor_detections_can_be_loaded_without_localization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'run.json').write_text(json.dumps(dict(mode='odometry', success=False)))
            (root/'localization.jsonl').write_text('')
            person = observation(1)
            person.update(health='degraded', world_from_lidar=None, frame_id='rslidar', tracks=[])
            (root/'people.jsonl').write_text(json.dumps(person)+'\n')
            result = load_result(root)
            self.assertEqual((result.kind, result.rows), ('recorded_states', []))
            people = RecordedPeople(root/'people.jsonl', dict(history_s=3, max_history_points=32))
            result = merge_people(result, people)
            self.assertEqual(sensor_frame(result), 'rslidar')
            self.assertIsNone(display_transform(result, result.rows[0], 'odom', rotation_matrix))

    def test_detection_only_and_failed_localization_observations_remain_seekable_without_world_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'people.jsonl'
            rows = [observation(1), observation(2), observation(3)]
            rows[0].update(health='degraded', world_from_lidar=None, tracks=[], frame_id='rslidar')
            rows[1].update(health='degraded', world_from_lidar=None, tracks=[], frame_id='rslidar')
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            people = RecordedPeople(path, dict(history_s=3, max_history_points=32))
            original = Result(Path(folder), [sample(2, health=2), sample(3)])
            result = merge_people(original, people)
            self.assertEqual([r['stamp_s'] for r in result.rows], [1, 2, 3])
            self.assertEqual(result.duration, 2)
            self.assertEqual(result.index_at(.5), 0)
            self.assertEqual(sensor_frame(result), 'rslidar')
            self.assertEqual(len(original.rows), 2)
            for row in result.rows[:2]:
                self.assertIsNone(display_transform(result, row, 'map', rotation_matrix))
                np.testing.assert_equal(display_transform(result, row, 'lidar', rotation_matrix), np.eye(4))
                specs = people.at(row['stamp_s'], sensor_only=True)
                self.assertTrue(any(spec['namespace'] == 'detections' for spec in specs))
                self.assertTrue(all(spec['frame_id'] == 'rslidar' for spec in specs))
                self.assertFalse(any(spec['namespace'] == 'tracks' for spec in specs))
            self.assertEqual(people.at(1.5, sensor_only=True), [])

    def test_cloud_uses_the_people_recorded_pose_without_changing_localization_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'people.jsonl'
            person = observation(3)
            person['world_from_lidar'][0][3] = 2.9
            path.write_text(json.dumps(person)+'\n')
            people = RecordedPeople(path, dict(history_s=3, max_history_points=32))
            state = sample(3)
            state['map_position'][0] = 13
            result = merge_people(Result(Path(folder), [state]), people)
            np.testing.assert_allclose(display_transform(result, result.rows[0], 'map', rotation_matrix)[:3, 3], [12.9, 0, 0])
            self.assertEqual(result.rows[0]['map_position'][0], 13)

    def test_timeline_pause_rate_seek_and_restart_are_wall_time_bound(self):
        timeline = Timeline(10, now=0)
        timeline.toggle(now=0)
        self.assertEqual(timeline.tick(now=2), 2)
        timeline.set_rate(2, now=2)
        self.assertEqual(timeline.tick(now=3), 4)
        timeline.toggle(now=3)
        self.assertEqual(timeline.tick(now=8), 4)
        timeline.seek(7, now=8)
        timeline.toggle(now=8)
        self.assertEqual(timeline.tick(now=10), 10)
        self.assertFalse(timeline.playing)
        timeline.toggle(now=10)
        self.assertEqual(timeline.position, 0)
        with self.assertRaises(ValueError):
            timeline.set_rate(float('nan'))

    def test_lookup_never_selects_a_future_observation(self):
        result = Result(Path('synthetic'), [sample(10), sample(11), sample(14)])
        self.assertEqual(result.index_at(.9), 0)
        self.assertEqual(result.index_at(2), 1)
        self.assertEqual(result.index_at(4), 2)

    def test_trajectory_breaks_at_failures_generations_and_map_revisions(self):
        rows = [sample(0), sample(1), sample(2, health=2), sample(3),
                sample(4, generation=1), sample(5, generation=1), sample(6, generation=1, revision=1)]
        result = Result(Path('synthetic'), rows)
        self.assertEqual([item[0] for item in result.trajectory('map')], [1, 5])
        self.assertEqual([item[0] for item in result.trajectory('odom')], [1, 5, 6])

    def test_rejects_time_reversal_and_fabricated_valid_poses(self):
        with self.assertRaisesRegex(ValueError, 'reversed'):
            Result(Path('synthetic'), [sample(2), sample(1)])
        bad = sample(1)
        bad['map_position'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'map pose'):
            Result(Path('synthetic'), [bad])
        bad = sample(1, health=2)
        bad['quaternion_xyzw'] = [0, 0, 0, 0]
        result = Result(Path('synthetic'), [bad])
        self.assertEqual(result.trajectory('odom'), [])

    def test_map_only_has_sparse_unknown_health_and_preserves_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'map'
            complete_map(directory)
            before = map_fingerprint(directory)
            result = load_result(directory)
            self.assertEqual(result.kind, 'optimized_keyframes')
            self.assertEqual(result.duration, 2)
            self.assertTrue(all(row['health'] is None and row['generation'] is None for row in result.rows))
            self.assertEqual(map_fingerprint(directory), before)

    def test_run_relocation_checks_original_binding_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete_map(root / 'run/map')
            metadata = dict(mode='mapping', playback={'records': 'localization.jsonl',
                            'map_fingerprint': map_fingerprint(root / 'run/map')})
            (root / 'run/run.json').write_text(json.dumps(metadata))
            (root / 'run/localization.jsonl').write_text(json.dumps(sample(10))+'\n')
            shutil.copytree(root / 'run/map', root / 'moved')
            self.assertEqual(load_result(root / 'run', map_override=root / 'moved').kind, 'recorded_states')
            (root / 'moved/GlobalMap.pcd').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'binding'):
                load_result(root / 'run', map_override=root / 'moved')
            self.assertEqual(json.loads((root / 'run/run.json').read_text()), metadata)

    def test_summary_only_localization_never_uses_mapping_keyframes_as_localization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete_map(root / 'map')
            (root / 'run.json').write_text(json.dumps(dict(mode='localization', map=str(root / 'map'))))
            result = load_result(root)
            self.assertEqual((result.kind, result.rows), ('map_only', []))
            (root / 'run.json').write_text(json.dumps(dict(mode='odometry')))
            with self.assertRaisesRegex(ValueError, 'summary'):
                load_result(root)
            with self.assertRaisesRegex(ValueError, 'alignment'):
                load_result(root, map_override=root / 'map')

    def test_comparison_requires_explicit_series_and_never_auto_overlays_a_map(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'comparison.json').write_text('{}')
            names = ['phase', 'backend', 'stamp_s', 'health', 'status', 'reason', 'map_revision']
            names += [prefix+'_'+axis for prefix in ('odom', 'map') for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
            with (root / 'trajectories.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                row = dict.fromkeys(names, '0')
                row.update(phase='localization', backend='PIN_latest', stamp_s='10', health='0',
                           status='TRACKING', reason='', odom_qw='1', map_qw='1')
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, 'phase'):
                load_result(root)
            result = load_result(root, phase='localization', backend='PIN_latest')
            self.assertIsNone(result.map_dir)
            self.assertEqual(len(result.rows), 1)
            with self.assertRaisesRegex(ValueError, 'separate'):
                load_result(root, phase='localization', backend='PIN_latest', map_override=root)

    def test_ascii_binary_pcd_display_budget_and_read_only_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            for binary in (False, True):
                path = Path(folder) / ('binary.pcd' if binary else 'ascii.pcd')
                pcd(path, binary)
                before = file_fingerprint(path)
                np.testing.assert_array_equal(read_pcd(path), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                self.assertLessEqual(len(read_pcd(path, limit=2)), 2)
                self.assertEqual(file_fingerprint(path), before)
                if binary:
                    path.write_bytes(path.read_bytes()[:-4])
                    with self.assertRaises(ValueError):
                        read_pcd(path)


if __name__ == '__main__':
    unittest.main()
