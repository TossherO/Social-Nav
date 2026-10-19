"""Manual seeds retain observation/map identity; pure CPU, no ROS or models."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from perception.mapping_localization.__main__ import arguments
from perception.mapping_localization.initial_pose import load_seed, quaternion_rpy, rotation_matrix, save_seed
from perception.mapping_localization.results import file_fingerprint, map_fingerprint, rpy_quaternion


class InitialPoseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.map = self.root / 'map'
        (self.map / 'keyframes').mkdir(parents=True)
        (self.map / 'COMPLETE').write_text('synthetic')
        (self.map / 'manifest.yaml').write_text('frame: map\n')
        (self.map / 'keyframes/poses.txt').write_text('1 0 0 0 0 0 0 0 1\n')
        (self.map / 'GlobalMap.pcd').write_bytes(b'synthetic map geometry, not loaded')
        self.bag = self.root / 'bag.bag'
        self.bag.write_bytes(b'synthetic input, not a real recording')
        self.binding = dict(map=dict(path=str(self.map), frame_id='map', fingerprint=map_fingerprint(self.map)),
                            bag=dict(path=str(self.bag), fingerprint=file_fingerprint(self.bag)),
                            observation=dict(start_offset_s=0.0, frame_id='rslidar', lidar_stamp_ns=1234567890))
        self.path = self.root / 'seed.json'
        self.quaternion = rpy_quaternion(.1, -.2, 1.2)
        save_seed(self.path, self.binding, [1, 2, 3], self.quaternion)

    def test_pose_roundtrip_and_preview_use_the_same_convention(self):
        np.testing.assert_allclose(load_seed(self.path, self.map, self.bag, 0), [.1, -.2, 1.2, 1, 2, 3])
        np.testing.assert_allclose(rotation_matrix(rpy_quaternion(0, 0, np.pi/2)) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
        for pitch in (-np.pi/2, np.pi/2):
            q = rpy_quaternion(.3, pitch, .5)
            np.testing.assert_allclose(rotation_matrix(rpy_quaternion(*quaternion_rpy(q))), rotation_matrix(q), atol=1e-15)
        record = json.loads(self.path.read_text())
        self.assertFalse(record['registration_verified'])
        self.assertEqual(record['observation']['lidar_stamp_ns'], 1234567890)

    def test_annotation_is_never_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_seed(self.path, self.binding, [9, 9, 9], self.quaternion)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.seed.json.*')), [])

    def test_changed_map_bag_offset_or_sensor_frame_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'offset'):
            load_seed(self.path, self.map, self.bag, 1)
        record = json.loads(self.path.read_text())
        record['observation']['frame_id'] = 'camera_link'
        self.path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'rslidar'):
            load_seed(self.path, self.map, self.bag, 0)
        record['observation']['frame_id'] = 'rslidar'
        self.path.write_text(json.dumps(record))
        self.bag.write_bytes(b'changed input')
        with self.assertRaisesRegex(ValueError, 'bag'):
            load_seed(self.path, self.map, self.bag, 0)
        self.bag.write_bytes(b'synthetic input, not a real recording')
        (self.map / 'GlobalMap.pcd').write_bytes(b'changed map')
        with self.assertRaisesRegex(ValueError, 'map'):
            load_seed(self.path, self.map, self.bag, 0)

    def test_localization_reads_seed_and_rejects_incompatible_modes(self):
        command = ['--mode', 'localization', '--bag', str(self.bag), '--map', str(self.map),
                   '--out', str(self.root / 'new_run'), '--pose-file', str(self.path)]
        parsed = arguments(command + ['--rviz'])
        self.assertEqual(parsed.initialization, 'pose')
        np.testing.assert_allclose(parsed.pose, [.1, -.2, 1.2, 1, 2, 3])
        for extra in (['--initialization', 'rviz'], ['--pose', '0', '0', '0', '0', '0', '0'],
                      ['--mode', 'mapping'], ['--start', '2']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(command + extra)
        self.assertFalse((self.root / 'new_run').exists())


if __name__ == '__main__':
    unittest.main()
