"""Rigid map levelling must preserve geometry and fail on weak evidence."""
import unittest

import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.level_map import floor_rotation, transform_row


class LevelMapTests(unittest.TestCase):
    def test_robust_floor_fit_and_rigid_pose_transform(self):
        y, x = np.mgrid[:90, :90]*.1
        ground = (.03*x-.04*y+.8).astype(np.float32)
        ground[20:24, 30:34] += 2.0
        rotation, report = floor_rotation(ground, np.ones_like(ground, bool), [0, 0], .1)
        expected = np.array([-.03, .04, 1.]); expected /= np.linalg.norm(expected)
        self.assertLess(np.linalg.norm(rotation@expected-np.array([0., 0., 1.])), .002)
        self.assertLess(abs(np.linalg.det(rotation)-1), 1e-10)
        self.assertEqual(report['assumption'], 'single_approximately_horizontal_floor')
        row = {'has_map_pose': True, 'map_position': [3., 2., 1.],
               'map_quaternion_xyzw': [0., 0., 0., 1.], 'position': [9., 8., 7.]}
        after = transform_row(row, rotation)
        original = pose_matrix(row['map_position'], row['map_quaternion_xyzw'])
        changed = pose_matrix(after['map_position'], after['map_quaternion_xyzw'])
        point = np.array([.2, -.3, .4, 1.])
        np.testing.assert_allclose((changed@point)[:3], rotation@(original@point)[:3], atol=1e-8)
        self.assertEqual(after['position'], row['position'])
        self.assertEqual(row['map_position'], [3., 2., 1.])

    def test_missing_or_steep_floor_fails(self):
        ground = np.zeros((100, 100), np.float32)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            floor_rotation(ground, np.zeros_like(ground, bool), [0, 0], .1)
        ground[:] = np.arange(100)[None, :]*.1*.3
        with self.assertRaisesRegex(ValueError, 'tilt'):
            floor_rotation(ground, np.ones_like(ground, bool), [0, 0], .1)


if __name__ == '__main__': unittest.main()
