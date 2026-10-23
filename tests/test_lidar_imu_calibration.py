"""Calibration physics and split isolation; no ROS, bag replay or GPU."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from perception.sensors.calibration.lidar_imu_joint import check_time_separation, solve_rotation


class JointCalibrationTests(unittest.TestCase):
    @staticmethod
    def synthetic_sources():
        rotation = Rotation.from_rotvec([.015, -.012, 0.]).as_matrix()

        def omega(t, phase):
            return np.column_stack([.015 * np.sin(2*np.pi*.4*t + phase),
                                    .012 * np.cos(2*np.pi*.6*t + phase),
                                    .3 * np.sin(2*np.pi*.2*t + phase) + .08*np.cos(2*np.pi*.5*t)])

        time = np.arange(0., 25., .01)
        grid = np.arange(-1., 26., .001)
        sources = []
        for j, (group, tau) in enumerate([('first', .03), ('second', .055), ('second', .055)]):
            bias = np.array([.002, -.001, .003]) + j*.0002
            sources.append(dict(id=str(j), clock_group=group, time=time, imu_grid=grid,
                                omega=omega(time, j*.3), filtered=omega(grid-tau, j*.3) @ rotation.T + bias,
                                mask=(time >= .5) & (time < 24.5)))
        return sources, rotation

    def test_recovers_shared_rotation_and_distinct_clock_offsets(self):
        sources, rotation = self.synthetic_sources()
        result = solve_rotation(sources, np.eye(3))
        self.assertTrue(result['converged'])
        self.assertTrue(result['bounded'])
        error = Rotation.from_matrix(result['rotation'] @ rotation.T).magnitude()
        self.assertLess(error, .003)
        self.assertAlmostEqual(result['taus']['first'], .03, delta=.003)
        self.assertAlmostEqual(result['taus']['second'], .055, delta=.003)
        # Runtime header correction is negative tau; returning the opposite lag
        # would fail these independent signal-based timing assertions.
        self.assertLess(-result['taus']['second'], 0.)

    def test_refits_clocks_without_changing_frozen_spatial_calibration(self):
        sources, rotation = self.synthetic_sources()
        result = solve_rotation(sources, rotation, refine_rotation=False)
        self.assertTrue(result['converged'])
        self.assertTrue(result['bounded'])
        self.assertFalse(result['rotation_refined'])
        np.testing.assert_array_equal(result['rotation'], rotation)
        self.assertAlmostEqual(result['taus']['first'], .03, delta=.003)
        self.assertAlmostEqual(result['taus']['second'], .055, delta=.003)

    def test_rejects_overlapping_acquisition_intervals_despite_different_bags(self):
        training = [dict(origin=100., time=np.array([5., 15.]), mask=np.array([True, True]))]
        validation = [dict(origin=105., time=np.array([5., 15.]), mask=np.array([True, True]))]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_time_separation(training, validation)
        validation[0]['origin'] = 200.
        check_time_separation(training, validation)


if __name__ == '__main__':
    unittest.main()
