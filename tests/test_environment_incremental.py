"""Check evidence, height seams, and old grid identity during static updates."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from perception.environment.fusion import fuse_scans_tiled
from perception.environment.incremental import _extend_parent, merge_survey_evidence, plan_survey_ground
from perception.environment.layers import GridSnapshot, load_config
from perception.environment.maps import endpoint_layers
from perception.environment.poses import PoseTimeline
from perception.environment.scans import ScanJournal


def _layers(cells, height):
    cells = np.atleast_2d(np.asarray(cells, np.int8))
    ground = np.atleast_2d(np.asarray(height, np.float32))
    return dict(occupancy=cells, ground_z=ground, ground_support=np.isfinite(ground),
        observed=(cells != -1), last_observed=np.where(cells != -1, 10., np.nan),
        coverage=(cells == 0).astype(np.float32), obstacle_min_z=np.where(cells == 100, .2, np.nan),
        obstacle_max_z=np.where(cells == 100, .8, np.nan), endpoint_count=(cells == 100).astype(np.int32))


class IncrementalEnvironmentTests(unittest.TestCase):
    def test_merge_retains_times_and_manual_edits_and_only_accepts_supported_changes(self):
        config = load_config()
        parent = _layers(
            [0, 0, 100, 100, 0, -1, -1, 0, 100, -1, 100, 0],
            [0, 0, 0, 0, .3, 0, np.nan, np.nan, 0, np.nan, .02, 0])
        parent['last_observed'][0, 7] = np.nan
        parent['last_observed'][0, 8] = 30.
        parent['manual_edit'] = np.zeros((1,12), bool)
        parent['manual_edit'][0, 7] = True
        parent['assumed_free'] = parent['manual_edit'].copy()
        parent['ground_support'][0, 10] = False
        parent['ground_z'][0, 7] = .05
        parent['ground_estimated'] = parent['manual_edit'].copy()
        survey = _layers(
            [-1, 100, 0, 0, 0, 0, 0, 100, 0, -1, 100, 0],
            [np.nan, 0, .02, 0, 0, 0, 0, 0, 0, .4, .07, .02])
        survey['last_observed'][:] = 20.
        survey['ground_scan_column_observed'] = survey['ground_support'].copy()
        projection, masks = plan_survey_ground(parent, survey, config)
        protected = np.zeros((1,12), bool)
        protected[0, 3] = True
        masks, counts = merge_survey_evidence(parent, survey, projection, masks, config,
                                               survey_start=19., survey_index=1, protected_occupied=protected)
        np.testing.assert_array_equal(parent['occupancy'], [[0, 100, 0, 100, -1, 0, 0, 0, 100, -1, 100, 0]])
        np.testing.assert_allclose(parent['last_observed'], [[10,20,20,10,np.nan,20,20,np.nan,30,np.nan,10,10]])
        np.testing.assert_allclose(parent['ground_z'], [[0,0,0,0,.3,0,0,.05,0,np.nan,.07,0]])
        self.assertFalse(parent['observed'][0, 4])
        self.assertEqual(parent['coverage'][0, 4], 0.)
        self.assertTrue(masks['ground_seam_conflict'][0, 9])
        self.assertTrue(masks['manual_review'][0, 7])
        self.assertTrue(parent['ground_estimated'][0, 7])
        self.assertTrue(parent['assumed_free'][0, 7])
        self.assertAlmostEqual(parent['obstacle_min_z'][0, 10], .15)
        self.assertEqual(counts['cleared_occupied'], 1)

    def test_refines_weak_ground_with_direct_scan_support_and_unchanged_collision_bands(self):
        config = load_config()
        parent = _layers([0, 100, 0, 0, -1, -1, 100, -1, -1],
                         [0., 0., 0., 0., 0., 0., 0., .17, .15])
        parent['ground_weak_support'] = np.array([[1,1,0,1,1,1,1,0,1]], bool)
        parent['manual_edit'] = np.zeros((1,9), bool)
        parent['manual_edit'][0, 3] = True
        parent['ground_scan_undersampled'] = parent['ground_weak_support'].copy()
        survey = _layers([-1, -1, -1, -1, -1, 100, -1, -1, -1],
                         [.02, .02, .02, .02, .05, .05, -.06, .17, .11])
        survey['ground_scan_column_observed'] = np.ones((1,9), bool)
        survey['ground_scan_column_observed'][0, 8] = False
        survey['last_observed'][:] = 20.
        ground, masks = plan_survey_ground(parent, survey, config)
        masks, counts = merge_survey_evidence(parent, survey, ground, masks, config, survey_start=19., survey_index=1)

        # Same physical collision columns preserve original classifications
        # and timestamps. Unknown without any collision evidence can improve
        # even when its potential column changes.
        np.testing.assert_allclose(parent['ground_z'], [[.02,.02,0.,0.,.05,0.,0.,.17,.15]])
        np.testing.assert_allclose(parent['last_observed'], [[10,10,10,10,np.nan,20,10,np.nan,np.nan]])
        np.testing.assert_array_equal(parent['occupancy'], [[0,100,0,0,-1,100,100,-1,-1]])
        self.assertAlmostEqual(parent['obstacle_min_z'][0,1]+parent['ground_z'][0,1], .2)
        self.assertTrue(masks['ground_refinement_blocked_collision_band'][0, 5])
        self.assertTrue(masks['ground_refinement_blocked_collision_band'][0, 6])
        self.assertEqual(counts['ground_refined'], 3)

    def test_refinement_never_creates_a_new_height_step(self):
        config = load_config()
        parent = _layers([-1, -1], [0., .14])
        parent['ground_weak_support'] = np.array([[True, False]])
        survey = _layers([-1, -1], [-.04, .14])
        survey['ground_scan_column_observed'] = np.ones((1,2), bool)
        ground, masks = plan_survey_ground(parent, survey, config)
        masks, _ = merge_survey_evidence(parent, survey, ground, masks, config, survey_start=19., survey_index=1)
        self.assertTrue(masks['ground_refinement_blocked_seam'][0,0])
        self.assertEqual(parent['ground_z'][0,0], 0.)

    def test_old_cell_lattice_and_values_survive_extension(self):
        cfg = load_config()
        parent = GridSnapshot(dict(origin_xy=[-.100003, .299997], shape=[2, 2], resolution=.1),
                              _layers([[0, -1], [-1, 100]], [[0, np.nan], [np.nan, .02]]))
        points = np.array([[-.205, .45, 0.], [.02, .34, .02]], np.float32)
        origin, extended, region = _extend_parent(parent, points, cfg)
        self.assertEqual(extended['occupancy'].shape, (2, 4))
        np.testing.assert_allclose(origin, [-.300003, .299997], atol=1e-12)
        for name, old in parent.layers.items():
            np.testing.assert_array_equal(extended[name][region], old)
        self.assertTrue(np.isnan(extended['last_observed'][0,0]))

    def test_repeated_ray_prefix_classifies_free_but_single_hits_do_not(self):
        config = load_config()
        config['min_obstacle_height_m'], config['max_height_m'] = .10, .19
        config['static']['tile_size_m'] = .5
        floor = np.zeros((3, 20), np.float32)
        origin = np.array([0., 0.])
        empty = np.empty((0,3), np.float32)
        candidate = GridSnapshot(dict(shape=[3,20], origin_xy=origin.tolist(), resolution=.1,
            snapshot_id='test', map_id='same_map'), dict(
            ground_z=floor, ground_support=np.ones(floor.shape,bool),
            **endpoint_layers(empty, floor, origin, config)))
        sensor = [.15, .15, .15]
        transform = np.eye(4)
        transform[:3,3] = sensor
        stamps = (1., 1.6, 2.2, 2.8)
        rows = [dict(stamp_s=t, stamp_ns=round(t*1e9), position=sensor, quaternion_xyzw=[0, 0, 0, 1],
                     session_id='test', generation=0, frame_id='odom', health=0) for t in stamps]
        timeline = PoseTimeline(rows, config['poses'], [[1., *sensor, 0., 0., 0., 0, 0]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scans.sqlite'
            writer = ScanJournal(path, config['scan_recording'], {})
            meta = dict(session_id='test', generation=0, frame_id='odom', rays_used=True,
                        ray_origin_model='pose_interpolated_origins')
            for index, stamp in enumerate(stamps):
                points = [[1.75,.15,.15]]
                if index == 0:
                    points.append([1.25,.25,.15])
                writer.append(np.asarray(points)-sensor, transform, round(stamp*1e9), meta,
                              np.broadcast_to(sensor, (len(points),3)))
            writer.close()
            survey = fuse_scans_tiled(candidate, path, timeline, config, {}, candidate_points=empty,
                                      require_repeated_evidence=True)
        self.assertEqual(survey.layers['occupancy'][1,5], 0)
        self.assertEqual(survey.layers['occupancy'][1,17], 100)
        self.assertEqual(survey.layers['occupancy'][2,12], -1)
        self.assertEqual(survey.layers['last_observed'][1,5], stamps[-1])


if __name__ == '__main__':
    unittest.main()
