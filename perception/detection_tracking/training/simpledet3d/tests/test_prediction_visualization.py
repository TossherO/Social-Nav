import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.visualize_wheelchair_predictions import (
    _plot_point_subsets,
    _write_csv,
    filter_prediction_anno_by_point_support,
    frame_audit_stats,
    greedy_center_matches,
    oriented_bev_corners,
    select_audit_frames,
    uniform_subset,
)
from simple_det3d.datasets.weak_label_utils import PedestrianBoxFilterConfig


class PredictionVisualizationTest(unittest.TestCase):
    def test_oriented_bev_corners_rotate_with_yaw(self):
        box = np.array([1, 2, 0, 4, 2, 1, 0], dtype=np.float32)
        corners = oriented_bev_corners(box)
        np.testing.assert_allclose(corners[0], [3, 3], atol=1e-6)
        np.testing.assert_allclose(corners[-1], corners[0], atol=1e-6)

        rotated = box.copy()
        rotated[6] = np.pi / 2
        rotated_corners = oriented_bev_corners(rotated)
        np.testing.assert_allclose(
            rotated_corners.max(axis=0) - rotated_corners.min(axis=0),
            [2, 4], atol=1e-5,
        )

    def test_center_matching_is_one_to_one(self):
        predictions = np.array([[0, 0], [0.15, 0], [2, 0]], dtype=np.float32)
        manual = np.array([[0.1, 0], [2.2, 0]], dtype=np.float32)
        matches = greedy_center_matches(predictions, manual, radius=0.25)
        self.assertEqual(len(matches), 2)
        self.assertEqual({match[2] for match in matches}, {0, 1})
        self.assertEqual(len({match[1] for match in matches}), 2)

    def test_frame_stats_apply_score_and_range_filters(self):
        info = {
            'token': 'indoor1/sequence/0001',
            'scene_id': 'indoor1/sequence',
            'frame_index': 1,
            'weak_centers': np.array([[1.1, 1.0], [20.0, 0.0]], dtype=np.float32),
            'gt_supervision_types': np.array([2, 1, 0], dtype=np.int32),
        }
        anno = {
            'frame_id': info['token'],
            'name': np.array(['Pedestrian', 'Pedestrian', 'Pedestrian']),
            'score': np.array([0.9, 0.2, 0.8], dtype=np.float32),
            'boxes_lidar': np.array([
                [1.0, 1.0, 0.0, 0.6, 0.6, 1.7, 0.0],
                [3.0, 3.0, 0.0, 0.6, 0.6, 1.7, 0.0],
                [4.0, 4.0, 0.0, -0.6, 0.6, 1.7, 0.0],
            ], dtype=np.float32),
        }
        record = frame_audit_stats(
            4, info, anno, [-5, -5, -2, 5, 5, 2],
            score_thresh=0.3, match_radius=0.5,
        )
        self.assertEqual(record['raw_predictions'], 3)
        self.assertEqual(record['shown_predictions'], 1)
        self.assertEqual(record['manual_centers_total'], 2)
        self.assertEqual(record['manual_centers_in_range'], 1)
        self.assertEqual(record['manual_matched'], 1)
        self.assertEqual(record['manual_missed'], 0)
        self.assertAlmostEqual(record['center_hit'], 1.0)
        self.assertEqual(record['weak_box_targets'], 1)
        self.assertEqual(record['weak_center_targets'], 1)

    def test_selection_and_uniform_subset(self):
        records = [
            {'dataset_index': 0, 'manual_missed': 1,
             'max_nearest_manual_distance': 0.8, 'shown_predictions': 2},
            {'dataset_index': 1, 'manual_missed': 2,
             'max_nearest_manual_distance': math.inf, 'shown_predictions': 0},
            {'dataset_index': 2, 'manual_missed': 0,
             'max_nearest_manual_distance': 0.1, 'shown_predictions': 5},
        ]
        selected = select_audit_frames(records, 'missed', 1)
        self.assertEqual(selected[0]['dataset_index'], 1)
        self.assertEqual(uniform_subset(list(range(10)), 3), [0, 4, 9])

    def test_csv_accepts_fields_added_only_to_selected_rows(self):
        records = [{'frame_id': 'a'}, {'frame_id': 'b', 'image_name': 'b.png'}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'frames.csv'
            _write_csv(output, records)
            with output.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['image_name'], '')
        self.assertEqual(rows[1]['image_name'], 'b.png')

    def test_plot_point_budget_prioritizes_current_frame(self):
        current = np.arange(30, dtype=np.float32).reshape(10, 3)
        history = np.arange(60, dtype=np.float32).reshape(20, 3)
        current_plot, history_plot = _plot_point_subsets(current, history, 6)
        self.assertEqual(len(current_plot), 6)
        self.assertEqual(len(history_plot), 0)

        current_plot, history_plot = _plot_point_subsets(current[:2], history, 6)
        self.assertEqual(len(current_plot), 2)
        self.assertEqual(len(history_plot), 4)

    def test_distance_adaptive_point_support_filter(self):
        anno = {
            'frame_id': np.asarray('frame'),
            'name': np.asarray(['Pedestrian', 'Pedestrian']),
            'score': np.asarray([0.9, 0.8], dtype=np.float32),
            'pred_labels': np.asarray([1, 1], dtype=np.int64),
            'boxes_lidar': np.asarray([
                [2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0],
                [10.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0],
            ], dtype=np.float32),
        }
        points = np.asarray([
            [1.9, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.1, 0.0, 0.0],
            [9.9, 0.0, 0.0],
        ], dtype=np.float32)
        cfg = PedestrianBoxFilterConfig(
            near_distance=8.0,
            far_distance=15.0,
            near_min_points=5,
            mid_min_points=3,
            far_min_points=1,
        )

        filtered, audit = filter_prediction_anno_by_point_support(
            points, anno, cfg,
        )

        self.assertEqual(audit['input_predictions'], 2)
        self.assertEqual(audit['rejected_predictions'], 1)
        self.assertEqual(len(filtered['boxes_lidar']), 1)
        self.assertAlmostEqual(float(filtered['boxes_lidar'][0, 0]), 10.0)
        np.testing.assert_array_equal(filtered['point_support'], [3])
        np.testing.assert_array_equal(filtered['required_point_support'], [3])


if __name__ == '__main__':
    unittest.main()
