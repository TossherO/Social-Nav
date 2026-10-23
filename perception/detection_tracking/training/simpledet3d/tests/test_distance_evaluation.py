"""Distance weighting, annotation boundaries and score-override integration."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from simple_det3d.datasets.coda_dataset import CodaDataset
from simple_det3d.datasets.distance_evaluation import DistanceCenterMetrics
from tools import task_scene


PROFILE = {'EDGES_M': [0, 3, 6, 10], 'WEIGHTS': [1., .5, .25, .1]}


def accumulate(gt, pred, *, complete=True):
    metrics = DistanceCenterMetrics(PROFILE, .5)
    matches = CodaDataset._greedy_center_matches(pred, gt, .5)
    metrics.add_frame(gt, pred, matches, complete=complete, empty=not len(gt))
    return metrics


class DistanceMetricTests(unittest.TestCase):
    def test_missing_near_target_costs_more_than_missing_far_target(self):
        gt = [[1., 0.], [8., 0.]]
        near_hit = accumulate(gt, gt[:1]).result()[0]
        far_hit = accumulate(gt, gt[1:]).result()[0]
        self.assertAlmostEqual(near_hit['DistanceWeightedF1@0.50m'], 2 / 2.25)
        self.assertAlmostEqual(far_hit['DistanceWeightedF1@0.50m'], .5 / 1.5)

    def test_false_positive_distance_and_empty_frames(self):
        near = accumulate([[1., 0.]], [[1., 0.], [2., 0.]]).result()[0]
        far = accumulate([[1., 0.]], [[1., 0.], [12., 0.]]).result()[0]
        self.assertLess(near['DistanceWeightedF1@0.50m'], far['DistanceWeightedF1@0.50m'])
        empty = accumulate([], [[2., 0.], [8., 0.]]).result()[0]
        self.assertAlmostEqual(empty['DistanceWeightedFalsePositivesPerFrame'], 1.25)
        self.assertTrue(np.isnan(empty['DistanceWeightedRecall@0.50m']))
        self.assertTrue(np.isnan(empty['DistanceWeightedF1@0.50m']))
        self.assertEqual(empty['Distance/6-10m/EmptyFalsePositiveCount'], 1)

    def test_global_match_crosses_bin_boundary_without_double_penalty(self):
        result = accumulate([[2.99, 0.]], [[3.01, 0.]]).result()[0]
        self.assertEqual(result['Distance/0-3m/CenterTruePositiveCount@0.50m'], 1)
        self.assertEqual(result['Distance/3-6m/CenterFalsePositiveCount@0.50m'], 0)
        self.assertEqual(result['DistanceWeightedF1@0.50m'], 1)

    def test_partial_annotations_only_supply_known_positive_hits(self):
        metric = accumulate([[1., 0.]], [[7., 0.], [12., 0.]], complete=False)
        metric.add_frame([[8., 0.]], [[8., 0.]], [(0., 0, 0)], complete=True, empty=False)
        result = metric.result()[0]
        self.assertEqual(result['Distance/0-3m/LabeledCenterCount'], 1)
        self.assertEqual(result['Distance/0-3m/CenterHit@0.50m'], 0)
        self.assertTrue(np.isnan(result['Distance/0-3m/CenterRecall@0.50m']))
        self.assertEqual(result['DistanceWeightedF1@0.50m'], 1)
        self.assertEqual(result['DistanceWeightedFalsePositiveCount@0.50m'], 0)
        partial_only = accumulate([[1., 0.]], [], complete=False).result()[0]
        self.assertNotIn('DistanceWeightedF1@0.50m', partial_only)

    def test_exact_boundaries_and_no_prediction(self):
        result = accumulate([[-3., 0.], [0., 6.], [10., 0.]], []).result()[0]
        for name in ('3-6m', '6-10m', '10-infm'):
            self.assertEqual(result['Distance/' + name + '/CenterMissedCount@0.50m'], 1)
        self.assertAlmostEqual(result['DistanceWeightedMissedCount@0.50m'], .85)
        self.assertEqual(result['DistanceWeightedF1@0.50m'], 0)
        self.assertTrue(np.isnan(result['DistanceWeightedPrecision@0.50m']))

    def test_invalid_profiles_are_rejected(self):
        for key,value in [('EDGES_M', [1, 3, 6, 10]), ('EDGES_M', [0, 3, 3, 10]),
                          ('EDGES_M', [0, 3, 6, float('inf')]), ('WEIGHTS', [1, .5]),
                          ('WEIGHTS', [1, .5, .6, .1]), ('WEIGHTS', [1, .5, .25, 0])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                DistanceCenterMetrics(dict(PROFILE, **{key:value}), .5)


class DistanceEvaluationIntegrationTests(unittest.TestCase):
    def test_counts_reconcile_and_unknown_background_is_not_scored(self):
        dataset = CodaDataset.__new__(CodaDataset)
        dataset.point_cloud_range = np.array([-5., -10., -2., 15., 10., 5.])
        def info(token, centers, complete):
            boxes = np.zeros((len(centers), 7), dtype=np.float32)
            boxes[:, :2] = np.asarray(centers).reshape(-1, 2)
            return dict(token=token, lidar_path=token+'.bin', gt_boxes=boxes,
                        weak_centers=centers, center_label_is_exhaustive=complete)
        dataset.infos = [info('full', [[1.,0.], [8.,0.]], True), info('empty', [], True),
                         info('draft', [[2.,0.]], False)]
        def prediction(token, xy):
            boxes = np.zeros((len(xy), 7), dtype=np.float32)
            boxes[:, :2] = np.asarray(xy).reshape(-1, 2)
            return dict(frame_id=token, boxes_lidar=boxes, score=np.full(len(xy), .8),
                        name=np.full(len(xy), 'Pedestrian'))
        predictions = [prediction('full', [[1.,0.], [12.,0.]]), prediction('empty', [[2.,0.]]),
                       prediction('draft', [[2.,0.], [4.,0.], [9.,0.]])]
        config = dict(CENTER_SCORE_THRESH=.3, CENTER_HIT_THRESHOLDS=[.5, 1.],
                      DISTANCE_EVAL=dict(PROFILE, ENABLED=True))
        _, result = dataset._evaluation_center_only(predictions, config)
        prefixes = ['Distance/'+name+'/' for name in ('0-3m','3-6m','6-10m','10-infm')]
        self.assertEqual(sum(result[p+'CenterMissedCount@0.50m'] for p in prefixes),
                         result['CenterMissedCount@0.50m'])
        self.assertEqual(sum(result[p+'CenterFalsePositiveCount@0.50m'] for p in prefixes),
                         result['CenterFalsePositiveCount@0.50m'])
        self.assertAlmostEqual(result['DistanceWeightedF1@0.50m'], 2/(2+.25+1.1))
        self.assertEqual(result['Distance/0-3m/CenterHit@0.50m'], 1)
        # Missing prediction records are still misses, as in the original metric.
        _, missing = dataset._evaluation_center_only(predictions[1:], config)
        self.assertAlmostEqual(missing['DistanceWeightedMissedCount@0.50m'], 1.25)
        _, disabled = dataset._evaluation_center_only(predictions, dict(config, DISTANCE_EVAL={}))
        self.assertNotIn('DistanceWeightedF1@0.50m', disabled)

    def test_cli_overrides_both_detection_and_evaluation_thresholds(self):
        config = Path(__file__).resolve().parents[1]/'configs/voxelrcnn_ros_centers_finetune.yaml'
        argv = ['task_scene.py','compare','--cfg_file',str(config),'--baseline','base.pth',
                '--candidate','candidate.pth','--out','unused.json','--score-threshold','.2']
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(task_scene, 'compare') as compare:
            task_scene.main()
        cfg = compare.call_args.args[1]
        self.assertEqual(cfg.MODEL.POST_PROCESSING.SCORE_THRESH, .2)
        self.assertEqual(cfg.DATA_CONFIG.WEAK_SUPERVISION.CENTER_SCORE_THRESH, .2)
        invalid = copy.copy(argv)
        invalid[-1] = 'nan'
        with mock.patch.object(sys, 'argv', invalid), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            task_scene.main()


if __name__ == '__main__':
    unittest.main()
