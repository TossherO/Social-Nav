"""Weighted target adaptation: supervision, split boundaries and comparisons."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from simple_det3d.datasets.task_scene_dataset import (
    TaskSceneDataset, build_task_scene_dataset, check_train_validation,
    data_snapshot, make_task_sampler, save_data_snapshot, check_resume_inputs,
)
from simple_det3d.train.sampling import CyclicCoverageSampler, WeightedDistributedSampler
from simple_det3d.utils.config import load_config
from tools.task_scene import finite_json, metric_comparison


class SamplingTests(unittest.TestCase):
    def test_relative_probability_and_rank_partition(self):
        sampler = WeightedDistributedSampler([1., .2], 100000, seed=41)
        draws = list(sampler)
        counts = np.bincount(draws)
        self.assertAlmostEqual(counts[0] / counts[1], 5., delta=.15)
        first = WeightedDistributedSampler([1., .2], 100000, seed=41, num_replicas=2, rank=0)
        second = WeightedDistributedSampler([1., .2], 100000, seed=41, num_replicas=2, rank=1)
        self.assertEqual(list(first), draws[::2])
        self.assertEqual(list(second), draws[1::2])
        sampler.set_epoch(3)
        third = list(sampler)
        self.assertNotEqual(third, draws)
        resumed = WeightedDistributedSampler([1., .2], 100000, seed=41)
        resumed.set_epoch(3)
        self.assertEqual(list(resumed), third)

    def test_budget_and_invalid_weights(self):
        sampler = WeightedDistributedSampler([1., 1., .2], num_replicas=2)
        self.assertEqual((sampler.requested_size, sampler.total_size, len(sampler)), (3, 4, 2))
        for weights in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                WeightedDistributedSampler(weights)


class CyclicSamplingTests(unittest.TestCase):
    def test_full_coverage_and_old_pool_coverage_across_epochs(self):
        sampler = CyclicCoverageSampler([1.] * 5 + [.2] * 100, seed=41)
        seen = set()
        for epoch in range(5):
            sampler.set_epoch(epoch)
            draws = list(sampler)
            self.assertEqual(len(draws), 25)
            self.assertEqual(len(set(draws)), 25)
            self.assertEqual({i for i in draws if i < 5}, set(range(5)))
            old = {i for i in draws if i >= 5}
            self.assertEqual(len(old), 20)
            self.assertFalse(seen & old)
            seen.update(old)
        self.assertEqual(seen, set(range(5, 105)))

    def test_cycle_boundary_keeps_epoch_unique_and_resume_deterministic(self):
        weights = [1.] * 3 + [.2] * 11
        sampler = CyclicCoverageSampler(weights, seed=17)
        old_seen, history = set(), []
        for epoch in range(9):
            sampler.set_epoch(epoch)
            draws = list(sampler)
            self.assertEqual(len(draws), len(set(draws)))
            self.assertEqual(len(draws), 6)
            self.assertEqual({i for i in draws if i < 3}, {0, 1, 2})
            if epoch < 4:
                old_seen.update(i for i in draws if i >= 3)
            history.append(draws)
        self.assertEqual(old_seen, set(range(3, 14)))
        resumed = CyclicCoverageSampler(weights, seed=17)
        resumed.set_epoch(8)
        self.assertEqual(list(resumed), history[8])
        self.assertEqual(list(resumed), list(resumed))
        with self.assertRaises(ValueError):
            resumed.set_epoch(-1)

    def test_ddp_uses_disjoint_shards_and_preserves_new_coverage(self):
        weights = [1.] * 4 + [.2] * 23
        samplers = [CyclicCoverageSampler(weights, num_replicas=2, rank=i) for i in range(2)]
        for sampler in samplers:
            sampler.set_epoch(4)
        shards = [list(s) for s in samplers]
        self.assertEqual([len(s) for s in shards], [5, 5])
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual({i for shard in shards for i in shard if i < 4}, set(range(4)))
        whole = CyclicCoverageSampler(weights, epoch_size=10)
        whole.set_epoch(4)
        draws = list(whole)
        self.assertEqual(shards[0], draws[::2])
        self.assertEqual(shards[1], draws[1::2])

    def test_budget_only_changes_fractional_replay_and_rejects_impossible_requests(self):
        weights = [1.] * 3 + [.2] * 11
        sampler = CyclicCoverageSampler(weights, epoch_size=8, num_replicas=2)
        counts = {g['weight']: g['per_epoch'] for g in sampler.coverage}
        self.assertEqual(counts, {1.: 3, .2: 5})
        for budget in (2, 15):
            with self.assertRaises(ValueError):
                CyclicCoverageSampler(weights, epoch_size=budget)
        with self.assertRaisesRegex(ValueError, 'DDP-aligned'):
            CyclicCoverageSampler([1., 1., .2], num_replicas=2)
        with self.assertRaisesRegex(ValueError, 'weights in'):
            CyclicCoverageSampler([1., 2.])

    def test_multiple_weight_groups_and_capacity_capping(self):
        weights = [1.] * 2 + [.5] * 4 + [.2] * 10
        default = CyclicCoverageSampler(weights)
        self.assertEqual({g['weight']: g['per_epoch'] for g in default.coverage}, {1.:2, .5:2, .2:2})
        enlarged = CyclicCoverageSampler(weights, epoch_size=14)
        self.assertEqual({g['weight']: g['per_epoch'] for g in enlarged.coverage}, {1.:2, .5:4, .2:8})
        self.assertEqual(len(set(enlarged)), 14)


class TaskSceneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        new, old = self.root / 'new', self.root / 'old'
        new.mkdir(); old.mkdir()
        cfg_path = Path(__file__).resolve().parents[1] / 'configs/voxelrcnn_ros_centers_finetune.yaml'
        self.cfg = load_config(cfg_path)
        self.cfg.DATA_CONFIG.DATA_PATH = str(new)
        self.cfg.DATA_CONFIG.RAW_DATA_PATH = str(new)
        self.cfg.DATA_CONFIG.ROS_GROUP_ASSIGNMENTS = {'capture': dict(split='train', location_group='route')}
        clouds, samples = [], []
        for index, (status, count) in enumerate([('draft', 1), ('complete', 0), ('draft', 0), ('complete', 1)]):
            ident = str(index)
            points = np.array([[2., 1., .5, 255.], [3., 2., .4, 128.]], dtype=np.float32)
            points.tofile(new / (ident + '.bin'))
            clouds.append(dict(id=ident, path=ident+'.bin', source_id='a', stamp_ns=1000000000+index*100000000,
                               sensor_frame='rslidar'))
            samples.append(dict(id=ident, pointcloud_id=ident, source_id='a', ordinal=index, status=status,
                                image=None, history=[], annotation=dict(centers=[dict(xy=[2., 1.])]*count,
                                  center_label_is_exhaustive=status=='complete', ignored_candidate_count=0)))
        self.document = dict(format='rosbag_people_centers', unverified_policy='ignore',
                             point_cloud_range=list(self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE), history={},
                             sources=[dict(id='a', root_group='capture', location_group='', split='unassigned', bag_name='a.bag')],
                             pointclouds=clouds, samples=samples)
        self.path = new / 'annotations.json'
        self.path.write_text(json.dumps(self.document))
        def legacy(ident):
            points.tofile(old / (ident+'.bin'))
            return dict(token=ident, scene_id=ident, lidar_path=ident+'.bin', sweeps=[],
                        is_target_domain=True, center_label_is_exhaustive=True,
                        roi_supervision_enabled=False, gt_boxes=np.array([[2., 1., .5, .6, .6, 1.7, 0.]], dtype=np.float32),
                        gt_names=np.array(['Pedestrian']), gt_box_reg_mask=np.array([[1,1,0,0,0,0,0,0]], dtype=np.float32))
        for name, infos in (('train', [legacy('old1'), legacy('old2')]), ('val', [legacy('old_val')])):
            (old/(name+'.pkl')).write_bytes(pickle.dumps(infos))
        for spec, mode in ((self.cfg.TASK_SCENE.TRAIN_DATASETS[1], 'train'),
                           (self.cfg.TASK_SCENE.VALIDATION_DATASETS[0], 'test')):
            spec.CONFIG.DATA_PATH = spec.CONFIG.RAW_DATA_PATH = str(old)
            spec.CONFIG.INFO_PATH[mode] = ['train.pkl' if mode == 'train' else 'val.pkl']

    def test_actual_mixed_batch_and_per_frame_weights(self):
        train = build_task_scene_dataset(self.cfg, training=True)
        validation = build_task_scene_dataset(self.cfg, training=False)
        check_train_validation(train, validation)
        self.assertEqual([len(d) for d in train.datasets], [3, 2])
        np.testing.assert_allclose(train.sample_weights, [1, 1, 1, .2, .2])
        with self.assertRaisesRegex(ValueError, 'Duplicate frame'):
            TaskSceneDataset([train.datasets[0], train.datasets[0]], ['first', 'second'])
        batch = train.collate_batch([train[i] for i in (0, 1, 3)])
        np.testing.assert_allclose(batch['heatmap_background_weight'], [0., .5, .5])
        np.testing.assert_allclose(batch['roi_center_supervision_enabled'], [0, 1, 1])
        self.assertEqual([len(train[i]['gt_boxes']) for i in (0, 1, 3)], [1, 0, 1])
        self.assertTrue(np.isfinite(batch['points']).all())
        self.assertTrue(all(not train[i]['gt_box_reg_mask'][:, 2:].any() for i in range(len(train))))
        self.assertEqual(batch['frame_id'].tolist(), ['task::0', 'task::1', 'legacy::old1'])
        self.assertEqual(json.loads(self.path.read_text())['sources'][0]['split'], 'unassigned')

    def test_complete_only_and_overlap_rejected(self):
        self.cfg.DATA_CONFIG.ROS_COMPLETE_ONLY = True
        train = build_task_scene_dataset(self.cfg, training=True)
        self.assertEqual(len(train.datasets[0]), 2)
        with self.assertRaisesRegex(ValueError, 'share current'):
            check_train_validation(train, train)
        validation = build_task_scene_dataset(self.cfg, training=False)
        validation.infos[0]['root_group'] = 'capture'
        with self.assertRaisesRegex(ValueError, 'root_group'):
            check_train_validation(train, validation)

    def test_snapshot_prevents_silent_data_changes(self):
        train = build_task_scene_dataset(self.cfg, training=True)
        validation = build_task_scene_dataset(self.cfg, training=False)
        snapshot = data_snapshot(train, validation, make_task_sampler(train, self.cfg), self.cfg)
        target = self.root / 'training_data.json'
        save_data_snapshot(target, snapshot)
        save_data_snapshot(target, snapshot)
        checkpoint = self.root / 'run/ckpt/checkpoint.pth'
        checkpoint.parent.mkdir(parents=True)
        save_data_snapshot(checkpoint.parent.parent / 'training_data.json', snapshot)
        check_resume_inputs(checkpoint, snapshot)
        self.assertEqual(snapshot['sampling']['strategy'], 'cyclic_coverage')
        self.assertFalse(snapshot['sampling']['replacement'])
        self.assertEqual(snapshot['training'][0]['exposure_per_frame'], 1)
        self.cfg.TASK_SCENE.SAMPLER = 'weighted_replacement'
        legacy_train = build_task_scene_dataset(self.cfg, training=True)
        legacy_snapshot = data_snapshot(legacy_train, validation,
                                        make_task_sampler(legacy_train, self.cfg), self.cfg)
        self.assertTrue(legacy_snapshot['sampling']['replacement'])
        with self.assertRaisesRegex(ValueError, 'Full resume'):
            check_resume_inputs(checkpoint, legacy_snapshot)
        self.cfg.TASK_SCENE.SAMPLER = 'cyclic_coverage'
        changed = copy.deepcopy(snapshot)
        changed['training'][0]['per_frame_weight'] = .2
        with self.assertRaisesRegex(ValueError, 'Full resume'):
            check_resume_inputs(checkpoint, changed)
        with self.assertRaisesRegex(ValueError, 'changed'):
            save_data_snapshot(target, changed)
        self.cfg.TASK_SCENE.INPUT_BINDINGS = [train.input_files[0]]
        self.path.write_text(self.path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'export changed'):
            build_task_scene_dataset(self.cfg, training=True)

    def test_empty_frame_metrics_and_prediction_identity(self):
        dataset = build_task_scene_dataset(self.cfg, training=False, diagnostic_training=True, names=['task'])
        predictions = []
        for info in dataset.infos:
            xy = [2., 1.] if len(info['gt_boxes']) else [4., 2.]
            predictions.append(dict(frame_id=info['token'], name=np.array(['Pedestrian', 'Pedestrian']),
                                    score=np.array([.8, .9]), boxes_lidar=np.array([xy+[0.]*5, [30.,0.]+[0.]*5])))
        _, metrics = dataset.evaluation(predictions, dataset.class_names)
        self.assertEqual(metrics['EmptyFrameCount'], 1)
        self.assertEqual(metrics['EmptyFalsePositivesPerFrame'], 1)
        self.assertEqual(metrics['EmptyFalsePositiveFrameRate'], 1)
        self.assertEqual(metrics['CenterRecall@0.50m'], 1)
        self.assertEqual(metrics['CenterFalsePositivesPerFrame'], .5)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            dataset.evaluation(predictions[:-1], dataset.class_names)
        component = dataset.datasets[0]
        component.infos = [i for i in component.infos if i.get('center_label_is_exhaustive') and not len(i['gt_boxes'])]
        empty = TaskSceneDataset([component], ['task'])
        _, metrics = empty.evaluation([predictions[1]], empty.class_names)
        self.assertTrue(np.isnan(metrics['CenterRecall@0.50m']))
        self.assertIsNone(finite_json(metrics)['CenterRecall@0.50m'])

    def test_comparison_does_not_accept_training_or_legacy_results(self):
        baseline = {'CenterRecall@0.50m': .9, 'CenterFalsePositivesPerFrame': 2.}
        candidate = {'CenterRecall@0.50m': .89, 'CenterFalsePositivesPerFrame': 1.}
        for role in ('training_resubstitution', 'legacy_weak_regression'):
            self.assertFalse(metric_comparison(baseline, candidate, role=role, max_recall_drop=.02)['acceptance_assessed'])
        self.assertTrue(metric_comparison(baseline, candidate, role='task_development', max_recall_drop=.02)['meets_requested_tradeoff'])
        self.assertFalse(metric_comparison(baseline, candidate, role='task_development', max_recall_drop=.005)['meets_requested_tradeoff'])
        self.assertFalse(metric_comparison(baseline, candidate, role='task_development', max_recall_drop=.02,
                                          complete_human_labels=False)['acceptance_assessed'])


if __name__ == '__main__':
    unittest.main()
