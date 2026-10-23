import tempfile
import pickle
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np
from easydict import EasyDict
from simple_det3d.utils.common_utils import load_checkpoint

from simple_det3d.train.train_utils import (
    checkpoint_state,
    load_best_metric_state,
    select_training_checkpoints,
    train_model,
)


METRIC_NAME = 'CenterHit@0.50m'


class TrainingResumeTest(unittest.TestCase):
    def _task_config(self, project_root, sweeps):
        return EasyDict(
            ROOT_DIR=project_root / 'perception/detection_tracking/training/simpledet3d',
            CLASS_NAMES=['Pedestrian'], MODEL={'NAME': 'VoxelRCNN'},
            DATA_CONFIG={'MAX_SWEEPS': sweeps}, TASK_SCENE={'ENABLED': True},
        )

    def test_task_initializer_matches_configured_frames_without_cuda(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sweeps in (1, 2, 3, 4):
                path = root / f'models/wheelchair_detector/{sweeps}frame/checkpoints/detector_best.pth'
                path.parent.mkdir(parents=True)
                path.touch()
            for sweeps in (0, 1, 2, 3, 4):
                cfg = self._task_config(root, sweeps)
                resumes, pretrained = select_training_checkpoints(cfg, root / 'new/ckpt')
                self.assertEqual(resumes, [])
                self.assertEqual(Path(pretrained), root / f'models/wheelchair_detector/{max(1, sweeps)}frame/checkpoints/detector_best.pth')
            with self.assertRaisesRegex(FileNotFoundError, '5frame'):
                select_training_checkpoints(self._task_config(root, 5), root / 'new/ckpt')

    def test_explicit_initializer_overrides_automatic_choice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            custom = root / 'custom.pth'
            custom.touch()
            cfg = self._task_config(root, 5)
            self.assertEqual(select_training_checkpoints(cfg, root / 'ckpt', pretrained_model=custom),
                             ([], str(custom)))
            with self.assertRaisesRegex(FileNotFoundError, 'missing.pth'):
                select_training_checkpoints(cfg, root / 'ckpt', pretrained_model=root / 'missing.pth')

    def test_resume_precedes_initialization_even_when_initializer_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ckpt_dir = root / 'ckpt'
            ckpt_dir.mkdir()
            older, newer, explicit = [ckpt_dir / name for name in
                                     ('checkpoint_epoch_1.pth', 'checkpoint_epoch_2.pth', 'selected.pth')]
            for index, path in enumerate((older, newer, explicit)):
                path.touch()
                os.utime(path, (100 + index, 100 + index))
            cfg = self._task_config(root, 5)
            self.assertEqual(select_training_checkpoints(cfg, ckpt_dir, pretrained_model=root / 'missing.pth'),
                             ([str(newer), str(older)], None))
            self.assertEqual(select_training_checkpoints(cfg, ckpt_dir, ckpt=explicit,
                                                         pretrained_model=root / 'missing.pth'),
                             ([str(explicit)], None))
            with self.assertRaisesRegex(FileNotFoundError, 'Resume checkpoint'):
                select_training_checkpoints(cfg, ckpt_dir, ckpt=root / 'missing.pth')

    def test_automatic_initialization_scope_and_dataset_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = self._task_config(root, 3)
            cfg.TASK_SCENE.ENABLED = False
            self.assertEqual(select_training_checkpoints(cfg, root / 'ckpt'), ([], None))
            cfg.TASK_SCENE.ENABLED = True
            cfg.MODEL.NAME = 'CenterPoint'
            with self.assertRaisesRegex(ValueError, 'VoxelRCNN'):
                select_training_checkpoints(cfg, root / 'ckpt')
            cfg.MODEL.NAME = 'VoxelRCNN'
            cfg.TASK_SCENE.TRAIN_DATASETS = [EasyDict(CONFIG={'MAX_SWEEPS': 1})]
            with self.assertRaisesRegex(ValueError, 'share MAX_SWEEPS'):
                select_training_checkpoints(cfg, root / 'ckpt')

    def test_sampler_epoch_is_set_before_iterator_creation_after_resume(self):
        for merged in (False, True):
            with self.subTest(merge_all_iters=merged):
                sampler = SimpleNamespace(epoch=0)
                sampler.set_epoch = lambda epoch: setattr(sampler, 'epoch', epoch)
                class Loader:
                    dataset = SimpleNamespace(merge_all_iters_to_one_epoch=mock.Mock())
                    def __init__(self):
                        self.created_epochs = []
                    def __len__(self):
                        return 6
                    def __iter__(self):
                        self.created_epochs.append(sampler.epoch)
                        return iter([(sampler.epoch, i) for i in range(6)])
                loader = Loader()
                def consume_epoch(*args, **kwargs):
                    batches = [next(kwargs['dataloader_iter']) for _ in range(kwargs['total_it_each_epoch'])]
                    if not merged:
                        self.assertTrue(all(epoch == kwargs['cur_epoch'] for epoch, _ in batches))
                    return kwargs['accumulated_iter'] + len(batches)
                with mock.patch('simple_det3d.train.train_utils.train_one_epoch', side_effect=consume_epoch):
                    train_model(model=None, optimizer=None, train_loader=loader, model_func=None,
                                lr_scheduler=None, optim_cfg=SimpleNamespace(WARMUP_EPOCH=0),
                                start_epoch=1, total_epochs=3, start_iter=0, rank=1, tb_log=None,
                                ckpt_save_dir=Path('/unused'), train_sampler=sampler, scaler=mock.Mock(),
                                merge_all_iters_to_one_epoch=merged)
                self.assertEqual(loader.created_epochs, [1] if merged else [1, 2])

    def test_restricted_checkpoint_accepts_legacy_numpy_float_scalars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'legacy.pth'
            torch.save({'lr': np.float64(0.0003), 'score': np.float32(0.75),
                        'model_state': {'weight': torch.ones(2)}}, path)
            loaded = load_checkpoint(path, map_location='cpu')
        self.assertEqual(float(loaded['lr']), 0.0003)
        self.assertEqual(float(loaded['score']), 0.75)
        torch.testing.assert_close(loaded['model_state']['weight'], torch.ones(2))

    def test_restricted_checkpoint_still_rejects_unlisted_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'unlisted.pth'
            torch.save({'unlisted': Path('asset')}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(path, map_location='cpu')

    def test_checkpoint_records_best_model_selection_state(self):
        state = checkpoint_state(
            model=torch.nn.Linear(2, 1),
            epoch=4,
            it=20,
            best_metric=0.75,
            best_epoch=3,
            best_metric_name=METRIC_NAME,
        )
        self.assertEqual(state['best_metric'], 0.75)
        self.assertEqual(state['best_epoch'], 3)
        self.assertEqual(state['best_metric_name'], METRIC_NAME)

    def test_load_best_metric_uses_best_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            epoch_path = root / 'checkpoint_epoch_4.pth'
            best_path = root / 'best_model.pth'
            incompatible_path = root / 'other_metric.pth'
            torch.save({
                'best_metric': 0.70,
                'best_epoch': 3,
                'best_metric_name': METRIC_NAME,
            }, epoch_path)
            torch.save({
                'best_metric': 0.80,
                'best_epoch': 2,
                'best_metric_name': METRIC_NAME,
            }, best_path)
            torch.save({
                'best_metric': 0.99,
                'best_epoch': 4,
                'best_metric_name': 'different_metric',
            }, incompatible_path)

            state = load_best_metric_state(
                [epoch_path, best_path, incompatible_path],
                expected_metric_name=METRIC_NAME,
            )

        self.assertEqual(state['best_metric'], 0.80)
        self.assertEqual(state['best_epoch'], 2)
        self.assertEqual(state['source'], str(best_path))

    def _run_one_resumed_epoch(self, current_metric, initial_best_metric):
        cfg = SimpleNamespace(DATA_CONFIG={
            'WEAK_SUPERVISION': {'BEST_METRIC': METRIC_NAME},
        })
        logger = mock.Mock()
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch(
                    'simple_det3d.train.train_utils.train_one_epoch',
                    return_value=10,
                ), \
                mock.patch(
                    'simple_det3d.train.train_utils.eval_epoch',
                    return_value={METRIC_NAME: current_metric},
                ), \
                mock.patch(
                    'simple_det3d.train.train_utils.save_checkpoint',
                ) as save_checkpoint_mock:
            train_model(
                model=model,
                optimizer=None,
                train_loader=[object()],
                model_func=None,
                lr_scheduler=None,
                optim_cfg=SimpleNamespace(WARMUP_EPOCH=0),
                start_epoch=1,
                total_epochs=2,
                start_iter=5,
                rank=0,
                tb_log=None,
                ckpt_save_dir=Path(temp_dir),
                ckpt_save_interval=1,
                logger=logger,
                eval_interval=1,
                test_loader=[object()],
                cfg=cfg,
                initial_best_metric=initial_best_metric,
                initial_best_epoch=1,
                best_metric_name=METRIC_NAME,
            )
            return save_checkpoint_mock

    def test_worse_metric_after_resume_does_not_overwrite_best_model(self):
        save_checkpoint_mock = self._run_one_resumed_epoch(
            current_metric=0.60,
            initial_best_metric=0.80,
        )
        best_calls = [
            call for call in save_checkpoint_mock.call_args_list
            if str(call.kwargs['filename']).endswith('best_model')
        ]
        self.assertEqual(best_calls, [])
        self.assertEqual(save_checkpoint_mock.call_count, 1)
        epoch_state = save_checkpoint_mock.call_args.args[0]
        self.assertEqual(epoch_state['best_metric'], 0.80)
        self.assertEqual(epoch_state['best_epoch'], 1)

    def test_better_metric_after_resume_updates_best_metadata(self):
        save_checkpoint_mock = self._run_one_resumed_epoch(
            current_metric=0.90,
            initial_best_metric=0.80,
        )
        self.assertEqual(save_checkpoint_mock.call_count, 2)
        for call in save_checkpoint_mock.call_args_list:
            state = call.args[0]
            self.assertEqual(state['best_metric'], 0.90)
            self.assertEqual(state['best_epoch'], 2)
            self.assertEqual(state['best_metric_name'], METRIC_NAME)


if __name__ == '__main__':
    unittest.main()
