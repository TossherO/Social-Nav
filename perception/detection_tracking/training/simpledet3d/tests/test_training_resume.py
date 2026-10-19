import tempfile
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np
from simple_det3d.utils.common_utils import load_checkpoint

from simple_det3d.train.train_utils import (
    checkpoint_state,
    load_best_metric_state,
    train_model,
)


METRIC_NAME = 'CenterHit@0.50m'


class TrainingResumeTest(unittest.TestCase):
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
