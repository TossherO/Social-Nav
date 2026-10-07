import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import test as test_tool


class _DummyDataset:
    class_names = ['Pedestrian']

    def __len__(self):
        return 1

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names):
        return [{
            'frame_id': 'dummy',
            'name': np.empty(0, dtype='<U10'),
            'boxes_lidar': np.empty((0, 7), dtype=np.float32),
            'score': np.empty(0, dtype=np.float32),
        }]

    def evaluation(self, det_annos, class_names):
        return 'dummy evaluation', {'dummy_metric': 1.0}


class _DummyLoader:
    dataset = _DummyDataset()

    def __len__(self):
        return 1

    def __iter__(self):
        yield {}


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, batch_dict):
        return [{}], {'gt': 0, 'roi_0.3': 0, 'rcnn_0.3': 0}


class _DummyProgress:
    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


class EvaluationArgumentCompatibilityTest(unittest.TestCase):
    def test_train_namespace_does_not_require_test_only_save_vis(self):
        cfg = SimpleNamespace(
            MODEL=SimpleNamespace(
                POST_PROCESSING=SimpleNamespace(RECALL_THRESH_LIST=[0.3]),
            ),
        )
        # Mirrors the relevant part of tools/train.py's Namespace: precision
        # exists, while the test-only save_vis option intentionally does not.
        args = argparse.Namespace(precision='fp32')

        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(test_tool, 'load_data_to_gpu'), \
                mock.patch.object(test_tool.tqdm, 'tqdm', return_value=_DummyProgress()):
            result_dir = Path(temp_dir)
            result = test_tool.eval_one_epoch(
                _DummyModel(), _DummyLoader(), cfg, args,
                logging.getLogger(__name__), result_dir=result_dir,
            )

            self.assertEqual(result['dummy_metric'], 1.0)
            self.assertTrue((result_dir / 'result.pkl').is_file())


if __name__ == '__main__':
    unittest.main()
