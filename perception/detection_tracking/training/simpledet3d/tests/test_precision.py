import unittest

import torch

from simple_det3d.train.train_utils import checkpoint_state
from simple_det3d.utils.precision_utils import (
    normalize_precision,
    precision_dtype,
    resolve_precision,
)


class _FakeScaler:
    def state_dict(self):
        return {'scale': 1024.0}


class PrecisionUtilsTest(unittest.TestCase):
    def test_normalize_precision_aliases(self):
        self.assertEqual(normalize_precision('half'), 'fp16')
        self.assertEqual(normalize_precision('float32'), 'fp32')
        self.assertEqual(normalize_precision('bfloat16'), 'bf16')
        self.assertIs(precision_dtype('fp16'), torch.float16)
        self.assertIs(precision_dtype('bf16'), torch.bfloat16)

    def test_resolve_precision_precedence_and_compatibility_alias(self):
        self.assertEqual(resolve_precision(None, 'bf16'), 'bf16')
        self.assertEqual(resolve_precision('fp16', 'bf16'), 'fp16')
        self.assertEqual(resolve_precision(None, 'fp32', use_amp=True), 'fp16')
        with self.assertRaises(ValueError):
            resolve_precision('bf16', 'fp32', use_amp=True)

    def test_rejects_unknown_precision(self):
        with self.assertRaises(ValueError):
            normalize_precision('int8')

    def test_checkpoint_keeps_fp32_weights_and_scaler_state(self):
        model = torch.nn.Linear(2, 1).float()
        state = checkpoint_state(
            model=model,
            epoch=2,
            it=10,
            scaler=_FakeScaler(),
            precision='fp16',
        )
        self.assertEqual(state['precision'], 'fp16')
        self.assertEqual(state['scaler_state'], {'scale': 1024.0})
        self.assertTrue(all(x.dtype == torch.float32 for x in state['model_state'].values()))

    def test_checkpoint_keeps_ema_state_on_cpu(self):
        model = torch.nn.Linear(2, 1)
        teacher = torch.nn.Linear(2, 1)
        state = checkpoint_state(
            model=model,
            epoch=1,
            it=2,
            precision='fp32',
            ema_teacher=teacher,
        )
        self.assertIn('ema_model_state', state)
        self.assertTrue(
            all(x.device.type == 'cpu' for x in state['ema_model_state'].values())
        )


if __name__ == '__main__':
    unittest.main()
