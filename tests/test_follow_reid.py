import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from decision.osnet.reid import FollowReid, OsnetEmbedder


class OsnetEmbedderTest(unittest.TestCase):
    def test_gpu_chunks_keep_every_candidate_in_order(self):
        class Engine:
            max_batch_size = 2
            onnx_sha256 = 'source'
            device = 'cuda:0'
            tensorrt_version = 'test'

            def __init__(self):
                self.inputs = []

            def __call__(self, images):
                self.inputs.append(images)
                return np.repeat(images[:, 0, 0, :1], 512, axis=1)

        engine = Engine()
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'osnet.engine'
            plan.write_bytes(b'example')
            with patch('decision.osnet.reid_trt.TensorRtOsnet', return_value=engine):
                encoder = OsnetEmbedder(plan)
        values = np.array([17, 82, 139, 198, 253], dtype=np.float32)
        crops = [Image.new('RGB', (17, 47), (int(red), 0, 0)) for red in values]
        features = encoder(crops)
        self.assertEqual(encoder.backend, 'tensorrt_cuda')
        self.assertEqual([images.shape for images in engine.inputs],
                         [(2, 3, 256, 128), (2, 3, 256, 128), (1, 3, 256, 128)])
        self.assertEqual(features.shape, (5, 512))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_allclose(features[:, 0], (values / 255 - .485) / .229)
        self.assertEqual(encoder([]).shape, (0, 512))

    def test_engine_rejects_missing_or_mismatched_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / 'osnet.engine'
            plan.write_bytes(b'example')
            with self.assertRaisesRegex(ValueError, 'requires readable engine metadata'):
                OsnetEmbedder(plan)
            plan.with_suffix('.engine.json').write_text('{"engine_sha256": "wrong"}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata does not match'):
                OsnetEmbedder(plan)


class Encoder:
    def __init__(self):
        self.batches = []

    def __call__(self, crops):
        labels = [crop.getpixel((0, 0))[0] for crop in crops]
        self.batches.append(labels)
        vectors = {1: [1, 0, 0], 2: [.96, .28, 0], 3: [0, 0, 1], 4: [.8, .6, 0]}
        return np.asarray([vectors[label] for label in labels])


class FollowReidTest(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        buf = io.BytesIO()
        Image.new('RGB', (24, 80), (1, 0, 0)).save(buf, format='PNG')
        crop = buf.getvalue()
        reference = dict(crop_sha256=hashlib.sha256(crop).hexdigest(), clock_domain='runtime_bag',
                         local_identity=['observer', 2, 'map'], rgb_stamp_s=1., rgb_stamp_ns=1_000_000_000)
        self.reid = FollowReid(self.encoder, crop, reference)

    def step(self, offset, items, *, scope=('observer', 2, 'map'), association_status='complete'):
        pixels = np.zeros((80, 24 * max(1, len(items)), 3), dtype=np.uint8)
        candidates = []
        positions = {}
        for i, (label, identity, lifecycle, position) in enumerate(items):
            pixels[:, i * 24:(i + 1) * 24, 0] = label
            track = None if identity is None else dict(identity=['observer', 2, identity],
                                                       lifecycle_generation=lifecycle)
            candidates.append(dict(id=i, box=[i * 24, 0, (i + 1) * 24, 80], score=.99, track=track))
            if position is not None:
                positions[i] = position
        record = dict(rgb_stamp_ns=1_000_000_000 + offset * 100_000_000,
                      clock_domain='runtime_bag', local_identity=scope, candidates=candidates,
                      association_status=association_status)
        return self.reid.step(pixels, record, track_positions=positions)

    def test_rank_all_people_and_reject_unassociated_best_or_close_competition(self):
        runner_up = (4, 'a', 1, (0., 0.))
        similarly_dressed = (2, 'a', 1, (0., 0.))
        unassociated = (1, None, None, None)
        result = self.step(1, [runner_up, unassociated])
        self.assertEqual(self.encoder.batches, [[1], [4, 1]])
        self.assertEqual(result['reason'], 'appearance_without_track')
        self.assertIsNone(result['identity'])
        self.assertIsNone(result['candidate_id'])
        self.assertEqual(result['best_candidate_id'], 1)
        self.assertEqual(result['second_candidate_id'], 0)
        result = self.step(2, [similarly_dressed, unassociated])
        self.assertEqual(result['reason'], 'appearance_ambiguous')

    def test_unavailable_association_never_yields_identity(self):
        strong = (1, 'a', 1, (0., 0.))
        weak = (3, 'b', 1, (0., 0.))
        for offset, items in enumerate(([], [strong], [weak]), 1):
            result = self.step(offset, items, association_status='pose_unavailable')
            self.assertIsNone(result['identity'])

    def test_recovery_needs_a_new_source_and_same_exact_lifecycle(self):
        a = (1, 'a', 1, (0., 0.))
        b = (1, 'b', 2, (.5, 0.))
        recycled_b = (1, 'b', 3, (.5, 0.))
        self.assertEqual(self.step(1, [a])['reason'], 'awaiting_confirmation')
        self.assertEqual(self.step(1, [a])['reason'], 'source_not_new')
        self.assertEqual(self.step(2, [a])['status'], 'recovered')
        self.assertEqual(self.step(3, [a])['status'], 'same')
        self.assertEqual(self.step(4, [])['reason'], 'no_detections')
        self.assertEqual(self.step(5, [b])['reason'], 'awaiting_confirmation')
        self.assertEqual(self.step(6, [recycled_b])['confirmation_count'], 1)
        result = self.step(7, [recycled_b])
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual(result['identity'], ['observer', 2, 'b'])
        self.assertEqual(result['lifecycle_generation'], 3)

    def test_missing_input_time_gap_scope_and_position_reset_continuity(self):
        a = (1, 'a', 1, (0., 0.))
        far = (1, 'b', 2, (10., 0.))
        self.step(1, [a])
        self.step(2, [a])
        self.assertEqual(self.step(3, [far])['reason'], 'position_inconsistent')
        self.assertEqual(self.step(4, [a])['confirmation_count'], 1)
        self.assertEqual(self.step(5, [a], scope=('observer', 3, 'map'))['reason'], 'source_scope_changed')
        self.assertEqual(self.step(6, [a])['confirmation_count'], 1)
        self.reid.interrupt()
        self.assertEqual(self.step(7, [a])['confirmation_count'], 1)
        self.assertEqual(self.step(18, [a])['confirmation_count'], 1)

    def test_gallery_preserves_original_and_excludes_ambiguous_observations(self):
        a = (1, 'a', 1, (0., 0.))
        hard_negative = (2, 'b', 1, (0., 0.))
        original = self.reid.anchor.copy()
        for offset in range(1, 5):
            self.step(offset, [a, hard_negative])
        self.assertEqual(len(self.reid.gallery), 0)
        for offset in range(5, 30):
            self.step(offset, [a])
        self.assertEqual(len(self.reid.gallery), self.reid.policy.gallery_size)
        np.testing.assert_equal(self.reid.anchor, original)
        self.assertFalse(self.reid.anchor.flags.writeable)
        result = self.step(30, [(3, 'b', 1, (.5, 0.))])
        self.assertEqual(result['reason'], 'below_similarity')
        self.assertIsNone(result['candidate_id'])
