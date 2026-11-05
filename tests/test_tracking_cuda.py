"""Explicit small CUDA geometry and complete tracking parity checks; no detector."""
import json
import unittest

import numpy as np

from perception.detection_tracking.inference.config import DEFAULT
from perception.detection_tracking.inference.association import Association
from perception.detection_tracking.inference.association_cuda import CudaOverlap
from perception.detection_tracking.inference.tracker import SimpleTrack

CONFIG = json.loads(DEFAULT.read_text())['tracking']


class CudaAssociationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('Explicit CUDA tests need a visible GPU')

    def test_rotated_matrices_and_touching_boxes_match_cpu(self):
        random = np.random.default_rng(13)
        a = np.c_[random.uniform(-.6, .6, (32, 3)), random.uniform(.15, 2, (32, 3)),
                  random.uniform(-np.pi, np.pi, 32), np.ones(32)]
        b = a.copy();b[:, :3] += random.normal(0, .2, (32, 3))
        a[:3] = [0, 0, 0, .6, .5, 1.7, 0, 1]
        b[0] = a[0];b[1] = a[0];b[1, 0] = .605;b[2] = a[0];b[2, 0] = .595
        covariance = np.repeat(np.eye(10)[None], len(b), axis=0)
        for metric in ('iou', 'giou', 'diou', 'xy_diou'):
            cpu = Association(dict(CONFIG, association=metric, association_backend='cpu'))
            gpu = Association(dict(CONFIG, association=metric, association_backend='cuda'))
            expected, actual = cpu.costs(a, b, covariance), gpu.costs(a, b, covariance)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_translation_height_and_size_distance_terms(self):
        box = np.array([[0., 0, 0, .6, .5, 1.7, 0, 1]])
        other = np.repeat(box, 3, axis=0);other[0, 2] = .3;other[1, 3] = 1.2;other[2, 0] = .3
        gpu = CudaOverlap('cuda:0')
        original = gpu.similarities(box, other, bev=False)
        moved = box.copy();moved[:, :3] += [10000, -20000, 3000]
        shifted = other.copy();shifted[:, :3] += [10000, -20000, 3000]
        np.testing.assert_allclose(original, gpu.similarities(moved, shifted, bev=False), atol=2e-5)
        np.testing.assert_allclose(gpu.similarities(box, other[:1], bev=True), [[1]], atol=2e-5)
        self.assertEqual(gpu.similarities(box[:0], other, bev=True).shape, (0, 3))

    def test_track_ids_and_low_score_support_match_cpu(self):
        for metric in ('giou', 'diou', 'xy_diou'):
            trackers = [SimpleTrack(dict(CONFIG, association=metric, association_backend=backend)) for backend in ('cpu', 'cuda')]
            for i in range(12):
                boxes = np.array([[.08*i, 0, 0, .3 if i%2 else .9, .6, 1.7, .1*i, .2 if i in (4, 5) else .9],
                                  [2-.04*i, 1, 0, .6, .5, 1.7, 0, .9]])
                rows = [tracker.step(boxes, 1+i*.1) for tracker in trackers]
                self.assertEqual(rows[0], rows[1])

    def test_sparse_pairs_reuse_buffers_and_preserve_returned_results(self):
        import torch
        random = np.random.default_rng(37)
        gpu = CudaOverlap('cuda:0')
        for count in (2, 20, 3):
            boxes = np.c_[random.normal(size=(count, 3)), random.uniform(.1, 2, (count, 3)),
                          random.uniform(-np.pi, np.pi, count), np.ones(count)]
            mask = random.uniform(size=(count, count)) < .2
            settings = dict(CONFIG, association='giou', association_backend='cpu', center_gate_m=100)
            expected = 1-Association(settings).costs(boxes, boxes, np.repeat(np.eye(10)[None], count, 0))
            actual = gpu.similarities(boxes, boxes, bev=False, metric='giou', eligible=mask)
            np.testing.assert_allclose(actual[mask], expected[mask], atol=2e-8)
            np.testing.assert_array_equal(actual[~mask], 0)
            saved = actual.copy()
            with torch.cuda.stream(torch.cuda.Stream()):
                gpu.similarities(boxes[:1], boxes, bev=False)
            np.testing.assert_array_equal(actual, saved)
        empty = gpu.similarities(boxes, boxes, bev=False, eligible=np.zeros((count, count), bool))
        np.testing.assert_array_equal(empty, 0)
        with self.assertRaisesRegex(ValueError, 'pair budget'):
            gpu.similarities(np.zeros((257, 7)), np.zeros((257, 7)), bev=False)

    def test_nearly_parallel_thin_and_large_world_geometry(self):
        from perception.detection_tracking.inference.geometry import similarity
        boxes = np.array([[0., 0, 0, .02, 3., 1.7, 0., 1.],
                          [0., 0, 0, .02, 3., 1.7, 1.e-7, 1.],
                          [.02+1.e-6, 0, 0, .02, 3., 1.7, 0., 1.],
                          [0., 0, 0, .02, 3., 1.7, np.pi/2, 1.]])
        boxes[:, :3] += [1.e6, -2.e6, 3000]
        gpu = CudaOverlap('cuda:0')
        # CPU reference uses the same local-origin translation to avoid its own
        # uncentred shoelace cancellation in large world coordinates.
        local = boxes.copy();local[:, :3] -= boxes[0, :3]
        for metric in ('iou', 'giou', 'diou'):
            expected = [[similarity(a, b, metric) for b in local] for a in local]
            np.testing.assert_allclose(gpu.similarities(boxes, boxes, bev=False, metric=metric),
                                       expected, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
