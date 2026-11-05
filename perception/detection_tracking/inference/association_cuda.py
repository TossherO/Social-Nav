"""Buffered CUDA association; one fused kernel and one result synchronization."""
from __future__ import annotations

import numpy as np


class CudaOverlap:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA association was requested but is unavailable; select cpu explicitly for replay')
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise ValueError('Tracking association CUDA device is not visible')
        try:
            from .association_kernel import overlap_kernel
        except ImportError as exc:
            raise RuntimeError('CUDA association needs the Triton package matching PyTorch; select cpu explicitly if unavailable') from exc
        self.kernel = overlap_kernel
        self.box_capacity = self.pair_capacity = 0
        with torch.cuda.device(self.device):
            self.ready = torch.cuda.Event()
        self.metadata = dict(device=str(self.device), geometry='rotated_rectangle_intersection',
                             implementation='triton_fused', dtype='float64', pairs='eligible_only')

    def _reserve(self, boxes, pairs):
        t = self.torch
        # One tracker owns these buffers; calls are serial and synchronize before
        # returning, so the next call can safely overwrite pinned host inputs.
        if boxes > self.box_capacity:
            self.box_capacity = max(32, 1 << (boxes-1).bit_length())
            self.host_boxes = t.empty((self.box_capacity, 7), dtype=t.float64, pin_memory=True)
            self.box_array = self.host_boxes.numpy()
            self.device_boxes = t.empty_like(self.host_boxes, device=self.device)
        if pairs > self.pair_capacity:
            self.pair_capacity = max(64, 1 << (pairs-1).bit_length())
            self.host_pairs = t.empty((self.pair_capacity, 2), dtype=t.int32, pin_memory=True)
            self.pair_array = self.host_pairs.numpy()
            self.device_pairs = t.empty_like(self.host_pairs, device=self.device)
            self.device_result = t.empty(self.pair_capacity, dtype=t.float64, device=self.device)
            self.host_result = t.empty(self.pair_capacity, dtype=t.float64, pin_memory=True)
            self.result_array = self.host_result.numpy()

    def warmup(self, *, bev, metric):
        """Compile/cache the selected specialization before live frame timing."""
        box = np.array([[0., 0., 0., .6, .5, 1.7, 0.]])
        self.similarities(box, box, bev=bev, metric=metric)

    def similarities(self, detections, predicted, *, bev, metric='diou', eligible=None):
        if metric not in ('iou', 'giou', 'diou'):
            raise ValueError('Unknown CUDA association similarity')
        n, m = len(detections), len(predicted)
        if n*m > 65536:
            raise ValueError('CUDA tracking association pair budget exceeded')
        scores = np.zeros((n, m))
        if not n or not m:
            return scores
        if eligible is None:
            rows, cols = np.indices((n, m)).reshape(2, -1)
        else:
            if eligible.shape != (n, m):
                raise ValueError('Association eligibility shape does not match boxes')
            rows, cols = np.nonzero(eligible)
        count = len(rows)
        if not count:
            return scores
        t = self.torch
        with t.cuda.device(self.device), t.no_grad():
            self._reserve(n+m, count)
            self.box_array[:n] = detections[:, :7]
            self.box_array[n:n+m] = predicted[:, :7]
            self.pair_array[:count, 0], self.pair_array[:count, 1] = rows, cols+n
            self.device_boxes[:n+m].copy_(self.host_boxes[:n+m], non_blocking=True)
            self.device_pairs[:count].copy_(self.host_pairs[:count], non_blocking=True)
            self.kernel[(count,)](self.device_boxes, self.device_pairs, self.device_result,
                                  BEV=bev, METRIC=metric, num_warps=4, enable_fp_fusion=False)
            self.host_result[:count].copy_(self.device_result[:count], non_blocking=True)
            self.ready.record(t.cuda.current_stream(self.device))
            self.ready.synchronize()
        scores[rows, cols] = self.result_array[:count]
        return scores
