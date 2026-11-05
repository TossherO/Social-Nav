"""Batched association costs; KF uncertainty and gates stay in the local world frame."""
from __future__ import annotations

import numpy as np

from .geometry import corners_xy, intersection_polygon, polygon_area, similarity


def bev_diou(a, b):
    """Rotated BEV DIoU: deliberately independent of height and vertical position."""
    pa, pb = corners_xy(a), corners_xy(b)
    intersection = polygon_area(intersection_polygon(pa, pb))
    union = np.prod(a[3:5]) + np.prod(b[3:5]) - intersection
    corners = np.r_[pa, pb]
    diagonal = np.sum((corners.max(axis=0)-corners.min(axis=0))**2)
    return float(intersection/max(union, 1e-12) - np.sum((a[:2]-b[:2])**2)/max(diagonal, 1e-12))


class Association:
    def __init__(self, config):
        self.config = config
        self.metric = config['association']
        self.backend = config.get('association_backend', 'cpu')
        self.weight = config.get('diou_weight', .2)
        self.motion_gate = config.get('motion_gate_chi2', 9.21)
        if self.metric not in ('iou', 'giou', 'diou', 'xy_diou'):
            raise ValueError('Unknown tracking association metric')
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('association_backend must be cpu or cuda')
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError('diou_weight must be in [0,1]')
        if not np.isfinite(self.motion_gate) or self.motion_gate <= 0:
            raise ValueError('motion_gate_chi2 must be positive')
        self.gpu = None
        if self.backend == 'cuda':
            from .association_cuda import CudaOverlap
            self.gpu = CudaOverlap(config.get('association_device', 'cuda:0'))
            self.gpu.warmup(bev=self.metric == 'xy_diou',
                            metric='diou' if self.metric == 'xy_diou' else self.metric)
        self.metadata = dict(metric=self.metric, backend=self.backend)
        if self.metric == 'xy_diou':
            self.metadata.update(diou_weight=self.weight, motion_gate_chi2=self.motion_gate)
        if self.gpu is not None:
            self.metadata.update(self.gpu.metadata)

    def costs(self, detections, predicted, covariance):
        count, tracks = len(detections), len(predicted)
        costs = np.full((count, tracks), 1e6)
        if not count or not tracks:
            return costs
        delta = detections[:, None, :2]-predicted[None, :, :2]
        eligible = np.sum(delta**2, axis=-1) <= self.config['center_gate_m']**2
        motion = None
        if self.metric == 'xy_diou':
            innovation = covariance[:, :2, :2] + np.eye(2)*self.config['position_noise_m']**2
            precision = np.linalg.inv(innovation)
            distance = np.maximum(0, np.einsum('nmi,mij,nmj->nm', delta, precision, delta))
            eligible &= distance <= self.motion_gate
            motion = np.sqrt(distance/self.motion_gate)
        if not eligible.any():
            return costs
        if self.gpu is not None:
            scores = self.gpu.similarities(detections, predicted, bev=self.metric == 'xy_diou',
                                           metric='diou' if self.metric == 'xy_diou' else self.metric,
                                           eligible=eligible)
        else:
            scores = np.zeros((count, tracks))
            for i, j in zip(*np.nonzero(eligible)):
                scores[i, j] = (bev_diou(detections[i], predicted[j]) if self.metric == 'xy_diou'
                                else similarity(detections[i], predicted[j], self.metric))
        if motion is None:
            costs[eligible] = 1-scores[eligible]
        else:
            # Scale to [0,2], like 1-IoU/GIoU/DIoU; thresholds remain metric-specific.
            mixed = 2*(1-self.weight)*motion + self.weight*(1-scores)
            costs[eligible] = mixed[eligible]
        if not np.isfinite(costs).all():
            raise ValueError('Nonfinite tracking association cost')
        return costs
