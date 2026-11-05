"""SimpleTrack-style KF/Hungarian/GIoU with explicit causal lifecycle semantics.

Reference: supplied simpletrack.zip and tusen-ai/SimpleTrack (MIT). This is an
adaptation, not a byte-identical benchmark implementation. Low-score matches
only support existing tracks; they do not create fake KF measurements.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import validate_boxes, wrap_angle
from .association import Association


def prediction_matrices(dt, acceleration_noise):
    """Same constant-velocity model, shared by tracks with the same elapsed time."""
    f = np.eye(10)
    f[:3, 7:10] = np.eye(3)*dt
    g = np.array([dt*dt/2, dt])
    motion = np.outer(g, g)*acceleration_noise**2
    q = np.zeros((10, 10))
    axes = np.arange(3)
    q[axes, axes], q[axes, axes+7] = motion[0, 0], motion[0, 1]
    q[axes+7, axes], q[axes+7, axes+7] = motion[1, 0], motion[1, 1]
    q[3:6, 3:6] = np.eye(3)*0.03**2*dt
    q[6, 6] = 0.2**2*dt
    return f, q


class Track:
    def __init__(self, identity, box, stamp, position_noise, detection_index=None):
        self.id = identity
        self.x = np.r_[box[:7], np.zeros(3)]
        self.p = np.diag([position_noise**2]*3 + [0.3**2]*3 + [0.5**2] + [4.0]*3)
        self.score = float(box[7])
        self.last_observed = self.last_supported = self.stamp = stamp
        self.hits = 1
        self.observed = True
        self.support = 'high_score_detection'
        self.detection_index = detection_index

    def predict(self, stamp, acceleration_noise, *, matrices=None):
        dt = stamp-self.stamp
        if dt <= 0:
            raise ValueError('Track timestamps must increase')
        f, q = prediction_matrices(dt, acceleration_noise) if matrices is None else matrices
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.x[6] = wrap_angle(self.x[6])
        self.stamp, self.observed, self.support = stamp, False, 'motion_prediction'
        self.detection_index = None

    def update(self, box, stamp, position_noise, detection_index=None):
        measurement = box[:7].copy()
        # Pedestrian geometry has a pi heading ambiguity; never interpret as gaze.
        difference = (measurement[6]-self.x[6]+np.pi/2) % np.pi - np.pi/2
        measurement[6] = self.x[6]+difference
        r = np.diag([position_noise**2]*3 + [0.3**2]*3 + [0.5**2])
        # H=[I7,0]: slicing preserves the same Joseph covariance update while
        # avoiding repeated construction and dense multiplication by H.
        innovation = self.p[:7, :7] + r
        gain = np.linalg.solve(innovation, self.p[:7]).T
        self.x += gain @ (measurement-self.x[:7])
        residual = np.eye(10)
        residual[:, :7] -= gain
        self.p = residual @ self.p @ residual.T + gain @ r @ gain.T
        self.x[6] = wrap_angle(self.x[6])
        self.last_observed = self.last_supported = stamp
        self.hits += 1
        self.score, self.observed, self.support = float(box[7]), True, 'high_score_detection'
        self.detection_index = detection_index

    def box(self):
        return np.r_[self.x[:7], self.score]


class SimpleTrack:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        c = self.config
        for key, value in c.items():
            if key not in ('association', 'association_backend', 'association_device') and (
                    not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'Invalid tracking parameter: {key}')
        for key in ('center_gate_m', 'max_unobserved_s', 'max_unsupported_s', 'reset_gap_s',
                    'acceleration_noise_mps2', 'position_noise_m'):
            if c[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if not 0 <= c['support_score'] <= c['birth_score'] <= 1:
            raise ValueError('Expected 0 <= support_score <= birth_score <= 1')
        if not 0 <= c['association_cost_max'] <= 2 or not -1 <= c['support_similarity_min'] <= 1:
            raise ValueError('Invalid association cost/similarity threshold')
        if not isinstance(c['min_hits'], int) or c['min_hits'] < 1:
            raise ValueError('min_hits must be a positive integer')
        self.association = Association(c)
        self.tracks = []
        self.next_id = 0
        self.last_stamp = None

    def reset(self):
        self.tracks.clear()
        self.next_id, self.last_stamp = 0, None

    def _match(self, detections, tracks, max_cost):
        if not len(detections) or not tracks:
            return []
        costs = self.association.costs(detections, np.asarray([t.box() for t in tracks]),
                                       np.asarray([t.p for t in tracks]))
        costs[costs > max_cost] = 1e6
        # Explicit unmatched slots prevent rejected pairs from stealing valid matches.
        augmented = np.c_[costs, np.full((len(detections), len(detections)), max_cost+1e-6)]
        rows, cols = linear_sum_assignment(augmented)
        return [(int(i), int(j)) for i, j in zip(rows, cols)
                if j < len(tracks) and costs[i, j] <= max_cost]

    def step(self, boxes, stamp):
        boxes = validate_boxes(boxes)
        if not math.isfinite(stamp) or (self.last_stamp is not None and stamp <= self.last_stamp):
            raise ValueError('Non-finite, duplicate or reversed tracking timestamp')
        c = self.config
        # Never associate across tracks whose lifetime already expired in a dropped-frame gap.
        self.tracks = [t for t in self.tracks
                       if stamp-t.last_observed < c['max_unobserved_s']
                       and stamp-t.last_supported < c['max_unsupported_s']]
        matrices = {}
        for track in self.tracks:
            dt = stamp-track.stamp
            if dt not in matrices:
                matrices[dt] = prediction_matrices(dt, c['acceleration_noise_mps2'])
            track.predict(stamp, c['acceleration_noise_mps2'], matrices=matrices[dt])
        high_indices = np.flatnonzero(boxes[:, 7] >= c['birth_score'])
        high = boxes[high_indices]
        low = boxes[(boxes[:, 7] >= c['support_score']) & (boxes[:, 7] < c['birth_score'])]
        matched = self._match(high, self.tracks, c['association_cost_max'])
        used_dets, used_tracks = set(), set()
        for d, t in matched:
            self.tracks[t].update(high[d], stamp, c['position_noise_m'], int(high_indices[d]))
            used_dets.add(d)
            used_tracks.add(t)
        remaining = [t for i, t in enumerate(self.tracks) if i not in used_tracks]
        for _, t in self._match(low, remaining, 1-c['support_similarity_min']):
            remaining[t].last_supported = stamp
            remaining[t].support = 'low_score_support'
        for index, box in enumerate(high):
            if index not in used_dets:
                self.tracks.append(Track(self.next_id, box, stamp, c['position_noise_m'], int(high_indices[index])))
                self.next_id += 1
        self.last_stamp = stamp
        output = []
        for track in self.tracks:
            output.append({
                'track_id': str(track.id), 'state': 'observed' if track.observed else 'predicted',
                'confirmed': track.hits >= c['min_hits'], 'box': track.box().tolist(),
                'velocity_xyz_mps': track.x[7:].tolist() if track.hits >= max(3, c['min_hits']) else None,
                'covariance': track.p.reshape(-1).tolist(),
                'last_observed_stamp_s': track.last_observed, 'last_supported_stamp_s': track.last_supported,
                'support': track.support, 'hits': track.hits,
                'detection_index': track.detection_index,
            })
        return output
