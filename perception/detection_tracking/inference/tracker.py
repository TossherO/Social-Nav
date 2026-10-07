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

from .geometry import similarity, validate_boxes, wrap_angle


class Track:
    def __init__(self, identity, box, stamp, position_noise):
        self.id = identity
        self.x = np.r_[box[:7], np.zeros(3)]
        self.p = np.diag([position_noise**2]*3 + [0.3**2]*3 + [0.5**2] + [4.0]*3)
        self.score = float(box[7])
        self.last_observed = self.last_supported = self.stamp = stamp
        self.hits = 1
        self.observed = True
        self.support = 'high_score_detection'

    def predict(self, stamp, acceleration_noise):
        dt = stamp-self.stamp
        if dt <= 0:
            raise ValueError('Track timestamps must increase')
        f = np.eye(10)
        f[:3, 7:10] = np.eye(3)*dt
        q = np.zeros((10, 10))
        for axis in range(3):
            g = np.array([dt*dt/2, dt])
            q[np.ix_([axis, axis+7], [axis, axis+7])] = np.outer(g, g)*acceleration_noise**2
        q[3:6, 3:6] = np.eye(3)*0.03**2*dt
        q[6, 6] = 0.2**2*dt
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.x[6] = wrap_angle(self.x[6])
        self.stamp, self.observed, self.support = stamp, False, 'motion_prediction'

    def update(self, box, stamp, position_noise):
        measurement = box[:7].copy()
        # Pedestrian geometry has a pi heading ambiguity; never interpret as gaze.
        difference = (measurement[6]-self.x[6]+np.pi/2) % np.pi - np.pi/2
        measurement[6] = self.x[6]+difference
        h = np.c_[np.eye(7), np.zeros((7, 3))]
        r = np.diag([position_noise**2]*3 + [0.3**2]*3 + [0.5**2])
        innovation = h @ self.p @ h.T + r
        gain = np.linalg.solve(innovation, h @ self.p).T
        self.x += gain @ (measurement-h @ self.x)
        residual = np.eye(10)-gain @ h
        self.p = residual @ self.p @ residual.T + gain @ r @ gain.T
        self.x[6] = wrap_angle(self.x[6])
        self.last_observed = self.last_supported = stamp
        self.hits += 1
        self.score, self.observed, self.support = float(box[7]), True, 'high_score_detection'

    def box(self):
        return np.r_[self.x[:7], self.score]


class SimpleTrack:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        c = self.config
        if c['association'] not in ('giou', 'iou', 'diou'):
            raise ValueError('Only explicitly implemented CPU association metrics are accepted')
        for key, value in c.items():
            if key != 'association' and (not isinstance(value, (int, float)) or not math.isfinite(value)):
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
        self.tracks = []
        self.next_id = 0
        self.last_stamp = None

    def reset(self):
        self.tracks.clear()
        self.next_id, self.last_stamp = 0, None

    def _match(self, detections, tracks, max_cost):
        if not len(detections) or not tracks:
            return []
        costs = np.full((len(detections), len(tracks)), 1e6)
        for i, box in enumerate(detections):
            for j, track in enumerate(tracks):
                if np.linalg.norm(box[:2]-track.x[:2]) <= self.config['center_gate_m']:
                    cost = 1-similarity(box, track.box(), self.config['association'])
                    if cost <= max_cost:
                        costs[i, j] = cost
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
        for track in self.tracks:
            track.predict(stamp, c['acceleration_noise_mps2'])
        high = boxes[boxes[:, 7] >= c['birth_score']]
        low = boxes[(boxes[:, 7] >= c['support_score']) & (boxes[:, 7] < c['birth_score'])]
        matched = self._match(high, self.tracks, c['association_cost_max'])
        used_dets, used_tracks = set(), set()
        for d, t in matched:
            self.tracks[t].update(high[d], stamp, c['position_noise_m'])
            used_dets.add(d)
            used_tracks.add(t)
        remaining = [t for i, t in enumerate(self.tracks) if i not in used_tracks]
        for _, t in self._match(low, remaining, 1-c['support_similarity_min']):
            remaining[t].last_supported = stamp
            remaining[t].support = 'low_score_support'
        for index, box in enumerate(high):
            if index not in used_dets:
                self.tracks.append(Track(self.next_id, box, stamp, c['position_noise_m']))
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
            })
        return output
