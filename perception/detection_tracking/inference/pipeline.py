"""Shared causal pipeline. Untrusted/raw odometry never silently becomes localization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
import uuid

import numpy as np

from .geometry import transform_boxes, validate_boxes, validate_transform
from .tracker import SimpleTrack


@dataclass
class Frame:
    observation_id: str
    sequence_id: str
    stamp_s: float
    points: np.ndarray
    sensor_frame: str = 'lidar'
    clock_domain: str = 'dataset'
    timestamp_source: str = 'sensor'
    world_from_lidar: np.ndarray | None = None
    world_frame: str = 'dataset_world'
    pose_source: str = 'missing'
    pose_health: str = 'invalid'
    pose_stamp_s: float | None = None
    rgb_asset_uri: str | None = None
    rgb_stamp_s: float | None = None

    def input_fingerprint(self):
        points = np.ascontiguousarray(self.points, dtype='<f4')
        digest = hashlib.sha256()
        digest.update(f'{self.sequence_id}\n{self.observation_id}\n{self.stamp_s:.17g}\n{self.sensor_frame}\n{points.shape}\n'.encode())
        digest.update(points.tobytes())
        return digest.hexdigest()


class PerceptionPipeline:
    def __init__(self, detector, tracking_config, session_id=None):
        self.detector = detector
        self.tracker = SimpleTrack(tracking_config)
        self.session_id = session_id or uuid.uuid4().hex
        self.generation = 0
        self.sequence = self.world_frame = self.clock_domain = None
        self.last_stamp = None
        self.localized = False
        self.last_transform = None

    def reset(self):
        self.tracker.reset()
        self.generation += 1
        self.localized = False
        self.last_transform = None

    def _begin(self, frame):
        if not math.isfinite(frame.stamp_s) or not frame.observation_id or not frame.sequence_id or not frame.sensor_frame:
            raise ValueError('Invalid observation identity/time/frame')
        changed = (frame.sequence_id != self.sequence or frame.clock_domain != self.clock_domain
                   or frame.world_frame != self.world_frame)
        if not changed and self.last_stamp is not None and frame.stamp_s <= self.last_stamp:
            raise ValueError('Duplicate/reversed timestamp; start a new sequence on replay reset')
        gap = self.last_stamp is not None and frame.stamp_s-self.last_stamp > self.tracker.config['reset_gap_s']
        if changed or gap:
            self.reset()
        self.sequence, self.world_frame, self.clock_domain = frame.sequence_id, frame.world_frame, frame.clock_domain
        self.last_stamp = frame.stamp_s

    def step(self, frame):
        start = time.perf_counter()
        self._begin(frame)
        boxes = validate_boxes(self.detector.predict(frame))
        detector_done = time.perf_counter()
        localized = (frame.world_from_lidar is not None and frame.pose_health == 'valid'
                     and frame.pose_source in ('dataset_pose', 'localization'))
        reason = None
        transform = validate_transform(frame.world_from_lidar) if localized else None
        if localized and frame.pose_stamp_s is not None:
            if not math.isfinite(frame.pose_stamp_s) or frame.pose_stamp_s > frame.stamp_s:
                raise ValueError('Localization timestamp must be finite and not later than the observation')
        if localized and self.last_transform is not None:
            displacement = np.linalg.norm(transform[:3, 3]-self.last_transform[:3, 3])
            relative_rotation = self.last_transform[:3, :3].T @ transform[:3, :3]
            rotation = np.arccos(np.clip((np.trace(relative_rotation)-1)/2, -1, 1))
            if (displacement > self.tracker.config['pose_jump_translation_m']
                    or rotation > self.tracker.config['pose_jump_rotation_rad']):
                self.reset()
                localized, reason = False, 'localization_discontinuity'
        if not localized:
            if self.localized:
                self.reset()
            tracks = []
            reason = reason or 'reliable_localization_unavailable'
        else:
            tracks = self.tracker.step(transform_boxes(boxes, transform), frame.stamp_s)
        self.localized = localized
        self.last_transform = transform if localized else None
        for track in tracks:
            track['track_id'] = f'{self.session_id}:{frame.sequence_id}:{self.generation}:{track["track_id"]}'
            track['lifecycle_generation'] = self.generation
            track['box_geometry'] = 'upright_envelope'
        complete = time.perf_counter()
        return {
            'observation_id': frame.observation_id, 'sequence_id': frame.sequence_id,
            'session_id': self.session_id, 'stamp_s': frame.stamp_s, 'clock_domain': frame.clock_domain,
            'timestamp_source': frame.timestamp_source, 'frame_id': frame.world_frame if localized else frame.sensor_frame,
            'detection_frame_id': frame.sensor_frame, 'generation': self.generation,
            'health': 'valid' if localized else 'degraded', 'reason': reason,
            'pose_source': frame.pose_source, 'pose_health': frame.pose_health,
            'pose_stamp_s': (frame.pose_stamp_s if frame.pose_stamp_s is not None else frame.stamp_s) if localized else None,
            'world_from_lidar': transform.tolist() if localized else None,
            'detections': boxes.tolist(), 'tracks': tracks, 'input_fingerprint': frame.input_fingerprint(),
            'rgb_asset_uri': frame.rgb_asset_uri, 'rgb_stamp_s': frame.rgb_stamp_s,
            'covariance_order': ['x','y','z','dx','dy','dz','yaw','vx','vy','vz'],
            'velocity_semantics': 'world_motion' if localized else 'unavailable',
            'timing_s': {'detector': detector_done-start, 'tracking': complete-detector_done,
                         'pipeline': complete-start},
        }

    def failure(self, frame, error):
        self.reset()
        return {'observation_id': frame.observation_id, 'sequence_id': frame.sequence_id,
                'session_id': self.session_id, 'stamp_s': frame.stamp_s, 'clock_domain': frame.clock_domain,
                'timestamp_source': frame.timestamp_source, 'frame_id': frame.sensor_frame,
                'detection_frame_id': frame.sensor_frame, 'generation': self.generation,
                'health': 'invalid', 'reason': str(error), 'detections': [], 'tracks': [],
                'pose_source': frame.pose_source, 'pose_health': frame.pose_health,
                'pose_stamp_s': None,
                'world_from_lidar': None, 'input_fingerprint': frame.input_fingerprint(),
                'rgb_asset_uri': frame.rgb_asset_uri, 'rgb_stamp_s': frame.rgb_stamp_s,
                'timing_s': {}, 'covariance_order': ['x','y','z','dx','dy','dz','yaw','vx','vy','vz'],
                'velocity_semantics': 'unavailable'}


def to_snapshot(result):
    """Validated perception result to shared records; does not call/change the VLM."""
    from interfaces.contracts import Header, Box3D, TrackIdentity, PersonTrack, PoseEstimate, PerceptionSnapshot
    header = Header(result['observation_id'], result['stamp_s'], result['frame_id'],
                    result['clock_domain'], result['sequence_id'], result['generation'],
                    timestamp_source=result['timestamp_source'])
    tracks = tuple(PersonTrack(
        TrackIdentity(t['track_id'], t['lifecycle_generation']), t['state'], t['last_observed_stamp_s'],
        Box3D(tuple(t['box'][:3]), tuple(t['box'][3:6]), t['box'][6]),
        None if t['velocity_xyz_mps'] is None else tuple(t['velocity_xyz_mps']),
        tuple(t['covariance']), None, confirmed=t['confirmed'], score=t['box'][7],
        covariance_order=tuple(result['covariance_order']),
    ) for t in result['tracks'])
    pose = None
    if result['world_from_lidar'] is not None:
        pose_header = Header(result['observation_id'], result['pose_stamp_s'], result['frame_id'],
                             result['clock_domain'], result['sequence_id'], result['generation'],
                             timestamp_source=result['timestamp_source'])
        pose = PoseEstimate(pose_header, result['frame_id'], result['detection_frame_id'],
                            tuple(tuple(row) for row in result['world_from_lidar']), None, 'valid')
    return PerceptionSnapshot(header, tracks, result['generation'], pose, (),
                              (result['rgb_asset_uri'],) if result['rgb_asset_uri'] else (), None,
                              result['pose_source'], health=result['health'], reason=result['reason'])
