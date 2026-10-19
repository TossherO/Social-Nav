"""Testable ROS message decoding/buffers; no ROS imports or raw odometry fallback."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from .geometry import validate_transform
from perception.sensors.pointcloud import pointcloud_fields


def decode_pointcloud(message, intensity_field='intensity'):
    selected = ('x', 'y', 'z', intensity_field)
    fields = pointcloud_fields(message, selected)
    return np.stack([fields[name] for name in selected], axis=1).astype(np.float32)


@dataclass
class LocalizedPose:
    stamp_s: float
    parent_frame: str
    child_frame: str
    parent_from_child: np.ndarray | None
    health: str = 'valid'
    source: str = 'localization'
    session_id: str = ''
    generation: int = 0


class PoseBuffer:
    def __init__(self, size=256):
        self.rows = []
        self.size = size
        self.lock = threading.Lock()
        self.ready = threading.Condition(self.lock)
        self.identity = None
        self.retired_sessions = []

    def add(self, pose):
        if pose.health not in ('valid', 'degraded', 'invalid') or pose.generation < 0:
            raise ValueError('Invalid localization health/lifecycle')
        pose.parent_from_child = validate_transform(pose.parent_from_child) if pose.health == 'valid' else None
        if not np.isfinite(pose.stamp_s) or not pose.parent_frame or not pose.child_frame:
            raise ValueError('Invalid localized pose metadata')
        with self.ready:
            identity = (pose.session_id, pose.generation)
            if self.identity is not None and identity != self.identity:
                old_session, old_generation = self.identity
                if (pose.session_id in self.retired_sessions or
                        pose.session_id == old_session and pose.generation < old_generation):
                    return
                if pose.session_id != old_session:
                    self.retired_sessions = (self.retired_sessions + [old_session])[-16:]
                self.rows.clear()
            self.identity = identity
            self.rows = [p for p in self.rows if p.stamp_s != pose.stamp_s]+[pose]
            self.rows.sort(key=lambda p: p.stamp_s)
            self.rows = self.rows[-self.size:]
            self.ready.notify_all()

    def clear(self):
        with self.ready:
            self.rows.clear()
            self.ready.notify_all()

    def resolve(self, stamp, max_age, parent_frame, sensor_frame, child_frame, child_from_lidar=None):
        _, matrix, age = self.resolve_record(stamp, max_age, parent_frame, sensor_frame, child_frame,
                                            child_from_lidar)
        return matrix, age

    def resolve_record(self, stamp, max_age, parent_frame, sensor_frame, child_frame,
                       child_from_lidar=None, wait_s=0):
        # Nearest PAST pose only; no future interpolation/smoothing in offline comparisons.
        deadline = time.monotonic() + wait_s
        with self.ready:
            while True:
                rows = [p for p in self.rows if p.parent_frame == parent_frame
                        and p.child_frame == child_frame and 0 <= stamp-p.stamp_s <= max_age]
                if rows:
                    pose = rows[-1]
                    break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    return None, None, None
                self.ready.wait(remaining)
        age = stamp-pose.stamp_s
        # A failed current estimate must never fall back to an older valid pose.
        if pose.health != 'valid':
            return pose, None, age
        if pose.child_frame == sensor_frame:
            extrinsic = np.eye(4) if child_from_lidar is None else validate_transform(child_from_lidar)
            if not np.allclose(extrinsic, np.eye(4)):
                raise ValueError('Nonidentity extrinsic supplied for identical LiDAR frame')
        else:
            if child_from_lidar is None:
                return pose, None, age
            extrinsic = validate_transform(child_from_lidar)
        return pose, pose.parent_from_child @ extrinsic, age


class LatestSlot:
    """One pending frame; a busy worker replaces obsolete pending input, never the active input."""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.dropped = 0
        self.closed = False

    def put(self, value):
        with self.condition:
            if self.closed:
                return
            if self.value is not None:
                self.dropped += 1
            self.value = value
            self.condition.notify()

    def take(self, timeout=0.1):
        with self.condition:
            self.condition.wait_for(lambda: self.value is not None or self.closed, timeout)
            value, self.value = self.value, None
            return value

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
