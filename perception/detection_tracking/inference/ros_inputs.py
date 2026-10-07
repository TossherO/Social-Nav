"""Testable ROS message decoding/buffers; no ROS imports or raw odometry fallback."""
from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

from .geometry import validate_transform


def decode_pointcloud(message, intensity_field='intensity'):
    codes = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4', 8:'f8'}
    fields = {}
    for field in message.fields:
        if field.name in fields:
            raise ValueError('Duplicate PointCloud2 field')
        fields[field.name] = field
    selected = ('x', 'y', 'z', intensity_field)
    if len(set(selected)) != 4 or any(name not in fields for name in selected):
        raise ValueError('PointCloud2 needs explicit XYZ/intensity fields')
    byteorder = '>' if message.is_bigendian else '<'
    formats, offsets = [], []
    for name in selected:
        field = fields[name]
        if field.count != 1 or field.datatype not in codes:
            raise ValueError('Unsupported PointCloud2 field type/count')
        dtype = np.dtype(byteorder+codes[field.datatype])
        if field.offset < 0 or field.offset+dtype.itemsize > message.point_step:
            raise ValueError('PointCloud2 field exceeds point_step')
        formats.append(dtype)
        offsets.append(field.offset)
    if (message.width < 1 or message.height < 1 or message.point_step < 1
            or message.row_step < message.width*message.point_step
            or len(message.data) != message.row_step*message.height):
        raise ValueError('Invalid/empty PointCloud2 layout or payload')
    dtype = np.dtype({'names': list(selected), 'formats': formats, 'offsets': offsets,
                     'itemsize': message.point_step})
    rows = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.row_step, message.point_step))
    return np.stack([rows[name].reshape(-1) for name in selected], axis=1).astype(np.float32)


@dataclass
class LocalizedPose:
    stamp_s: float
    parent_frame: str
    child_frame: str
    parent_from_child: np.ndarray


class PoseBuffer:
    def __init__(self, size=256):
        self.rows = []
        self.size = size
        self.lock = threading.Lock()

    def add(self, pose):
        pose.parent_from_child = validate_transform(pose.parent_from_child)
        if not np.isfinite(pose.stamp_s) or not pose.parent_frame or not pose.child_frame:
            raise ValueError('Invalid localized pose metadata')
        with self.lock:
            self.rows = [p for p in self.rows if p.stamp_s != pose.stamp_s]+[pose]
            self.rows.sort(key=lambda p: p.stamp_s)
            self.rows = self.rows[-self.size:]

    def clear(self):
        with self.lock:
            self.rows.clear()

    def resolve(self, stamp, max_age, parent_frame, sensor_frame, child_frame, child_from_lidar=None):
        # Nearest PAST pose only; no future interpolation/smoothing in offline comparisons.
        with self.lock:
            rows = [p for p in self.rows if p.parent_frame == parent_frame
                    and p.child_frame == child_frame and 0 <= stamp-p.stamp_s <= max_age]
        if not rows:
            return None, None
        pose = rows[-1]
        if pose.child_frame == sensor_frame:
            extrinsic = np.eye(4) if child_from_lidar is None else validate_transform(child_from_lidar)
            if not np.allclose(extrinsic, np.eye(4)):
                raise ValueError('Nonidentity extrinsic supplied for identical LiDAR frame')
        else:
            if child_from_lidar is None:
                return None, None
            extrinsic = validate_transform(child_from_lidar)
        return pose.parent_from_child @ extrinsic, stamp-pose.stamp_s


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
