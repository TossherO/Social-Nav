"""Acquisition-bound LiDAR decoding; no ROS or GPU imports."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perception.sensors.pointcloud import pointcloud_fields


@dataclass
class Scan:
    points: np.ndarray
    times: np.ndarray | None
    stamp_s: float
    duration_s: float


def decode_scan(message, deskew=True, time_field='timestamp', max_scan_s=0.2):
    stamp = message.header.stamp.to_sec()
    if not np.isfinite(stamp) or stamp <= 0:
        raise ValueError('Invalid acquisition timestamp')
    names = ('x', 'y', 'z', 'intensity') + ((time_field,) if deskew else ())
    fields = pointcloud_fields(message, names)
    points = np.stack([fields[name] for name in names[:4]], axis=1)
    valid = np.isfinite(points).all(axis=1)
    times, duration = None, 0.0
    if deskew:
        # This sensor supplies absolute FLOAT64 seconds; never cast to FLOAT32 first.
        raw = fields[time_field].astype(np.float64)
        finite = np.isfinite(raw)
        if not finite.any():
            raise ValueError('Missing finite per-point acquisition times')
        begin, end = float(raw[finite].min()), float(raw[finite].max())
        duration = stamp - begin
        if not (0 < duration <= max_scan_s) or abs(end - stamp) > 0.005:
            raise ValueError('Per-point times must describe an absolute scan ending at the header')
        # Normalize before XYZ filtering: deleted near/far/NaN points retain the same clock.
        times = (raw - begin) / duration
        valid &= finite & (times >= 0) & (times <= 1 + 1e-5)
        times = np.clip(times[valid], 0, 1).astype(np.float32)
    points = points[valid].astype(np.float32)
    if len(points) < 30:
        raise ValueError('Too few finite LiDAR points')
    return Scan(points, times, float(stamp), duration)
