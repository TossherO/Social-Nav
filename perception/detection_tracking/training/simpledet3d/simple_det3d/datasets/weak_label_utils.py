"""Utilities for conservative Wheelchair weak-label filtering.

The functions in this module are deliberately NumPy-only so the offline data
preparation tool and its unit tests do not require CUDA or a built detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


SUPERVISION_IGNORE = 0
SUPERVISION_CENTER = 1
SUPERVISION_BOX = 2
SUPERVISION_SOURCE = 3


@dataclass(frozen=True)
class PedestrianBoxFilterConfig:
    """Broad physical and point-support limits for pseudo pedestrian boxes."""

    min_horizontal_size: float = 0.15
    max_horizontal_size: float = 1.50
    min_height: float = 0.60
    max_height: float = 2.40
    near_distance: float = 8.0
    far_distance: float = 15.0
    near_min_points: int = 5
    mid_min_points: int = 3
    far_min_points: int = 1
    min_bidirectional_box_score: float = 0.60
    temporal_base_distance: float = 0.25
    max_pedestrian_speed: float = 4.0


def validate_box_geometry(
    box: Sequence[float],
    cfg: PedestrianBoxFilterConfig,
    point_cloud_range: Sequence[float] | None = None,
) -> tuple[bool, str]:
    """Validate ``[x, y, z, dx, dy, dz, yaw]`` pseudo-box geometry."""

    box = np.asarray(box, dtype=np.float64)
    if box.shape != (7,):
        return False, 'invalid_shape'
    if not np.all(np.isfinite(box)):
        return False, 'non_finite'

    dx, dy, dz = box[3:6]
    if dx <= 0 or dy <= 0 or dz <= 0:
        return False, 'non_positive_size'
    if not (
        cfg.min_horizontal_size <= dx <= cfg.max_horizontal_size
        and cfg.min_horizontal_size <= dy <= cfg.max_horizontal_size
        and cfg.min_height <= dz <= cfg.max_height
    ):
        return False, 'implausible_size'

    if point_cloud_range is not None:
        pc_range = np.asarray(point_cloud_range, dtype=np.float64)
        if pc_range.shape != (6,):
            raise ValueError('point_cloud_range must have six elements')
        if np.any(box[:3] < pc_range[:3]) or np.any(box[:3] > pc_range[3:]):
            return False, 'outside_range'

    return True, 'ok'


def count_points_in_oriented_box(points: np.ndarray, box: Sequence[float]) -> int:
    """Count finite XYZ points inside a z-axis oriented 3D box."""

    points = np.asarray(points)
    box = np.asarray(box, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points must have shape (N, >=3)')
    if box.shape != (7,):
        raise ValueError('box must have shape (7,)')

    xyz = points[:, :3]
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) == 0:
        return 0

    rel = xyz - box[None, :3]
    cos_yaw = np.cos(box[6])
    sin_yaw = np.sin(box[6])
    local_x = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
    local_y = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw

    inside = (
        (np.abs(local_x) <= box[3] * 0.5)
        & (np.abs(local_y) <= box[4] * 0.5)
        & (np.abs(rel[:, 2]) <= box[5] * 0.5)
    )
    return int(inside.sum())


def required_point_count(distance: float, cfg: PedestrianBoxFilterConfig) -> int:
    if distance <= cfg.near_distance:
        return cfg.near_min_points
    if distance <= cfg.far_distance:
        return cfg.mid_min_points
    return cfg.far_min_points


def transform_centers(centers: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform an ``(N, 3)`` center array with a homogeneous matrix."""

    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('transform must be a finite 4x4 matrix')
    if len(centers) == 0:
        return centers.copy()
    homogeneous = np.concatenate(
        [centers, np.ones((len(centers), 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ transform.T)[:, :3]


def temporal_support_counts(
    frame_centers: Mapping[int, np.ndarray],
    frame_poses: Mapping[int, np.ndarray],
    frame_rate_hz: float,
    cfg: PedestrianBoxFilterConfig,
    max_frame_gap: int = 2,
) -> dict[tuple[int, int], int]:
    """Count previous/next-frame center matches in a common global frame.

    A count of two means that a candidate has support on both temporal sides.
    Missing poses are expected to be represented by identity matrices only for
    scenes whose platform is known to be stationary.
    """

    if frame_rate_hz <= 0:
        raise ValueError('frame_rate_hz must be positive')

    world_centers = {}
    for frame_idx, centers in frame_centers.items():
        pose = frame_poses.get(frame_idx, np.eye(4, dtype=np.float64))
        world_centers[frame_idx] = transform_centers(centers, pose)

    support = {}
    frame_indices = set(world_centers)
    for frame_idx, centers in world_centers.items():
        for center_idx, center in enumerate(centers):
            matched_sides = 0
            for direction in (-1, 1):
                side_matched = False
                for gap in range(1, max_frame_gap + 1):
                    other_idx = frame_idx + direction * gap
                    if other_idx not in frame_indices:
                        continue
                    dt = gap / frame_rate_hz
                    gate = cfg.temporal_base_distance + cfg.max_pedestrian_speed * dt
                    other = world_centers[other_idx]
                    if len(other) and np.linalg.norm(other[:, :2] - center[None, :2], axis=1).min() <= gate:
                        side_matched = True
                        break
                matched_sides += int(side_matched)
            support[(frame_idx, center_idx)] = matched_sides
    return support


def classify_pseudo_box(
    box: Sequence[float],
    points: np.ndarray,
    temporal_support: int,
    cfg: PedestrianBoxFilterConfig,
    point_cloud_range: Sequence[float] | None = None,
    pseudo_score: float | None = None,
) -> tuple[int, dict]:
    """Classify one raw pseudo box into BOX, CENTER, or IGNORE supervision.

    BOX requires valid geometry, distance-aware point support, image
    confidence, and support on both temporal sides.  Manual BEV centers are a
    complementary annotation set: an object with an existing pseudo 3D box is
    intentionally not clicked again, so clicks must not validate or reject a
    box.  Later Teacher refreshes can promote additional candidates.
    """

    valid, reason = validate_box_geometry(box, cfg, point_cloud_range)
    box = np.asarray(box, dtype=np.float64)
    center_finite = box.shape == (7,) and np.all(np.isfinite(box[:2]))
    diagnostics = {
        'geometry_valid': bool(valid),
        'reason': reason,
        'temporal_support': int(temporal_support),
        'point_count': 0,
        'required_points': 0,
        'pseudo_score': float(pseudo_score) if pseudo_score is not None else float('nan'),
    }
    if not center_finite:
        return SUPERVISION_IGNORE, diagnostics

    if not valid:
        return SUPERVISION_CENTER, diagnostics

    point_count = count_points_in_oriented_box(points, box)
    distance = float(np.linalg.norm(box[:2]))
    min_points = required_point_count(distance, cfg)
    diagnostics['point_count'] = point_count
    diagnostics['required_points'] = min_points
    if point_count < min_points:
        diagnostics['reason'] = 'insufficient_points'
        return SUPERVISION_CENTER, diagnostics

    bidirectional_temporal = temporal_support >= 2
    score_support = (
        pseudo_score is not None
        and np.isfinite(pseudo_score)
        and pseudo_score >= cfg.min_bidirectional_box_score
    )
    if bidirectional_temporal and score_support:
        diagnostics['reason'] = 'box_supported'
        return SUPERVISION_BOX, diagnostics

    diagnostics['reason'] = (
        'low_box_score' if bidirectional_temporal and not score_support
        else 'center_only_support'
    )
    return SUPERVISION_CENTER, diagnostics


def normalize_string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=str)
