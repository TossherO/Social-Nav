"""CPU box geometry in metres/radians; no Torch, MMCV or ROS dependency."""
from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError('Expected a finite rigid parent_from_child 4x4 transform')
    return matrix.copy()


def validate_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        if boxes.shape not in ((0,), (0, 8)):
            raise ValueError('Empty detections must have shape (0,8)')
        return np.empty((0, 8), dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 8 or not np.all(np.isfinite(boxes))
            or np.any(boxes[:, 3:6] <= 0) or np.any(boxes[:, 7] < 0)
            or np.any(boxes[:, 7] > 1)):
        raise ValueError('Boxes must be finite [x,y,z,dx,dy,dz,yaw,score] with positive sizes')
    boxes = boxes.copy()
    boxes[:, 6] = (boxes[:, 6] + np.pi) % (2*np.pi) - np.pi
    return boxes


def corners_xy(box):
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[c, -s], [s, c]])
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * box[3:5] / 2
    return local @ rotation.T + box[:2]


def corners_xyz(box):
    xy = corners_xy(box)
    return np.concatenate([np.c_[xy, np.full(4, box[2] + sign*box[5]/2)]
                           for sign in (-1, 1)])


def transform_boxes(boxes, parent_from_sensor):
    """Enclose transformed corners in an upright box, including roll/pitch.

    A yaw-only detector box cannot represent a full SE(3) rotated box exactly.
    This conservative envelope is explicit; heading is not human gaze.
    """
    boxes = validate_boxes(boxes)
    transform = validate_transform(parent_from_sensor)
    result = boxes.copy()
    for index, box in enumerate(boxes):
        points = corners_xyz(box) @ transform[:3, :3].T + transform[:3, 3]
        direction = transform[:3, :3] @ np.array([np.cos(box[6]), np.sin(box[6]), 0])
        if np.linalg.norm(direction[:2]) < 1e-6:
            raise ValueError('Box longitudinal direction is vertical; upright heading undefined')
        yaw = np.arctan2(direction[1], direction[0])
        c, s = np.cos(yaw), np.sin(yaw)
        basis = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        local = points @ basis
        low, high = local.min(axis=0), local.max(axis=0)
        result[index, :3] = ((low + high) / 2) @ basis.T
        result[index, 3:6] = high - low
        result[index, 6] = wrap_angle(yaw)
    return result


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    points = np.asarray(points)
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], -1)))) / 2


def intersection_polygon(subject, clip):
    points = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        old, points = points, []
        if not old:
            break
        def side(p):
            delta, edge = p-a, b-a
            return edge[0]*delta[1] - edge[1]*delta[0]
        previous = old[-1]
        for current in old:
            before, after = side(previous), side(current)
            if (before >= -1e-9) != (after >= -1e-9):
                denominator = before - after
                if abs(denominator) > 1e-12:
                    points.append(previous + before/denominator*(current-previous))
            if after >= -1e-9:
                points.append(current)
            previous = current
    return points


def similarity(a, b, metric='giou'):
    pa, pb = corners_xy(a), corners_xy(b)
    overlap_height = max(0.0, min(a[2]+a[5]/2, b[2]+b[5]/2)
                         - max(a[2]-a[5]/2, b[2]-b[5]/2))
    intersection = polygon_area(intersection_polygon(pa, pb)) * overlap_height
    union = np.prod(a[3:6]) + np.prod(b[3:6]) - intersection
    iou = intersection / max(union, 1e-12)
    if metric == 'iou':
        return float(iou)
    if metric == 'giou':
        all_xy = np.r_[pa, pb]
        from scipy.spatial import ConvexHull
        hull = ConvexHull(all_xy)
        height = max(a[2]+a[5]/2, b[2]+b[5]/2) - min(a[2]-a[5]/2, b[2]-b[5]/2)
        enclosure = hull.volume * height
        return float(iou - (enclosure-union)/max(enclosure, 1e-12))
    if metric == 'diou':
        corners = np.r_[corners_xyz(a), corners_xyz(b)]
        diagonal_squared = np.sum((corners.max(axis=0)-corners.min(axis=0))**2)
        return float(iou - np.sum((a[:3]-b[:3])**2)/max(diagonal_squared, 1e-12))
    raise ValueError(f'Unsupported association metric: {metric}')
