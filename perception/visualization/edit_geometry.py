"""Map-frame polygon and bounded-height selections for manual editing."""
import numpy as np
from matplotlib.path import Path


def validate_polygon(vertices):
    points = np.asarray(vertices, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
            not np.isfinite(points).all() or
            abs(np.sum(points[:, 0]*np.roll(points[:, 1], -1)-
                       points[:, 1]*np.roll(points[:, 0], -1))) < 1e-9):
        raise ValueError('需要至少三个有限坐标，且选区面积不能为零')
    return points


def polygon_mask(points, vertices):
    polygon = validate_polygon(vertices)
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('选区输入必须是有限的二维坐标')
    closed = np.vstack([polygon, polygon[0]])
    return Path(closed, closed=True).contains_points(xy, radius=1e-9)


def grid_polygon_mask(shape, origin, resolution, vertices):
    polygon = validate_polygon(vertices)
    origin = np.asarray(origin, dtype=float)
    low = np.maximum(0, np.floor((polygon.min(axis=0)-origin)/resolution).astype(int))
    high = np.minimum(np.asarray(shape[::-1]), np.ceil((polygon.max(axis=0)-origin)/resolution).astype(int))
    result = np.zeros(shape, dtype=bool)
    if (high <= low).any():
        return result
    y, x = np.mgrid[low[1]:high[1], low[0]:high[0]]
    centers = np.c_[x.ravel()+.5, y.ravel()+.5]*resolution+origin
    result[y.ravel(), x.ravel()] = polygon_mask(centers, polygon)
    return result
