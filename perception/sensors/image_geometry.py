"""CPU camera/LiDAR geometry shared by ROS readers and image association."""
import numpy as np


def corners(box):
    local = np.array([[x, y, z] for z in (-.5, .5) for x, y in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]) * box[3:6]
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def sensor_box(box, inverse):
    box = np.asarray(box, dtype=float)
    points = corners(box) @ inverse[:3, :3].T + inverse[:3, 3]
    center = box[:3] @ inverse[:3, :3].T + inverse[:3, 3]
    direction = inverse[:3, :3] @ np.array([np.cos(box[6]), np.sin(box[6]), 0])
    yaw = np.arctan2(direction[1], direction[0])
    c, s = np.cos(yaw), np.sin(yaw)
    axis = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = (points-center) @ axis
    return np.r_[center, np.ptp(local, axis=0), yaw, box[7]].tolist()


def project_box(box, transform, info, size):
    if info is None:
        return None
    import cv2
    optical = corners(np.asarray(box)) @ transform[:3, :3].T + transform[:3, 3]
    if np.any(optical[:, 2] <= .05):
        return None
    if info.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
        return None
    k = np.asarray(info.K).reshape(3, 3)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or (info.width, info.height) != tuple(size):
        return None
    pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), k, np.asarray(info.D, dtype=float))
    pixels = pixels.reshape(-1, 2)
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    if not np.isfinite(pixels).all() or high[0] < 0 or high[1] < 0 or low[0] >= size[0] or low[1] >= size[1]:
        return None
    return [float(max(0, low[0])), float(max(0, low[1])), float(min(size[0], high[0])), float(min(size[1], high[1]))]


def camera_fov(transform, info, size):
    """Display-only horizontal view: principal-row rays projected into LiDAR XY."""
    if info is None or not size or info.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
        return None
    k = np.asarray(info.K, dtype=float).reshape(3, 3)
    distortion = np.asarray(info.D, dtype=float)
    if ((info.width, info.height) != tuple(size) or min(size) < 2 or
            not np.isfinite(k).all() or not np.isfinite(distortion).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0):
        return None
    import cv2
    pixels = np.array([[[0., k[1, 2]]], [[k[0, 2], k[1, 2]]], [[size[0]-1., k[1, 2]]]])
    normalized = cv2.undistortPoints(pixels, k, distortion).reshape(-1, 2)
    rays = (np.c_[normalized, np.ones(3)] @ transform[:3, :3])[:, :2]
    norms = np.linalg.norm(rays, axis=1)
    if not np.isfinite(rays).all() or np.any(norms < 1e-6):
        return None
    rays /= norms[:, None]
    if np.any(rays[[0, 2]] @ rays[1] <= 0):
        return None
    origin = -transform[:3, :3].T @ transform[:3, 3]
    return dict(origin_xy_m=origin[:2].tolist(), boundary_directions_xy=rays[[0, 2]].tolist(),
                forward_xy=rays[1].tolist())

def camera_model(info, size):
    """Serializable calibration bound to the original image size."""
    if info is None or not size:
        return None
    if (info.width, info.height) != tuple(size):
        return None
    return dict(K=list(info.K), D=list(info.D), width=int(info.width), height=int(info.height),
                distortion_model=info.distortion_model)


def project_point(point, transform, info):
    if info is None:
        return None
    import cv2
    optical = np.asarray(point) @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(optical).all() or optical[2] <= .05:
        return None
    pixel, _ = cv2.projectPoints(optical.reshape(1, 3), np.zeros(3), np.zeros(3),
                                 np.asarray(info.K).reshape(3, 3), np.asarray(info.D, dtype=float))
    return pixel.reshape(2).tolist() if np.isfinite(pixel).all() else None

