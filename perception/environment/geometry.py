"""Observed ground support; no missing region is filled with a presumed floor."""
import numpy as np
import cv2


def nearest_support(mask):
    """Return an actual observed seed; verify its exact Euclidean support distance."""
    _, labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8),cv2.DIST_L2,5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    seeds = np.argwhere(mask)
    nearest = seeds[labels-1]
    yy,xx = np.indices(mask.shape)
    distance = np.hypot(yy-nearest[:,:,0],xx-nearest[:,:,1])
    return distance,(nearest[:,:,0],nearest[:,:,1])


def distance_to_zero(mask):
    return cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)


def interpolate_origins(offset_ms, stamp, samples, max_gap):
    """Interpolate measured positions from signed first-point-relative offsets.

    Organized LiDAR packets can contain firings earlier than point zero. Keep
    those offsets unchanged; scan-end time anchors the maximum, not zero.
    """
    offsets = np.asarray(offset_ms,dtype=float)/1000
    if not len(offsets) or not np.isfinite(offsets).all():
        return None
    duration = np.ptp(offsets)
    if duration <= 0 or duration > max_gap or len(samples)<2:
        return None
    times = np.asarray([s[0] for s in samples],dtype=float)
    points = np.asarray([s[1] for s in samples],dtype=float)
    required = stamp-offsets.max()+offsets
    if (not np.isfinite(points).all() or times[-1]>stamp+1e-6 or times[0]>required.min()+1e-6 or
            times[-1]<required.max()-1e-6 or np.any(np.diff(times)<=0)):
        return None
    lo = max(0,np.searchsorted(times,required.min(),side='right')-1)
    if np.any(np.diff(times[lo:])>max_gap):
        return None
    return np.column_stack([np.interp(required,times,points[:,axis]) for axis in range(3)]).astype(np.float32)


def ground_plane(points, sensor, config):
    d = points - sensor
    keep = ((np.linalg.norm(d[:,:2], axis=1) < config['ground_fit_radius_m']) &
            (d[:,2] < -config['ground_min_below_sensor_m']) &
            (d[:,2] > -config['ground_max_below_sensor_m']))
    samples = points[keep]
    if len(samples) < config['ground_min_points']:
        return None
    # The fit budget is independent of full scan density.
    samples = samples[::max(1, int(np.ceil(len(samples)/3000)))]
    xy = np.c_[samples[:,:2], np.ones(len(samples))]
    random = np.random.default_rng(0)
    best = None
    for _ in range(40):
        index = random.choice(len(samples), 3, replace=False)
        if abs(np.linalg.det(xy[index])) < 1e-6:
            continue
        plane = np.linalg.solve(xy[index], samples[index,2])
        if np.linalg.norm(plane[:2]) > np.tan(config['ground_max_slope_rad']):
            continue
        mask = np.abs(xy @ plane-samples[:,2]) <= config['ground_tolerance_m']
        if best is None or mask.sum() > best[0]:
            best = (int(mask.sum()), mask)
    if best is None or best[0] < max(config['ground_min_points'], len(samples)*config['ground_min_ratio']):
        return None
    plane = np.linalg.lstsq(xy[best[1]], samples[best[1],2], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > np.tan(config['ground_max_slope_rad']):
        return None
    return plane


def shift_array(values, dx, dy, fill):
    """Move cells between grid origins, without changing the world frame."""
    h, w = values.shape
    result = np.full_like(values, fill)
    x0, x1, y0, y1 = max(0, dx), min(w,w+dx), max(0,dy), min(h,h+dy)
    if x1 > x0 and y1 > y0:
        result[y0:y1,x0:x1] = values[y0-dy:y1-dy,x0-dx:x1-dx]
    return result
