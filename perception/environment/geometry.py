"""Observed ground support; no missing region is filled with a presumed floor."""
from functools import lru_cache
import numpy as np
import cv2


def voxel_height_band(ground, resolution, minimum, maximum):
    """Integer world-Z keys and counts for an inclusive voxel-centre band.

    The 1e-5-cell boundary tolerance matches resident CPU/CUDA projection.
    Promote ground before arithmetic so float32 and float64 use the same lattice.
    """
    ground = np.asarray(ground, dtype=np.float64)
    low = np.zeros(ground.shape, np.int64)
    count = np.zeros(ground.shape, np.int32)
    finite = np.isfinite(ground)
    first = np.ceil((ground[finite]+minimum)/resolution-.5-1e-5).astype(np.int64)
    last = np.floor((ground[finite]+maximum)/resolution-.5+1e-5).astype(np.int64)
    low[finite] = first
    count[finite] = np.maximum(0, last-first+1).astype(np.int32)
    return low, count


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


def ground_cell_statistics(points, origin, resolution, width):
    """Order-independent median/MAD of unique observed points in each XY cell."""
    points=np.unique(np.asarray(points),axis=0)
    if not len(points):return np.empty(0,int),np.empty(0),np.empty(0),np.empty(0,int)
    xy=np.floor((points[:,:2]-origin)/resolution).astype(int)
    keys=xy[:,1]*width+xy[:,0]
    order=np.lexsort((points[:,2],keys));keys=keys[order];z=points[order,2]
    cells,start,count=np.unique(keys,return_index=True,return_counts=True)
    def middle(values):return (values[start+(count-1)//2]+values[start+count//2])/2
    median=middle(z)
    deviations=np.abs(z-np.repeat(median,count))
    deviation_order=np.lexsort((deviations,keys))
    return cells,median,middle(deviations[deviation_order]),count


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
    return fit_ground_plane(samples, config)


@lru_cache(maxsize=64)
def _ground_sample_indices(count):
    """The original forty independent draws, reused only for the same count."""
    random = np.random.default_rng(0)
    indices = np.array([random.choice(count, 3, replace=False) for _ in range(40)])
    indices.flags.writeable = False
    return indices


def fit_ground_plane(samples, config):
    """Fit already range/height-filtered points with the original sample budget."""
    if len(samples) < config['ground_min_points']:
        return None
    # The fit budget is independent of full scan density.
    samples = samples[::max(1, int(np.ceil(len(samples)/3000)))]
    xy = np.c_[samples[:,:2], np.ones(len(samples))]
    indices = _ground_sample_indices(len(samples))
    matrices = xy[indices]
    valid = np.abs(np.linalg.det(matrices)) >= 1e-6
    if not valid.any():
        return None
    planes = np.linalg.solve(matrices[valid], samples[indices[valid],2])
    planes = planes[np.linalg.norm(planes[:,:2],axis=1) <= np.tan(config['ground_max_slope_rad'])]
    if not len(planes):
        return None
    # Batched matrix-vector products keep each hypothesis's point order.
    residual = np.abs((xy[None] @ planes[:,:,None])[...,0]-samples[None,:,2])
    inliers = residual <= config['ground_tolerance_m']
    counts = inliers.sum(axis=1)
    best = int(np.argmax(counts))  # First maximum, matching the original tie rule.
    if counts[best] < max(config['ground_min_points'], len(samples)*config['ground_min_ratio']):
        return None
    mask = inliers[best]
    plane = np.linalg.lstsq(xy[mask], samples[mask,2], rcond=None)[0]
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
