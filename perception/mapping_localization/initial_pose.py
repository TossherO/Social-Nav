"""Manual LiDAR-in-map seeds and their bag/map bindings; no ROS or SLAM."""
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .results import checked_pose, file_fingerprint, map_fingerprint


def quaternion_rpy(quaternion):
    x, y, z, w = quaternion
    r00, r10 = 1-2*(y*y+z*z), 2*(x*y+z*w)
    r20, r21, r22 = 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)
    cy = math.hypot(r00, r10)
    if cy > 1e-8:
        return [math.atan2(r21, r22), math.atan2(-r20, cy), math.atan2(r10, r00)]
    return [math.atan2(-2*(y*z-x*w), 1-2*(x*x+z*z)), math.atan2(-r20, cy), 0.0]


def rotation_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def supported_ground_height(grid, position, *, radius_m=.5, min_cells=5, min_fraction=.25, max_spread_m=.25):
    """Return a local static-ground estimate, or an explicit reason it is unusable."""
    xy = np.asarray(position, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('Ground query requires finite map XY')
    metadata, layers = grid.metadata, grid.layers
    ground = np.asarray(layers['ground_z'])
    support = np.asarray(layers['ground_support'], dtype=bool)
    resolution = float(metadata['resolution'])
    origin = np.asarray(metadata['origin_xy'], dtype=float)
    if (ground.ndim != 2 or support.shape != ground.shape or origin.shape != (2,) or
            not np.isfinite(origin).all() or not np.isfinite(resolution) or resolution <= 0):
        raise ValueError('Invalid static ground grid')
    index = np.floor((xy-origin)/resolution).astype(int)
    ix, iy = map(int,index)
    if not (0 <= iy < ground.shape[0] and 0 <= ix < ground.shape[1]):
        return dict(status='unavailable', reason='outside_static_map')
    if not support[iy,ix] or not np.isfinite(ground[iy,ix]):
        return dict(status='unavailable', reason='selected_cell_has_no_ground_support')
    n = int(math.ceil(radius_m/resolution))
    x0,x1 = max(0,ix-n),min(ground.shape[1],ix+n+1)
    y0,y1 = max(0,iy-n),min(ground.shape[0],iy+n+1)
    rows,cols = np.ogrid[y0:y1,x0:x1]
    xx=origin[0]+(cols+.5)*resolution
    yy=origin[1]+(rows+.5)*resolution
    circle=(xx-xy[0])**2+(yy-xy[1])**2 <= radius_m**2
    values=ground[y0:y1,x0:x1]
    valid=circle & support[y0:y1,x0:x1] & np.isfinite(values)
    count=int(valid.sum());total=int(circle.sum())
    if total == 0 or count < min_cells or count/total < min_fraction:
        return dict(status='unavailable', reason='insufficient_nearby_ground_support',
                    supported_cells=count, neighborhood_cells=total)
    samples=np.asarray(values[valid],dtype=float)
    low,median,high=np.percentile(samples,[5,50,95])
    spread=float(high-low)
    if spread > max_spread_m:
        return dict(status='unavailable', reason='nearby_ground_height_discontinuous',
                    supported_cells=count, neighborhood_cells=total, height_spread_m=spread)
    return dict(status='available',ground_z_m=float(median),supported_cells=count,
                neighborhood_cells=total,height_spread_m=spread,radius_m=radius_m)


def map_ground_initial_height(grid, hint, position, *, min_offset_m=.2, max_offset_m=2.2):
    """Translate a map-bound historical seed by supported local terrain change."""
    if grid.metadata.get('frame_id') != hint['map']['frame_id']:
        return dict(status='unavailable', reason='static_ground_frame_mismatch')
    reference = hint['pose']['position_m']
    if len(reference) != 3 or not np.isfinite(reference).all():
        raise ValueError('Initial height hint must contain finite XYZ')
    anchor=supported_ground_height(grid,reference[:2])
    if anchor['status'] != 'available':
        return dict(status='unavailable',reason='reference_'+anchor['reason'],reference_ground=anchor)
    target=supported_ground_height(grid,position)
    if target['status'] != 'available':
        return dict(status='unavailable',reason='target_'+target['reason'],target_ground=target,reference_ground=anchor)
    offset=float(reference[2]-anchor['ground_z_m'])
    if not min_offset_m <= offset <= max_offset_m:
        return dict(status='unavailable',reason='reference_height_offset_out_of_range',
                    reference_ground=anchor,target_ground=target,reference_offset_m=offset)
    return dict(status='available',method='static_ground_relative_to_map_bound_hint',
                lidar_z_m=float(target['ground_z_m']+offset),reference_lidar_z_m=float(reference[2]),
                reference_xy_m=list(reference[:2]),reference_offset_m=offset,
                reference_ground=anchor,target_ground=target,
                static_snapshot_id=grid.metadata.get('snapshot_id'),
                offset_scope='derived_from_historical_unverified_seed_not_measured_lidar_height')


def save_seed(path, binding, position, quaternion):
    if checked_pose(position, quaternion) is None:
        raise ValueError('Initial pose must be finite with a unit quaternion')
    record = dict(binding, format='manual_localization_start', source='manual_map_selection',
                  registration_verified=False,
                  pose=dict(position_m=list(position), quaternion_xyzw=list(quaternion)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(name, str(path))  # Atomic, and never overwrite an existing annotation.
    finally:
        Path(name).unlink(missing_ok=True)
    return record


def load_seed(path, map_path, bag_path, start):
    """Read a seed only for the same map, bag and requested playback offset."""
    import yaml
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict) or record.get('format') != 'manual_localization_start':
        raise ValueError('Not a manual localization start file')
    try:
        if record['map']['fingerprint'] != map_fingerprint(map_path):
            raise ValueError('Initial pose belongs to a different map')
        if record['bag']['fingerprint'] != file_fingerprint(bag_path):
            raise ValueError('Initial pose belongs to a different bag')
        manifest = yaml.safe_load((Path(map_path) / 'manifest.yaml').read_text())
        if record['map']['frame_id'] != manifest['frame']:
            raise ValueError('Initial pose uses a different map frame')
        if record['observation']['frame_id'] != 'rslidar':
            raise ValueError('Initial pose must refer to the configured rslidar frame')
        if not math.isclose(float(record['observation']['start_offset_s']), start, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Use --start matching the annotated bag offset')
        pose = checked_pose(record['pose']['position_m'], record['pose']['quaternion_xyzw'])
        if pose is None:
            raise ValueError('Invalid initial position/quaternion')
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete initial pose record') from error
    position, quaternion = pose
    return quaternion_rpy(quaternion) + position.tolist()
