"""Final-map endpoint projection; missing rays never become free space."""
import hashlib
import json
import numpy as np

from .geometry import ground_plane, nearest_support, distance_to_zero
from .layers import GridSnapshot


def static_geometry(points, sensor_positions, config, *, map_id, frame, stamp):
    points = np.asarray(points, dtype=np.float32)
    sensors = np.asarray(sensor_positions, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
            not np.isfinite(points).all() or sensors.ndim != 2 or sensors.shape[1] != 3 or
            not len(sensors) or not np.isfinite(sensors).all()):
        raise ValueError('Finite map endpoints and same-map sensor positions are required')
    settings = config['static']
    if (any(type(settings[key]) is not int or settings[key] <= 0
            for key in ('max_cells', 'max_points', 'max_ground_poses')) or
            not np.isfinite(settings['ground_pose_spacing_m']) or settings['ground_pose_spacing_m'] <= 0):
        raise ValueError('Static-map counts must be positive integers and pose spacing finite/positive')
    resolution = config['resolution']
    origin = np.floor(points[:, :2].min(axis=0)/resolution)*resolution
    size = np.floor((points[:, :2].max(axis=0)-origin)/resolution).astype(int)+1
    if int(size[0])*int(size[1]) > settings['max_cells'] or len(points) > settings['max_points']:
        raise ValueError('Static map exceeds its budget; select an explicit map region or resolution')
    shape = (int(size[1]), int(size[0]))
    cells = np.floor((points[:, :2]-origin)/resolution).astype(int)
    floor = np.full(shape, np.nan, np.float32)
    seeds = np.zeros(shape, bool)
    selected = []
    for sensor in sensors:
        if not selected or min(np.linalg.norm(sensor-s) for s in selected) >= settings['ground_pose_spacing_m']:
            selected.append(sensor)
    if len(selected) > settings['max_ground_poses']:
        raise ValueError('Ground-fit pose budget exceeded; split the map explicitly')
    for sensor in selected:
        nearby = np.linalg.norm(points[:, :2]-sensor[:2], axis=1) <= config['ground_fit_radius_m']
        local = points[nearby]
        plane = ground_plane(local, sensor, config)
        if plane is None:
            continue
        ground = np.abs(local[:, 2]-np.c_[local[:, :2], np.ones(len(local))] @ plane) <= config['ground_tolerance_m']
        samples = local[ground]
        ij = np.floor((samples[:, :2]-origin)/resolution).astype(int)
        x, y = ij.T
        floor[y, x], seeds[y, x] = samples[:, 2], True
    if seeds.any():
        distance, nearest = nearest_support(seeds)
        support = distance*resolution <= config['ground_support_radius_m']
        floor = floor[nearest].copy()
        floor[~support] = np.nan
    else:
        support = seeds
    x, y = cells.T
    heights = points[:, 2]-floor[y, x]
    hit = np.isfinite(heights)
    finite_height = heights[hit]
    hit[hit] = ((finite_height >= config['min_obstacle_height_m']) &
                (finite_height <= config['max_height_m']))
    occupancy = np.full(shape, -1, np.int8)
    occupancy[y[hit], x[hit]] = 100
    bottom, top = np.full(shape, np.inf), np.full(shape, -np.inf)
    np.minimum.at(bottom, (y[hit], x[hit]), heights[hit])
    np.maximum.at(top, (y[hit], x[hit]), heights[hit])
    bottom[~np.isfinite(bottom)], top[~np.isfinite(top)] = np.nan, np.nan
    observed = np.zeros(shape, bool)
    observed[y, x] = True
    counts = np.zeros(shape, np.int32)
    np.add.at(counts, (y, x), 1)
    clearance = (distance_to_zero(occupancy != 100)*resolution if hit.any() else np.full(shape, np.nan))
    clearance[occupancy == -1] = np.nan
    # Source point times are absent: never stamp every cell with export time.
    layers = dict(occupancy=occupancy, ground_z=floor, ground_support=support,
                  observed=observed, last_observed=np.full(shape, np.nan),
                  coverage=np.zeros(shape, np.float32), obstacle_min_z=bottom.astype(np.float32),
                  obstacle_max_z=top.astype(np.float32), clearance=clearance.astype(np.float32),
                  unknown_distance=np.zeros(shape, np.float32), frontier=np.zeros(shape, bool),
                  endpoint_count=counts)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    metadata = dict(snapshot_id=map_id+':endpoints:'+signature, map_id=map_id, frame_id=frame,
                    stamp_s=float(stamp), session_id=map_id, generation=0, resolution=resolution,
                    origin_xy=origin.tolist(), shape=list(shape), association={}, rays_used=False,
                    geometry_status='STATIC_ENDPOINT_CANDIDATES', source_evidence='final_map_endpoints_only',
                    time_scope='offline_final_prior', per_cell_observation_time_available=False,
                    world_from_lidar=None, point_count=len(points), ground_fit_poses=len(selected),
                    free_space_supported=False, navigation_reachability_verified=False)
    for value in layers.values():
        value.flags.writeable = False
    return GridSnapshot(metadata, layers)
