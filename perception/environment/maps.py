"""Final-map endpoint projection; missing rays never become free space."""
import hashlib
import json
import numpy as np

from .geometry import ground_plane, nearest_support, distance_to_zero
from .layers import GridSnapshot


def endpoint_layers(points, floor, origin, config):
    """Project a selected final-map subset onto an existing ground/grid geometry."""
    shape = floor.shape
    cells = np.floor((points[:, :2]-origin)/config['resolution']).astype(int)
    inside=(cells>=0).all(axis=1)&(cells[:,0]<shape[1])&(cells[:,1]<shape[0])
    points,cells=points[inside],cells[inside]
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
    return dict(occupancy=occupancy, observed=observed, endpoint_count=counts,
                obstacle_min_z=bottom.astype(np.float32), obstacle_max_z=top.astype(np.float32))


def static_geometry(points, sensor_positions, config, *, map_id, frame, stamp, bounds=None):
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
    region=None
    if bounds is None:
        origin = np.floor(points[:, :2].min(axis=0)/resolution)*resolution
        size = np.floor((points[:, :2].max(axis=0)-origin)/resolution).astype(int)+1
    else:
        bounds=np.asarray(bounds,dtype=float)
        if bounds.shape!=(4,) or not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
            raise ValueError('Map bounds must be finite [xmin,ymin,xmax,ymax] with positive area')
        origin=np.floor(bounds[:2]/resolution)*resolution
        upper=np.ceil(bounds[2:]/resolution)*resolution
        size=np.rint((upper-origin)/resolution).astype(int)
        region=dict(requested_bounds=bounds.tolist(),effective_bounds=[*origin.tolist(),*upper.tolist()],
                    policy='explicit_region_outward_aligned_to_source_grid')
    inside=((points[:,:2]>=origin)&(points[:,:2]<origin+size*resolution)).all(axis=1)
    if not inside.any():raise ValueError('Selected map region contains no source endpoints')
    if int(size[0])*int(size[1]) > settings['max_cells'] or len(points) > settings['max_points']:
        raise ValueError('Static map exceeds its budget; select an explicit map region or resolution')
    shape = (int(size[1]), int(size[0]))
    floor = np.full(shape, np.nan, np.float32)
    seeds = np.zeros(shape, bool)
    ground_points = np.zeros(len(points),bool)
    selected = []
    for sensor in sensors:
        if region is not None:
            gap=np.maximum(0,np.maximum(origin-sensor[:2],sensor[:2]-(origin+size*resolution)))
            if np.linalg.norm(gap)>config['ground_fit_radius_m']:continue
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
        ground_points[np.flatnonzero(nearby)[ground]]=True
    # A point can be selected by several nearby fitting poses. Count it once,
    # and do not let the last point or last pose choose the cell's elevation.
    samples=points[ground_points]
    if len(samples):
        ij=np.floor((samples[:,:2]-origin)/resolution).astype(int)
        inside_samples=(ij>=0).all(axis=1)&(ij<size).all(axis=1)
        ij,samples=ij[inside_samples],samples[inside_samples]
        flat=ij[:,1]*shape[1]+ij[:,0];order=np.lexsort((samples[:,2],flat))
        keys,starts,counts=np.unique(flat[order],return_index=True,return_counts=True)
        z=samples[order,2]
        median=(z[starts+(counts-1)//2].astype(float)+z[starts+counts//2])/2
        floor.ravel()[keys]=median;seeds.ravel()[keys]=True
    if seeds.any():
        distance, nearest = nearest_support(seeds)
        support = distance*resolution <= config['ground_support_radius_m']
        floor = floor[nearest].copy()
        floor[~support] = np.nan
    else:
        support = seeds
    layers = endpoint_layers(points, floor, origin, config)
    occupancy = layers['occupancy']
    clearance = (distance_to_zero(occupancy != 100)*resolution if (occupancy == 100).any() else np.full(shape, np.nan))
    clearance[occupancy == -1] = np.nan
    # Source point times are absent: never stamp every cell with export time.
    layers.update(ground_z=floor, ground_support=support, last_observed=np.full(shape, np.nan),
                  coverage=np.zeros(shape, np.float32), clearance=clearance.astype(np.float32),
                  unknown_distance=np.zeros(shape, np.float32), frontier=np.zeros(shape, bool))
    ground_estimator='per_cell_median_of_unique_ground_points'
    signature = hashlib.sha256(json.dumps([config,region,ground_estimator],sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    metadata = dict(entity_type='map',map_type='static_environment_map',snapshot_id=map_id+':endpoints:'+signature, map_id=map_id, frame_id=frame,
                    stamp_s=float(stamp), session_id=map_id, generation=0, resolution=resolution,
                    origin_xy=origin.tolist(), shape=list(shape), association={}, rays_used=False,
                    geometry_status='STATIC_ENDPOINT_CANDIDATES', source_evidence='final_map_endpoints_only',
                    ground_estimator=ground_estimator,
                    time_scope='offline_final_prior', per_cell_observation_time_available=False,
                    world_from_lidar=None, point_count=int(inside.sum()), ground_fit_poses=len(selected),
                    free_space_supported=False, navigation_reachability_verified=False)
    if region is not None:metadata.update(region=region,source_point_count=len(points))
    for value in layers.values():
        value.flags.writeable = False
    return GridSnapshot(metadata, layers)
