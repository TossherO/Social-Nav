"""Point projections with explicitly assumed, rather than observed, free space."""
import hashlib
import json

import numpy as np

from .geometry import distance_to_zero
from .layers import GridSnapshot
from .maps import static_geometry


def docker_projection(points, sensor_positions, config, *, map_id, frame, stamp, bounds=None,
                      z_min=-10., z_max=10., radius=.2, min_neighbors=4):
    """Project filtered map points as occupied; absence of points is an assumption.

    Radius counts exclude the query point, matching PCL RadiusOutlierRemoval.
    This is a geometric planning prior, not a source of measured free space.
    """
    values=(z_min,z_max,radius)
    if (any(type(x) not in (int,float) or not np.isfinite(x) for x in values) or
            z_min >= z_max or radius <= 0 or type(min_neighbors) is not int or
            not 1 <= min_neighbors <= 1024):
        raise ValueError('Invalid point-projection height or radius filter')
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Point projection needs finite XYZ points')
    selected=points[(points[:,2]>=z_min)&(points[:,2]<=z_max)]
    if len(selected)<=min_neighbors:
        raise ValueError('Not enough points after projection height filter')
    from scipy.spatial import cKDTree
    distances,_=cKDTree(selected).query(selected,k=min_neighbors+1,
        distance_upper_bound=np.nextafter(float(radius),np.inf),workers=1)
    kept=selected[np.isfinite(distances[:,-1])]
    if not len(kept):
        raise ValueError('No points remain after projection radius filter')
    base=static_geometry(kept,sensor_positions,config,map_id=map_id,frame=frame,
                         stamp=stamp,bounds=bounds)
    metadata=base.metadata;shape=tuple(metadata['shape']);resolution=metadata['resolution']
    cells=np.floor((kept[:,:2]-metadata['origin_xy'])/resolution).astype(np.int64)
    inside=((cells>=0)&(cells<np.asarray(shape[::-1]))).all(axis=1)
    xy=cells[inside];local=kept[inside];x,y=xy.T
    occupancy=np.zeros(shape,np.int8);occupancy[y,x]=100
    assumed=occupancy==0
    floor=base.layers['ground_z'];heights=local[:,2]-floor[y,x]
    finite=np.isfinite(heights)
    bottom=np.full(shape,np.inf);top=np.full(shape,-np.inf)
    np.minimum.at(bottom,(y[finite],x[finite]),heights[finite])
    np.maximum.at(top,(y[finite],x[finite]),heights[finite])
    bottom[~np.isfinite(bottom)]=np.nan;top[~np.isfinite(top)]=np.nan
    layers=dict(base.layers,occupancy=occupancy,assumed_free=assumed,
                obstacle_min_z=bottom.astype(np.float32),obstacle_max_z=top.astype(np.float32),
                clearance=(distance_to_zero(~(occupancy==100))*resolution).astype(np.float32))
    # Assumed-free cells have no ray evidence or acquisition time. They must
    # never become measured static support in the independent safety path.
    layers['last_observed']=np.full(shape,np.nan,dtype=np.float64)
    layers['coverage']=np.zeros(shape,np.float32)
    layers['unknown_distance']=np.zeros(shape,np.float32)
    layers['frontier']=np.zeros(shape,bool)
    settings=dict(z_min_m=float(z_min),z_max_m=float(z_max),radius_m=float(radius),
                  min_neighbors=min_neighbors,resolution_m=float(resolution))
    signature=hashlib.sha256(json.dumps([metadata['snapshot_id'],settings],sort_keys=True).encode()).hexdigest()[:16]
    metadata=dict(metadata,snapshot_id=map_id+':docker_projection:'+signature,
                  geometry_status='STATIC_ASSUMED_FREE_PROJECTION',
                  source_evidence='height_and_radius_filtered_map_endpoints',
                  projection_strategy='docker_assume_free',projection_settings=settings,
                  projection_counts=dict(input_points=len(points),height_filtered_points=len(selected),
                                         radius_filtered_points=len(kept),projected_points=int(inside.sum())),
                  free_assumption='no_filtered_endpoint_in_cell',reference_policy='unobserved_cells_assumed_free',
                  free_space_supported=False,
                  per_cell_observation_time_available=False,navigation_reachability_verified=False)
    for value in layers.values():value.flags.writeable=False
    return GridSnapshot(metadata,layers)


def ground_relative_projection(points, sensor_positions, config, *, map_id, frame, stamp,
                               bounds=None, radius=None, min_neighbors=None,
                               height_min=None, height_max=None):
    """Project points in the ground-relative collision band after a 3-D density filter.

    Without a ground reference, keep a surviving endpoint as occupied rather
    than guessing its height. Non-hit cells are assumed free, not observed.
    """
    low = config['min_obstacle_height_m'] if height_min is None else height_min
    high = config['max_height_m'] if height_max is None else height_max
    settings = config['static'].get('ground_relative_projection', {})
    radius = settings.get('radius_m', .2) if radius is None else radius
    min_neighbors = settings.get('min_neighbors', 2) if min_neighbors is None else min_neighbors
    ground_radius = settings.get('ground_support_radius_m', .75)
    if (any(type(x) not in (int, float) or not np.isfinite(x) for x in (low, high, radius, ground_radius))
            or low < 0 or high <= low or radius <= 0 or ground_radius <= 0 or type(min_neighbors) is not int
            or not 1 <= min_neighbors <= 1024):
        raise ValueError('Invalid ground-relative height band or radius filter')
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Point projection needs finite XYZ points')
    if len(points) <= min_neighbors:
        raise ValueError('Not enough points for projection radius filter')
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(points).query(points, k=min_neighbors+1,
        distance_upper_bound=np.nextafter(float(radius), np.inf), workers=1)
    kept = points[np.isfinite(distances[:, -1])]
    if not len(kept):
        raise ValueError('No points remain after projection radius filter')
    geometry_config = dict(config, min_obstacle_height_m=float(low), max_height_m=float(high),
                           ground_support_radius_m=float(ground_radius))
    base = static_geometry(kept, sensor_positions, geometry_config, map_id=map_id,
                           frame=frame, stamp=stamp, bounds=bounds)
    metadata = base.metadata
    shape, resolution = tuple(metadata['shape']), metadata['resolution']
    cells = np.floor((kept[:, :2]-metadata['origin_xy'])/resolution).astype(np.int64)
    inside = ((cells >= 0) & (cells < np.asarray(shape[::-1]))).all(axis=1)
    local, xy = kept[inside], cells[inside]
    x, y = xy.T
    heights = local[:, 2]-base.layers['ground_z'][y, x]
    supported = base.layers['ground_support'][y, x] & np.isfinite(heights)
    hit = ~supported | ((heights >= low) & (heights <= high))
    occupancy = np.zeros(shape, np.int8)
    occupancy[y[hit], x[hit]] = 100
    layers = dict(base.layers, occupancy=occupancy, assumed_free=occupancy == 0,
                  clearance=(distance_to_zero(occupancy != 100)*resolution).astype(np.float32))
    layers['last_observed'] = np.full(shape, np.nan, dtype=np.float64)
    layers['coverage'] = np.zeros(shape, np.float32)
    layers['unknown_distance'] = np.zeros(shape, np.float32)
    layers['frontier'] = np.zeros(shape, bool)
    settings = dict(height_min_m=float(low), height_max_m=float(high), radius_m=float(radius),
                    min_neighbors=min_neighbors, resolution_m=float(resolution),
                    ground_support_radius_m=float(ground_radius),
                    missing_ground_policy='retain_endpoint_as_occupied')
    signature = hashlib.sha256(json.dumps([metadata['snapshot_id'], settings], sort_keys=True).encode()).hexdigest()[:16]
    metadata = dict(metadata, snapshot_id=map_id+':ground_relative_projection:'+signature,
                    geometry_status='STATIC_ASSUMED_FREE_PROJECTION',
                    source_evidence='ground_relative_and_radius_filtered_map_endpoints',
                    projection_strategy='ground_relative_assume_free', projection_settings=settings,
                    projection_counts=dict(input_points=len(points), radius_filtered_points=len(kept),
                                           projected_points=int(inside.sum()),
                                           height_band_points=int((supported & hit).sum()),
                                           missing_ground_points=int((~supported).sum())),
                    free_assumption='no_filtered_endpoint_in_cell',
                    reference_policy='unobserved_cells_assumed_free', free_space_supported=False,
                    per_cell_observation_time_available=False, navigation_reachability_verified=False)
    for value in layers.values():
        value.flags.writeable = False
    return GridSnapshot(metadata, layers)
