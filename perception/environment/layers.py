"""Bounded rolling occupancy evidence and immutable observation snapshots."""
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import cv2

from perception.detection_tracking.inference.geometry import validate_transform
from .geometry import ground_plane, shift_array, nearest_support, distance_to_zero
from .occupancy import OccupancyTree


def load_config(path=None):
    import yaml
    source = Path(path) if path else Path(__file__).with_name('configs')/'environment.yaml'
    values = yaml.safe_load(source.read_text())
    positive = ['resolution','window_m','max_height_m','voxel_ttl_s','max_range_m','max_rays',
                'max_voxels','update_hz','ground_fit_radius_m','ground_min_points',
                'ground_support_radius_m','pose_wait_s','snapshot_cache_size']
    positive += ['max_origin_interpolation_gap_s','association_max_age_s','association_confirm_matches']
    if any(not np.isfinite(values[k]) or values[k] <= 0 for k in positive):
        raise ValueError('Environment sizes and budgets must be finite and positive')
    for key in ('max_rays','max_voxels','ground_min_points','snapshot_cache_size','association_confirm_matches'):
        if not isinstance(values[key],int) or isinstance(values[key],bool):
            raise ValueError('Environment count budgets must be integers: '+key)
    if not 0 < values['free_vertical_coverage'] <= 1 or not 0 < values['free_probability'] < .5 < values['occupied_probability'] < 1:
        raise ValueError('Invalid occupancy/coverage thresholds')
    if values['max_height_m'] <= values['min_obstacle_height_m'] or values['window_m']/values['resolution'] > 600:
        raise ValueError('Invalid collision height or grid budget')
    if not isinstance(values['output_topic'],str) or not values['output_topic'].startswith('/'):
        raise ValueError('Environment output_topic must be an explicit absolute ROS namespace')
    return values


@dataclass(frozen=True)
class GridSnapshot:
    metadata: dict
    layers: dict

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Refusing to overwrite an environment snapshot')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.saving.npz')
        try:
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(self.metadata, allow_nan=False)), **self.layers)
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data['metadata'].item())
            layers = {key:data[key].copy() for key in data.files if key != 'metadata'}
        shape = tuple(metadata['shape'])
        if not layers or any(array.shape != shape for array in layers.values()):
            raise ValueError('Map layer dimensions disagree with metadata')
        for array in layers.values():
            array.flags.writeable = False
        return cls(metadata, layers)


def project_evidence(voxels, floor, support, origin, config, stamp, *, endpoints=None, candidates=None):
    """Shared local/static projection: free evidence must cover the collision column."""
    height, width = floor.shape
    resolution = config['resolution']
    occupancy = np.full(floor.shape, -1, np.int8)
    observed = np.zeros(floor.shape, bool)
    seen = np.full(floor.shape, -np.inf, np.float64)
    top, bottom = np.full(floor.shape, -np.inf), np.full(floor.shape, np.inf)
    bins = int(np.ceil((config['max_height_m']-config['min_obstacle_height_m'])/resolution))
    free_columns = np.zeros((height*width, bins), bool)
    # The floor-relative band rarely aligns with the world voxel lattice.
    # Count the voxel centres actually inside each column's band; a fixed ceil
    # can demand a nonexistent twelfth sample from a band containing eleven.
    required = np.ones(floor.shape, np.int32)
    finite = np.isfinite(floor)
    low = np.ceil((floor[finite]+config['min_obstacle_height_m'])/resolution-.5-1e-5)
    high = np.floor((floor[finite]+config['max_height_m'])/resolution-.5+1e-5)
    required[finite] = np.maximum(1, high-low+1).astype(np.int32)
    if len(voxels):
        cells = np.floor((voxels[:, :2]-origin)/resolution).astype(int)
        keep = (cells >= 0).all(axis=1) & (cells[:, 0] < width) & (cells[:, 1] < height)
        values, cells = voxels[keep], cells[keep]
        x, y = cells.T
        observed[y, x] = True
        # NumPy promotes a Python scalar to float32 when subtracting from a
        # float32 age array. Promote ages first, before restoring epoch seconds.
        np.maximum.at(seen, (y, x), stamp-values[:, 4].astype(np.float64))
        heights = values[:, 2]-floor[y, x]
        band = np.isfinite(heights)
        valid = heights[band]
        band[band] = (valid >= config['min_obstacle_height_m']) & (valid <= config['max_height_m'])
        hit = band & (values[:, 3] >= config['occupied_probability'])
        occupancy[y[hit], x[hit]] = 100
        np.maximum.at(top, (y[hit], x[hit]), heights[hit])
        np.minimum.at(bottom, (y[hit], x[hit]), heights[hit])
        free = band & (values[:, 3] <= config['free_probability'])
        zbin = np.floor((heights[free]-config['min_obstacle_height_m'])/resolution).astype(int)
        free_columns[y[free]*width+x[free], np.clip(zbin, 0, bins-1)] = True
    coverage = np.minimum(1,free_columns.sum(axis=1).reshape(floor.shape)/required).astype(np.float32)
    occupancy[(coverage >= config['free_vertical_coverage']) & support & (occupancy != 100)] = 0
    if endpoints is not None and len(endpoints):
        cells = np.floor((endpoints[:, :2]-origin)/resolution).astype(int)
        keep = (cells >= 0).all(axis=1) & (cells[:, 0] < width) & (cells[:, 1] < height)
        x, y = cells[keep].T
        heights = endpoints[keep, 2]-floor[y, x]
        hit = np.isfinite(heights)
        valid = heights[hit]
        hit[hit] = (valid >= config['min_obstacle_height_m']) & (valid <= config['max_height_m'])
        occupancy[y[hit], x[hit]] = 100
        np.maximum.at(top, (y[hit], x[hit]), heights[hit])
        np.minimum.at(bottom, (y[hit], x[hit]), heights[hit])
    top[~np.isfinite(top)], bottom[~np.isfinite(bottom)] = np.nan, np.nan
    if candidates is not None:
        # Retain thin final-map endpoints conservatively; dynamic cleanup is not implied.
        occupancy[candidates['occupancy'] == 100] = 100
        top = np.fmax(top, candidates['obstacle_max_z'])
        bottom = np.fmin(bottom, candidates['obstacle_min_z'])
        observed |= candidates['observed']
    occupied = occupancy == 100
    clearance = distance_to_zero(~occupied)*resolution if occupied.any() else np.full(floor.shape, np.nan)
    clearance[occupancy == -1] = np.nan
    unknown_distance = distance_to_zero(np.pad(occupancy != -1, 1, constant_values=False))[1:-1, 1:-1]*resolution
    neighbors = cv2.dilate((occupancy == -1).astype(np.uint8), np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8))
    frontier = (occupancy == 0) & (neighbors != 0)
    frontier[[0, -1], :] = False; frontier[:, [0, -1]] = False
    return dict(occupancy=occupancy, ground_z=floor.astype(np.float32), ground_support=support,
                observed=observed, last_observed=seen, coverage=coverage,
                obstacle_min_z=bottom.astype(np.float32), obstacle_max_z=top.astype(np.float32),
                clearance=clearance.astype(np.float32), unknown_distance=unknown_distance.astype(np.float32),
                frontier=frontier)


class LocalEnvironment:
    def __init__(self, config=None, tree_factory=OccupancyTree):
        self.config = config or load_config()
        self.tree_factory = tree_factory
        self.tree = None
        self.identity, self.last_stamp, self.last_pose = None, None, None
        self.ground_z, self.ground_seen, self.origin = None, None, None
        self.snapshots = OrderedDict()
        self.revision = 0

    def close(self):
        if self.tree is not None:
            self.tree.close()
            self.tree = None

    def reset(self):
        self.close()
        self.snapshots.clear()
        self.ground_z = self.ground_seen = self.origin = None
        self.last_stamp = self.last_pose = None

    def update(self, points, transform, stamp, identity, *, frame='odom', association=None,
               scan_duration=None, ray_origins=None):
        config = self.config
        if not np.isfinite(stamp):
            raise ValueError('Environment observation time must be finite')
        transform = validate_transform(transform)
        if identity != self.identity:
            self.reset()
            self.identity = identity
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError('Out-of-order environment observation; reset the session explicitly')
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected Nx3 sensor points')
        finite = np.isfinite(points).all(axis=1)
        if ray_origins is not None:
            ray_origins = np.asarray(ray_origins,dtype=np.float32)
            if ray_origins.shape != points.shape or not np.isfinite(ray_origins).all():
                raise ValueError('Invalid per-point ray origins')
            ray_origins = ray_origins[finite]
        points = points[finite]
        world = points @ transform[:3,:3].T + transform[:3,3]
        sensor = transform[:3,3]
        resolution = config['resolution']
        size = int(np.ceil(config['window_m']/resolution))
        origin = np.floor((sensor[:2]-size*resolution/2)/resolution)*resolution
        if self.origin is None:
            self.ground_z = np.full((size,size), np.nan, dtype=np.float32)
            self.ground_seen = np.full((size,size), -np.inf, dtype=np.float64)
        else:
            dx, dy = np.rint((self.origin-origin)/resolution).astype(int)
            self.ground_z = shift_array(self.ground_z,dx,dy,np.nan)
            self.ground_seen = shift_array(self.ground_seen,dx,dy,-np.inf)
        self.origin = origin
        plane = ground_plane(world,sensor,config)
        indices = np.floor((world[:,:2]-origin)/resolution).astype(int)
        inside = (indices >= 0).all(axis=1) & (indices < size).all(axis=1)
        if plane is not None:
            ground = np.abs(world[:,2]-(np.c_[world[:,:2],np.ones(len(world))] @ plane)) <= config['ground_tolerance_m']
            chosen = np.flatnonzero(ground & inside)
            x,y = indices[chosen].T
            self.ground_z[y,x], self.ground_seen[y,x] = world[chosen,2], stamp
        supported = (stamp-self.ground_seen <= config['voxel_ttl_s']) & np.isfinite(self.ground_z)
        if supported.any():
            distances, nearest = nearest_support(supported)
            support = distances*resolution <= config['ground_support_radius_m']
            floor = self.ground_z[nearest].copy()
            floor[~support] = np.nan
        else:
            support, floor = np.zeros((size,size),bool), np.full((size,size),np.nan,np.float32)
        # A single origin is an explicit, bounded approximation. No first-scan or long-gap rays.
        rays = ray_origins is not None
        if not rays and self.last_pose is not None and scan_duration is not None:
            dt = stamp-self.last_stamp
            if 0 < scan_duration <= dt*1.25 and dt <= .5:
                translation = np.linalg.norm(sensor-self.last_pose[:3,3])*scan_duration/dt
                # Endpoints are already deskewed: rotation does not move their ray origin.
                rays = translation <= config['max_scan_translation_m']
        if self.tree is None:
            self.tree = self.tree_factory(resolution)
        bounds = [*origin, sensor[2]-config['ground_max_below_sensor_m']-.5,
                  *(origin+size*resolution), sensor[2]+config['max_height_m']+.5]
        voxels = self.tree.update(world,sensor,bounds,stamp,config,rays=rays,ray_origins=ray_origins)
        layers = project_evidence(voxels, floor, support, origin, config, stamp, endpoints=world)
        self.revision += 1
        metadata = dict(snapshot_id='%s:%s:%d' % (*identity,self.revision), stamp_s=float(stamp),
                        frame_id=frame, session_id=identity[0], generation=identity[1],
                        resolution=resolution, origin_xy=origin.tolist(), shape=[size,size],
                        world_from_lidar=transform.tolist(), association=association or {},
                        rays_used=bool(rays), ray_origin_model=('pose_interpolated_origins' if ray_origins is not None
                            else 'bounded_scan_end_approximation' if rays else 'hits_only'),
                        scan_duration_s=scan_duration, voxel_count=len(voxels),
                        geometry_status='OBSERVED' if supported.any() else 'GROUND_UNAVAILABLE')
        for array in layers.values():
            array.flags.writeable = False
        snapshot = GridSnapshot(metadata,layers)
        self.snapshots[metadata['snapshot_id']] = snapshot
        while len(self.snapshots)>config['snapshot_cache_size']:
            self.snapshots.popitem(last=False)
        self.last_stamp, self.last_pose = stamp,transform
        return snapshot

    def get_snapshot(self, ident):
        return self.snapshots[ident]
