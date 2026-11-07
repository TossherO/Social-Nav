"""Bounded rolling occupancy evidence and immutable observation snapshots."""
from collections import OrderedDict
from dataclasses import dataclass, field
import copy
import json
from pathlib import Path

import numpy as np
import cv2

from perception.detection_tracking.inference.geometry import validate_transform
from .geometry import shift_array, nearest_support, distance_to_zero, voxel_height_band
from .occupancy import OccupancyTree
from .dynamic import people_boxes, person_points
from .ground import GroundTracker
from .near_field import NearFieldFreeMemory


STATIC_FILTER_MODES = ('geometric', 'none')


def validate_current_policy(config):
    """Reject retired algorithm requests instead of silently changing behavior."""
    for name in ('terrain', 'navigation_fusion'):
        if config.get(name, {}).get('enabled'):
            raise ValueError('Unsupported environment policy: '+name)


def load_config(path=None):
    import yaml
    source = Path(path) if path else Path(__file__).with_name('configs')/'environment.yaml'
    values = yaml.safe_load(source.read_text())
    validate_current_policy(values)
    if type(values.get('hit_invalidates_free',False)) is not bool:
        raise ValueError('hit_invalidates_free must be boolean')
    near=values.setdefault('near_field',dict(enabled=False))
    if not isinstance(near,dict) or type(near.get('enabled')) is not bool:
        raise ValueError('near_field.enabled must be boolean')
    if near['enabled']:
        NearFieldFreeMemory(values,{k:v for k,v in near.items() if k!='enabled'})
    values.setdefault('obstacle_ttl_s', values['voxel_ttl_s'])
    if not np.isfinite(values['obstacle_ttl_s']) or not 0 < values['obstacle_ttl_s'] <= values['voxel_ttl_s']:
        raise ValueError('Obstacle lifetime must be positive and no longer than evidence retention')
    compute = values.setdefault('compute',{})
    compute.setdefault('backend','cuda')
    compute.setdefault('device',0)
    if compute['backend'] not in ('cpu','cuda') or type(compute['device']) is not int or compute['device']<0:
        raise ValueError('Environment compute requires cpu/cuda and a nonnegative visible device')
    ground=values.setdefault('ground_tracking',dict(enabled=False))
    if type(ground['enabled']) is not bool:raise ValueError('Ground tracking enabled must be boolean')
    if near['enabled'] and not ground['enabled']:
        raise ValueError('Near-field history requires the tracked ground plane')
    if ground['enabled']:
        for key,default in GroundTracker.DEFAULTS.items():ground.setdefault(key,default)
        if any(not np.isfinite(ground[k]) or ground[k]<=0 for k in
               ('min_span_m','max_height_mad_m','height_tolerance_m','search_band_m',
                'support_cell_m','max_cell_vertical_span_m','max_extrapolation_error_m')):
            raise ValueError('Invalid ground tracking distances')
        if any(type(ground[k]) is not int or ground[k]<1 for k in ('min_observations','min_inliers','reacquire_observations')):
            raise ValueError('Invalid ground tracking counts')
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
    dynamic = values.get('dynamic')
    if dynamic is not None:
        dynamic.setdefault('object_radius_limit_m',.55)
        for name in ('max_people_age_s', 'wait_s', 'padding_m', 'max_radius_m', 'seed_distance_m',
                     'static_observation_interval_s','object_radius_limit_m'):
            if not np.isfinite(dynamic[name]) or dynamic[name] <= 0:
                raise ValueError('Dynamic-filter sizes and time budgets must be positive')
        if dynamic['object_radius_limit_m']>.98:
            raise ValueError('Person ownership radius plus padding must fit the 1 m object budget')
        if not np.isfinite(dynamic['transient_ttl_s']) or dynamic['transient_ttl_s'] < 0:
            raise ValueError('Person lifetime must be finite and nonnegative')
        for name in ('min_hits', 'min_cluster_points', 'static_min_observations'):
            if type(dynamic[name]) is not int or not 1 <= dynamic[name] <= 255:
                raise ValueError('Dynamic-filter count budgets must be integers in [1,255]')
        if not 0 < dynamic['min_score'] <= 1 or type(dynamic['enabled']) is not bool:
            raise ValueError('Invalid dynamic-filter confidence or enabled flag')
    static = values['static']
    static.setdefault('free_probability', values['free_probability'])
    if not 0 < static['free_probability'] < .5:
        raise ValueError('Invalid static free probability')
    static.setdefault('max_rays',values['max_rays'])
    if type(static['max_rays']) is not int or static['max_rays']<=0:
        raise ValueError('Static ray budget must be a positive integer')
    static.setdefault('dynamic_filter', 'geometric')
    if static['dynamic_filter'] not in STATIC_FILTER_MODES:
        raise ValueError('Unknown static dynamic-filter mode')
    for key, default in dict(free_min_observations=3, free_to_hit_ratio=2.,
                             ray_end_margin_m=.2, mask_ground_extrapolation_m=.3,
                             conditional_max_area_m2=.8, conditional_max_extent_m=2.).items():
        static.setdefault(key, default)
        if not np.isfinite(static[key]) or static[key] <= 0:
            raise ValueError('Static review thresholds must be finite and positive')
    if type(static['free_min_observations']) is not int or static['free_to_hit_ratio'] < 1:
        raise ValueError('Static review requires an integer observation count and free/hit ratio >= 1')
    return values


@dataclass(frozen=True)
class GridSnapshot:
    """Array container shared by maps and views; metadata identifies the role."""
    metadata: dict
    layers: dict
    _binding_cache: object = field(default=None, compare=False, repr=False)

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
            # NpzFile materializes each array; closing it does not invalidate them.
            layers = {}
            for key in data.files:
                if key == 'metadata':
                    continue
                value = data[key]
                layers[key] = np.frombuffer(value.tobytes(order='C'), value.dtype).reshape(value.shape)
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
    bins = int(np.ceil((config['max_height_m']-config['min_obstacle_height_m'])/resolution))+1
    free_columns = np.zeros((height*width, bins), bool)
    low, required = voxel_height_band(floor, resolution, config['min_obstacle_height_m'], config['max_height_m'])
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
        keys = np.rint(values[:, 2].astype(np.float64)/resolution-.5).astype(np.int64)
        zbin = keys-low[y, x]
        band = np.isfinite(floor[y, x]) & (zbin >= 0) & (zbin < required[y, x])
        hit = band & (values[:, 3] >= config['occupied_probability'])
        occupancy[y[hit], x[hit]] = 100
        np.maximum.at(top, (y[hit], x[hit]), heights[hit])
        np.minimum.at(bottom, (y[hit], x[hit]), heights[hit])
        free = band & (values[:, 3] <= config['free_probability'])
        free_columns[y[free]*width+x[free], zbin[free]] = True
    coverage = (free_columns.sum(axis=1).reshape(floor.shape)/np.maximum(1,required)).astype(np.float32)
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
    return finish_evidence(occupancy, observed, seen, coverage, bottom, top, floor, support, resolution)


def finish_evidence(occupancy, observed, seen, coverage, bottom, top, floor, support, resolution):
    """Common exact CPU distance fields and boundary rules for both native backends."""
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


class LocalEnvironmentMap:
    """Maintain a rolling map of current local evidence in the continuous odom frame."""
    def __init__(self, config=None, tree_factory=OccupancyTree):
        self.config = copy.deepcopy(config or load_config())
        validate_current_policy(self.config)
        compute = self.config.get('compute', {})
        self.tree_factory = (lambda resolution: OccupancyTree(resolution,
            backend=compute.get('backend','cuda'), device=compute.get('device',0))) if tree_factory is OccupancyTree else tree_factory
        self.tree = None
        self.support_observation = None
        self.identity, self.last_stamp, self.last_pose = None, None, None
        self.frame=None
        self.ground_z, self.ground_seen, self.origin = None, None, None
        self.snapshots = OrderedDict()
        self.revision = 0
        self.ground_tracker=GroundTracker(self.config.get('ground_tracking',dict(enabled=False)))
        self.near_field=self._new_near_field()
        if self.near_field and not self.config.get('ground_tracking',{}).get('enabled'):
            raise ValueError('Near-field history requires tracked plane ground')
        if compute.get('backend','cuda') == 'cuda':
            # Fail before announcing the node ready, not in the first live frame.
            self.tree = self.tree_factory(self.config['resolution'])

    def close(self):
        if self.tree is not None:
            self.tree.close()
            self.tree = None

    def _new_near_field(self):
        settings=self.config.get('near_field',{})
        return NearFieldFreeMemory(self.config,{k:v for k,v in settings.items() if k!='enabled'}) if settings.get('enabled') else None

    def reset(self):
        self.close()
        self.support_observation=None
        self.snapshots.clear()
        self.ground_z = self.ground_seen = self.origin = None
        self.last_stamp = self.last_pose = None
        self.identity=None;self.frame=None
        self.ground_tracker=GroundTracker(self.config.get('ground_tracking',dict(enabled=False)))
        self.near_field=self._new_near_field()

    def _plane_ground(self,world,sensor,origin,size,stamp):
        config=self.config;resolution=config['resolution']
        if self.origin is None:
            self.ground_z = np.full((size,size), np.nan, dtype=np.float32)
            self.ground_seen = np.full((size,size), -np.inf, dtype=np.float64)
        else:
            dx, dy = np.rint((self.origin-origin)/resolution).astype(int)
            self.ground_z = shift_array(self.ground_z,dx,dy,np.nan)
            self.ground_seen = shift_array(self.ground_seen,dx,dy,-np.inf)
        plane,status = self.ground_tracker.fit(world,sensor,config)
        if status.get('clear_ground_history'):
            self.ground_z.fill(np.nan);self.ground_seen.fill(-np.inf)
            if self.near_field:self.near_field.reset()
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
        if status.get('history_suspended'):
            # Retain the height for obstacle classification, not free certification.
            # Stored ground values and original observation times remain untouched.
            support.fill(False)
        return floor,support,status

    def update(self, points, transform, stamp, identity, *, frame='odom', association=None,
               scan_duration=None, ray_origins=None, people=None, clock_domain=None, device=None, metrics=None,
               localization_source=None,sensor_frame=None):
        config = self.config
        if not np.isfinite(stamp):
            raise ValueError('Environment observation time must be finite')
        transform = validate_transform(transform)
        if identity != self.identity or frame!=self.frame:
            self.reset()
            self.identity = identity
            self.frame=frame
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
        floor,support,ground_status=self._plane_ground(world,sensor,origin,size,stamp)
        self.origin = origin
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
        masking, person_status = {}, dict(status='disabled')
        owners=np.full(len(world),-1,np.int32);objects=None
        if config.get('dynamic', {}).get('enabled'):
            boxes, person_status = people_boxes(people, stamp, identity, frame, config['dynamic'], clock_domain=clock_domain)
            if (people is not None and person_status['status']=='valid' and abs(people['stamp_s']-stamp)<1e-6
                    and all(key is not None for key in person_status['track_ids'])
                    and 'session_id' in people and 'generation' in people):
                from .person_objects import point_owners,ownership_record
                plane=ground_status.get('plane_world') if ground_status.get('status')=='tracked' else None
                owners=point_owners(world,boxes,config,sensor,plane)
                transient=owners>=0
                if transient.any():
                    objects=ownership_record(world,owners,boxes,person_status['track_ids'],people,stamp,identity,frame)
            else:
                transient = person_points(world, boxes, floor, origin, config)
            person_status.update(masked_points=int(transient.sum()), transient_ttl_s=config['dynamic']['transient_ttl_s'])
            masking = dict(transient_mask=transient, transient_ttl=config['dynamic']['transient_ttl_s'],omit_transient=objects is not None)
        count = self.tree.update(world,sensor,bounds,stamp,config,rays=rays,ray_origins=ray_origins,read=False,**masking)
        if metrics is not None and self.tree.backend=='cuda':
            metrics.transfer('h2d',world.astype(np.float32,copy=False).nbytes+
                (0 if ray_origins is None else ray_origins.nbytes)+
                (0 if not masking else masking['transient_mask'].nbytes),'environment_evidence')
            metrics.transfer('d2h',4,'environment_voxel_count')
        self.revision += 1
        metadata = dict(entity_type='map',map_type='local_environment_map',snapshot_id='%s:%s:%d' % (*identity,self.revision), stamp_s=float(stamp),
                        frame_id=frame, session_id=identity[0], generation=identity[1],
                        resolution=resolution, origin_xy=origin.tolist(), shape=[size,size],
                        world_from_lidar=transform.tolist(), association=association or {},
                        rays_used=bool(rays), ray_origin_model=('pose_interpolated_origins' if ray_origins is not None
                            else 'bounded_scan_end_approximation' if rays else 'hits_only'),
                        scan_duration_s=scan_duration, voxel_count=count,
                        evidence_retention_s=config['voxel_ttl_s'],
                        obstacle_retention_s=config.get('obstacle_ttl_s',config['voxel_ttl_s']),
                        free_vertical_coverage_required=config['free_vertical_coverage'],
                        geometry_status='OBSERVED' if support.any() else 'GROUND_UNAVAILABLE')
        metadata['compute_backend'] = self.config.get('compute',{}).get('backend','cuda')
        if localization_source is not None:metadata['localization_source']=localization_source
        if sensor_frame is not None:metadata['sensor_frame']=sensor_frame
        metadata['ground_tracking'] = ground_status
        metadata['dynamic_filter'] = person_status
        if objects is not None:metadata['person_objects']=objects
        self.support_observation = dict(stamp_s=float(stamp),identity=(*identity,frame),
            endpoints=np.ascontiguousarray(world,np.float64),
            transient_mask=np.ascontiguousarray(masking.get('transient_mask',np.zeros(len(world),bool)),np.uint8),
            person_owner=owners)
        backend=self.config.get('compute',{}).get('backend','cuda')
        metadata['hit_invalidates_free']=config.get('hit_invalidates_free',False)
        metadata['compute_components']=dict(occupancy=backend,projection=backend,distance_fields=backend,
            near_field=backend if self.near_field else 'disabled',ground_fit='cpu',ground_support='cpu',people_mask='cpu')
        remaining=owners<0;view_points=world[remaining]
        view_mask=masking.get('transient_mask');view_mask=None if view_mask is None else view_mask[remaining]
        layers = self.tree.project(floor,support,origin,config,stamp,endpoints=view_points,
            view_metadata=metadata,near_field=self.near_field if backend=='cuda' else None,
            transient_mask=view_mask,device=device,metrics=metrics)
        if self.near_field and backend=='cuda':
            metadata.update(snapshot_id=metadata['snapshot_id']+':near_field:'+self.near_field.binding,
                            near_field_memory=self.tree.view_metadata)
        if device is not None:
            from .device import DeviceGridSnapshot
            if backend!='cuda':raise ValueError('Device snapshots require CUDA')
            snapshot=DeviceGridSnapshot.seal(metadata,layers)
            self.snapshots[metadata['snapshot_id']]=snapshot
            while len(self.snapshots)>config['snapshot_cache_size']:self.snapshots.popitem(last=False)
            self.last_stamp,self.last_pose=stamp,transform
            return snapshot
        snapshot=GridSnapshot(metadata,layers)
        if self.near_field and backend=='cpu':
            snapshot=self.near_field.update(snapshot,view_points,transient_mask=view_mask,
                person_only=getattr(self.tree,'person_only',None))
        if 'person_background_free' in snapshot.layers:
            snapshot.metadata['dynamic_navigation']=dict(policy='exclusive_person_on_certified_historical_free',
                eligible_cells=int(snapshot.layers['person_background_free'].sum()),current_occupancy_preserved=True)
        metadata,layers=snapshot.metadata,snapshot.layers
        for array in layers.values():
            array.flags.writeable = False
        self.snapshots[metadata['snapshot_id']] = snapshot
        while len(self.snapshots)>config['snapshot_cache_size']:
            self.snapshots.popitem(last=False)
        self.last_stamp, self.last_pose = stamp,transform
        return snapshot

    def get_snapshot(self, ident):
        return self.snapshots[ident]
