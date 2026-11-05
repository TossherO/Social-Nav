"""Offline final-pose fusion of recorded, moving-origin deskewed scans."""
import hashlib

import numpy as np

from .layers import GridSnapshot, project_evidence, STATIC_FILTER_MODES
from .context import encoded
from .occupancy import OccupancyTree
from .scans import read_scans
from .dynamic import people_boxes, person_points, obstacle_band, floor_at, StaticEvidence
from .maps import endpoint_layers
from perception.detection_tracking.inference.geometry import transform_boxes


def fuse_scans(candidate, path, timeline, config, fingerprint, *, people=None, filter_mode='none', candidate_points=None):
    config = dict(config, free_probability=config['static'].get('free_probability',config['free_probability']))
    if filter_mode not in STATIC_FILTER_MODES:
        raise ValueError('Unknown static dynamic-filter mode')
    geometric = filter_mode == 'geometric'
    if geometric and candidate_points is None:
        raise ValueError('Static filtering requires final-map endpoints')
    meta = candidate.metadata
    origin = np.asarray(meta['origin_xy'])
    shape = np.asarray(meta['shape'])
    floor, support = candidate.layers['ground_z'], candidate.layers['ground_support']
    sensors = timeline.sensor_positions
    finite_floor = floor[np.isfinite(floor)]
    low_z = float(finite_floor.min()-.5) if len(finite_floor) else float(sensors[:, 2].min()-3)
    high_z = float(finite_floor.max()+config['max_height_m']+.5) if len(finite_floor) else float(sensors[:, 2].max()+3)
    bounds = [*origin, low_z, *(origin+shape[::-1]*config['resolution']), high_z]
    settings = dict(config, voxel_ttl_s=max(config['voxel_ttl_s'], float(timeline.times[-1]-timeline.times[0]+1)))
    # Historical export preserves evidence across the whole survey. Online
    # obstacle expiration must not erase walls merely visited early in the bag.
    settings['obstacle_ttl_s'] = settings['voxel_ttl_s']
    settings['hit_invalidates_free'] = False
    settings['max_rays']=config['static'].get('max_rays',config['max_rays'])
    settings['max_voxels'] = config['static']['max_voxels']
    count, unbound, rays, last_stamp = 0, 0, 0, None
    review = StaticEvidence(config) if filter_mode != 'none' and people is not None else None
    filter_counts = dict(people_valid_scans=0, people_unavailable_scans=0, masked_scans=0, masked_points=0,
                         extrapolated_ground_mask_points=0)
    tree = OccupancyTree(config['resolution'], backend='cpu')
    try:
        if geometric:
            tree.enable_static_review(config['dynamic']['static_observation_interval_s'],
                                      config['static']['ray_end_margin_m'])
        for record, data in read_scans(path):
            stamp = record['stamp_ns']/1e9
            pose = timeline.at(stamp, exact_ns=record['stamp_ns'])
            if pose is None:
                unbound += 1
                continue
            if ((pose['session_id'], pose['generation'], pose['local_frame']) !=
                    (record['session_id'], record['generation'], record['frame_id']) or
                    not np.allclose(pose['local_from_lidar'], data['local_from_lidar'], atol=1e-8)):
                raise ValueError('Recorded scan and localization source do not agree')
            local, final = np.asarray(pose['local_from_lidar']), np.asarray(pose['world_from_lidar'])
            correction = final @ np.linalg.inv(local)
            points = data['points'] @ final[:3, :3].T+final[:3, 3]
            origins = data.get('origins')
            if origins is not None:
                origins = origins @ correction[:3, :3].T+correction[:3, 3]
            model = record['ray_origin_model']
            if model == 'pose_interpolated_origins' and origins is None:
                raise ValueError('Moving-origin evidence was lost from the scan journal')
            use_rays = bool(record['rays_used'])
            if use_rays and model not in ('pose_interpolated_origins', 'bounded_scan_end_approximation'):
                raise ValueError('Unsupported scan ray origin model')
            masking = {}
            if review is not None:
                row = people.at(stamp)
                boxes, status = people_boxes(row, stamp, (pose['session_id'], pose['generation']),
                    pose['local_frame'], config['dynamic'], clock_domain=people.manifest['clock_domain'])
                boxes = transform_boxes(boxes, correction)
                mask = person_points(points, boxes, floor, origin, config,
                    ground_extrapolation=config['static']['mask_ground_extrapolation_m'] if geometric else 0.)
                review.add(points, mask, stamp, floor, origin)
                filter_counts['people_valid_scans'] += int(status['status'] == 'valid')
                filter_counts['people_unavailable_scans'] += int(status['status'] != 'valid')
                filter_counts['masked_scans'] += int(mask.any())
                filter_counts['masked_points'] += int(mask.sum())
                if geometric:
                    filter_counts['extrapolated_ground_mask_points'] += int((mask & ~np.isfinite(
                        floor_at(points, floor, origin, config['resolution']))).sum())
                masking = dict(transient_mask=mask, transient_ttl=config['dynamic']['transient_ttl_s'], omit_transient=True)
            tree.update(points, final[:3, 3], bounds, stamp, settings, rays=use_rays,
                        ray_origins=origins, read=False, **masking)
            count += 1; rays += int(use_rays); last_stamp = stamp
        if not count:
            raise ValueError('No scan could be bound to a final-map pose')
        voxels = tree.snapshot(last_stamp)
        if geometric:
            voxel_votes, endpoint_votes = tree.static_votes(voxels[:, :3]), tree.static_votes(candidate_points)
    finally:
        tree.close()
    candidates = candidate.layers
    filter_metadata = None
    retained = np.ones(len(voxels), bool)
    keep = np.ones(len(candidate_points), bool) if candidate_points is not None else None
    if review is not None:
        retained = review.keep(voxels[:, :3])
        rejected = (voxels[:, 3] >= config['occupied_probability']) & ~retained
        obstacle = obstacle_band(candidate_points, floor, origin, config)
        keep = ~obstacle | review.keep(candidate_points)
        filter_metadata = dict(mode=filter_mode, people=people.reference, counts=filter_counts,
            static_support_policy='fresh_nonperson_observations_after_last_person',
            mask_policy='bounded_xy_component_above_supported_ground',
            support=review.summary(), settings=config['dynamic'], rejected_occupied_voxels=int(rejected.sum()),
            rejected_endpoints=int((~keep).sum()))
        retained = ~rejected
    if geometric:
        def contradicted(points, votes):
            hits, free = votes.T
            return (obstacle_band(points, floor, origin, config) &
                    (free >= config['static']['free_min_observations']) &
                    (free >= np.maximum(hits, 1)*config['static']['free_to_hit_ratio']))
        rejected_voxels = contradicted(voxels[:, :3], voxel_votes)
        rejected_endpoints = contradicted(candidate_points, endpoint_votes)
        filter_metadata = filter_metadata or dict(mode=filter_mode, people=None, counts=filter_counts)
        filter_metadata.update(geometric_policy='independent_visible_ray_prefix_free_to_hit_votes',
            geometric_settings={k:config['static'][k] for k in ('free_min_observations','free_to_hit_ratio',
                'ray_end_margin_m','mask_ground_extrapolation_m')},
            geometric_rejected_occupied_voxels=int((rejected_voxels & retained &
                                                   (voxels[:, 3] >= config['occupied_probability'])).sum()),
            geometric_rejected_endpoints=int((rejected_endpoints & keep).sum()),
            mask_policy='bounded_xy_component_with_constrained_local_ground_extrapolation')
        filter_metadata['geometric_settings']['observation_interval_s'] = config['dynamic']['static_observation_interval_s']
        retained &= ~(rejected_voxels & (voxels[:, 3] >= config['occupied_probability']))
        keep &= ~rejected_endpoints
    if filter_metadata is not None:
        voxels = voxels[retained]
        candidates = endpoint_layers(candidate_points[keep], floor, origin, config)
        filter_metadata.update(candidate_cells_before=int((candidate.layers['occupancy'] == 100).sum()),
                               candidate_cells_after=int((candidates['occupancy'] == 100).sum()))
    layers = project_evidence(voxels, floor, support, origin, config, last_stamp, candidates=candidates)
    layers['last_observed'][~np.isfinite(layers['last_observed'])] = np.nan
    layers['endpoint_count'] = candidates['endpoint_count']
    from .changes import conditional_obstacles
    layers['conditional_obstacle'] = conditional_obstacles(layers, config)
    binding = [meta['snapshot_id'], fingerprint]
    if filter_metadata is not None:
        binding.append(filter_metadata)
    signature = hashlib.sha256(encoded(binding).encode()).hexdigest()[:20]
    metadata = dict(meta, snapshot_id=meta['map_id']+':scan_fusion:'+signature, stamp_s=last_stamp,
                    geometry_status='STATIC_SCAN_OBSERVED' if rays else 'STATIC_SCAN_HITS_ONLY',
                    source_evidence='deskewed_scans_and_final_map_endpoints', rays_used=bool(rays),
                    free_space_supported=bool(rays), per_cell_observation_time_available=True,
                    voxel_count=len(voxels), evidence_scans=count, unbound_scans=unbound, ray_scans=rays,
                    final_endpoint_candidates_retained=filter_mode == 'none', dynamic_residual_cleanup_verified=False,
                    occupancy_policy='revisable_historical_prior', conditional_policy=config['static'],
                    static_sensor_model=dict(hit_invalidates_free=False,retention='entire_survey'),
                    conditional_obstacle_cells=int(layers['conditional_obstacle'].sum()))
    if filter_metadata is not None:
        metadata['dynamic_filter'] = filter_metadata
    for array in layers.values():
        array.flags.writeable = False
    return GridSnapshot(metadata, layers)
