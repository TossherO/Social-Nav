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


def _scan_geometry(record, data, timeline):
    stamp = record['stamp_ns']/1e9
    pose = timeline.at(stamp, exact_ns=record['stamp_ns'])
    if pose is None:
        return None
    if ((pose['session_id'], pose['generation'], pose['local_frame']) !=
            (record['session_id'], record['generation'], record['frame_id']) or
            not np.allclose(pose['local_from_lidar'], data['local_from_lidar'], atol=1e-8)):
        raise ValueError('Recorded scan and localization source do not agree')
    local, final = np.asarray(pose['local_from_lidar']), np.asarray(pose['world_from_lidar'])
    correction = final @ np.linalg.inv(local)
    points = data['points'] @ final[:3, :3].T+final[:3, 3]
    origins = data.get('origins')
    if origins is not None:
        if np.shape(origins) != np.shape(data['points']):
            raise ValueError('Ray origins must match the point observations')
        origins = origins @ correction[:3, :3].T+correction[:3, 3]
    model = record['ray_origin_model']
    if model == 'pose_interpolated_origins' and origins is None:
        raise ValueError('Moving-origin evidence was lost from the scan journal')
    if record['rays_used'] and model not in ('pose_interpolated_origins', 'bounded_scan_end_approximation'):
        raise ValueError('Unsupported scan ray origin model')
    return stamp, pose, correction, final, points, origins


def _z_bounds(candidate, timeline, config):
    floor = candidate.layers['ground_z']
    finite_floor = floor[np.isfinite(floor)]
    sensors = timeline.sensor_positions
    low = float(finite_floor.min()-.5) if len(finite_floor) else float(sensors[:, 2].min()-3)
    high = float(finite_floor.max()+config['max_height_m']+.5) if len(finite_floor) else float(sensors[:, 2].max()+3)
    return low, high


def fuse_scans(candidate, path, timeline, config, fingerprint, *, people=None, filter_mode='none', candidate_points=None,
               scan_stamps=None, global_candidate=None, global_totals=None, z_bounds=None, defer_derived=False,
               scan_masks=None, grid_offset=None):
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
    reference_floor = global_candidate.layers['ground_z'] if global_candidate is not None else floor
    reference_origin = np.asarray(global_candidate.metadata['origin_xy']) if global_candidate is not None else origin
    low_z, high_z = z_bounds if z_bounds is not None else _z_bounds(candidate, timeline, config)
    lower, upper = origin, origin+shape[::-1]*config['resolution']
    if global_candidate is not None:
        global_origin = np.asarray(global_candidate.metadata['origin_xy'])
        global_upper = global_origin+np.asarray(global_candidate.metadata['shape'][::-1])*config['resolution']
        # Point coordinates and voxel centers can fall on opposite sides of a
        # tile edge after float32 quantization. Keep a one-cell evidence halo.
        lower = np.maximum(global_origin, lower-config['resolution'])
        upper = np.minimum(global_upper, upper+config['resolution'])
    bounds = [*lower, low_z, *upper, high_z]
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
        tree.enable_static_retention()
        if geometric:
            tree.enable_static_review(config['dynamic']['static_observation_interval_s'],
                                      config['static']['ray_end_margin_m'])
        for record, data in read_scans(path, stamps=scan_stamps):
            scan = _scan_geometry(record, data, timeline)
            if scan is None:
                unbound += 1
                continue
            stamp, pose, correction, final, points, origins = scan
            use_rays = bool(record['rays_used'])
            masking = {}
            if review is not None:
                person_floor, person_origin = reference_floor, reference_origin
                if scan_masks is None:
                    row = people.at(stamp)
                    boxes, status = people_boxes(row, stamp, (pose['session_id'], pose['generation']),
                        pose['local_frame'], config['dynamic'], clock_domain=people.manifest['clock_domain'])
                    boxes = transform_boxes(boxes, correction)
                    mask = person_points(points, boxes, person_floor, person_origin, config,
                        ground_extrapolation=config['static']['mask_ground_extrapolation_m'] if geometric else 0.)
                    filter_counts['people_valid_scans'] += int(status['status'] == 'valid')
                    filter_counts['people_unavailable_scans'] += int(status['status'] != 'valid')
                else:
                    mask = np.unpackbits(scan_masks[record['stamp_ns']], count=len(points)).astype(bool)
                if global_candidate is not None:
                    # Use the same float32 coordinates as the native voxel update.
                    xy = np.asarray(points, np.float32)[:, :2]
                    selected = ((xy >= lower) & (xy < upper)).all(axis=1)
                    review.add(points[selected], mask[selected], stamp, person_floor, person_origin)
                else:
                    review.add(points, mask, stamp, floor, origin)
                filter_counts['masked_scans'] += int(mask.any())
                filter_counts['masked_points'] += int(mask.sum())
                if geometric:
                    filter_counts['extrapolated_ground_mask_points'] += int((mask & ~np.isfinite(
                        floor_at(points, person_floor, person_origin, config['resolution']))).sum())
                masking = dict(transient_mask=mask, transient_ttl=config['dynamic']['transient_ttl_s'], omit_transient=True)
            tree.update(points, final[:3, 3], bounds, stamp, settings, rays=use_rays,
                        ray_origins=origins, read=False, **masking)
            count += 1; rays += int(use_rays); last_stamp = stamp
        if global_totals is not None:
            count, unbound, rays, last_stamp = global_totals
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
    if grid_offset is not None:
        cells = np.floor((voxels[:, :2]-reference_origin)/config['resolution']).astype(int)-grid_offset
        core = ((cells >= 0) & (cells < shape[::-1])).all(axis=1)
    else:
        core = np.ones(len(voxels), bool)
    keep = np.ones(len(candidate_points), bool) if candidate_points is not None else None
    if review is not None:
        retained = review.keep(voxels[:, :3])
        rejected = (voxels[:, 3] >= config['occupied_probability']) & ~retained
        obstacle = obstacle_band(candidate_points, reference_floor, reference_origin, config)
        keep = ~obstacle | review.keep(candidate_points)
        filter_metadata = dict(mode=filter_mode, people=people.reference, counts=filter_counts,
            static_support_policy='fresh_nonperson_observations_after_last_person',
            mask_policy='bounded_xy_component_above_supported_ground',
            support=review.summary(), settings=config['dynamic'], rejected_occupied_voxels=int((rejected & core).sum()),
            rejected_endpoints=int((~keep).sum()))
        retained = ~rejected
    if geometric:
        def contradicted(points, votes):
            hits, free = votes.T
            return (obstacle_band(points, reference_floor, reference_origin, config) &
                    (free >= config['static']['free_min_observations']) &
                    (free >= np.maximum(hits, 1)*config['static']['free_to_hit_ratio']))
        rejected_voxels = contradicted(voxels[:, :3], voxel_votes)
        rejected_endpoints = contradicted(candidate_points, endpoint_votes)
        filter_metadata = filter_metadata or dict(mode=filter_mode, people=None, counts=filter_counts)
        filter_metadata.update(geometric_policy='independent_visible_ray_prefix_free_to_hit_votes',
            geometric_settings={k:config['static'][k] for k in ('free_min_observations','free_to_hit_ratio',
                'ray_end_margin_m','mask_ground_extrapolation_m')},
            geometric_rejected_occupied_voxels=int((rejected_voxels & retained & core &
                                                   (voxels[:, 3] >= config['occupied_probability'])).sum()),
            geometric_rejected_endpoints=int((rejected_endpoints & keep).sum()),
            mask_policy='bounded_xy_component_with_constrained_local_ground_extrapolation')
        filter_metadata['geometric_settings']['observation_interval_s'] = config['dynamic']['static_observation_interval_s']
        retained &= ~(rejected_voxels & (voxels[:, 3] >= config['occupied_probability']))
        keep &= ~rejected_endpoints
    if filter_metadata is not None:
        voxels = voxels[retained]
        candidates = endpoint_layers(candidate_points[keep], floor, origin, config,
            grid_origin=reference_origin if grid_offset is not None else None, grid_offset=grid_offset)
        filter_metadata.update(candidate_cells_before=int((candidate.layers['occupancy'] == 100).sum()),
                               candidate_cells_after=int((candidates['occupancy'] == 100).sum()))
    layers = project_evidence(voxels, floor, support, origin, config, last_stamp, candidates=candidates,
                              defer_derived=defer_derived,
                              grid_origin=reference_origin if grid_offset is not None else None, grid_offset=grid_offset)
    layers['last_observed'][~np.isfinite(layers['last_observed'])] = np.nan
    layers['endpoint_count'] = candidates['endpoint_count']
    from .changes import conditional_obstacles
    if not defer_derived:
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
                    conditional_obstacle_cells=int(layers['conditional_obstacle'].sum()) if not defer_derived else None)
    if filter_metadata is not None:
        metadata['dynamic_filter'] = filter_metadata
    for array in layers.values():
        array.flags.writeable = False
    return GridSnapshot(metadata, layers)


def fuse_scans_tiled(candidate, path, timeline, config, fingerprint, *, people=None, filter_mode='none', candidate_points,
                     progress=None):
    """Fuse bounded tiles using the same global ground and original scan order."""
    from .changes import conditional_obstacles
    from .layers import finish_evidence

    resolution = config['resolution']
    tile_cells = max(1, int(config['static']['tile_size_m']/resolution))
    height, width = candidate.metadata['shape']
    budget_errors = ('Occupancy voxel budget exceeded; map was reset to unknown', 'Static review voxel budget exceeded')
    failed_single = False
    if height <= tile_cells and width <= tile_cells:
        try:
            if progress:progress('正在融合整块地图；超过体素预算时会自动细分。')
            result = fuse_scans(candidate, path, timeline, config, fingerprint, people=people,
                                filter_mode=filter_mode, candidate_points=candidate_points)
            details = dict(strategy='single_tile', tile_size_m=config['static']['tile_size_m'], initial_tiles=1,
                           completed_tiles=1, adaptive_splits=0, peak_tile_voxels=result.metadata['voxel_count'])
            return GridSnapshot(dict(result.metadata, fusion=details), result.layers)
        except (RuntimeError, ValueError) as error:
            if str(error) not in budget_errors:raise
            failed_single = True
    origin = np.asarray(candidate.metadata['origin_xy'])
    floor = candidate.layers['ground_z']
    points = np.asarray(candidate_points)
    indices = np.floor((points[:, :2]-origin)/resolution).astype(np.int64)
    # Keep the original scan indices: subsampling in the native updater depends
    # on their positions and filtering points would change which rays are used.
    index = []
    scan_masks = {} if people is not None and filter_mode != 'none' else None
    mask_bytes = 0
    count = unbound = rays = 0
    last_stamp = None
    mask_counts = dict(people_valid_scans=0, people_unavailable_scans=0, masked_scans=0, masked_points=0,
                       extrapolated_ground_mask_points=0)
    if progress:progress('正在建立扫描空间索引。')
    for record, data in read_scans(path):
        scan = _scan_geometry(record, data, timeline)
        if scan is None:
            unbound += 1
            continue
        stamp, pose, correction, final, cloud, origins = scan
        if not np.isfinite(cloud).all() or not np.isfinite(final).all() or (origins is not None and not np.isfinite(origins).all()):
            raise ValueError('Invalid occupancy observation')
        count += 1; rays += int(bool(record['rays_used'])); last_stamp = stamp
        if people is not None and filter_mode != 'none':
            boxes, status = people_boxes(people.at(stamp), stamp, (pose['session_id'], pose['generation']),
                pose['local_frame'], config['dynamic'], clock_domain=people.manifest['clock_domain'])
            boxes = transform_boxes(boxes, correction)
            mask = person_points(cloud, boxes, floor, origin, config,
                ground_extrapolation=config['static']['mask_ground_extrapolation_m'] if filter_mode == 'geometric' else 0.)
            if scan_masks is not None:
                packed = np.packbits(mask)
                mask_bytes += packed.nbytes
                if mask_bytes <= 64*1024*1024:
                    scan_masks[record['stamp_ns']] = packed
                else:
                    scan_masks = None  # Recompute per tile instead of retaining unbounded masks.
            mask_counts['people_valid_scans'] += int(status['status'] == 'valid')
            mask_counts['people_unavailable_scans'] += int(status['status'] != 'valid')
            mask_counts['masked_scans'] += int(mask.any())
            mask_counts['masked_points'] += int(mask.sum())
            if filter_mode == 'geometric':
                mask_counts['extrapolated_ground_mask_points'] += int((mask & ~np.isfinite(
                    floor_at(cloud, floor, origin, resolution))).sum())
        if len(cloud):
            xy = np.asarray(cloud, np.float32)[:, :2]
            lo, hi = xy.min(axis=0).astype(float), xy.max(axis=0).astype(float)
            if record['rays_used']:
                ray_start = np.asarray(origins if origins is not None else final[None, :3, 3], np.float32)[:, :2]
                lo = np.minimum(lo, ray_start.min(axis=0)); hi = np.maximum(hi, ray_start.max(axis=0))
            index.append((record['stamp_ns'], lo, hi))
    if not count:
        raise ValueError('No scan could be bound to a final-map pose')
    totals = (count, unbound, rays, last_stamp)
    z_bounds = _z_bounds(candidate, timeline, config)
    pending = [(x, y, min(x+tile_cells, width), min(y+tile_cells, height))
               for y in range(0, height, tile_cells) for x in range(0, width, tile_cells)]
    layers = None
    summary = None
    tile_count = splits = voxel_count = max_voxels = candidate_cells_after = 0
    summed = ('geometric_rejected_occupied_voxels', 'geometric_rejected_endpoints',
              'rejected_occupied_voxels', 'rejected_endpoints')
    for_progress = len(pending)
    if failed_single:
        if width == 1 and height == 1:
            raise ValueError('A single-cell fusion tile exceeds the voxel budget')
        xm, ym = width//2, height//2
        xs = [(0, xm), (xm, width)] if xm else [(0, width)]
        ys = [(0, ym), (ym, height)] if ym else [(0, height)]
        pending = [(a, c, b, d) for c, d in ys for a, b in xs]
        splits = 1
    if progress:progress(f'开始融合 {for_progress} 个初始地图块，必要时会自动细分。')
    while pending:
        x0, y0, x1, y1 = pending.pop()
        tile_origin = origin + np.array([x0, y0])*resolution
        tile_upper = origin + np.array([x1, y1])*resolution
        stamps = [stamp for stamp, lo, hi in index if (hi >= tile_origin-resolution-1e-5).all() and
                  (lo <= tile_upper+resolution+1e-5).all()]
        use_points = ((indices[:, 0] >= x0) & (indices[:, 0] < x1) &
                      (indices[:, 1] >= y0) & (indices[:, 1] < y1))
        region = np.s_[y0:y1, x0:x1]
        meta = dict(candidate.metadata, origin_xy=tile_origin.tolist(), shape=[y1-y0, x1-x0])
        tile = GridSnapshot(meta, {key:value[region] for key,value in candidate.layers.items()})
        try:
            result = fuse_scans(tile, path, timeline, config, fingerprint, people=people, filter_mode=filter_mode,
                candidate_points=points[use_points], scan_stamps=stamps, global_candidate=candidate,
                global_totals=totals, z_bounds=z_bounds, defer_derived=True, scan_masks=scan_masks,
                grid_offset=np.array([x0, y0]))
        except (RuntimeError, ValueError) as error:
            if str(error) not in budget_errors or (x1-x0 == 1 and y1-y0 == 1):
                raise
            xm, ym = (x0+x1)//2, (y0+y1)//2
            xs = [(x0, xm), (xm, x1)] if xm > x0 else [(x0, x1)]
            ys = [(y0, ym), (ym, y1)] if ym > y0 else [(y0, y1)]
            pending.extend((a,c,b,d) for c,d in ys for a,b in xs)
            splits += 1
            continue
        if layers is None:
            layers = {name:np.empty((height,width), dtype=value.dtype) for name,value in result.layers.items()}
        for name, value in result.layers.items():
            layers[name][region] = value
        voxel_count += result.metadata['voxel_count']
        max_voxels = max(max_voxels, result.metadata['voxel_count'])
        tile_count += 1
        if progress and (tile_count % 5 == 0 or not pending):
            progress(f'已完成 {tile_count} 块，待处理 {len(pending)} 块，已细分 {splits} 次。')
        item = result.metadata.get('dynamic_filter')
        if item:
            candidate_cells_after += item['candidate_cells_after']
            if summary is None:
                summary = dict(item)
                for key in summed:
                    if key in summary:summary[key] = 0
                if 'support' in summary:summary['support'] = {k:0 for k in summary['support']}
            for key in summed:
                if key in item:summary[key] += item[key]
            if 'support' in item:
                for key, value in item['support'].items():summary['support'][key] += value
    # Distance fields and obstacle components must cross tile boundaries.
    layers.update(finish_evidence(layers['occupancy'], layers['observed'], layers['last_observed'],
        layers['coverage'], layers['obstacle_min_z'], layers['obstacle_max_z'],
        layers['ground_z'], layers['ground_support'], resolution))
    layers['conditional_obstacle'] = conditional_obstacles(layers, config)
    if summary is not None:
        summary['counts'] = mask_counts
        summary['candidate_cells_before'] = int((candidate.layers['occupancy'] == 100).sum())
        summary['candidate_cells_after'] = candidate_cells_after
        if 'support' in summary:summary['support_count_semantics'] = 'sum_per_tile_includes_boundary_halos'
    details = dict(strategy='spatial_tiles', tile_size_m=config['static']['tile_size_m'],
                   initial_tiles=for_progress, completed_tiles=tile_count, adaptive_splits=splits,
                   peak_tile_voxels=max_voxels, scan_index_entries=len(index),
                   cached_person_masks=scan_masks is not None,
                   voxel_count_semantics='sum_per_tile_includes_boundary_halos')
    binding = [candidate.metadata['snapshot_id'], fingerprint, details, summary]
    signature = hashlib.sha256(encoded(binding).encode()).hexdigest()[:20]
    metadata = dict(candidate.metadata, snapshot_id=candidate.metadata['map_id']+':scan_fusion:'+signature,
        stamp_s=last_stamp, geometry_status='STATIC_SCAN_OBSERVED' if rays else 'STATIC_SCAN_HITS_ONLY',
        source_evidence='deskewed_scans_and_final_map_endpoints', rays_used=bool(rays),
        free_space_supported=bool(rays), per_cell_observation_time_available=True,
        voxel_count=voxel_count, evidence_scans=count, unbound_scans=unbound, ray_scans=rays,
        final_endpoint_candidates_retained=filter_mode == 'none', dynamic_residual_cleanup_verified=False,
        occupancy_policy='revisable_historical_prior', conditional_policy=config['static'],
        static_sensor_model=dict(hit_invalidates_free=False, retention='entire_survey'),
        conditional_obstacle_cells=int(layers['conditional_obstacle'].sum()), fusion=details)
    if summary is not None:metadata['dynamic_filter'] = summary
    for value in layers.values():value.flags.writeable = False
    return GridSnapshot(metadata, layers)
