"""Build a static-map candidate by merging an old raster with one registered survey."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint, map_fingerprint, read_pcd
from .changes import conditional_obstacles
from .context import encoded
from .export import _save_package
from .fusion import fuse_scans_tiled
from .geometry import distance_to_zero, voxel_height_band
from .layers import GridSnapshot, finish_evidence, load_config
from .manual_edit import source_binding
from .maps import endpoint_layers
from .offline_ground import scan_ground
from .storage import StaticEnvironmentMap, asset_reference, content_hash


_MEASUREMENT_LAYERS = ('occupancy', 'observed', 'last_observed', 'coverage',
                       'obstacle_min_z', 'obstacle_max_z', 'endpoint_count')
_GROUND_DIAGNOSTICS = ('ground_scan_unreliable', 'ground_scan_undersampled',
                       'ground_scan_column_observed')
_DERIVED_LAYERS = {'clearance', 'unknown_distance', 'frontier', 'conditional_obstacle'}
_ADJACENT_PAIRS = ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :]))


def validate_static_parent(old_static, old_map):
    """Validate the existing static package against the actual fixed ROLL map."""
    asset = StaticEnvironmentMap(old_static)
    if asset.manifest['source'].get('map_fingerprint') != map_fingerprint(old_map):
        raise ValueError('Old static map does not bind to the selected fixed localization map')
    meta = asset.snapshot.metadata
    map_manifest = yaml.safe_load((Path(old_map)/'manifest.yaml').read_text())
    if map_manifest['frame'] != meta['frame_id']:
        raise ValueError('Old static map and fixed localization map frames disagree')
    if (np.asarray(meta['origin_xy']).shape != (2,) or not np.isfinite(meta['origin_xy']).all() or
            not np.isfinite(meta['resolution']) or meta['resolution'] <= 0 or
            not np.isfinite(meta['stamp_s']) or len(meta['shape']) != 2 or min(meta['shape']) <= 0 or
            not set(_MEASUREMENT_LAYERS+('ground_support', 'ground_z')).issubset(asset.snapshot.layers)):
        raise ValueError('Old static map has invalid grid geometry or missing evidence')
    return asset


def plan_survey_ground(layers, surveyed_ground, config):
    """Keep historical ground at overlap and stop new floor support at height jumps."""
    old_z, new_z = layers['ground_z'], surveyed_ground['ground_z']
    manual = layers.get('manual_edit', np.zeros(old_z.shape, bool))
    estimated = layers.get('ground_estimated', np.zeros(old_z.shape, bool))
    old_support = layers['ground_support'] & np.isfinite(old_z) & ~estimated
    new_support = surveyed_ground['ground_support'] & np.isfinite(new_z)
    unreliable = surveyed_ground.get('ground_scan_unreliable', np.zeros(old_z.shape, bool))
    new_support &= ~unreliable
    tolerance = config['ground_tracking']['height_tolerance_m']
    overlap = old_support & new_support
    difference = np.zeros(old_z.shape, np.float32)
    np.subtract(new_z, old_z, out=difference, where=overlap)
    conflict = overlap & (np.abs(difference) > tolerance)
    extension = new_support & ~old_support & ~manual & ~estimated
    seam = np.zeros(old_z.shape, bool)
    for old_side, new_side in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:, 1:], np.s_[:, :-1]),
                               (np.s_[:-1, :], np.s_[1:, :]), (np.s_[1:, :], np.s_[:-1, :])):
        seam[new_side] |= (old_support[old_side] & extension[new_side] &
                           (np.abs(old_z[old_side]-new_z[new_side]) > tolerance))
    extension &= ~seam
    manual_reference = manual & ~old_support & new_support
    usable = (old_support | extension | manual_reference) & ~conflict & ~unreliable
    floor = np.where(usable, np.where(extension | manual_reference, new_z, old_z), np.nan).astype(np.float32)
    ground = dict(ground_z=floor, ground_support=usable, survey_ground_z=new_z)
    for name in _GROUND_DIAGNOSTICS:
        ground[name] = surveyed_ground.get(name, np.zeros(old_z.shape, bool))
    masks = dict(ground_added=extension, ground_overlap=overlap, ground_height_conflict=conflict,
                 ground_seam_conflict=seam, survey_ground_unreliable=unreliable,
                 survey_ground_supported=new_support,
                 ground_compatible=overlap & ~conflict)
    return ground, masks


def measure_registered_survey(candidate, layers, item, config, *, progress=None):
    """Measure a survey on a common lattice using its own hits and actual rays."""
    source = item['source']
    digest = source['scans_fingerprint']
    tracked = scan_ground(candidate, item['path'], item['timeline'], config, digest,
                          progress=progress, allow_empty=True)
    ground, masks = plan_survey_ground(layers, tracked.layers, config)
    empty_points = np.empty((0, 3), np.float32)
    blank = endpoint_layers(empty_points, ground['ground_z'], np.asarray(candidate.metadata['origin_xy']), config)
    seed = GridSnapshot(tracked.metadata, dict(blank, **ground))
    # Require independent time buckets in every classified voxel and complete
    # vertical coverage to establish a free collision column.
    config = dict(config, free_vertical_coverage=1.)
    observed = fuse_scans_tiled(seed, item['path'], item['timeline'], config, digest,
        candidate_points=empty_points, require_repeated_evidence=True, progress=progress)
    bound, unbound, rays = (observed.metadata[k] for k in ('evidence_scans', 'unbound_scans', 'ray_scans'))
    if bound < 3 or bound/(bound+unbound) < .5 or rays < 3:
        raise ValueError(f'Insufficient registered ray evidence: bound={bound}, unbound={unbound}, ray_scans={rays}')
    return observed, ground, masks, tracked.metadata['ground_scan_diagnostics']


def _refine_weak_ground(layers, ground, masks, config):
    """Accept better overlap estimates only without changing their collision evidence."""
    old_z, new_z = layers['ground_z'], ground['survey_ground_z']
    weak = layers.get('ground_weak_support', layers.get('ground_scan_undersampled', np.zeros(old_z.shape, bool)))
    manual = layers.get('manual_edit', np.zeros(old_z.shape, bool))
    delta = np.abs(new_z-old_z)
    eligible = (weak & ~manual & masks['ground_compatible'] & ground['ground_scan_column_observed'] &
                (delta > 1e-4) & (delta <= config['ground_tolerance_m']))
    old_band = voxel_height_band(old_z, config['resolution'], config['min_obstacle_height_m'], config['max_height_m'])
    new_band = voxel_height_band(np.where(eligible, new_z, np.nan), config['resolution'],
                                 config['min_obstacle_height_m'], config['max_height_m'])
    same_band = (old_band[0] == new_band[0]) & (old_band[1] == new_band[1])
    no_collision_evidence = ((layers['occupancy'] == -1) & (layers['coverage'] == 0.) &
                             ~np.isfinite(layers['obstacle_min_z']) & ~np.isfinite(layers['obstacle_max_z']))
    band_conflict = eligible & ~(same_band | no_collision_evidence)
    eligible &= ~band_conflict
    seam = np.zeros(old_z.shape, bool)
    tolerance = config['ground_tracking']['height_tolerance_m']
    for first, second in _ADJACENT_PAIRS:
        for own, adjacent in ((first, second), (second, first)):
            # Check both possible neighbour heights: that neighbour may itself
            # be rejected by the band test or by a seam on its opposite side.
            with_old = layers['ground_support'][adjacent] & (np.abs(new_z[own]-old_z[adjacent]) > tolerance)
            with_new = eligible[adjacent] & (np.abs(new_z[own]-new_z[adjacent]) > tolerance)
            seam[own] |= eligible[own] & (with_old | with_new)
    refined = eligible & ~seam
    for name in ('obstacle_min_z', 'obstacle_max_z'):
        layers[name][refined] += old_z[refined]-new_z[refined]
    old_z[refined] = new_z[refined]
    if 'ground_weak_support' in layers:
        layers['ground_weak_support'][refined] = False
    for name in _GROUND_DIAGNOSTICS:
        if name in layers:
            layers[name][refined] = ground[name][refined]
    return dict(ground_refined=refined, ground_refinement_blocked_collision_band=band_conflict,
                ground_refinement_blocked_seam=seam)


def merge_survey_evidence(layers, survey, ground, masks, config, *, survey_start, survey_index,
                          protected_occupied=None):
    """Merge classified changes while carrying the original evidence everywhere else."""
    old_cells, new_cells = layers['occupancy'].copy(), survey['occupancy']
    manual = layers.get('manual_edit', np.zeros(old_cells.shape, bool))
    assumed = layers.get('assumed_free', np.zeros(old_cells.shape, bool))
    last = layers['last_observed']
    new_time = survey['last_observed']
    valid = (survey['observed'] & np.isfinite(new_time) & (new_time >= survey_start-1e-6) &
             (~np.isfinite(last) | (new_time > last)) & ~manual & ground['ground_support'])
    free = (valid & (new_cells == 0) & (survey['coverage'] >= 1.-1e-6))
    hit = valid & (new_cells == 100)
    # A historical obstacle is cleared only with supported, compatible ground
    # and a new fully observed free column. Candidate endpoint guards may veto.
    protected = np.zeros(old_cells.shape, bool) if protected_occupied is None else protected_occupied
    clear = free & (old_cells == 100) & masks['ground_compatible'] & ~protected
    changed = (hit & (old_cells != 100)) | (free & ((old_cells == -1) | assumed)) | clear
    for name in _MEASUREMENT_LAYERS:
        layers[name][changed] = survey[name][changed]
    if 'assumed_free' in layers:
        layers['assumed_free'][changed] = False
    if 'survey_index' in layers:
        layers['survey_index'][changed] = survey_index

    old_z = layers['ground_z']
    retained_obstacle = (old_cells == 100) & ~changed
    added = (masks['ground_added'] & (~np.isfinite(last) | (last < survey_start) | changed) &
             (~retained_obstacle | np.isfinite(old_z)))
    rebase = added & retained_obstacle
    for name in ('obstacle_min_z', 'obstacle_max_z'):
        layers[name][rebase] += old_z[rebase]-ground['ground_z'][rebase]
    for name in ('ground_z', 'ground_support')+_GROUND_DIAGNOSTICS:
        if name in layers:
            layers[name][added] = ground[name][added]
    if 'ground_weak_support' in layers:
        layers['ground_weak_support'][added] = ~ground['ground_scan_column_observed'][added]
    # A newly measured incompatible elevation withdraws old automatic free
    # classification. The original measurement is preserved in the report and
    # the parent asset, while the fused unknown cell has no collision support.
    # Carried-forward manual edits always remain available for explicit review.
    height_disputed = (masks['ground_height_conflict'] & ground['ground_scan_column_observed'] &
                       (old_cells == 0) & ~manual & (~np.isfinite(last) | (last < survey_start)))
    for name, absent in (('occupancy', -1), ('observed', False), ('last_observed', np.nan), ('coverage', 0.),
                         ('obstacle_min_z', np.nan), ('obstacle_max_z', np.nan), ('endpoint_count', 0)):
        layers[name][height_disputed] = absent
    if 'assumed_free' in layers:
        layers['assumed_free'][height_disputed] = False
    if 'survey_index' in layers:
        layers['survey_index'][height_disputed] = -1
    manual_disputed = (manual & ((survey['observed'] & np.isin(new_cells, (0, 100)) &
                                   (new_cells != old_cells)) | masks['ground_height_conflict']))
    refinements = _refine_weak_ground(layers, ground, masks, config)
    masks = dict(masks, **refinements, ground_added=added, survey_selected=changed, height_disputed_free=height_disputed,
                 manual_review=manual_disputed, retained_occupied_against_free=free & (old_cells == 100) & ~clear)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    counts.update(classified_cells=int((valid & np.isin(new_cells, (0, 100))).sum()),
                  newly_free=int((changed & (old_cells == -1) & (new_cells == 0)).sum()),
                  newly_occupied=int((changed & (old_cells != 100) & (new_cells == 100)).sum()),
                  cleared_occupied=int(clear.sum()),
                  occupancy_changed=int((old_cells != layers['occupancy']).sum()),
                  retained_parent_free=int(((old_cells == 0) & (layers['occupancy'] == 0) & ~changed).sum()),
                  retained_parent_ground=int((layers['ground_support'] & ~added & ~masks['ground_refined']).sum()))
    return masks, counts


def _fill_value(name, value):
    if name in ('occupancy', 'survey_index'):
        return -1
    if np.issubdtype(value.dtype, np.floating) and name != 'coverage':
        return np.nan
    return 0


def _extend_parent(parent, points, config):
    origin = np.asarray(parent.metadata['origin_xy'], dtype=float)
    resolution = parent.metadata['resolution']
    height, width = parent.metadata['shape']
    coordinates = (points[:, :2].astype(float)-origin)/resolution
    lower = np.minimum(0, np.floor(coordinates.min(axis=0)).astype(int))
    upper = np.maximum([width, height], np.floor(coordinates.max(axis=0)).astype(int)+1)
    size = upper-lower
    if int(size[0])*int(size[1]) > config['static']['max_cells']:
        raise ValueError('Combined static map exceeds its cell budget at the original resolution')
    region = np.s_[-lower[1]:height-lower[1], -lower[0]:width-lower[0]]
    shape = tuple(int(value) for value in size[::-1])
    layers = {}
    for name, value in parent.layers.items():
        if name in _DERIVED_LAYERS:
            continue
        layers[name] = np.full(shape, _fill_value(name, value), value.dtype)
        layers[name][region] = value
    for name in _GROUND_DIAGNOSTICS:
        layers.setdefault(name, np.zeros(shape, bool))
    if 'ground_weak_support' not in layers:
        endpoint_ground = (parent.metadata.get('geometry_status') == 'STATIC_ENDPOINT_CANDIDATES' or
                           parent.metadata.get('ground_estimator') == 'per_cell_median_of_unique_ground_points')
        layers['ground_weak_support'] = layers['ground_support'] & (
            endpoint_ground | layers['ground_scan_undersampled'])
    return origin+lower*resolution, layers, region


def _seam_statistics(layers, masks, tolerance):
    """Measure 4-neighbour height differences between retained and new ground."""
    supported = layers['ground_support']
    updated = masks['ground_added'] | masks['ground_refined']
    ground = layers['ground_z']
    values = []
    for first, second in _ADJACENT_PAIRS:
        edges = supported[first] & supported[second] & (updated[first] != updated[second])
        values.append(np.abs(ground[first][edges]-ground[second][edges]))
    differences = np.concatenate(values)
    return dict(adjacent_parent_to_survey_edges=int(len(differences)),
                p95_absolute_height_step_m=float(np.percentile(differences, 95)) if len(differences) else None,
                max_absolute_height_step_m=float(differences.max()) if len(differences) else None,
                edges_over_tolerance=int((differences > tolerance).sum()))


def _candidate_lineage(candidate_map, old_map, item):
    """Bind the new 3D package to the exact inputs that supplied its coordinates."""
    from perception.mapping_localization.manual_edit import source_binding as roll_binding

    lineage_path = candidate_map/'update_source.json'
    manifest = yaml.safe_load((candidate_map/'manifest.yaml').read_text())
    lineage_digest = content_hash(lineage_path)
    if lineage_digest != manifest.get('update_source_sha256'):
        raise ValueError('Candidate localization map lineage does not match its manifest')
    lineage = json.loads(lineage_path.read_text())
    source = item['source']
    inputs = dict(mapping_run=source['evidence_run_fingerprint'],
                  mapping_poses=source['evidence_records_fingerprint'],
                  localization_run=source['reference_run_fingerprint'],
                  localization_poses=source['reference_records_fingerprint'],
                  scans=source['scans_fingerprint'])
    if (lineage.get('old_binding') != roll_binding(old_map) or
            lineage.get('bag_fingerprint') != source['bag_fingerprint'] or
            lineage.get('input_fingerprints') != inputs):
        raise ValueError('Candidate localization map must come from the same fixed map and registered survey')
    guard_m = lineage['evidence']['deletion_hit_guard_m']
    if not np.isfinite(guard_m) or guard_m < 0:
        raise ValueError('Candidate localization map has an invalid obstacle deletion guard')
    return manifest['frame'], lineage_digest, guard_m


def export_incremental_environment(old_static, old_map, candidate_map, evidence_run, reference_run, output, *,
                                   config=None, export_pgm=False, progress=None, candidate_map_reference=None):
    """Write a portable, source-bound candidate static package and change record."""
    from .multi_export import _registered_source

    old_static, old_map, candidate_map, evidence_run, reference_run = [
        Path(path).resolve() for path in (old_static, old_map, candidate_map, evidence_run, reference_run)]
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Select a new environment output directory')
    if any(path == output.resolve() or path in output.resolve().parents or output.resolve() in path.parents
           for path in (old_static, old_map, candidate_map, evidence_run, reference_run)):
        raise ValueError('Static candidate output must not overlap its inputs')
    old_files = source_binding(old_static)
    parent_asset = validate_static_parent(old_static, old_map)
    parent = parent_asset.snapshot
    old_config = parent_asset.manifest['configuration']
    inherited = ('resolution', 'min_obstacle_height_m', 'max_height_m')
    # With no override, let the existing lattice and collision band govern.
    if config is None:
        config = dict(load_config(), **{key: old_config[key] for key in inherited})
    if any(not np.isclose(config[key], old_config[key], rtol=0, atol=1e-9) for key in inherited):
        raise ValueError('Static candidate must use the old map resolution and collision height band')
    if not np.isclose(config['resolution'], parent.metadata['resolution'], rtol=0, atol=1e-9):
        raise ValueError('Old static map raster resolution disagrees with its configuration')
    config = dict(config, compute=dict(config.get('compute', {}), backend='cpu'))
    frame = parent.metadata['frame_id']
    old_digest, candidate_digest = map_fingerprint(old_map), map_fingerprint(candidate_map)
    item = _registered_source(evidence_run, reference_run, old_digest, frame, config)
    candidate_frame, lineage_digest, guard_m = _candidate_lineage(candidate_map, old_map, item)
    if candidate_frame != frame:
        raise ValueError('Candidate localization map is not expressed in the old map frame')
    first, end = item['source']['observation_interval_s']
    times = parent.layers['last_observed']
    previous_end = max(parent.metadata['stamp_s'], float(times[np.isfinite(times)].max(initial=-np.inf)))
    if first <= previous_end:
        raise ValueError('New and old static acquisition times overlap or are not increasing')

    points = read_pcd(candidate_map/'GlobalMap.pcd', config['static']['max_points'], strict=True)
    if not len(points):
        raise ValueError('Candidate localization map has no endpoints')
    origin, layers, parent_region = _extend_parent(parent, points, config)
    resolution = config['resolution']
    shape = layers['occupancy'].shape
    sensors = item['timeline'].sensor_positions[:, :2]
    lower = np.maximum(0, np.floor((sensors.min(axis=0)-config['max_range_m']-origin)/resolution).astype(int))
    upper = np.minimum(shape[::-1],
        np.ceil((sensors.max(axis=0)+config['max_range_m']-origin)/resolution).astype(int)+1)
    if (upper <= lower).any():
        raise ValueError('Registered survey does not intersect the candidate raster')
    region = np.s_[lower[1]:upper[1], lower[0]:upper[0]]
    survey_origin = origin+lower*resolution
    local = {name: value[region] for name, value in layers.items()}
    map_id = 'roll:'+hashlib.sha256(encoded(candidate_digest).encode()).hexdigest()
    seed_meta = dict(parent.metadata, map_id=map_id, origin_xy=survey_origin.tolist(),
                     shape=list(local['occupancy'].shape))
    seed = GridSnapshot(seed_meta, {name: local[name] for name in ('ground_z','ground_support')})
    survey, ground, masks, diagnostics = measure_registered_survey(seed, local, item, config, progress=progress)
    endpoints = endpoint_layers(points, ground['ground_z'], survey_origin, config)
    nearby_hit = (endpoints['occupancy'] == 100) | (survey.layers['occupancy'] == 100)
    protected = distance_to_zero(~nearby_hit)*resolution <= guard_m+1e-6 if nearby_hit.any() else nearby_hit
    # Copy only the crop used for visual comparison, before mutating the output.
    before = {name: local[name].copy() for name in _MEASUREMENT_LAYERS+('ground_support', 'ground_z')}
    masks, counts = merge_survey_evidence(local, survey.layers, ground, masks, config, survey_start=first,
                                           survey_index=-3, protected_occupied=protected)
    layers.update(finish_evidence(layers['occupancy'], layers['observed'], layers['last_observed'],
        layers['coverage'], layers['obstacle_min_z'], layers['obstacle_max_z'],
        layers['ground_z'], layers['ground_support'], resolution))
    layers['conditional_obstacle'] = conditional_obstacles(layers, config)
    if 'manual_edit' in layers:
        layers['conditional_obstacle'] &= ~layers['manual_edit']
    assumed = layers.get('assumed_free', np.zeros(shape, bool))
    layers['frontier'] &= ~assumed
    measured_free = ((layers['occupancy'] == 0) & ~assumed & layers['observed'] &
                     np.isfinite(layers['last_observed']))

    source = dict(map=asset_reference(candidate_map_reference or candidate_map), map_fingerprint=candidate_digest,
                  input_kind='incremental_registered_recorded_rays', candidate_lineage_sha256=lineage_digest,
                  parent=dict(static=asset_reference(old_static), static_files=old_files,
                              snapshot_id=parent.metadata['snapshot_id'], map=asset_reference(old_map),
                              map_fingerprint=old_digest, ground_estimator=parent.metadata.get('ground_estimator'),
                              limitations=parent_asset.manifest.get('limitations', [])),
                  survey=item['source'])
    metadata = dict(entity_type='map', map_type='static_environment_map',
        snapshot_id=map_id+':incremental:'+hashlib.sha256(encoded([old_files, lineage_digest, config]).encode()).hexdigest()[:20],
        map_id=map_id, session_id=map_id, frame_id=frame, generation=0, origin_xy=origin.tolist(), shape=list(shape),
        resolution=resolution, stamp_s=survey.metadata['stamp_s'], association={}, world_from_lidar=None,
        time_scope='offline_final_prior', geometry_status='STATIC_SCAN_OBSERVED', rays_used=True,
        free_space_supported=bool(measured_free.any()),
        per_cell_observation_time_available=bool(np.isfinite(layers['last_observed']).any()),
        source_evidence='parent_and_registered_survey',
        occupancy_policy='retained_parent_with_repeated_registered_changes', ground_estimator='retained_parent_and_tracked_survey',
        navigation_reachability_verified=False, dynamic_residual_cleanup_verified=False,
        conditional_obstacle_cells=int(layers['conditional_obstacle'].sum()), fusion=survey.metadata['fusion'])
    if 'manual_edit' in layers:
        metadata['manual_edit'] = parent.metadata.get('manual_edit')
        metadata['manual_free_cells_total'] = int((layers['manual_edit'] & (layers['occupancy'] == 0)).sum())
    if parent.metadata.get('free_assumption') and assumed.any():
        metadata['free_assumption'] = parent.metadata['free_assumption']
        metadata['reference_policy'] = parent.metadata['reference_policy']
    if 'ground_estimated' in layers:
        metadata['ground_estimated_cells'] = int(layers['ground_estimated'].sum())
    if 'survey_index' in layers:
        metadata['survey_index_meaning'] = 'parent_values_retained_minus_three_incremental_survey'
    limits = ['not_navigation_reachability', 'cross_session_clock_comparability_not_independently_verified',
              'repeated_time_buckets_do_not_prove_independent_viewpoints',
              'registration_and_dynamic_cleanup_not_independently_verified',
              'height_conflicts_and_manual_disagreements_require_review']
    if assumed.any():
        limits.append('retained_assumed_free_is_not_sensor_observed')
    overlap = masks['ground_overlap']
    dz = ground['survey_ground_z'][overlap]-before['ground_z'][overlap]
    alignment = dict(method='same_observation_fixed_map_localization_on_parent_lattice',
        supported_ground_overlap_cells=int(overlap.sum()),
        median_survey_minus_parent_ground_m=float(np.median(dz)) if len(dz) else None,
        p95_absolute_ground_difference_m=float(np.percentile(np.abs(dz), 95)) if len(dz) else None,
        max_registered_pose_innovation_m=item['source']['max_pose_innovation_m'])
    seam = _seam_statistics(local, masks, config['ground_tracking']['height_tolerance_m'])
    report = dict(status='candidate_not_accepted', source=source,
        grid=dict(origin_xy=origin.tolist(), shape=list(shape), resolution_m=resolution,
                  parent_offset_yx=[int(axis.start) for axis in parent_region],
                  survey_offset_yx=lower[::-1].tolist(), survey_shape=list(local['occupancy'].shape),
                  survey_origin_xy=survey_origin.tolist()),
        policy=dict(acquisition_interval_s=[first, end], prior_acquisition_end_s=previous_end,
                    grid_extent='union_of_parent_grid_and_candidate_localization_endpoints',
                    ground_conflict_m=config['ground_tracking']['height_tolerance_m'], hit_guard_m=guard_m,
                    max_weak_ground_refinement_m=config['ground_tolerance_m'],
                    free_vertical_coverage=1., free_min_observations=config['static']['free_min_observations'],
                    free_to_hit_ratio=config['static']['free_to_hit_ratio'],
                    observation_bucket_s=config['dynamic']['static_observation_interval_s'],
                    ray_end_margin_m=config['static']['ray_end_margin_m']),
        registration=alignment, ground_seam=seam, survey=dict(ground=diagnostics, **{name:survey.metadata[name] for name in
                    ('fusion', 'evidence_scans', 'unbound_scans', 'ray_scans')}),
        counts=dict(counts, output_shape=list(shape), surveyed_cells=int(survey.layers['observed'].sum()),
                    parent_free_total=int((parent.layers['occupancy'] == 0).sum()),
                    parent_ground_total=int(parent.layers['ground_support'].sum()),
                    output_free_total=int((layers['occupancy'] == 0).sum()),
                    output_ground_total=int(layers['ground_support'].sum()),
                    preserved_manual_cells=int(layers.get('manual_edit', np.empty(0,bool)).sum())),
        limitations=limits)

    def write_extras(target):
        masks_path = target/'changes.npz'
        np.savez_compressed(masks_path, **masks,
            **{'parent_'+name: value for name,value in before.items()},
            survey_ground_z=ground['survey_ground_z'], survey_coverage=survey.layers['coverage'],
            **{'survey_'+name: survey.layers[name] for name in ('occupancy','last_observed','observed')},
            **{'candidate_'+name: local[name] for name in before})
        changes_hash = content_hash(masks_path)
        result_path = target/'update_report.json'
        result_path.write_text(json.dumps(dict(report, changes_sha256=changes_hash), ensure_ascii=False, indent=2)+'\n')
        return {'changes.npz': changes_hash, 'update_report.json': content_hash(result_path)}

    if (source_binding(old_static) != old_files or map_fingerprint(old_map) != old_digest or
            map_fingerprint(candidate_map) != candidate_digest or
            content_hash(candidate_map/'update_source.json') != lineage_digest):
        raise ValueError('A map input changed during incremental fusion')
    survey_source = item['source']
    survey_files = [(item['bag'], survey_source['bag_fingerprint']),
                    (item['path'], survey_source['scans_fingerprint'])]
    for run, prefix in ((evidence_run, 'evidence'), (reference_run, 'reference')):
        survey_files.extend(((run/'run.json', survey_source[prefix+'_run_fingerprint']),
                             (run/'localization.jsonl', survey_source[prefix+'_records_fingerprint'])))
    for path, expected in survey_files:
        if file_fingerprint(path) != expected:
            raise ValueError('Survey input changed during incremental fusion: '+str(path))
    return _save_package(GridSnapshot(metadata, layers), output, source, config, frame, 0, export_pgm, limits,
                         write_extras=write_extras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-static', 'old-map', 'candidate-map', 'mapping-run', 'localization-run', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--config')
    parser.add_argument('--export-pgm', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export_incremental_environment(
        args.old_static, args.old_map, args.candidate_map, args.mapping_run, args.localization_run, args.out,
        config=load_config(args.config) if args.config else None, export_pgm=args.export_pgm,
        progress=lambda message: print(message, file=sys.stderr, flush=True)),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
