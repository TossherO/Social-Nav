"""Time-ordered static export from independently registered survey sessions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint, map_fingerprint, read_pcd
from .context import encoded
from .layers import GridSnapshot, finish_evidence, load_config
from .maps import static_geometry
from .poses import PoseTimeline
from .scans import sealed_scan_summary
from .storage import asset_reference


def _registered_source(evidence, reference, map_digest, frame, config):
    from perception.mapping_localization.update_map import _pose_innovation, _sensor_parameters
    evidence, reference = Path(evidence), Path(reference)
    run = json.loads((evidence/'run.json').read_text())
    ref = json.loads((reference/'run.json').read_text())
    if (not run.get('success') or run.get('source') != 'bag' or not ref.get('success') or
            ref.get('source') != 'bag' or ref.get('mode') != 'localization'):
        raise ValueError('Each survey needs a successful bag scan run and fixed-map localization')
    bag_digest = run['playback']['bag_fingerprint']
    if ref['playback']['bag_fingerprint'] != bag_digest or ref['playback']['map_fingerprint'] != map_digest:
        raise ValueError('Evidence/reference must share a bag and use the selected final map')
    first, second = _sensor_parameters(run), _sensor_parameters(ref)
    if any(not np.allclose(a,b,atol=1e-9,rtol=0.) for a,b in zip(first,second)):
        raise ValueError('Evidence/reference use different sensor extrinsics or IMU timing')
    if file_fingerprint(run['bag']) != bag_digest:
        raise ValueError('Evidence source bag changed')
    scans = evidence/'environment/scans.sqlite'
    archive = sealed_scan_summary(scans)
    scans_digest = file_fingerprint(scans)
    if scans_digest != run.get('environment_scans_fingerprint'):
        raise ValueError('Evidence scan journal changed or was never bound')
    rows_path, targets_path = evidence/'localization.jsonl', reference/'localization.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    targets = [json.loads(line) for line in targets_path.read_text().splitlines() if line.strip()]
    if any(row.get('map_frame') != frame for row in targets if row.get('has_map_pose')):
        raise ValueError('Registered poses use a different target map frame')
    timeline = PoseTimeline(rows, config['poses'], registered_rows=targets)
    if not len(timeline.sensor_positions):
        raise ValueError('A survey has no valid registered map poses')
    previous = None
    maximum = 0.
    for row in rows:
        pose = timeline.at(row['stamp_s'], exact_ns=row.get('stamp_ns'))
        if pose is None:
            continue
        local, mapped = np.asarray(pose['local_from_lidar']), np.asarray(pose['world_from_lidar'])
        if previous is not None and row['stamp_s']-previous[0] <= 2.:
            distance, angle = _pose_innovation(previous[1], previous[2], mapped, local)
            maximum = max(maximum, distance)
            if distance > .5 or angle > .35:
                raise ValueError('Registered survey has a large map correction jump at '
                                 f'{row["stamp_s"]:.6f}: {distance:.3f} m, {np.degrees(angle):.2f} deg')
        previous = (row['stamp_s'], mapped, local)
    source = dict(evidence_run=asset_reference(evidence), evidence_run_fingerprint=file_fingerprint(evidence/'run.json'),
                  evidence_records_fingerprint=file_fingerprint(rows_path), reference_run=asset_reference(reference),
                  reference_run_fingerprint=file_fingerprint(reference/'run.json'),
                  reference_records_fingerprint=file_fingerprint(targets_path), bag_fingerprint=bag_digest,
                  scans=asset_reference(scans), scans_fingerprint=scans_digest, recorded_scans=archive['records'],
                  observation_interval_s=[float(timeline.times[0]), float(timeline.times[-1])],
                  max_pose_innovation_m=maximum)
    return dict(path=scans, timeline=timeline, source=source, bag=Path(run['bag']),
                evidence=evidence, reference=reference)


def _map_feature_times(map_path, keys, candidate, config):
    """Latest contributing keyframe time per obstacle cell; never a ray timestamp."""
    from perception.mapping_localization.manual_edit import read_xyzi, _world
    shape=candidate.metadata['shape'];origin=np.asarray(candidate.metadata['origin_xy'])
    floor=candidate.layers['ground_z'];resolution=config['resolution']
    result=np.full(shape,-np.inf,dtype=np.float64)
    for row in keys:
        stamp=float(row[0]);index=int(row[7])
        for kind in ('corner','surf'):
            cloud=read_xyzi(Path(map_path)/'keyframes'/f'{kind}{index}.pcd')
            if not len(cloud):continue
            world=_world(cloud,row[1:7]);cells=np.floor((world[:,:2]-origin)/resolution).astype(int)
            inside=((cells>=0)&(cells<np.asarray(shape[::-1]))).all(axis=1)
            world,cells=world[inside],cells[inside]
            if not len(cells):continue
            x,y=cells.T;height=world[:,2]-floor[y,x]
            hit=np.isfinite(height)&(height>=config['min_obstacle_height_m'])&(height<=config['max_height_m'])
            np.maximum.at(result,(y[hit],x[hit]),stamp)
    result[~np.isfinite(result)]=np.nan
    return result


def _merge_session(layers, session, index, *, baseline_occupied=None, feature_times=None, survey_start=None):
    """Replace only cells with classified, timestamped measurements from this survey."""
    values = session.layers
    valid = (np.isin(values['occupancy'], [0, 100]) & values['observed'] &
             np.isfinite(values['last_observed']))
    if baseline_occupied is not None:
        # Endpoints from the final map may be newer than this whole survey.
        # Only clear one if every measurement in the survey is later than its
        # latest contributing keyframe. Unknown endpoint ages remain occupied.
        blocked=(baseline_occupied & (values['occupancy']==0) &
                 (~np.isfinite(feature_times) | (feature_times >= survey_start)))
        valid &= ~blocked
    # Session evidence without enough vertical free support remains unknown.
    previous = layers['occupancy'].copy()
    for name in ('occupancy', 'observed', 'last_observed', 'coverage', 'obstacle_min_z', 'obstacle_max_z'):
        layers[name][valid] = values[name][valid]
    layers['endpoint_count'][valid] = values['endpoint_count'][valid]
    layers['survey_index'][valid] = index
    return dict(classified_cells=int(valid.sum()), changed_cells=int((valid & (previous != layers['occupancy'])).sum()),
                cleared_occupied=int((valid & (previous == 100) & (layers['occupancy'] == 0)).sum()),
                newly_occupied=int((valid & (previous != 100) & (layers['occupancy'] == 100)).sum()))


def export_registered_multi_environment(map_path, evidence_runs, reference_runs, output, *, config=None,
                                        bounds=None, export_pgm=False, progress=None):
    """Fuse distinct, nonoverlapping surveys from old to new without rewriting times."""
    from .changes import conditional_obstacles
    from .export import _save_package
    from .fusion import fuse_scans_tiled

    map_path, output = Path(map_path), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Select a new environment output directory')
    if len(evidence_runs) < 2 or len(evidence_runs) != len(reference_runs):
        raise ValueError('Provide at least two paired evidence and reference runs')
    config = config or load_config()
    config = dict(config, compute=dict(config.get('compute', {}), backend='cpu'))
    digest = map_fingerprint(map_path)
    frame = yaml.safe_load((map_path/'manifest.yaml').read_text())['frame']
    surveys = [_registered_source(e, r, digest, frame, config) for e,r in zip(evidence_runs, reference_runs)]
    surveys.sort(key=lambda item: item['timeline'].times[0])
    identities = [encoded(item['source']['bag_fingerprint']) for item in surveys]
    if len(set(identities)) != len(identities):
        raise ValueError('Each survey must use a different bag')
    for earlier, later in zip(surveys, surveys[1:]):
        if earlier['timeline'].times[-1] >= later['timeline'].times[0]:
            raise ValueError('Survey timestamps overlap; chronological priority cannot be determined')
    points = read_pcd(map_path/'GlobalMap.pcd', config['static']['max_points'], strict=True)
    keys = np.loadtxt(map_path/'keyframes/poses.txt', ndmin=2)
    if keys.shape[1] != 9 or not np.isfinite(keys).all() or np.any(np.diff(keys[:,0]) <= 0):
        raise ValueError('Invalid target localization-map keyframe poses')
    map_id = 'roll:'+hashlib.sha256(encoded(digest).encode()).hexdigest()
    sensors = np.concatenate([keys[:,1:4]]+[item['timeline'].sensor_positions for item in surveys])
    candidate = static_geometry(points, sensors, config, map_id=map_id, frame=frame,
                                stamp=float(surveys[-1]['timeline'].times[-1]), bounds=bounds)
    layers = {name: value.copy() for name,value in candidate.layers.items()}
    layers['survey_index'] = np.full(candidate.layers['occupancy'].shape, -1, dtype=np.int16)
    if progress: progress('正在核对最终地图关键帧时间，保护较新的障碍特征。')
    feature_times=_map_feature_times(map_path,keys,candidate,config)
    layers['map_feature_time_s']=feature_times
    baseline_occupied=candidate.layers['occupancy']==100
    # The geometric baseline is separate: only measured session voxels can
    # override an earlier session, never unstamped final-map endpoints.
    empty_points = np.empty((0, 3), dtype=np.float32)
    for index, item in enumerate(surveys):
        if progress: progress(f'融合第 {index+1}/{len(surveys)} 次采集：{item["source"]["evidence_run"]["path"]}')
        result = fuse_scans_tiled(candidate, item['path'], item['timeline'], config,
                                  item['source']['scans_fingerprint'], filter_mode='geometric',
                                  candidate_points=empty_points, progress=progress)
        bound, unbound, ray_count = (result.metadata[key] for key in ('evidence_scans','unbound_scans','ray_scans'))
        if bound < 3 or bound/(bound+unbound) < .5 or ray_count < 3:
            raise ValueError(f'Insufficient registered ray evidence in survey {index+1}: '
                             f'bound={bound}, unbound={unbound}, ray_scans={ray_count}')
        item['source']['fusion'] = result.metadata.get('fusion')
        item['source']['evidence_scans'] = result.metadata['evidence_scans']
        item['source']['unbound_scans'] = result.metadata['unbound_scans']
        item['source']['ray_scans'] = result.metadata['ray_scans']
        item['source']['merge'] = _merge_session(layers, result, index, baseline_occupied=baseline_occupied,
                                                feature_times=feature_times,
                                                survey_start=float(item['timeline'].times[0]))
    layers.update(finish_evidence(layers['occupancy'], layers['observed'], layers['last_observed'],
        layers['coverage'], layers['obstacle_min_z'], layers['obstacle_max_z'], layers['ground_z'],
        layers['ground_support'], config['resolution']))
    layers['conditional_obstacle'] = conditional_obstacles(layers, config)
    sources = [item['source'] for item in surveys]
    if map_fingerprint(map_path) != digest:
        raise ValueError('Target map changed during static fusion')
    for item in surveys:
        source=item['source']
        for path, expected in ((item['bag'],source['bag_fingerprint']),
                               (item['path'],source['scans_fingerprint']),
                               (item['evidence']/'run.json',source['evidence_run_fingerprint']),
                               (item['evidence']/'localization.jsonl',source['evidence_records_fingerprint']),
                               (item['reference']/'run.json',source['reference_run_fingerprint']),
                               (item['reference']/'localization.jsonl',source['reference_records_fingerprint'])):
            if file_fingerprint(path) != expected:
                raise ValueError('Survey input changed during static fusion: '+str(path))
    signature = hashlib.sha256(encoded([candidate.metadata['snapshot_id'], sources,
                                        'latest_classified_survey_v1']).encode()).hexdigest()[:20]
    metadata = dict(candidate.metadata, snapshot_id=map_id+':multi_survey:'+signature,
        geometry_status='STATIC_SCAN_OBSERVED', source_evidence='time_ordered_registered_surveys',
        rays_used=any(item['ray_scans'] for item in sources),
        free_space_supported=any(item['ray_scans'] for item in sources),
        per_cell_observation_time_available=True, occupancy_policy='latest_classified_survey',
        survey_index_meaning='zero_based_manifest_sources_minus_one_final_map_endpoints',
        map_feature_time_semantics='latest_obstacle_keyframe_acquisition_time_not_ray_observation',
        evidence_scans=sum(item['evidence_scans'] for item in sources),
        unbound_scans=sum(item['unbound_scans'] for item in sources),
        ray_scans=sum(item['ray_scans'] for item in sources),
        dynamic_residual_cleanup_verified=False,
        conditional_obstacle_cells=int(layers['conditional_obstacle'].sum()))
    for value in layers.values(): value.flags.writeable = False
    snapshot = GridSnapshot(metadata, layers)
    source = dict(map=asset_reference(map_path), map_fingerprint=digest,
                  input_kind='time_ordered_registered_recorded_rays', surveys=sources,
                  temporal_policy='latest_classified_survey_preserves_unobserved_and_newer_map_features',
                  ordering_basis='recorded_sensor_timestamps_assumed_comparable')
    return _save_package(snapshot, output, source, config, frame, 0, export_pgm,
        ['not_navigation_reachability', 'cross_session_clock_comparability_not_independently_verified',
         'registration_and_dynamic_cleanup_not_independently_verified',
         'only_classified_observations_supersede_earlier_sessions',
         'same_session_feature_and_free_order_not_proven'])
