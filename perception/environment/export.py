"""Export static geometry from a bound final map and optional deskewed rays."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint, map_fingerprint, read_pcd
from .context import encoded
from .layers import load_config, STATIC_FILTER_MODES
from .maps import static_geometry
from .poses import PoseTimeline
from .storage import asset_reference, content_hash, export_map_server_grid


def export_environment(run_directory, output, *, config=None, bag=None, export_pgm=False, people=None, filter_mode=None,
                       bounds=None, progress=None):
    run_directory, output = Path(run_directory), Path(output)
    if output.exists():
        raise FileExistsError('Select a new environment output directory')
    config = config or load_config()
    # Full-map historical evidence and static review use the CPU implementation.
    config = dict(config, compute=dict(config.get('compute',{}), backend='cpu'))
    run = json.loads((run_directory/'run.json').read_text())
    if not run.get('success') or run.get('source') not in ('bag','live') or run.get('mode') != 'mapping':
        raise ValueError('Use a successful recorded mapping run for static geometry export')
    filter_mode = config['static']['dynamic_filter'] if filter_mode is None else filter_mode
    if filter_mode not in STATIC_FILTER_MODES:
        raise ValueError('Unknown static dynamic-filter mode')
    config = dict(config, static=dict(config['static'], dynamic_filter=filter_mode))
    if filter_mode != 'none' and not run.get('environment_scans_fingerprint'):
        raise ValueError('Static filtering requires a bound sealed scan journal; select none explicitly for endpoint-only export')
    bag_path,bag_digest=None,None
    if run['source']=='bag':
        bag_path = Path(bag) if bag else Path(run['bag'])
        bag_digest = file_fingerprint(bag_path)
        if bag_digest != run['playback']['bag_fingerprint']:
            raise ValueError('Bag does not match the original localization input')
    elif bag is not None or not run.get('environment_scans_fingerprint'):
        raise ValueError('Live export requires its own bound sealed scan journal; a bag cannot replace that source')
    records = run_directory/'localization.jsonl'
    with records.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    source = dict(run=asset_reference(run_directory), run_fingerprint=file_fingerprint(run_directory/'run.json'),
                  records_fingerprint=file_fingerprint(records), input_kind=run['source'],bag_fingerprint=bag_digest)
    if bag_path is not None:source['bag']=asset_reference(bag_path)
    observations = None
    if filter_mode != 'none' and people is not None:
        from .people import PeopleObservations
        observations = PeopleObservations(people, run_fingerprint=source['run_fingerprint'], bag_fingerprint=bag_digest)
        if observations.manifest['source']['records_fingerprint'] != source['records_fingerprint']:
            raise ValueError('People observations used different recorded poses')
        source['people'] = observations.reference
    map_path = run_directory/'map'
    map_digest = map_fingerprint(map_path)
    if map_digest != run['playback']['map_fingerprint']:
        raise ValueError('Final map no longer matches this mapping run')
    manifest = yaml.safe_load((map_path/'manifest.yaml').read_text())
    if not isinstance(manifest.get('frame'), str) or not manifest['frame']:
        raise ValueError('Final map has no frame')
    frame = manifest['frame']
    keys = np.loadtxt(map_path/'keyframes/poses.txt', ndmin=2)
    if keys.shape[1] != 9 or not np.isfinite(keys).all() or np.any(np.diff(keys[:, 0]) <= 0):
        raise ValueError('Invalid final ROLL keyframe poses')
    timeline = PoseTimeline(rows, config['poses'], keys)
    points = read_pcd(map_path/'GlobalMap.pcd', config['static']['max_points'], strict=True)
    map_id = 'roll:'+hashlib.sha256(encoded(map_digest).encode()).hexdigest()
    snapshot = static_geometry(points, keys[:, 1:4], config, map_id=map_id, frame=frame, stamp=keys[-1, 0],bounds=bounds)
    if bounds is not None:
        origin=np.asarray(snapshot.metadata['origin_xy']);extent=origin+np.asarray(snapshot.metadata['shape'][::-1])*config['resolution']
        points=points[((points[:,:2]>=origin)&(points[:,:2]<extent)).all(axis=1)]
    source.update(map=asset_reference(map_path), map_fingerprint=map_digest)
    scans = run_directory/'environment/scans.sqlite'
    expected_scans = run.get('environment_scans_fingerprint')
    if expected_scans is not None:
        from .scans import sealed_scan_summary
        sealed_scan_summary(scans)
        digest = file_fingerprint(scans)
        if digest != expected_scans:
            raise ValueError('Deskewed scan journal no longer matches this mapping run')
        from .fusion import fuse_scans_tiled
        snapshot = fuse_scans_tiled(snapshot, scans, timeline, config, digest, people=observations,
                              filter_mode=filter_mode, candidate_points=points, progress=progress)
        source.update(scans=asset_reference(scans), scans_fingerprint=digest)
    return _save_package(snapshot, output, source, config, frame, timeline.bound_keyframes, export_pgm,
                         ['not_navigation_reachability', 'final_map_is_not_causal_input_to_its_own_mapping_run'])


def export_endpoint_environment(map_path, output, *, config=None, bounds=None, export_pgm=False,
                                projection_strategy='observed_endpoints', projection_settings=None):
    """Import a localization map with an explicit endpoint projection strategy."""
    map_path, output = Path(map_path), Path(output)
    if output.exists():
        raise FileExistsError('Select a new environment output directory')
    config = config or load_config()
    config = dict(config, static=dict(config['static'], dynamic_filter='none'),
                  compute=dict(config.get('compute',{}), backend='cpu'))
    digest = map_fingerprint(map_path)
    frame = yaml.safe_load((map_path/'manifest.yaml').read_text())['frame']
    if not isinstance(frame, str) or not frame:
        raise ValueError('Localization map has no frame')
    keys = np.loadtxt(map_path/'keyframes/poses.txt', ndmin=2)
    if keys.shape[1] != 9 or not np.isfinite(keys).all() or np.any(np.diff(keys[:, 0]) <= 0):
        raise ValueError('Invalid localization-map keyframe poses')
    points = read_pcd(map_path/'GlobalMap.pcd', config['static']['max_points'], strict=True)
    map_id = 'roll:'+hashlib.sha256(encoded(digest).encode()).hexdigest()
    if projection_strategy=='observed_endpoints':
        if projection_settings is not None:raise ValueError('Projection settings require an assumed-free strategy')
        snapshot = static_geometry(points, keys[:, 1:4], config, map_id=map_id, frame=frame,
                                   stamp=keys[-1, 0], bounds=bounds)
    elif projection_strategy in ('docker_assume_free', 'ground_relative_assume_free'):
        from .projection import docker_projection, ground_relative_projection
        project = ground_relative_projection if projection_strategy == 'ground_relative_assume_free' else docker_projection
        snapshot = project(points,keys[:,1:4],config,map_id=map_id,frame=frame,
                                     stamp=keys[-1,0],bounds=bounds,**(projection_settings or {}))
    else:
        raise ValueError('Unknown endpoint projection strategy')
    source = dict(map=asset_reference(map_path), map_fingerprint=digest, input_kind='localization_map_endpoints')
    limitations=['not_navigation_reachability', 'no_recorded_rays_or_free_space',
                 'source_point_observation_times_unavailable']
    if projection_strategy in ('docker_assume_free', 'ground_relative_assume_free'):
        source['projection_strategy']=projection_strategy
        limitations.append('unobserved_cells_assumed_free_not_measured')
    return _save_package(snapshot, output, source, config, frame, 0, export_pgm,
                         limitations)


def export_registered_environment(map_path, evidence_run, reference_run, output, *, config=None,
                                  bounds=None, export_pgm=False, progress=None):
    """Fuse existing scan rays using an independent same-bag localization in the target map."""
    map_path, evidence_run, reference_run, output = map(Path, (map_path,evidence_run,reference_run,output))
    if output.exists():
        raise FileExistsError('Select a new environment output directory')
    run = json.loads((evidence_run/'run.json').read_text())
    reference = json.loads((reference_run/'run.json').read_text())
    digest = map_fingerprint(map_path)
    if (not run.get('success') or run.get('source') != 'bag' or not reference.get('success') or
            reference.get('source') != 'bag' or reference.get('mode') != 'localization'):
        raise ValueError('Use a successful scan recording and successful fixed-map bag localization')
    if (run['playback']['bag_fingerprint'] != reference['playback']['bag_fingerprint'] or
            reference['playback']['map_fingerprint'] != digest):
        raise ValueError('Evidence/reference must share the same bag and the requested target map')
    if file_fingerprint(run['bag']) != run['playback']['bag_fingerprint']:
        raise ValueError('Evidence source bag changed')
    from .scans import sealed_scan_summary
    scans = evidence_run/'environment/scans.sqlite'
    archive = sealed_scan_summary(scans)
    if file_fingerprint(scans) != run.get('environment_scans_fingerprint'):
        raise ValueError('Evidence scan journal changed or was never bound')
    rows = [json.loads(x) for x in (evidence_run/'localization.jsonl').read_text().splitlines() if x.strip()]
    targets = [json.loads(x) for x in (reference_run/'localization.jsonl').read_text().splitlines() if x.strip()]
    frame = yaml.safe_load((map_path/'manifest.yaml').read_text())['frame']
    if any(r.get('map_frame') != frame for r in targets if r.get('has_map_pose')):
        raise ValueError('Registered poses use a different target map frame')
    config = config or load_config()
    config = dict(config, compute=dict(config.get('compute',{}),backend='cpu'))
    timeline = PoseTimeline(rows,config['poses'],registered_rows=targets)
    points = read_pcd(map_path/'GlobalMap.pcd',config['static']['max_points'],strict=True)
    keys = np.loadtxt(map_path/'keyframes/poses.txt',ndmin=2)
    if keys.shape[1]!=9 or not np.isfinite(keys).all() or np.any(np.diff(keys[:,0])<=0):
        raise ValueError('Invalid target localization-map keyframe poses')
    map_id = 'roll:'+hashlib.sha256(encoded(digest).encode()).hexdigest()
    # Preserve ground-supported structure from the complete historical map,
    # including places the additional scan recording never visited. Only rays
    # from the registered recording can add measured free space there.
    sensors=np.r_[keys[:,1:4],timeline.sensor_positions]
    candidate = static_geometry(points,sensors,config,map_id=map_id,frame=frame,
                                stamp=targets[-1]['stamp_s'],bounds=bounds)
    if bounds is not None:
        origin=np.asarray(candidate.metadata['origin_xy']);upper=origin+np.asarray(candidate.metadata['shape'][::-1])*config['resolution']
        points=points[((points[:,:2]>=origin)&(points[:,:2]<upper)).all(axis=1)]
    from .fusion import fuse_scans_tiled
    snapshot=fuse_scans_tiled(candidate,scans,timeline,config,file_fingerprint(scans),
                        filter_mode='geometric',candidate_points=points,progress=progress)
    source=dict(map=asset_reference(map_path),map_fingerprint=digest,input_kind='registered_recorded_rays',
        evidence_run=asset_reference(evidence_run),evidence_run_fingerprint=file_fingerprint(evidence_run/'run.json'),
        evidence_records_fingerprint=file_fingerprint(evidence_run/'localization.jsonl'),
        reference_run=asset_reference(reference_run),reference_run_fingerprint=file_fingerprint(reference_run/'run.json'),
        reference_records_fingerprint=file_fingerprint(reference_run/'localization.jsonl'),
        bag_fingerprint=run['playback']['bag_fingerprint'],scans=asset_reference(scans),
        scans_fingerprint=file_fingerprint(scans),recorded_scans=archive['records'],
        ground_seed_source='target_map_keyframes_and_registered_sensor_positions')
    return _save_package(snapshot,output,source,config,frame,0,export_pgm,
        ['not_navigation_reachability','only_same_observation_valid_map_poses_contribute',
         'not_independent_of_evidence_and_reference_recordings'])


def _save_package(snapshot, output, source, config, frame, bound_keyframes, export_pgm, limitations):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=str(output.parent)) as temporary:
        target = Path(temporary)/'package'
        target.mkdir()
        snapshot.save(target/'layers.npz')
        data = dict(entity_type='map',map_type='static_environment_map',kind='static_environment', snapshot_id=snapshot.metadata['snapshot_id'], frame_id=frame,
                    time_scope='offline_final_prior', source=source, configuration=config,
                    bound_final_keyframes=bound_keyframes,
                    layers_sha256=content_hash(target/'layers.npz'),
                    limitations=limitations)
        (target/'manifest.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        if export_pgm:
            export_map_server_grid(snapshot, target)
        if output.exists():
            raise FileExistsError('Output appeared during export; refusing to replace it')
        target.rename(output)
    return data
