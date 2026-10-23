"""Offline final-map export and indexed RGB references from the same source bag."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint, map_fingerprint, read_pcd
from .layers import GridSnapshot, load_config
from .maps import static_geometry
from .memory import (PoseTimeline, KeyObservations, create_index, add_observation, encoded,
                     camera_info_record, visual_record)
from .storage import asset_reference, content_hash, export_navigation


def index_images(bag_path, db, timeline, binding, settings, frame, calibration=None):
    import rosbag
    from genpy import Time
    selector = KeyObservations(settings)
    info = []
    counts = dict(images_seen=0, missing_pose=0, selected=0, missing_camera_info=0)
    # Receipt and acquisition clocks can differ slightly. Bind on acquisition, not read order.
    lower, upper = timeline.times[0]-2, timeline.times[-1]+2
    previous_image_stamp = None
    with rosbag.Bag(str(bag_path), 'r') as recording:
        for topic, message, receipt in recording.read_messages(
                topics=[settings['rgb_topic'], settings['camera_info_topic']],
                start_time=Time.from_sec(lower), end_time=Time.from_sec(upper)):
            if topic == settings['camera_info_topic']:
                info.append(camera_info_record(message))
                info = info[-64:]
                continue
            stamp_ns = message.header.stamp.to_nsec()
            stamp = stamp_ns/1e9
            if previous_image_stamp is not None and stamp_ns <= previous_image_stamp:
                raise ValueError('RGB acquisition time reversed or duplicated; split the recording explicitly')
            previous_image_stamp = stamp_ns
            counts['images_seen'] += 1
            pose = timeline.at(stamp)
            if pose is None:
                counts['missing_pose'] += 1
                continue
            if not selector.accept(stamp, pose):
                continue
            candidates = [item for item in info if item['stamp_ns'] <= stamp_ns and
                          item['frame_id'] == message.header.frame_id and
                          (item['width'], item['height']) == (message.width, message.height)]
            intrinsics = max(candidates, key=lambda item: item['stamp_ns']) if candidates else None
            if intrinsics is None:
                counts['missing_camera_info'] += 1
            row = visual_record(message, pose, binding, frame=frame, intrinsics=intrinsics,
                                calibration=calibration, receipt_ns=receipt.to_nsec())
            add_observation(db, row)
            counts['selected'] += 1
    db.commit()
    if not counts['selected']:
        raise ValueError('No RGB observation could be bound to this localization run')
    return counts


def export_environment(run_directory, output, *, config=None, bag=None, calibration=None, navigation=False):
    run_directory, output = Path(run_directory), Path(output)
    if output.exists():
        raise FileExistsError('Select a new environment output directory')
    config = config or load_config()
    settings = config['memory']
    for name in ('max_pose_age_s', 'keyframe_time_tolerance_s', 'displacement_m', 'rotation_rad', 'max_interval_s', 'max_observations'):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError('Visual sampling settings must be finite and positive')
    if type(settings['max_observations']) is not int:
        raise ValueError('Visual observation budget must be an integer')
    run = json.loads((run_directory/'run.json').read_text())
    if not run.get('success') or run.get('source') != 'bag' or run.get('mode') not in ('mapping', 'odometry'):
        raise ValueError('Use a successful recorded mapping or local-odometry run')
    bag_path = Path(bag) if bag else Path(run['bag'])
    bag_digest = file_fingerprint(bag_path)
    if bag_digest != run['playback']['bag_fingerprint']:
        raise ValueError('Bag does not match the original localization input')
    records = run_directory/'localization.jsonl'
    with records.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    calibration_path = Path(calibration) if calibration else None
    calibration_data = yaml.safe_load(calibration_path.read_text()) if calibration_path else None
    if calibration_data is not None:
        # The command explicitly selects the installation scope; no implicit current calibration.
        actual_frames = {row['child_frame_id'] for row in rows if row['health'] == 0}
        if actual_frames != {calibration_data['lidar_to_camera_optical']['source_frame']}:
            raise ValueError('Recorded LiDAR frame and camera calibration source differ')
    source = dict(run=asset_reference(run_directory), run_fingerprint=file_fingerprint(run_directory/'run.json'),
                  records_fingerprint=file_fingerprint(records), bag=asset_reference(bag_path), bag_fingerprint=bag_digest)
    map_digest = None
    if run['mode'] == 'mapping':
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
        timeline = PoseTimeline(rows, settings, keys)
        points = read_pcd(map_path/'GlobalMap.pcd', config['static']['max_points'], strict=True)
        map_id = 'roll:'+hashlib.sha256(encoded(map_digest).encode()).hexdigest()
        snapshot = static_geometry(points, keys[:, 1:4], config, map_id=map_id, frame=frame, stamp=keys[-1, 0])
        source.update(map=asset_reference(map_path), map_fingerprint=map_digest)
        scans = run_directory/'environment/scans.sqlite'
        expected_scans = run.get('environment_scans_fingerprint')
        if expected_scans is not None:
            digest = file_fingerprint(scans)
            if digest != expected_scans:
                raise ValueError('Deskewed scan journal no longer matches this mapping run')
            from .fusion import fuse_scans
            snapshot = fuse_scans(snapshot, scans, timeline, config, digest)
            source.update(scans=asset_reference(scans), scans_fingerprint=digest)
        kind = 'static_environment'
    else:
        timeline = PoseTimeline(rows, settings)
        snapshot = GridSnapshot.load(run_directory/'environment/layers.npz')
        frame = snapshot.metadata['frame_id']
        valid_identities = {(row['session_id'], row['generation'], row['frame_id']) for row in rows if row['health'] == 0}
        expected = (snapshot.metadata['session_id'], snapshot.metadata['generation'], frame)
        if valid_identities != {expected}:
            raise ValueError('Local memory export requires one continuous local frame/session')
        kind = 'local_session_memory'
    binding = dict(snapshot_id=snapshot.metadata['snapshot_id'], bag=source['bag'], bag_fingerprint=bag_digest,
                   source_id=hashlib.sha256(encoded(bag_digest).encode()).hexdigest(),
                   calibration_sha256=content_hash(calibration_path) if calibration_path else None,
                   rgb_topic=settings['rgb_topic'],
                   time_scope='offline_final_prior' if map_digest else 'recorded_local_session')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=str(output.parent)) as temporary:
        target = Path(temporary)/'package'
        target.mkdir()
        snapshot.save(target/'layers.npz')
        db = create_index(target/'observations.sqlite', binding)
        try:
            counts = index_images(bag_path, db, timeline, binding, settings, frame, calibration_data)
        finally:
            db.close()
        data = dict(kind=kind, snapshot_id=snapshot.metadata['snapshot_id'], frame_id=frame,
                    time_scope=binding['time_scope'], source=source, memory_binding=binding,
                    configuration=config, calibration=calibration_data, images=counts,
                    bound_final_keyframes=timeline.bound_keyframes,
                    layers_sha256=content_hash(target/'layers.npz'),
                    limitations=['not_navigation_reachability', 'observed_transitions_are_historical',
                                 'final_map_is_not_causal_input_to_its_own_mapping_run'])
        (target/'manifest.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        if navigation:
            export_navigation(snapshot, target)
        # A partially written package is never exposed at the requested output path.
        if output.exists():
            raise FileExistsError('Output appeared during export; refusing to replace it')
        target.rename(output)
    return data
