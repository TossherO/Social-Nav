"""Source-bound people observations for geometric map filtering; no ROS in readers."""
from bisect import bisect_right
import json
from pathlib import Path
import time

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint
from .layers import load_config
from .poses import PoseTimeline
from .storage import asset_reference, content_hash


class PeopleObservations:
    def __init__(self, directory, *, run_fingerprint, bag_fingerprint):
        directory = Path(directory)
        if (directory/'manifest.yaml').is_file():
            self.manifest = yaml.safe_load((directory/'manifest.yaml').read_text())
            manifest_hash = content_hash(directory/'manifest.yaml')
        else:
            run = json.loads((directory/'run.json').read_text())
            if not run.get('success'):
                raise ValueError('People observations require a successful source run')
            self.manifest = dict(kind='environment_people', source=dict(
                run_fingerprint=file_fingerprint(directory/'run.json'),
                bag_fingerprint=run['playback']['bag_fingerprint'],
                records_fingerprint=file_fingerprint(directory/'localization.jsonl')),
                clock_domain='ros_sim_time' if run.get('use_sim_time') else 'ros_time',
                people_sha256=content_hash(directory/'people.jsonl'))
            manifest_hash = content_hash(directory/'run.json')
        source = self.manifest['source']
        if (self.manifest.get('kind') != 'environment_people' or
                source['run_fingerprint'] != run_fingerprint or source['bag_fingerprint'] != bag_fingerprint):
            raise ValueError('People observations belong to another localization run/bag')
        path = directory/'people.jsonl'
        if content_hash(path) != self.manifest['people_sha256']:
            raise ValueError('People observations changed after source binding')
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.times = [row['stamp_s'] for row in self.rows]
        if (not self.rows or not np.isfinite(self.times).all() or
                np.any(np.diff(self.times) <= 0)):
            raise ValueError('People observations must be finite and strictly ordered')
        self.reference = dict(package=asset_reference(directory), manifest_sha256=manifest_hash,
                              people_sha256=self.manifest['people_sha256'])

    def at(self, stamp):
        index = bisect_right(self.times, stamp)-1
        return self.rows[index] if index >= 0 else None


def prepare_people(run_directory, output, *, detector_python, gpu, detector_config=None):
    """Read the original bag with system ROS; reuse the existing isolated detector worker."""
    import rosbag
    from perception.detection_tracking.annotation.worker import DetectorClient
    from perception.detection_tracking.inference.config import load_config as detector_settings, project_path
    from perception.sensors.pointcloud import pointcloud_fields
    from perception.sensors.pointcloud import input_fingerprint

    run_directory, output = Path(run_directory), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Select a new people-observation directory')
    root = Path(__file__).resolve().parents[2]
    if not any(parent in output.resolve().parents for parent in ((root/'outputs').resolve(), Path('/tmp'))):
        raise ValueError('People preparation output must stay under outputs or /tmp')
    run = json.loads((run_directory/'run.json').read_text())
    if not run.get('success') or run.get('source') != 'bag':
        raise ValueError('People preparation requires a successful recorded localization run')
    bag = Path(run['bag'])
    bag_digest = file_fingerprint(bag)
    if bag_digest != run['playback']['bag_fingerprint']:
        raise ValueError('Bag no longer matches the recorded localization run')
    records = run_directory/'localization.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
    timeline = PoseTimeline(rows, load_config()['poses'])
    settings = detector_settings(detector_config)
    effective = yaml.safe_load(project_path(settings['detector']['config']).read_text())
    if '_BASE_CONFIG_' in effective:
        raise ValueError('Use the standalone deployed detector configuration')
    output.mkdir(parents=True)
    client = DetectorClient(detector_python, gpu, settings, effective, tracking=settings['tracking'])
    sequence = 'environment:'+bag_digest['sha256'][:16]
    clock = 'ros_sim_time' if run.get('use_sim_time') else 'ros_time'

    started = time.monotonic()
    count, valid, tracks = 0, 0, 0
    try:
        with rosbag.Bag(str(bag), 'r') as source, (output/'people.jsonl').open('x') as stream:
            for _, message, _ in source.read_messages(topics=[settings['ros']['pointcloud_topic']]):
                stamp_ns = message.header.stamp.to_nsec()
                stamp = stamp_ns/1e9
                if not timeline.times[0] <= stamp <= timeline.times[-1]:
                    continue
                pose = timeline.at(stamp, exact_ns=stamp_ns)
                if pose is not None and pose['pose_age_s'] > settings['ros']['max_pose_age_s']:
                    pose = None
                pose_ns = None
                if pose is not None:
                    index = int(np.searchsorted(timeline.times, pose['pose_stamp_s'], side='right'))-1
                    pose_ns = rows[index]['stamp_ns']
                    if pose_ns > stamp_ns:
                        pose = None
                fields = pointcloud_fields(message, ('x', 'y', 'z', 'intensity'))
                points = np.column_stack([fields[name] for name in ('x', 'y', 'z', 'intensity')]).astype(np.float32)
                ident, sensor = sequence+'/'+str(stamp_ns), message.header.frame_id.lstrip('/')
                frame = dict(id=ident, source_id=sequence, stamp_ns=stamp_ns, sensor_frame=sensor,
                    clock_domain=clock, timestamp_source='ros_sensor_header', world_frame=rows[0]['frame_id'],
                    fingerprint=input_fingerprint(points, sequence, ident, stamp, sensor))
                bound_pose = dict(source_id=sequence, query_stamp_ns=stamp_ns, valid=True,
                    stamp_ns=pose_ns, frame_id=pose['local_frame'],
                    session_id=pose['session_id'], generation=pose['generation'],
                    local_from_lidar=pose['local_from_lidar']) if pose else None
                result = client.track(frame, points, bound_pose)
                result['localization_binding'] = dict(session_id=pose['session_id'], generation=pose['generation'],
                    frame_id=pose['local_frame'], source='recorded_localization') if pose else None
                result['stamp_ns'] = stamp_ns
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
                count += 1; valid += int(result['health'] == 'valid'); tracks += len(result['tracks'])
                if count % 100 == 0:
                    print('people: %d scans, %d valid, %.1fs' % (count, valid, time.monotonic()-started), flush=True)
    finally:
        client.close()
    if not count:
        raise ValueError('No raw scans matched the recorded localization time range')
    manifest = dict(kind='environment_people', source=dict(run=asset_reference(run_directory),
        run_fingerprint=file_fingerprint(run_directory/'run.json'), bag_fingerprint=bag_digest,
        records_fingerprint=file_fingerprint(records)), detector=client.metadata,
        tracking=settings['tracking'], clock_domain=clock, scans=count, valid=valid, tracks=tracks,
        wall_s=time.monotonic()-started, people_sha256=content_hash(output/'people.jsonl'),
        semantic_accuracy_evaluated=False)
    (output/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    return manifest
