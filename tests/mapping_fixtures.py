"""Synthetic mapping runs and reference localization for offline map tests."""
import json
from pathlib import Path

import numpy as np
import yaml

from perception.environment.layers import load_config
from perception.environment.scans import ScanJournal
from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.manual_edit import write_xyzi
from perception.mapping_localization.results import file_fingerprint, map_fingerprint


def _write_map(path, rows, clouds):
    keyframes = path / 'keyframes'
    keyframes.mkdir(parents=True)
    clouds = [np.asarray(cloud, np.float32) for cloud in clouds]
    for index, cloud in enumerate(clouds):
        write_xyzi(keyframes / f'corner{index}.pcd', np.empty((0, 4), np.float32))
        write_xyzi(keyframes / f'surf{index}.pcd', cloud)
    np.savetxt(keyframes / 'poses.txt', np.asarray(rows), fmt='%.17g')
    write_xyzi(path / 'GlobalMap.pcd', np.concatenate(clouds))
    (path / 'manifest.yaml').write_text(yaml.safe_dump(dict(
        format='roll_keyframes_v1', frame='map', units='metres_radians',
        keyframes=len(rows), visualization_resolution=.1)))
    (path / 'COMPLETE').write_text('roll_keyframes_v1\n')


def write_parent_map(root):
    old = Path(root) / 'old'
    _write_map(old, [[100, 0, 0, 0, 0, 0, 0, 0, 1], [101, 0, 0, 0, 0, 0, 0, 1, 1]],
               [[[1, 0, 0, 1], [1, 1, 0, 1], [4, 4, 1, 1]], [[1, 0, 0, 1], [1, 1, 0, 1]]])
    return old


def registered_map_inputs(root):
    root = Path(root)
    old = write_parent_map(root)
    mapping = root / 'mapping'
    reference = root / 'reference'
    mapping.mkdir()
    reference.mkdir()
    bag = root / 'input.bag'
    bag.write_bytes(b'synthetic bag')
    _write_map(mapping / 'map', [[200, 0, 0, 0, 0, 0, 0, 0, 1], [201.2, .2, 0, 0, 0, 0, 0, 1, 1]],
               [[[3, 0, 0, 1]], [[2.8, 0, 0, 1]]])
    scans = mapping / 'environment/scans.sqlite'
    journal = ScanJournal(scans, load_config()['scan_recording'], {})
    rows = []
    targets = []
    metadata = dict(session_id='synthetic', generation=0, frame_id='odom', sensor_frame='rslidar',
                    rays_used=True, ray_origin_model='pose_interpolated_origins')
    try:
        for index in range(6):
            stamp = 200 + index * .6
            x = index * .1
            ns = round(stamp * 1e9)
            row = dict(stamp_s=stamp, stamp_ns=ns, frame_id='odom', child_frame_id='rslidar',
                       position=[x, 0, 0], quaternion_xyzw=[0, 0, 0, 1], health=0,
                       session_id='synthetic', generation=0)
            rows.append(row)
            targets.append(dict(row, has_map_pose=True, map_frame='map', map_position=[x, 0, 0],
                                map_quaternion_xyzw=[0, 0, 0, 1]))
            points = np.array([[3 - x, 0, 0]], np.float32)
            origins = np.array([[x, 0, 0]], np.float32)
            journal.append(points, pose_matrix(row['position'], row['quaternion_xyzw']), ns,
                           metadata, origins, np.array([0], np.float32))
    finally:
        journal.close()
    for folder, records in ((mapping, rows), (reference, targets)):
        (folder / 'localization.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    digest = file_fingerprint(bag)

    def run(mode, map_path):
        return dict(success=True, source='bag', mode=mode, bag=str(bag),
                    algorithm_parameters=dict(mapping=dict(extrinsic_R=np.eye(3).reshape(-1).tolist(),
                                                          extrinsic_T=[0., 0., 0.], extrinsic_est_en=False),
                                              common=dict(imu_time_offset=0., time_sync_en=False)),
                    playback=dict(bag_fingerprint=digest, map_fingerprint=map_fingerprint(map_path)))

    mapping_run = run('mapping', mapping / 'map')
    mapping_run['environment_scans_fingerprint'] = file_fingerprint(scans)
    (mapping / 'run.json').write_text(json.dumps(mapping_run))
    (reference / 'run.json').write_text(json.dumps(run('localization', old)))
    return old, mapping, reference
