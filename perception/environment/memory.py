"""Bounded visual observations and manual places, with original sensor times."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.results import rpy_quaternion


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class PoseTimeline:
    """Earlier valid poses only; final corrections remain explicitly offline."""
    def __init__(self, rows, settings, keyframes=None):
        self.rows, self.settings = list(rows), settings
        self.times = np.asarray([row['stamp_s'] for row in rows], dtype=float)
        self.nanoseconds = {row['stamp_ns']: i for i, row in enumerate(rows) if 'stamp_ns' in row}
        if not len(rows) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('A strictly increasing recorded localization timeline is required')
        self.segments, self.local = [], []
        identity, segment = None, 0
        for row in rows:
            current = (row['session_id'], row['generation'], row['frame_id'])
            if row['health'] != 0:
                self.segments.append(None); self.local.append(None)
                identity = None
                continue
            if current != identity:
                segment += 1
                identity = current
            self.segments.append(segment)
            self.local.append(pose_matrix(row['position'], row['quaternion_xyzw']))
        self.corrections = {}
        self.bound_keyframes = 0
        if keyframes is not None:
            for number, key in enumerate(keyframes):
                index = int(np.argmin(np.abs(self.times-key[0])))
                if (abs(self.times[index]-key[0]) > settings['keyframe_time_tolerance_s'] or
                        self.local[index] is None):
                    continue
                final = pose_matrix(key[1:4], rpy_quaternion(*key[4:7]))
                entry = (self.times[index], final @ np.linalg.inv(self.local[index]), number)
                self.corrections.setdefault(self.segments[index], []).append(entry)
                self.bound_keyframes += 1
            if not self.bound_keyframes:
                raise ValueError('Final keyframes cannot be bound to this localization run')
            for entries in self.corrections.values():
                entries.sort(key=lambda item: item[0])
        self.final_map = keyframes is not None

    def at(self, stamp, *, exact_ns=None):
        index = self.nanoseconds.get(exact_ns, -1) if exact_ns is not None else bisect_right(self.times, stamp)-1
        if exact_ns is not None and index >= 0:
            stamp = self.times[index]
        if index < 0 or stamp-self.times[index] > self.settings['max_pose_age_s'] or self.local[index] is None:
            return None
        row, local, segment = self.rows[index], self.local[index], self.segments[index]
        correction_key, final = None, local
        if self.final_map:
            entries = self.corrections.get(segment, [])
            key = bisect_right([item[0] for item in entries], self.times[index])-1
            if key < 0:
                return None
            _, correction, correction_key = entries[key]
            final = correction @ local
        return dict(pose_stamp_s=float(self.times[index]), pose_age_s=float(stamp-self.times[index]),
                    session_id=row['session_id'], generation=row['generation'], segment=segment,
                    local_frame=row['frame_id'], local_from_lidar=local.tolist(),
                    sensor_frame=row.get('child_frame_id'),
                    world_from_lidar=final.tolist(), final_keyframe=correction_key,
                    pose_source='offline_final_keyframe_correction' if self.final_map else 'recorded_local_pose')


class KeyObservations:
    def __init__(self, settings):
        self.settings, self.previous, self.count = settings, None, 0

    def accept(self, stamp, pose):
        transform = np.asarray(pose['world_from_lidar'])
        previous = self.previous
        choose = previous is None or pose['segment'] != previous[2]
        if not choose:
            angle = np.arccos(np.clip((np.trace(previous[1][:3, :3].T @ transform[:3, :3])-1)/2, -1, 1))
            choose = (stamp-previous[0] >= self.settings['max_interval_s'] or
                      np.linalg.norm(transform[:3, 3]-previous[1][:3, 3]) >= self.settings['displacement_m'] or
                      angle >= self.settings['rotation_rad'])
        if choose:
            if self.count >= self.settings['max_observations']:
                raise ValueError('Visual observation budget exceeded; split this dataset explicitly')
            self.previous = (stamp, transform, pose['segment'])
            self.count += 1
        return choose


def create_index(path, binding):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to replace an observation index')
    db = sqlite3.connect(str(path))
    db.executescript('''
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE observations(id TEXT PRIMARY KEY, stamp_s REAL NOT NULL,
            segment INTEGER NOT NULL, record_json TEXT NOT NULL);
        CREATE INDEX observations_time ON observations(stamp_s);
        CREATE TABLE places(id TEXT PRIMARY KEY, name TEXT NOT NULL,
            observation_id TEXT NOT NULL REFERENCES observations(id), record_json TEXT NOT NULL);
        CREATE TABLE transitions(source TEXT NOT NULL REFERENCES observations(id),
            target TEXT NOT NULL REFERENCES observations(id), kind TEXT NOT NULL,
            PRIMARY KEY(source,target));
        CREATE TABLE image_assets(observation_id TEXT PRIMARY KEY REFERENCES observations(id),
            sha256 TEXT NOT NULL, data BLOB NOT NULL);
    ''')
    db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', encoded(binding)))
    db.commit()
    return db


def add_observation(db, record):
    row = dict(record)
    ident = hashlib.sha256(encoded([row['source_id'], row['image_stamp_ns'], row['receipt_ns'],
                                    row.get('session_id'), row.get('generation')]).encode()).hexdigest()[:24]
    row['id'] = ident
    previous = db.execute('SELECT id,segment FROM observations ORDER BY rowid DESC LIMIT 1').fetchone()
    db.execute('INSERT INTO observations VALUES (?,?,?,?)',
               (ident, row['stamp_s'], row['segment'], encoded(row)))
    if previous and previous[1] == row['segment']:
        db.execute('INSERT INTO transitions VALUES (?,?,?)', (previous[0], ident, 'observed_transition'))
    return row


def read_index(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    return db


def query_observations(db, *, as_of, position=None, radius=None, heading=None, heading_tolerance=.8, limit=20,
                       session_id=None, generation=None, frame=None):
    if not np.isfinite(as_of) or type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('A finite observation cutoff and a bounded limit are required')
    if position is not None:
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all() or radius is None or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Expected finite XY position and a positive radius')
    elif radius is not None:
        raise ValueError('A radius query requires an explicit XY position')
    if heading is not None and (not np.isfinite(heading) or not 0 < heading_tolerance <= np.pi):
        raise ValueError('Invalid heading query')
    result = []
    for entry in db.execute('SELECT record_json FROM observations WHERE stamp_s<=? ORDER BY stamp_s DESC', (float(as_of),)):
        row = json.loads(entry[0])
        if ((session_id is not None and row['session_id'] != session_id) or
                (generation is not None and row['generation'] != generation) or
                (frame is not None and row['frame_id'] != frame)):
            continue
        pose = np.asarray(row['world_from_camera'] if row.get('world_from_camera') is not None else row['world_from_lidar'])
        if position is not None and np.linalg.norm(pose[:2, 3]-position) > radius:
            continue
        # Camera forward is optical +Z; LiDAR fallback is explicitly +X.
        forward = pose[:3, 2 if row.get('world_from_camera') is not None else 0]
        if heading is not None and np.linalg.norm(forward[:2]) < 1e-6:
            continue  # A vertical optical axis has no observed horizontal heading.
        yaw = np.arctan2(forward[1], forward[0])
        if heading is not None and abs(np.arctan2(np.sin(yaw-heading), np.cos(yaw-heading))) > heading_tolerance:
            continue
        result.append(row)
        if len(result) == limit:
            break
    return result


def camera_info_record(message):
    return dict(stamp_ns=message.header.stamp.to_nsec(), frame_id=message.header.frame_id,
                width=message.width, height=message.height, distortion_model=message.distortion_model,
                K=list(message.K), D=list(message.D), R=list(message.R), P=list(message.P))


def visual_record(message, pose, binding, *, frame, intrinsics=None, calibration=None, receipt_ns=None,
                  association=None):
    """Share original RGB and prior-pose binding between offline and live indexing."""
    from perception.sensors.image import rgb_array
    from perception.detection_tracking.inference.geometry import validate_transform
    rgb = rgb_array(message)
    future = (pose['pose_stamp_ns'] > message.header.stamp.to_nsec() if 'pose_stamp_ns' in pose
              else pose['pose_stamp_s'] > message.header.stamp.to_sec())
    if future:
        raise ValueError('Cannot bind a future pose to this RGB observation')
    if intrinsics is not None and (intrinsics['stamp_ns'] > message.header.stamp.to_nsec() or
            intrinsics['frame_id'] != message.header.frame_id or
            (intrinsics['width'], intrinsics['height']) != (message.width, message.height)):
        raise ValueError('CameraInfo does not bind to this image')
    camera = None
    if calibration is not None:
        value = calibration['lidar_to_camera_optical']
        if message.header.frame_id != value['target_frame'] or pose['sensor_frame'] != value['source_frame']:
            raise ValueError('Camera calibration frames do not bind to this observation')
        camera = (np.asarray(pose['world_from_lidar']) @ np.linalg.inv(validate_transform(value['matrix']))).tolist()
    return dict(pose, source_id=binding['source_id'], stamp_s=message.header.stamp.to_sec(),
                image_stamp_ns=message.header.stamp.to_nsec(), receipt_ns=receipt_ns,
                rgb_topic=binding['rgb_topic'], optical_frame=message.header.frame_id, frame_id=frame,
                image_sha256=hashlib.sha256(message.data).hexdigest(), width=message.width,
                height=message.height, encoding=message.encoding, camera_info=intrinsics,
                world_from_camera=camera, camera_pose_status='explicit_calibration_at_prior_pose_time'
                    if camera is not None else 'calibration_not_supplied',
                calibration_sha256=binding['calibration_sha256'], time_scope=binding['time_scope'],
                association=association or {}, decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                image_storage='bag_reference' if binding.get('bag') is not None else 'embedded_png')


def add_place(db, ident, name, observation_id, *, position=None, bounds=None, aliases=()):
    if not ident or not name or not isinstance(ident, str) or not isinstance(name, str):
        raise ValueError('A nonempty place identity and name are required')
    entry = db.execute('SELECT record_json FROM observations WHERE id=?', (observation_id,)).fetchone()
    if entry is None:
        raise ValueError('A place must reference an observation in this environment')
    observation = json.loads(entry[0])
    position = np.asarray(position if position is not None else np.asarray(observation['world_from_lidar'])[:3, 3], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Place coordinates must be finite XYZ metres in the environment frame')
    if bounds is not None:
        box = np.asarray(bounds, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError('Invalid place region bounds')
        bounds = box.tolist()
    if not all(isinstance(alias, str) and alias for alias in aliases):
        raise ValueError('Aliases must be nonempty strings')
    row = dict(id=ident, name=name, observation_id=observation_id, position=position.tolist(),
               frame_id=observation['frame_id'], bounds=bounds, aliases=list(aliases),
               semantic_source='manual', stamp_s=observation['stamp_s'], navigation_verified=False)
    db.execute('INSERT INTO places VALUES (?,?,?,?)', (ident, name, observation_id, encoded(row)))
    db.commit()
    return row


def query_places(db, *, name=None, bounds=None, as_of):
    if not np.isfinite(as_of):
        raise ValueError('A finite observation cutoff is required')
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
            raise ValueError('Invalid place query bounds')
    result = []
    for entry in db.execute('SELECT record_json FROM places ORDER BY id'):
        row = json.loads(entry[0])
        if row['stamp_s'] > as_of or (name is not None and name not in [row['id'], row['name'], *row['aliases']]):
            continue
        if bounds is not None:
            region = row.get('bounds')
            low, high = (np.array(region[:2]), np.array(region[2:])) if region is not None else (
                np.array(row['position'][:2]), np.array(row['position'][:2]))
            if np.any(high < bounds[:2]) or np.any(low > bounds[2:]):
                continue
        result.append(row)
    return result
