"""Causal pose lookup and final-keyframe correction for geometric observations."""
from bisect import bisect_right
from collections import deque

import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.results import rpy_quaternion


class PoseTimeline:
    """Earlier valid poses only; final corrections remain explicitly offline."""
    def __init__(self, rows, settings, keyframes=None, *, registered_rows=None):
        if keyframes is not None and registered_rows is not None:
            raise ValueError('Choose final keyframes or same-observation registered poses')
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
        self.registered = None
        if registered_rows is not None:
            self.registered = {row['stamp_ns']: row for row in registered_rows}
            if len(self.registered) != len(registered_rows):
                raise ValueError('Registered observations must have unique timestamps')
        self.sensor_positions = np.asarray([row['map_position'] for row in registered_rows
            if row['health'] == 0 and row.get('has_map_pose')] if registered_rows is not None else
            [row['position'] for row in rows if row['health'] == 0], dtype=float).reshape(-1, 3)

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
        if self.registered is not None:
            target = self.registered.get(row.get('stamp_ns'))
            if target is None or target['health'] != 0 or not target.get('has_map_pose'):
                return None
            if (abs(target['stamp_s']-row['stamp_s'])>1e-6 or not row.get('child_frame_id') or
                    target.get('child_frame_id')!=row.get('child_frame_id')):
                raise ValueError('Registered pose is not the same sensor observation')
            final = pose_matrix(target['map_position'], target['map_quaternion_xyzw'])
        return dict(pose_stamp_s=float(self.times[index]), pose_age_s=float(stamp-self.times[index]),
                    session_id=row['session_id'], generation=row['generation'], segment=segment,
                    local_frame=row['frame_id'], local_from_lidar=local.tolist(),
                    sensor_frame=row.get('child_frame_id'),
                    world_from_lidar=final.tolist(), final_keyframe=correction_key,
                    pose_source=('same_observation_registered_map_pose' if self.registered is not None else
                                 'offline_final_keyframe_correction' if self.final_map else 'recorded_local_pose'))


class PoseBuffer:
    def __init__(self, settings):
        self.settings = settings
        self.rows = deque(maxlen=settings['pose_history_size'])
        self.retired = deque(maxlen=32)
        self.identity, self.segment, self.healthy, self.last_stamp = None, 0, False, None

    def add(self, row):
        identity = (row['session_id'], row['generation'], row['frame_id'])
        stamp = row['stamp_ns']
        if identity in self.retired:
            return False
        if identity != self.identity:
            if self.identity is not None:
                self.retired.append(self.identity)
            self.rows.clear(); self.identity = identity; self.last_stamp = None; self.healthy = False
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        matrix = pose_matrix(row['position'], row['quaternion_xyzw']) if row['health'] == 0 else None
        if matrix is not None and not self.healthy:
            self.segment += 1
        self.rows.append(dict(row, matrix=matrix, segment=self.segment))
        self.last_stamp, self.healthy = stamp, matrix is not None
        return True

    def at(self, stamp_ns):
        row = next((item for item in reversed(self.rows) if item['stamp_ns'] <= stamp_ns), None)
        if row is None or row['matrix'] is None or (stamp_ns-row['stamp_ns'])/1e9 > self.settings['max_pose_age_s']:
            return None
        return dict(pose_stamp_s=row['stamp_ns']/1e9, pose_stamp_ns=row['stamp_ns'],
                    pose_age_s=(stamp_ns-row['stamp_ns'])/1e9, session_id=row['session_id'],
                    generation=row['generation'], segment=row['segment'], local_frame=row['frame_id'],
                    sensor_frame=row['child_frame_id'], local_from_lidar=row['matrix'].tolist(),
                    world_from_lidar=row['matrix'].tolist(), final_keyframe=None, pose_source='causal_local_pose')
