"""Causal local-pose history and one bounded visual-memory writer."""
from collections import deque
import hashlib
import sqlite3

import cv2
import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.sensors.image import rgb_array
from .memory import KeyObservations, create_index, add_observation, visual_record, query_observations


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


class VisualWriter:
    """Owns its SQLite connection in the image worker, outside geometry callbacks."""
    def __init__(self, path, binding, settings, calibration=None):
        self.db = create_index(path, binding)
        self.binding, self.settings, self.calibration = binding, settings, calibration
        self.selector = KeyObservations(settings)
        self.images_bytes, self.count = 0, 0
        self.last_image = None

    def append(self, message, pose, *, intrinsics=None, association=None):
        stamp = message.header.stamp.to_sec()
        identity = (pose['session_id'], pose['generation'], message.header.stamp.to_nsec())
        if self.last_image is not None and identity[:2] == self.last_image[:2] and identity[2] <= self.last_image[2]:
            return None
        self.last_image = identity
        if not self.selector.accept(stamp, pose):
            return None
        row = visual_record(message, pose, self.binding, frame=pose['local_frame'], intrinsics=intrinsics,
                            calibration=self.calibration, association=association)
        with self.db:
            row = add_observation(self.db, row)
            if row['image_storage'] == 'embedded_png':
                success, data = cv2.imencode('.png', rgb_array(message)[:, :, ::-1])
                if not success or self.images_bytes+len(data) > self.settings['max_image_bytes']:
                    raise ValueError('Live image budget exceeded or PNG encoding failed')
                payload = data.tobytes()
                self.db.execute('INSERT INTO image_assets VALUES (?,?,?)',
                    (row['id'], hashlib.sha256(payload).hexdigest(), sqlite3.Binary(payload)))
                self.images_bytes += len(payload)
        self.count += 1
        return row

    def close(self):
        self.db.close()


def query_live_memory(db, binding, request):
    allowed = {'operation','memory_id','session_id','generation','frame','as_of','position','radius',
               'heading','limit'}
    if not isinstance(request, dict) or set(request)-allowed or request.get('memory_id') != binding['memory_id']:
        raise ValueError('Query must bind the intended live memory')
    if request.get('operation') != 'query_observations':
        raise ValueError('Live memory currently exposes read-only observation queries')
    if (not isinstance(request.get('session_id'), str) or not request['session_id'] or
            type(request.get('generation')) is not int or request['generation'] < 0 or
            not isinstance(request.get('frame'), str) or not request['frame']):
        raise ValueError('Live local coordinates require a session, generation and frame')
    rows = query_observations(db, as_of=request['as_of'], session_id=request['session_id'],
        generation=request['generation'], frame=request['frame'], position=request.get('position'),
        radius=request.get('radius'), heading=request.get('heading'), limit=request.get('limit',20))
    return dict(memory_id=binding['memory_id'], as_of=request['as_of'], frame=request['frame'],
                session_id=request['session_id'], generation=request['generation'], observations=rows)
