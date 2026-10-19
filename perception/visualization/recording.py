"""Union of recorded observations, without synthesizing missing localization."""
from dataclasses import replace

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform


def merge_people(result, people):
    rows = {row['stamp_s']: dict(row) for row in result.rows}
    if len(rows) != len(result.rows):
        raise ValueError('Duplicate localization observations; split the recorded sessions')
    for person in people.rows:
        stamp = person['stamp_s']
        if stamp not in rows:
            rows[stamp] = dict(stamp_s=stamp, health=1, status='DETECTION_ONLY',
                               reason='localization_not_recorded', has_map_pose=False)
        row = rows[stamp]
        row['_people'] = person
        if row.get('child_frame_id') and row['child_frame_id'] != person['detection_frame_id']:
            raise ValueError('Detection and localization refer to different scan frames')
        row['child_frame_id'] = person['detection_frame_id']
    return replace(result, rows=[rows[stamp] for stamp in sorted(rows)])


def sensor_frame(result):
    names = {row['child_frame_id'] for row in result.rows if row.get('child_frame_id')}
    if len(names) != 1:
        raise ValueError('Sensor view needs one explicitly recorded LiDAR frame')
    return next(iter(names))


def display_transform(result, row, frame, rotation_matrix):
    """Raw cloud follows the selected observation; no neighboring pose is borrowed."""
    if frame == 'lidar':
        return np.eye(4) if row.get('child_frame_id') else None
    pose = result.pose(row, frame)
    if row.get('health') != 0 or pose is None:
        return None
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = rotation_matrix(pose[1]), pose[0]
    person = row.get('_people')
    if person and person['health'] == 'valid' and person['world_from_lidar'] is not None:
        source = result.pose(row, 'odom')
        if source is None or person['frame_id'] != row.get('frame_id'):
            raise ValueError('Person pose does not belong to this recorded localization frame')
        local = np.eye(4)
        local[:3, :3], local[:3, 3] = rotation_matrix(source[1]), source[0]
        target = target @ np.linalg.inv(local) @ validate_transform(person['world_from_lidar'])
    return target
