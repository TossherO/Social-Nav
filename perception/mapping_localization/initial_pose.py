"""Manual LiDAR-in-map seeds and their bag/map bindings; no ROS or SLAM."""
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .results import checked_pose, file_fingerprint, map_fingerprint


def quaternion_rpy(quaternion):
    x, y, z, w = quaternion
    r00, r10 = 1-2*(y*y+z*z), 2*(x*y+z*w)
    r20, r21, r22 = 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)
    cy = math.hypot(r00, r10)
    if cy > 1e-8:
        return [math.atan2(r21, r22), math.atan2(-r20, cy), math.atan2(r10, r00)]
    return [math.atan2(-2*(y*z-x*w), 1-2*(x*x+z*z)), math.atan2(-r20, cy), 0.0]


def rotation_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def save_seed(path, binding, position, quaternion):
    if checked_pose(position, quaternion) is None:
        raise ValueError('Initial pose must be finite with a unit quaternion')
    record = dict(binding, format='manual_localization_start', source='manual_map_selection',
                  registration_verified=False,
                  pose=dict(position_m=list(position), quaternion_xyzw=list(quaternion)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(name, str(path))  # Atomic, and never overwrite an existing annotation.
    finally:
        Path(name).unlink(missing_ok=True)
    return record


def load_seed(path, map_path, bag_path, start):
    """Read a seed only for the same map, bag and requested playback offset."""
    import yaml
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict) or record.get('format') != 'manual_localization_start':
        raise ValueError('Not a manual localization start file')
    try:
        if record['map']['fingerprint'] != map_fingerprint(map_path):
            raise ValueError('Initial pose belongs to a different map')
        if record['bag']['fingerprint'] != file_fingerprint(bag_path):
            raise ValueError('Initial pose belongs to a different bag')
        manifest = yaml.safe_load((Path(map_path) / 'manifest.yaml').read_text())
        if record['map']['frame_id'] != manifest['frame']:
            raise ValueError('Initial pose uses a different map frame')
        if record['observation']['frame_id'] != 'rslidar':
            raise ValueError('Initial pose must refer to the configured rslidar frame')
        if not math.isclose(float(record['observation']['start_offset_s']), start, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Use --start matching the annotated bag offset')
        pose = checked_pose(record['pose']['position_m'], record['pose']['quaternion_xyzw'])
        if pose is None:
            raise ValueError('Invalid initial position/quaternion')
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete initial pose record') from error
    position, quaternion = pose
    return quaternion_rpy(quaternion) + position.tolist()
