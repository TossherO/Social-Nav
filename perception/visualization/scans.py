"""Observation-bound display geometry, independent of ROS transport and inference."""
from collections import deque
import math

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from perception.sensors.pointcloud import pointcloud_fields


SCAN_TOPIC = '/perception/visualization/scan'


def rotation_matrix(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ObservationOrder:
    """Ignore duplicates and late results, including a retired session/generation."""
    def __init__(self):
        self.identity, self.stamp = None, None
        self.retired = deque(maxlen=32)

    def accept(self, row):
        stamp = float(row['stamp_s'])
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError('Invalid display observation time')
        identity = (row['session_id'], row.get('sequence_id'), row['generation'])
        if identity in self.retired or identity == self.identity and stamp <= self.stamp:
            return False
        if self.identity is not None and identity != self.identity:
            self.retired.append(self.identity)
        self.identity, self.stamp = identity, stamp
        return True


class ScanCache:
    def __init__(self, size=16):
        self.rows = deque(maxlen=size)

    def add(self, cloud):
        stamp = cloud.header.stamp.to_sec()
        if self.rows and stamp < self.rows[-1][0]-.5:
            self.rows.clear()  # New replay clock, never reuse old-cycle raw scans.
        self.rows.append((stamp, cloud))

    def at(self, stamp, frame):
        matches = [cloud for t, cloud in self.rows if abs(t-stamp) <= 1e-6
                   and cloud.header.frame_id.lstrip('/') == frame.lstrip('/')]
        return matches[-1] if matches else None


def scan_geometry(cloud, row, sensor_only=False):
    """Use the exact observation's pose; absent/invalid pose yields no world cloud."""
    sensor_frame = row['detection_frame_id']
    frame = sensor_frame if sensor_only else row['frame_id']
    if cloud is None or row['health'] == 'invalid':
        return frame, np.empty((0, 3), dtype=np.float32)
    if (abs(cloud.header.stamp.to_sec()-row['stamp_s']) > 1e-6 or
            cloud.header.frame_id.lstrip('/') != sensor_frame.lstrip('/')):
        raise ValueError('Display cloud is not bound to this observation/frame')
    transform = None
    if not sensor_only:
        if row['health'] != 'valid' or row['world_from_lidar'] is None:
            return frame, np.empty((0, 3), dtype=np.float32)
        transform = validate_transform(row['world_from_lidar'])
    fields = pointcloud_fields(cloud, ('x', 'y', 'z'))
    points = np.column_stack([fields[axis] for axis in ('x', 'y', 'z')])
    points = points[np.isfinite(points).all(axis=1)]
    if transform is not None:
        points = points @ transform[:3, :3].T + transform[:3, 3]
    return frame, np.asarray(points, dtype=np.float32)


def pointcloud_message(points, frame, stamp=0):
    import rospy
    from sensor_msgs.msg import PointCloud2, PointField
    cloud = PointCloud2()
    cloud.header.frame_id, cloud.header.stamp = frame, rospy.Time.from_sec(stamp)
    points = np.asarray(points, dtype='<f4').reshape(-1, 3)
    cloud.height, cloud.width, cloud.point_step = 1, len(points), 12
    cloud.row_step = len(points)*12
    cloud.fields = [PointField(name=axis, offset=i*4, datatype=PointField.FLOAT32, count=1)
                    for i, axis in enumerate(('x', 'y', 'z'))]
    cloud.is_dense, cloud.data = True, points.tobytes()
    return cloud


def configure_rviz(configuration, transport, frame=None):
    """Resolve display topics from the same transport policy used by the node."""
    manager = configuration['Visualization Manager']
    if frame:
        manager['Global Options']['Fixed Frame'] = frame
    for display in manager['Displays']:
        if display.get('Marker Topic') == '/perception/people/markers':
            display['Marker Topic'] = transport['output_topic']+'/markers'
        if display.get('Image Topic') == '/perception/people/rgb':
            display['Image Topic'] = transport.get('people_image_topic',transport['output_topic']+'/rgb')
    return configuration
