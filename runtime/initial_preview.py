"""Live initial-pose preview, never a localization result or a TF transform."""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from perception.sensors.pointcloud import pointcloud_fields


def preview_points(cloud, pose, sensor_frame):
    """Project one raw LiDAR scan using a tentative map-from-LiDAR pose."""
    if cloud.header.frame_id.lstrip('/') != sensor_frame.lstrip('/'):
        raise ValueError('Preview scan frame differs from the configured LiDAR frame')
    if len(pose) != 6 or not np.isfinite(pose).all():
        raise ValueError('Preview pose must be finite roll, pitch, yaw, x, y, z')
    fields = pointcloud_fields(cloud, ('x', 'y', 'z'))
    points = np.column_stack([fields[axis] for axis in ('x', 'y', 'z')])
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('Preview scan contains no finite points')
    return np.asarray(Rotation.from_euler('xyz', pose[:3]).apply(points) + pose[3:], dtype=np.float32)


class PreviewState:
    """Bound confirmation to the current candidate and a recently rendered scan."""
    def __init__(self, max_age_s=1.):
        self.max_age_s = max_age_s
        self.pose = self.intent_stamp = None
        self.revision = 0
        self.cloud = self.received_mono = self.source_stamp = None
        self.rendered = None

    def select(self, pose, intent_stamp):
        if len(pose) != 6 or not np.isfinite(pose).all() or not math.isfinite(intent_stamp):
            raise ValueError('Invalid candidate pose')
        self.pose = list(pose)
        self.intent_stamp = float(intent_stamp)
        self.revision += 1
        self.rendered = None

    def scan(self, cloud, received_mono, *, now_source=None):
        stamp = cloud.header.stamp.to_sec()
        if not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(received_mono):
            raise ValueError('Invalid preview scan time')
        if now_source is not None and (not math.isfinite(now_source) or abs(now_source-stamp) > 1.5):
            raise ValueError('Preview scan measurement time is stale or in the future')
        if self.source_stamp is not None and stamp <= self.source_stamp:
            return False
        self.cloud, self.received_mono, self.source_stamp = cloud, received_mono, stamp
        return True

    def fresh(self, now):
        return self.cloud is not None and 0 <= now-self.received_mono <= self.max_age_s

    def ready(self, now):
        return (self.pose is not None and self.fresh(now) and self.rendered is not None and
                self.rendered[0] == self.revision and 0 <= now-self.rendered[2] <= self.max_age_s)
