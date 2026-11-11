"""Live initial-pose preview, never a localization result or a TF transform."""
import math
import copy
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

from perception.sensors.pointcloud import pointcloud_fields


class PreviewGroundEstimator:
    """Bounded multi-scan floor estimate in the tilted LiDAR frame."""
    def __init__(self, roll, pitch, config, *, sample_period_s=.2, max_age_s=1.5):
        from perception.environment.ground import GroundTracker
        self.config=config
        self.settings=config['ground_tracking']
        self.tracker=GroundTracker(self.settings)
        self.rotation=Rotation.from_euler('xyz',[roll,pitch,0.])
        self.sample_period_s,self.max_age_s=sample_period_s,max_age_s
        self.last_attempt=self.source_stamp=self.last_sample_mono=self.measurement=None

    def scan(self, cloud, received_mono, *, sensor_frame):
        if cloud.header.frame_id.lstrip('/') != sensor_frame.lstrip('/'):
            raise ValueError('Ground estimate scan frame differs from configured LiDAR frame')
        stamp=float(cloud.header.stamp.to_sec())
        if self.source_stamp is not None and stamp <= self.source_stamp:return False
        if self.last_attempt is not None and received_mono-self.last_attempt < self.sample_period_s:return False
        if self.last_attempt is not None and received_mono-self.last_attempt > self.max_age_s:
            from perception.environment.ground import GroundTracker
            self.tracker=GroundTracker(self.settings)
            self.measurement=None
        self.last_attempt,self.source_stamp=received_mono,stamp
        fields=pointcloud_fields(cloud,('x','y','z'))
        points=np.column_stack([fields[k] for k in ('x','y','z')])
        points=points[np.isfinite(points).all(axis=1)]
        if not len(points):return False
        _,info=self.tracker.fit(self.rotation.apply(points),np.zeros(3),self.config)
        if info.get('status') == 'tracked' and info.get('reference_height_m') is not None:
            self.measurement=dict(status='available',height_m=float(info['reference_height_m']),
                source='current_lidar_ground_tracker',source_stamp_s=stamp,
                latest_scan_sha256=hashlib.sha256(cloud.data).hexdigest(),sensor_frame=cloud.header.frame_id,
                height_uncertainty_m=float(info['height_uncertainty_m']),
                inlier_points=info['inlier_points'],support_cells=info['support_cells'],
                observations=len(self.tracker.heights),reference_scope='local_ground_observation_not_calibration')
            self.last_sample_mono=received_mono
        else:
            # A failed current fit must not reuse a previously accepted floor.
            self.measurement=self.last_sample_mono=None
        return True

    def current(self, now):
        if self.measurement is None or not 0 <= now-self.last_sample_mono <= self.max_age_s:
            return None
        return copy.deepcopy(self.measurement)


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
        self.height_seed = None
        self.revision = 0
        self.cloud = self.received_mono = self.source_stamp = None
        self.rendered = None

    def select(self, pose, intent_stamp, *, height_seed=None):
        if len(pose) != 6 or not np.isfinite(pose).all() or not math.isfinite(intent_stamp):
            raise ValueError('Invalid candidate pose')
        self.pose = list(pose)
        self.height_seed = copy.deepcopy(height_seed)
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
