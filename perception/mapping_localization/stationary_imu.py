"""Explicit stationary-assumption inertial input; never publishes a robot pose.

The virtual IMU shares the LiDAR axes. Specific force, not zero acceleration,
is assumed constant; timestamps come from the observed scan interval.
"""
import json
import math
import threading

import numpy as np

from perception.sensors.pointcloud import pointcloud_fields


SOURCE_SUFFIX = '_stationary_assumption'


def localization_source(backend, assume_stationary=False):
    return backend + (SOURCE_SUFFIX if assume_stationary else '')


class StationaryImuClock:
    def __init__(self, rate_hz=200.0, max_scan_s=0.2, max_gap_s=0.5):
        if (not all(math.isfinite(v) and v > 0 for v in (rate_hz, max_scan_s, max_gap_s))
                or not 10 <= rate_hz <= 1000 or max_scan_s > 1 or max_gap_s < max_scan_s):
            raise ValueError('Invalid stationary IMU sampling or scan budget')
        self.period_ns = round(1e9 / rate_hz)
        self.max_scan_s, self.max_gap_s = max_scan_s, max_gap_s
        self.last_end_ns = None

    def sample_times(self, cloud):
        end_ns = cloud.header.stamp.to_nsec()
        raw = pointcloud_fields(cloud, ('timestamp',))['timestamp'].astype(np.float64)
        finite = raw[np.isfinite(raw)]
        if end_ns <= 0 or not len(finite):
            raise ValueError('Missing valid acquisition time for stationary IMU')
        begin_s, last_s = float(finite.min()), float(finite.max())
        duration = end_ns / 1e9 - begin_s
        if not 0 < duration <= self.max_scan_s or abs(last_s-end_ns/1e9) > 0.005:
            raise ValueError('Expected absolute per-point seconds ending at cloud header')
        begin_ns = round(begin_s * 1e9)
        if self.last_end_ns is not None:
            if not 0 < (end_ns-self.last_end_ns)/1e9 <= self.max_gap_s:
                raise ValueError('Scan time reversed, repeated or exceeded stationary input gap')
            begin_ns = max(begin_ns, self.last_end_ns + self.period_ns)
        samples = list(range(begin_ns, end_ns, self.period_ns))
        if not samples or samples[-1] != end_ns:
            samples.append(end_ns)
        self.last_end_ns = end_ns
        return samples


def main():
    import rospy
    from rosgraph import Master
    from sensor_msgs.msg import Imu, PointCloud2
    from std_msgs.msg import String
    from std_srvs.srv import Trigger, TriggerResponse

    rospy.init_node('stationary_imu_input')
    if not rospy.get_param('/stationary_imu/enabled', False):
        raise RuntimeError('Stationary assumption must be explicitly enabled')
    cfg = rospy.get_param('/stationary_imu')
    raw_topic = rospy.resolve_name(cfg['raw_topic'])
    cloud_topic = rospy.resolve_name(rospy.get_param('/common/lid_topic'))
    imu_topic = rospy.resolve_name(rospy.get_param('/common/imu_topic'))
    if len({raw_topic, cloud_topic, imu_topic}) != 3:
        raise ValueError('Raw, adapted cloud and synthetic IMU topics must be distinct')
    existing = dict(Master(rospy.get_name()).getSystemState()[0])
    if existing.get(cloud_topic) or existing.get(imu_topic):
        raise RuntimeError('Adapted topics already have publishers; do not mix inertial sources')
    force = tuple(float(v) for v in cfg['specific_force_m_s2'])
    if len(force) != 3 or not all(math.isfinite(v) for v in force) or not 9 < np.linalg.norm(force) < 11:
        raise ValueError('Stationary specific force must include a finite gravity-scale vector')
    clock = StationaryImuClock(cfg['rate_hz'], cfg['max_scan_s'], cfg['max_gap_s'])
    lock = threading.Lock()
    stats = dict(source='stationary_assumption', scans=0, imu_samples=0,
                 last_scan_ns=None, fatal_error=None, orientation_measured=False,
                 imu_frame=cfg['imu_frame'], specific_force_m_s2=force,
                 raw_topic=raw_topic, cloud_topic=cloud_topic, imu_topic=imu_topic)
    cloud_pub = rospy.Publisher(cloud_topic, PointCloud2, queue_size=4)
    imu_pub = rospy.Publisher(imu_topic, Imu, queue_size=256)
    status_pub = rospy.Publisher('/stationary_imu/input_status', String, queue_size=1, latch=True)

    def status(_):
        with lock:
            return TriggerResponse(success=stats['fatal_error'] is None, message=json.dumps(stats))

    def on_cloud(cloud):
        with lock:
            try:
                if cloud.header.frame_id != cfg['lidar_frame']:
                    raise ValueError('LiDAR frame differs from the explicitly assumed IMU axes')
                times = clock.sample_times(cloud)
                for stamp_ns in times:
                    message = Imu()
                    message.header.stamp = rospy.Time(stamp_ns//10**9, stamp_ns%10**9)
                    message.header.frame_id = cfg['imu_frame']
                    message.orientation.w = 1
                    message.orientation_covariance[0] = -1  # No measured orientation.
                    # Zero gyro; covariance zero means unknown, not noise-free measurement.
                    message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = force
                    imu_pub.publish(message)
                cloud_pub.publish(cloud)  # Same raw bytes/header; no fixed pose injection.
                stats.update(scans=stats['scans']+1, imu_samples=stats['imu_samples']+len(times),
                             last_scan_ns=cloud.header.stamp.to_nsec())
            except (ValueError, TypeError) as error:
                stats['fatal_error'] = str(error)
                status_pub.publish(String(data=json.dumps(stats)))
                rospy.logfatal('Stationary IMU input rejected: %s', error)
                rospy.signal_shutdown(str(error))
    subscriber = rospy.Subscriber(raw_topic, PointCloud2, on_cloud, queue_size=4, buff_size=32*1024*1024)
    service = rospy.Service('/stationary_imu/status', Trigger, status)
    status_pub.publish(String(data=json.dumps(stats)))
    rospy.logwarn('EXPERIMENTAL stationary assumption: synthetic IMU, no measured inertial motion or orientation')
    rospy.spin()


if __name__ == '__main__':
    main()
