"""ROS 1 PIN worker: raw LiDAR only, acquisition-bound health and one TF chain."""
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import torch
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse
from tf2_msgs.msg import TFMessage
from social_nav_localization.msg import LocalizationState

from .core import PinSlam
from .inputs import decode_scan


def fill_pose(destination, matrix):
    destination.position.x, destination.position.y, destination.position.z = matrix[:3, 3]
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    destination.orientation.x, destination.orientation.y, destination.orientation.z, destination.orientation.w = q


def transform(stamp, parent, child, matrix):
    value = TransformStamped()
    value.header.stamp, value.header.frame_id, value.child_frame_id = stamp, parent, child
    value.transform.translation.x, value.transform.translation.y, value.transform.translation.z = matrix[:3, 3]
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    value.transform.rotation.x, value.transform.rotation.y, value.transform.rotation.z, value.transform.rotation.w = q
    return value


class Node:
    def __init__(self):
        self.mode = rospy.get_param('~mode')
        self.map_path = Path(rospy.get_param('~map_dir'))
        self.source = rospy.get_param('~input_source', 'live_lidar')
        self.map_frame, self.odom_frame, self.lidar_frame = 'map', 'odom', 'rslidar'
        self.lock = threading.RLock()
        self.busy, self.saved, self.fatal = False, False, ''
        self.received, self.rejected = 0, 0
        self.registration_rejections = {}
        self.processing_times_s = []
        self.started = time.monotonic()
        self.core = PinSlam(rospy.get_param('~config'), self.mode,
                            rospy.get_param('~initial_pose'), self.map_path)
        rospy.set_param('~algorithm_parameters', self.core.parameters())
        self.publisher = rospy.Publisher('/localization/state', LocalizationState, queue_size=64)
        # Pure Python TFMessage avoids loading system-Python-only tf2/PyKDL bindings.
        self.tf = rospy.Publisher('/tf', TFMessage, queue_size=64)
        self.map_pub = rospy.Publisher('~map', PointCloud2, queue_size=1, latch=True)
        self.status_service = rospy.Service('~status', Trigger, self.status)
        self.save_service = rospy.Service('~save_map', Trigger, self.save)
        self.subscriber = rospy.Subscriber('/rslidar_points', PointCloud2, self.on_scan,
                                            queue_size=64, buff_size=64*1024*1024)
        rospy.on_shutdown(self.shutdown)

    def status(self, _request):
        # Lock-free counters allow a caller to detect an in-flight warm-up scan.
        samples = list(self.processing_times_s)
        timing = ({'count': len(samples), 'total_s': sum(samples), 'mean_ms': float(np.mean(samples)*1000),
                   'p50_ms': float(np.percentile(samples, 50)*1000), 'p95_ms': float(np.percentile(samples, 95)*1000),
                   'max_ms': max(samples)*1000} if samples else {})
        return TriggerResponse(not bool(self.fatal), json.dumps(dict(
            busy=self.busy, received=self.received, processed=self.core.frames,
            registered=self.core.valid_frames, lost=self.core.lost_frames, rejected=self.rejected,
            map_revision=self.core.map_revision, neural_points=self.core.neural.count(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            processing=timing,
            registration_rejections=dict(self.registration_rejections),
            last_registration=self.core.last_registration,
            map_point_frames_after_pgo=self.core.neural.after_pgo,
            final_map_refinement=self.core.final_map_refinement,
            saved=self.saved, fatal=self.fatal, wall_s=time.monotonic()-self.started)))

    def save(self, _request):
        with self.lock:
            if self.mode != 'mapping':
                return TriggerResponse(False, 'Fixed-map mode is read-only')
            if self.saved:
                return TriggerResponse(True, 'Map already completed')
            try:
                self.core.save(self.map_path, self.source)
                self.saved = True
                return TriggerResponse(True, 'Complete neural map saved')
            except Exception as error:
                rospy.logerr('PIN map save failed: %s', error)
                return TriggerResponse(False, str(error))

    def shutdown(self):
        if self.mode == 'mapping' and not self.saved and not self.fatal:
            response = self.save(None)
            if not response.success:
                rospy.logerr(response.message)

    def on_scan(self, cloud):
        with self.lock:
            if self.saved or self.fatal:
                return
            self.busy = True
            frame_started = time.monotonic()
            self.received += 1
            try:
                try:
                    if cloud.header.frame_id != self.lidar_frame:
                        raise ValueError('Unexpected LiDAR frame; no calibrated frame substitution')
                    scan = decode_scan(cloud, deskew=self.core.config.deskew)
                except ValueError as error:
                    self.rejected += 1
                    result = self.core.invalid(cloud.header.stamp.to_sec(), str(error))
                else:
                    result = self.core.process(scan)
                    if result.status == 'LOST':
                        count = self.registration_rejections.get(result.reason, 0)
                        self.registration_rejections[result.reason] = count+1
                        if count == 0:
                            rospy.logwarn('PIN registration rejected at %.9f: %s',
                                scan.stamp_s, json.dumps(self.core.last_registration))
                # The registration/mapping paths synchronize their GPU kernels upstream.
                torch.cuda.synchronize()
                self.processing_times_s.append(time.monotonic()-frame_started)
                self.publish(cloud.header.stamp, result)
                if result.valid and self.core.frames % 20 == 0:
                    coordinates = self.core.neural.neural_points.detach().cpu().numpy()[::5]
                    self.map_pub.publish(create_cloud_xyz32(Header(stamp=cloud.header.stamp, frame_id=self.map_frame), coordinates))
            except Exception as error:
                self.fatal = str(error)
                rospy.logerr('PIN failed; stopping without marking its map complete: %s', error)
                self.publish(cloud.header.stamp, self.core.invalid(cloud.header.stamp.to_sec(), 'ALGORITHM_EXCEPTION'))
                rospy.signal_shutdown('PIN algorithm exception')
            finally:
                self.busy = False

    def publish(self, stamp, result):
        state = LocalizationState()
        state.header.stamp, state.header.frame_id = stamp, self.odom_frame
        state.child_frame_id, state.source = self.lidar_frame, 'pin_slam'
        state.session_id, state.generation = self.core.session_id, self.core.generation
        state.map_revision = result.map_revision
        state.status, state.reason = result.status, result.reason
        state.health = state.VALID if result.valid else state.INVALID
        # Unknown covariance/matching statistics remain explicitly absent.
        state.has_covariance = state.has_matching_quality = False
        state.has_map_pose = result.valid
        state.map_frame = self.map_frame
        if result.valid:
            fill_pose(state.pose, result.local_pose)
            fill_pose(state.map_pose, result.map_pose)
            map_from_odom = result.map_pose @ np.linalg.inv(result.local_pose)
            self.tf.publish(TFMessage([transform(stamp, self.map_frame, self.odom_frame, map_from_odom),
                                       transform(stamp, self.odom_frame, self.lidar_frame, result.local_pose)]))
        self.publisher.publish(state)


def main():
    rospy.init_node('pin_slam')
    Node()
    rospy.spin()


if __name__ == '__main__':
    main()
