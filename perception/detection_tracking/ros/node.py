"""ROS 1 Python 3 transport. Raw /odom is never subscribed as trusted localization."""
from __future__ import annotations

from collections import deque
import json
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..inference.config import load_config
from ..inference.detector import SimpleDet3dDetector
from ..inference.pipeline import Frame, PerceptionPipeline, to_snapshot
from ..inference.ros_inputs import decode_pointcloud, LocalizedPose, PoseBuffer, LatestSlot


class PeopleNode:
    def __init__(self):
        import rospy
        from sensor_msgs.msg import PointCloud2, Image
        from std_msgs.msg import String
        from social_nav_people.msg import People, TrackState, Detection
        self.rospy, self.String = rospy, String
        self.People, self.TrackState, self.Detection = People, TrackState, Detection
        self.config = load_config(rospy.get_param('~config', None), rospy.get_param('~local_config', None))
        self.cfg = self.config['ros']
        # Topic overrides are explicit; an empty launch arg does not erase a configured source.
        for key in ('pointcloud_topic', 'rgb_topic', 'output_topic', 'localization_topic'):
            value = rospy.get_param('~'+key, None)
            if value:
                self.cfg[key] = value
        if (self.cfg['localization_topic'] and
                self.cfg['localization_topic'] == self.cfg['raw_odometry_topic']):
            raise ValueError('Raw odometry cannot be configured as the localization source')
        for key in ('max_pose_age_s', 'max_rgb_age_s', 'max_input_age_s'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key] <= 0:
                raise ValueError(f'Invalid ROS time tolerance: {key}')
        self.detector = SimpleDet3dDetector(self.config['detector'])
        self.pipeline = PerceptionPipeline(self.detector, self.config['tracking'])
        self.poses, self.pending = PoseBuffer(), LatestSlot()
        self.rgb, self.rgb_lock = deque(maxlen=64), threading.Lock()
        self.last_received_wall = None
        self.last_output_stamp = None
        self.last_result_health = 'invalid'
        self.sequence_generation = 0
        self.clock_domain = 'ros_sim_time' if rospy.get_param('/use_sim_time', False) else 'ros_time'
        self.publisher = rospy.Publisher(self.cfg['output_topic'], People, queue_size=1)
        self.json_publisher = rospy.Publisher(self.cfg['output_topic']+'/json', String, queue_size=1)
        self.status_publisher = rospy.Publisher(self.cfg['output_topic']+'/status', String, queue_size=1)
        self.rgb_publisher = rospy.Publisher(self.cfg['output_topic']+'/rgb', Image, queue_size=1)
        self.subscribers = [rospy.Subscriber(self.cfg['pointcloud_topic'], PointCloud2, self.cloud_callback,
                                             queue_size=1, buff_size=32*1024*1024)]
        if self.cfg['rgb_topic']:
            self.subscribers.append(rospy.Subscriber(self.cfg['rgb_topic'], Image, self.rgb_callback, queue_size=2))
        if self.cfg['localization_topic']:
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
            classes = {'odometry': Odometry, 'pose_covariance': PoseWithCovarianceStamped, 'pose_stamped': PoseStamped}
            if self.cfg['localization_message'] not in classes:
                raise ValueError('Unknown localization_message type')
            self.subscribers.append(rospy.Subscriber(self.cfg['localization_topic'],
                classes[self.cfg['localization_message']], self.pose_callback, queue_size=32))
        else:
            rospy.logwarn('No localization topic: detections only, no world tracking. Raw odometry is NOT a fallback.')
        self.watchdog = rospy.Timer(rospy.Duration(0.25), self.status_callback)
        self.worker = threading.Thread(target=self.run, name='person_inference', daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.pending.close)

    def cloud_callback(self, message):
        self.last_received_wall = time.monotonic()
        self.pending.put(message)

    def rgb_callback(self, message):
        stamp = message.header.stamp.to_sec()
        if np.isfinite(stamp):
            with self.rgb_lock:
                self.rgb.append((stamp, message))

    def pose_callback(self, message):
        try:
            kind = self.cfg['localization_message']
            pose = message.pose if kind == 'pose_stamped' else message.pose.pose
            child = message.child_frame_id if kind == 'odometry' else self.cfg['localization_child_frame']
            if child != self.cfg['localization_child_frame']:
                raise ValueError('Localization child frame differs from configuration')
            q = pose.orientation
            quaternion = np.array([q.x, q.y, q.z, q.w])
            if not np.all(np.isfinite(quaternion)) or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
                raise ValueError('Invalid localization quaternion')
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
            self.poses.add(LocalizedPose(message.header.stamp.to_sec(), message.header.frame_id, child, matrix))
        except (ValueError, TypeError) as error:
            self.rospy.logwarn_throttle(2, f'Rejected localization: {error}')

    def status_callback(self, _):
        stale = self.last_received_wall is None or time.monotonic()-self.last_received_wall > self.cfg['max_input_age_s']
        output_stale = (self.last_output_stamp is None or
                        self.rospy.Time.now().to_sec()-self.last_output_stamp > self.cfg['max_input_age_s'])
        self.status_publisher.publish(self.String(data=json.dumps({
            'health': 'invalid' if stale or output_stale else self.last_result_health,
            'pointcloud_stale': stale, 'output_stale': output_stale,
            'last_output_stamp_s': self.last_output_stamp, 'dropped_busy': self.pending.dropped,
            'localization_topic_configured': bool(self.cfg['localization_topic']),
            'notice': 'Transport heartbeat is not localization accuracy or navigation safety verification',
        })))

    def run(self):
        rospy = self.rospy
        while not rospy.is_shutdown():
            message = self.pending.take()
            if message is None:
                continue
            stamp = message.header.stamp.to_sec()
            sensor_frame = message.header.frame_id
            if not np.isfinite(stamp) or stamp < 0 or not sensor_frame:
                self.last_result_health = 'invalid'
                rospy.logerr_throttle(2, 'Rejected cloud with invalid time/frame; no fabricated observation published')
                continue
            frame = Frame(f'{self.pipeline.session_id}/{stamp:.9f}',
                          f'{self.cfg["sequence_id"]}:{self.sequence_generation}', stamp,
                          np.empty((0, 4), dtype=np.float32), sensor_frame=sensor_frame,
                          clock_domain=self.clock_domain, world_frame=self.cfg['localization_parent_frame'])
            image = None
            pose_age = None
            try:
                if not np.isfinite(stamp) or not sensor_frame:
                    raise ValueError('Invalid cloud timestamp/frame')
                age = rospy.Time.now().to_sec()-stamp
                if age < -self.cfg['max_input_age_s'] or age > self.cfg['max_input_age_s']:
                    raise ValueError('Cloud outside configured ROS-clock freshness bounds')
                if self.pipeline.last_stamp is not None and stamp <= self.pipeline.last_stamp:
                    if self.pipeline.last_stamp-stamp > self.config['tracking']['reset_gap_s']:
                        self.sequence_generation += 1
                        frame.sequence_id = f'{self.cfg["sequence_id"]}:{self.sequence_generation}'
                        self.poses.clear()
                        with self.rgb_lock:
                            self.rgb.clear()
                    else:
                        raise ValueError('Duplicate/out-of-order cloud')
                frame.points = decode_pointcloud(message)
                pose, pose_age = self.poses.resolve(stamp, self.cfg['max_pose_age_s'],
                    self.cfg['localization_parent_frame'], sensor_frame, self.cfg['localization_child_frame'],
                    self.cfg['localization_child_from_lidar'])
                if pose is not None and self.cfg['localization_topic']:
                    frame.world_from_lidar, frame.pose_source, frame.pose_health = pose, 'localization', 'valid'
                    frame.pose_stamp_s = stamp-pose_age
                with self.rgb_lock:
                    images = [(t, im) for t, im in self.rgb if 0 <= stamp-t <= self.cfg['max_rgb_age_s']]
                if images:
                    frame.rgb_stamp_s, image = max(images, key=lambda pair: pair[0])
                result = self.pipeline.step(frame)
                to_snapshot(result)
            except Exception as error:
                result = self.pipeline.failure(frame, f'{type(error).__name__}: {error}')
                rospy.logerr_throttle(2, result['reason'])
            self.publish(result, pose_age, image)

    def publish(self, result, pose_age, image):
        rospy = self.rospy
        message = self.People()
        message.header.stamp = rospy.Time.from_sec(result['stamp_s'])
        message.header.frame_id = result['frame_id']
        for key in ('observation_id','sequence_id','session_id','generation','health',
                    'detection_frame_id','timestamp_source','input_fingerprint'):
            setattr(message, key, result[key])
        message.reason = result['reason'] or ''
        message.has_localization = result['world_from_lidar'] is not None
        if message.has_localization:
            message.world_from_lidar = np.asarray(result['world_from_lidar']).reshape(-1).tolist()
        message.has_pose_age = pose_age is not None
        message.pose_age_s = pose_age or 0.0
        message.dropped_busy = self.pending.dropped
        message.has_rgb = image is not None
        if image is not None:
            message.rgb_stamp = image.header.stamp
            self.rgb_publisher.publish(image)
        for box in result['detections']:
            detection = self.Detection()
            detection.box, detection.class_name = box, 'Pedestrian'
            message.detections.append(detection)
        for row in result['tracks']:
            track = self.TrackState()
            track.track_id, track.lifecycle_generation = row['track_id'], row['lifecycle_generation']
            track.state = track.OBSERVED if row['state'] == 'observed' else track.PREDICTED
            track.confirmed, track.box = row['confirmed'], row['box']
            track.has_velocity = row['velocity_xyz_mps'] is not None
            if track.has_velocity:
                track.velocity_xyz_mps = row['velocity_xyz_mps']
            track.covariance = row['covariance']
            track.last_observed_stamp = rospy.Time.from_sec(row['last_observed_stamp_s'])
            track.last_supported_stamp = rospy.Time.from_sec(row['last_supported_stamp_s'])
            track.support, track.box_geometry = row['support'], row['box_geometry']
            message.tracks.append(track)
        self.publisher.publish(message)
        self.json_publisher.publish(self.String(data=json.dumps(result, allow_nan=False)))
        self.last_output_stamp = result['stamp_s']
        self.last_result_health = result['health']


def main():
    import rospy
    rospy.init_node('social_nav_people')
    node = PeopleNode()
    rospy.spin()
    node.pending.close()


if __name__ == '__main__':
    main()
