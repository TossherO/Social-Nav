"""Read-only live display: bound scans and optional people, with wall-time expiry."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np

from perception.detection_tracking.inference.config import load_config
from .people import PeopleScene, marker_array
from .scans import ObservationOrder, ScanCache, SCAN_TOPIC, configure_rviz, pointcloud_message, scan_geometry, rotation_matrix


def localization_observation(message):
    pose = None
    if message.health == message.VALID:
        p, q = message.pose.position, message.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w])
        if not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-3):
            raise ValueError('Invalid localization display quaternion')
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation_matrix(quaternion), [p.x, p.y, p.z]
    return dict(stamp_s=message.header.stamp.to_sec(), frame_id=message.header.frame_id,
                detection_frame_id=message.child_frame_id, session_id=message.session_id,
                generation=message.generation, health='valid' if pose is not None else 'invalid',
                world_from_lidar=pose, source=message.source)


def main():
    import rospy
    import yaml
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from social_nav_localization.msg import LocalizationState
    rospy.init_node('perception_visualization')
    config = load_config(rospy.get_param('~config', None), rospy.get_param('~local_config', None))
    transport, cfg = config['ros'], config['visualization']
    for key in ('pointcloud_topic', 'localization_source', 'localization_topic'):
        value = rospy.get_param('~'+key, '')
        if value:
            transport[key] = value
    mode = rospy.get_param('~mode', 'people')
    sensor_only = rospy.get_param('~sensor_only', False)
    if mode not in ('people', 'localization'):
        raise ValueError('Display mode must be people or localization')
    if mode == 'localization' and not rospy.get_param('~pointcloud_topic', ''):
        transport['pointcloud_topic'] = rospy.get_param('/stationary_imu/raw_topic') if rospy.get_param(
            '/stationary_imu/enabled', False) else rospy.get_param('/common/lid_topic', transport['pointcloud_topic'])
    if not math.isfinite(cfg['stale_s']) or cfg['stale_s'] <= 0:
        raise ValueError('Invalid display freshness budget')
    topic = transport['output_topic']
    cloud_pub = rospy.Publisher(SCAN_TOPIC, PointCloud2, queue_size=1, latch=True)
    markers = rospy.Publisher(topic+'/markers', MarkerArray, queue_size=1, latch=True) if mode == 'people' else None
    scene = PeopleScene(cfg['history_s'], cfg['max_history_points'])
    cache, order = ScanCache(), ObservationOrder()
    lock, stopped = threading.RLock(), threading.Event()
    current, received = [None], [None]
    world_frame = transport['localization_parent_frame']
    display_frame = [rospy.get_param('~sensor_frame', 'rslidar') if sensor_only else world_frame]

    def clear():
        current[0], received[0] = None, None
        scene.clear()
        cloud_pub.publish(pointcloud_message([], display_frame[0]))
        if markers is not None:
            markers.publish(marker_array([]))

    def publish_scan(row):
        cloud = cache.at(row['stamp_s'], row['detection_frame_id'])
        frame, points = scan_geometry(cloud, row, sensor_only)
        # Unknown world geometry must not be placed through an old sensor TF.
        frame = frame if sensor_only or row['health'] == 'valid' else world_frame
        if not len(points):
            # A clear is a display command, not an estimate at an invalid scan
            # time (there may be no map->odom TF at that time).
            cloud_pub.publish(pointcloud_message([], display_frame[0]))
            return
        display_frame[0] = frame
        cloud_pub.publish(pointcloud_message(points, frame, row['stamp_s']))

    def on_cloud(message):
        with lock:
            cache.add(message)
            row = current[0]
            if row is not None and abs(row['stamp_s']-message.header.stamp.to_sec()) <= 1e-6:
                try:
                    publish_scan(row)
                except (ValueError, TypeError, KeyError) as error:
                    clear()
                    rospy.logwarn_throttle(2, 'Display scan cleared: %s', error)

    def on_result(message):
        with lock:
            try:
                row = json.loads(message.data) if mode == 'people' else localization_observation(message)
                if mode == 'localization' and row['source'] != transport['localization_source']:
                    return
                age = rospy.Time.now().to_sec()-row['stamp_s']
                if not math.isfinite(age) or age < -cfg['stale_s'] or age > cfg['stale_s']:
                    return  # A stale delayed result must not clear a newer display.
                if not order.accept(row):
                    return
                if markers is not None:
                    specs = scene.update(row, sensor_only=sensor_only)
                    if specs is None:
                        return
                    if not sensor_only and (row['health'] != 'valid' or row['world_from_lidar'] is None):
                        specs = []
                    markers.publish(marker_array(specs, cfg['stale_s']))
                current[0], received[0] = row, time.monotonic()
                publish_scan(row)
            except (ValueError, KeyError, TypeError) as error:
                clear()
                rospy.logwarn_throttle(2, 'Perception display cleared: %s', error)

    clear()
    subscribers = [rospy.Subscriber(transport['pointcloud_topic'], PointCloud2, on_cloud,
                                    queue_size=4, buff_size=32*1024*1024)]
    subscribers.append(rospy.Subscriber(topic+'/json' if mode == 'people' else transport['localization_topic'],
        String if mode == 'people' else LocalizationState, on_result, queue_size=8))
    rospy.on_shutdown(stopped.set)
    view = None
    with tempfile.TemporaryDirectory(prefix='social-nav-live-view-') as temporary:
        try:
            if rospy.get_param('~rviz', False):
                configuration = yaml.safe_load((Path(__file__).parent/'configs/people.rviz').read_text())
                frame = rospy.get_param('~sensor_frame', 'rslidar') if sensor_only else world_frame
                configure_rviz(configuration, transport, frame)
                path = Path(temporary)/'people.rviz'
                path.write_text(yaml.safe_dump(configuration))
                view = subprocess.Popen(['rosrun', 'rviz', 'rviz', '-d', str(path)], start_new_session=True)
            while not rospy.is_shutdown() and not stopped.wait(.05):
                with lock:
                    if received[0] is not None and (time.monotonic()-received[0] > cfg['stale_s'] or
                            rospy.Time.now().to_sec()-current[0]['stamp_s'] > cfg['stale_s']):
                        clear()
                if view is not None and view.poll() is not None:
                    rospy.loginfo('RViz window closed; display adapter remains active until ROS shutdown')
                    view = None
        finally:
            if view is not None and view.poll() is None:
                os.killpg(view.pid, signal.SIGTERM)
                try:
                    view.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(view.pid, signal.SIGKILL)
                    view.wait()
            with lock:
                clear()


if __name__ == '__main__':
    main()
