"""Read-only bag replay for an isolated LI-Init pure-LiDAR motion reference.

Requires the private upstream build with calibration/lidar_only and pose_file.
Original point timestamps and IMU samples are preserved. Only the published
PointCloud2 header changes from scan end to actual first point for LI-Init.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import xmlrpc.client

import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Imu, PointCloud2

from perception.sensors.pointcloud import pointcloud_fields


PARAMETERS = {
    'use_sim_time': False,
    'calibration/lidar_only': True,
    'common/lid_topic': '/calibration/lidar',
    'common/imu_topic': '/calibration/imu',
    'max_iteration': 4, 'point_filter_num': 2, 'cube_side_length': 200.,
    'preprocess/lidar_type': 6, 'preprocess/scan_line': 16,
    'preprocess/blind': .8, 'preprocess/feature_extract_en': False,
    'initialization/cut_frame': True, 'initialization/cut_frame_num': 3,
    'initialization/orig_odom_freq': 10, 'initialization/mean_acc_norm': 9.81,
    'initialization/data_accum_length': 500., 'initialization/online_refine_time': 0.,
    'initialization/Rot_LI_cov': [.00005] * 3,
    'initialization/Trans_LI_cov': [.0001] * 3,
    'mapping/filter_size_surf': .10, 'mapping/filter_size_map': .15,
    'mapping/det_range': 40., 'mapping/acc_cov': .5, 'mapping/gyr_cov': .5,
    'mapping/grav_cov': .001, 'mapping/b_acc_cov': .0001, 'mapping/b_gyr_cov': .0001,
    'publish/path_en': False, 'publish/scan_publish_en': False,
    'publish/dense_publish_en': False, 'publish/scan_bodyframe_pub_en': False,
    'runtime_pos_log_enable': False, 'pcd_save/pcd_save_en': False,
}


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New CSV path')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--rate', type=float, default=2.)
    parser.add_argument('--subframes', type=int, choices=(1, 3), default=1,
                        help='Use full scans for sparse 16-beam scenes; split mode is diagnostic')
    args = parser.parse_args()
    if args.rate <= 0 or not np.isfinite(args.rate):
        parser.error('rate must be positive and finite')
    bag_path, output, binary = args.bag.resolve(), args.output.resolve(), args.binary.resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stat = bag_path.stat()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    master = node = None
    with tempfile.TemporaryDirectory(prefix='liinit-replay-') as tmp:
        os.environ.update(ROS_MASTER_URI='http://127.0.0.1:%d' % port,
                          ROS_IP='127.0.0.1', ROS_HOME=tmp, ROS_LOG_DIR=tmp + '/roslog')
        with open(tmp + '/master.log', 'w') as master_log, open(tmp + '/node.log', 'w') as node_log:
            try:
                master = subprocess.Popen(['roscore', '-p', str(port)], stdout=master_log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                server = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        if server.getPid('/calibration')[0] == 1:
                            break
                    except (OSError, xmlrpc.client.Error):
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('Isolated ROS master did not start')
                rospy.init_node('calibration_replay', disable_signals=True)
                config = dict(PARAMETERS, **{'calibration/pose_file': str(output)})
                config['initialization/cut_frame_num'] = args.subframes
                for key, value in config.items():
                    rospy.set_param('/' + key, value)
                node = subprocess.Popen([str(binary), '__name:=li_calibration'], stdout=node_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                cloud_pub = rospy.Publisher('/calibration/lidar', PointCloud2, queue_size=128)
                imu_pub = rospy.Publisher('/calibration/imu', Imu, queue_size=10000)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if node.poll() is not None:
                        raise RuntimeError('Upstream node exited before replay')
                    if cloud_pub.get_num_connections() and imu_pub.get_num_connections():
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Upstream subscribers did not connect')
                counts = {'scans': 0, 'imu': 0}; last_stamp = None; next_progress = 10.
                with rosbag.Bag(str(bag_path), 'r') as bag:
                    start = bag.get_start_time(); wall = time.monotonic()
                    for topic, message, receipt in bag.read_messages(topics=['/rslidar_points', '/camera/imu']):
                        wait = wall + (receipt.to_sec() - start) / args.rate - time.monotonic()
                        if wait > .001:
                            time.sleep(wait)
                        if node.poll() is not None:
                            raise RuntimeError('Upstream node stopped during replay')
                        if topic == '/rslidar_points':
                            points = pointcloud_fields(message, ('timestamp',))['timestamp']
                            times = points[np.isfinite(points) & (points > 0)]
                            if len(times) == 0:
                                raise ValueError('No absolute per-point times')
                            last_stamp = float(times.max())
                            if not 0 <= last_stamp - float(times.min()) <= .2:
                                raise ValueError('Unexpected scan time span')
                            message.header.stamp = rospy.Time.from_sec(float(times.min()))
                            cloud_pub.publish(message); counts['scans'] += 1
                        else:
                            imu_pub.publish(message); counts['imu'] += 1
                        relative = receipt.to_sec() - start
                        if relative >= next_progress:
                            print(bag_path.parent.name, bag_path.name, 'bag+%.1fs' % relative, flush=True)
                            next_progress += 10
                    duration = bag.get_end_time() - start
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    if output.exists():
                        lines = output.read_text().splitlines()
                        if len(lines) > 2 and float(lines[-1].split(',')[0]) >= last_stamp - .06:
                            break
                    if node.poll() is not None:
                        raise RuntimeError('Upstream node stopped before drain')
                    time.sleep(.2)
                else:
                    raise RuntimeError('LiDAR reference did not drain to the final scan')
                stop(node); node = None
                record = dict(bag=str(bag_path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                              duration_s=duration, record_start_s=start, input_messages=counts,
                              pose_count=len(lines) - 1, parameters=config, binary=str(binary),
                              source='LI-Init FAST-LO, forced LiDAR-only; IMU used for synchronization gate only',
                              time_convention='pose at subscan end; original point times preserved',
                              frame='first lidar scan', wall_seconds=time.monotonic() - wall)
                output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
                print('completed',output,'poses',len(lines) - 1,flush=True)
            except BaseException:
                node_log.flush()
                print(Path(tmp + '/node.log').read_text()[-6000:], flush=True)
                raise
            finally:
                stop(node)
                rospy.signal_shutdown('calibration replay complete')
                stop(master)


if __name__ == '__main__':
    main()
