"""Bounded sensor-only recording and local LIO for supervised stop measurements."""
import argparse
import json
from pathlib import Path
import queue
import signal
import threading
import time

import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, Imu
import yaml

from perception.mapping_localization.configuration import algorithm_config
from perception.sensors.observations import bag_observations
from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.runtime import LocalizationSession


def main(output):
    root = Path(output).resolve()
    config, _ = algorithm_config(mode='odometry')
    config_path = root/'algorithm.yaml'
    with config_path.open('x') as stream:
        yaml.safe_dump(config, stream)
    geometry = yaml.safe_load(Path('configs/wheelchair_vehicle.yaml').read_text())['geometry']
    lidar_from_base = np.linalg.inv(np.asarray(geometry['base_from_lidar'], dtype=float))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    began = time.monotonic()
    inbox = queue.Queue(24000)
    errors = []
    counts = dict(lidar=0, imu=0, scans=0, valid_scans=0)
    report = dict(scope='sensor_only_independent_local_lio_not_controller_truth',
                  started_mono=began, maximum_wall_s=90., success=False,
                  counts=counts, motor_commands_sent=False,
                  clock_basis='original_source_header_and_original_callback_receipt',
                  base_from_lidar=geometry['base_from_lidar'])
    rospy.init_node('social_nav_combined_stop_observer', anonymous=True, disable_signals=True)
    if rospy.get_param('/use_sim_time', False):
        raise RuntimeError('Real sensor clock required')
    def callback(message, topic):
        mono = time.monotonic()
        wall = time.time()
        try:
            inbox.put_nowait((topic, message, mono, wall))
        except queue.Full:
            errors.append('Sensor recording backlog exceeded');stop.set()
    listeners = [rospy.Subscriber(config['common']['lid_topic'], PointCloud2, callback,
                                 callback_args=config['common']['lid_topic'], queue_size=256,
                                 buff_size=8*1024*1024, tcp_nodelay=True),
                 rospy.Subscriber(config['common']['imu_topic'], Imu, callback,
                                  callback_args=config['common']['imu_topic'], queue_size=4096,
                                  tcp_nodelay=True)]
    timer = threading.Timer(90., stop.set)
    timer.daemon = True;timer.start()
    session = None
    try:
        session = LocalizationSession(config_path, mode='odometry', allow_local_only=True)
        with rosbag.Bag(str(root/'sensors.bag'), 'w', compression='lz4') as bag, \
                (root/'lidar_receipts.jsonl').open('x') as receipts, \
                (root/'localization.jsonl').open('x') as poses, \
                (root/'imu_samples.jsonl').open('x') as imu_samples:
            def messages():
                while not stop.is_set() and not rospy.is_shutdown():
                    try:
                        topic, message, mono, wall = inbox.get(timeout=.1)
                    except queue.Empty:
                        continue
                    bag.write(topic, message, t=rospy.Time.from_sec(wall))
                    lidar = topic == config['common']['lid_topic']
                    counts['lidar' if lidar else 'imu'] += 1
                    if lidar:
                        receipts.write(json.dumps(dict(stamp_ns=message.header.stamp.to_nsec(),
                            received_mono=mono, received_wall=wall))+'\n')
                    else:
                        g=message.angular_velocity;a=message.linear_acceleration
                        imu_samples.write(json.dumps(dict(stamp_ns=message.header.stamp.to_nsec(),
                            received_mono=mono,received_wall=wall,frame_id=message.header.frame_id,
                            gyro_xyz=[g.x,g.y,g.z],accel_xyz=[a.x,a.y,a.z]))+'\n')
                    yield topic, message, None
            consecutive = 0
            for batch in bag_observations(None, config, messages=messages()):
                acquired = time.monotonic()
                state, _, _, front, _ = session.step(batch)
                counts['scans'] += 1
                valid = state['health'] == 0
                consecutive = consecutive+1 if valid else 0
                base = None
                if valid:
                    counts['valid_scans'] += 1
                    world_from_base = pose_matrix(state['position'], state['quaternion_xyzw']) @ lidar_from_base
                    base = world_from_base[:3, 3].tolist()
                completed = time.monotonic()
                record = dict(state=state, base_position=base, acquired_mono=acquired,
                              completed_mono=completed, processing_ms=front['processing_ms'])
                poses.write(json.dumps(record, allow_nan=False)+'\n');poses.flush()
                if consecutive >= 10:
                    marker = root/'ready.tmp'
                    marker.write_text(json.dumps(dict(valid_consecutive=consecutive,
                        completed_mono=completed, source_stamp_s=state['stamp_s'], base_position=base)))
                    marker.replace(root/'ready.json')
                elif (root/'ready.json').exists():
                    (root/'ready.json').unlink()
                if (root/'STOP').exists():
                    stop.set()
            if errors:
                raise RuntimeError(errors[0])
            report['success'] = True
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
        (root/'observer_error.json').write_text(json.dumps(dict(error=report['error'],mono=time.monotonic()))+'\n')
    finally:
        stop.set();timer.cancel()
        if (root/'ready.json').exists():(root/'ready.json').unlink()
        for listener in listeners:
            listener.unregister()
        if session is not None:
            session.close()
        report.update(ended_mono=time.monotonic(), unconsumed_messages=inbox.qsize())
        (root/'observer_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    raise SystemExit(main(parser.parse_args().output))
