"""Pick a bag's initial LiDAR pose on a ROLL map, with first-scan/RGB previews; no SLAM."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

import numpy as np
from perception.sensors.image import rgb_array

from .__main__ import ROOT, project_path, stop
from .initial_pose import quaternion_rpy, rotation_matrix, save_seed
from .results import file_fingerprint, map_fingerprint, read_pcd, rpy_quaternion


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--localization-map','--map',dest='map', required=True, help='Complete ROLL 3D localization map bundle')
    parser.add_argument('--bag', required=True, help='Read one scan and its preceding RGB; never replay the bag')
    parser.add_argument('--out', help='New JSON annotation; default: BAG.initial_pose.json beside the bag')
    parser.add_argument('--start', type=float, default=0.0, help='Select the first scan after this recorded-time offset')
    parser.add_argument('--lidar-topic', help='Default: current FAST-LIO lidar topic')
    parser.add_argument('--rgb-topic', help='Default: current detection RGB topic; empty string disables RGB')
    parser.add_argument('--max-points', type=int, default=250000, help='Map display point budget')
    args = parser.parse_args(argv)
    args.map, args.bag = project_path(args.map), project_path(args.bag)
    args.out = project_path(args.out) if args.out else args.bag.with_suffix('.initial_pose.json')
    if not math.isfinite(args.start) or args.start < 0 or args.max_points < 1:
        parser.error('Start must be finite/nonnegative; display point budget must be positive')
    if not args.bag.is_file():
        parser.error('Bag does not exist')
    if not (args.map / 'COMPLETE').is_file() or not (args.map / 'GlobalMap.pcd').is_file():
        parser.error('Select a complete ROLL map; import legacy maps first')
    if args.out.exists() or args.out.is_symlink():
        parser.error('Annotation exists; use --out with a new filename')
    return args


def read_first_frame(args):
    import rosbag
    import genpy
    from perception.sensors.pointcloud import pointcloud_fields
    print('读取首帧及相邻RGB（只读，不回放bag）……', flush=True)
    with rosbag.Bag(str(args.bag), 'r') as recording:
        begin = recording.get_start_time()
        if args.start > recording.get_end_time()-begin:
            raise ValueError('Start is outside this recording')
        cutoff = genpy.Time.from_sec(begin + args.start) if args.start else None
        first = next(recording.read_messages(topics=[args.lidar_topic], start_time=cutoff), None)
        if first is None:
            raise ValueError('No LiDAR scan at/after the requested start')
        _, cloud, recorded = first
        if cloud._type != 'sensor_msgs/PointCloud2' or cloud.header.stamp.to_nsec() <= 0:
            raise ValueError('Expected a PointCloud2 with a positive observation timestamp')
        if cloud.header.frame_id.lstrip('/') != 'rslidar':
            raise ValueError('This picker records the current rslidar pose; no other-frame transform is assumed')
        fields = pointcloud_fields(cloud, ('x', 'y', 'z'))
        points = np.column_stack([fields[axis] for axis in ('x', 'y', 'z')])
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError('First scan has no finite XYZ points')
        rgb = None
        if args.rgb_topic:
            for _, image, _ in recording.read_messages(topics=[args.rgb_topic],
                    start_time=genpy.Time.from_sec(max(begin, recorded.to_sec()-1)),
                    end_time=genpy.Time.from_sec(recorded.to_sec()+1)):
                stamp = image.header.stamp.to_nsec()
                age = cloud.header.stamp.to_nsec()-stamp
                if 0 <= age <= 500000000 and (rgb is None or stamp > rgb.header.stamp.to_nsec()):
                    rgb = image
        observation = dict(start_offset_s=args.start, lidar_topic=args.lidar_topic,
                           frame_id=cloud.header.frame_id.lstrip('/'),
                           lidar_stamp_ns=cloud.header.stamp.to_nsec(), record_stamp_ns=recorded.to_nsec(),
                           rgb_topic=args.rgb_topic, rgb_stamp_ns=rgb.header.stamp.to_nsec() if rgb else None)
    return points, rgb, observation


def first_scan_config(frame, points):
    """Independent sensor-frame view, with camera tools only and no map pose."""
    distance = max(10.0, float(np.percentile(np.linalg.norm(points, axis=1), 95))*2.5)
    return {
        'Preferences': {'PromptSaveOnExit': False}, 'Panels': [],
        'Visualization Manager': {
            'Class': '',
            'Global Options': {'Fixed Frame': frame, 'Background Color': '28; 28; 28'},
            'Displays': [
                {'Class': 'rviz/PointCloud2', 'Name': 'First scan', 'Enabled': True,
                 'Topic': '/start_picker/first_scan', 'Queue Size': 1, 'Style': 'Points',
                 'Size (Pixels)': 2, 'Color Transformer': 'AxisColor', 'Axis': 'Z'},
                {'Class': 'rviz/Axes', 'Name': 'Sensor axes', 'Enabled': True,
                 'Reference Frame': frame, 'Length': 1, 'Radius': .025}],
            'Tools': [{'Class': 'rviz/MoveCamera'}],
            'Views': {'Current': {'Class': 'rviz/Orbit', 'Pitch': .8, 'Yaw': -float(np.pi/2),
                                  'Distance': distance, 'Focal Point': {'X': 0, 'Y': 0, 'Z': 0}}}}}


def make_window(QtWidgets, QtCore, QtGui, args, frame, points, rgb, binding, publish, scan_config):
    class Window(QtWidgets.QWidget):
        picked = QtCore.pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.setWindowTitle('起点标注 · ' + args.bag.name)
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel('RViz中选择 2D Pose Estimate，点击位置并拖动朝向。\n'
                                    '青色首帧叠加为人工预览；调整坐标后点击保存。')
            layout.addWidget(title)
            tabs = QtWidgets.QTabWidget()
            layout.addWidget(tabs)
            # A separate RViz manager keeps raw sensor viewing independent of map selection.
            from rviz import bindings as rviz
            scan = QtWidgets.QWidget()
            scan_layout = QtWidgets.QVBoxLayout(scan)
            scan_layout.setContentsMargins(0, 0, 0, 0)
            self.scan_view = rviz.VisualizationFrame()
            self.scan_view.setSplashPath('')
            self.scan_view.initialize(str(scan_config))
            self.scan_view.setMenuBar(None)
            self.scan_view.setStatusBar(None)
            self.scan_view.setHideButtonVisibility(False)
            for dock in self.scan_view.findChildren(QtWidgets.QDockWidget):
                dock.hide()
            self.scan_view.setMinimumSize(520, 390)
            scan_layout.addWidget(self.scan_view)
            scan_layout.addWidget(QtWidgets.QLabel('左键旋转 · 滚轮／右键缩放 · 中键／Shift+左键平移'))
            buttons = QtWidgets.QHBoxLayout()
            for label, overview in (('俯视', False), ('全览', True)):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _, fit=overview: self.reset_scan_view(fit))
                buttons.addWidget(button)
            scan_layout.addLayout(buttons)
            tabs.addTab(scan, '首帧雷达（三维）')
            image_label = QtWidgets.QLabel('首帧之前500毫秒内没有可用RGB。')
            image_label.setMinimumSize(520, 390)
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            if rgb is not None:
                try:
                    image = rgb_array(rgb)
                    h, w = image.shape[:2]
                    image = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format_RGB888).copy()
                    image_label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(520, 390, QtCore.Qt.KeepAspectRatio))
                except ValueError as error:
                    image_label.setText(str(error))
            tabs.addTab(image_label, '对应RGB（仅作场景参考）')
            observation = binding['observation']
            layout.addWidget(QtWidgets.QLabel('雷达观测时间：%.9f；地图系：%s；雷达系：%s' %
                (observation['lidar_stamp_ns']/1e9, frame, observation['frame_id'])))
            grid = QtWidgets.QGridLayout()
            self.values, self.selected = [], False
            for i, name in enumerate(('X（米）', 'Y（米）', 'Z（米）', 'Roll（度）', 'Pitch（度）', 'Yaw（度）')):
                value = QtWidgets.QDoubleSpinBox()
                value.setDecimals(4)
                value.setRange(-1000000, 1000000) if i < 3 else value.setRange(-180, 180)
                value.setSingleStep(.1 if i < 3 else 1)
                value.setEnabled(False)
                value.valueChanged.connect(self.preview)
                self.values.append(value)
                grid.addWidget(QtWidgets.QLabel(name), i//3*2, i%3)
                grid.addWidget(value, i//3*2+1, i%3)
            layout.addLayout(grid)
            self.status = QtWidgets.QLabel('尚未选点；未设置任何默认起点。')
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            self.save = QtWidgets.QPushButton('保存初值并退出')
            self.save.setEnabled(False)
            self.save.clicked.connect(self.save_pose)
            layout.addWidget(self.save)
            self.picked.connect(self.select)

        def reset_scan_view(self, overview):
            camera = self.scan_view.getManager().getViewManager().getCurrent()
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low+high)/2 if overview else np.zeros(3)
            distance = (max(10.0, float((high-low).max())*1.5) if overview else
                        max(10.0, float(np.percentile(np.linalg.norm(points, axis=1), 95))*2.5))
            for name, value in (('Pitch', .8 if overview else 1.56), ('Yaw', -float(np.pi/2)),
                                ('Distance', distance)):
                camera.subProp(name).setValue(value)
            for name, value in zip(('X', 'Y', 'Z'), center):
                camera.subProp('Focal Point').subProp(name).setValue(float(value))

        def closeEvent(self, event):
            self.scan_view.getManager().stopUpdate()
            super().closeEvent(event)

        def select(self, message):
            from .results import checked_pose
            p, q = message.pose.pose.position, message.pose.pose.orientation
            quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
            norm = np.linalg.norm(quaternion)
            if (message.header.frame_id.lstrip('/') != frame.lstrip('/') or
                    not np.isfinite(norm) or norm < 1e-8 or
                    checked_pose([p.x, p.y, p.z], quaternion/norm) is None):
                self.status.setText('拒绝无效选点：请使用当前地图坐标系和有限位姿。')
                return
            distances = np.linalg.norm(keyframes[:, :2]-[p.x, p.y], axis=1)
            height = float(keyframes[np.argmin(distances), 2])
            values = [p.x, p.y, height] + list(np.degrees(quaternion_rpy(quaternion/norm)))
            for spin, value in zip(self.values, values):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
                spin.setEnabled(True)
            self.selected = True
            self.save.setEnabled(True)
            self.preview()

        def pose(self):
            values = [spin.value() for spin in self.values]
            return values[:3], rpy_quaternion(*np.radians(values[3:]))

        def preview(self):
            if not self.selected:
                return
            position, quaternion = self.pose()
            publish(position, quaternion)
            self.status.setText('人工初值，未经配准验证。Z取最近XY关键帧，可手动修改；\n'
                                '重叠楼层／倾斜地图请同时核对Z、Roll、Pitch。')

        def save_pose(self):
            try:
                position, quaternion = self.pose()
                save_seed(args.out, binding, position, quaternion)
            except (OSError, ValueError) as error:
                self.status.setText('保存失败：' + str(error))
                return
            print('已保存：%s\n对应--pose：%s' %
                  (args.out, ' '.join('%.9g' % value for value in quaternion_rpy(quaternion)+position)), flush=True)
            self.close()

    keyframes = np.loadtxt(args.map / 'keyframes/poses.txt', ndmin=2)[:, 1:4]
    if not len(keyframes) or not np.isfinite(keyframes).all():
        raise ValueError('Map keyframe positions are empty/nonfinite')
    return Window()


def main(argv=None):
    args = arguments(argv)
    import yaml
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('请从桌面终端或启用X11转发后运行选点脚本。')
    settings = yaml.safe_load((ROOT / 'perception/mapping_localization/configs/fast_lio.yaml').read_text())
    args.lidar_topic = args.lidar_topic or settings['common']['lid_topic']
    if args.rgb_topic is None:
        args.rgb_topic = json.loads((ROOT / 'perception/detection_tracking/configs/inference.json').read_text())['ros']['rgb_topic']
    frame = yaml.safe_load((args.map / 'manifest.yaml').read_text()).get('frame')
    if not isinstance(frame, str) or not frame:
        raise ValueError('Map has no recorded coordinate frame')
    map_cloud = read_pcd(args.map / 'GlobalMap.pcd', args.max_points)
    if not len(map_cloud):
        raise ValueError('Map has no finite display points')
    points, rgb, observation = read_first_frame(args)
    binding = dict(map=dict(path=str(args.map), frame_id=frame, fingerprint=map_fingerprint(args.map)),
                   bag=dict(path=str(args.bag), fingerprint=file_fingerprint(args.bag)), observation=observation)
    from PyQt5 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication([])
    master = view = None
    handlers = {}
    with tempfile.TemporaryDirectory(prefix='social-nav-pick-start-') as temporary:
        env = dict(os.environ, ROS_HOME=temporary, ROS_LOG_DIR=temporary+'/logs', ROS_IP='127.0.0.1',
                   DISABLE_ROS1_EOL_WARNINGS='1')
        env.pop('ROS_HOSTNAME', None)
        os.environ.pop('ROS_HOSTNAME', None)
        with socket.socket() as selector:
            selector.bind(('127.0.0.1', 0))
            port = selector.getsockname()[1]
        env['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % port
        os.environ.update({key: env[key] for key in
                           ('ROS_MASTER_URI', 'ROS_IP', 'ROS_HOME', 'ROS_LOG_DIR', 'DISABLE_ROS1_EOL_WARNINGS')})
        with (Path(temporary) / 'processes.log').open('w') as log:
            try:
                master = subprocess.Popen(['roscore', '-p', str(port)], env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                from xmlrpc.client import ServerProxy
                deadline = time.monotonic()+15
                while True:
                    try:
                        if ServerProxy(env['ROS_MASTER_URI']).getPid('/start_picker')[0] == 1:
                            break
                    except OSError:
                        pass
                    if master.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Isolated selection master did not start')
                    time.sleep(.1)
                import rospy
                from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
                from sensor_msgs.msg import PointCloud2
                from sensor_msgs import point_cloud2
                from std_msgs.msg import Header
                from tf2_msgs.msg import TFMessage
                rospy.init_node('start_picker', anonymous=True, disable_signals=True)
                reference = rospy.Publisher('/tf_static', TFMessage, queue_size=1, latch=True)
                transform = TransformStamped()
                transform.header.frame_id, transform.child_frame_id = frame, 'start_picker_reference'
                transform.transform.rotation.w = 1
                sensor_reference = TransformStamped()
                sensor_reference.header.frame_id = observation['frame_id']
                sensor_reference.child_frame_id = 'start_picker_sensor_reference'
                sensor_reference.transform.rotation.w = 1
                reference.publish(TFMessage(transforms=[transform, sensor_reference]))
                map_pub = rospy.Publisher('/start_picker/map', PointCloud2, queue_size=1, latch=True)
                raw_pub = rospy.Publisher('/start_picker/scan', PointCloud2, queue_size=1, latch=True)
                first_pub = rospy.Publisher('/start_picker/first_scan', PointCloud2, queue_size=1, latch=True)
                pose_pub = rospy.Publisher('/start_picker/pose', PoseStamped, queue_size=1, latch=True)
                def cloud(xyz):
                    return point_cloud2.create_cloud_xyz32(Header(frame_id=frame), xyz)
                map_pub.publish(cloud(map_cloud))
                first_pub.publish(point_cloud2.create_cloud_xyz32(Header(frame_id=observation['frame_id']), points))
                def publish(position, quaternion):
                    raw_pub.publish(cloud(points @ rotation_matrix(quaternion).T + position))
                    pose = PoseStamped(header=Header(frame_id=frame))
                    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
                    (pose.pose.orientation.x, pose.pose.orientation.y,
                     pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
                    pose_pub.publish(pose)
                scan_config = Path(temporary) / 'first_scan.rviz'
                scan_config.write_text(yaml.safe_dump(first_scan_config(observation['frame_id'], points)))
                window = make_window(QtWidgets, QtCore, QtGui, args, frame, points, rgb, binding, publish, scan_config)
                subscriber = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped,
                                              lambda message: window.picked.emit(message), queue_size=1)
                configuration = yaml.safe_load((ROOT / 'perception/visualization/configs/results.rviz').read_text())
                manager = configuration['Visualization Manager']
                manager['Global Options']['Fixed Frame'] = frame
                manager['Displays'] = [display for display in manager['Displays'] if display['Class'] != 'rviz/MarkerArray']
                for display in manager['Displays']:
                    if display['Class'] == 'rviz/PointCloud2':
                        display['Topic'] = '/start_picker/map' if display['Name'] == 'Saved map' else '/start_picker/scan'
                        display['Name'] = 'Map' if display['Name'] == 'Saved map' else 'Manual first-scan preview'
                manager['Displays'].append(dict(Class='rviz/Pose', Name='Manual seed', Enabled=True,
                                                Topic='/start_picker/pose', Shape='Arrow'))
                manager['Tools'].append(dict(Class='rviz/SetInitialPose', Topic='/initialpose'))
                low, high = map_cloud.min(axis=0), map_cloud.max(axis=0)
                manager['Views']['Current'].update(Pitch=1.45, Yaw=-float(np.pi/2),
                    Distance=max(10.0, float((high-low).max())*1.25),
                    **{'Focal Point': dict(zip(('X', 'Y', 'Z'), map(float, (low+high)/2)))})
                screen = app.primaryScreen().availableGeometry()
                configuration['Window Geometry'].update(X=screen.x(), Y=screen.y(),
                    Width=max(640, screen.width()-580), Height=max(500, screen.height()-60))
                config = Path(temporary) / 'pick_start.rviz'
                config.write_text(yaml.safe_dump(configuration))
                view = subprocess.Popen(['rosrun', 'rviz', 'rviz', '-d', str(config), '__name:=start_picker_rviz'],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, lambda *_: window.close())
                window.move(max(screen.x(), screen.right()-560), screen.y()+20)
                window.show()
                code = [0]
                def check_processes():
                    if master.poll() is not None or view.poll() is not None:
                        code[0] = 1 if master.poll() is not None or view.returncode else 0
                        if code[0]:
                            print((Path(temporary) / 'processes.log').read_text()[-3000:], flush=True)
                        window.close()
                monitor = QtCore.QTimer(window)
                monitor.timeout.connect(check_processes)
                monitor.start(400)
                return app.exec_() or code[0]
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
                try:
                    if 'window' in locals():
                        import sip
                        window.scan_view.getManager().stopUpdate()
                        sip.delete(window.scan_view)
                finally:
                    try:
                        stop(view, first_signal=signal.SIGTERM)
                    finally:
                        if 'rospy' in locals():
                            rospy.signal_shutdown('Selection closed')
                        stop(master)


if __name__ == '__main__':
    raise SystemExit(main())
