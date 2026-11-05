"""View completed results with RViz and a small Qt playback controller; never runs SLAM."""
import argparse
from bisect import bisect_right
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

from perception.mapping_localization.results import Timeline, file_fingerprint, load_result, map_points
from .people import RecordedPeople, marker_array, place_specs
from .recording import merge_people, sensor_frame, display_transform
from .scans import rotation_matrix


ROOT = Path(__file__).resolve().parents[2]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Run directory, complete map, JSONL or comparison directory')
    parser.add_argument('--localization-map','--map',dest='map', help='Relocate the same 3D localization map referenced by a run')
    parser.add_argument('--phase', choices=('mapping', 'localization'), help='Select a comparison phase')
    parser.add_argument('--backend', help='Select a comparison series, e.g. ROLL, PIN, PIN_latest')
    parser.add_argument('--frame', choices=('auto', 'map', 'odom', 'lidar'), default='auto',
                        help='lidar: raw sensor view, including detections without localization')
    parser.add_argument('--play', action='store_true', help='Start playback; default is a static overview')
    parser.add_argument('--start', type=float, help='Initial elapsed acquisition seconds')
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--points', action='store_true', help='Show matched raw bag scans, rigidly placed at scan end; no deskew')
    parser.add_argument('--bag', help='Explicit original bag if moved, or for a live run which recorded no bag path')
    parser.add_argument('--backend-python', help='Prepared PIN interpreter, only to read PIN map geometry on CPU')
    parser.add_argument('--max-points', type=int, default=250000, help='Display point budget, not a source-data filter')
    parser.add_argument('--max-trace-points', type=int, default=20000)
    args = parser.parse_args(argv)
    if not math.isfinite(args.rate) or not .05 <= args.rate <= 32:
        parser.error('Playback rate must be between 0.05 and 32')
    if args.start is not None and (not math.isfinite(args.start) or args.start < 0):
        parser.error('Start must be finite and nonnegative')
    if args.max_points < 1 or args.max_trace_points < 2:
        parser.error('Display budgets must be positive (trajectory: at least two)')
    return args


class BagScans:
    def __init__(self, path, expected, topic):
        import rosbag
        if expected is not None and file_fingerprint(path) != expected:
            raise ValueError('Bag no longer matches the recorded input fingerprint')
        self.recording = rosbag.Bag(str(path), 'r')
        self.topic = topic
        first = next(self.recording.read_messages(topics=[topic]), None)
        if first is None:
            self.recording.close()
            raise ValueError('Original LiDAR topic is absent from this bag')
        self.offset = first[2].to_sec()-first[1].header.stamp.to_sec()
        self.cached = None

    def read(self, row):
        import rospy
        from perception.sensors.pointcloud import pointcloud_fields
        stamp = row['stamp_s']
        if self.cached is not None and self.cached[0] == stamp:
            return self.cached[1]
        best, error = None, .001  # Same-scan timestamp tolerance, not a future-pose fallback.
        for _, cloud, _ in self.recording.read_messages(topics=[self.topic],
                start_time=rospy.Time.from_sec(max(0, stamp+self.offset-.5)),
                end_time=rospy.Time.from_sec(stamp+self.offset+.5)):
            difference = abs(cloud.header.stamp.to_sec()-stamp)
            if difference < error:
                best, error = cloud, difference
        if best is None:
            self.cached = (stamp, None)
            return None
        if not row.get('child_frame_id') or best.header.frame_id.lstrip('/') != row['child_frame_id'].lstrip('/'):
            raise ValueError('Bag scan frame differs from the recorded LiDAR frame; no identity transform is assumed')
        fields = pointcloud_fields(best, ('x', 'y', 'z'))
        points = np.column_stack([fields[axis] for axis in ('x', 'y', 'z')])
        points = points[np.isfinite(points).all(axis=1)]
        self.cached = stamp, points
        return points

    def close(self):
        self.recording.close()


class Scene:
    def __init__(self, result, frame, points, scans, max_trace_points, max_points, people=None):
        import rospy
        from geometry_msgs.msg import Point
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import ColorRGBA
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage
        from visualization_msgs.msg import Marker, MarkerArray
        self.rospy, self.Point, self.PointCloud2, self.PointField = rospy, Point, PointCloud2, PointField
        self.Color, self.Marker, self.MarkerArray = ColorRGBA, Marker, MarkerArray
        self.result, self.frame = result, frame
        self.frame_name = sensor_frame(result) if frame == 'lidar' else result.frame_name(frame)
        self.scans, self.max_points = scans, max_points
        self.markers = rospy.Publisher('/result_view/markers', MarkerArray, queue_size=1, latch=True)
        self.cloud = rospy.Publisher('/result_view/map', PointCloud2, queue_size=1, latch=True)
        self.scan = rospy.Publisher('/result_view/scan', PointCloud2, queue_size=1, latch=True)
        # A display-only coordinate reference, never a robot/map-to-odom estimate.
        # Registering the chosen frame also keeps RViz's frame status meaningful.
        self.reference = rospy.Publisher('/tf_static', TFMessage, queue_size=1, latch=True)
        reference = TransformStamped()
        reference.header.frame_id = self.frame_name
        reference.child_frame_id = 'result_view_reference'
        reference.transform.rotation.w = 1
        self.reference.publish(TFMessage(transforms=[reference]))
        self.cloud.publish(self.pointcloud(points))
        segments = result.trajectory(frame, max_trace_points) if frame != 'lidar' else []
        self.segment_ends = [item[0] for item in segments]
        self.path_points, self.path_colors = [], []
        for _, a, b, health in segments:
            self.path_points += [Point(*a), Point(*b)]
            color = ColorRGBA(.65, .65, .65, 1) if health is None else ColorRGBA(1, .45, .1, 1)
            self.path_colors += [color, color]
        eligible = [result.pose(row, frame) for row in result.rows if row.get('health') in (0, None)] if frame != 'lidar' else []
        eligible = [pose for pose in eligible if pose is not None]
        self.endpoints = [pose[0] for pose in (eligible[0], eligible[-1])] if eligible else []
        self.last_display = None
        self.last_note = ''
        self.people = people
        if self.people is not None:
            self.people_publisher = rospy.Publisher('/result_view/people', MarkerArray, queue_size=1, latch=True)
            result.notices.append('定位与检测按观测时间合并；缺定位检测可用--frame lidar查看，不补世界位姿。')

    def pointcloud(self, points):
        cloud = self.PointCloud2()
        cloud.header.frame_id = self.frame_name
        points = np.asarray(points, dtype='<f4').reshape(-1, 3)
        cloud.height, cloud.width, cloud.point_step = 1, len(points), 12
        cloud.row_step = len(points)*12
        cloud.fields = [self.PointField(name=name, offset=i*4, datatype=self.PointField.FLOAT32, count=1)
                        for i, name in enumerate(('x', 'y', 'z'))]
        cloud.is_dense, cloud.data = True, points.tobytes()
        return cloud

    def marker(self, ident, kind):
        marker = self.Marker()
        marker.header.frame_id = self.frame_name
        marker.ns, marker.id, marker.type, marker.action = 'result_view', ident, kind, self.Marker.ADD
        marker.pose.orientation.w = 1
        marker.color.a = 1
        return marker

    def show(self, index, overview=False):
        if self.last_display == (index, overview):
            return self.last_note
        self.last_display = index, overview
        line = self.marker(0, self.Marker.LINE_LIST)
        line.scale.x = .06
        count = len(self.path_points) if overview else 2*bisect_right(self.segment_ends, index)
        line.points, line.colors = self.path_points[:count], self.path_colors[:count]
        if not line.points:
            line.action = self.Marker.DELETE
        endpoints = self.marker(1, self.Marker.SPHERE_LIST)
        endpoints.scale.x = endpoints.scale.y = endpoints.scale.z = .3
        endpoints.points = [self.Point(*p) for p in self.endpoints] if overview else []
        endpoints.colors = [self.Color(0, .9, .2, 1), self.Color(.3, .4, 1, 1)] if overview and self.endpoints else []
        if not endpoints.points:
            endpoints.action = self.Marker.DELETE
        current = self.marker(2, self.Marker.ARROW)
        current.scale.x, current.scale.y, current.scale.z = .8, .15, .15
        current.color.r, current.color.g = 1, .85
        row = self.result.rows[index] if index >= 0 else None
        pose = self.result.pose(row, self.frame) if row and self.frame != 'lidar' else None
        usable = row is not None and row.get('health') in (0, None) and pose is not None
        if usable:
            current.pose.position.x, current.pose.position.y, current.pose.position.z = pose[0]
            current.pose.orientation.x, current.pose.orientation.y, current.pose.orientation.z, current.pose.orientation.w = pose[1]
            if row.get('health') is None:
                current.color.r = current.color.g = current.color.b = .65
        else:
            current.action = self.Marker.DELETE
        self.markers.publish(self.MarkerArray(markers=[line, endpoints, current]))
        message = ''
        points = np.empty((0, 3), dtype=np.float32)
        transform = display_transform(self.result, row, self.frame, rotation_matrix) if row else None
        if self.scans is not None and transform is not None:
            try:
                raw = self.scans.read(row)
                if raw is None:
                    message = '此观测没有匹配的原始扫描。'
                else:
                    raw = raw[::max(1, math.ceil(len(raw)/self.max_points))]
                    points = raw @ transform[:3, :3].T + transform[:3, 3]
            except (OSError, ValueError) as error:
                message = '原始扫描未显示：' + str(error)
        if self.scans is not None:
            self.scan.publish(self.pointcloud(points))
        if self.people is not None:
            specs = []
            if row is not None and self.frame == 'lidar':
                specs = place_specs(self.people.at(row['stamp_s'], sensor_only=True),
                                    self.frame_name, self.frame_name, np.eye(4))
            elif usable and row.get('health') == 0:
                source = self.result.pose(row, 'odom')
                if source is not None:
                    target_from_source = np.eye(4)
                    if self.frame == 'map':
                        source_pose, target_pose = np.eye(4), np.eye(4)
                        source_pose[:3, :3], source_pose[:3, 3] = rotation_matrix(source[1]), source[0]
                        target_pose[:3, :3], target_pose[:3, 3] = rotation_matrix(pose[1]), pose[0]
                        target_from_source = target_pose @ np.linalg.inv(source_pose)
                    specs = place_specs(self.people.at(row['stamp_s']), row['frame_id'],
                                        self.frame_name, target_from_source)
            self.people_publisher.publish(marker_array(specs))
            if row and row.get('_people') and not usable and self.frame != 'lidar':
                message += ' 无同刻有效定位；使用--frame lidar查看传感器检测。'
        self.last_note = message
        return message


def make_window(QtWidgets, QtCore, QtGui, result, scene, args):
    class HealthStrip(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedHeight(12)
            self.setToolTip('观测状态：绿=有效，红=无效或降级，灰=未记录；空白处没有记录。')
            self.cache = None

        def paintEvent(self, event):
            width = max(1, self.width())
            if self.cache is None or self.cache[0] != width:
                levels = np.zeros(width, dtype=np.int8)
                for elapsed, row in zip(result.times, result.rows):
                    column = min(width-1, int((width-1)*elapsed/result.duration)) if result.duration else 0
                    level = 1 if row.get('health') == 0 else 2 if row.get('health') is None else 3
                    levels[column] = max(levels[column], level)
                self.cache = width, levels
            painter = QtGui.QPainter(self)
            colors = ('#eeeeee', '#35ac6c', '#999999', '#d84b4b')
            levels = self.cache[1]
            begin = 0
            for end in range(1, width+1):
                if end == width or levels[end] != levels[begin]:
                    painter.fillRect(begin, 0, end-begin, self.height(), QtGui.QColor(colors[levels[begin]]))
                    begin = end

    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Social-Nav 结果回看')
            self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
            self.resize(920, 240)
            self.timeline = Timeline(result.duration, args.rate)
            self.overview = not args.play and args.start is None
            self.timeline.seek(result.duration if self.overview else args.start or 0)
            if args.play:
                self.timeline.toggle()
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel(str(result.path))
            title.setTextFormat(QtCore.Qt.PlainText)
            title.setWordWrap(True)
            layout.addWidget(title)
            caption = '；'.join(result.notices)
            if scene.scans:
                caption += (' 雷达坐标视图，不显示世界轨迹；原始扫描未去畸变。' if scene.frame == 'lidar' else
                            ' 原始点云仅按同观测记录位姿刚性叠加，未去畸变。')
            self.caption = QtWidgets.QLabel(caption)
            self.caption.setWordWrap(True)
            layout.addWidget(self.caption)
            controls = QtWidgets.QHBoxLayout()
            self.toggle = QtWidgets.QPushButton('播放')
            self.toggle.setObjectName('toggle')
            self.toggle.clicked.connect(self.toggle_play)
            controls.addWidget(self.toggle)
            overview = QtWidgets.QPushButton('全览')
            overview.setObjectName('overview')
            overview.clicked.connect(self.show_overview)
            controls.addWidget(overview)
            controls.addWidget(QtWidgets.QLabel('速度'))
            self.rate = QtWidgets.QDoubleSpinBox()
            self.rate.setObjectName('rate')
            self.rate.setRange(.05, 32)
            self.rate.setSingleStep(.25)
            self.rate.setSuffix(' ×')
            self.rate.setValue(args.rate)
            self.rate.valueChanged.connect(lambda value: self.timeline.set_rate(value))
            controls.addWidget(self.rate)
            controls.addWidget(QtWidgets.QLabel('跳转（秒）'))
            self.seek = QtWidgets.QDoubleSpinBox()
            self.seek.setObjectName('seek')
            self.seek.setRange(0, result.duration)
            self.seek.setDecimals(3)
            self.seek.setSingleStep(.1)
            self.seek.valueChanged.connect(self.jump)
            controls.addWidget(self.seek)
            controls.addStretch()
            layout.addLayout(controls)
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setObjectName('timeline')
            self.slider.setRange(0, 10000)
            self.slider.sliderPressed.connect(lambda: self.timeline.tick())
            self.slider.valueChanged.connect(lambda value: self.jump(result.duration*value/10000))
            layout.addWidget(self.slider)
            layout.addWidget(HealthStrip())
            self.status = QtWidgets.QLabel()
            self.status.setObjectName('status')
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            for widget in (self.toggle, self.seek, self.slider):
                widget.setEnabled(bool(result.rows) and result.duration > 0)
            QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self, activated=self.toggle_play)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update_view)
            self.timer.start(33)
            self.update_view()

        def toggle_play(self):
            self.overview = False
            self.timeline.toggle()
            self.update_view()

        def jump(self, seconds):
            self.overview = False
            self.timeline.seek(seconds)
            self.update_view()

        def show_overview(self):
            self.timeline.playing = False
            self.timeline.seek(result.duration)
            self.overview = True
            self.update_view()

        def update_view(self):
            elapsed = self.timeline.tick()
            index = result.index_at(elapsed)
            for widget, value in ((self.slider, int(10000*elapsed/result.duration) if result.duration else 0),
                                  (self.seek, elapsed)):
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
            self.toggle.setText('暂停' if self.timeline.playing else '播放')
            try:
                extra = scene.show(index, self.overview)
            except (OSError, ValueError) as error:
                self.timeline.playing = False
                extra = str(error)
            if index < 0:
                self.status.setText('仅地图；没有已记录的位姿轨迹。')
            else:
                row = result.rows[index]
                health = {0:'有效', 1:'降级', 2:'无效', None:'未记录'}[row.get('health')]
                self.status.setText(('全览' if self.overview else '播放记录') +
                    ' | %.3f / %.3f 秒 | 观测 %.9f | 健康：%s | %s %s | 地图修正 %s %s' %
                    (elapsed, result.duration, row['stamp_s'], health, row.get('status', ''),
                     row.get('reason', ''), row.get('map_revision') if row.get('map_revision') is not None else '未记录', extra))
                self.status.setStyleSheet('color: #b02a2a' if row.get('health') in (1, 2) else '')
            self.status.setProperty('record_index', index)

    return Window()


def main(argv=None):
    args = arguments(argv)
    result = load_result(args.result, phase=args.phase, backend=args.backend, map_override=args.map)
    people = None
    directory = result.path if result.path.is_dir() else result.path.parent
    person_records = directory/'people.jsonl'
    if result.kind == 'recorded_states' and person_records.is_file():
        configuration = json.loads((ROOT/'perception/detection_tracking/configs/inference.json').read_text())
        people = RecordedPeople(person_records, configuration['visualization'])
        result = merge_people(result, people)
    frame = result.default_frame if args.frame == 'auto' else args.frame
    if args.start is not None and args.start > result.duration:
        raise ValueError('Start exceeds recorded duration')
    if args.play and not result.rows:
        raise ValueError('This map has no recorded trajectory to play')
    if frame == 'lidar':
        sensor_frame(result)
    if frame == 'odom' and not any(row.get('frame_id') for row in result.rows):
        raise ValueError('This result has no continuous odom poses')
    if frame == 'map' and not result.map_dir and not any(row.get('has_map_pose') for row in result.rows):
        raise ValueError('This result has no map-frame poses; select odom')
    points = map_points(result.map_dir, args.max_points, args.backend_python) if result.map_dir and frame == 'map' else np.empty((0, 3))
    scans = None
    if args.points:
        if result.kind in ('comparison_trajectory', 'optimized_keyframes', 'map_only'):
            raise ValueError('Raw scans require per-scan frame-bound records from a run, not sparse or comparison poses')
        bag = args.bag or result.metadata.get('bag')
        if not bag or not Path(bag).is_file():
            raise ValueError('--points requires the original reachable bag (use --bag if moved)')
        parameters = result.metadata.get('algorithm_parameters', {})
        topic = (parameters.get('stationary_imu', {}).get('raw_topic', '/rslidar_points')
                 if result.metadata.get('assume_stationary') else
                 parameters.get('common', {}).get('lid_topic', '/rslidar_points'))
        scans = BagScans(bag, result.metadata.get('playback', {}).get('bag_fingerprint'), topic)
        if not result.metadata.get('playback', {}).get('bag_fingerprint'):
            result.notices.append('旧结果无bag指纹，仅按记录的时间与传感器系绑定扫描。')
    from PyQt5 import QtWidgets, QtCore, QtGui
    from perception.mapping_localization.__main__ import stop
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('Result viewing needs a desktop/X11 display; run from a desktop terminal or enable X forwarding')
    app = QtWidgets.QApplication([])
    master = view = None
    old_handlers = {}
    with tempfile.TemporaryDirectory(prefix='social-nav-result-view-') as temporary:
        env = os.environ.copy()
        env.pop('ROS_HOSTNAME', None)
        os.environ.pop('ROS_HOSTNAME', None)
        with socket.socket() as selector:
            selector.bind(('127.0.0.1', 0))
            port = selector.getsockname()[1]
        env.update(ROS_MASTER_URI='http://127.0.0.1:%d' % port, ROS_IP='127.0.0.1', DISABLE_ROS1_EOL_WARNINGS='1',
                   ROS_HOME=temporary, ROS_LOG_DIR=temporary+'/logs')
        os.environ.update({key: env[key] for key in ('ROS_MASTER_URI', 'ROS_IP', 'ROS_HOME', 'ROS_LOG_DIR')})
        with (Path(temporary) / 'processes.log').open('w') as log:
            try:
                master = subprocess.Popen(['roscore', '-p', str(port)], env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                from xmlrpc.client import ServerProxy
                proxy = ServerProxy(env['ROS_MASTER_URI'])
                deadline = time.monotonic()+15
                while True:
                    try:
                        if proxy.getPid('/result_view')[0] == 1:
                            break
                    except OSError:
                        pass
                    if master.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Isolated result-view master did not start')
                    time.sleep(.1)
                import rospy
                import yaml
                rospy.init_node('result_view', anonymous=True, disable_signals=True)
                scene = Scene(result, frame, points, scans, args.max_trace_points, args.max_points, people)
                window = make_window(QtWidgets, QtCore, QtGui, result, scene, args)
                screen = app.primaryScreen().availableGeometry()
                window.move(screen.x()+15, max(screen.y()+15, screen.bottom()-window.sizeHint().height()-40))
                configuration = yaml.safe_load((ROOT / 'perception/visualization/configs/results.rviz').read_text())
                manager = configuration['Visualization Manager']
                manager['Global Options']['Fixed Frame'] = scene.frame_name
                configuration['Window Geometry'].update(X=screen.x()+10, Y=screen.y()+10,
                    Width=min(1280, screen.width()-30), Height=max(350, screen.height()-window.sizeHint().height()-100))
                geometry = points if len(points) else np.array([pose[0] for row in result.rows
                    if frame != 'lidar' and row.get('health') in (0, None)
                    for pose in [result.pose(row, frame)] if pose is not None])
                if frame == 'lidar':
                    manager['Views']['Current']['Distance'] = 20
                if len(geometry):
                    low, high = geometry.min(axis=0), geometry.max(axis=0)
                    camera = manager['Views']['Current']
                    camera['Focal Point'] = dict(zip(('X', 'Y', 'Z'), map(float, (low+high)/2)))
                    camera['Distance'] = max(10.0, float((high-low).max())*1.25)
                if not window.overview and frame != 'lidar' and result.rows:
                    row = result.rows[result.index_at(window.timeline.position)]
                    pose = result.pose(row, frame) if row.get('health') in (0, None) else None
                    if pose is not None:
                        camera = manager['Views']['Current']
                        camera['Focal Point'] = dict(zip(('X', 'Y', 'Z'), map(float, pose[0])))
                        camera['Distance'] = 20.0
                config_path = Path(temporary) / 'results.rviz'
                config_path.write_text(yaml.safe_dump(configuration, allow_unicode=True))
                view = subprocess.Popen(['rosrun', 'rviz', 'rviz', '-d', str(config_path), '__name:=result_rviz'],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                for signum in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[signum] = signal.signal(signum, lambda *_: window.close())
                window.show()
                result_code = [0]
                monitor = QtCore.QTimer(window)
                def check_processes():
                    if master.poll() is not None or view.poll() is not None:
                        result_code[0] = 1 if master.poll() is not None or view.returncode else 0
                        if result_code[0]:
                            import sys
                            print('Result display process exited: ' +
                                  (Path(temporary) / 'processes.log').read_text()[-3000:], file=sys.stderr)
                        window.close()
                monitor.timeout.connect(check_processes)
                monitor.start(500)
                code = app.exec_()
                return code or result_code[0]
            finally:
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
                try:
                    stop(view, first_signal=signal.SIGTERM)
                finally:
                    if 'rospy' in locals():
                        rospy.signal_shutdown('Result view closed')
                    try:
                        stop(master)
                    finally:
                        if scans is not None:
                            scans.close()


if __name__ == '__main__':
    raise SystemExit(main())
