"""RViz review on private display topics; no TF, goals or velocity commands."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .footprint import VehicleGeometry
from .inputs import load_config, load_input
from .planner import se2, transform_poses
from .replay import resolve

TOPIC = '/navigation/planning/review'
FRAME = 'navigation_review'  # View-only coordinates relative to the selected start.


def messages(row, geometry, scan=None):
    import rospy
    from nav_msgs.msg import OccupancyGrid, Path as RosPath
    from geometry_msgs.msg import PoseStamped, Point
    from visualization_msgs.msg import Marker, MarkerArray
    from perception.visualization.scans import pointcloud_message
    from .visualization import display_scan
    source, result = row['source'], row['result']
    scene, asset = load_input(scene=resolve(source['scene']) if source.get('scene') else None,
                             static_map=resolve(source['static_map']) if source.get('static_map') else None)
    local_pose = (geometry.pose(scene.record['localization']['world_from_lidar'])
                  if scene is not None and scene.record['localization']['status'] == 'valid' else None)
    has_global = asset is not None and 'start' in result and result['request']['mode'] in ('static', 'mapped')
    reference = result.get('start', local_pose)
    base_from_reference = np.linalg.inv(se2(reference)) if reference is not None else np.eye(3)
    base_from_local = np.linalg.inv(se2(local_pose)) if local_pose is not None else None
    stamp = rospy.Time.from_sec(scene.record['as_of'] if scene else 0.)

    def grid_message(grid, transform):
        message = OccupancyGrid(); message.header.frame_id = FRAME; message.header.stamp = stamp
        message.info.origin.orientation.w = 1.; message.info.resolution = .1
        message.info.width = message.info.height = 1; message.data = [-1]
        if grid is not None:
            m = grid.metadata; origin = transform @ np.r_[m['origin_xy'], 1.]
            yaw = np.arctan2(transform[1, 0], transform[0, 0])
            message.info.width = m['shape'][1]; message.info.height = m['shape'][0]
            message.info.resolution = m['resolution']
            message.info.origin.position.x, message.info.origin.position.y = origin[:2]
            message.info.origin.orientation.z = np.sin(yaw/2); message.info.origin.orientation.w = np.cos(yaw/2)
            message.data = grid.layers['occupancy'].ravel().tolist()
        return message

    path = RosPath(); path.header.frame_id = FRAME; path.header.stamp = stamp
    poses = result.get('path', [])
    if poses:
        for x, y, yaw in transform_poses(poses, base_from_reference):
            pose = PoseStamped(); pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z, pose.pose.orientation.w = np.sin(yaw/2), np.cos(yaw/2)
            path.poses.append(pose)
    clear = Marker(); clear.action = Marker.DELETEALL
    markers = MarkerArray(markers=[clear])
    foot = Marker(); foot.header = path.header; foot.ns = 'assumed_footprint'; foot.id = 1
    foot.type = Marker.LINE_STRIP; foot.action = Marker.ADD; foot.pose.orientation.w = 1.
    foot.scale.x = .035; foot.color.r = 1.; foot.color.g = .35; foot.color.a = 1.
    foot.points = [Point(x=float(x), y=float(y), z=.08) for x, y in np.vstack((geometry.footprint, geometry.footprint[:1]))]
    if reference is not None: markers.markers.append(foot)
    if path.poses:
        route = Marker(); route.header = path.header; route.ns = 'reference'; route.id = 3
        route.type = Marker.LINE_STRIP; route.pose.orientation.w = 1.; route.scale.x = .05; route.color.a = 1.
        if result['current_path_valid']: route.color.b = 1.; route.color.g = .4
        else: route.color.r = .9; route.color.g = .55
        route.points = [Point(x=pose.pose.position.x, y=pose.pose.position.y, z=.12) for pose in path.poses]
        markers.markers.append(route)
    prefix=result.get('current_local_path',{}).get('poses',[])
    if prefix and base_from_local is not None:
        local=Marker();local.header=path.header;local.ns='verified_local_reference';local.id=4
        local.type=Marker.LINE_STRIP;local.pose.orientation.w=1.;local.scale.x=.07;local.color.g=.8;local.color.a=1.
        local.points=[Point(x=float(x),y=float(y),z=.15) for x,y,yaw in transform_poses(prefix,base_from_local)]
        markers.markers.append(local)
    text = Marker(); text.header = path.header; text.ns = 'status'; text.id = 2; text.type = Marker.TEXT_VIEW_FACING
    text.pose.orientation.w = 1.; text.pose.position.z = 1.8; text.scale.z = .25
    text.color.r = .1; text.color.g = .16; text.color.b = .22; text.color.a = 1.
    text.text = result['status']+' | executable=false'
    markers.markers.append(text)
    if scan is None and source.get('run'): scan = display_scan(source)
    points = np.empty((0, 3), np.float32)
    if scan is not None and base_from_local is not None:
        points = scan['points'].copy()
        points[:, :2] = points[:, :2] @ base_from_local[:2, :2].T+base_from_local[:2, 2]
    return dict(static=grid_message(asset.snapshot if has_global else None, base_from_reference),
                local=grid_message(scene.geometry if base_from_local is not None else None, base_from_local), path=path,
                markers=markers, scan=pointcloud_message(points, FRAME, stamp.to_sec()))


def run(results_path):
    import rospy
    import rosgraph
    import yaml
    from nav_msgs.msg import OccupancyGrid, Path as RosPath
    from visualization_msgs.msg import MarkerArray
    from sensor_msgs.msg import PointCloud2
    from PyQt5 import QtWidgets, QtCore
    from .visualization import display_scan
    rosgraph.Master('/navigation_planning_review').getPid()
    rospy.init_node('navigation_planning_review', anonymous=True, disable_signals=True)
    rows = [r for r in json.loads(Path(results_path).read_text())['queries'] if 'result' in r]
    geometry = VehicleGeometry.historical(load_config()['geometry_reference'])
    publishers = {k:rospy.Publisher(TOPIC+'/'+k, kind, queue_size=1, latch=True) for k, kind in
                  [('static', OccupancyGrid), ('local', OccupancyGrid), ('path', RosPath),
                   ('markers', MarkerArray), ('scan', PointCloud2)]}
    app = QtWidgets.QApplication([]); window = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(window)
    layout.addWidget(QtWidgets.QLabel('只读回看：所选起点居中；切帧清空旧路径；不是机器人TF或运动控制。'))
    choice = QtWidgets.QComboBox(); choice.addItems([r['label'] for r in rows]); layout.addWidget(choice)
    status = QtWidgets.QLabel(); status.setWordWrap(True); layout.addWidget(status)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider.setRange(0, len(rows)-1); layout.addWidget(slider)
    play = QtWidgets.QPushButton('播放／暂停'); layout.addWidget(play); timer = QtCore.QTimer()
    cache = {}
    def show(index):
        choice.blockSignals(True); slider.blockSignals(True); choice.setCurrentIndex(index); slider.setValue(index)
        choice.blockSignals(False); slider.blockSignals(False)
        row = rows[index]; key = (row['source'].get('scene'), row['source'].get('run'))
        r = row['result']; status.setText('%s | %s | current_path_valid=%s | source=%s' % (
            r['status'], r.get('reason') or r.get('local_check', {}).get('status', ''),
            r.get('current_path_valid', False), r.get('frame_id', 'unavailable')))
        if key not in cache: cache[key] = display_scan(row['source'])
        for name, message in messages(row, geometry, cache[key]).items(): publishers[name].publish(message)
    choice.currentIndexChanged.connect(show); slider.valueChanged.connect(show)
    timer.timeout.connect(lambda: show((choice.currentIndex()+1) % len(rows)))
    play.clicked.connect(lambda: timer.stop() if timer.isActive() else timer.start(1800))
    with tempfile.TemporaryDirectory(prefix='navigation-rviz-') as directory:
        config = {'Visualization Manager': {'Global Options': {'Fixed Frame': FRAME, 'Background Color': '235; 240; 245'},
                  'Views': {'Current': {'Class': 'rviz/TopDownOrtho', 'Scale': 32., 'X': 0., 'Y': 0., 'Angle': 0.}}, 'Displays': [
            {'Class': 'rviz/Map', 'Name': 'Static reference', 'Topic': TOPIC+'/static', 'Enabled': True, 'Alpha': .5},
            {'Class': 'rviz/Map', 'Name': 'Current local', 'Topic': TOPIC+'/local', 'Enabled': True, 'Alpha': .9},
            {'Class': 'rviz/MarkerArray', 'Name': 'Footprint and status', 'Marker Topic': TOPIC+'/markers', 'Enabled': True},
            {'Class': 'rviz/PointCloud2', 'Name': 'Current raw scan', 'Topic': TOPIC+'/scan', 'Enabled': True,
             'Style': 'Points', 'Size (Pixels)': 2, 'Color Transformer': 'FlatColor', 'Color': '0; 150; 180'}]}}
        screen = app.primaryScreen().availableGeometry()
        config['Window Geometry'] = dict(Width=min(1280, screen.width()-40), Height=max(400, screen.height()-210),
                                         X=screen.x()+20, Y=screen.y()+190)
        path = Path(directory)/'review.rviz'; path.write_text(yaml.safe_dump(config, allow_unicode=True))
        environment = dict(os.environ, DISABLE_ROS1_EOL_WARNINGS='1')
        process = subprocess.Popen(['rviz', '-d', str(path)], env=environment)
        try:
            show(0); window.setWindowTitle('Navigation read-only review'); window.resize(700, 150)
            window.move(screen.x()+20, screen.y()+20); window.show()
            app.exec_()
        finally:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            rospy.signal_shutdown('Review closed')
