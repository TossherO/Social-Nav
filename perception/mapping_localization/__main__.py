"""ROS 1 mapping, fixed-map localization and local odometry from bags or live topics."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

from .results import file_fingerprint, map_fingerprint
from .stationary_imu import localization_source


ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def stop(process, first_signal=signal.SIGINT):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, first_signal)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        raise RuntimeError('ROS process did not exit cleanly')


def wait_for_display_recording(path, timeout=10):
    """rosbag's launcher may exit before its recorder child closes/renames the bag."""
    path = Path(path)
    active = Path(str(path)+'.active')
    deadline = time.monotonic()+timeout
    while active.exists() or not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('Display recording was not finalized')
        time.sleep(.05)
    return file_fingerprint(path)


def show_final_map(map_path):
    """Keep the saved map and optimized keyframes visible after computation stops."""
    print('建图已完成并保存。正在显示最终地图和优化后的关键帧轨迹；'
          '关闭RViz或结果控制窗口，或在终端按Ctrl+C退出。', flush=True)
    view = subprocess.Popen(['/usr/bin/python3', '-m', 'perception.visualization.viewer',
                             '--result', str(map_path)], cwd=ROOT, start_new_session=True)
    try:
        code = view.wait()
        if code:
            print('建图结果已保存，但最终显示未正常退出；可使用view.sh --result MAP重新查看。', flush=True)
        return code
    except KeyboardInterrupt:
        return 0  # Leaving the view does not cancel an already completed map.
    finally:
        stop(view, first_signal=signal.SIGTERM)


def merge_timings(output):
    """Join process-local profiles by observation stamp, preserving absent measurements."""
    frontend, target = output / 'frontend_timings.csv', output / 'timings.csv'
    if not frontend.is_file():
        return
    if not target.is_file():
        frontend.rename(target)
        return
    with frontend.open() as stream:
        reader = csv.DictReader(stream)
        fields = [key for key in reader.fieldnames if key.startswith('fast_lio_')]
        measurements = {row['stamp_ns']: row for row in reader}
    with target.open() as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    temporary = target.with_suffix('.joining.csv')
    try:
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=names+fields)
            writer.writeheader()
            for row in rows:
                match = measurements.get(row['stamp_ns'], {})
                writer.writerow(dict(row, **{key: match.get(key, '') for key in fields}))
        temporary.replace(target)
        frontend.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def finalize_artifacts(output, settings, keep_details=False):
    """Keep summary, recorded states and map; retain diagnostics on request or failure."""
    if settings.get('success', False):
        if keep_details:
            merge_timings(output)
        disposable = ['master.log', 'playback.log']
        if not keep_details:
            disposable += ['algorithm.log', 'frontend.log', 'people.log', 'image_association.log', 'rviz.log', 'display.log', 'display_recording.log',
                           'trajectory.tum', 'timings.csv']
        for name in disposable:
            path = output / name
            if path.is_file():
                path.unlink()
    logs = output / 'logs'
    if logs.is_dir() and not logs.is_symlink() and not any(logs.iterdir()):
        logs.rmdir()
    settings['retained_artifacts'] = sorted(['run.json'] + [
        path.name + ('/' if path.is_dir() else '')
        for path in output.iterdir() if path.name != 'run.json'])
    (output / 'run.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')


def wait_for_scan_ack(condition, receipt, stamp_ns, previous_count, alive, timeout=60):
    """Bind completion to the published scan, never an earlier queued result."""
    deadline = time.monotonic() + timeout
    with condition:
        while receipt['count'] <= previous_count or receipt['stamp_ns'] != stamp_ns:
            if not alive():
                raise RuntimeError('PIN worker exited before acknowledging its scan')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('PIN scan processing exceeded the per-frame budget')
            condition.wait(min(.1, remaining))


def wait_for_initial_pose(ready, algorithm, view):
    print('地图已加载。请在RViz选择2D Pose Estimate，点击雷达当前位置并拖动朝向；'
          '确认初值后才启动定位输入。Ctrl+C或关闭RViz取消。', flush=True)
    while not ready.wait(.1):
        if algorithm.poll() is not None:
            raise RuntimeError('Localization backend exited while waiting for an initial pose')
        if view.poll() is not None:
            raise RuntimeError('Initial pose selection cancelled: RViz was closed')


def wait_for_native_drain(status, alive, last_state_wall, timeout=20):
    """Wait for actual ROLL work; the budget limits stalls, not productive work."""
    progress, last_progress = None, time.monotonic()
    while True:
        if not alive():
            raise RuntimeError('Algorithm exited while draining its input queue')
        stats = status()
        now = time.monotonic()
        current = (stats['processed'], stats['received_clouds'], stats['received_odometry'])
        if current != progress:
            progress, last_progress = current, now
        # An unmatched odometry tail is possible; unprocessed feature clouds are not.
        quiet = last_state_wall[0] is not None and now-last_state_wall[0] > 2
        if not stats['busy'] and not stats['pending_clouds'] and quiet and now-last_progress > 2:
            return stats
        if now-last_progress >= timeout:
            raise RuntimeError('ROLL input processing stalled: ' + json.dumps(stats))
        time.sleep(.2)


def prepare_decision_project(args, output):
    """Run only after ROS/model teardown, preserving the now-frozen perception run."""
    project = project_path(args.decision_project)
    cache = output/'rgb_association'
    run_metadata=json.loads((output/'run.json').read_text())
    transport = run_metadata.get('people_transport',{})
    command = [str(project_path(args.people_python)),'-B','-m','decision.vlm.annotation','prepare',
               '--results',str(output),'--bag',str(project_path(args.bag)),'--output',str(project),
               '--ros-python','/usr/bin/python3']
    if not isinstance(run_metadata.get('image_association'),dict) or not run_metadata['image_association'].get('complete'):
        command += ['--calibration',args.decision_calibration]
    if getattr(args,'image_association',True):
        command += ['--with-rgb-associations','--rgb-output',str(cache)]
    for option,key in (('--pointcloud-topic','pointcloud_topic'),('--rgb-topic','rgb_topic'),('--camera-info-topic','camera_info_topic')):
        if transport.get(key):command += [option,transport[key]]
    if args.decision_root_group:
        command += ['--root-group',args.decision_root_group]
    print('感知已结束并释放资源；开始建立决策标注索引并接入二维结果…',flush=True)
    process = subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    try:
        if process.wait():
            raise RuntimeError('决策标注准备失败；已有感知结果保留。索引已完成时可用'
                               ' decision.vlm.annotation prepare-rgb --project '+str(project)+' 恢复二维阶段')
    finally:
        stop(process,first_signal=signal.SIGTERM)
    if not getattr(args,'image_association',True):
        import sqlite3
        with sqlite3.connect((project/'observations.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            count=db.execute('SELECT count(*) FROM frames').fetchone()[0]
        return dict(status='ready',project=str(project),observations=count,rgb_box_source='projection')
    manifest = json.loads((cache/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise RuntimeError('YOLO关联缓存未完成，不能报告标注已就绪')
    # Do not rewrite run.json here: annotation provenance already binds its hash.
    return dict(status='ready',project=str(project),rgb_associations=str(cache),
                observations=manifest['observations'],association_identity=manifest['identity'])


def wait_for_image_drain(condition,status,last_people,received,alive,timeout=15):
    deadline=time.monotonic()+timeout
    with condition:
        while True:
            state=status()
            quiet=last_people[0] is not None and time.monotonic()-last_people[0]>.5
            if (quiet and not state.get('busy',True) and state.get('last_processed')==last_people[1] and
                    received()>=state.get('processed',0)):
                return state
            if not alive() or time.monotonic()>=deadline:
                raise RuntimeError('RGB association did not drain its bound observations')
            condition.wait(.1)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('mapping', 'localization', 'odometry'), required=True,
                        help='Save a reusable map, localize in a read-only map, or estimate local motion without a prior map')
    parser.add_argument('--source', choices=('bag', 'live'), default='bag',
                        help='bag: own isolated master and replay; live: connect to existing master and topics')
    parser.add_argument('--backend', choices=('roll_fast_lio', 'fast_lio', 'pin_slam'),
                        help='Default: FAST-LIO for odometry, ROLL + FAST-LIO for other tasks')
    parser.add_argument('--backend-python', help='PIN worker interpreter in its independent environment')
    parser.add_argument('--backend-config', help='Explicit PIN YAML; default is the project sensor configuration')
    parser.add_argument('--lidar-imu-calibration', help='Explicit spatial/time calibration YAML for FAST-LIO/ROLL')
    parser.add_argument('--step-replay', action='store_true',
                        help='PIN only: publish every raw scan and wait for its bound result; ignores playback rate')
    parser.add_argument('--bag', help='Input recording; required only for --source bag')
    parser.add_argument('--start', type=float, default=0.0, help='Bag offset in recorded seconds; bag input only')
    parser.add_argument('--assume-stationary', action='store_true',
                        help='Explicit verified-static experiment: synthesize IMU from scans; never a real-IMU fallback')
    parser.add_argument('--localization-map','--map',dest='map',
                        help='Complete 3D localization map bundle; not a static occupancy package or PGM/YAML')
    parser.add_argument('--out', required=True, help='New directory beneath project outputs/ or /tmp/')
    initial_pose = parser.add_mutually_exclusive_group()
    initial_pose.add_argument('--pose', type=float, nargs=6, metavar=('ROLL', 'PITCH', 'YAW', 'X', 'Y', 'Z'))
    initial_pose.add_argument('--pose-file', help='Manual pick_start JSON; ROLL bag localization only')
    parser.add_argument('--initialization', choices=('rviz', 'map-start'),
                        help='ROLL localization: wait for RViz selection, or use the known map start; '
                             'default is rviz with --rviz, otherwise map-start. --pose starts directly')
    parser.add_argument('--duration', type=float, help='Recorded seconds for bag, wall seconds for live; omit to run until completion/Ctrl+C')
    parser.add_argument('--rate', type=float, default=1.0)
    parser.add_argument('--rviz', action='store_true',
                        help='Show the running task; successful mapping then keeps the final map view open until closed')
    parser.add_argument('--loop-closure', choices=('on', 'off'),
                        help='Mapping only: override the default enabled ROLL loop closure')
    parser.add_argument('--people-python', help='Detector interpreter; enables 3D detection and tracking')
    images=parser.add_mutually_exclusive_group()
    images.add_argument('--image-association',action='store_true',help='Explicit VLM/diagnostic YOLO branch; not required by 3D perception/navigation')
    images.add_argument('--no-image-association',action='store_true',help='Disable optional image work, including a requested decision project')
    parser.add_argument('--image-device',default='0',help='YOLO GPU index within CUDA_VISIBLE_DEVICES')
    parser.add_argument('--decision-project',help='Bag only: create this NEW annotation project and reuse the saved joint perception results')
    parser.add_argument('--camera-calibration','--decision-calibration',dest='decision_calibration',help='Camera/LiDAR calibration shared by runtime YOLO and decision preparation')
    parser.add_argument('--decision-root-group',help='Common original recording group for subsequent annotation splits')
    parser.add_argument('--environment', action='store_true', help='Enable observed local geometry and shared scene queries')
    parser.add_argument('--environment-config', help='Explicit environment geometry YAML')
    parser.add_argument('--environment-backend',choices=('cpu','cuda'),help='Explicit environment computation override')
    parser.add_argument('--record-environment-scans', action='store_true',
                        help='Mapping + environment: retain bounded deskewed scans for final ray fusion')
    parser.add_argument('--record-display', action='store_true',
                        help='Environment: save processed scans, geometry grids, states and TF in display.bag for playback without recomputing SLAM')
    parser.add_argument('--scene', action='store_true',
                        help='Environment: join causal geometry/RGB/people/prior into one scene input')
    parser.add_argument('--association-coverage', help='Trusted-region polygon configuration bound to the localization map; mixed localization only')
    parser.add_argument('--static-map','--prior-map',dest='prior_map',
                        help='Static environment map asset used to build optional scene prior views')
    parser.add_argument('--allow-local-only', action='store_true',
                        help='Explicit mixed route: retain valid frontend pose when prior association is unavailable')
    parser.add_argument('--keep-details', action='store_true',
                        help='Write and retain per-scan results, trajectories and diagnostics')
    args = parser.parse_args(argv)
    if args.image_association and not args.people_python:
        parser.error('--image-association requires --people-python for source 3D tracks')
    args.image_association=bool(args.people_python and not args.no_image_association and
        (args.image_association or args.decision_project))
    args.backend = args.backend or ('fast_lio' if args.mode == 'odometry' else 'roll_fast_lio')
    if (args.mode == 'odometry') != (args.backend == 'fast_lio'):
        parser.error('odometry uses fast_lio; mapping/localization use roll_fast_lio or pin_slam')
    if args.source == 'bag' and not args.bag or args.source == 'live' and args.bag:
        parser.error('--bag is required for bag input and cannot be combined with live input')
    if args.source == 'live' and (args.step_replay or args.rate != 1.0 or args.start != 0):
        parser.error('--step-replay, --rate and --start apply only to bag input')
    if args.loop_closure is not None and (args.mode != 'mapping' or args.backend != 'roll_fast_lio'):
        parser.error('--loop-closure applies only to ROLL mapping')
    if args.mode == 'odometry' and (args.map or args.pose or args.pose_file):
        parser.error('odometry starts a new local frame; --localization-map and --pose do not apply')
    if args.pose_file and (args.mode != 'localization' or args.source != 'bag' or args.backend != 'roll_fast_lio'):
        parser.error('--pose-file applies only to ROLL bag localization')
    pin = args.backend == 'pin_slam'
    if args.environment and pin:
        parser.error('Environment observation integration currently supports FAST-LIO/ROLL')
    if args.record_environment_scans and (not args.environment or args.mode != 'mapping'):
        parser.error('--record-environment-scans requires --environment and mapping')
    if args.record_display and not args.environment:
        parser.error('--record-display requires --environment')
    if args.scene and not args.environment:
        parser.error('--scene requires --environment')
    if args.prior_map and (not args.scene or args.mode != 'localization'):
        parser.error('--static-map requires --scene and fixed-map localization')
    if args.allow_local_only and (not args.environment or args.mode != 'localization'):
        parser.error('--allow-local-only requires --environment and fixed-map localization')
    if args.association_coverage and (not args.allow_local_only or not project_path(args.association_coverage).is_file()):
        parser.error('--association-coverage requires --allow-local-only and an existing map-bound YAML')
    if args.environment_config and (not args.environment or not project_path(args.environment_config).is_file()):
        parser.error('--environment-config requires --environment and an existing YAML')
    if args.environment_backend and not args.environment:
        parser.error('--environment-backend requires --environment')
    if not pin:
        args.lidar_imu_calibration = args.lidar_imu_calibration or 'perception/sensors/calibration/lidar_imu.yaml'
    if args.lidar_imu_calibration:
        if pin or not project_path(args.lidar_imu_calibration).is_file():
            parser.error('--lidar-imu-calibration requires an existing YAML and FAST-LIO/ROLL')
    if pin and args.assume_stationary:
        parser.error('Synthetic IMU applies to FAST-LIO/ROLL; PIN is LiDAR-only')
    if pin and (not args.backend_python or not project_path(args.backend_python).is_file()):
        parser.error('PIN requires --backend-python pointing to its prepared interpreter')
    if not pin and (args.backend_python or args.backend_config):
        parser.error('Backend Python/config overrides apply only to PIN')
    if args.step_replay and not pin:
        parser.error('Acknowledged raw-LiDAR replay currently supports PIN only')
    if args.step_replay and args.start:
        parser.error('--start is not supported by PIN acknowledged replay')
    if pin and args.rviz:
        parser.error('PIN publishes /pin_slam/map and TF; use a separately configured RViz view')
    import math
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('Bag start must be finite and nonnegative')
    if not math.isfinite(args.rate) or args.rate <= 0 or (args.duration is not None and
            (not math.isfinite(args.duration) or args.duration <= 0)):
        parser.error('Rate and duration must be finite and positive')
    if args.pose and not all(math.isfinite(v) for v in args.pose):
        parser.error('Initial pose must contain six finite values')
    if args.initialization is not None:
        if args.mode != 'localization' or args.backend != 'roll_fast_lio':
            parser.error('--initialization applies only to ROLL fixed-map localization')
        if args.pose is not None or args.pose_file:
            parser.error('Choose --pose/--pose-file or --initialization, not both')
    bag, output = project_path(args.bag) if args.bag else None, project_path(args.out)
    if bag is not None and not bag.is_file():
        parser.error('Bag does not exist')
    if (not any(base in output.parents for base in ((ROOT / 'outputs').resolve(), Path('/tmp'))) or output.exists()):
        parser.error('Output must be a NEW directory beneath project outputs/ or /tmp/')
    map_path = project_path(args.map) if args.map else output / 'map'
    if not args.decision_project and args.decision_root_group:
        parser.error('--decision-root-group requires --decision-project')
    if not args.image_association and not args.decision_project and args.decision_calibration:
        parser.error('--camera-calibration requires image association or --decision-project')
    if args.image_association and args.step_replay:
        parser.error('PIN acknowledged replay supports 3D detection only; omit the image branch or use ordinary replay')
    if args.image_association or args.decision_project:
        camera = project_path(args.decision_calibration or 'perception/sensors/calibration/front_camera_lidar.yaml')
        if not camera.is_file():parser.error('Camera/LiDAR calibration does not exist')
        args.decision_calibration = str(camera)
    if args.decision_project:
        if args.source != 'bag' or not args.people_python or not project_path(args.people_python).is_file():
            parser.error('--decision-project requires bag input and an existing --people-python interpreter')
        project = project_path(args.decision_project)
        if project.exists() or project == output or project in output.parents or map_path == project or map_path in project.parents:
            parser.error('Decision project must be NEW and must not contain the run or be inside its map')
        args.decision_project = str(project)
        from perception.detection_tracking.inference.config import load_config
        if args.image_association:
            image = load_config()['image_association']
            artifact = project_path(image['engine'] if image['backend']=='tensorrt' else image['weights'])
            if not artifact.is_file() or image['backend']=='tensorrt' and not artifact.with_suffix('.engine.json').is_file():
                parser.error('Decision preparation requires prepared YOLO weights/engine and its build record; no automatic export')
    if args.mode == 'localization':
        if not args.map or not (map_path / 'COMPLETE').is_file():
            parser.error('Localization requires an explicitly selected complete 3D localization map bundle')
        if args.pose_file:
            from .initial_pose import load_seed
            args.pose_file = str(project_path(args.pose_file))
            try:
                args.pose = load_seed(args.pose_file, map_path, bag, args.start)
            except (OSError, ValueError, TypeError) as error:
                parser.error(str(error))
        if not pin:
            args.initialization = 'pose' if args.pose is not None else args.initialization or (
                'rviz' if args.rviz else 'map-start')
        if args.initialization == 'rviz':
            args.rviz = True
        elif args.pose is None:
            if pin:
                manifest = json.loads((map_path / 'manifest.json').read_text())
                if manifest.get('format') != 'pin_slam_neural_map':
                    parser.error('PIN requires a complete PIN neural map')
                args.pose = manifest['initial_pose']
            else:
                first = [float(x) for x in (map_path / 'keyframes/poses.txt').read_text().splitlines()[0].split()]
                if len(first) != 9:
                    parser.error('Map start fallback requires current 9-column pose format')
                args.pose = first[4:7] + first[1:4]
            print('Using this map\'s known mapping start; other starts require --pose.')
    elif args.mode == 'mapping':
        if map_path.exists() or output not in map_path.parents:
            parser.error('New map must be inside this run output and must not exist')
        args.pose = args.pose or [0.0] * 6
    else:
        args.pose = [0.0] * 6
    if args.pose is not None and (len(args.pose) != 6 or not all(math.isfinite(value) for value in args.pose)):
        parser.error('Map start must contain six finite pose values')
    return args


def algorithm_command(args, output, map_path, use_sim_time, *, coverage_path=None):
    """One launch contract shared by replay and live sessions."""
    pin = args.backend == 'pin_slam'
    command = ['roslaunch', str(ROOT / 'perception/mapping_localization/launch' /
                               ('pin_slam.launch' if pin else 'native.launch')),
               'mode:=' + args.mode, 'map_dir:=' + (str(map_path) if map_path else ''),
               'initial_pose:=' + json.dumps(args.pose if args.pose is not None else [0.0]*6),
               'use_sim_time:=' + str(use_sim_time).lower()]
    if pin:
        command += ['project_root:=' + str(ROOT),
                    'python_executable:=' + str(project_path(args.backend_python)),
                    'input_source:=' + (str(project_path(args.bag)) if args.bag else 'live_lidar')]
        if args.backend_config:
            command += ['config:=' + str(project_path(args.backend_config))]
    else:
        manual = args.initialization == 'rviz'
        command += ['run_dir:=' + str(output), 'rviz:=false',
                    'visualize:=' + str(args.rviz).lower(),
                    'keep_details:=' + str(args.keep_details).lower(),
                    'assume_stationary:=' + str(args.assume_stationary).lower(),
                    'start_frontend:=' + str(not manual).lower(),
                    'wait_for_initial_pose:=' + str(manual).lower()]
        if args.loop_closure is not None:
            command += ['loop_closure:=' + str(args.loop_closure == 'on').lower()]
        if args.lidar_imu_calibration:
            command += ['lidar_imu_calibration:=' + str(project_path(args.lidar_imu_calibration))]
        if args.environment:
            command += ['environment:=true','allow_local_only:='+str(args.allow_local_only).lower()]
            if args.environment_config:
                command += ['environment_config:='+str(project_path(args.environment_config))]
            if args.environment_backend:
                command += ['environment_backend:='+args.environment_backend]
            command += ['record_environment_scans:='+str(args.record_environment_scans).lower(),
                        'scene:='+str(args.scene).lower(),
                        'scene_people:='+str(bool(args.people_python)).lower()]
            if args.association_coverage:
                command += ['association_coverage:='+str(coverage_path or project_path(args.association_coverage))]
            if args.prior_map:
                command += ['static_map:='+str(project_path(args.prior_map))]
    return command


def replay_topics(args, parameters, people_transport=None, environment_config=None):
    """Replay actual configured inputs, including RGB only for joint perception."""
    common = parameters.get('common', {})
    raw = (parameters['stationary_imu']['raw_topic'] if args.assume_stationary else
           common.get('lid_topic', '/rslidar_points'))
    topics = [raw, '/tf_static']
    if args.backend != 'pin_slam' and not args.assume_stationary:
        topics.append(common.get('imu_topic', '/camera/imu'))
    if people_transport and people_transport.get('rgb_topic'):
        topics.append(people_transport['rgb_topic'])
    if getattr(args,'image_association',False):
        from perception.detection_tracking.inference.config import load_config
        topics.append((people_transport or {}).get('camera_info_topic') or load_config()['ros']['camera_info_topic'])
    if getattr(args,'scene',False):
        topics.append(environment_config['scene']['rgb_topic'])
    return list(dict.fromkeys(topics))


def main():
    args = arguments()
    pin, live = args.backend == 'pin_slam', args.source == 'live'
    bag, output = project_path(args.bag) if args.bag else None, project_path(args.out)
    map_path = (project_path(args.map) if args.map else output / 'map') if args.mode != 'odometry' else None
    from perception.detection_tracking.inference.config import load_config
    inference_config=load_config()
    display_transport = inference_config['ros']
    if args.image_association:
        policy=inference_config['image_association']
        artifact=project_path(policy['engine'] if policy['backend']=='tensorrt' else policy['weights'])
        if not artifact.is_file() or policy['backend']=='tensorrt' and not artifact.with_suffix('.engine.json').is_file():
            raise ValueError('The requested image branch requires prepared YOLO weights/engine; omit it for 3D perception')
    environment_config = None
    if args.environment:
        from perception.environment.layers import load_config as load_environment_config
        environment_config = load_environment_config(project_path(args.environment_config) if args.environment_config else None)
        if args.environment_backend:
            environment_config.setdefault('compute',{})['backend']=args.environment_backend
    people_topic = display_transport['output_topic']
    bag_topics = None
    if bag is not None:
        import rosbag
        with rosbag.Bag(str(bag)) as recording:
            bag_topics = recording.get_type_and_topic_info().topics
            if args.start >= recording.get_end_time()-recording.get_start_time():
                raise ValueError('Bag offset is outside this recording')
            if args.assume_stationary and any(v.msg_type == 'sensor_msgs/Imu' for v in bag_topics.values()):
                raise ValueError('Bag already contains IMU; do not silently replace recorded inertial measurements')
            if args.scene and environment_config['scene']['rgb_topic'] not in bag_topics:
                raise ValueError('Scene assembly requires the configured RGB topic in the bag')
            if (args.decision_project or args.image_association) and any(topic not in bag_topics for topic in
                    (display_transport['rgb_topic'],display_transport['camera_info_topic'])):
                raise ValueError('Decision preparation requires RGB and CameraInfo in the original bag')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    settings = dict(vars(args), bag=str(bag) if bag else None, map=str(map_path) if map_path else None,
                    out=str(output), scope='online_topics' if live else 'offline_replay',
                    geometry_accuracy_verified=False, success=False)
    if environment_config is not None:
        settings['environment_configuration'] = environment_config
    expected_source = localization_source(args.backend, args.assume_stationary)
    settings['localization_source'] = expected_source
    settings['map_type'] = 'localization_map' if map_path is not None else None
    settings['inertial_input'] = 'synthetic_stationary_assumption' if args.assume_stationary else (
        'not_used' if pin else 'recorded_or_live_imu')
    settings['playback'] = {'records': 'localization.jsonl', 'reference': 'scan_end',
                            'map_geometry': 'final_only', 'bag_fingerprint': None,
                            'map_fingerprint': None}
    if args.mode == 'localization':
        settings['playback']['map_fingerprint'] = map_fingerprint(map_path)
    if args.prior_map:
        from perception.environment.prior import StaticPriorView
        prior = StaticPriorView(project_path(args.prior_map),expected_map=settings['playback']['map_fingerprint'])
        settings['prior_package'] = dict(entity_type='map',map_type='static_environment_map',snapshot_id=prior.manifest['snapshot_id'],
            layers_sha256=prior.manifest['layers_sha256'],asset_fingerprint=prior.asset)
    if args.association_coverage:
        import hashlib
        from perception.mapping_localization.coverage import PriorCoverage
        digest = hashlib.sha256(json.dumps(settings['playback']['map_fingerprint'],sort_keys=True).encode()).hexdigest()
        coverage = PriorCoverage.load(project_path(args.association_coverage),asset_fingerprint=digest)
        settings['association_coverage_record'] = coverage.record
        settings['association_coverage_fingerprint'] = coverage.fingerprint
    record_path = output / 'run.json'
    if args.image_association:settings['image_association']=dict(complete=False,records='rgb_associations.jsonl')
    record_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    counts = {'localization': 0, 'valid': 0, 'invalid': 0, 'people': 0, 'people_valid': 0,
              'people_degraded': 0, 'people_rgb': 0, 'people_dropped_busy': 0,
              'detections': 0, 'confirmed_tracks': 0}
    if args.image_association:
        counts.update(rgb_records=0,rgb_matches=0,rgb_failed=0,rgb_inferred=0)
    image_status=[{}]
    last_people=[None,None]
    settings['localization_statuses'] = {}
    condition = threading.Condition()
    last_state_wall = [None]
    input_receipts = {}
    receipt = {'count': 0, 'stamp_ns': None}
    master = algorithm = frontend = people = image_node = playback = view = display = None
    with tempfile.TemporaryDirectory(prefix='social-nav-ros-replay-') as temporary:
        frozen_coverage = None
        if args.association_coverage:
            import yaml
            frozen_coverage = Path(temporary) / 'localization_coverage.yaml'
            frozen_coverage.write_text(yaml.safe_dump(settings['association_coverage_record'], sort_keys=False))
            settings['association_gate_configuration'] = yaml.safe_load(
                (ROOT / 'perception/mapping_localization/configs/roll.yaml').read_text())['roll']['coverageGate']
        env = os.environ.copy()
        if not live:
            with socket.socket() as selector:
                selector.bind(('127.0.0.1', 0))
                port = selector.getsockname()[1]
            env.pop('ROS_HOSTNAME', None)
            os.environ.pop('ROS_HOSTNAME', None)
            env.update(ROS_MASTER_URI='http://127.0.0.1:%d' % port, ROS_IP='127.0.0.1')
        env.update(ROS_HOME=temporary, ROS_LOG_DIR=temporary + '/logs')
        os.environ.update({key: env[key] for key in ('ROS_MASTER_URI', 'ROS_IP', 'ROS_HOME', 'ROS_LOG_DIR') if key in env})
        streams = []

        def launch(command, filename):
            stream = (output / filename).open('w')
            streams.append(stream)
            return subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)

        display_recording = None
        try:
            if bag:
                settings['playback']['bag_fingerprint'] = file_fingerprint(bag)
            if not live:
                master = launch(['roscore', '-p', str(port)], 'master.log')
            from xmlrpc.client import ServerProxy
            deadline = time.monotonic() + 15
            while True:
                try:
                    if ServerProxy(env.get('ROS_MASTER_URI', 'http://localhost:11311')).getPid('/social_nav_session')[0] == 1:
                        break
                except (OSError, ConnectionError):
                    pass
                if live or master.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Start ROS master before live input' if live else 'Temporary ROS master failed to start')
                time.sleep(.1)
            import rospy
            from social_nav_localization.msg import LocalizationState
            from std_msgs.msg import String
            if not live:
                # rospy decides whether to subscribe to /clock during init_node.
                ServerProxy(env['ROS_MASTER_URI']).setParam('/social_nav_session', '/use_sim_time', True)
            rospy.init_node('social_nav_session', anonymous=True, disable_signals=True)
            if args.record_display:
                prefix = environment_config['output_topic']
                display_topics = [prefix+'/scan', prefix+'/occupancy', prefix+'/grid',prefix+'/state',
                                  prefix+'/historical_free',prefix+'/recent_obstacles',
                                  '/localization/state', '/localization/map_association', '/tf', '/tf_static']
                if args.scene:
                    scene_prefix = environment_config['scene']['output_topic']
                    display_topics += [scene_prefix, scene_prefix+'/prior/occupancy']
                display_recording = launch(['rosbag','record','--lz4','--buffsize','128',
                    '-O',str(output/'display.bag')]+display_topics, 'display_recording.log')
                settings['display_recording_topics'] = display_topics
            use_sim_time = bool(rospy.get_param('/use_sim_time', False)) if live else True
            settings['use_sim_time'] = use_sim_time
            if not pin:
                from geometry_msgs.msg import PoseStamped

                def on_frontend_reference(message):
                    p, q = message.pose.position, message.pose.orientation
                    with condition:
                        settings['frontend_initial_reference'] = {
                            'stamp_ns': message.header.stamp.to_nsec(), 'frame_id': message.header.frame_id,
                            'position': [p.x, p.y, p.z], 'quaternion_xyzw': [q.x, q.y, q.z, q.w]}

                frontend_reference_sub = rospy.Subscriber('/fast_lio/initial_lidar_pose', PoseStamped,
                                                         on_frontend_reference, queue_size=1)
            if args.mode == 'localization' and not pin:
                from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
                from tf2_ros import StaticTransformBroadcaster
                pose_ready = threading.Event()

                def on_initial_pose(message):
                    p, q = message.pose.pose.position, message.pose.pose.orientation
                    with condition:
                        settings.setdefault('initial_pose_selections', []).append({
                            'selection_stamp_s': message.header.stamp.to_sec(),
                            'map_frame': message.header.frame_id, 'position': [p.x, p.y, p.z],
                            'quaternion_xyzw': [q.x, q.y, q.z, q.w]})
                    pose_ready.set()  # Backend acceptance is a seed, not a VALID registration.

                initial_pose_sub = rospy.Subscriber('/roll/initial_pose_accepted', PoseWithCovarianceStamped,
                                                    on_initial_pose, queue_size=8)
            state_file = (output / 'localization.jsonl').open('w')
            streams.append(state_file)

            def on_state(message):
                if message.source != expected_source:
                    return
                received_wall_ns = time.time_ns()
                received_monotonic_ns = time.monotonic_ns()
                received_ros = rospy.Time.now()
                p, q = message.pose.position, message.pose.orientation
                row = {'stamp_s': message.header.stamp.to_sec(), 'stamp_ns': message.header.stamp.to_nsec(),
                    'frame_id': message.header.frame_id, 'child_frame_id': message.child_frame_id,
                    'position': [p.x, p.y, p.z], 'quaternion_xyzw': [q.x, q.y, q.z, q.w],
                    'health': message.health, 'status': message.status, 'source': message.source,
                    'session_id': message.session_id, 'generation': message.generation,
                    'map_revision': message.map_revision, 'has_covariance': message.has_covariance,
                    'reason': message.reason, 'has_map_pose': message.has_map_pose,
                    'map_frame': message.map_frame}
                if args.keep_details:
                    # ROS clock age includes processing/transport, not sensor-clock calibration error.
                    row['received_wall_ns'] = received_wall_ns
                    row['observation_age_ms'] = (received_ros-message.header.stamp).to_sec()*1000
                    with condition:
                        input_received = input_receipts.pop(message.header.stamp.to_nsec(), None)
                    row['raw_to_state_ms'] = ((received_monotonic_ns-input_received)/1e6
                                             if input_received is not None and input_received <= received_monotonic_ns else None)
                if message.has_map_pose:
                    mp, mq = message.map_pose.position, message.map_pose.orientation
                    row['map_position'] = [mp.x, mp.y, mp.z]
                    row['map_quaternion_xyzw'] = [mq.x, mq.y, mq.z, mq.w]
                state_file.write(json.dumps(row) + '\n')
                with condition:
                    counts['localization'] += 1
                    counts['valid' if message.health == message.VALID else 'invalid'] += 1
                    status_key = message.status + (':' + message.reason if message.reason else '')
                    settings['localization_statuses'][status_key] = settings['localization_statuses'].get(status_key, 0) + 1
                    if message.health == message.VALID:
                        row = {'stamp_s': message.header.stamp.to_sec(), 'position': [p.x, p.y, p.z],
                               'quaternion_xyzw': [q.x, q.y, q.z, q.w], 'map_revision': message.map_revision}
                        settings.setdefault('first_valid_pose', row)
                        settings['last_valid_pose'] = row
                    last_state_wall[0] = time.monotonic()
                    receipt.update(count=receipt['count']+1, stamp_ns=message.header.stamp.to_nsec())
                    condition.notify_all()

            state_sub = rospy.Subscriber('/localization/state', LocalizationState, on_state, queue_size=256)
            if live:
                existing = rospy.get_published_topics()
                if any(topic == '/localization/state' for topic, _ in existing):
                    raise RuntimeError('Another localization publisher is active; stop it before starting this session')
            algorithm = launch(algorithm_command(args, output, map_path, use_sim_time,
                                                 coverage_path=frozen_coverage), 'algorithm.log')
            rospy.wait_for_service('/pin_slam/status' if pin else
                                   '/fast_lio/status' if args.mode == 'odometry' else '/roll/save_map',
                                   timeout=90 if pin else 30)
            if args.mode == 'localization' and not pin:
                # Declare only a display reference while no robot pose is known.
                reference = TransformStamped()
                reference.header.frame_id = rospy.get_param('/roll/mapFrame', 'map')
                reference.child_frame_id = 'localization_map_reference'
                reference.transform.rotation.w = 1
                map_reference = StaticTransformBroadcaster()
                map_reference.sendTransform(reference)
            if args.rviz:
                import yaml
                from perception.visualization.scans import configure_rviz
                display_transport.update(
                    pointcloud_topic=rospy.get_param('/stationary_imu/raw_topic') if args.assume_stationary else
                        rospy.get_param('/common/lid_topic', '/rslidar_points'),
                    localization_topic='/localization/state', localization_source=expected_source)
                settings['display_transport'] = display_transport
                rviz_transport=dict(display_transport)
                if args.image_association:rviz_transport['people_image_topic']=people_topic+'/rgb_associations/image'
                local_view = args.mode == 'odometry' or (args.environment and args.initialization != 'rviz')
                view_config = 'odometry.rviz' if local_view else 'roll.rviz'
                view_path = ROOT / 'perception/visualization/configs' / view_config
                configuration = configure_rviz(yaml.safe_load(view_path.read_text()), rviz_transport)
                if args.environment:
                    from perception.visualization.environment import add_environment_display
                    add_environment_display(configuration,environment_config['output_topic'],
                        prior_topic=environment_config['scene']['output_topic']+'/prior/occupancy' if args.prior_map else None,
                        target_frame=environment_config['sensor_frame'])
                if args.initialization == 'rviz':
                    import numpy as np
                    positions = np.loadtxt(map_path / 'keyframes/poses.txt', ndmin=2)[:, 1:4]
                    low, high = positions.min(axis=0), positions.max(axis=0)
                    center = (low+high)/2
                    configuration['Visualization Manager']['Global Options']['Fixed Frame'] = reference.header.frame_id
                    configuration['Visualization Manager']['Views']['Current'] = {
                        'Class': 'rviz/Orbit', 'Pitch': 1.45, 'Yaw': -float(np.pi/2),
                        'Distance': float(max(15, (np.linalg.norm(high-low)+12)*1.5)),
                        'Focal Point': dict(zip(('X', 'Y', 'Z'), map(float, center))),
                        'Target Frame': '<Fixed Frame>'}
                view_path = Path(temporary) / 'live.rviz'
                view_path.write_text(yaml.safe_dump(configuration))
                if not args.people_python:
                    display = launch(['roslaunch', str(ROOT / 'perception/detection_tracking/ros/launch/visualization.launch'),
                        'mode:=localization', 'pointcloud_topic:=' + display_transport['pointcloud_topic'],
                        'localization_topic:=/localization/state', 'localization_source:=' + expected_source], 'display.log')
                env['DISABLE_ROS1_EOL_WARNINGS'] = '1'
                view = launch(['rosrun', 'rviz', 'rviz', '-d',
                               str(view_path),
                               '__name:=localization_rviz'], 'rviz.log')
            if args.initialization == 'rviz':
                wait_for_initial_pose(pose_ready, algorithm, view)
                frontend = launch(['roslaunch', str(ROOT / 'perception/mapping_localization/launch/frontend.launch')],
                                  'frontend.log')
                from rosgraph import Master
                graph = Master('/social_nav_session')
                deadline = time.monotonic()+30
                raw_topic = rospy.resolve_name(rospy.get_param('/common/lid_topic'))
                feature_topic = rospy.resolve_name(rospy.get_param('/roll/pointCloudTopic'))
                while True:
                    subscribers = dict(graph.getSystemState()[1])
                    if ('/laserMapping' in subscribers.get(raw_topic, []) and
                            '/roll_scanRegistration' in subscribers.get(feature_topic, [])):
                        break
                    if frontend.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Localization frontend did not become ready after selection')
                    time.sleep(.1)
                if args.environment:
                    # Initial picking uses the prior map view. Once accepted, the
                    # running scene stays in continuous odom through unmapped roads.
                    stop(view,first_signal=signal.SIGTERM)
                    configuration = configure_rviz(yaml.safe_load((ROOT/'perception/visualization/configs/odometry.rviz').read_text()),rviz_transport)
                    add_environment_display(configuration,environment_config['output_topic'],
                        prior_topic=environment_config['scene']['output_topic']+'/prior/occupancy' if args.prior_map else None,
                        target_frame=environment_config['sensor_frame'])
                    view_path.write_text(yaml.safe_dump(configuration))
                    view = launch(['rosrun','rviz','rviz','-d',str(view_path),'__name:=localization_rviz'],'rviz.log')
            if pin:
                from std_srvs.srv import Trigger
                pin_status = rospy.ServiceProxy('/pin_slam/status', Trigger)
                settings['algorithm_parameters'] = rospy.get_param('/pin_slam/algorithm_parameters')
            else:
                if args.mode != 'odometry':
                    from std_srvs.srv import Trigger
                    rospy.wait_for_service('/roll/status', timeout=30)
                    native_status = rospy.ServiceProxy('/roll/status', Trigger)

                    def read_native_status():
                        response = native_status()
                        settings['backend_statistics'] = json.loads(response.message)
                        if not response.success:
                            raise RuntimeError('ROLL dropped input after queue overflow: ' + response.message)
                        return settings['backend_statistics']

                    # Advertisement precedes map loading in roscpp. A completed
                    # callback also confirms constructor/submap preparation ended.
                    read_native_status()

                keys = ['common', 'preprocess', 'mapping', 'publish', 'pcd_save', 'odometry', 'initialization',
                        'cube_side_length', 'max_iteration', 'point_filter_num', 'filter_size_surf',
                        'filter_size_map', 'feature_extract_enable', 'need_map_erected', 'matching_threads',
                        'matching_backend', 'deskew_backend', 'linearization_backend', 'filter_compact_solve']
                if args.mode != 'odometry':
                    keys.append('roll')
                if args.assume_stationary:
                    keys.append('stationary_imu')
                settings['algorithm_parameters'] = {key: rospy.get_param('/'+key) for key in keys
                                                     if rospy.has_param('/'+key)}
            if args.environment:
                from std_srvs.srv import Trigger
                rospy.wait_for_service(environment_config['output_topic']+'/status',timeout=30)
                environment_status = rospy.ServiceProxy(environment_config['output_topic']+'/status',Trigger)
            if args.scene:
                rospy.wait_for_service(environment_config['scene']['output_topic']+'/status',timeout=30)
                scene_status = rospy.ServiceProxy(environment_config['scene']['output_topic']+'/status',Trigger)
            if args.assume_stationary:
                from std_srvs.srv import Trigger
                rospy.wait_for_service('/stationary_imu/status', timeout=15)
                synthetic_status = rospy.ServiceProxy('/stationary_imu/status', Trigger)
                status = synthetic_status()
                if not status.success:
                    raise RuntimeError('Stationary input adapter is not ready: ' + status.message)
                settings['stationary_input'] = json.loads(status.message)
            if args.keep_details and not pin:
                from sensor_msgs.msg import PointCloud2

                def on_raw_cloud(message):
                    received = time.monotonic_ns()
                    with condition:
                        input_receipts[message.header.stamp.to_nsec()] = received
                        while len(input_receipts) > 256:
                            input_receipts.pop(next(iter(input_receipts)))

                raw_timing_topic = (rospy.get_param('/stationary_imu/raw_topic') if args.assume_stationary else
                                    rospy.get_param('/common/lid_topic', '/rslidar_points'))
                raw_timing_sub = rospy.Subscriber(raw_timing_topic,
                    PointCloud2, on_raw_cloud, queue_size=128)
            if args.people_python:
                people_file = (output / 'people.jsonl').open('w')
                streams.append(people_file)

                def on_people(message):
                    result = json.loads(message.data)
                    if args.keep_details:
                        result['received_wall_ns'] = time.time_ns()
                        result['observation_age_ms'] = (rospy.Time.now().to_sec()-result['stamp_s'])*1000
                    if people_file:
                        people_file.write(json.dumps(result, allow_nan=False) + '\n')
                    with condition:
                        last_people[:]=[time.monotonic(),[result.get(key) for key in
                            ('session_id','generation','sequence_id','clock_domain','observation_id','stamp_s')]]
                        counts['people'] += 1
                        counts['people_valid' if result['health'] == 'valid' else 'people_degraded'] += 1
                        counts['people_rgb'] += result.get('rgb_stamp_s') is not None
                        counts['people_dropped_busy'] = max(counts['people_dropped_busy'], result.get('dropped_busy', 0))
                        counts['detections'] += len(result['detections'])
                        counts['confirmed_tracks'] += sum(t['confirmed'] for t in result['tracks'])
                        condition.notify_all()

                people_sub = rospy.Subscriber(people_topic+'/json', String, on_people, queue_size=64)
                lidar_topic = (rospy.get_param('/stationary_imu/raw_topic') if args.assume_stationary else
                               settings['algorithm_parameters'].get('common', {}).get('lid_topic', '/rslidar_points'))
                people = launch(['roslaunch', str(ROOT / 'perception/detection_tracking/ros/launch/people.launch'),
                    'project_root:=' + str(ROOT), 'python_executable:=' + str(project_path(args.people_python)),
                    'visualize:=' + str(args.rviz).lower(),
                    'pointcloud_topic:=' + lidar_topic, 'localization_topic:=/localization/state',
                    'localization_source:=' + expected_source],
                    'people.log')
                settings['people_transport'] = json.loads(
                    rospy.wait_for_message(people_topic+'/status', String, timeout=90).data)
                settings['people_transport']['camera_info_topic']=display_transport['camera_info_topic']
                if args.image_association:
                    image_file=(output/'rgb_associations.jsonl').open('w')
                    streams.append(image_file)
                    image_topic=people_topic+'/rgb_associations'
                    def on_image_record(message):
                        record=json.loads(message.data)
                        with condition:
                            image_file.write(json.dumps(record,allow_nan=False)+'\n')
                            counts['rgb_records']+=1
                            counts['rgb_matches']+=record['result'].get('matches',0)
                            counts['rgb_failed']+=record['result']['status']=='failed'
                            counts['rgb_inferred']+=record.get('detector_s') is not None
                            condition.notify_all()
                    def on_image_status(message):
                        with condition:
                            image_status[0]=json.loads(message.data)
                            condition.notify_all()
                    image_sub=rospy.Subscriber(image_topic+'/records',String,on_image_record,queue_size=128,buff_size=16*1024*1024)
                    image_status_sub=rospy.Subscriber(image_topic+'/status',String,on_image_status,queue_size=8)
                    image_node=launch(['roslaunch',str(ROOT/'perception/detection_tracking/ros/launch/rgb_association.launch'),
                        'project_root:='+str(ROOT),'python_executable:='+str(project_path(args.people_python)),
                        'calibration:='+args.decision_calibration,'device:='+args.image_device,
                        'people_topic:='+people_topic+'/json','rgb_topic:='+people_topic+'/rgb',
                        'camera_info_topic:='+display_transport['camera_info_topic']], 'image_association.log')
                    deadline=time.monotonic()+90
                    with condition:
                        while image_status[0].get('state')!='ready':
                            if image_node.poll() is not None or time.monotonic()>=deadline:
                                raise RuntimeError('YOLO node did not become ready; inspect image_association.log')
                            condition.wait(.1)
            if args.rviz:
                from sensor_msgs.msg import PointCloud2
                from perception.visualization.scans import SCAN_TOPIC
                rospy.wait_for_message(SCAN_TOPIC, PointCloud2, timeout=15)
            if not live:
                command = ['rosbag', 'play', '--quiet', '--clock', '--rate', str(args.rate), '--delay', '1']
                if args.start:
                    command += ['--start', str(args.start)]
                if args.duration is not None:
                    command += ['--duration', str(args.duration)]
                topics = replay_topics(args, settings.get('algorithm_parameters', {}),
                                       settings.get('people_transport'), environment_config)
                settings['replayed_topics'] = topics
                required = [t for t in topics if t != '/tf_static' and
                            t != settings.get('people_transport', {}).get('rgb_topic')]
                if any(topic not in bag_topics for topic in required):
                    raise ValueError('Bag is missing required input topics: ' + str(required))
                command += [str(bag), '--topics'] + topics
            replay_started = time.monotonic()
            if live:
                ending = 'saves the map and stops this session' if args.mode == 'mapping' else 'stops this session'
                print('Session ready: %s, %s. Ctrl+C %s.' % (args.mode, args.backend, ending), flush=True)
                try:
                    while args.duration is None or time.monotonic()-replay_started < args.duration:
                        if any(p is not None and p.poll() is not None for p in (algorithm, frontend, people, image_node, display)):
                            raise RuntimeError('An algorithm node exited; inspect its log')
                        time.sleep(.1)
                except KeyboardInterrupt:
                    pass
            elif args.step_replay:
                import rosbag
                from sensor_msgs.msg import PointCloud2
                from rosgraph_msgs.msg import Clock
                clock_pub = rospy.Publisher('/clock', Clock, queue_size=1, latch=True)
                scan_pub = rospy.Publisher('/rslidar_points', PointCloud2, queue_size=1)
                deadline = time.monotonic()+15
                while scan_pub.get_num_connections() == 0:
                    if algorithm.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('PIN did not subscribe to the raw LiDAR publisher')
                    time.sleep(.1)
                published = 0
                settings['replay_policy'] = 'one_scan_one_ack_no_transport_frame_skipping'
                with rosbag.Bag(str(bag)) as recording:
                    settings['bag_lidar_frames'] = recording.get_message_count('/rslidar_points')
                    start = recording.get_start_time()
                    for _, cloud, recorded_time in recording.read_messages(topics=['/rslidar_points']):
                        if args.duration is not None and recorded_time.to_sec()-start >= args.duration:
                            break
                        with condition:
                            before = receipt['count']
                        clock_pub.publish(Clock(clock=cloud.header.stamp))
                        scan_pub.publish(cloud)
                        wait_for_scan_ack(condition, receipt, cloud.header.stamp.to_nsec(), before,
                                          lambda: algorithm.poll() is None)
                        published += 1
                        if published % 100 == 0:
                            print(json.dumps({'processed': published, 'valid': counts['valid'],
                                'invalid': counts['invalid'], 'wall_s': time.monotonic()-replay_started}), flush=True)
                settings['published_lidar_frames'] = published
                if args.duration is None and published != settings['bag_lidar_frames']:
                    raise RuntimeError('Incomplete raw LiDAR bag consumption')
            else:
                playback = launch(command, 'playback.log')
                while playback.poll() is None:
                    if any(p is not None and p.poll() is not None for p in (algorithm, frontend, people, image_node, display)):
                        raise RuntimeError('An algorithm node exited during replay; inspect its log')
                    time.sleep(.1)
                if playback.returncode != 0:
                    raise RuntimeError('rosbag replay failed')
            # Drain the bounded algorithm queues without reusing ROS time after /clock stops.
            if not pin and args.mode != 'odometry':
                if live:
                    read_native_status()
                else:
                    wait_for_native_drain(read_native_status,
                        lambda: all(p is None or p.poll() is None for p in (algorithm, frontend)),
                        last_state_wall)
            drain_deadline = time.monotonic() + (120 if pin else 20)
            while not live and (pin or args.mode == 'odometry') and time.monotonic() < drain_deadline:
                if any(p is not None and p.poll() is not None for p in (algorithm, frontend)):
                    raise RuntimeError('Algorithm exited while draining its input queue')
                busy = False
                if pin:
                    response = pin_status()
                    if not response.success:
                        raise RuntimeError('PIN worker reported a fatal error: ' + response.message)
                    settings['backend_statistics'] = json.loads(response.message)
                    busy = settings['backend_statistics']['busy']
                if not busy and last_state_wall[0] is not None and time.monotonic()-last_state_wall[0] > 2:
                    break
                time.sleep(.2)
            else:
                if not live and (pin or args.mode == 'odometry'):
                    raise RuntimeError('Algorithm did not drain within the replay budget')
            if not pin and rospy.has_param('/initialization/result'):
                settings['initialization_result'] = rospy.get_param('/initialization/result')
            settings['session_wall_s' if live else 'replay_wall_s'] = time.monotonic()-replay_started
            if args.assume_stationary:
                response = synthetic_status()
                settings['stationary_input'] = json.loads(response.message)
                if not response.success or not settings['stationary_input']['scans']:
                    raise RuntimeError('Stationary input failed or did not receive scans: ' + response.message)
            if counts['localization'] == 0:
                raise RuntimeError('No localization output was received')
            if counts['valid'] == 0:
                raise RuntimeError('No accepted registration; inspect algorithm diagnostics')
            if args.people_python and counts['people_valid'] == 0:
                raise RuntimeError('No localized detector output was received')
            if args.image_association and not live:
                wait_for_image_drain(condition,lambda:image_status[0],last_people,lambda:counts['rgb_records'],
                                     lambda:image_node.poll() is None and people.poll() is None)
            if args.environment:
                deadline = time.monotonic()+15
                while True:
                    response = environment_status()
                    settings['environment'] = json.loads(response.message)
                    if not response.success:
                        raise RuntimeError('Environment geometry failed: '+response.message)
                    if not settings['environment']['busy'] and not settings['environment']['pending']:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Environment observation queue did not drain')
                    time.sleep(.1)
                if not settings['environment']['counts']['processed']:
                    raise RuntimeError('No observation-bound environment geometry was generated')
            if args.scene:
                deadline = time.monotonic()+15
                while True:
                    response = scene_status()
                    settings['scene_statistics'] = json.loads(response.message)
                    if not response.success:
                        raise RuntimeError('Scene assembly failed: '+response.message)
                    if not settings['scene_statistics']['busy'] and not settings['scene_statistics']['pending']:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Scene queue did not drain')
                    time.sleep(.1)
                if not settings['scene_statistics']['counts']['published']:
                    raise RuntimeError('No observation-bound scene was assembled')
            if args.mode == 'mapping' and not (map_path / 'COMPLETE').is_file():
                if pin:
                    response = rospy.ServiceProxy('/pin_slam/save_map', Trigger)()
                    settings['backend_statistics'] = json.loads(pin_status().message)
                else:
                    from roll.srv import save_map
                    response = rospy.ServiceProxy('/roll/save_map', save_map)(0, 0, 0, 0)
                if not response.success:
                    raise RuntimeError('Map save failed; inspect algorithm diagnostics')
            settings['success'] = True
        except BaseException as error:
            settings['success'] = False
            settings['error'] = str(error)
            raise
        finally:
            cleanup_errors = []
            try:
                # The optional view owns no map output. Stop its own process group
                # directly, without waiting for ROS timers on a paused bag clock.
                stop(view, first_signal=signal.SIGTERM)
            except Exception as error:
                cleanup_errors.append(str(error))
            for process in (playback, display, people):
                try:
                    stop(process)
                except Exception as error:
                    cleanup_errors.append(str(error))
            if live and image_node is not None and last_people[0] is not None and image_node.poll() is None:
                try:
                    wait_for_image_drain(condition,lambda:image_status[0],last_people,lambda:counts['rgb_records'],
                                         lambda:image_node.poll() is None)
                except Exception as error:cleanup_errors.append(str(error))
            for process in (image_node,frontend,algorithm):
                try:stop(process)
                except Exception as error:cleanup_errors.append(str(error))
            try:
                stop(display_recording)
                if args.record_display:
                    settings['display_recording_fingerprint'] = wait_for_display_recording(output/'display.bag')
            except Exception as error:
                cleanup_errors.append(str(error))
            if 'rospy' in locals():
                rospy.signal_shutdown('Replay finished')
            try:
                stop(master)
            except Exception as error:
                cleanup_errors.append(str(error))
            for stream in streams:
                stream.close()
            settings['counts'] = counts
            settings['cleanup_errors'] = cleanup_errors
            if args.mode == 'mapping':
                settings['map_saved'] = (map_path / 'COMPLETE').is_file()
                settings['success'] = settings.get('success', False) and settings['map_saved']
            settings['success'] = settings.get('success', False) and not cleanup_errors
            if args.image_association:
                metadata=dict(image_status[0],records='rgb_associations.jsonl')
                metadata['complete']=bool(settings['success'] and metadata.get('state')=='ready')
                metadata['recorded']=counts['rgb_records']
                metadata['not_processed']=max(0,counts['people']-counts['rgb_records'])
                path=output/'rgb_associations.jsonl'
                if path.is_file():
                    from perception.detection_tracking.inference.rgb_cache import file_digest
                    metadata['records_sha256']=file_digest(path)
                settings['image_association']=metadata
            if map_path is not None and (map_path / 'COMPLETE').is_file():
                try:
                    actual_fingerprint = map_fingerprint(map_path)
                    if args.mode == 'localization' and settings['playback']['map_fingerprint'] != actual_fingerprint:
                        raise ValueError('Read-only localization map changed during this session')
                    settings['playback']['map_fingerprint'] = actual_fingerprint
                except (OSError, ValueError) as error:
                    settings['success'] = False
                    settings['error'] = 'Cannot bind playback map: ' + str(error)
            if args.record_environment_scans and settings.get('success'):
                try:
                    from perception.environment.scans import sealed_scan_summary
                    scans = output/'environment/scans.sqlite'
                    archive=sealed_scan_summary(scans)
                    final_environment=json.loads((output/'environment/summary.json').read_text())
                    recording=final_environment['scan_recording']
                    if recording.get('state','sealed')!='sealed' or recording.get('error'):
                        raise ValueError('Environment reported an incomplete or failed archive')
                    if archive['records']!=final_environment['counts']['processed']:
                        raise ValueError('Sealed scan journal does not cover all processed environment observations')
                    settings['environment']={key:value for key,value in final_environment.items() if key!='config'}
                    settings['environment_scans_fingerprint'] = file_fingerprint(scans)
                except (OSError, ValueError) as error:
                    settings.update(success=False, error='Invalid scan journal: '+str(error))
            finalize_artifacts(output, settings, keep_details=args.keep_details)
    summary = {'success': settings['success'], 'counts': counts, 'out': str(output)}
    if settings['success'] and args.decision_project:
        summary['decision_preparation'] = prepare_decision_project(args,output)
    print(json.dumps(summary,
                     ensure_ascii=False), flush=True)
    if settings['success'] and args.mode == 'mapping' and args.rviz:
        # Read the frozen map, not historical per-scan estimates which are not
        # retrospectively corrected by loop closure. Computation is already stopped.
        return show_final_map(map_path)
    return 0 if settings['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
