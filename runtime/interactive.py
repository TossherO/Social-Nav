"""RViz-selected localization, live goals, and paused-bag planning review."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import yaml

from .command import send_request

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def session_inputs(config_path):
    from .static_map import load_static_map
    from perception.mapping_localization.coverage import PriorCoverage, asset_digest, full_map_coverage
    from perception.mapping_localization.results import map_fingerprint
    config = yaml.safe_load(Path(config_path).read_text())
    config = dict({key: str(resolve(config[key])) for key in
              ('map', 'static_map', 'initial_pose_hint', 'motion_config')},
              coverage=str(resolve(config['coverage'])) if config.get('coverage') else None,
              catalog=str(resolve(config['catalog'])) if config.get('catalog') else None,
              motion_profile=config.get('motion_profile'))
    if not Path(config['map'], 'COMPLETE').is_file():
        raise ValueError('Select a complete localization map')
    asset = load_static_map(config['static_map'], config['catalog'])
    fingerprint = map_fingerprint(config['map'])
    if asset.manifest['source'].get('map_fingerprint') != fingerprint:
        raise ValueError('Static environment and localization map do not match')
    if config['coverage']:
        PriorCoverage.load(config['coverage'], asset_fingerprint=asset_digest(config['map']))
    else:
        full_map_coverage(config['map'])
    hint = json.loads(Path(config['initial_pose_hint']).read_text())
    if hint['map']['fingerprint'] != fingerprint:
        raise ValueError('Initial height/attitude hint belongs to another map')
    return config, asset, hint


def selected_initial_pose(frame, position, quaternion, hint, *, initial_z=None):
    """RViz selects the LiDAR XY/yaw; map Z and tilt remain explicit seed guesses."""
    from perception.mapping_localization.initial_pose import quaternion_rpy
    if frame != hint['map']['frame_id']:
        raise ValueError('Initial pose must use the selected map frame')
    if (len(position) != 3 or len(quaternion) != 4 or
            not np.isfinite([*position, *quaternion]).all() or
            abs(np.dot(quaternion, quaternion)-1) > 1e-3):
        raise ValueError('Initial pose must be finite with a normalized quaternion')
    roll, pitch, _ = quaternion_rpy(hint['pose']['quaternion_xyzw'])
    yaw = quaternion_rpy(quaternion)[2]
    z = hint['pose']['position_m'][2] if initial_z is None else initial_z
    if not math.isfinite(z):
        raise ValueError('Initial map Z must be finite')
    return [roll, pitch, yaw, float(position[0]), float(position[1]), float(z)]


def selected_initial_height(grid, hint, position, *, initial_z=None, observed_height=None, height_range=(.2,2.2)):
    """Choose a LiDAR map-Z seed and retain its source, never a localization result."""
    if initial_z is not None:
        if not math.isfinite(initial_z):raise ValueError('Initial map Z must be finite')
        return dict(status='available',method='explicit_initial_z',lidar_z_m=float(initial_z))
    from perception.mapping_localization.initial_pose import map_ground_initial_height, supported_ground_height
    if observed_height is not None:
        try:
            ground=supported_ground_height(grid,position[:2])
            height=float(observed_height['height_m'])
            if (observed_height.get('status')=='available' and observed_height.get('source')=='current_lidar_ground_tracker' and
                    type(observed_height.get('source_stamp_s')) in (int,float) and math.isfinite(observed_height['source_stamp_s']) and
                    grid.metadata.get('frame_id') == hint['map']['frame_id'] and ground['status']=='available' and
                    math.isfinite(height) and height_range[0] <= height <= height_range[1]):
                return dict(status='available',method='static_ground_plus_current_lidar_ground',
                    lidar_z_m=ground['ground_z_m']+height,target_ground=ground,
                    static_snapshot_id=grid.metadata.get('snapshot_id'),observed_height=observed_height)
        except (KeyError, TypeError, ValueError):
            pass
    try:
        result=map_ground_initial_height(grid,hint,position[:2],min_offset_m=height_range[0],max_offset_m=height_range[1])
    except (KeyError, TypeError, ValueError) as error:
        result=dict(status='unavailable',reason='invalid_static_ground_or_height_hint',error_type=type(error).__name__)
    if result['status'] == 'available':
        result['reference_source']=hint.get('source')
        return result
    fallback=float(hint['pose']['position_m'][2])
    if not math.isfinite(fallback):raise ValueError('Initial map Z must be finite')
    return dict(status='fallback',method='fixed_map_bound_hint',lidar_z_m=fallback,
                reason=result.get('reason','ground_estimate_unavailable'),ground_diagnostic=result,
                reference_source=hint.get('source'))


def bag_initial_pose(path, config, *, bag, start, live=False):
    """Reuse the existing seed validator; a saved bag pose never initializes live input."""
    from perception.mapping_localization.initial_pose import load_seed
    if live or not bag:
        raise ValueError('--pose-file is only valid for the matching bag and start offset')
    return load_seed(resolve(path), config['map'], resolve(bag), start)


def handoff_initial_pose(value, asset, hint):
    """Apply map-specific height and attitude to a provisional LiDAR XY/yaw."""
    from perception.mapping_localization.initial_pose import quaternion_rpy
    transform=value.get('destination') or {};pose=value.get('lidar_xy_yaw')
    if (value.get('verified_localization') is not False or
            value.get('source') not in ('agent_destination_scan_and_map','standard_destination_start') or not value.get('scan_frame') or
            not all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in ('scan_stamp_s','exit_stamp_s')) or
            value['scan_stamp_s']<=value['exit_stamp_s'] or
            transform.get('to_map_id')!=asset.snapshot.metadata.get('map_id') or
            transform.get('to_floor')!=value.get('floor_id') or
            not isinstance(pose,list) or len(pose)!=3 or
            not all(type(v) in (int,float) and math.isfinite(v) for v in pose)):
        raise ValueError('Floor handoff seed does not match the destination localization map')
    if value['source']=='standard_destination_start':
        full=value.get('lidar_pose_rpy_xyz')
        if (not isinstance(full,list) or len(full)!=6 or
                not all(type(v) in (int,float) and math.isfinite(v) for v in full) or
                not np.allclose([full[3],full[4],full[2]],pose,rtol=0,atol=1e-9)):
            raise ValueError('Invalid standard destination LiDAR pose')
        return [float(x) for x in full]
    height=selected_initial_height(asset.snapshot,hint,pose[:2])
    roll,pitch,_=quaternion_rpy(hint['pose']['quaternion_xyzw'])
    return [roll,pitch,pose[2],pose[0],pose[1],height['lidar_z_m']]


def require_rviz_tools(python='/usr/bin/python3'):
    from .rviz_tools import verify_rviz_tools
    verify_rviz_tools(python)


def session_notice(status, *, selected=False, candidate=False):
    """Operator-visible state independent of the missing map-to-local transform."""
    if not selected:
        if candidate:
            return ('initial_pose_preview', 'PREVIEW ONLY: align the scan with the map.\nReselect as needed, then Confirm Initial Pose or c + Enter.', False)
        return ('waiting_for_initial_pose', 'Use Ground Pose Estimate to select\nthe actual LiDAR XY and heading on the ground.', False)
    observation = (status or {}).get('observation') or {}
    row = observation.get('localization') or {}
    if not row:
        return ('initializing', 'Starting perception / waiting for localization...', False)
    paused = bool(status.get('paused'))
    prefix = 'PAUSED: ' if paused else ''
    if row.get('health') != 0:
        return ('localization_initializing', prefix+'LiDAR / IMU initialization is not ready.', True)
    if not row.get('has_map_pose'):
        return ('map_unlocalized', prefix+'MAP LOCALIZATION NOT ESTABLISHED\nCheck initial pose (XY / yaw) and restart.\nNo map transform; map goals unavailable.', True)
    if not (observation.get('prior') or {}).get('valid'):
        return ('map_association_unavailable', prefix+'Map association is not valid for planning.\nCheck coverage and association freshness.', True)
    if paused:
        if status.get('source_ended'):
            return ('paused_ready', 'END OF BAG: select a goal for frozen planning.', False)
        return ('paused_ready', 'PAUSED: select a goal for frozen planning.\nr + Enter resumes playback.', False)
    if status.get('live'):
        return ('ready', 'Map localization ready. Select a goal.\ns cancels; q exits.', False)
    return ('ready', 'Map localization ready. Select a goal.\np + Enter pauses; s cancels; q exits.', False)


def panel_controls(status, *, live, execute, preview_selected, preview_ready, runtime_ready):
    status = status or {}
    ended = bool(status.get('source_ended'))
    paused = status.get('pause_requested', status.get('paused', False))
    playback_available = not live and bool(status) and not ended
    shadow = status.get('shadow_navigation') or {}
    return dict(adjust=preview_selected, confirm=preview_ready,
                pause=playback_available and not paused, resume=playback_available and paused,
                review=not live and bool(status.get('review_available')),
                shadow_start=not live and bool(shadow.get('available')) and not shadow.get('enabled', False),
                shadow_stop=not live and bool(shadow.get('enabled')),
                arm=live and execute and bool(status), cancel=runtime_ready, exit=True)


def cancel_operator_goal(bridge, send, *, bag_ready, execute):
    try:
        if bridge.last_submitted is not None:
            return bridge.cancel()
        if bag_ready:
            return send('shadow_navigation', dict(enabled=False))
        return dict(accepted=True)
    finally:
        if execute:
            send('arm', dict(enabled=False))


def motion_selection(config, *, path=None, profile=None):
    """A custom file uses its own default unless a profile is explicitly selected."""
    if profile is None and path is None:profile=config.get('motion_profile')
    return resolve(path or config['motion_config']), profile


def adjust_initial_pose(pose, command, *, step_m=.05, step_rad=math.pi/36):
    """Adjust the unconfirmed LiDAR pose in its forward/left frame."""
    value=list(pose)
    if len(value)!=6 or not all(math.isfinite(v) for v in value):
        raise ValueError('Initial pose is unavailable or invalid')
    if command in ('w','s','a','d'):
        forward={'w':1,'s':-1}.get(command,0)*step_m
        left={'a':1,'d':-1}.get(command,0)*step_m
        yaw=value[2]
        value[3]+=forward*math.cos(yaw)-left*math.sin(yaw)
        value[4]+=forward*math.sin(yaw)+left*math.cos(yaw)
    elif command in ('q','e'):
        value[2]=math.atan2(math.sin(value[2]+(step_rad if command=='q' else -step_rad)),
                            math.cos(value[2]+(step_rad if command=='q' else -step_rad)))
    else:raise ValueError('Unknown initial pose adjustment')
    return value


def runtime_command(args, config, pose):
    motion_path, profile = motion_selection(config, path=args.motion_config, profile=args.motion_profile)
    command = [sys.executable, '-B', '-m', 'runtime', '--mode', 'localization',
               '--map', config['map'], '--static-map', config['static_map'],
               '--pose', *map(str, pose),
               '--motion-config', str(motion_path),
               '--output', str(Path(args.output)/'run'), '--display', '--interactive-session', '--record-planning']
    if config.get('coverage'):command += ['--coverage', config['coverage']]
    if config.get('catalog'):command += ['--catalog', config['catalog']]
    if profile is not None:command += ['--motion-profile', profile]
    if args.live:
        command += ['--live']
    else:
        command += ['--bag', str(resolve(args.bag)), '--start', str(args.start), '--rate', str(args.rate), '--bag-shadow']
        if getattr(args,'shadow_route_run',None):
            command += ['--shadow-route-run', str(resolve(args.shadow_route_run))]
        if args.pause_at is not None:
            command += ['--pause-at', str(args.pause_at)]
    if args.gpu is not None:
        command += ['--gpu', args.gpu]
    if args.runtime_config:
        command += ['--runtime-config', str(resolve(args.runtime_config))]
    if getattr(args,'agent_observations',False):
        command += ['--agent-observations']
    if getattr(args,'agent_follow_candidates',False):
        command += ['--agent-follow-candidates']
        # Freeze the effective backend for later floor handoffs even when the
        # original launcher relied on the local override or the default.
        from decision.vlm.t2.hosted import follow_settings
        backend,weights=follow_settings(getattr(args,'t2_backend',None),weights=getattr(args,'t2_reid_weights',None),
                                        raw_rgb=getattr(args,'t2_raw_rgb',False))
        command += ['--t2-backend',backend]
        if weights is not None:command += ['--t2-reid-weights',weights]
        if backend!='reid' and getattr(args,'t2_jpeg_quality',None) is not None:
            command += ['--t2-jpeg-quality',str(args.t2_jpeg_quality)]
    if getattr(args,'t2_raw_rgb',False):command += ['--t2-raw-rgb']
    if getattr(args,'record_t2_requests',False):command += ['--record-t2-requests']
    if getattr(args,'test_person_extraction',False):command += ['--test-person-extraction']
    if getattr(args,'agent_seating',False):
        command += ['--agent-seating']
    if getattr(args,'agent_execution',False):
        command += ['--agent-execution']
    if getattr(args,'trace_timing',False):
        command += ['--trace-timing']
    if getattr(args,'roll_gravity_deg',None) is not None:
        command += ['--roll-gravity-deg',str(args.roll_gravity_deg)]
    if args.environment_update_hz is not None:
        command += ['--environment-update-hz', str(args.environment_update_hz)]
    if args.environment_window_m is not None:
        command += ['--environment-window-m', str(args.environment_window_m)]
    if args.vehicle_profile:
        command += ['--vehicle-profile', str(resolve(args.vehicle_profile))]
    if args.execute:
        command += ['--execute', '--chassis-port', args.chassis_port]
    return command


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag')
    source.add_argument('--live', action='store_true')
    p.add_argument('--config', default=str(ROOT/'navigation/planning/configs/interactive.yaml'))
    p.add_argument('--output', required=True)
    p.add_argument('--gpu')
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--rate', type=float, default=1.)
    p.add_argument('--pause-at', type=float)
    p.add_argument('--shadow-route-run', help='Reuse fixed-map localization recorded for this bag and map')
    p.add_argument('--initial-z', type=float, help='LiDAR map Z initial guess, overriding the selected map hint')
    p.add_argument('--pose-file', help='Explicit saved LiDAR start, validated against this bag, map and --start; skips mouse initialization')
    p.add_argument('--handoff-seed', help='Provisional cross-floor LiDAR seed for a nonexecuting live Agent observation')
    p.add_argument('--motion-config')
    p.add_argument('--motion-profile')
    p.add_argument('--runtime-config', help='Same machine scheduling override inspected by runtime.preflight')
    p.add_argument('--agent-observations', action='store_true', help='Publish bounded RGB observations for the nonexecuting Agent')
    p.add_argument('--agent-follow-candidates', action='store_true', help='Prepare optional YOLO person candidates for Agent follow selection')
    p.add_argument('--t2-backend',choices=('dspark','hosted','reid'),help='Select the T2 decision backend')
    p.add_argument('--t2-reid-weights',help='Local OSNet .onnx (OpenCV CPU) or .engine (TensorRT CUDA) for reid T2')
    p.add_argument('--t2-jpeg-quality',type=int,help='Optional T2 JPEG quality (1..95)')
    p.add_argument('--t2-raw-rgb',action='store_true',help='Experimental nonexecuting T2 without 2D detection')
    p.add_argument('--record-t2-requests',action='store_true',help='Save T2 inputs and results including RGB locally')
    p.add_argument('--agent-seating',action='store_true',help='Prepare RGB and lidar for current table-edge proposals')
    p.add_argument('--agent-execution', action='store_true', help='Allow Agent goal intents in this manually armed execution session')
    p.add_argument('--trace-timing', action='store_true', help='Record bounded scalar stage timings for diagnosis')
    p.add_argument('--roll-gravity-deg', type=float, help='Fixed-map gravity correction limit in degrees (bag default: 8)')
    p.add_argument('--environment-update-hz', type=float, help='Override environment update rate (at most 10 Hz)')
    p.add_argument('--environment-window-m', type=float, help='Override the square local environment window in metres')
    p.add_argument('--vehicle-profile')
    p.add_argument('--test-person-extraction',action='store_true',
                   help='Bag only: accept all confidence scores for tracked person point extraction')
    p.add_argument('--execute', action='store_true', help='Enable supervised live hardware output; separate arming is still required')
    p.add_argument('--chassis-port')
    p.add_argument('--no-rviz', action='store_true', help='Use an already open RViz or ROS test publisher')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not math.isfinite(args.start) or args.start < 0 or not math.isfinite(args.rate) or args.rate <= 0:
        p.error('Bag start/rate must be finite and nonnegative/positive')
    if args.test_person_extraction and not args.bag:
        p.error('--test-person-extraction requires --bag')
    if args.environment_update_hz is not None and (not math.isfinite(args.environment_update_hz) or not 0 < args.environment_update_hz <= 10):
        p.error('Environment update rate must be finite and in (0,10] Hz')
    if args.environment_window_m is not None and (not math.isfinite(args.environment_window_m) or not 0 < args.environment_window_m <= 60):
        p.error('Environment window must be finite and in (0,60] metres')
    if args.roll_gravity_deg is not None and (not math.isfinite(args.roll_gravity_deg) or not 0 < args.roll_gravity_deg <= 15):
        p.error('--roll-gravity-deg requires a finite value in (0,15] degrees')
    if args.live and (args.start != 0 or args.rate != 1 or args.pause_at is not None or args.shadow_route_run):
        p.error('Live input does not accept bag playback options')
    if args.handoff_seed and (not args.live or not args.agent_observations or args.execute or args.pose_file or args.initial_z is not None):
        p.error('--handoff-seed requires nonexecuting live Agent observations, without --pose-file or --initial-z')
    if args.pause_at is not None and (not math.isfinite(args.pause_at) or args.pause_at < 0):
        p.error('--pause-at must be finite and nonnegative')
    if args.execute and not (args.live and args.vehicle_profile and args.chassis_port):
        p.error('--execute requires --live, --vehicle-profile and --chassis-port')
    if args.agent_execution and not (args.execute and args.agent_observations):
        p.error('--agent-execution requires --execute and --agent-observations')
    if args.agent_follow_candidates and not args.agent_observations:
        p.error('--agent-follow-candidates requires --agent-observations')
    if args.t2_backend and not args.agent_follow_candidates:
        p.error('--t2-backend requires --agent-follow-candidates')
    if args.t2_reid_weights is not None and not args.agent_follow_candidates:
        p.error('--t2-reid-weights requires --agent-follow-candidates')
    if args.t2_raw_rgb and (not args.agent_follow_candidates or args.execute or args.agent_execution):
        p.error('--t2-raw-rgb requires follow candidates and nonexecuting observation')
    if args.t2_jpeg_quality is not None and (not args.agent_follow_candidates or not 1<=args.t2_jpeg_quality<=95):
        p.error('--t2-jpeg-quality requires --agent-follow-candidates and quality in 1..95')
    if args.record_t2_requests and not args.agent_follow_candidates:
        p.error('--record-t2-requests requires --agent-follow-candidates')
    if args.agent_follow_candidates:
        from decision.vlm.t2.hosted import follow_settings
        try:
            args.t2_backend,args.t2_reid_weights=follow_settings(args.t2_backend,weights=args.t2_reid_weights,
                                                                raw_rgb=args.t2_raw_rgb)
        except ValueError as error:
            p.error(str(error))
    if args.agent_seating and not (args.live and args.agent_observations):
        p.error('--agent-seating requires --live and --agent-observations')
    if args.chassis_port and not args.execute:
        p.error('--chassis-port is only used with explicit --execute')
    if not args.no_rviz:
        from .config import runtime_settings
        require_rviz_tools(runtime_settings(resolve(args.runtime_config or 'configs/runtime.yaml'))['input_python'])
    config, asset, hint = session_inputs(args.config)
    seed = None
    if args.pose_file:
        if args.initial_z is not None:p.error('--initial-z cannot modify a saved --pose-file')
        seed = bag_initial_pose(args.pose_file,config,bag=args.bag,start=args.start,live=args.live)
    if args.handoff_seed:
        seed=handoff_initial_pose(json.loads(resolve(args.handoff_seed).read_text()),asset,hint)
    suggested = None
    if not args.live:
        try:suggested = bag_initial_pose(config['initial_pose_hint'],config,bag=args.bag,start=args.start)
        except ValueError:pass  # A different bag/offset has no validated start hint.
    if args.vehicle_profile:
        from navigation.control.vehicle import load_vehicle_profile
        from navigation.planning.motion import load_motion_config
        motion_path, profile = motion_selection(config, path=args.motion_config, profile=args.motion_profile)
        load_vehicle_profile(resolve(args.vehicle_profile), load_motion_config(motion_path,profile=profile),
                             require_execution=args.execute)
    output = resolve(args.output); output.mkdir(parents=True, exist_ok=False); args.output = str(output)
    (output/'session.json').write_text(json.dumps(dict(source='live' if args.live else 'bag', config=config,
        start=args.start, rate=args.rate, pause_at=args.pause_at, execute_requested=args.execute,
        owner_pid=os.getpid(),agent_observations=args.agent_observations,
        initial_pose_reference='rslidar', map_frame=asset.manifest['frame_id'],
        navigation_window=asset.navigation_window,
        initial_guess_source=config['initial_pose_hint'],pose_file=args.pose_file,handoff_seed=args.handoff_seed), ensure_ascii=False, indent=2))

    import rospy
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped, Point
    from nav_msgs.msg import OccupancyGrid, Path as RosPath
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Empty, String
    from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
    from visualization_msgs.msg import Marker, MarkerArray
    import rosgraph
    try:
        rosgraph.Master('/social_nav_interactive').getPid()
    except Exception as error:
        raise RuntimeError('ROS master unavailable; source ros1_env.sh and start roscore first') from error
    rospy.init_node('social_nav_interactive', disable_signals=True)
    stopped = threading.Event(); selection = []; lock = threading.Lock()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    state_pub = rospy.Publisher('/social_nav_interactive/state', String, queue_size=1, latch=True)
    panel_state_pub = rospy.Publisher('/social_nav_interactive/panel_state', String, queue_size=1, latch=True)
    panel_reply_pub = rospy.Publisher('/social_nav_interactive/panel_reply', String, queue_size=10)
    map_pub = rospy.Publisher('/social_nav_interactive/map', OccupancyGrid, queue_size=1, latch=True)
    static_basemap_pub = rospy.Publisher('/social_nav_interactive/static_basemap', PointCloud2, queue_size=1, latch=True)
    terrain_pub = rospy.Publisher('/social_nav_interactive/ground_map', PointCloud2, queue_size=1, latch=True)
    reference_pub = rospy.Publisher('/social_nav_interactive/preview_reference', RosPath, queue_size=1, latch=True)
    trajectory_pub = rospy.Publisher('/social_nav_interactive/preview_trajectory', RosPath, queue_size=1, latch=True)
    label_pub = rospy.Publisher('/social_nav_interactive/preview_status', Marker, queue_size=1, latch=True)
    session_pub = rospy.Publisher('/social_nav_interactive/session_markers', MarkerArray, queue_size=1, latch=True)
    curves_pub = rospy.Publisher('/social_nav_interactive/preview_paths', MarkerArray, queue_size=1, latch=True)
    goal_pub = rospy.Publisher('/social_nav_interactive/accepted_goal', MarkerArray, queue_size=1, latch=True)
    initial_scan_pub = rospy.Publisher('/social_nav_interactive/initial_scan_preview', PointCloud2, queue_size=1, latch=True)
    initial_marker_pub = rospy.Publisher('/social_nav_interactive/initial_pose_preview', MarkerArray, queue_size=1, latch=True)
    message = OccupancyGrid(); grid = asset.navigation_snapshot
    message.header.frame_id = grid.metadata['frame_id']; message.header.stamp = rospy.Time.from_sec(grid.metadata['stamp_s'])
    message.info.resolution = grid.metadata['resolution']; message.info.height, message.info.width = grid.layers['occupancy'].shape
    message.info.origin.position.x, message.info.origin.position.y = grid.metadata['origin_xy']; message.info.origin.orientation.w = 1.
    message.data = grid.layers['occupancy'].reshape(-1).tolist(); map_pub.publish(message)
    from perception.visualization.surface import ground_cloud,annotation_height
    static_basemap_pub.publish(ground_cloud(grid,offset=-.015,fill_known=True))
    terrain_pub.publish(ground_cloud(grid,offset=-.015))
    runtime = rviz = log = None
    from .initial_preview import PreviewState, PreviewGroundEstimator, preview_points
    from perception.visualization.scans import pointcloud_message
    from perception.detection_tracking.inference.config import load_config as detection_config
    sensor_settings = detection_config()['ros']
    initial_preview = PreviewState()
    review_frame = None
    review_corners = []
    review_lock = threading.Lock()
    from perception.mapping_localization.initial_pose import quaternion_rpy
    from perception.environment.layers import load_config as environment_config
    initial_roll,initial_pitch,_ = quaternion_rpy(hint['pose']['quaternion_xyzw'])
    ground_config=environment_config() if args.live or not args.no_rviz else None
    initial_ground = PreviewGroundEstimator(initial_roll,initial_pitch,ground_config) if args.live and args.initial_z is None else None
    preview_visible = False
    socket_path = output/'run/commands.sock'
    # Instructions live in map itself, so they remain visible when localization fails.
    anchor = hint['pose']['position_m'][:2]
    anchor_z = annotation_height(grid,anchor,hint['pose']['position_m'][2])
    last_notice = None

    def show_notice(status=None):
        nonlocal last_notice
        state,text,error = session_notice(status,selected=bool(selection),candidate=args.live and initial_preview.pose is not None)
        key=(state,text)
        if key==last_notice:return
        last_notice=key
        label=Marker();label.header.frame_id=asset.manifest['frame_id']
        label.ns='session';label.id=0;label.type=Marker.TEXT_VIEW_FACING;label.action=Marker.ADD
        label.pose.orientation.w=1.;label.pose.position.x,label.pose.position.y=anchor
        label.pose.position.y-=3.  # Keep instructions beside, rather than above, the vehicle labels.
        label.pose.position.z=anchor_z+3.8;label.scale.z=.3
        label.color.r,label.color.g,label.color.b,label.color.a=(1.,.35,.2,1.) if error else (1.,1.,.7,1.)
        label.text=text
        markers=[label]
        if suggested is not None and not selection:
            arrow=Marker();arrow.header.frame_id=asset.manifest['frame_id'];arrow.ns='session';arrow.id=1
            arrow.type=Marker.ARROW;arrow.action=Marker.ADD
            arrow.pose.position.x,arrow.pose.position.y=suggested[3:5]
            arrow.pose.position.z=annotation_height(grid,suggested[3:5],suggested[5])+.12
            arrow.pose.orientation.z=math.sin(suggested[2]/2);arrow.pose.orientation.w=math.cos(suggested[2]/2)
            arrow.scale.x,arrow.scale.y,arrow.scale.z=1.,.18,.18
            arrow.color.r,arrow.color.g,arrow.color.b,arrow.color.a=.2,1.,.2,1.
            label.text+='\nGreen arrow: saved start for this bag / offset.'
            markers.append(arrow)
        else:
            arrow=Marker();arrow.header.frame_id=asset.manifest['frame_id'];arrow.ns='session';arrow.id=1;arrow.action=Marker.DELETE
            markers.append(arrow)
        session_pub.publish(MarkerArray(markers=markers))
        print('[%s] %s' % (state,text.replace('\n',' ')),flush=True)

    if seed is not None:
        selection.append(seed)
        (output/'initial_pose.json').write_text(json.dumps(dict(source='cross_floor_provisional_seed' if args.handoff_seed else 'validated_bag_pose_file',
            source_file=str(resolve(args.handoff_seed or args.pose_file)),reference='rslidar',map=config['map'],pose_rpy_xyz=seed,
            registration_verified=False),indent=2))

    def send(kind, value, *, timeout=2.):
        if runtime is None or runtime.poll() is not None or not socket_path.exists():
            raise ValueError('Initialize the LiDAR pose and wait for the runtime to start')
        return send_request(socket_path, kind, value, timeout=timeout)

    from navigation.planning.manual_goal import ManualGoals
    bridge = ManualGoals(send, task_id='interactive-'+uuid.uuid4().hex,
                         frame=asset.manifest['frame_id'], map_snapshot_id=asset.manifest['snapshot_id'])

    def report(action):
        try:
            result = action()
        except Exception as error:
            result = dict(accepted=False, error=str(error), executable=False)
            rospy.logwarn('%s', error)
        reply = result.get('reply')
        if isinstance(reply, dict) and 'accepted' in reply:
            result['accepted'] = reply['accepted']
            if 'error' in reply:
                result['error'] = reply['error']
        state_pub.publish(String(data=json.dumps(result, ensure_ascii=False, allow_nan=False)))
        return result

    last_panel_state = None

    def publish_panel_state(status=None):
        nonlocal last_panel_state
        with lock:
            preview_selected = args.live and not selection and initial_preview.pose is not None
            preview_ready = preview_selected and initial_preview.ready(time.monotonic())
        runtime_ready = runtime is not None and runtime.poll() is None and socket_path.exists()
        controls = panel_controls(status, live=args.live, execute=args.execute,
                                  preview_selected=preview_selected, preview_ready=preview_ready,
                                  runtime_ready=runtime_ready)
        status = status or {}
        playback_state = None
        if not args.live and status:
            playback_state = ('ended' if status.get('source_ended') else
                              'paused' if status.get('paused') else 'running')
        chassis = status.get('chassis') or {}
        shadow = status.get('shadow_navigation') or {}
        shadow_fields = ('state','reason')
        if shadow.get('state') == 'planning':
            shadow_fields += ('planning_status','planning_reason','dwa_status','dwa_reason')
        message = dict(controls=controls, playback_state=playback_state, hardware_enabled=args.execute,
                       chassis={key:chassis.get(key) for key in ('state','status_fresh') if key in chassis},
                       shadow_navigation={key:shadow[key] for key in shadow_fields if key in shadow})
        if message != last_panel_state:
            last_panel_state = message
            panel_state_pub.publish(String(data=json.dumps(message, ensure_ascii=False, allow_nan=False)))

    def show_initial_candidate(pose, error=None):
        marker=Marker();marker.header.frame_id=asset.manifest['frame_id']
        marker.ns='initial_candidate';marker.id=0;marker.action=Marker.ADD;marker.type=Marker.ARROW
        marker.pose.position=Point(*pose[3:]);marker.pose.orientation.z=math.sin(pose[2]/2)
        marker.pose.orientation.w=math.cos(pose[2]/2)
        marker.scale.x,marker.scale.y,marker.scale.z=1.,.12,.18
        marker.color.r,marker.color.g,marker.color.b,marker.color.a=0.,1.,1.,1.
        label=Marker();label.header.frame_id=marker.header.frame_id;label.ns=marker.ns;label.id=1
        label.action=Marker.ADD;label.type=Marker.TEXT_VIEW_FACING;label.pose.orientation.w=1.
        label.pose.position=Point(pose[3],pose[4],pose[5]+1.2);label.scale.z=.25
        label.color.r,label.color.g,label.color.b,label.color.a=0.,1.,1.,1.
        seed=initial_preview.height_seed or {}
        height=('HEIGHT FALLBACK: '+str(seed.get('reason')) if seed.get('status')=='fallback' else
                'Z %.3f m (%s)' % (pose[5],seed.get('method','fixed hint')))
        label.text=('CONFIRMATION FAILED: '+error if error else
                    'UNVERIFIED INITIAL GUESS\n'+height+'\nCyan: raw scan preview; confirm to start')
        initial_marker_pub.publish(MarkerArray(markers=[marker,label]))

    def initial(message):
        def choose():
            with lock:
                if runtime is not None or selection:
                    raise ValueError('This session is already initialized; stop and restart to change the initial pose')
                pose = message.pose.pose
                height_seed=(selected_initial_height(asset.snapshot,hint,[pose.position.x,pose.position.y],initial_z=args.initial_z,
                    observed_height=None if initial_ground is None else initial_ground.current(time.monotonic()),
                    height_range=(ground_config['ground_min_below_sensor_m'],ground_config['ground_max_below_sensor_m']))
                             if args.live else None)
                value = selected_initial_pose(message.header.frame_id,
                    [pose.position.x, pose.position.y, pose.position.z],
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], hint,
                    initial_z=height_seed['lidar_z_m'] if height_seed is not None else args.initial_z)
                if args.live:
                    initial_preview.select(value, message.header.stamp.to_sec(),height_seed=height_seed)
                    show_initial_candidate(value)
                    print('[initial_height] '+json.dumps(height_seed,ensure_ascii=False,allow_nan=False),flush=True)
                    return dict(accepted=True, state='initial_pose_preview', executable=False,
                                preview_revision=initial_preview.revision)
                selection.append(value)
                (output/'initial_pose.json').write_text(json.dumps(dict(source='rviz_initialpose',
                    reference='rslidar', map=config['map'], pose_rpy_xyz=value,
                    intent_stamp_s=message.header.stamp.to_sec(), registration_verified=False), indent=2))
                return dict(accepted=True, state='initializing', executable=False)
        report(choose)

    def scan_for_initial_preview(message):
        if not args.live or message.header.frame_id.lstrip('/') != sensor_settings['localization_child_frame'].lstrip('/'):
            return
        with lock:
            if selection or runtime is not None:
                return
            try:
                now=time.monotonic()
                if initial_preview.scan(message,now,now_source=rospy.Time.now().to_sec()) and initial_ground is not None:
                    initial_ground.scan(message,now,sensor_frame=sensor_settings['localization_child_frame'])
            except (ValueError, np.linalg.LinAlgError) as error: rospy.logwarn_throttle(5., 'Initial preview scan: %s', error)

    def draw_initial_preview():
        nonlocal preview_visible
        if not args.live: return
        with lock:
            now=time.monotonic()
            if selection or runtime is not None or initial_preview.pose is None or not initial_preview.fresh(now):
                if preview_visible:
                    initial_scan_pub.publish(pointcloud_message([], asset.manifest['frame_id']))
                    preview_visible=False
                if not initial_preview.fresh(now): initial_preview.rendered=None
                return
            try:
                points=preview_points(initial_preview.cloud, initial_preview.pose, sensor_settings['localization_child_frame'])
                initial_scan_pub.publish(pointcloud_message(points, asset.manifest['frame_id'],initial_preview.source_stamp))
                initial_preview.rendered=(initial_preview.revision,initial_preview.source_stamp,initial_preview.received_mono)
                preview_visible=True
            except ValueError as error:
                initial_preview.rendered=None
                if preview_visible:initial_scan_pub.publish(pointcloud_message([],asset.manifest['frame_id']));preview_visible=False
                rospy.logwarn_throttle(5., 'Initial preview: %s', error)

    def confirm_initial():
        with lock:
            if not args.live or runtime is not None or selection:
                raise ValueError('Initial pose confirmation is only available before live initialization')
            if not initial_preview.ready(time.monotonic()):
                raise ValueError('Select a pose and wait for a fresh LiDAR preview before confirming')
            pending=output/'initial_pose.tmp'
            pending.write_text(json.dumps(dict(source='confirmed_rviz_initialpose',
                reference='rslidar', map=config['map'], pose_rpy_xyz=initial_preview.pose,
                intent_stamp_s=initial_preview.intent_stamp, preview_scan_stamp_s=initial_preview.rendered[1],
                confirmed_monotonic_s=time.monotonic(),height_seed=initial_preview.height_seed,
                registration_verified=False), indent=2))
            pending.replace(output/'initial_pose.json')
            selection.append(list(initial_preview.pose))
            clear=Marker();clear.action=Marker.DELETEALL
            initial_marker_pub.publish(MarkerArray(markers=[clear]))
        return dict(accepted=True, state='initializing', executable=False)

    def confirm_initial_service(_):
        result=report(confirm_initial)
        return TriggerResponse(success=result.get('accepted',False),message=json.dumps(result,ensure_ascii=False))

    def confirm_initial_topic(_):
        result=report(confirm_initial)
        if not result.get('accepted') and initial_preview.pose is not None:
            show_initial_candidate(initial_preview.pose,result.get('error','try again'))

    def goal(message):
        pose = message.pose
        report(lambda: bridge.goal(frame=message.header.frame_id,
            position=[pose.position.x, pose.position.y, pose.position.z],
            quaternion=[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            stamp_s=message.header.stamp.to_sec()))

    def point(message):
        position = [message.point.x,message.point.y,message.point.z]
        def accept_point():
            with review_lock:
                if review_frame is not None:
                    if message.header.frame_id!=asset.manifest['frame_id'] or not all(map(math.isfinite,position)):
                        raise ValueError('请在地图坐标系内选择有效的对角点。')
                    status=send('session_status',{})
                    observed_frame=(status.get('epoch'),status.get('review_snapshot_id'))
                    if not status.get('review_available') or observed_frame!=review_frame:
                        raise ValueError('暂停画面已变化，请重新开始圈定。')
                    if not review_corners:
                        review_corners.append(position)
                        rospy.loginfo('已选第一个对角点，请选择矩形的另一角。')
                        return dict(accepted=True)
                    result=send('save_anomaly_review',dict(epoch=review_frame[0],snapshot_id=review_frame[1],frame=message.header.frame_id,
                                corners=[review_corners[0],position]),timeout=8.)
                    if result.get('accepted'):
                        review_corners.clear()
                        mark_review(result)
                        rospy.loginfo('异常审核证据已保存到 %s',result['path'])
                    return result
            return bridge.point(frame=message.header.frame_id,position=position,stamp_s=message.header.stamp.to_sec())
        report(accept_point)

    def mark_review(result=None):
        from geometry_msgs.msg import Point
        marker=Marker()
        marker.header.frame_id=asset.manifest['frame_id']
        marker.ns='anomaly_review'
        marker.action=Marker.DELETE if result is None else Marker.ADD
        marker.type=Marker.LINE_STRIP
        marker.pose.orientation.w=1.
        marker.scale.x=.04
        marker.color.r,marker.color.g,marker.color.b,marker.color.a=1.,.9,.2,1.
        if result is not None:
            x0,y0,x1,y1=result['bounds_xy']
            z=max(p[2] for p in result['corners_xyz'])+.06
            marker.points=[Point(x,y,z) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0))]
        label_pub.publish(marker)

    def begin_review():
        nonlocal review_frame
        if args.live:
            raise ValueError('区域审核只用于 bag 回放。')
        status=send('session_status',{})
        if not status.get('review_available'):
            raise ValueError('请暂停回放并等待地图定位就绪。')
        with review_lock:
            review_frame=(status['epoch'],status['review_snapshot_id'])
            review_corners.clear()
            mark_review()
        text='用 RViz 的 Publish Point 依次点击矩形的两个对角点。'
        rospy.loginfo('%s',text)
        return dict(accepted=True,message=text)

    def cancel():
        bag_ready=not args.live and runtime is not None and runtime.poll() is None and socket_path.exists()
        return cancel_operator_goal(bridge,send,bag_ready=bag_ready,execute=args.execute)

    def set_playback_paused(paused):
        nonlocal review_frame
        if args.live:
            raise ValueError('Playback controls require a bag session')
        result=send('playback', dict(paused=paused))
        if not paused and result.get('accepted'):
            with review_lock:
                review_frame=None
                review_corners.clear()
                mark_review()
        return result

    def set_armed(enabled):
        if not args.execute:
            raise ValueError('Execution is not enabled for this session')
        return send('arm', dict(enabled=enabled))

    def run_operator_command(command):
        def act():
            if stopped.is_set() and command != 'exit':
                raise ValueError('The interactive session is shutting down')
            if command in ('w','a','s','d','q','e'):
                with lock:
                    if not args.live or runtime is not None or selection or initial_preview.pose is None:
                        raise ValueError('Select an unconfirmed live initial pose first')
                    value=adjust_initial_pose(initial_preview.pose,command)
                    height=selected_initial_height(asset.snapshot,hint,value[3:5],initial_z=args.initial_z,
                        observed_height=None if initial_ground is None else initial_ground.current(time.monotonic()),
                        height_range=(ground_config['ground_min_below_sensor_m'],ground_config['ground_max_below_sensor_m']))
                    value[5]=height['lidar_z_m']
                    initial_preview.select(value,rospy.Time.now().to_sec(),height_seed=height)
                    show_initial_candidate(value)
                    return dict(accepted=True,state='initial_pose_preview',preview_revision=initial_preview.revision)
            if command=='confirm':return confirm_initial()
            if command=='review':return begin_review()
            if command in ('pause', 'resume'):return set_playback_paused(command=='pause')
            if command in ('shadow_start', 'shadow_stop'):
                return send('shadow_navigation', dict(enabled=command=='shadow_start'))
            if command=='arm':return set_armed(True)
            if command=='cancel':return cancel()
            if command=='exit':
                stopped.set()
                return dict(accepted=True,state='stopping')
            raise ValueError('Unknown operator command')
        return report(act)

    def panel_command(message):
        if not message.data.startswith('{'):
            run_operator_command(message.data)
            return
        try:
            request = json.loads(message.data)
            command, request_id = request['command'], request['request_id']
            if not isinstance(command, str) or not isinstance(request_id, str) or not request_id:
                raise ValueError('Expected a command and request identifier')
        except (KeyError, TypeError, ValueError) as error:
            rospy.logwarn('Invalid RViz panel command: %s', error)
            return
        result = run_operator_command(command)
        response = dict(request_id=request_id, accepted=result.get('accepted') is True)
        if not response['accepted'] and result.get('error'):
            response['error'] = result['error']
        if response['accepted'] and result.get('message'):
            response['message'] = result['message']
        panel_reply_pub.publish(String(data=json.dumps(response, ensure_ascii=False)))

    last_goal = object()
    def show_goal(request):
        nonlocal last_goal
        key=None if request is None else request['request_id']
        if key==last_goal:return
        last_goal=key
        clear=Marker();clear.header.frame_id=asset.manifest['frame_id'];clear.action=Marker.DELETEALL
        markers=[clear]
        from perception.visualization.interactive import goal_geometry
        geometry=goal_geometry(request,grid)
        if geometry is not None:
            marker=Marker();marker.header.frame_id=request['frame_id']
            marker.header.stamp=rospy.Time.from_sec(request.get('source_intent_stamp_s',0.))
            marker.ns='accepted_goal';marker.id=0;marker.action=Marker.ADD
            marker.pose.position=Point(*geometry['position']);marker.pose.orientation.w=1.
            marker.color.r,marker.color.g,marker.color.b,marker.color.a=1.,.3,1.,1.
            if geometry['yaw'] is None:
                marker.type=Marker.SPHERE;marker.scale.x=marker.scale.y=marker.scale.z=.18
            else:
                marker.type=Marker.ARROW;marker.scale.x,marker.scale.y,marker.scale.z=.8,.10,.14
                marker.pose.orientation.z=math.sin(geometry['yaw']/2);marker.pose.orientation.w=math.cos(geometry['yaw']/2)
            markers.append(marker)
        goal_pub.publish(MarkerArray(markers=markers))

    def pause_service(request):
        result = report(lambda: set_playback_paused(request.data))
        return SetBoolResponse(success=result.get('accepted', False), message=json.dumps(result, ensure_ascii=False))

    def arm_service(request):
        result = report(lambda: set_armed(request.data))
        return SetBoolResponse(success=result.get('accepted', False), message=json.dumps(result, ensure_ascii=False))

    def cancel_service(_):
        result = report(cancel)
        return TriggerResponse(success=result.get('accepted', result.get('reply', {}).get('accepted', False)), message=json.dumps(result))

    subscribers = [rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, initial, queue_size=1),
                   rospy.Subscriber('/social_nav_interactive/initial_pose_candidate', PoseWithCovarianceStamped, initial, queue_size=1),
                   rospy.Subscriber('/social_nav_interactive/panel_command', String, panel_command, queue_size=10),
                   rospy.Subscriber('/move_base_simple/goal', PoseStamped, goal, queue_size=1),
                   rospy.Subscriber('/clicked_point', PointStamped, point, queue_size=1),
                   rospy.Subscriber('/social_nav_interactive/confirm_initial_pose', Empty, confirm_initial_topic, queue_size=1)]
    if args.live:
        subscribers.append(rospy.Subscriber(sensor_settings['pointcloud_topic'], PointCloud2, scan_for_initial_preview, queue_size=1, buff_size=4*1024*1024))
    services = [rospy.Service('/social_nav_interactive/pause', SetBool, pause_service),
                rospy.Service('/social_nav_interactive/arm', SetBool, arm_service),
                rospy.Service('/social_nav_interactive/cancel', Trigger, cancel_service),
                rospy.Service('/social_nav_interactive/confirm_initial_pose', Trigger, confirm_initial_service)]

    def keyboard():
        commands = {'c':'confirm', 'p':'pause', 'r':'resume', 's':'cancel', 'a':'arm', 'q':'exit',
                    'shadow_start':'shadow_start', 'shadow_stop':'shadow_stop', 'review':'review'}
        for line in sys.stdin:
            command = commands.get(line.strip().lower())
            if command is not None:
                run_operator_command(command)
            if command == 'exit':
                return
    if sys.stdin.isatty():
        threading.Thread(target=keyboard, daemon=True, name='interactive-keyboard').start()

    def publish_path(publisher, poses, frame, stamp):
        result = RosPath(); result.header.frame_id = frame; result.header.stamp = rospy.Time.from_sec(stamp)
        for x, y, z, yaw in poses:
            pose = PoseStamped(); pose.header = result.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
            pose.pose.orientation.z = math.sin(yaw/2); pose.pose.orientation.w = math.cos(yaw/2)
            result.poses.append(pose)
        publisher.publish(result)

    last_preview = None
    try:
        if not args.no_rviz:
            view=yaml.safe_load((ROOT/'perception/visualization/configs/interactive.rviz').read_text())
            xy=anchor if seed is None else seed[3:5]
            z=annotation_height(grid,xy,anchor_z)
            manager=view['Visualization Manager']
            manager['Global Options']['Fixed Frame']=grid.metadata['frame_id']
            manager['Views']['Current']['Focal Point']=dict(X=float(xy[0]),Y=float(xy[1]),Z=z)
            static_resolution=float(grid.metadata['resolution'])
            local_resolution=float(ground_config['resolution'])
            cell_sizes={
                '/social_nav_interactive/static_basemap':static_resolution,
                '/social_nav_runtime_view/reference_search_map':static_resolution,
                '/social_nav_runtime_view/navigation_basemap':static_resolution,
                '/social_nav_runtime_view/local_basemap':local_resolution,
            }
            for layer in manager['Displays']:
                if layer.get('Topic') in cell_sizes:
                    layer['Size (m)']=cell_sizes[layer['Topic']]
            for tool in manager['Tools']:
                if tool['Class'] in ('social_nav_rviz/GroundInitialPose','social_nav_rviz/GroundGoal'):
                    tool['Cell size']=static_resolution
            view_path=output/'view.rviz';view_path.write_text(yaml.safe_dump(view,sort_keys=False))
            rviz = subprocess.Popen(['rviz', '-d', str(view_path)])
        state_pub.publish(String(data=json.dumps(dict(state='initializing' if selection else 'waiting_for_initial_pose', reference='rslidar', executable=False))))
        publish_panel_state()
        show_notice()
        show_goal(None)
        print(('Cross-floor provisional start loaded; wait for map localization. ' if args.handoff_seed else
               'Saved bag start loaded; wait for map localization. ' if seed is not None else
               'RViz: Ground Pose Estimate selects LiDAR start. ')+
              'Ground Nav Goal selects pose; Publish Point selects XY.', flush=True)
        terminal_actions = ['s cancels/disarms', 'q exits']
        if args.live:
            terminal_actions.insert(0, 'c confirms the live initial pose')
        else:
            terminal_actions[:0] = ['p pauses', 'r resumes', 'shadow_start / shadow_stop control shadow navigation']
        if args.execute:
            terminal_actions.append('a arms')
        print('Terminal (press Enter after each command): '+', '.join(terminal_actions)+'.', flush=True)
        while not stopped.is_set() and not rospy.is_shutdown():
            draw_initial_preview()
            status = None
            with lock:
                if runtime is None and selection:
                    command = runtime_command(args, config, selection[0])
                    (output/'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2))
                    log = (output/'runtime.log').open('x')
                    runtime = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if runtime is not None:
                if runtime.poll() is not None:
                    if runtime.returncode: raise RuntimeError('Runtime failed; see '+str(output/'runtime.log'))
                    break
                if socket_path.exists():
                    try:
                        status = send('session_status', {})
                        if status.get('accepted') is False:
                            raise OSError(status.get('error', 'Cannot read interactive session status'))
                        show_notice(status)
                        show_goal(status.get('goal_intent'))
                        state_pub.publish(String(data=json.dumps(status, ensure_ascii=False, allow_nan=False)))
                        preview = status.get('preview')
                        key = None if preview is None else (preview.get('scene_binding'), (preview.get('request') or {}).get('request_id'))
                        if key != last_preview:
                            last_preview = key
                            display = {} if preview is None else preview.get('display') or {}
                            stamp = 0. if preview is None else preview.get('as_of', 0.)
                            clear=Marker();clear.header.frame_id=asset.manifest['frame_id'];clear.action=Marker.DELETEALL
                            curves=[clear]
                            for number,(name,publisher,color,width) in enumerate((
                                    ('reference',reference_pub,(.96,.96,.96,1.),.075),
                                    ('trajectory',trajectory_pub,(.27,.86,1.,1.),.10))):
                                curve=display.get('curves',{}).get(name,{})
                                frame=curve.get('frame_id',asset.manifest['frame_id'])
                                publish_path(publisher,curve.get('prefix',[]),frame,stamp)
                                if not curve.get('segments'):continue
                                line=Marker();line.header.frame_id=frame;line.header.stamp=rospy.Time.from_sec(stamp)
                                line.ns=name;line.id=number;line.type=Marker.LINE_LIST;line.action=Marker.ADD
                                line.pose.orientation.w=1.;line.scale.x=width
                                line.color.r,line.color.g,line.color.b,line.color.a=color
                                line.points=[Point(*xyz) for xyz in curve['segments']];curves.append(line)
                            curves_pub.publish(MarkerArray(markers=curves))
                            marker = Marker(); marker.header.frame_id = display.get('label_frame',asset.manifest['frame_id']); marker.header.stamp = rospy.Time.from_sec(stamp)
                            marker.ns = 'frozen'; marker.id = 0; marker.type = Marker.TEXT_VIEW_FACING
                            marker.action = Marker.DELETE if preview is None else Marker.ADD
                            marker.pose.orientation.w = 1.; marker.scale.z = .3
                            marker.color.r, marker.color.g, marker.color.b, marker.color.a = .3, .9, 1., 1.
                            marker.pose.position=Point(*display.get('label_position',[*anchor,anchor_z+2.2]))
                            marker.text = '' if preview is None else 'FROZEN BAG | REF %s | DWA %s\n%s' % (
                                preview.get('reference_available'), preview.get('generated'), preview.get('reason') or 'preview only')
                            for name,curve in display.get('curves',{}).items():
                                if curve['height_count']<curve['point_count']:
                                    marker.text+='\n%s height %d/%d (gaps omitted)'%(name,curve['height_count'],curve['point_count'])
                                if curve.get('approximate_count'):
                                    marker.text+='\n%s approximate heights: %d'%(name,curve['approximate_count'])
                            label_pub.publish(marker)
                    except (OSError, EOFError) as error:
                        status = None
                        rospy.logwarn_throttle(2., 'Waiting for runtime: %s', error)
                else:show_notice()
            publish_panel_state(status)
            stopped.wait(.2)
    finally:
        if runtime is not None and runtime.poll() is None:
            try:
                send('override', dict(manual_override=False, emergency_stop=True))
            except Exception:
                pass
            runtime.send_signal(signal.SIGINT)
            try: runtime.wait(timeout=15)
            except subprocess.TimeoutExpired:
                runtime.terminate()
                try: runtime.wait(timeout=3)
                except subprocess.TimeoutExpired: runtime.kill(); runtime.wait(timeout=3)
        if rviz is not None and rviz.poll() is None:
            rviz.terminate()
            try: rviz.wait(timeout=3)
            except subprocess.TimeoutExpired: rviz.kill(); rviz.wait(timeout=3)
        if log is not None: log.close()
        for service in services: service.shutdown()
        for subscriber in subscribers: subscriber.unregister()
        asset.close()
    if runtime is not None and runtime.returncode:
        raise RuntimeError('Runtime failed during shutdown; see '+str(output/'runtime.log'))


if __name__ == '__main__':
    main()
