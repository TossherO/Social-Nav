"""Direct perception/navigation; shadow by default, explicitly armed live tests optional."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import yaml

from interfaces.shared import SharedArrayReader
from .transport import Connection,frame_ref
from .metrics import Metrics

ROOT=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',help='Existing geometry/calibration, never computed poses; execution backends use current defaults')
    p.add_argument('--bag',help='Override the recorded bag while retaining explicit algorithm parameters')
    p.add_argument('--live',action='store_true',help='Subscribe only to sensor topics at the ROS boundary')
    p.add_argument('--mode',choices=['mapping','localization','odometry'])
    p.add_argument('--lio-backend',choices=['cpu','cuda'],help='Override module CUDA default for FAST-LIO matching')
    p.add_argument('--roll-backend',choices=['cpu','cuda'],help='Override module CUDA default for ROLL; unused by pure odometry')
    p.add_argument('--roll-gravity-deg',type=float,help='Explicit live fixed-map gravity correction limit in degrees')
    p.add_argument('--deskew-backend',choices=['cpu','cuda'],help='Override module CUDA default for deskew')
    p.add_argument('--linearization-backend',choices=['cpu','cuda'],help='Experimental CUDA observation reduction; CPU by default')
    p.add_argument('--map',dest='map_directory');p.add_argument('--pose',nargs=6,type=float);p.add_argument('--pose-file')
    p.add_argument('--start',type=float);p.add_argument('--lidar-imu-calibration');p.add_argument('--camera-calibration')
    p.add_argument('--sensor-clock-group',help='Declared continuous sensor/driver clock session; never inferred from bag dates')
    p.add_argument('--sensor-session',help='Bag-bound clock-session JSON; defaults to BAG.clock.json')
    p.add_argument('--output',required=True);p.add_argument('--gpu',help='Physical visible-device selection; otherwise preserve CUDA_VISIBLE_DEVICES');p.add_argument('--duration',type=float)
    p.add_argument('--rate',type=float,default=1.);p.add_argument('--static-map');p.add_argument('--goal',nargs='+',type=float)
    p.add_argument('--goal-frame',default='odom');p.add_argument('--coverage');p.add_argument('--record-views',action='store_true')
    p.add_argument('--record-planning',action='store_true',help='Bounded asynchronous planning evidence: 2 Hz maps plus footprint/ground/budget failure snapshots; preserves normal map scheduling')
    p.add_argument('--trace-timing',action='store_true',help='Bounded scalar timing events, flushed at shutdown; no map/image recording or scheduling change')
    p.add_argument('--record-decision-inputs',action='store_true',help='Optional VLM worker: prepare YOLO associations and freeze causal three-frame task windows; no VLM request')
    p.add_argument('--agent-observations',action='store_true',help='Bounded optional Agent RGB/scene outlet; no YOLO, API or chassis output')
    p.add_argument('--agent-follow-candidates',action='store_true',help='Prepare optional YOLO person candidates for Agent target selection')
    p.add_argument('--agent-execution',action='store_true',help='Allow owned Agent goal intents through the existing safety path; requires manual arming')
    p.add_argument('--no-image',action='store_true',help='Compatibility flag: the core never loads YOLO; disallows optional VLM image preparation')
    p.add_argument('--runtime-config',default=str(ROOT/'configs/runtime.yaml'))
    p.add_argument('--environment-update-hz',type=float,help='Override the environment update rate without changing observation age limits')
    p.add_argument('--environment-window-m',type=float,help='Override the square local environment window in metres')
    p.add_argument('--motion-config',help='Explicit motion YAML shared by reference planning, DWA and independent safety')
    p.add_argument('--motion-profile',help='Named profile in the selected motion YAML')
    p.add_argument('--display',action='store_true',help='Publish optional RViz display topics; no core subscribes to them')
    p.add_argument('--interactive-session',action='store_true',help='Enable session status and bag pause/preview commands')
    p.add_argument('--pause-at',type=float,help='Pause an interactive bag after this many source seconds; hold at bag end')
    p.add_argument('--vehicle-profile',help='Explicit measured vehicle geometry, limits and protocol')
    p.add_argument('--execute',action='store_true',help='Enable supervised live hardware output; separate operator arming required')
    p.add_argument('--chassis-port',help='Exclusive serial device, only with --execute')
    p.add_argument('--commands',help='Explicit timed goal/cancel/override list for causal shadow replay')
    p.add_argument('--route',help='Source-bound recorded demonstration route JSON; bag shadow replay only')
    p.add_argument('--route-floor',help='Explicit floor selection within --route; never inferred from the elevator')
    p.add_argument('--record-scans','--record-environment-scans',action='store_true',default=None)
    p.add_argument('--no-record-scans',action='store_false',dest='record_scans')
    return p


def main():
    args=parser().parse_args()
    if args.execute and (not args.live or not args.interactive_session or not args.vehicle_profile or not args.chassis_port):
        raise ValueError('--execute requires --live --interactive-session --vehicle-profile and --chassis-port')
    if args.agent_execution and not (args.execute and args.agent_observations):
        raise ValueError('--agent-execution requires --execute and --agent-observations')
    if args.agent_follow_candidates and (not args.agent_observations or args.no_image):
        raise ValueError('--agent-follow-candidates requires --agent-observations and RGB input')
    if args.chassis_port and not args.execute:raise ValueError('--chassis-port requires --execute')
    if args.record_decision_inputs and args.no_image:
        raise ValueError('--record-decision-inputs explicitly enables VLM image preparation; remove --no-image')
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    from .config import runtime_settings
    from perception.mapping_localization.configuration import algorithm_config,DEFAULT_CALIBRATION
    from perception.sensors.clock_session import check_time_calibration
    source=Path(args.source_run).resolve() if args.source_run else None
    run={} if source is None else json.loads((source/'run.json').read_text())
    args.start=(0. if args.live else float(run.get('start',0.) or 0.)) if args.start is None else args.start
    if args.coverage is None and args.map_directory is None:
        args.coverage=run.get('association_coverage')
    if args.duration is None and source is not None and not args.live:args.duration=run.get('duration')
    mode=args.mode or run.get('mode','odometry')
    if args.live and (args.start or args.rate!=1. or args.pose_file):raise ValueError('Live input uses ROS time; bag offset/rate/pose file are not applicable')
    if args.live and mode=='localization' and args.pose is None:raise ValueError('Live map localization requires an explicit current --pose')
    if args.live and args.bag:raise ValueError('Select either bag or live input')
    if args.pause_at is not None and (not args.interactive_session or args.live or not np.isfinite(args.pause_at) or args.pause_at<0):
        raise ValueError('--pause-at requires an interactive bag and a finite nonnegative offset')
    if args.interactive_session and (args.route or args.commands):
        raise ValueError('Interactive goals cannot share a recorded route or scheduled commands')
    if not args.live and not args.bag and not run.get('bag'):raise ValueError('Select --bag, --live or a source run with an original bag')
    if args.route and (args.live or not args.route_floor or args.goal is not None or args.commands):
        raise ValueError('--route needs a bag and --route-floor, without --goal or --commands')
    if args.route and mode!='localization':raise ValueError('Recorded mixed-route replay needs --mode localization')
    if args.route_floor and not args.route:raise ValueError('--route-floor requires --route')
    bag=None if args.live else Path(args.bag or run['bag']).expanduser().resolve()
    config,run=algorithm_config(mode=mode,source_run=source,map_directory=args.map_directory,pose=args.pose,pose_file=args.pose_file,
        bag=bag,start=args.start,calibration=args.lidar_imu_calibration,coverage=args.coverage,lio_backend=args.lio_backend,
        roll_backend=args.roll_backend,deskew_backend=args.deskew_backend,linearization_backend=args.linearization_backend)
    if args.roll_gravity_deg is not None:
        if not args.live or mode!='localization' or not np.isfinite(args.roll_gravity_deg) or not 0<args.roll_gravity_deg<=15:
            raise ValueError('--roll-gravity-deg requires live localization and a finite value in (0,15] degrees')
        config['roll']['coverageGate']['maxGravity']=float(np.deg2rad(args.roll_gravity_deg))
    if args.rate<=0:raise ValueError('Integrated timing/independent safety requires positive real-time replay rate')
    if args.goal is not None and len(args.goal) not in (2,3):raise ValueError('Goal requires x y [yaw]')
    if args.environment_update_hz is not None and (not np.isfinite(args.environment_update_hz) or not 0<args.environment_update_hz<=10):
        raise ValueError('Environment update rate must be finite and in (0,10] Hz')
    if args.environment_window_m is not None and (not np.isfinite(args.environment_window_m) or not 0<args.environment_window_m<=60):
        raise ValueError('Environment window must be finite and in (0,60] metres')
    settings=runtime_settings(args.runtime_config)
    from perception.environment.layers import load_config as environment_config
    env_config=environment_config();env_config['compute'].update(backend='cuda',device=0)
    if args.environment_update_hz is not None:env_config['update_hz']=args.environment_update_hz
    if args.environment_window_m is not None:
        support=env_config.get('navigation_support',{})
        minimum=2*max(env_config['ground_fit_radius_m'],support.get('radius_m',0.) if support.get('enabled') else 0.)
        if args.environment_window_m<minimum or args.environment_window_m/env_config['resolution']>600:
            raise ValueError('Environment window must cover configured ground fitting and navigation support radii')
        env_config['window_m']=args.environment_window_m
    with_rgb=bool((args.display and settings['display']['rgb_hz']>0) or args.record_views or
        args.record_decision_inputs or args.agent_observations)
    if (not np.isfinite([args.start,args.rate]).all() or args.start<0 or
            args.duration is not None and (not np.isfinite(args.duration) or args.duration<=0)):
        raise ValueError('Start/rate/duration must be finite with nonnegative start and positive rate/duration')
    for key in ('control_period_s','scene_export_hz','sensor_input_timeout_s','startup_timeout_s','finalization_timeout_s'):
        if not np.isfinite(settings[key]) or settings[key]<=0:raise ValueError('Positive finite runtime setting required: '+key)
    if not .01<=settings['control_period_s']<=.2:raise ValueError('Control period must stay in [0.01,0.2] seconds')
    calibration_source=Path(args.lidar_imu_calibration) if args.lidar_imu_calibration else source/'run.json' if source else DEFAULT_CALIBRATION
    timing=check_time_calibration(config,bag=bag,group=args.sensor_clock_group,session_file=args.sensor_session,
        previous=run.get('imu_time_calibration'),source_file=calibration_source)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Selected CUDA device unavailable; check device access before changing dependencies')
    torch.cuda.set_device(0)
    from perception.detection_tracking.inference.config import load_config as detection_config
    from perception.detection_tracking.inference.detector import SimpleDet3dDetector
    from perception.detection_tracking.inference.pipeline import PerceptionPipeline,Frame
    from perception.environment.layers import LocalEnvironmentMap
    from perception.environment.occupancy import OccupancyTree
    from perception.environment.scene import assemble_scene,people_observation
    from perception.environment.storage import StaticEnvironmentMap
    from perception.mapping_localization.context import pose_matrix
    from perception.mapping_localization.results import file_fingerprint
    from navigation.planning.inputs import load_config as planning_config
    from navigation.planning.motion import load_motion_config
    from navigation.planning.footprint import VehicleGeometry
    from navigation.planning.reference import ReferenceRuntime
    from navigation.control.device import DeviceDwaBatch
    from navigation.control.shadow import DwaShadowController,load_config as shadow_config
    from navigation.planning.runtime import Worker
    from .safety import SafetyProcess,SafetyTelemetry
    from .export import SceneExporter
    from .command import CommandInbox
    from .control import ControlLoop
    clock_domain='runtime_live' if args.live else 'runtime_bag'
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    metrics=Metrics(trace_path=output/'timing.jsonl' if args.trace_timing else None);startup=time.perf_counter();device=torch.device('cuda:0')
    det_cfg=detection_config();detector=SimpleDet3dDetector(det_cfg['detector'],metrics=metrics);pipeline=PerceptionPipeline(detector,det_cfg['tracking'])
    camera_record=None
    if args.camera_calibration:
        from decision.vlm.image_inputs import camera_calibration
        camera_record=camera_calibration(args.camera_calibration)
    support_enabled=env_config.get('navigation_support',{}).get('enabled',False)
    factory=lambda resolution:OccupancyTree(resolution,library=ROOT/settings['environment_library'],backend='cuda',device=0)
    environment=LocalEnvironmentMap(env_config,tree_factory=factory)
    from perception.environment.scans import AsyncScanJournal
    record_scans=mode=='mapping' if args.record_scans is None else args.record_scans
    journal=AsyncScanJournal(output/'environment/scans.sqlite',env_config['scan_recording'],dict(config=env_config,
        sensor_frame=det_cfg['ros']['localization_child_frame'],evidence='frontend_accepted_deskewed_points_not_feature_keyframes')) if record_scans else None
    display=None
    exporter=SceneExporter(output/'scenes',env_config['scene'],metrics=metrics,prior_directory=args.static_map,clock_domain=clock_domain) if args.record_views else None
    decision_recorder=None
    if args.record_decision_inputs:
        from .decision import DecisionRecorder
        decision_recorder=DecisionRecorder(output/'decision_inputs',metrics=metrics,
            image_policy=det_cfg['image_association'],camera_calibration=args.camera_calibration)
    motion_path=Path(args.motion_config or ROOT/'navigation/planning/configs/motion.yaml').expanduser().resolve()
    motion_document=yaml.safe_load(motion_path.read_text())
    motion_source=dict(path=str(motion_path),fingerprint=file_fingerprint(motion_path),
        profile=args.motion_profile or (motion_document.get('default_profile') if isinstance(motion_document,dict) else None))
    planning=planning_config();motion=load_motion_config(motion_path,profile=args.motion_profile);geometry=VehicleGeometry.historical(ROOT/planning['geometry_reference'])
    if args.vehicle_profile:
        from navigation.control.vehicle import load_vehicle_profile
        geometry,motion,_=load_vehicle_profile(args.vehicle_profile,motion,require_execution=args.execute)
    planning['motion_constraints']={k:motion[k] for k in ('min_turning_radius_m','allow_reverse','allow_in_place_rotation')}
    planning['pose_search']['native_library']=str(ROOT/'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
    motion.update(collision_backend='cuda',collision_library=str(ROOT/settings['collision_library']))
    asset=StaticEnvironmentMap(args.static_map) if args.static_map else None
    if asset is not None and asset.snapshot.metadata.get('projection_strategy') in ('docker_assume_free', 'ground_relative_assume_free'):
        print('[static map] Non-hit cells are assumed free; measured local safety checks remain required.',
              file=sys.stderr,flush=True)
    from perception.environment.navigation import CAPSULE_FIELDS, safety_capsule
    from perception.environment.support import NavigationSupport, finalize
    navigation_support=NavigationSupport(env_config,library=ROOT/settings['environment_library'])
    route_replay=None
    if args.route:
        from navigation.planning.route_replay import RouteReplay
        route_replay=RouteReplay(args.route,bag,args.route_floor,geometry,planning,settings['route_replay'],asset)
    reference=ReferenceRuntime(planning,motion,geometry,worker_factory=lambda *parts:Worker(*parts,metrics=metrics))
    batch=DeviceDwaBatch(device=device,library=ROOT/settings['dwa_library'],metrics=metrics)
    shadow=shadow_config();shadow['period_s']=settings['control_period_s']
    controller=DwaShadowController(planning,motion,geometry,asset=asset,device_batch=batch,settings=shadow,execution=args.execute)
    planning_recorder=None
    if args.record_planning:
        from .planning_record import PlanningRecorder
        planning_recorder=PlanningRecorder(output/'planning',metrics=metrics)
        reference.diagnostics=controller.controller.diagnostics=True
        (output/'planning/config.json').write_text(json.dumps(dict(planning=reference.planning,motion=motion,
            reference=reference.settings,dwa=controller.controller.settings,safety=controller.controller.safety_settings,
            geometry=dict(footprint=geometry.footprint.tolist(),base_from_lidar=geometry.base_from_lidar.tolist(),source=geometry.source),
            static_map=args.static_map),allow_nan=False,indent=2)+'\n')
    if args.display:
        from perception.visualization.runtime import RuntimeDisplay
        display=RuntimeDisplay(settings['display'],geometry,planning,asset=asset,
            reference_max_age_s=reference.settings['max_age_s'],
            trajectory_max_age_s=controller.controller.safety_settings['max_input_age_s'],metrics=metrics)
    state_parent,state_input=socket.socketpair();safety=SafetyProcess(state_parent,planning,motion,geometry,clock_domain=clock_domain,asset=asset,
        slots=settings['shared_safety_slots'],slot_bytes=settings['shared_safety_slot_bytes'],settings=shadow,
        chassis_port=args.chassis_port,chassis_output=output/'chassis')
    main_pose,pose_input=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM);main_pose.setblocking(False)
    from .association import AssociationInbox
    association_socket,association_input=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
    associations=AssociationInbox(association_socket,metrics)
    interactive=None
    if args.interactive_session:
        from .session import InteractiveSession
        interactive=InteractiveSession(live=args.live,safety=safety,output=output,execute=args.execute)
    commands=CommandInbox(output/'commands.sock',safety,session=interactive,agent_execution=args.agent_execution,
                          follow_enabled=args.agent_follow_candidates)
    commands.agent_state.configure_vehicle(geometry,motion)
    agent_observer=None
    if args.agent_observations:
        from decision.agent.adapters.runtime import ObservationWorker
        commands.agent_state.enable_goal_observations()
        agent_observer=ObservationWorker(commands.agent_state.publish_observation,metrics=metrics,geometry=geometry,
                                         follow_candidates=args.agent_follow_candidates)
    schedule=[] if args.commands is None else json.loads(Path(args.commands).read_text())
    if not isinstance(schedule,list) or any(set(v)!={'offset_s','kind','value'} or not np.isfinite(v['offset_s']) or v['offset_s']<0 for v in schedule):
        raise ValueError('Command schedule requires finite offset_s, kind and value')
    schedule=deque(sorted(schedule,key=lambda v:v['offset_s']));first_observation=None
    state_parent.close();parent,child=socket.socketpair();wire=Connection(parent);worker=None;reader=None
    names=('localization','people','control','safety','reference')+(('route',) if route_replay is not None else ())
    streams={name:(output/(name+'.jsonl')).open('x') for name in names}
    def record(name,value):streams[name].write(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    scene=None;next_env=-np.inf;next_export=-np.inf;control_loop=None;safety_telemetry=None;initial_goal=None;map_worker=None;map_stream=None;scan_inbox=None;input_epoch=None
    if args.goal is not None:
        initial_goal=dict(request_id='runtime-goal',task_id='runtime-shadow',goal_revision=0,mode='mapped' if args.goal_frame=='map' else 'local',
            goal=args.goal,frame_id=args.goal_frame,target_reference='base_link',algorithm='hybrid_astar',goal_behavior='stop',motion_intent='forward_preferred')
        if asset is not None:initial_goal['map_snapshot_id']=asset.manifest['snapshot_id']
    counts=dict(scans=0,maps=0,controls=0,safety=0);source_metrics=None;last_device=None;failure=None;run_start=None;cpu_start=None
    def reset_map():
        environment.reset();navigation_support.reset()
    def compute_map(job):
        nonlocal map_stream
        metrics.trace('map_start',measurement_stamp_s=job['stamp'])
        torch.cuda.set_device(0)
        if map_stream is None:map_stream=torch.cuda.Stream()
        with torch.cuda.stream(map_stream):
            with metrics.span('environment_device'):
                grid=environment.update(job['points'],job['pose'],job['stamp'],job['identity'],**job['options'],device=device,metrics=metrics)
                if support_enabled:
                    grid=navigation_support.apply(grid,asset,observation=environment.support_observation,metrics=metrics)
            metrics.trace('map_device_done',measurement_stamp_s=job['stamp'])
            with metrics.span('safety_capsule_materialize'):
                fields=CAPSULE_FIELDS if support_enabled else ('occupancy','person_background_free')
                capsule=grid.materialize(consumer='safety_capsule',fields=[k for k in fields if k in grid._layers],metrics=metrics)
                if support_enabled:
                    from perception.environment.device import DeviceGridSnapshot
                    capsule=safety_capsule(finalize(capsule))
                    grid=DeviceGridSnapshot.seal(capsule.metadata,grid.consume())
                current=assemble_scene(capsule,clock_domain=clock_domain,people=job['options']['people'],settings=env_config['scene'])
            metrics.trace('map_ready',measurement_stamp_s=job['stamp'],scene_binding=current.record['input_fingerprint'])
            return current,grid
    def publish_map(result):
        if result is None:control_loop.offer(None,None);return
        current,grid=result
        metrics.trace('map_publish_start',measurement_stamp_s=current.record['as_of'],scene_binding=current.record['input_fingerprint'])
        safety.scene(current)
        metrics.trace('map_safety_sent',measurement_stamp_s=current.record['as_of'],scene_binding=current.record['input_fingerprint'])
        control_loop.offer(current,grid);counts['maps']+=1
    previous_interrupt=None;stop_requested=False
    try:
        with tempfile.TemporaryDirectory(prefix='social-nav-runtime-') as tmp:
            deadline=time.monotonic()+settings['startup_timeout_s']
            while not safety.ready or not reference.worker.ready:
                safety.poll();reference.worker.poll()
                if time.monotonic()>deadline:raise RuntimeError('CPU worker startup exceeded its budget')
                time.sleep(.01)
            safety_telemetry=SafetyTelemetry(safety,record)
            control_loop=ControlLoop(reference,controller,batch,safety,main_pose,commands,settings,env_config['scene'],metrics,record,asset=asset,initial_goal=initial_goal,clock_domain=clock_domain,display=display,route_replay=route_replay,planning_recorder=planning_recorder,session=interactive)
            # Archives retain every selected scan and its matching optional
            # inputs. Normal navigation bounds map work independently of the
            # detector; the native tree has exactly one owning worker.
            if not (journal or exporter or decision_recorder or agent_observer):
                from .environment import LatestMapWorker
                map_worker=LatestMapWorker(compute_map,publish_map,reset_map)
            cfg=Path(tmp)/'algorithm.yaml';cfg.write_text(yaml.safe_dump(config))
            command=[settings['input_python'],'-B','-m','runtime.input_worker','--socket-fd',str(child.fileno()),'--pose-fd',str(state_input.fileno()),
                '--main-pose-fd',str(pose_input.fileno()),
                '--association-fd',str(association_input.fileno()),
                '--shared-slots',str(settings['shared_scan_slots']),'--slot-bytes',str(settings['shared_scan_slot_bytes']),
                '--origin-gap',str(env_config['max_origin_interpolation_gap_s']),
                '--config',str(cfg),'--mode',mode,'--rate',str(args.rate),'--start',str(args.start)]
            command+=['--live'] if args.live else ['--bag',str(bag)]
            if args.trace_timing:command+=['--timing-output',str(output/'timing_input.jsonl')]
            if with_rgb:command+=['--with-rgb']
            if args.duration:command+=['--duration',str(args.duration)]
            if args.coverage:command+=['--coverage',str(Path(args.coverage).resolve())]
            if mode=='mapping':command+=['--map-output',str(output/'map')]
            pass_fds=(child.fileno(),state_input.fileno(),pose_input.fileno(),association_input.fileno())
            if interactive is not None and interactive.child is not None:
                command+=['--playback-fd',str(interactive.child.fileno())]
                pass_fds+= (interactive.child.fileno(),)
                if args.pause_at is not None:command+=['--pause-at',str(args.pause_at)]
            child_env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT)+':/opt/ros/noetic/lib/python3/dist-packages')
            child_env['LD_LIBRARY_PATH']='/opt/ros/noetic/lib:'+str(ROOT/'outputs/perception/dependencies/gtsam/lib')+':'+child_env.get('LD_LIBRARY_PATH','')
            worker_log=(output/'input.log').open('x')
            worker=subprocess.Popen(command,cwd=ROOT,env=child_env,pass_fds=pass_fds,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
            if interactive is not None and interactive.child is not None:interactive.child.close()
            def request_stop(*_):
                nonlocal stop_requested
                if stop_requested:raise KeyboardInterrupt
                stop_requested=True;safety.command('override',dict(manual_override=False,emergency_stop=True));worker.send_signal(signal.SIGINT)
            previous_interrupt=signal.signal(signal.SIGINT,request_stop)
            child.close();state_input.close();pose_input.close();association_input.close();parent.settimeout(settings['startup_timeout_s'])
            fd=wire.recv_fd();header=wire.recv();reader=SharedArrayReader(fd,header['owner'],header['slots'],header['slot_bytes'])
            ready=wire.recv()
            if ready.get('kind')!='ready':raise RuntimeError('Input initialization failed: '+str(ready))
            metrics.add('startup',time.perf_counter()-startup)
            run_start=time.monotonic();cpu_start=time.process_time()
            parent.settimeout(settings['sensor_input_timeout_s'])
            if map_worker is not None:
                from .scan_inbox import LatestScanInbox
                def observe_scan(packet):
                    nonlocal first_observation
                    metrics.trace('scan_received',measurement_stamp_s=packet['stamp_s'])
                    if first_observation is None:first_observation=packet['stamp_s']
                    record('localization',dict(packet['state'],clock_domain=clock_domain))
                scan_inbox=LatestScanInbox(wire,observe_scan,map_worker.invalidate,settings['finalization_timeout_s'])
            while True:
                with metrics.span('input_wait'):packet=wire.recv() if scan_inbox is None else scan_inbox.recv()
                if packet['kind']=='end':source_metrics=packet['metrics'];break
                if packet['kind']=='draining':parent.settimeout(settings['finalization_timeout_s']);continue
                if packet['kind']=='error':raise RuntimeError(packet['error'])
                if packet['kind']=='playback':
                    if interactive is None:raise RuntimeError('Unexpected playback event')
                    if packet['paused'] and not interactive.paused and map_worker is not None:map_worker.flush()
                    interactive.update_playback(packet)
                    if control_loop.error:raise RuntimeError(control_loop.error)
                    continue
                ref=frame_ref(packet['buffer']);arrays=reader.arrays(ref);began=time.monotonic()
                try:
                    row=dict(packet['state'],clock_domain=clock_domain);stamp=packet['stamp_s'];counts['scans']+=1
                    if first_observation is None:first_observation=stamp
                    while schedule and schedule[0]['offset_s']<=stamp-first_observation:
                        item=schedule.popleft();commands.offer(item['kind'],item['value'])
                    if control_loop.error:raise RuntimeError(control_loop.error)
                    if safety_telemetry.error:raise RuntimeError(safety_telemetry.error)
                    if scan_inbox is None:record('localization',row)
                    elif packet['_input_epoch']!=input_epoch:
                        pipeline.reset();input_epoch=packet['_input_epoch']
                    pose=pose_matrix(row['position'],row['quaternion_xyzw']) if row['health']==0 else None
                    frame=Frame(packet['observation_id'],str(bag) if bag else 'live',stamp,arrays['raw'],sensor_frame=packet['sensor_frame'],clock_domain=clock_domain,
                        world_from_lidar=pose,world_frame=row['frame_id'],pose_source='localization',pose_health='valid' if pose is not None else 'invalid',
                        pose_stamp_s=stamp,pose_identity=(row['session_id'],row['generation']),rgb_stamp_s=packet['rgb_stamp'])
                    metrics.trace('detector_start',measurement_stamp_s=stamp)
                    with metrics.span('detector_tracking'):people=pipeline.step(frame)
                    metrics.trace('detector_done',measurement_stamp_s=stamp)
                    people['localization_binding']=dict(session_id=row['session_id'],generation=row['generation'],frame_id=row['frame_id'],source=row['source'])
                    record('people',people)
                    if map_worker is not None:
                        latest=map_worker.latest()
                        scene,last_device=(None,None) if latest is None else latest
                    if pose is None:
                        if map_worker is None:reset_map();control_loop.offer(None,None)
                        else:map_worker.invalidate()
                        scene=None;last_device=None
                    elif (scan_inbox is None or scan_inbox.current(packet)) and 'deskewed' in arrays and stamp>=next_env-1e-6:
                        job=dict(points=arrays['deskewed'][:,:3],pose=pose,stamp=stamp,identity=(row['session_id'],row['generation']),
                            options=dict(frame=row['frame_id'],association=associations.select(packet['association']),scan_duration=packet['scan_duration_s'],
                                ray_origins=arrays.get('ray_origins'),people=people,clock_domain=clock_domain,
                                localization_source=row['source'],sensor_frame=packet['sensor_frame']))
                        if map_worker is None:
                            metrics.trace('map_submit',measurement_stamp_s=stamp)
                            scene,last_device=compute_map(job);publish_map((scene,last_device))
                        else:
                            import copy
                            # Own bounded input storage before releasing the
                            # shared scan slot; never retain a writable lease.
                            job=copy.deepcopy(job)
                            job['points'].flags.writeable=False
                            origins=job['options']['ray_origins']
                            if origins is not None:origins.flags.writeable=False
                            metrics.transfer('host_copy',job['points'].nbytes+(0 if origins is None else origins.nbytes),'environment_job')
                            metrics.trace('map_submit',measurement_stamp_s=stamp)
                            scan_inbox.submit_if_current(packet,lambda:map_worker.submit(job,identity=(*job['identity'],row['frame_id'])))
                        if journal is not None:
                            if not journal.submit(arrays['deskewed'][:,:3],pose,row['stamp_ns'],last_device.metadata,
                                arrays.get('ray_origins'),arrays['deskewed'][:,4]):raise RuntimeError('Scan journal failed: '+str(journal.summary()))
                        next_env=stamp if not np.isfinite(next_env) else next_env
                        while next_env<=stamp+1e-6:next_env+=1/env_config['update_hz']
                        if args.record_views and stamp>=next_export-1e-6:
                            rgb=None if 'rgb' not in arrays else dict(pixels=arrays['rgb'].copy(),stamp_s=packet['rgb_stamp'],
                                frame_id=packet['rgb_frame'],
                                reference=dict(source='live' if args.live else 'bag',bag=None if bag is None else str(bag),stamp_s=packet['rgb_stamp']))
                            exporter.submit(last_device,people,rgb=rgb,name=packet['observation_id'])
                            next_export=stamp+1/settings['scene_export_hz']
                    if agent_observer is not None and scene is not None and abs(scene.record['as_of']-stamp)<1e-6:
                        agent_observer.submit(scene,arrays.get('rgb'),packet['rgb_stamp'],packet['rgb_frame'],packet['camera_info'],packet['rgb_stamp_ns'])
                    if decision_recorder is not None:
                        current_scene=scene if scene is not None and abs(scene.record['as_of']-stamp)<1e-6 else None
                        decision_recorder.submit(people,arrays,scene=current_scene,
                            image_metadata=dict(rgb_stamp_ns=packet['rgb_stamp_ns'],
                                rgb_frame=packet['rgb_frame'],camera_info=packet['camera_info']))
                    if display is not None:
                        display.submit(arrays['raw'],row,people,rgb=arrays.get('rgb'),rgb_frame=packet['rgb_frame'],grid=None if scene is None else last_device)
                    now=packet['source_clock_s']+(time.monotonic()-packet['source_mono'])*args.rate
                    metrics.add('sensor_ready_to_compute_complete',time.monotonic()-packet['source_mono'])
                    metrics.add('source_observation_age',max(0.,now-stamp))
                    metrics.add('scan_consumer_compute',time.monotonic()-began)
                finally:
                    arrays.clear();frame=None
                    if scan_inbox is None:wire.send(dict(slot=ref.slot,epoch=ref.epoch))
                    else:scan_inbox.release(packet)
            worker.wait(timeout=20)
            if worker.returncode:raise RuntimeError('Native input worker failed: '+str(worker.returncode))
            if map_worker is not None:
                map_worker.flush();latest=map_worker.latest()
                scene,last_device=(None,None) if latest is None else latest
            if last_device is not None:
                final=last_device.materialize(consumer='final_review',metrics=metrics)
                final_scene=assemble_scene(final,clock_domain=clock_domain,people=None if scene is None else people_observation(scene),settings=env_config['scene'])
                final_scene.save(output/'final_scene')
    except (Exception,KeyboardInterrupt) as error:
        failure=type(error).__name__+': '+str(error);raise
    finally:
        if previous_interrupt is not None:signal.signal(signal.SIGINT,previous_interrupt)
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=2)
        process_cpu_s=None if cpu_start is None else time.process_time()-cpu_start
        run_wall_s=None if run_start is None else time.monotonic()-run_start
        cleanup_errors=[];export_report=display_report=decision_report=planning_report=association_report=agent_report=None
        for name,close in [('scan_inbox',None if scan_inbox is None else scan_inbox.close),('map_worker',None if map_worker is None else map_worker.close),('commands',commands.close),('control',None if control_loop is None else control_loop.close),
                           ('agent_observations',None if agent_observer is None else agent_observer.close),
                           ('safety_telemetry',None if safety_telemetry is None else safety_telemetry.close),('safety',safety.close),('reference',reference.close),('environment',environment.close),('wire',wire.close),('pose_channel',main_pose.close),
                           ('reader',None if reader is None else reader.close),('journal',None if journal is None else journal.close),
                           ('exporter',None if exporter is None else exporter.close),
                           ('planning_recorder',None if planning_recorder is None else planning_recorder.close),
                           ('associations',associations.close),
                           ('decision_inputs',None if decision_recorder is None else decision_recorder.close),
                           ('interactive_session',None if interactive is None else interactive.close),
                           ('display',None if display is None else display.close)]:
            if close is None:continue
            try:
                result=close()
                if name=='exporter':export_report=result
                if name=='planning_recorder':planning_report=result
                if name=='associations':association_report=result
                if name=='agent_observations':agent_report=result
                if name=='decision_inputs':decision_report=result
                if name=='display':
                    display_report=result
                    if not result['complete']:cleanup_errors.append('display: '+str(result['error'] or 'worker failed to drain'))
            except Exception as error:cleanup_errors.append(name+': '+str(error))
        try:metrics.write_trace()
        except Exception as error:cleanup_errors.append('timing_trace: '+str(error))
        if cleanup_errors:failure=failure or '; '.join(cleanup_errors)
        if control_loop is not None:counts['controls']=control_loop.count
        if safety_telemetry is not None:counts['safety']=safety_telemetry.count
        counts['source_scans']=counts['scans'] if scan_inbox is None else scan_inbox.received
        for stream in streams.values():stream.close()
        report=dict(success=failure is None,error=failure,counts=counts,metrics=metrics.summary(),source_metrics=source_metrics,imu_time_calibration=timing,
            source_bag=None if bag is None else str(bag),configuration_source=str(source/'run.json') if source else 'current_module_configs',mode=mode,rate=args.rate,
            internal_ros_transport=False,native_localization=True,gpu_device=str(torch.cuda.get_device_name(0)),
            scope='supervised_live_navigation; inspect chassis events for transmitted targets; not physical acceptance' if args.execute else
                'nonexecuting_shadow; bag motion is observed input, never navigation execution feedback',
            detector=detector.metadata,environment_config=env_config,runtime_config=settings,scene_export=export_report,display=display_report,
            decision_inputs=decision_report,planning_record=planning_report,association_delivery=association_report,
            map_scheduling=dict(policy='synchronous_archive') if map_worker is None else dict(map_worker.counts,policy='one_active_one_latest_pending'),
            scan_scheduling=dict(policy='synchronous_archive') if scan_inbox is None else dict(policy='one_active_one_latest_pending',received=scan_inbox.received,replaced=scan_inbox.replaced),
            main_process_cpu_s=process_cpu_s,run_wall_s=run_wall_s,cleanup_errors=cleanup_errors)
        report['route_replay']=None if route_replay is None else route_replay.summary()
        report['agent_observations']=agent_report
        report['termination']='user_stop' if stop_requested else 'source_end'
        (output/'summary.json').write_text(json.dumps(report,allow_nan=False,indent=2)+'\n')
        manifest=dict(success=failure is None,source='live' if args.live else 'bag',clock_domain=clock_domain,backend='native_roll_fast_lio',mode=mode,bag=None if bag is None else str(bag),imu_time_calibration=timing,
            algorithm_parameters=config,localization_source='fast_lio' if mode=='odometry' else 'roll_fast_lio',
            map=str(output/'map') if mode=='mapping' else str(Path(config['roll']['saveKeyframeMapDirectory']).parent) if mode=='localization' else None,
            playback=dict(records='localization.jsonl',bag_fingerprint=None if bag is None else file_fingerprint(bag),start_offset_s=args.start),
            people=dict(configuration=det_cfg,detector=detector.metadata),
            navigation_motion=dict(source=motion_source,parameters=motion),
            runtime=dict(summary='summary.json',internal_ros_transport=False,core_image_association=False,
                rgb_requested=with_rgb))
        manifest['termination']=report['termination']
        manifest['start']=args.start;manifest['duration']=args.duration
        manifest['association_coverage']=args.coverage
        manifest['interactive']=dict(enabled=args.interactive_session,execution_requested=args.execute,
            vehicle_source=motion.get('vehicle_source'),chassis_port=args.chassis_port,
            physical_acceptance=False,task_completed=False)
        if route_replay is not None:manifest['route_replay']=route_replay.summary()
        if decision_report is not None:manifest['decision_inputs']=dict(directory='decision_inputs',**decision_report)
        if mode in ('mapping','localization'):
            from perception.mapping_localization.results import map_fingerprint
            if Path(manifest['map'],'COMPLETE').is_file():manifest['playback']['map_fingerprint']=map_fingerprint(manifest['map'])
        if journal is not None and not cleanup_errors:manifest['environment_scans_fingerprint']=file_fingerprint(output/'environment/scans.sqlite')
        manifest['detector_metadata']=detector.metadata;manifest['people_transport']=det_cfg['ros']
        if camera_record is None and decision_report and decision_report.get('image_preparation'):
            camera_record=decision_report['image_preparation']['calibration']
        if camera_record is not None:manifest['camera_calibration']=camera_record
        (output/'run.json').write_text(json.dumps(manifest,allow_nan=False,indent=2)+'\n')
    if failure:raise RuntimeError(failure)
    print(json.dumps(dict(success=True,output=str(output),counts=counts,metrics=metrics.summary())))


if __name__=='__main__':main()
