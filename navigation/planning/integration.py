"""Explicit isolated raw-bag perception/planning load probe; never robot commands."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from xmlrpc.client import ServerProxy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def process_sample(root,previous,interval):
    """Owned descendants only; CPU percent uses one core=100%, RSS is summed."""
    pending=[root];seen=set();cpu=0.;rss=0.;ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            base=Path('/proc')/str(pid);fields=(base/'stat').read_text().rsplit(')',1)[1].split()
            value=(int(fields[11])+int(fields[12]))/ticks;key=(pid,fields[19])
            if key in previous:cpu+=max(0.,value-previous[key])
            previous[key]=value;rss+=int(fields[21])*page
            pending.extend(map(int,(base/'task'/str(pid)/'children').read_text().split()))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return dict(cpu_percent=cpu/max(interval,1e-6)*100,rss_sum_mib=rss/2**20,processes=len(seen))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True,help='Read-only existing run providing bag/map/seed provenance')
    parser.add_argument('--static-map',required=True)
    parser.add_argument('--goal',type=float,nargs=2,required=True,help='Explicit map XY, metres')
    parser.add_argument('--duration',type=float,default=65.)
    parser.add_argument('--local-at',type=float,help='Explicit local-mode test time; default duration minus 10 seconds')
    parser.add_argument('--output',required=True)
    parser.add_argument('--shadow',action='store_true',help='Include independent ROS shadow tracking/safety processes')
    parser.add_argument('--faults',action='store_true',help='Inject shadow overrides, clock pause and owned planner exit')
    parser.add_argument('--cpu-cores',type=int,help='Constrain this probe and descendants; not a device-equivalence claim')
    parser.add_argument('--collision-backend',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if not 20<=args.duration<=900 or not np.isfinite(args.goal).all():raise ValueError('Bounded finite probe required')
    if args.faults and (not args.shadow or args.duration<40):raise ValueError('Fault probe requires shadow and at least 40 seconds')
    available=sorted(os.sched_getaffinity(0))
    if args.cpu_cores is not None:
        if not 1<=args.cpu_cores<=len(available):raise ValueError('Invalid CPU core constraint')
        os.sched_setaffinity(0,available[:args.cpu_cores])
    local_at=args.duration-10 if args.local_at is None else args.local_at
    if not 2<local_at<args.duration:raise ValueError('Local-mode test must be within replay')
    from .inputs import load_config as navigation_config
    from .motion import load_motion_config
    from navigation.control.shadow import load_config as shadow_config
    import yaml
    motion=load_motion_config();motion['collision_backend']=args.collision_backend
    from .reference import load_config as reference_config,planning_config
    from navigation.control.dwa import load_config as dwa_config
    config_snapshot=dict(planning=planning_config(navigation_config()),motion=motion,runtime=reference_config(),
        controller='dwa',reference_algorithm='hybrid_astar',
        dwa=dwa_config(),shadow=shadow_config(),
        safety=yaml.safe_load((ROOT/'navigation/safety/config.yaml').read_text()))
    out=Path(args.output).resolve()
    if not any(p in out.parents for p in [ROOT/'outputs',Path('/tmp')]):raise ValueError('Output must be in outputs or /tmp')
    out.mkdir(parents=True,exist_ok=False)
    source=json.loads((Path(args.source_run)/'run.json').read_text())
    from perception.environment.storage import StaticEnvironmentMap
    from perception.mapping_localization.results import file_fingerprint,map_fingerprint
    asset=StaticEnvironmentMap(args.static_map)
    if file_fingerprint(source['bag'])!=source['playback']['bag_fingerprint']:raise ValueError('Source bag changed')
    if map_fingerprint(source['map'])!=source['playback']['map_fingerprint']:raise ValueError('Source map changed')
    import rospy
    from sensor_msgs.msg import PointCloud2
    from social_nav_localization.msg import LocalizationState
    from social_nav_environment.msg import EnvironmentGrid
    from rospy.numpy_msg import numpy_msg
    from std_msgs.msg import String
    from perception.environment.scene import snapshot_from_message
    from .online import planning_scene
    from perception.environment.layers import load_config
    lock=threading.RLock();counts=Counter();reasons=Counter();shadow_reasons=Counter();samples={};state=[None];grids={};ready={};metrics=[]
    start=[None];phases=[];processes=[];roles={};resource_stop=threading.Event();resource_thread=None
    with tempfile.TemporaryDirectory(prefix='navigation-full-load-') as temporary:
        scratch=Path(temporary)
        with socket.socket() as selector:selector.bind(('127.0.0.1',0));port=selector.getsockname()[1]
        env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_IP='127.0.0.1',
            ROS_HOME=str(scratch/'home'),ROS_LOG_DIR=str(scratch/'logs'))
        env.pop('ROS_HOSTNAME',None);os.environ.pop('ROS_HOSTNAME',None);os.environ.update(env)
        handles=[]
        def launch(command,name):
            handle=(scratch/(name+'.log')).open('w');handles.append(handle)
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(p);return p
        def wait(predicate,timeout=90):
            deadline=time.monotonic()+timeout
            while not predicate():
                if any(p.poll() is not None for p in processes):raise RuntimeError('Probe process exited; see logs')
                if time.monotonic()>deadline:raise TimeoutError('Probe readiness timeout')
                time.sleep(.05)
        def event(kind,stamp,**fields):
            with lock:
                counts[kind]+=1
                metrics.append(dict(kind=kind,stamp_s=stamp,receive_ros_s=rospy.Time.now().to_sec(),
                    receive_mono_s=time.monotonic(),**fields))
        def pose(m):
            state[0]=m;event('pose',m.header.stamp.to_sec(),health=m.health)
        def grid(m):
            stamp=m.header.stamp.to_sec();meta=json.loads(m.metadata_json)
            with lock:
                grids[stamp]=time.monotonic()
                while len(grids)>256:del grids[next(iter(grids))]
            event('grid',stamp,geometry_status=m.geometry_status,dynamic_status=meta.get('dynamic_filter',{}).get('status'))
            # A handful of causal observations is enough for review; no full grid archive.
            if start[0] is not None:
                offset=stamp-start[0]
                for t in (20,35,47,50,60):
                    if t<=offset<t+.3 and t not in samples:
                        scene=planning_scene(snapshot_from_message(m),clock_domain='ros_sim_time',settings=load_config()['scene'])
                        scene.save(out/'samples'/str(t));samples[t]=dict(scene=str((out/'samples'/str(t)).relative_to(ROOT)),stamp_s=stamp)
        def people(m):
            row=json.loads(m.data)
            event('people',row['stamp_s'],health=row['health'],timing_s=row.get('timing_s'),detections=len(row['detections']),
                tracks=len(row['tracks']),dropped_busy=row.get('dropped_busy'))
        def planning(m):
            row=json.loads(m.data);r=row.get('result')
            with lock:
                reasons[row.get('reason') or row['status']]+=1
                if r is not None and r['result_binding'] not in ready:
                    ready[r['result_binding']]=r
                    event('trajectory',r['input_binding']['as_of'],binding=r['result_binding'],scene=r['input_binding']['scene'],
                        progress=r['trajectory'].get('route_progress',{}).get('progress_type'),
                        grid_to_receive_ms=(time.monotonic()-grids[r['input_binding'].get('geometry_as_of',r['input_binding']['as_of'])])*1000 if r['input_binding'].get('geometry_as_of',r['input_binding']['as_of']) in grids else None)
                if row['status'] not in ('trajectory_ready','control_ready'):event('planning_unavailable',rospy.Time.now().to_sec(),reason=row.get('reason'))
                for entry in row.get('worker_events',[]):
                    if 'role' in entry:event('worker',entry.get('as_of',rospy.Time.now().to_sec()),worker_event=entry['event'],
                        role=entry['role'],elapsed_ms=entry.get('elapsed_ms'),compute_ms=entry.get('compute_ms'),reason=entry.get('reason'))
        def shadow(m):
            row=json.loads(m.data)
            with lock:shadow_reasons[row['reason'] or 'allowed']+=1
            event('shadow',row['stamp_s'],row=row)
        def resources():
            previous={};last=time.monotonic()
            while not resource_stop.wait(1.):
                now=time.monotonic();cpu={name:process_sample(p.pid,previous,now-last) for name,p in list(roles.items())};last=now
                gpu=None;device=env.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
                if device:
                    try:
                        probe=subprocess.run(['nvidia-smi','-i',device,'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                            capture_output=True,text=True,timeout=2,check=True)
                        usage,memory=map(float,probe.stdout.strip().split(','));gpu=dict(device=device,utilization_percent=usage,memory_mib=memory)
                    except (ValueError,OSError,subprocess.SubprocessError):gpu=dict(status='unavailable')
                event('resource',rospy.Time.now().to_sec(),processes=cpu,gpu=gpu)
        try:
            master=launch(['roscore','-p',str(port)],'master')
            roles['ros_master']=master
            def master_ready():
                try:return ServerProxy(env['ROS_MASTER_URI']).getPid('/load_probe')[0]==1
                except OSError:return False
            wait(master_ready,20);ServerProxy(env['ROS_MASTER_URI']).setParam('/load_probe','/use_sim_time',True)
            rospy.init_node('navigation_load_probe',disable_signals=True)
            subscribers=[rospy.Subscriber('/localization/state',LocalizationState,pose,queue_size=128),
                rospy.Subscriber('/perception/environment/grid',numpy_msg(EnvironmentGrid),grid,queue_size=2,buff_size=16*1024*1024),
                rospy.Subscriber('/perception/people/json',String,people,queue_size=32),
                rospy.Subscriber('/navigation/planning/state',String,planning,queue_size=64),
                rospy.Subscriber('/rslidar_points',PointCloud2,lambda m:event('raw_scan',m.header.stamp.to_sec()),queue_size=32)]
            goal_pub=rospy.Publisher('/navigation/planning/goal',String,queue_size=1,latch=True)
            cancel_pub=rospy.Publisher('/navigation/planning/cancel',String,queue_size=1)
            if args.shadow:
                subscribers.append(rospy.Subscriber('/navigation/shadow/state',String,shadow,queue_size=128))
                override_pub=rospy.Publisher('/navigation/shadow/override',String,queue_size=8)
            command=['bash','perception/scripts/run_ros_diagnostic.sh','--source','live','--mode','localization','--localization-map',source['map'],
                '--pose',*map(str,source['pose']),'--environment','--environment-backend','cuda','--scene',
                '--static-map',args.static_map,'--allow-local-only','--people-python',sys.executable,'--out',str(out/'perception')]
            if source.get('association_coverage'):command+=['--association-coverage',source['association_coverage']]
            perception=launch(command,'perception')
            roles['perception']=perception
            # The session announces readiness after localization, 3D tracking and
            # requested environment consumers are ready; YOLO is not required.
            wait(lambda:'Session ready:' in (scratch/'perception.log').read_text())
            command=[sys.executable,'-B','-m','navigation.planning','online','--geometry','historical','--static-map',args.static_map,
                '--collision-backend',args.collision_backend]
            if args.shadow:command.append('--shadow')
            node=launch(command,'planning');roles['planning_and_tracking']=node
            safety=None
            if args.shadow:
                safety=launch([sys.executable,'-B','-m','navigation.safety.shadow','--geometry','historical','--static-map',args.static_map,
                    '--collision-backend',args.collision_backend],'safety')
                roles['independent_safety']=safety
            wait(lambda:goal_pub.get_num_connections()>=(2 if args.shadow else 1),20)
            resource_thread=threading.Thread(target=resources,daemon=True);resource_thread.start()
            # Only raw sensors are replayed. SLAM, detectors, tracking and maps run anew.
            topics=['/rslidar_points','/camera/imu','/camera/color/image_raw','/camera/color/camera_info','/tf_static']
            playback=launch(['rosbag','play','--clock','--hz','100','--duration',str(args.duration),source['bag'],'--topics',*topics],'playback')
            roles['bag_playback']=playback
            request=dict(request_id='load:mapped',task_id='load_probe',goal_revision=1,mode='mapped',frame_id='map',
                target_reference='base_link',map_snapshot_id=asset.manifest['snapshot_id'],goal=args.goal,goal_behavior='pass_through')
            goal_pub.publish(String(data=json.dumps(request)));sent_local=False;cancelled=False
            began=time.monotonic();last_print=began;fault_done=set();pause_end=None;planner_killed=False
            def phase(kind,stamp,**fields):phases.append(dict(kind=kind,stamp_s=stamp,monotonic_s=time.monotonic(),**fields))
            while playback.poll() is None:
                if (perception.poll() is not None or node.poll() is not None and not planner_killed or
                        safety is not None and safety.poll() is not None):raise RuntimeError('Perception/planning/safety exited')
                now=rospy.Time.now().to_sec()
                if start[0] is None and now>0:start[0]=now
                offset=0. if start[0] is None else now-start[0]
                if pause_end is not None and time.monotonic()>=pause_end:
                    os.killpg(playback.pid,signal.SIGCONT);phase('clock_resume',now);pause_end=None
                if args.faults:
                    for name,at in [('manual_override',args.duration*.2),('manual_release',args.duration*.2+1),
                            ('emergency_stop',args.duration*.3),('emergency_release',args.duration*.3+1),
                            ('clock_pause',args.duration*.4),('planner_exit',args.duration-3)]:
                        if offset<at or name in fault_done:continue
                        fault_done.add(name);phase(name,now)
                        if name=='clock_pause':os.killpg(playback.pid,signal.SIGSTOP);pause_end=time.monotonic()+.8
                        elif name=='planner_exit':os.killpg(node.pid,signal.SIGKILL);planner_killed=True
                        else:override_pub.publish(String(data=json.dumps(dict(manual_override=name=='manual_override',emergency_stop=name=='emergency_stop'))))
                if offset>=local_at-2 and not cancelled:
                    cancel_pub.publish(String(data=json.dumps(dict(task_id='load_probe',goal_revision=1))));cancelled=True
                    phase('cancel',now)
                if offset>=local_at and not sent_local and state[0] is not None and state[0].health==0:
                    from perception.mapping_localization.context import pose_matrix
                    from .footprint import VehicleGeometry
                    from .inputs import load_config as planning_config
                    m=state[0];p,q=m.pose.position,m.pose.orientation
                    g=VehicleGeometry.historical(planning_config()['geometry_reference'])
                    p=g.pose(pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w]))
                    request.update(request_id='load:local',goal_revision=2,mode='local',frame_id=m.header.frame_id,
                        goal=(p[:2]+2*np.array([np.cos(p[2]),np.sin(p[2])])).tolist())
                    request.pop('map_snapshot_id',None);goal_pub.publish(String(data=json.dumps(request)));sent_local=True
                    phase('explicit_local_goal',now,request=request.copy())
                if time.monotonic()-began>args.duration+40:raise TimeoutError('Raw replay exceeded budget')
                if time.monotonic()-last_print>10:
                    print(json.dumps(dict(offset_s=offset,counts=dict(counts),last_reason=next(reversed(reasons),None))),flush=True);last_print=time.monotonic()
                time.sleep(.02)
            if playback.returncode:raise RuntimeError('Raw bag playback failed')
            time.sleep(1.)
            if args.shadow:
                graph=ServerProxy(env['ROS_MASTER_URI']).getSystemState('/load_probe')[2]
                unsafe=[name for name,publishers in graph[0] if name in ('/cmd_vel','/mobile_base/commands/velocity')]
                if unsafe:raise RuntimeError('Unexpected chassis publisher in isolated shadow graph')
                phase('shadow_graph_checked_no_chassis_publishers',rospy.Time.now().to_sec())
            perception.send_signal(signal.SIGINT);perception.wait(timeout=40)
            if perception.returncode:raise RuntimeError('Perception session failed to finalize')
            run=json.loads((out/'perception/run.json').read_text())
            if not run['success']:raise RuntimeError('Perception run unsuccessful')
            def percentiles(values):
                a=np.asarray([v for v in values if v is not None],float)
                return None if not len(a) else dict(samples=len(a),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),maximum=float(a.max()))
            summary=dict(scope='raw_sensor_one_speed_full_perception_and_planning_no_chassis',source_run=args.source_run,
                bag_fingerprint=file_fingerprint(source['bag']),duration_s=args.duration,phases=phases,
                counts=dict(counts),reasons=dict(reasons),unique_trajectories=len(ready),
                age_ms={kind:percentiles([(r['receive_ros_s']-r['stamp_s'])*1000 for r in metrics if r['kind']==kind])
                    for kind in ['pose','grid','people','trajectory']},
                grid_to_trajectory_receipt_ms=percentiles([r['grid_to_receive_ms'] for r in metrics if r['kind']=='trajectory']),
                environment=run.get('environment'),scene=run.get('scene_statistics'),perception_counts=run['counts'],samples=samples,
                initial_pose_source=args.source_run,executable=False,task_completed=False)
            summary.update(cpu_affinity=sorted(os.sched_getaffinity(0)),shadow=args.shadow,shadow_reasons=dict(shadow_reasons),collision_backend=args.collision_backend,
                navigation_config=config_snapshot,
                evaluation='open_loop_bag_shadow_not_command_execution',GPU_scope='shared_perception_device_load',
                shadow_age_ms=percentiles([(r['receive_ros_s']-r['row']['as_of'])*1000 for r in metrics if r['kind']=='shadow' and r['row'].get('as_of') is not None]),
                shadow_compute_ms={key:percentiles([r['row']['timing_ms'].get(key) for r in metrics if r['kind']=='shadow'])
                    for key in ('observation_join','feedback','independent_gate','total')},
                controller_ms=percentiles([r['row'].get('controller_ms') for r in metrics if r['kind']=='shadow']),
                worker_compute_ms={role:percentiles([r['compute_ms'] for r in metrics if r['kind']=='worker' and r['role']==role])
                    for role in ('global','local','repair','reference')},fault_responses=[])
            expected=dict(manual_override='manual_override',emergency_stop='emergency_stop',clock_pause='clock_stalled',
                planner_exit='proposal_timeout',cancel='no_active_goal')
            for entry in phases:
                if entry['kind'] not in expected:continue
                found=next((r for r in metrics if r['kind']=='shadow' and 0<=r['receive_mono_s']-entry['monotonic_s']<=.75 and
                    r['row']['reason']==expected[entry['kind']] and r['row']['velocity']==[0.,0.]),None)
                summary['fault_responses'].append(dict(kind=entry['kind'],stop_recommendation_observed=found is not None,
                    response_ms=None if found is None else (found['receive_mono_s']-entry['monotonic_s'])*1000,actual_stop_verified=False))
            (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
            with (out/'metrics.jsonl').open('w') as f:
                for row in metrics:f.write(json.dumps(row,allow_nan=False)+'\n')
            (out/'trajectories.json').write_text(json.dumps(list(ready.values()),allow_nan=False))
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException:
            for name in ['perception','planning','playback','safety']:
                path=scratch/(name+'.log')
                if path.exists():(out/(name+'_failure.log')).write_text(path.read_text())
            raise
        finally:
            resource_stop.set()
            if resource_thread is not None:resource_thread.join(timeout=3)
            if 'playback' in locals() and playback.poll() is None:
                try:os.killpg(playback.pid,signal.SIGCONT)
                except ProcessLookupError:pass
            for process in reversed(processes):
                if process.poll() is None:
                    # Signal parent first so its existing session owner cleans its children.
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=45)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
            for handle in handles:handle.close()
            rospy.signal_shutdown('Full load probe complete')


if __name__=='__main__':main()
