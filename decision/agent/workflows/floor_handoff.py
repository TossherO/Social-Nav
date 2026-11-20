"""Supervise an observation-only runtime replacement for a floor transition."""
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import hashlib

import numpy as np
from decision.agent import ROOT, CONFIG_DIR
from decision.agent.adapters.manual import ManualNavigation
from decision.agent.adapters.runtime import ManualRuntimeClient


REGISTRATION_TIMEOUT_S = 30.


def scan_alignment_preview(crop,bounds,base_pose,base_xy_dm):
    """Plot measured scan returns under a proposed pose, in the original metric crop."""
    from PIL import Image, ImageDraw
    from ..tools.spatial import compact_map_layout
    xy=np.asarray(base_xy_dm,dtype=float)*.1
    x,y,yaw=base_pose
    c,s=np.cos(yaw),np.sin(yaw)
    mapped=xy@np.array([[c,s],[-s,c]])+[x,y]
    x0,y0,x1,y1=bounds
    mapped=mapped[(mapped[:,0]>=x0)&(mapped[:,0]<x1)&(mapped[:,1]>y0)&(mapped[:,1]<=y1)]
    layout=compact_map_layout(bounds,max_edge=1024)
    with Image.open(io.BytesIO(crop)) as source:
        canvas=source.convert('RGB')
    draw=ImageDraw.Draw(canvas)
    for px,py in mapped:
        col=round(layout['left']+(px-x0)*layout['scale'])
        row=round(layout['top']+(y1-py)*layout['scale'])
        draw.ellipse((col-1,row-1,col+1,row+1),fill='#c01a99')
    out=io.BytesIO()
    canvas.save(out,format='PNG')
    return out.getvalue()


def observation_owner(socket_path, *, allow_missing=False):
    """Identify only the same-user interactive process that owns this session."""
    socket_path=Path(socket_path).resolve()
    if socket_path.name!='commands.sock' or socket_path.parent.name!='run':
        raise ValueError('Expected an interactive observation socket')
    directory=socket_path.parent.parent
    record=json.loads((directory/'session.json').read_text())
    pid=record.get('owner_pid')
    if (record.get('source')!='live' or record.get('execute_requested') is not False or
            record.get('agent_observations') is not True or type(pid) is not int or pid<=1):
        raise ValueError('The prior session is not an owned observation-only Agent runtime')
    proc=Path('/proc')/str(pid)
    try:
        owner=proc.stat().st_uid
        status=(proc/'status').read_text()
        if allow_missing and any(line.startswith('State:') and '\tZ' in line for line in status.splitlines()):
            return None
        args=(proc/'cmdline').read_bytes().decode('utf-8').split('\0')
    except FileNotFoundError:
        if allow_missing:return None
        raise ValueError('Observation owner has already stopped') from None
    if owner!=os.getuid():raise ValueError('Observation owner belongs to another user')
    if 'runtime.interactive' not in args or '--output' not in args:
        raise ValueError('Observation owner is not the expected interactive process')
    i=args.index('--output')
    if i+1>=len(args) or (ROOT/args[i+1]).resolve()!=directory:
        raise ValueError('Observation owner output does not match the selected socket')
    return pid


def follow_runtime_options(socket_path):
    """Preserve the source session's explicitly selected follow input settings."""
    path=Path(socket_path).resolve().parent.parent/'command.json'
    command=json.loads(path.read_text())
    if not isinstance(command,list) or not all(isinstance(x,str) for x in command):
        raise ValueError('Source runtime command is invalid')
    result=[]
    for flag in ('--t2-backend','--t2-jpeg-quality'):
        indexes=[i for i,value in enumerate(command) if value==flag]
        if len(indexes)>1 or indexes and indexes[0]+1>=len(command):
            raise ValueError('Source T2 option is ambiguous')
        if indexes:
            value=command[indexes[0]+1]
            if flag=='--t2-backend' and value not in ('dspark','hosted'):
                raise ValueError('Source T2 backend is invalid')
            if flag=='--t2-jpeg-quality' and (not value.isdigit() or not 1<=int(value)<=95):
                raise ValueError('Source T2 JPEG quality is invalid')
            result.extend((flag,value))
    if '--t2-backend' not in result:
        raise ValueError('Source T2 backend was not recorded; restart the observation with an explicit backend')
    if command.count('--t2-raw-rgb')>1:raise ValueError('Source T2 mode is ambiguous')
    if '--t2-raw-rgb' in command:result.append('--t2-raw-rgb')
    return result


def load_standard_start(target,floor):
    standards=json.loads((CONFIG_DIR/'elevator_standard_starts.json').read_text())
    standard=standards.get(floor)
    if standard is None:return None,None
    pose=standard.get('pose_rpy_xyz')
    if (standard.get('map_id')!=target.value['localization_map_id'] or
            standard.get('frame_id')!=target.value['frame_id'] or standard.get('reference')!='rslidar' or
            not isinstance(pose,list) or len(pose)!=6 or not np.isfinite(pose).all()):
        raise ValueError('Standard elevator start does not match the destination map')
    path=(ROOT/standard['reference_observation']).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Standard reference must be inside the project')
    meta=json.loads(path.read_text());rgb=(path.parent/'rgb.jpg').read_bytes();scan=(path.parent/'lidar.png').read_bytes()
    if (meta.get('map_id')!=standard['map_id'] or hashlib.sha256(rgb).hexdigest()!=meta.get('rgb_sha256') or
            hashlib.sha256(scan).hexdigest()!=meta.get('lidar_sha256') or
            not rgb.startswith(b'\xff\xd8') or not rgb.endswith(b'\xff\xd9') or
            not scan.startswith(b'\x89PNG\r\n\x1a\n')):
        raise ValueError('Standard elevator reference is missing or changed')
    return copy.deepcopy(standard),(meta,rgb,scan)


class ElevatorRide:
    """Observation-only equipment used by the ordinary task loop."""
    def __init__(self, configs, socket_path, output):
        self.configs,self.socket_path,self.output=configs,socket_path,output
        self.transition=self.camera=self.handoff=self.frame=None
        self.retained=[]
        self.last_attempt=None
        self.attempt_overlay=None
        self.last_stamp=None
        self.wait_reason=None
        self.standard=None
        self.standard_reference=None
        self.standard_attempted=False

    def start(self, session, goal_id):
        from ..core.elevator_transition import ElevatorTransition
        from ..adapters.elevator_camera import ElevatorCamera
        from ..adapters.shadow import load_catalog_map
        from ..tools.spatial import render
        requirement=session.task_goals.current(goal_id)['requirement']
        if (self.transition or session.navigation.scope!='manual' or requirement.get('stage_type')!='elevator' or
                session.skill.record['handle'] is not None):
            raise ValueError('Elevator observation requires a manual elevator stage without an active navigation call')
        enabled=session.navigation.value.get('local_skills') or {}
        if enabled.get('follow') is True:
            # Detect unrecorded legacy settings before stopping the source
            # observer; otherwise a later floor could silently switch backend.
            follow_runtime_options(self.socket_path)
        target=session.task_goals.floor_catalogs[requirement['to_floor']]
        standard,reference=load_standard_start(target,requirement['to_floor'])
        links=json.loads((CONFIG_DIR/'elevator_links.json').read_text())['links']
        exits=np.array([target.places[item[requirement['to_floor']]]['pose'][:2] for item in links])
        self.bounds=np.r_[exits.min(axis=0)-5,exits.max(axis=0)+5].tolist()
        asset=load_catalog_map(target)
        try:self.crop=render(target,asset.snapshot,self.bounds,None,compact=True)
        finally:asset.close()
        transition=ElevatorTransition(task_id=session.state['task_id'],goal_id=goal_id,
            from_floor=requirement['floor_id'],to_floor=requirement['to_floor'],
            destination_map_id=target.value['localization_map_id'])
        pid=observation_owner(self.socket_path)
        self.camera=ElevatorCamera()
        try:
            self.camera.start(session.navigation.geometry.base_from_lidar)
            os.kill(pid,signal.SIGINT)
        except BaseException:
            self.camera.close()
            raise
        self.transition=transition
        self.standard=copy.deepcopy(standard)
        self.standard_reference=reference
        self.standard_attempted=False
        self.frame=None
        self.last_attempt=None
        self.feedback=None
        return dict(status='accepted',source_floor=transition.from_floor,destination_floor=transition.to_floor)

    def poll(self,session):
        if self.handoff is not None:
            result=self.handoff.tick()
            if result['status']=='complete':
                self.socket_path=str(self.handoff.run_dir/'run/commands.sock')
                self.close()
                session.state.update(elevator_active=False,status='ready',reason=None)
                session._save('destination_localization_available')
                return
            if result['status']!='failed':
                return
            self.last_attempt['result']=result['error']
            if result.get('wait_reason'):
                self.last_attempt['wait_reason']=result['wait_reason']
            self.handoff.close()
            self.handoff=None
            if self.standard is not None:
                self.wait_reason=result['error']
                if self.last_attempt['source']!='operator_rviz' or result.get('manual_pose_confirmed'):
                    self.start_manual_selection(session)
                return
        if session.state['status']!='waiting_monitor':
            return
        if self.standard is not None and self.standard_attempted:
            return
        sample=self.camera.snapshot()
        if sample.get('status')!='available':
            self.wait_reason=sample.get('reason','sensor_observation_unavailable')
            return
        stamp=sample['record']['stamp_s']
        if self.last_stamp is not None and stamp<=self.last_stamp:
            return
        if self.transition.status=='awaiting_destination_localization':
            scan=sample.get('scan_record')
            latest_used=(self.last_attempt or {}).get('scan_stamp_s',self.transition.last_observation_stamp_s)
            if not scan or scan['stamp_s']<=latest_used:
                self.wait_reason='awaiting_post_exit_lidar_scan'
                return
        self.last_stamp=stamp
        self.wait_reason=None
        if self.standard is not None and self.transition.status=='awaiting_destination_localization':
            self.frame=sample
            try:self.restart_standard(session)
            except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                self.standard_attempted=True
                self.wait_reason='standard_start_localization_failed: '+str(error)[:180]
                self.start_manual_selection(session)
            return
        return sample

    def observe(self,judgment,evidence,retain_observation):
        if self.frame is None or self.transition.status!='monitoring_ride':
            raise ValueError('Ride observation no longer pending')
        self.transition.observe(camera_record=self.frame['record'],image=self.frame['image'],
                                judgment=judgment,evidence=evidence)
        if retain_observation:
            rgb=dict(record=self.frame['record'],image=self.frame['image'])
            self.retained=(self.retained+[rgb])[-4:]
        self.transition.observations.clear()
        if self.transition.status=='awaiting_destination_localization':
            self.retained.clear()
        return dict(status=self.transition.status,judgment=judgment,evidence=evidence,
                    source_stamp_s=self.frame['record']['stamp_s'])

    def restart(self,session,pose,evidence):
        from ..core.elevator_links import destination_seed
        if (self.transition.status!='awaiting_destination_localization' or not self.frame or
                not self.frame.get('scan_record') or self.handoff is not None):
            raise ValueError('Destination scan and observed exit are required')
        x,y,_=pose
        if not (self.bounds[0]<=x<=self.bounds[2] and self.bounds[1]<=y<=self.bounds[3]):
            raise ValueError('Selected base pose outside the destination waiting-area crop')
        seed=destination_seed(self.transition,pose,self.frame['scan_record'],session.navigation.geometry.base_from_lidar)
        points=self.frame.get('base_xy_dm')
        self.attempt_overlay=scan_alignment_preview(self.crop,self.bounds,pose,points) if points else None
        self.handoff=self._new_handoff(session,seed)
        self.last_attempt=dict(base_pose=pose,scan_stamp_s=seed['scan_stamp_s'],evidence=evidence)
        return dict(status='waiting_destination_localization',attempt=self.handoff.run_dir.name)

    def restart_standard(self,session):
        from ..core.elevator_links import standard_destination_seed
        if (self.standard is None or self.standard_attempted or not self.frame or
                not self.frame.get('scan_record') or self.handoff is not None):
            raise ValueError('Standard start needs a fresh scan and no active localization attempt')
        seed=standard_destination_seed(self.transition,self.standard,self.frame['scan_record'])
        self.handoff=self._new_handoff(session,seed)
        self.last_attempt=dict(source='configured_standard_start',scan_stamp_s=seed['scan_stamp_s'])
        self.standard_attempted=True

    def start_manual_selection(self,session):
        if not self.standard_attempted or self.handoff is not None:
            raise ValueError('Manual selection requires a failed standard start')
        self.last_attempt=dict(source='operator_rviz')
        try:
            self.handoff=self._new_handoff(session,seed=None)
        except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
            self.wait_reason='manual_selection_unavailable: '+str(error)[:180]

    def reopen_manual_selection(self,session):
        if (self.last_attempt or {}).get('source')!='operator_rviz':
            return
        if self.handoff is not None:
            if self.handoff.status!='waiting_manual_pose' or (self.handoff.run_dir/'initial_pose.json').exists():
                return
            self.handoff.close()
            self.handoff=None
        self.start_manual_selection(session)

    def _new_handoff(self,session,seed):
        floor=self.transition.to_floor
        return FloorHandoff(session=session,transition=self.transition,target_catalog=session.task_goals.floor_catalogs[floor],
            config_path=self.configs[floor],seed=seed,output=self.output,source_socket=self.socket_path)

    def attachments(self):
        history=self.retained[-2:] if self.standard_reference else self.retained
        items=[('historical_elevator_rgb',f['record'],f['image'],'JPEG') for f in history]
        items.append(('current_elevator_rgb',self.frame['record'],self.frame['image'],'JPEG'))
        if self.frame.get('scan_record'):
            items.append(('current_local_lidar',self.frame['scan_record'],self.frame['scan_image'],'PNG'))
        if self.standard_reference and self.transition.status=='monitoring_ride':
            meta,rgb,scan=self.standard_reference
            items.extend([('standard_start_reference_rgb',meta['rgb'],rgb,'JPEG'),
                          ('standard_start_reference_lidar',meta['lidar'],scan,'PNG')])
        if self.transition.status=='awaiting_destination_localization':
            items.append(('destination_waiting_area',dict(floor_id=self.transition.to_floor,
                map_id=self.transition.destination_map_id,bounds_xy_m=self.bounds),self.crop,'PNG'))
            if self.attempt_overlay and self.last_attempt and self.last_attempt.get('result'):
                items.append(('failed_pose_scan_overlay',dict(floor_id=self.transition.to_floor,
                    scan_stamp_s=self.last_attempt['scan_stamp_s'],proposed_base_pose=self.last_attempt['base_pose'],
                    meaning='magenta returns at the previous proposed pose, unverified'),self.attempt_overlay,'PNG'))
        return items

    def close(self):
        if self.handoff:self.handoff.close()
        if self.camera:self.camera.close()
        self.transition=self.camera=self.handoff=self.frame=None
        self.retained.clear()
        self.feedback=self.last_attempt=None
        self.standard=self.standard_reference=None
        self.standard_attempted=False
        self.attempt_overlay=None
        self.last_stamp=None
        self.wait_reason=None


class FloorHandoff:
    def __init__(self, *, session, transition, target_catalog, config_path, seed, output, source_socket):
        if transition.status!='awaiting_destination_localization':
            raise ValueError('The Agent has not observed elevator exit')
        if session.skill.record['handle'] is not None:
            raise ValueError('The source-floor navigation call is still active')
        stage=session.task_goals.current(transition.goal_id)['requirement']
        if (transition.task_id!=session.state['task_id'] or stage.get('stage_type')!='elevator' or
                stage.get('floor_id')!=transition.from_floor or stage.get('to_floor')!=transition.to_floor or
                session.catalog.value.get('floor_id')!=transition.from_floor or
                target_catalog.value.get('floor_id')!=transition.to_floor or
                target_catalog.value.get('localization_map_id')!=transition.destination_map_id):
            raise ValueError('Floor handoff does not match the active task and catalogs')
        self.session,self.transition,self.target_catalog=session,transition,target_catalog
        self.config=Path(config_path).resolve();self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        identity=uuid.uuid4().hex
        self.seed_path=self.seed_stamp=self.deadline=None
        if seed is not None:
            self._validate_seed(seed)
            self.seed_path=self.output/('floor_seed_'+identity+'.json')
            self.seed_path.write_text(json.dumps(seed,ensure_ascii=False,allow_nan=False,indent=2))
            self.seed_stamp=seed['scan_stamp_s']
            self.deadline=time.monotonic()+REGISTRATION_TIMEOUT_S
        self.run_dir=self.output/('floor_observation_'+identity)
        self.log_path=self.output/('floor_observation_'+identity+'.log')
        enabled=self.session.navigation.value.get('local_skills') or {}
        self.follow_options=follow_runtime_options(source_socket) if enabled.get('follow') is True else []
        self.owner_pid=observation_owner(source_socket,allow_missing=True)
        self.process=None;self.log=None;self.status='stopping_source';self.error=None;self.wait_reason=None
        if self.owner_pid is not None:
            try:os.kill(self.owner_pid,signal.SIGINT)
            except ProcessLookupError:self.owner_pid=None

    def _validate_seed(self,seed):
        transition=self.transition
        transform=seed.get('destination') or {}
        if (seed.get('verified_localization') is not False or seed.get('floor_id')!=transition.to_floor or
                seed.get('source') not in ('agent_destination_scan_and_map','standard_destination_start') or
                (seed.get('task_id'),seed.get('goal_id'))!=(transition.task_id,transition.goal_id) or
                seed.get('exit_stamp_s')!=transition.last_observation_stamp_s or
                not isinstance(seed.get('scan_stamp_s'),(int,float)) or not seed['scan_stamp_s']>seed['exit_stamp_s'] or
                transform.get('from_floor')!=transition.from_floor or transform.get('to_floor')!=transition.to_floor or
                transform.get('to_map_id')!=transition.destination_map_id):
            raise ValueError('Expected a provisional seed for the destination floor')
        if seed['source']=='standard_destination_start':
            pose=seed.get('lidar_pose_rpy_xyz');xy_yaw=seed.get('lidar_xy_yaw')
            if (not isinstance(pose,list) or len(pose)!=6 or
                    not isinstance(xy_yaw,list) or len(xy_yaw)!=3 or
                    not all(type(v) in (int,float) and np.isfinite(v) for v in pose+xy_yaw) or
                    not np.allclose([pose[3],pose[4],pose[2]],xy_yaw,rtol=0,atol=1e-9) or
                    not isinstance(seed.get('scan_frame'),str) or seed['scan_frame'].lstrip('/')!='rslidar'):
                raise ValueError('Invalid standard destination LiDAR seed')

    def _source_stopped(self):
        if self.owner_pid is None:return True
        proc=Path('/proc')/str(self.owner_pid)
        if not proc.exists():return True
        try:
            state=next(line for line in (proc/'status').read_text().splitlines() if line.startswith('State:'))
            return '\tZ' in state
        except (FileNotFoundError,StopIteration):return True

    def tick(self):
        if self.status in ('complete','failed'):
            return self.snapshot()
        if self.status=='waiting_manual_pose':
            path=self.run_dir/'initial_pose.json'
            if path.exists():
                try:
                    pose=json.loads(path.read_text())
                    stamp=pose['preview_scan_stamp_s']
                    confirmed=pose['confirmed_monotonic_s']
                    if (pose['source']!='confirmed_rviz_initialpose' or
                            not np.isfinite(stamp) or stamp<=self.transition.last_observation_stamp_s or
                            not np.isfinite(confirmed) or not 0<=confirmed<=time.monotonic()):
                        raise ValueError('Invalid destination pose confirmation')
                except (OSError,KeyError,TypeError,ValueError) as error:
                    self.status='failed';self.error='invalid_manual_pose: '+str(error)[:180]
                    return self.snapshot()
                self.seed_stamp=stamp
                self.deadline=confirmed+REGISTRATION_TIMEOUT_S
                self.status='starting_destination'
        if self.deadline is not None and time.monotonic()>=self.deadline:
            self.status='failed';self.error='destination_registration_timeout'
            return self.snapshot()
        if self.status=='stopping_source' and self._source_stopped():
            # All runtime arguments are passed positionally, not interpolated as shell text.
            args=[sys.executable,'-B','-m','runtime.interactive','--live','--config',str(self.config),
                  '--output',str(self.run_dir),'--agent-observations',
                  '--vehicle-profile','configs/wheelchair_vehicle.yaml',
                  '--gpu',os.environ.get('GPU_ID','0')]
            if self.seed_path is not None:args.extend(['--handoff-seed',str(self.seed_path)])
            enabled=self.session.navigation.value.get('local_skills') or {}
            if enabled.get('follow') is True:args.extend(['--agent-follow-candidates',*self.follow_options])
            if enabled.get('seating') is True:args.append('--agent-seating')
            try:
                self.log=self.log_path.open('x')
                self.process=subprocess.Popen(['bash','-c',
                    'source perception/scripts/ros1_env.sh && source navigation/commissioning/performance.env && exec taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" "$@" --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M" --roll-gravity-deg "$SOCIAL_NAV_COMMISSIONING_ROLL_GRAVITY_DEG"',
                    'floor-observer',*args],cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            except OSError as error:
                self.status='failed';self.error='destination_start_failed: '+type(error).__name__
                return self.snapshot()
            self.status='waiting_manual_pose' if self.seed_path is None else 'starting_destination'
        if self.status in ('waiting_manual_pose','starting_destination'):
            if self.process.poll() is not None:
                self.status='failed';self.error='destination_observation_exited';return self.snapshot()
            if self.status=='waiting_manual_pose':return self.snapshot()
            socket_path=self.run_dir/'run/commands.sock'
            if not socket_path.exists():return self.snapshot()
            try:
                navigation=ManualNavigation(self.target_catalog,ManualRuntimeClient(socket_path),automatic_arrival=True)
                state=navigation.tick();mapped=(navigation.value.get('control') or {}).get('agent_mapped_pose') or {}
                binding=state.get('map_binding') or {}
                if (state.get('input_stale') or state.get('origin_pose') is None or
                        state.get('localization_status')!='valid' or mapped.get('source')!='current_localization_and_scene_association' or
                        self.seed_stamp is not None and (mapped.get('association_stamp_s') is None or
                                                        mapped['association_stamp_s']<self.seed_stamp)):
                    self.wait_reason='destination_localization_not_ready'
                    return self.snapshot()
                association=dict(binding,local_session_id=state['session_id'],local_generation=state['generation'],
                                 local_frame=state['frame_id'],measurement_stamp_s=mapped.get('association_stamp_s'))
                if self.seed_path is None:association['initial_pose_source']='operator_rviz'
                candidate=copy.deepcopy(self.transition)
                candidate.accept_localization(association=association,now_s=state['stamp_s'],max_age_s=1.,
                                              clock_domain=state['clock_domain'])
                self.session.rebind_manual_floor(self.target_catalog,navigation,candidate)
                self.transition.association=candidate.association
                self.transition.status=candidate.status
            except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                self.wait_reason=type(error).__name__+': '+str(error)[:180]
                return self.snapshot()
            self.wait_reason=None
            self.status='complete'
        return self.snapshot()

    def snapshot(self):
        return dict(status=self.status,error=self.error,wait_reason=self.wait_reason,
                    manual_pose_confirmed=self.seed_path is None and self.seed_stamp is not None,
                    observation_dir=str(self.run_dir),log=str(self.log_path))

    def close(self):
        if self.process is not None and self.status!='complete' and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            # The interactive owner shuts down its runtime and RViz before a replacement can reuse ROS.
            try:self.process.wait(timeout=26.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2.)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:self.log.close();self.log=None
