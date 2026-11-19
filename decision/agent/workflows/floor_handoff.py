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

import numpy as np
from decision.agent import ROOT, CONFIG_DIR
from decision.agent.adapters.manual import ManualNavigation
from decision.agent.adapters.runtime import ManualRuntimeClient


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

    def start(self, session, goal_id):
        from ..core.elevator_transition import ElevatorTransition
        from ..adapters.elevator_camera import ElevatorCamera
        from ..adapters.shadow import load_catalog_map
        from ..tools.spatial import render
        requirement=session.task_goals.current(goal_id)['requirement']
        state=session.navigation.snapshot()
        if (self.transition or requirement.get('stage_type')!='elevator' or session.skill.record['handle'] or
                state.get('stopped') is not True or state.get('input_stale') or state.get('localization_status')!='valid'):
            raise ValueError('Pausing localization requires the current elevator stage, stopped and localized')
        target=session.task_goals.floor_catalogs[requirement['to_floor']]
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
            if result['status']!='failed' and time.monotonic()<self.attempt_deadline:
                return
            self.last_attempt.update(result=result.get('error') or result.get('wait_reason') or 'association_wait_expired')
            self.handoff.close()
            self.handoff=None
        if session.state['status']!='waiting_monitor':
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
        return sample

    def observe(self,judgment,evidence,retain_observation):
        if self.frame is None or self.transition.status not in ('awaiting_entry','awaiting_exit'):
            raise ValueError('Ride observation no longer pending')
        self.transition.observe(camera_record=self.frame['record'],image=self.frame['image'],
                                judgment=judgment,evidence=evidence)
        if retain_observation:
            rgb=dict(record=self.frame['record'],image=self.frame['image'])
            self.retained=(self.retained+[rgb])[-4:]
        self.transition.observations.clear()
        if self.transition.status=='awaiting_destination_localization':
            self.retained.clear()
        return dict(status=self.transition.status)

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
        target=session.task_goals.floor_catalogs[self.transition.to_floor]
        self.handoff=FloorHandoff(session=session,transition=self.transition,target_catalog=target,
            config_path=self.configs[self.transition.to_floor],seed=seed,output=self.output,source_socket=self.socket_path)
        self.last_attempt=dict(base_pose=pose,scan_stamp_s=seed['scan_stamp_s'],evidence=evidence)
        self.attempt_deadline=time.monotonic()+90.
        return dict(status='waiting_destination_localization',attempt=self.handoff.run_dir.name)

    def attachments(self):
        items=[('historical_elevator_rgb',f['record'],f['image'],'JPEG') for f in self.retained]
        items.append(('current_elevator_rgb',self.frame['record'],self.frame['image'],'JPEG'))
        if self.frame.get('scan_record'):
            items.append(('current_local_lidar',self.frame['scan_record'],self.frame['scan_image'],'PNG'))
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
        self.attempt_overlay=None
        self.last_stamp=None


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
        transform=seed.get('destination') or {}
        if (seed.get('verified_localization') is not False or seed.get('floor_id')!=target_catalog.value.get('floor_id') or
                seed.get('source')!='agent_destination_scan_and_map' or
                (seed.get('task_id'),seed.get('goal_id'))!=(transition.task_id,transition.goal_id) or
                seed.get('exit_stamp_s')!=transition.last_observation_stamp_s or
                not isinstance(seed.get('scan_stamp_s'),(int,float)) or not seed['scan_stamp_s']>seed['exit_stamp_s'] or
                transform.get('from_floor')!=transition.from_floor or transform.get('to_floor')!=transition.to_floor or
                transform.get('to_map_id')!=transition.destination_map_id):
            raise ValueError('Expected a provisional seed for the destination floor')
        self.session,self.transition,self.target_catalog=session,transition,target_catalog
        self.config=Path(config_path).resolve();self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        identity=uuid.uuid4().hex
        self.seed_path=self.output/('floor_seed_'+identity+'.json')
        self.seed_stamp=seed.get('scan_stamp_s')
        self.seed_path.write_text(json.dumps(seed,ensure_ascii=False,allow_nan=False,indent=2))
        self.run_dir=self.output/('floor_observation_'+identity)
        self.log_path=self.output/('floor_observation_'+identity+'.log')
        self.owner_pid=observation_owner(source_socket,allow_missing=True)
        self.process=None;self.log=None;self.status='stopping_source';self.error=None;self.wait_reason=None
        if self.owner_pid is not None:
            try:os.kill(self.owner_pid,signal.SIGINT)
            except ProcessLookupError:self.owner_pid=None

    def _source_stopped(self):
        if self.owner_pid is None:return True
        proc=Path('/proc')/str(self.owner_pid)
        if not proc.exists():return True
        try:
            state=next(line for line in (proc/'status').read_text().splitlines() if line.startswith('State:'))
            return '\tZ' in state
        except (FileNotFoundError,StopIteration):return True

    def tick(self):
        if self.status=='stopping_source' and self._source_stopped():
            # All runtime arguments are passed positionally, not interpolated as shell text.
            args=[sys.executable,'-B','-m','runtime.interactive','--live','--config',str(self.config),
                  '--output',str(self.run_dir),'--agent-observations','--handoff-seed',str(self.seed_path),
                  '--vehicle-profile','configs/wheelchair_vehicle.yaml',
                  '--gpu',os.environ.get('GPU_ID','0')]
            enabled=self.session.navigation.value.get('local_skills') or {}
            if enabled.get('follow') is True:args.append('--agent-follow-candidates')
            if enabled.get('seating') is True:args.append('--agent-seating')
            try:
                self.log=self.log_path.open('x')
                self.process=subprocess.Popen(['bash','-c',
                    'source perception/scripts/ros1_env.sh && source navigation/commissioning/performance.env && exec taskset -c "$SOCIAL_NAV_COMMISSIONING_CPU_LIST" "$@" --environment-update-hz "$SOCIAL_NAV_COMMISSIONING_ENVIRONMENT_HZ" --environment-window-m "$SOCIAL_NAV_COMMISSIONING_WINDOW_M" --roll-gravity-deg "$SOCIAL_NAV_COMMISSIONING_ROLL_GRAVITY_DEG"',
                    'floor-observer',*args],cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            except OSError as error:
                self.status='failed';self.error='destination_start_failed: '+type(error).__name__
                return self.snapshot()
            self.status='starting_destination'
        if self.status=='starting_destination':
            if self.process.poll() is not None:
                self.status='failed';self.error='destination_observation_exited';return self.snapshot()
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
                    observation_dir=str(self.run_dir),log=str(self.log_path))

    def close(self):
        if self.process is not None and self.status!='complete' and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=2.)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.log is not None:self.log.close();self.log=None
