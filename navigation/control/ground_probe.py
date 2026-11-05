"""One supervised, bounded straight-line protocol probe; never autonomous navigation.

Without --execute this module only prints the selected test profile. The serial
owner runs separately from the parent, enforces its own pulse deadline and
heartbeat, and returns to zero/raw mode 3. Physical stopping remains necessary
for host, OS or controller failures. Feedback layout/checksum are candidates.
"""
import argparse
from collections import deque
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import signal
import time

from .chassis_protocol import SOURCE,FeedbackFrames,encode_command


PROFILE=dict(linear_mps=.03,angular_rps=0.,pulse_s=.3,acceleration_mps2=.24,
    command_period_s=.05,heartbeat_timeout_s=.2,feedback_timeout_s=.25,
    settling_s=1.2,stationary_check_s=.5,arm_zero_s=.5,restore_zero_s=1.,
    stop_confirmation_timeout_s=3.,stationary_min_span_s=.2,
    max_observed_linear_mps=.15,max_observed_angular_rps=.15,
    max_average_linear_mps=.11,startup_turn_grace_s=1.,
    turn_warning_rps=.05,feedback_average_window_s=.4,feedback_average_min_frames=5)
# Development test boundaries, not measured hardware capability or safe travel.
LIMITS=dict(min_linear_mps=.001,max_linear_mps=.1,min_pulse_s=.1,max_pulse_s=3.)


def probe_profile(*,speed=None,pulse_s=None):
    """Freeze one explicit request before opening a device or output directory."""
    profile=dict(PROFILE)
    for name,value,low,high in (
            ('linear_mps',PROFILE['linear_mps'] if speed is None else speed,LIMITS['min_linear_mps'],LIMITS['max_linear_mps']),
            ('pulse_s',PROFILE['pulse_s'] if pulse_s is None else pulse_s,LIMITS['min_pulse_s'],LIMITS['max_pulse_s'])):
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('%s must be finite and in [%s,%s]'%(name,low,high))
        profile[name]=float(value)
    return profile


class ReceiptWindow:
    """Time-weighted candidate rates on host receipt time, not measurement time."""
    def __init__(self,duration,min_frames,max_gap):
        self.duration=duration;self.min_frames=min_frames;self.max_gap=max_gap;self.rows=deque(maxlen=128)

    def push(self,stamp,velocity):
        if self.rows and stamp<=self.rows[-1][0]:return
        if self.rows and stamp-self.rows[-1][0]>self.max_gap:self.rows.clear()
        self.rows.append((stamp,float(velocity[0]),float(velocity[1])))
        cutoff=stamp-self.duration
        while len(self.rows)>1 and self.rows[1][0]<=cutoff:self.rows.popleft()

    def average(self):
        if len(self.rows)<self.min_frames:return None
        end=self.rows[-1][0];begin=end-self.duration
        if self.rows[0][0]>begin+1e-9:return None
        totals=[0.,0.]
        for previous,current in zip(self.rows,list(self.rows)[1:]):
            left=max(previous[0],begin);right=current[0]
            if right<=left:continue
            fraction=(left-previous[0])/(right-previous[0])
            for k in (0,1):
                initial=previous[k+1]+fraction*(current[k+1]-previous[k+1])
                totals[k]+=(initial+current[k+1])*.5*(right-left)
        return dict(span_s=self.duration,frames=len(self.rows),
            mean_linear_mps=totals[0]/self.duration,mean_angular_rps=totals[1]/self.duration)


class PulseWindow:
    """A one-shot deadline; new feedback/heartbeats cannot prolong motion."""
    def __init__(self,start,*,expect_inhibited=True,duration=None,speed=None):
        self.profile=probe_profile(speed=speed,pulse_s=duration)
        self.start=self.previous=start;self.reason=None;self.expect_inhibited=expect_inhibited
        self.duration=self.profile['pulse_s'];self.last_feedback=-math.inf;self.stage='startup'
        args=(self.profile['feedback_average_window_s'],self.profile['feedback_average_min_frames'],self.profile['feedback_timeout_s'])
        self.all_window=ReceiptWindow(*args);self.steady_window=ReceiptWindow(*args)

    def diagnostics(self):
        return dict(stage=self.stage,clock_basis='host_monotonic_receipt_not_measurement',
            all_window=self.all_window.average(),steady_window=self.steady_window.average(),reason=self.reason)

    def command(self,now,*,heartbeat_at,feedback_at,feedback,parent_alive=True):
        if self.reason:return 0.
        reason=None
        if not all(math.isfinite(v) for v in (now,heartbeat_at,feedback_at)) or now<self.previous:
            reason='monotonic_clock_invalid'
        elif not parent_alive or not 0<=now-heartbeat_at<=self.profile['heartbeat_timeout_s']:reason='parent_heartbeat_lost'
        elif not 0<=now-feedback_at<=self.profile['feedback_timeout_s']:reason='serial_feedback_stale'
        elif not feedback.get('candidate_checksum_matches'):reason='candidate_checksum_mismatch'
        elif feedback.get('mode_raw')!=1:reason='raw_mode_changed'
        else:
            velocity=feedback.get('velocity_mps_rps',[])
            if (len(velocity)!=2 or not all(math.isfinite(v) for v in velocity) or
                    abs(velocity[0])>self.profile['max_observed_linear_mps'] or
                    abs(velocity[1])>self.profile['max_observed_angular_rps']):reason='unexpected_velocity'
            elif any(feedback.get('joystick_axes_raw',[])) or any(feedback.get('joystick_mps_rps',[])):reason='joystick_takeover'
            elif self.expect_inhibited and (abs(velocity[0])>.005 or abs(velocity[1])>.01):reason='motion_while_inhibited'
            elif not self.expect_inhibited and velocity[0]<-.005:reason='unexpected_motion_direction'
            elif not self.expect_inhibited and feedback_at>=self.start and feedback_at>self.last_feedback:
                self.last_feedback=feedback_at;self.all_window.push(feedback_at,velocity)
                average=self.all_window.average()
                if average and average['mean_linear_mps']>self.profile['max_average_linear_mps']:
                    reason='sustained_linear_overspeed'
                if feedback_at-self.start>=self.profile['startup_turn_grace_s']:
                    self.stage='steady';self.steady_window.push(feedback_at,velocity)
                    steady=self.steady_window.average()
                    if steady and abs(steady['mean_angular_rps'])>self.profile['turn_warning_rps']:
                        reason=reason or 'persistent_unexpected_turn'
        self.previous=now
        if reason:self.reason=reason;return 0.
        if now>=self.start+self.duration:return 0.
        if now<self.start:self.reason='pulse_clock_reversed';return 0.
        return min(self.profile['linear_mps'],self.profile['acceleration_mps2']*(now-self.start))


class ProbeAbort(RuntimeError):pass


def summarize_motion(events,profile):
    """Recorded candidate response by stage; no physical verification implied."""
    begin=next((r['mono'] for r in events if r['kind']=='phase' and r['name']=='single_forward_pulse'),None)
    stop=next((r['mono'] for r in events if r['kind']=='phase' and r['name']=='restore_zero_raw_mode_3'),None)
    if begin is None:return None
    groups={name:[] for name in ('startup','steady','stopping')};first=None
    for row in events:
        if row['kind']!='feedback' or row['mono']<begin:continue
        velocity=row['feedback']['velocity_mps_rps']
        name='stopping' if stop is not None and row['mono']>=stop else (
            'startup' if row['mono']-begin<profile['startup_turn_grace_s'] else 'steady')
        groups[name].append(velocity)
        if first is None and any(velocity):first=row['mono']-begin
    return dict(clock_basis='host_monotonic_receipt_not_measurement',
        first_nonzero_candidate_delay_s=first,command_window_elapsed_s=None if stop is None else stop-begin,
        stages={name:dict(feedback_frames=len(values),max_abs_candidate_velocity=[max([abs(v[k]) for v in values],default=0) for k in (0,1)])
            for name,values in groups.items()})


def _serial_owner(connection,port,output,expect_inhibited,pulse_s=None,speed=None):
    import fcntl
    import os
    import select
    import subprocess
    import termios
    profile=probe_profile(speed=speed,pulse_s=pulse_s)
    cancelled=[False]
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,lambda *_:cancelled.__setitem__(0,True))
    out=Path(output);parser=FeedbackFrames();recent=deque(maxlen=16)
    fd=None;original=None;last_rx=None;heartbeat=time.monotonic();parent_alive=True;sent=[];latest=None;pulse_deadline=None
    report=dict(scope='supervised_ground_protocol_probe',autonomous_navigation=False,
        expect_inhibited=expect_inhibited,profile=profile,limits=LIMITS,protocol_source=SOURCE,
        execution_feedback=False,feedback_protocol_verified=False,nonzero_hardware_commands_sent=False,
        completed=False,stop_feedback_observed=False,motion_verified=False,operator_observation_pending=True,phases=[])
    events=(out/'events.jsonl').open('x');raw_capture=(out/'feedback.bin').open('xb')
    event_rows=[];raw_bytes=bytearray();observed_velocities=[]

    def record(kind,**fields):
        if len(event_rows)<10000:event_rows.append(dict(kind=kind,mono=time.monotonic(),**fields))
        else:report['event_log_truncated']=True

    def write(linear,mode):
        if not 0<=linear<=profile['linear_mps']:raise ProbeAbort('Nonzero target outside the selected forward profile')
        if mode!=1 and linear:raise ProbeAbort('Only the probed raw mode may carry a nonzero target')
        if linear and (pulse_deadline is None or time.monotonic()>=pulse_deadline):linear=0.
        frame=encode_command(linear,0.,mode=mode)
        if os.write(fd,frame)!=len(frame):raise ProbeAbort('Partial command write')
        wire_linear=int.from_bytes(frame[2:4],byteorder='big',signed=True)/1000.
        sent.append((wire_linear,mode));report['nonzero_hardware_commands_sent']|=wire_linear!=0.
        record('write',linear_mps=wire_linear,requested_linear_mps=linear,angular_rps=0.,raw_mode=mode,frame_hex=frame.hex())

    def receive(*,cleanup=False,window=None):
        nonlocal latest,last_rx,heartbeat,parent_alive
        if parent_alive:
            try:
                while connection.poll():
                    message=connection.recv()
                    if isinstance(message,tuple) and len(message)==2 and message[0]=='heartbeat':
                        if 0<=time.monotonic()-message[1]<=profile['heartbeat_timeout_s']:
                            heartbeat=max(heartbeat,message[1])
                    else:cancelled[0]=True
            except (EOFError,OSError):parent_alive=False;cancelled[0]=True
        if not cleanup and (cancelled[0] or time.monotonic()-heartbeat>profile['heartbeat_timeout_s']):
            raise ProbeAbort('Parent cancelled, disconnected or stopped heartbeats')
        if select.select([fd],[],[],.01)[0]:
            try:data=os.read(fd,4096)
            except BlockingIOError:return
            if not data:return
            if len(raw_bytes)+len(data)<=131072:raw_bytes.extend(data)
            else:report['raw_log_truncated']=True
            for row in parser.feed(data):
                now=time.monotonic();last_rx=now;latest=row;recent.append((now,row))
                observed_velocities.append(row['velocity_mps_rps'])
                record('feedback',received_mono=now,feedback=row)
                if not cleanup and not row['candidate_checksum_matches']:raise ProbeAbort('Candidate feedback checksum mismatch')
                if window:
                    window.command(now,heartbeat_at=heartbeat,feedback_at=last_rx,feedback=row,parent_alive=parent_alive)
                    diagnostics=window.diagnostics()
                    if report.get('turning_guard',{}).get('stage')!=diagnostics['stage']:
                        record('control_stage',name=diagnostics['stage'])
                    report['turning_guard']=diagnostics
                    if window.reason:raise ProbeAbort(window.reason)

    def phase(name,duration,*,mode=None,pulse=False,cleanup=False,until_stationary=False):
        nonlocal pulse_deadline
        began=time.monotonic();next_write=began
        report['phases'].append(dict(name=name,started_mono=began))
        record('phase',name=name)
        try:connection.send(dict(phase=name,duration_s=duration,maximum_linear_mps=profile['linear_mps'] if pulse else 0.))
        except (BrokenPipeError,OSError):pass
        window=PulseWindow(began,expect_inhibited=expect_inhibited,duration=duration,speed=profile['linear_mps']) if pulse else None
        if pulse:pulse_deadline=began+duration
        while time.monotonic()-began<duration:
            receive(cleanup=cleanup,window=window);now=time.monotonic()
            if name!='receive_settling' and not cleanup and (last_rx is None or now-last_rx>profile['feedback_timeout_s']):
                raise ProbeAbort('Serial feedback stopped')
            if mode is not None and now>=next_write:
                linear=0.
                if window:
                    linear=window.command(now,heartbeat_at=heartbeat,feedback_at=last_rx,
                        feedback=latest or {},parent_alive=parent_alive)
                    report['turning_guard']=window.diagnostics()
                    if window.reason:raise ProbeAbort(window.reason)
                write(linear,mode);next_write=now+profile['command_period_s']
            if until_stationary and now-began>=profile['restore_zero_s'] and stationary(3):break
        if pulse:write(0.,1)
        if window:report['turning_guard']=window.diagnostics()
        if pulse:pulse_deadline=None
        report['phases'][-1]['ended_mono']=time.monotonic()

    def stationary(expected_mode):
        now=time.monotonic();fresh=[r for stamp,r in recent if 0<=now-stamp<=.3]
        fresh_times=[stamp for stamp,_ in recent if 0<=now-stamp<=.3]
        return (len(fresh)>=4 and fresh_times[-1]-fresh_times[0]>=profile['stationary_min_span_s'] and
            all(r['candidate_checksum_matches'] and r['mode_raw']==expected_mode and
            abs(r['velocity_mps_rps'][0])<=.005 and abs(r['velocity_mps_rps'][1])<=.01 and
            not any(r['joystick_axes_raw']) and not any(r['joystick_mps_rps']) for r in fresh))

    try:
        fd=os.open(port,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK);fcntl.ioctl(fd,termios.TIOCEXCL)
        owners=subprocess.run(['fuser',os.path.realpath(port)],capture_output=True,text=True,timeout=2.,
            env=dict(os.environ,LC_ALL='C'))
        pids={int(v) for v in owners.stdout.split() if v.isdigit()}
        if owners.returncode!=0 or pids!={os.getpid()} or 'Permission denied' in owners.stderr:
            raise ProbeAbort('Cannot establish exclusive serial ownership; no commands sent')
        original=termios.tcgetattr(fd);settings=copy.deepcopy(original)
        settings[0]=settings[1]=settings[3]=0
        settings[2]=(settings[2]&~(termios.CSIZE|termios.PARENB|termios.CSTOPB|termios.CRTSCTS))|termios.CS8|termios.CLOCAL|termios.CREAD
        settings[4]=settings[5]=termios.B19200;settings[6][termios.VMIN]=settings[6][termios.VTIME]=0
        termios.tcsetattr(fd,termios.TCSANOW,settings)
        phase('receive_settling',profile['settling_s'])
        phase('stationary_precheck',profile['stationary_check_s'])
        if not stationary(3):raise ProbeAbort('Expected a stationary neutral raw-mode-3 stream')
        phase('zero_raw_mode_1',profile['arm_zero_s'],mode=1)
        if not stationary(1):raise ProbeAbort('Raw-mode-1 zero acknowledgement or neutral feedback missing')
        phase('single_forward_pulse',profile['pulse_s'],mode=1,pulse=True)
        report['completed']=True
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        if fd is not None:
            try:
                if sent:
                    # Only zero commands after any cancellation or failure.
                    stop_started=time.monotonic()
                    phase('restore_zero_raw_mode_3',profile['stop_confirmation_timeout_s'],mode=3,cleanup=True,until_stationary=True)
                    report['stop_feedback_observed']=stationary(3)
                    report['stop_confirmation_elapsed_s']=time.monotonic()-stop_started
                    report['stop_confirmation_timed_out']=not report['stop_feedback_observed']
                elif latest is not None:report['stop_feedback_observed']=stationary(3)
            except BaseException as error:report['restore_error']=type(error).__name__+': '+str(error)
            finally:
                if original is not None:
                    try:termios.tcsetattr(fd,termios.TCSANOW,original)
                    except OSError as error:report['settings_restore_error']=str(error)
                try:fcntl.ioctl(fd,termios.TIOCNXCL)
                except OSError:pass
                os.close(fd)
        report.update(frames_written=len(sent),last_candidate_feedback=latest,
            nonzero_frames=sum(v!=0 for v,_ in sent),candidate_frames_seen=len(observed_velocities),
            nonzero_candidate_feedback_observed=any(abs(v[0])>.005 for v in observed_velocities),
            max_abs_candidate_velocity=[max([abs(v[i]) for v in observed_velocities],default=0) for i in (0,1)])
        report['response_summary']=summarize_motion(event_rows,profile)
        for row in event_rows:events.write(json.dumps(row,allow_nan=False)+'\n')
        raw_capture.write(raw_bytes)
        events.close();raw_capture.close()
        (out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        try:connection.send(dict(done=True,completed=report['completed'],error=report.get('error'),
            completion_scope='command_sequence_only',motion_verified=False,operator_observation_pending=True,
            nonzero_frames=report['nonzero_frames'],stop_feedback_observed=report['stop_feedback_observed'],
            nonzero_candidate_feedback_observed=report['nonzero_candidate_feedback_observed'],
            response_summary=report['response_summary']))
        except (BrokenPipeError,OSError):pass
        connection.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port');p.add_argument('--output')
    p.add_argument('--execute',action='store_true',help='Run one hardware probe; requires the supervised physical preparation')
    p.add_argument('--motion',action='store_true',help='Expect actual movement; otherwise any observed movement aborts the inhibited test')
    p.add_argument('--speed',type=float,help='Forward target cap in m/s, 0.001 to 0.10; default from PROFILE')
    p.add_argument('--pulse-s',type=float,help='Explicit test window in seconds, 0.1 to 3.0; default from PROFILE')
    args=p.parse_args(argv)
    try:profile=probe_profile(speed=args.speed,pulse_s=args.pulse_s)
    except ValueError as error:p.error(str(error))
    if args.motion and not args.execute:p.error('--motion requires --execute')
    if not args.execute:
        print(json.dumps(dict(scope='plan_only_no_device_opened',profile=profile,limits=LIMITS,
            physical_motor_inhibit_expected=True,feedback_protocol_verified=False),ensure_ascii=False));return
    if not args.port or not args.output:p.error('--execute requires --port and --output')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    context=mp.get_context('spawn');parent,child=context.Pipe()
    worker=context.Process(target=_serial_owner,args=(child,args.port,str(output),not args.motion,
        profile['pulse_s'],profile['linear_mps']),daemon=False)
    worker.start();child.close();cancelled=False;result=None
    try:
        while worker.is_alive():
            try:
                parent.send('cancel' if cancelled else ('heartbeat',time.monotonic()))
                if parent.poll(.05):
                    message=parent.recv()
                    if message.get('done'):result=message;break
                    print(json.dumps(message,ensure_ascii=False),flush=True)
            except KeyboardInterrupt:
                cancelled=True
                print('Cancellation requested; serial owner is sending zero commands. Keep the physical stop available.',flush=True)
            except (EOFError,OSError):break
    finally:
        parent.close();worker.join(timeout=3.)
        if worker.is_alive():
            print('SERIAL OWNER HAS NOT EXITED: press the physical stop. No software stop is being claimed.',flush=True)
            worker.terminate();worker.join(timeout=3.)
    if result is None:raise SystemExit('Probe did not return a result; keep physical motor inhibit engaged')
    print(json.dumps(dict(output=str(output),**result),ensure_ascii=False))
    if not result['completed'] or not result['stop_feedback_observed']:raise SystemExit(1)
    if args.motion and not result['nonzero_candidate_feedback_observed']:
        raise SystemExit('No nonzero candidate motion feedback was observed; do not automatically increase the profile')


if __name__=='__main__':main()
