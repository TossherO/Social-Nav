"""One fixed measurement batch; plan only unless explicitly executed."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from navigation.control.ground_probe import probe_profile, validate_silence
from navigation.control.response_probe import _run_segment

ROOT=Path(__file__).resolve().parents[2]
PORT='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
STEPS=(('left_braking',dict(maneuver='spin_left',angular_speed=.2,pulse_s=2.),None),
       ('right_braking',dict(maneuver='spin_right',angular_speed=.2,pulse_s=2.),None),
       ('forward_silence',dict(maneuver='forward',speed=.1,pulse_s=1.5),1.5))


def selected_steps(names=None):
    if names is None:return STEPS
    if not names or len(names)!=len(set(names)) or any(name not in {row[0] for row in STEPS} for name in names):
        raise ValueError('Select distinct existing measurement steps')
    return tuple(row for row in STEPS if row[0] in names)


def plan(names=None):
    profiles=[]
    chosen=selected_steps(names)
    for name,request,silence in chosen:
        profile=probe_profile(**request);validate_silence(silence,profile)
        profiles.append(dict(name=name,profile=profile,silence_s=silence))
    return dict(scope='fixed_remaining_measurements_not_navigation',profiles=profiles,
        expected_elapsed_s=[30,45] if len(chosen)==3 else None,motor_command_limit='No reverse; spin <=0.20 rad/s, forward <=0.10 m/s; each stage returns to zero.',
        preparation='Stationary unoccupied wheelchair, joystick centered, electric mode available, clear flat front and full turning footprint, immediate physical-stop supervision.',
        calibration_acceptance=False)


def master_available():
    import rosgraph
    socket.setdefaulttimeout(1.)
    try:rosgraph.Master('/social_nav_measure_remaining').getPid();return True
    except (OSError,rosgraph.MasterException):return False


def stop_process(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=10.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3.)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3.)


def stage_ok(source,expected):
    if not source.get('completed') or not source.get('stop_feedback_observed') or not source.get('nonzero_candidate_feedback_observed'):
        return False
    if any(source.get(k) for k in ('error','restore_error','settings_restore_error','stop_confirmation_timed_out','event_log_truncated','raw_log_truncated')):
        return False
    if any(source['profile'].get(k)!=expected['profile'][k] for k in ('maneuver','linear_mps','angular_rps','pulse_s')):
        return False
    actual=source.get('silence_test')
    return actual is None if expected['silence_s'] is None else actual is not None and actual.get('requested_s')==expected['silence_s']


def observer_ready(output,process):
    if process.poll() is not None or (output/'observer_error.json').exists() or (output/'observer_summary.json').exists():return False
    try:return time.monotonic()-json.loads((output/'ready.json').read_text())['completed_mono']<.6
    except (OSError,ValueError,KeyError):return False


def execute(names=None):
    chosen=selected_steps(names);selected=plan(names)
    output=ROOT/'outputs/navigation/remaining_measurements'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    print('本轮记录：'+str(output),flush=True)
    sensors=observer=None
    report=dict(scope=selected['scope'],stages=[],vehicle_profile_updated=False,calibration_accepted=False,
                all_selected_sequences_completed=False,operator_observation_pending=True)
    try:
        with (output/'sensors.console.log').open('x') as sensor_log,(output/'observer.console.log').open('x') as observer_log:
            if not master_available():
                sensors=subprocess.Popen(['bash','perception/capture/start_sensors.sh'],cwd=ROOT,
                    stdout=sensor_log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+15.
                while not master_available():
                    if sensors.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Sensor master did not start')
                    time.sleep(.1)
            observer=subprocess.Popen([sys.executable,'-B','navigation/commissioning/measure_observer.py','--output',str(output)],
                cwd=ROOT,stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+15.
            while True:
                if observer_ready(output,observer):break
                if observer.poll() is not None or (output/'observer_error.json').exists() or time.monotonic()>=deadline:raise RuntimeError('Independent LIO not ready; no motion sent')
                time.sleep(.1)
            for index,((name,request,silence),expected) in enumerate(zip(chosen,selected['profiles']),1):
                if not observer_ready(output,observer):
                    raise RuntimeError('Independent observer unavailable; no next maneuver')
                target=output/f'{index:02d}_{name}'
                command=[sys.executable,'-B','-m','navigation.control.ground_probe','--port',PORT,'--output',str(target),
                    '--execute','--motion','--maneuver',request['maneuver'],'--pulse-s',str(request['pulse_s'])]
                if 'speed' in request:command+=['--speed',str(request['speed'])]
                if 'angular_speed' in request:command+=['--angular-speed',str(request['angular_speed'])]
                if silence is not None:command+=['--silence-s',str(silence)]
                entry=dict(name=name,status='started');report['stages'].append(entry)
                print(f'阶段 {index}/{len(chosen)}：{name}',flush=True)
                code=_run_segment(command,output/f'{index:02d}_{name}.console.log')
                source=json.loads((target/'result.json').read_text())
                ok=code in (0,1) and stage_ok(source,expected)
                entry.update(status='finished',exit_code=code,sequence_completed=source['completed'],
                    stopped=source['stop_feedback_observed'],safe_to_continue=ok,error=source.get('error'))
                if not ok:raise RuntimeError('Stage stopped; no later maneuver: '+name)
                entry['independent_observer_current_at_end']=observer_ready(output,observer)
                if not entry['independent_observer_current_at_end']:
                    raise RuntimeError('Independent observer lost during stage; no later maneuver')
            report['all_selected_sequences_completed']=True
            time.sleep(1.)
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        (output/'STOP').write_text('Measurement sequence ended; no chassis commands from observer.\n')
        for name,process in (('observer',observer),('owned_sensors',sensors)):
            try:stop_process(process)
            except Exception as error:report[name+'_cleanup_error']=str(error)
        (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),**report),ensure_ascii=False),flush=True)
    return 0 if report['all_selected_sequences_completed'] and not any(k.endswith('_cleanup_error') for k in report) else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--steps',nargs='+',choices=[row[0] for row in STEPS],help='Only remaining steps, in the original fixed order')
    args=parser.parse_args(argv)
    if not args.execute:print(json.dumps(plan(args.steps),ensure_ascii=False,indent=2));return 0
    def cancel(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,cancel)
    return execute(args.steps)


if __name__=='__main__':raise SystemExit(main())
