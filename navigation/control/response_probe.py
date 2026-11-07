"""Fixed supervised low-speed response batch; default is a plan, never navigation."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from .ground_probe import probe_profile


STEPS=(('forward_001','forward',.01),('forward_003','forward',.03),('forward_006','forward',.06),
       ('reverse','reverse',None),('spin_left','spin_left',None),('spin_right','spin_right',None))


def selected_plan():
    return dict(scope='fixed_supervised_motor_response_batch',autonomous_navigation=False,
        execution_requested=False,nominal_total_command_window_s=12.,
        profiles=[dict(name=name,**probe_profile(maneuver=kind,speed=speed,pulse_s=2.)) for name,kind,speed in STEPS],
        sequencing='Fixed beforehand; no safety failure may advance to another stage. No measured response in a safely completed stage is retained as an observation, not a reason to choose a new speed.',
        expected_physical_preparation='Stationary, unoccupied wheelchair, neutral joystick, clear flat space in front/behind and around the full footprint, and immediate physical-stop supervision.')


def _run_segment(command,console):
    """The child retains its own deadline and heartbeat-based zero recovery."""
    with console.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        try:return process.wait(timeout=12.)
        except (KeyboardInterrupt,subprocess.TimeoutExpired):
            if process.poll() is None:process.send_signal(signal.SIGINT)
            try:process.wait(timeout=7.)
            except subprocess.TimeoutExpired:
                print('Child has not exited: operate the physical stop. No software stop is claimed.',flush=True)
                process.terminate()
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:pass
            raise


def run_batch(port,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=selected_plan();plan['execution_requested']=True
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    result=dict(scope=plan['scope'],actual_execution_requested=True,stages=[],
        all_command_sequences_completed=False,motion_verified=False,execution_feedback=False,
        vehicle_profile_updated=False,firmware_watchdog_verified=False,operator_observation_pending=True)
    try:
        for index,(name,kind,speed) in enumerate(STEPS,1):
            target=output/f'{index:02d}_{name}'
            command=[sys.executable,'-B','-m','navigation.control.ground_probe','--port',port,
                     '--output',str(target),'--execute','--motion','--maneuver',kind,'--pulse-s','2']
            if speed is not None:command+=['--speed',str(speed)]
            print(json.dumps(dict(stage=index,total=len(STEPS),name=name,profile=plan['profiles'][index-1]),ensure_ascii=False),flush=True)
            entry=dict(name=name,command=command,safe_to_continue=False,status='started')
            result['stages'].append(entry)
            started=time.monotonic();code=_run_segment(command,output/f'{index:02d}_{name}.console.log')
            source=json.loads((target/'result.json').read_text())
            expected=plan['profiles'][index-1]
            matches=all(source['profile'].get(k)==expected[k] for k in ('maneuver','linear_mps','angular_rps','pulse_s'))
            safe=(matches and code in (0,1) and source.get('completed') is True and source.get('stop_feedback_observed') is True and
                  not source.get('stop_confirmation_timed_out') and not any(source.get(k) for k in
                  ('error','restore_error','settings_restore_error','event_log_truncated','raw_log_truncated')))
            entry.update(exit_code=code,wall_s=time.monotonic()-started,status='finished',
                command_sequence_completed=source.get('completed'),stop_feedback_observed=source.get('stop_feedback_observed'),
                candidate_response_observed=source.get('nonzero_candidate_feedback_observed'),
                scope='unverified_candidate_feedback_not_physical_acceptance',error=source.get('error'),safe_to_continue=safe)
            if not safe:
                result['error']='Stage did not safely finish: '+name;break
        result['all_command_sequences_completed']=len(result['stages'])==len(STEPS) and all(x['safe_to_continue'] for x in result['stages'])
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        if isinstance(error,KeyboardInterrupt):result['cancelled']=True
    finally:
        (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Explicit supervised hardware batch; six fixed stages, each at most 2 seconds')
    parser.add_argument('--port');parser.add_argument('--output')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(selected_plan(),ensure_ascii=False));return 0
    if not args.port or not args.output:parser.error('--execute requires --port and --output')
    def cancel(*_):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,cancel)
    try:result=run_batch(args.port,args.output)
    finally:signal.signal(signal.SIGTERM,previous)
    print(json.dumps(dict(output=str(Path(args.output).resolve()),**result),ensure_ascii=False))
    return 0 if result['all_command_sequences_completed'] else 1


if __name__=='__main__':sys.exit(main())
