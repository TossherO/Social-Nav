"""Single video-synchronized silence observation; no action without --execute."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from navigation.control.ground_probe import probe_profile, validate_silence
from navigation.control.response_probe import _run_segment

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'navigation/commissioning/measure_remaining.py'
spec=importlib.util.spec_from_file_location('local_measure_remaining',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
PORT=helper.PORT
SPEED=.10
PULSE_S=1.5
SILENCE_S=1.5
SOURCES=(Path(__file__).resolve(),ROOT/'navigation/commissioning/video_sync_marker.py',
    ROOT/'navigation/commissioning/video_alignment.py',HELPER,
    ROOT/'navigation/commissioning/measure_observer.py',
    ROOT/'navigation/control/ground_probe.py',ROOT/'navigation/control/response_probe.py',
    ROOT/'navigation/control/chassis_protocol.py')


def make_plan():
    profile=probe_profile(maneuver='forward',speed=SPEED,pulse_s=PULSE_S)
    validate_silence(SILENCE_S,profile)
    return dict(scope='single_supervised_video_silence_not_navigation',
        execution_requested=False,profile=profile,silence_s=SILENCE_S,
        marker=dict(period_s=.5,clock='host_monotonic_gui_request_not_screen_emission',
            physical_screen_latency_unmeasured=True),
        video='Fixed camera sees wheel, ground reference and numbered screen marker; start recording before execution.',
        preparation='Stationary unoccupied wheelchair, neutral joystick, electric drive available, clear flat forward path and immediate physical-stop supervision.',
        firmware_watchdog_verified=False,vehicle_profile_updated=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES})


def marker_ready(ready,process,*,minimum_id=0):
    if process is None or process.poll() is not None:return False
    try:
        row=json.loads(ready.read_text())
        age=time.monotonic()-row['draw_queued_mono']
        return row['marker_id']>=minimum_id and 0<=age<=1.5
    except (OSError,ValueError,KeyError,TypeError):return False


def marker_spans_motion(directory,probe):
    try:
        rows=[json.loads(x) for x in (directory/'marker_events.jsonl').read_text().splitlines()]
        events=[json.loads(x) for x in (probe/'events.jsonl').read_text().splitlines()]
        nonzero=[e for e in events if e['kind']=='write' and (e['linear_mps'] or e['angular_rps'])]
        first=nonzero[0]['mono'];last=nonzero[-1]['mono']
        zero=next(e['mono'] for e in events if e['kind']=='write' and e['mono']>last and not e['linear_mps'] and not e['angular_rps'])
        stamps=[r['draw_queued_mono'] for r in rows]
        return len(rows)>=4 and stamps[0]<first and stamps[-1]>zero
    except (OSError,ValueError,KeyError,IndexError,StopIteration,TypeError):return False


def execute(*, output_parent=None):
    selected=make_plan()
    parent=output_parent or ROOT/'outputs/navigation/video_silence'
    output=parent/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    selected['execution_requested']=True
    (output/'plan.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'video_marks_template.csv').write_text('marker_id,video_time_before_s,video_time_after_s\n')
    source=output/'source';source.mkdir()
    for path in SOURCES:shutil.copyfile(path,source/path.name)
    print('本轮记录：'+str(output),flush=True)
    marker=observer=sensors=None
    summary=dict(scope=selected['scope'],execution_requested=True,probe_started=False,
        command_sequence_completed=False,stop_feedback_observed=False,marker_spans_motion=False,
        independent_observer_valid_at_end=False,video_recording_not_verified=True,
        firmware_watchdog_verified=False,vehicle_profile_updated=False,operator_observation_pending=True)
    try:
        with (output/'marker.console.log').open('x') as marker_log, \
             (output/'observer.console.log').open('x') as observer_log, \
             (output/'sensors.console.log').open('x') as sensor_log:
            marker=subprocess.Popen([sys.executable,'-B',str(ROOT/'navigation/commissioning/video_sync_marker.py'),
                '--events',str(output/'marker_events.jsonl'),'--ready',str(output/'marker_ready.json'),
                '--stop-file',str(output/'MARKER_STOP')],cwd=ROOT,stdout=marker_log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+8.
            while not marker_ready(output/'marker_ready.json',marker,minimum_id=2):
                if marker.poll() is not None or time.monotonic()>=deadline:
                    raise RuntimeError('Video marker unavailable; no motion sent')
                time.sleep(.1)
            if not helper.master_available():
                sensors=subprocess.Popen(['bash','perception/capture/start_sensors.sh'],cwd=ROOT,
                    stdout=sensor_log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+15.
                while not helper.master_available():
                    if sensors.poll() is not None or time.monotonic()>=deadline:raise RuntimeError('Sensor master did not start')
                    time.sleep(.1)
            observer=subprocess.Popen([sys.executable,'-B',str(ROOT/'navigation/commissioning/measure_observer.py'),
                '--output',str(output)],cwd=ROOT,stdout=observer_log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+15.
            while not helper.observer_ready(output,observer):
                if observer.poll() is not None or (output/'observer_error.json').exists() or time.monotonic()>=deadline:
                    raise RuntimeError('Independent LIO not ready; no motion sent')
                time.sleep(.1)
            if not marker_ready(output/'marker_ready.json',marker,minimum_id=2):
                raise RuntimeError('Video marker stopped before probe; no motion sent')
            probe=output/'probe'
            command=[sys.executable,'-B','-m','navigation.control.ground_probe','--port',PORT,
                '--output',str(probe),'--execute','--motion','--maneuver','forward',
                '--speed',str(SPEED),'--pulse-s',str(PULSE_S),'--silence-s',str(SILENCE_S)]
            print('开始单段前进与断发；请保持镜头同时拍到车轮、地面参照和编号屏幕。',flush=True)
            summary['probe_started']=True
            code=_run_segment(command,output/'probe.console.log')
            raw=json.loads((probe/'result.json').read_text())
            expected=dict(profile=selected['profile'],silence_s=SILENCE_S)
            summary.update(probe_exit_code=code,command_sequence_completed=raw.get('completed',False),
                stop_feedback_observed=raw.get('stop_feedback_observed',False),
                probe_checks_passed=(code in (0,1) and helper.stage_ok(raw,expected)),
                probe_error=raw.get('error'),
                independent_observer_valid_at_end=helper.observer_ready(output,observer),
                marker_alive_at_end=marker_ready(output/'marker_ready.json',marker,minimum_id=2))
            # Preserve transitions following active-zero recovery for the video.
            time.sleep(1.)
    except BaseException as error:
        summary['error']=type(error).__name__+': '+str(error)
        if isinstance(error,KeyboardInterrupt):summary['cancelled']=True
    finally:
        (output/'STOP').write_text('Stop sensor-only observer after probe.\n')
        for name,process in (('observer',observer),('owned_sensors',sensors)):
            try:helper.stop_process(process)
            except Exception as error:summary[name+'_cleanup_error']=str(error)
        (output/'MARKER_STOP').write_text('Stop display marker.\n')
        if marker is not None:
            try:marker.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                try:os.killpg(marker.pid,signal.SIGTERM);marker.wait(timeout=3.)
                except (OSError,subprocess.TimeoutExpired) as error:summary['marker_cleanup_error']=str(error)
        summary['marker_spans_motion']=marker_spans_motion(output,output/'probe')
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),**summary),ensure_ascii=False),flush=True)
    return 0 if (summary.get('probe_checks_passed') and summary['independent_observer_valid_at_end'] and
        summary.get('marker_alive_at_end') and summary['marker_spans_motion'] and
        not any(k.endswith('_cleanup_error') for k in summary)) else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Run the single supervised physical test after camera and site preparation')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(make_plan(),ensure_ascii=False,indent=2,allow_nan=False));return 0
    def cancel(*_):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,cancel)
    try:return execute()
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':raise SystemExit(main())
