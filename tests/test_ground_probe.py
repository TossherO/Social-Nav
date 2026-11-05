"""Bounded probe guards and a simulated serial controller; no physical device."""
import json
import contextlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import pty
import select
import struct
import tempfile
import threading
import time
import tty
import unittest
from unittest.mock import patch

from navigation.control.chassis_protocol import decode_feedback
from navigation.control.ground_probe import PROFILE,PulseWindow,ReceiptWindow,_serial_owner,probe_profile,main,summarize_motion


def frame(mode=1,velocity=0,angular=0):
    payload=struct.pack('>2h2i2HBHB2h2hB',int(velocity*1000),int(angular*1000),0,0,0,80,0,0,mode,0,0,0,0,0)
    return b'\xbf\x01'+payload+bytes([sum(payload)&255])


class PulseGuardTests(unittest.TestCase):
    def ask(self,window,now=10.1,**changes):
        values=dict(heartbeat_at=now,feedback_at=now,feedback=decode_feedback(frame()))
        values.update(changes)
        return window.command(now,**values)

    def test_payload_candidate_checksum_does_not_upgrade_protocol_validation(self):
        row=decode_feedback(frame());self.assertTrue(row['candidate_checksum_matches'])
        self.assertFalse(row['checksum_verified']);self.assertFalse(row['execution_feedback'])
        corrupt=bytearray(frame());corrupt[6]^=1
        self.assertFalse(decode_feedback(corrupt)['candidate_checksum_matches'])

    def test_stage_summary_uses_original_receipt_times_and_does_not_claim_execution(self):
        events=[dict(kind='phase',name='single_forward_pulse',mono=10.),
            dict(kind='feedback',mono=10.2,feedback=dict(velocity_mps_rps=[.02,0.])),
            dict(kind='feedback',mono=11.3,feedback=dict(velocity_mps_rps=[.08,.01])),
            dict(kind='phase',name='restore_zero_raw_mode_3',mono=12.),
            dict(kind='feedback',mono=12.1,feedback=dict(velocity_mps_rps=[0.,0.]))]
        result=summarize_motion(events,PROFILE)
        self.assertAlmostEqual(result['first_nonzero_candidate_delay_s'],.2)
        self.assertEqual(result['clock_basis'],'host_monotonic_receipt_not_measurement')
        self.assertEqual([result['stages'][s]['feedback_frames'] for s in ('startup','steady','stopping')],[1,1,1])
        self.assertEqual(result['stages']['steady']['max_abs_candidate_velocity'],[.08,.01])

    def test_profile_ramp_and_deadline_cannot_be_extended_by_fresh_inputs(self):
        window=PulseWindow(10.)
        self.assertEqual(self.ask(window,10.),0.)
        self.assertAlmostEqual(self.ask(window,10.05),.012)
        self.assertEqual(self.ask(window,10.2),.03)
        self.assertEqual(self.ask(window,10.3),0.)
        self.assertEqual(self.ask(window,11.),0.)

    def test_explicit_longer_window_has_its_own_fixed_deadline_and_same_speed_cap(self):
        window=PulseWindow(10.,duration=1.)
        self.assertEqual(self.ask(window,10.2),.03)
        self.assertEqual(self.ask(window,10.8),.03)
        self.assertEqual(self.ask(window,11.),0.)
        self.assertEqual(self.ask(window,12.),0.)
        for duration in (0.,.09,3.01,-1.,True,float('nan')):
            with self.assertRaises(ValueError):PulseWindow(10.,duration=duration)

    def test_explicit_speed_and_duration_do_not_change_defaults_or_feedback_limits(self):
        profile=probe_profile(speed=.08,pulse_s=2.)
        self.assertEqual(profile['linear_mps'],.08);self.assertEqual(profile['pulse_s'],2.)
        self.assertEqual(profile['max_observed_linear_mps'],PROFILE['max_observed_linear_mps'])
        self.assertEqual(probe_profile()['linear_mps'],.03)
        window=PulseWindow(10.,duration=2.,speed=.08)
        self.assertAlmostEqual(self.ask(window,10.1),.024)
        self.assertEqual(self.ask(window,10.5),.08)
        self.assertEqual(self.ask(window,12.),0.)
        self.assertEqual(self.ask(window,12.5),0.)

    def test_invalid_requested_limits_are_rejected_before_resource_creation(self):
        for speed in (0.,-.01,.1001,float('nan'),float('inf'),True):
            with self.subTest(speed=speed),self.assertRaises(ValueError):probe_profile(speed=speed)
        for duration in (0.,-.1,3.01,float('nan'),True):
            with self.subTest(duration=duration),self.assertRaises(ValueError):probe_profile(pulse_s=duration)
        with tempfile.TemporaryDirectory() as folder,patch('multiprocessing.get_context') as spawn:
            output=Path(folder)/'must-not-exist'
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                main(['--speed','.2','--execute','--port','/not/a/device','--output',str(output)])
            self.assertEqual(error.exception.code,2);self.assertFalse(output.exists());spawn.assert_not_called()

    def test_selected_plan_never_starts_a_serial_owner(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output),patch('multiprocessing.get_context') as spawn:
            main(['--speed','.05','--pulse-s','2'])
        record=json.loads(output.getvalue());self.assertEqual(record['scope'],'plan_only_no_device_opened')
        self.assertEqual(record['profile']['linear_mps'],.05);self.assertEqual(record['profile']['pulse_s'],2.)
        spawn.assert_not_called()

    def test_startup_turn_does_not_disable_hard_limits(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=3.)
        turning=decode_feedback(frame(angular=.058))
        for i in range(1,20):self.assertGreater(self.ask(window,10+i*.05,feedback=turning),0.)
        self.assertEqual(window.stage,'startup');self.assertIsNone(window.reason)
        for i in range(20,30):self.assertGreater(self.ask(window,10+i*.05),0.)
        self.assertEqual(window.stage,'steady')
        hard=PulseWindow(10.,expect_inhibited=False)
        self.assertEqual(self.ask(hard,feedback=decode_feedback(frame(angular=.16))),0.)
        self.assertEqual(hard.reason,'unexpected_velocity')

    def test_sustained_turn_after_startup_stops_and_remains_latched(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=3.)
        turning=decode_feedback(frame(angular=.058))
        for i in range(1,28):self.assertGreater(self.ask(window,10+i*.05,feedback=turning),0.)
        self.assertEqual(self.ask(window,11.4,feedback=turning),0.)
        self.assertEqual(window.reason,'persistent_unexpected_turn')
        self.assertEqual(self.ask(window,11.45),0.)

    def test_receipt_window_rejects_duplicates_bursts_and_long_gaps(self):
        window=ReceiptWindow(.4,5,.25)
        for t in (10.,10.,10.001,10.002,10.003,10.004):window.push(t,[.03,.06])
        self.assertIsNone(window.average())
        window.push(10.4,[.03,.06]);self.assertIsNone(window.average())
        for i in range(1,9):window.push(10.4+i*.05,[.03,.06])
        self.assertAlmostEqual(window.average()['mean_angular_rps'],.06)

    def test_alternating_turn_and_short_linear_spike_differ_from_sustained_overspeed(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=3.,speed=.1)
        for i in range(1,45):
            angular=.058 if i%2 else -.058
            v=.12 if i==21 else .08
            self.assertGreater(self.ask(window,10+i*.05,feedback=decode_feedback(frame(velocity=v,angular=angular))),0.)
        self.assertIsNone(window.reason)
        window=PulseWindow(10.,expect_inhibited=False,duration=3.,speed=.1)
        for i in range(1,9):self.assertGreater(self.ask(window,10+i*.05,feedback=decode_feedback(frame(velocity=.12))),0.)
        self.assertEqual(self.ask(window,10.45,feedback=decode_feedback(frame(velocity=.12))),0.)
        self.assertEqual(window.reason,'sustained_linear_overspeed')
        hard=PulseWindow(10.,expect_inhibited=False)
        self.assertEqual(self.ask(hard,feedback=decode_feedback(frame(velocity=.16))),0.)
        self.assertEqual(hard.reason,'unexpected_velocity')

    def test_stale_or_disconnected_inputs_latch_zero(self):
        for changes,reason in ((dict(heartbeat_at=9.),'parent_heartbeat_lost'),
                (dict(parent_alive=False),'parent_heartbeat_lost'),(dict(feedback_at=9.),'serial_feedback_stale'),
                (dict(feedback_at=11.),'serial_feedback_stale')):
            window=PulseWindow(10.)
            with self.subTest(reason=reason):
                self.assertEqual(self.ask(window,**changes),0.);self.assertEqual(window.reason,reason)
                self.assertEqual(self.ask(window,10.2),0.)

    def test_corruption_takeover_and_mode_changes_abort_without_recovery(self):
        for fields,reason in ((dict(candidate_checksum_matches=False),'candidate_checksum_mismatch'),
                (dict(mode_raw=3),'raw_mode_changed'),(dict(joystick_axes_raw=[1,0]),'joystick_takeover'),
                (dict(velocity_mps_rps=[.2,0]),'unexpected_velocity')):
            window=PulseWindow(10.);row=decode_feedback(frame());row.update(fields)
            with self.subTest(reason=reason):
                self.assertEqual(self.ask(window,feedback=row),0.);self.assertEqual(window.reason,reason)
                self.assertEqual(self.ask(window,10.2),0.)

    def test_inhibited_motion_or_wrong_direction_is_not_accepted(self):
        window=PulseWindow(10.,expect_inhibited=True)
        self.assertEqual(self.ask(window,feedback=decode_feedback(frame(velocity=.02))),0.)
        self.assertEqual(window.reason,'motion_while_inhibited')
        window=PulseWindow(10.,expect_inhibited=False)
        self.assertEqual(self.ask(window,feedback=decode_feedback(frame(velocity=-.02))),0.)
        self.assertEqual(window.reason,'unexpected_motion_direction')

    def test_reversed_or_nonfinite_clock_cannot_produce_motion(self):
        window=PulseWindow(10.);self.ask(window,10.1)
        self.assertEqual(self.ask(window,10.05),0.)
        self.assertEqual(window.reason,'monotonic_clock_invalid')
        self.assertEqual(self.ask(PulseWindow(10.),float('nan')),0.)


class SerialOwnerTests(unittest.TestCase):
    def exercise(self,disconnect_on_motion=False,simulate_motion=False,pulse_s=.3,speed=.03,stop_jitter_s=0.):
        master,slave=pty.openpty();tty.setraw(slave);port=os.ttyname(slave)
        os.set_blocking(master,False);halt=threading.Event();moving=threading.Event();writes=[]
        def controller():
            pending=bytearray();mode=3;velocity=0.;stopped_at=None;next_emit=time.monotonic()
            while not halt.is_set():
                if select.select([master],[],[],.005)[0]:
                    try:data=os.read(master,4096)
                    except OSError:data=b''
                    pending.extend(data)
                    while len(pending)>=12:
                        start=pending.find(b'\xaf\x01')
                        if start<0:pending.clear();break
                        del pending[:start]
                        if len(pending)<12:break
                        packet=bytes(pending[:12]);del pending[:12]
                        before_mode=mode
                        _,_,linear,angular,mode,limit,error,led,checksum=struct.unpack('>2B2hBhBBB',packet)
                        if checksum!=(0xAF+1+linear+angular+mode+limit+error)&255:continue
                        writes.append((time.monotonic(),linear,angular,mode))
                        velocity=linear/1000. if simulate_motion else 0.
                        if linear:moving.set()
                        if before_mode==1 and mode==3 and moving.is_set():stopped_at=time.monotonic()
                if time.monotonic()>=next_emit:
                    jitter=stopped_at is not None and time.monotonic()-stopped_at<stop_jitter_s
                    try:os.write(master,frame(mode=mode,velocity=.018 if jitter else velocity))
                    except OSError:pass
                    next_emit=time.monotonic()+.025
        emulator=threading.Thread(target=controller);emulator.start()
        context=mp.get_context('spawn');parent,child=context.Pipe()
        worker=None
        try:
            with tempfile.TemporaryDirectory() as directory:
                worker=context.Process(target=_serial_owner,args=(child,port,directory,not simulate_motion,pulse_s,speed));worker.start();child.close()
                os.close(slave);slave=None
                began=time.monotonic();result=None;disconnected=False
                while worker.is_alive() and time.monotonic()-began<10:
                    if disconnect_on_motion and moving.is_set() and not disconnected:
                        parent.close();disconnected=True
                    if not disconnected:
                        parent.send(('heartbeat',time.monotonic()))
                        if parent.poll(.02):
                            message=parent.recv()
                            if message.get('done'):result=message;break
                    else:time.sleep(.02)
                worker.join(timeout=3.)
                self.assertFalse(worker.is_alive(),'Serial owner did not finish its bounded test')
                self.assertEqual(worker.exitcode,0)
                report=json.loads((Path(directory)/'result.json').read_text())
                events=[json.loads(s) for s in (Path(directory)/'events.jsonl').read_text().splitlines()]
                self.assertTrue(report['stop_feedback_observed'],report)
                self.assertFalse(report['motion_verified']);self.assertFalse(report['execution_feedback'])
                self.assertEqual(report['nonzero_candidate_feedback_observed'],simulate_motion)
                self.assertTrue(moving.is_set(),'Simulated serial controller received no nonzero test frame')
                self.assertEqual(writes[-1][1:],(0,0,3))
                self.assertLessEqual(max(abs(r[1]) for r in writes),int(speed*1000))
                self.assertEqual(report['profile']['linear_mps'],speed)
                self.assertEqual(report['profile']['pulse_s'],pulse_s)
                if stop_jitter_s:self.assertGreater(report['stop_confirmation_elapsed_s'],1.1)
                self.assertFalse(report['stop_confirmation_timed_out'])
                self.assertIsNotNone(report['response_summary'])
                self.assertTrue(all(r[2]==0 for r in writes))
                self.assertEqual(report['nonzero_frames'],sum(r[1]!=0 for r in writes))
                for row in events:
                    if row['kind']=='write':
                        wire=int.from_bytes(bytes.fromhex(row['frame_hex'])[2:4],byteorder='big',signed=True)/1000.
                        self.assertEqual(row['linear_mps'],wire)
                if disconnect_on_motion:
                    self.assertFalse(report['completed']);self.assertIn('Parent',report['error'])
                    first_motion=next(r[0] for r in writes if r[1])
                    after=next(r[0] for r in writes if r[0]>first_motion and r[1]==0)
                    self.assertLess(after-first_motion,.5)
                else:
                    self.assertTrue(report['completed'],report)
                    start=next(r['mono'] for r in events if r['kind']=='phase' and r['name']=='single_forward_pulse')
                    self.assertTrue(all(r['mono']-start<pulse_s+.03 for r in events if r['kind']=='write' and r['linear_mps']))
        finally:
            if worker is not None and worker.is_alive():worker.kill();worker.join()
            parent.close();halt.set();emulator.join(timeout=1);os.close(master)
            if slave is not None:os.close(slave)

    def test_fixed_pulse_restores_zero_and_original_mode(self):self.exercise()
    def test_parent_disconnect_during_pulse_stops_without_parent_help(self):self.exercise(disconnect_on_motion=True)
    def test_positive_motion_feedback_followed_by_zero_is_recorded_without_claiming_verification(self):self.exercise(simulate_motion=True)
    def test_explicit_longer_window_restores_zero_without_raising_speed(self):self.exercise(simulate_motion=True,pulse_s=1.)
    def test_expanded_request_uses_selected_speed_and_still_restores_zero(self):self.exercise(simulate_motion=True,pulse_s=2.,speed=.08)
    def test_expanded_request_stops_on_parent_disconnect_before_its_deadline(self):self.exercise(disconnect_on_motion=True,pulse_s=2.,speed=.08)
    def test_stop_confirmation_waits_for_stable_feedback_after_jitter(self):self.exercise(simulate_motion=True,stop_jitter_s=1.25)


if __name__=='__main__':unittest.main()
