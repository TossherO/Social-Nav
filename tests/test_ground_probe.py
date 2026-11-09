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
from navigation.control.ground_probe import PROFILE,PulseWindow,ReceiptWindow,_serial_owner,probe_profile,main,summarize_motion,validate_silence
from tests.chassis_fixtures import frame


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

    def test_silence_plan_and_invalid_requests_never_open_resources(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output),patch('multiprocessing.get_context') as spawn:
            main(['--speed','.1','--pulse-s','1.5','--silence-s','.8'])
        record=json.loads(output.getvalue());self.assertTrue(record['silence_test']['no_control_frames_during_observation'])
        self.assertEqual(record['silence_test']['duration_s'],.8);spawn.assert_not_called()
        for value in (True,0.,.19,1.51,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_silence(value,probe_profile(pulse_s=1.5))
        for pulse,gap in ((.3,.8),(2.5,.8)):
            with self.assertRaises(ValueError):validate_silence(gap,probe_profile(pulse_s=pulse))
        with tempfile.TemporaryDirectory() as folder,patch('multiprocessing.get_context') as spawn:
            dest=Path(folder)/'not_created'
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(['--execute','--speed','.1','--pulse-s','1.5','--silence-s','.8','--port','/not/a/device','--output',str(dest)])
            self.assertFalse(dest.exists());spawn.assert_not_called()

    def test_longer_silence_remains_within_the_same_total_window(self):
        validate_silence(1.5,probe_profile(speed=.1,pulse_s=1.5))
        with self.assertRaises(ValueError):validate_silence(1.5,probe_profile(speed=.1,pulse_s=1.6))

    def test_silence_small_rebound_is_retained_but_sustained_or_large_reverse_stops(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=1.5,silence=True)
        self.ask(window,10.1,feedback=decode_feedback(frame(velocity=-.018,angular=.06)))
        self.assertIsNone(window.reason);self.assertEqual(window.silence_reverse_total_frames,1)
        self.ask(window,10.15,feedback=decode_feedback(frame()))
        self.assertIsNone(window.reason);self.assertEqual(window.silence_reverse_frames,0)
        for i in range(5):self.ask(window,10.2+i*.05,feedback=decode_feedback(frame(velocity=-.018)))
        self.assertEqual(window.reason,'unexpected_motion_direction')
        hard=PulseWindow(10.,expect_inhibited=False,duration=1.5,silence=True)
        self.ask(hard,10.1,feedback=decode_feedback(frame(velocity=-.04)))
        self.assertEqual(hard.reason,'unexpected_motion_direction')
        forward=PulseWindow(10.,expect_inhibited=False,duration=1.5)
        self.ask(forward,10.1,feedback=decode_feedback(frame(velocity=-.018)))
        self.assertEqual(forward.reason,'unexpected_motion_direction')

    def test_silence_guard_never_returns_nonzero_and_allows_firmware_mode_change(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=.8,speed=.1,silence=True)
        self.assertEqual(self.ask(window,10.1,feedback=decode_feedback(frame(mode=3))),0.)
        self.assertIsNone(window.reason)
        self.assertEqual(self.ask(window,10.2,feedback=decode_feedback(frame(velocity=.05))),0.)
        row=decode_feedback(frame());row['joystick_axes_raw']=[1,0]
        self.assertEqual(self.ask(window,10.3,feedback=row),0.);self.assertEqual(window.reason,'joystick_takeover')
        stale=PulseWindow(10.,expect_inhibited=False,duration=.8,silence=True)
        self.assertEqual(self.ask(stale,10.2,heartbeat_at=9.),0.)
        self.assertEqual(stale.reason,'parent_heartbeat_lost')

    def test_fixed_reverse_and_spins_have_separate_units_caps_and_directions(self):
        reverse=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='reverse')
        self.assertEqual(self.ask(reverse,10.5,feedback=decode_feedback(frame(velocity=-.04))),-.05)
        self.assertEqual(reverse.angular_command(10.5),0.)
        self.assertEqual(self.ask(reverse,12.),0.)
        for kind,direction in [('spin_left',1),('spin_right',-1)]:
            window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver=kind)
            self.assertEqual(self.ask(window,10.5,feedback=decode_feedback(frame(angular=direction*.1))),0.)
            self.assertIsNone(window.reason)
            self.assertEqual(window.angular_command(10.5),direction*.1)
            self.assertEqual(window.angular_command(12.),0.)
            self.assertEqual(self.ask(window,10.6,heartbeat_at=9.),0.)
            self.assertEqual(window.angular_command(10.6),0.)

    def test_spin_rejects_translation_and_cannot_be_silence_mode(self):
        window=PulseWindow(10.,expect_inhibited=False,maneuver='spin_left')
        self.assertEqual(self.ask(window,10.1,feedback=decode_feedback(frame(velocity=.061,angular=.1))),0.)
        self.assertEqual(window.reason,'unexpected_translation_during_spin');self.assertEqual(window.angular_command(10.1),0.)
        with self.assertRaises(ValueError):PulseWindow(10.,silence=True,maneuver='reverse')
        for kind in ('reverse','spin_left','spin_right'):
            with self.assertRaises(ValueError):probe_profile(maneuver=kind,speed=.1)

    def test_spin_angle_request_has_its_own_bounds_and_does_not_change_straight_limits(self):
        profile=probe_profile(maneuver='spin_left',angular_speed=.2,pulse_s=2.)
        self.assertEqual(profile['angular_rps'],.2)
        self.assertEqual(profile['max_observed_angular_rps'],.3)
        self.assertEqual(profile['linear_mps'],0.)
        self.assertEqual(probe_profile()['max_observed_angular_rps'],.15)
        self.assertEqual(probe_profile(maneuver='spin_left')['angular_rps'],.1)
        for value in (0.,.049,.301,float('nan'),float('inf'),True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                probe_profile(maneuver='spin_left',angular_speed=value)
        for kind in ('forward','reverse'):
            with self.assertRaises(ValueError):probe_profile(maneuver=kind,angular_speed=.2)
        with patch('multiprocessing.get_context') as spawn,contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--maneuver','spin_right','--angular-speed','.2','--pulse-s','2'])
            spawn.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['profile']['angular_rps'],.2)
        with patch('multiprocessing.get_context') as spawn,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--execute','--motion','--maneuver','spin_left','--angular-speed','.31',
                  '--port','/not/a/device','--output','/must/not/be/created'])
        spawn.assert_not_called()

    def test_spin_short_reverse_sample_is_recorded_but_does_not_abort(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_left',angular_speed=.2)
        self.ask(window,10.417,feedback=decode_feedback(frame(velocity=.018,angular=-.06)))
        self.assertIsNone(window.reason);self.assertEqual(window.spin_reverse_frames,1)
        self.assertGreater(window.angular_command(10.417),0.)
        self.ask(window,10.467,feedback=decode_feedback(frame(velocity=.018,angular=.058)))
        self.assertIsNone(window.reason);self.assertEqual(window.spin_reverse_frames,0)
        self.assertEqual(window.angular_command(12.),0.)

    def test_spin_single_small_translation_does_not_abort_but_recovery_resets_it(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_right',angular_speed=.2)
        self.ask(window,10.1,feedback=decode_feedback(frame(velocity=-.036,angular=-.116)))
        self.assertIsNone(window.reason);self.assertEqual(window.spin_translation_frames,1)
        self.ask(window,10.15,feedback=decode_feedback(frame(velocity=0.,angular=-.116)))
        self.assertIsNone(window.reason);self.assertEqual(window.spin_translation_frames,0)

    def test_spin_sustained_small_translation_still_aborts(self):
        for sign in (-1,1):
            window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_left',angular_speed=.2)
            for now in (10.1,10.15,10.2,10.25):
                self.ask(window,now,feedback=decode_feedback(frame(velocity=sign*.04,angular=.1)))
                self.assertIsNone(window.reason)
            self.ask(window,10.3,feedback=decode_feedback(frame(velocity=sign*.04,angular=.1)))
            self.assertEqual(window.reason,'unexpected_translation_during_spin')
            self.assertEqual(window.angular_command(10.3),0.)

    def test_spin_translation_requires_fresh_spanning_frames_and_resets_on_gap(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_left',angular_speed=.2)
        moving=decode_feedback(frame(velocity=.04,angular=.1))
        for now in (10.1,10.15,10.2,10.25,10.3):self.ask(window,now,feedback_at=10.1,feedback=moving)
        self.assertIsNone(window.reason);self.assertEqual(window.spin_translation_frames,1)
        self.ask(window,10.4,feedback=moving)
        self.assertIsNone(window.reason);self.assertEqual(window.spin_translation_frames,1)
        for now in (10.45,10.5,10.55,10.6):self.ask(window,now,feedback=moving)
        self.assertEqual(window.reason,'unexpected_translation_during_spin')

    def test_spin_sustained_reverse_still_aborts_both_directions(self):
        for kind,sign in [('spin_left',1),('spin_right',-1)]:
            window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver=kind,angular_speed=.2)
            for i in range(4):
                self.ask(window,10.1+i*.05,feedback=decode_feedback(frame(angular=-sign*.06)))
                self.assertIsNone(window.reason)
            self.ask(window,10.3,feedback=decode_feedback(frame(angular=-sign*.06)))
            self.assertEqual(window.reason,'unexpected_spin_direction')
            self.assertEqual(window.angular_command(10.3),0.)
            self.ask(window,10.35,feedback=decode_feedback(frame(angular=sign*.2)))
            self.assertEqual(window.reason,'unexpected_spin_direction')

    def test_spin_reverse_check_needs_fresh_spanning_feedback_and_resets_on_a_gap(self):
        window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_left',angular_speed=.2)
        reverse=decode_feedback(frame(angular=-.06))
        for now in (10.1,10.15,10.2,10.25,10.3):
            self.ask(window,now,feedback_at=10.1,feedback=reverse)
        self.assertIsNone(window.reason);self.assertEqual(window.spin_reverse_frames,1)
        self.ask(window,10.4,feedback=reverse)
        self.assertIsNone(window.reason);self.assertEqual(window.spin_reverse_frames,1)
        for now in (10.401,10.402,10.403):self.ask(window,now,feedback=reverse)
        self.assertIsNone(window.reason)
        for now in (10.45,10.5,10.55,10.6):self.ask(window,now,feedback=reverse)
        self.assertEqual(window.reason,'unexpected_spin_direction')

    def test_faster_spin_retains_instant_feedback_and_joystick_protection(self):
        for fields,reason in ((dict(velocity_mps_rps=[0.,.301]),'unexpected_velocity'),
                (dict(mode_raw=30),'raw_mode_changed'),(dict(joystick_axes_raw=[1,0]),'joystick_takeover'),
                (dict(candidate_checksum_matches=False),'candidate_checksum_mismatch')):
            window=PulseWindow(10.,expect_inhibited=False,duration=2.,maneuver='spin_left',angular_speed=.2)
            row=decode_feedback(frame(angular=.2));row.update(fields)
            self.ask(window,10.1,feedback=row)
            self.assertEqual(window.reason,reason);self.assertEqual(window.angular_command(10.1),0.)

    def test_nonforward_plans_and_incompatible_options_do_not_open_resources(self):
        with patch('multiprocessing.get_context') as spawn:
            with contextlib.redirect_stdout(io.StringIO()):main(['--maneuver','spin_right','--pulse-s','2'])
            spawn.assert_not_called()
            for extra in (['--maneuver','reverse','--speed','.1'],['--maneuver','spin_left','--pulse-s','2','--silence-s','.5'],['--maneuver','reverse','--execute']):
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(extra)
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
    def exercise(self,disconnect_on_motion=False,simulate_motion=False,pulse_s=.3,speed=.03,stop_jitter_s=0.,
                 silence_s=None,firmware_timeout_s=None,watchdog_mode=1,disconnect_during_silence=False,drop_silence_feedback=False,
                 expect_silence_entry_failure=False,maneuver='forward',angular_speed=None,spin_startup_reverse=False):
        master,slave=pty.openpty();tty.setraw(slave);port=os.ttyname(slave)
        os.set_blocking(master,False);halt=threading.Event();moving=threading.Event();writes=[];silencing=threading.Event()
        def controller():
            pending=bytearray();mode=3;velocity=0.;rotation=0.;stopped_at=None;next_emit=time.monotonic();last_command=None;rotation_began=None
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
                        last_command=time.monotonic()
                        velocity=linear/1000. if simulate_motion else 0.
                        rotation=angular/1000. if simulate_motion else 0.
                        if angular and rotation_began is None:rotation_began=time.monotonic()
                        if linear or angular:moving.set()
                        if before_mode==1 and mode==3 and moving.is_set():stopped_at=time.monotonic()
                if firmware_timeout_s is not None and last_command is not None and mode==1 and time.monotonic()-last_command>firmware_timeout_s:
                    velocity=rotation=0.;mode=watchdog_mode
                if time.monotonic()>=next_emit:
                    jitter=stopped_at is not None and time.monotonic()-stopped_at<stop_jitter_s
                    if not (drop_silence_feedback and silencing.is_set() and mode==1):
                        emitted_rotation=-rotation if spin_startup_reverse and rotation_began is not None and time.monotonic()-rotation_began<.06 else rotation
                        try:os.write(master,frame(mode=mode,velocity=.018 if jitter else velocity,angular=emitted_rotation))
                        except OSError:pass
                    next_emit=time.monotonic()+.025
        emulator=threading.Thread(target=controller);emulator.start()
        context=mp.get_context('spawn');parent,child=context.Pipe()
        worker=None
        try:
            with tempfile.TemporaryDirectory() as directory:
                selected=probe_profile(speed=speed if maneuver=='forward' else None,pulse_s=pulse_s,maneuver=maneuver,angular_speed=angular_speed)
                worker=context.Process(target=_serial_owner,args=(child,port,directory,not simulate_motion and silence_s is None,pulse_s,
                    speed if maneuver=='forward' else None,silence_s,maneuver,angular_speed));worker.start();child.close()
                os.close(slave);slave=None
                began=time.monotonic();result=None;disconnected=False
                while worker.is_alive() and time.monotonic()-began<10:
                    if ((disconnect_on_motion and moving.is_set()) or (disconnect_during_silence and silencing.is_set())) and not disconnected:
                        parent.close();disconnected=True
                    if not disconnected:
                        parent.send(('heartbeat',time.monotonic()))
                        if parent.poll(.02):
                            message=parent.recv()
                            if message.get('phase')=='firmware_command_silence':silencing.set()
                            if message.get('done'):result=message;break
                    else:time.sleep(.02)
                worker.join(timeout=3.)
                self.assertFalse(worker.is_alive(),'Serial owner did not finish its bounded test')
                self.assertEqual(worker.exitcode,0)
                report=json.loads((Path(directory)/'result.json').read_text())
                events=[json.loads(s) for s in (Path(directory)/'events.jsonl').read_text().splitlines()]
                self.assertTrue(report['stop_feedback_observed'],report)
                self.assertFalse(report['motion_verified']);self.assertFalse(report['execution_feedback'])
                # Disconnection may happen before the first moving reply is read.
                # This scenario checks independent zero recovery, not response capture.
                if not (disconnect_on_motion and simulate_motion):
                    self.assertEqual(report['nonzero_candidate_feedback_observed'],simulate_motion)
                self.assertTrue(moving.is_set(),'Simulated serial controller received no nonzero test frame')
                self.assertEqual(writes[-1][1:],(0,0,3))
                self.assertLessEqual(max(abs(r[1]) for r in writes),int(selected['linear_mps']*1000))
                self.assertLessEqual(max(abs(r[2]) for r in writes),int(selected['angular_rps']*1000))
                self.assertEqual(report['profile']['linear_mps'],selected['linear_mps'])
                self.assertEqual(report['profile']['pulse_s'],pulse_s)
                if stop_jitter_s:self.assertGreater(report['stop_confirmation_elapsed_s'],1.1)
                self.assertFalse(report['stop_confirmation_timed_out'])
                self.assertIsNotNone(report['response_summary'])
                if selected['angular_rps']:self.assertTrue(all(r[1]==0 for r in writes))
                else:self.assertTrue(all(r[2]==0 for r in writes))
                self.assertEqual(report['nonzero_frames'],sum(r[1]!=0 or r[2]!=0 for r in writes))
                for row in events:
                    if row['kind']=='write':
                        wire=int.from_bytes(bytes.fromhex(row['frame_hex'])[2:4],byteorder='big',signed=True)/1000.
                        self.assertEqual(row['linear_mps'],wire)
                if disconnect_on_motion or disconnect_during_silence:
                    self.assertFalse(report['completed']);self.assertIn('Parent',report['error'])
                    first_motion=next(r['mono'] for r in events if r['kind']=='phase' and r['name']=='firmware_command_silence') if disconnect_during_silence else next(r[0] for r in writes if r[1] or r[2])
                    after=next(r[0] for r in writes if r[0]>first_motion and r[1]==0 and r[2]==0)
                    self.assertLess(after-first_motion,.5)
                elif drop_silence_feedback:
                    self.assertFalse(report['completed']);self.assertIn('feedback',report['error'].lower())
                elif expect_silence_entry_failure:
                    self.assertFalse(report['completed']);self.assertIn('Moving neutral feedback required',report['error'])
                else:
                    self.assertTrue(report['completed'],report)
                    start=next(r['mono'] for r in events if r['kind']=='phase' and r['name']=='single_'+maneuver+'_pulse')
                    self.assertTrue(all(r['mono']-start<pulse_s+.03 for r in events if r['kind']=='write' and (r['linear_mps'] or r['angular_rps'])))
                return report,events,writes
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

    def test_firmware_silence_observes_stop_without_sending_zero_first(self):
        report,events,_=self.exercise(simulate_motion=True,pulse_s=1.2,speed=.08,silence_s=.6,firmware_timeout_s=.15)
        proof=report['silence_test'];self.assertTrue(proof['stationary_before_rescue'])
        self.assertFalse(proof['firmware_watchdog_verified'])
        self.assertLess(proof['confirmation_delay_from_last_write_s'],.6)
        self.assertFalse(any(r['kind']=='write' and proof['started_mono']<=r['mono']<=proof['ended_mono'] for r in events))
        self.assertTrue(all(r['linear_mps']==0 for r in events if r['kind']=='write' and r['mono']>=proof['started_mono']))
        self.assertLess(report['response_summary']['command_window_elapsed_s'],1.25)
        self.assertGreater(report['response_summary']['silence_window_elapsed_s'],.2)

    def test_firmware_may_switch_to_manual_mode_during_silence(self):
        report,_,_=self.exercise(simulate_motion=True,pulse_s=1.2,speed=.08,silence_s=.6,firmware_timeout_s=.15,watchdog_mode=3)
        self.assertTrue(report['silence_test']['stationary_before_rescue'])

    def test_longer_silence_observes_stationary_without_extending_the_combined_limit(self):
        report,events,_=self.exercise(simulate_motion=True,pulse_s=1.5,speed=.1,silence_s=1.5,firmware_timeout_s=.8)
        proof=report['silence_test'];self.assertTrue(proof['stationary_before_rescue'])
        self.assertLessEqual(proof['elapsed_from_last_write_s'],1.55)
        self.assertFalse(any(r['kind']=='write' and proof['started_mono']<=r['mono']<=proof['ended_mono'] for r in events))

    def test_missing_firmware_watchdog_requires_zero_rescue_and_never_claims_stop_before_it(self):
        report,events,_=self.exercise(simulate_motion=True,pulse_s=1.2,speed=.08,silence_s=.5)
        proof=report['silence_test'];self.assertFalse(proof['stationary_before_rescue'])
        self.assertEqual(proof['status'],'not_observed_within_window')
        self.assertTrue(report['stop_feedback_observed'])
        self.assertLess(proof['elapsed_from_last_write_s'],.56)
        self.assertFalse(any(r['kind']=='write' and proof['started_mono']<=r['mono']<=proof['ended_mono'] for r in events))

    def test_silence_preserves_parent_disconnect_and_feedback_loss_stop(self):
        for options in [dict(disconnect_during_silence=True),dict(drop_silence_feedback=True)]:
            with self.subTest(options=options):
                report,_,_=self.exercise(simulate_motion=True,pulse_s=1.2,speed=.08,silence_s=.7,**options)
                self.assertEqual(report['silence_test']['status'],'aborted')
                self.assertFalse(report['silence_test']['stationary_before_rescue'])
                self.assertLess(report['silence_test']['elapsed_from_last_write_s'],.5)

    def test_no_moving_feedback_prevents_silence_experiment(self):
        report,events,_=self.exercise(pulse_s=1.2,speed=.08,silence_s=.5,expect_silence_entry_failure=True)
        self.assertEqual(report['silence_test']['status'],'not_started')
        self.assertFalse(any(r.get('name')=='firmware_command_silence' for r in events))

    def test_fixed_reverse_restores_zero_and_does_not_send_forward_or_rotation(self):
        report,_,writes=self.exercise(simulate_motion=True,pulse_s=1.,maneuver='reverse')
        self.assertTrue(any(r[1]<0 for r in writes));self.assertTrue(all(r[1]<=0 and r[2]==0 for r in writes))

    def test_fixed_left_and_right_spins_restore_zero_and_never_translate(self):
        for kind,sign in [('spin_left',1),('spin_right',-1)]:
            with self.subTest(kind=kind):
                report,_,writes=self.exercise(simulate_motion=True,pulse_s=1.,maneuver=kind)
                self.assertTrue(any(r[2]*sign>0 for r in writes));self.assertTrue(all(r[1]==0 and r[2]*sign>=0 for r in writes))

    def test_selected_faster_spins_pass_short_startup_reversal_and_restore_zero(self):
        for kind,sign in [('spin_left',1),('spin_right',-1)]:
            with self.subTest(kind=kind):
                report,_,writes=self.exercise(simulate_motion=True,pulse_s=1.1,maneuver=kind,angular_speed=.2,spin_startup_reverse=True)
                self.assertEqual(report['profile']['angular_rps'],.2)
                self.assertTrue(any(r[2]==sign*200 for r in writes))

    def test_faster_spin_parent_disconnect_keeps_independent_zero_recovery(self):
        self.exercise(disconnect_on_motion=True,simulate_motion=True,pulse_s=2.,maneuver='spin_left',angular_speed=.2)


if __name__=='__main__':unittest.main()
