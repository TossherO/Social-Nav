"""Synthetic chassis serial, process-loss and telemetry tests; no real device."""
import json
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

from navigation.control.chassis import ChassisProcess, DriveState, COMMAND_PERIOD_S, _command_due, _read_available
from navigation.control.telemetry import ChassisTelemetry
from navigation.planning.motion import load_motion_config
from tests.chassis_fixtures import frame

class SerialOwnerLossTests(unittest.TestCase):
    def test_owner_killed_after_motion_cannot_stop_a_controller_without_watchdog(self):
        # This emulator deliberately holds the last command forever: it models
        # one possible controller, not the behavior of the real wheelchair.
        master, slave = pty.openpty()
        tty.setraw(slave)
        port = os.ttyname(slave)
        os.close(slave)
        os.set_blocking(master, False)
        shutdown = threading.Event()
        seen_motion = threading.Event()
        commands = []

        def controller():
            pending = bytearray()
            mode = 3
            while not shutdown.wait(.01):
                if select.select([master], [], [], 0)[0]:
                    try:
                        pending.extend(os.read(master, 4096))
                    except OSError:
                        pass
                while len(pending) >= 12:
                    start = pending.find(b'\xaf\x01')
                    if start < 0:
                        pending.clear()
                        break
                    del pending[:start]
                    if len(pending) < 12:
                        break
                    packet = bytes(pending[:12])
                    del pending[:12]
                    _, _, v, w, mode, _, _, _, _ = struct.unpack('>2B2hBhBBB', packet)
                    commands.append((v, w, mode))
                    if v != 0 or w != 0:
                        seen_motion.set()
                try:
                    os.write(master, frame(mode=mode))
                except OSError:
                    pass

        emulator = threading.Thread(target=controller)
        emulator.start()
        driver = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                motion = load_motion_config()
                motion['controller'] = dict(velocity_limit_field=1500,
                    feedback_max_linear_mps=1.5, feedback_max_angular_rps=1.5)
                driver = ChassisProcess(port, Path(folder) / 'chassis', motion)
                deadline = time.monotonic() + 5.
                while not seen_motion.is_set() and time.monotonic() < deadline:
                    now = time.monotonic()
                    packet = dict(mono=now, arm=True, arm_revision=1, allowed=True,
                                  units=[20, 0], expires_mono=now+.1)
                    driver.channel.send(json.dumps(packet).encode())
                    time.sleep(.025)
                self.assertTrue(seen_motion.is_set(), 'Synthetic controller did not reach motion')
                driver.process.kill()
                driver.process.join(timeout=2.)
                self.assertFalse(driver.process.is_alive())
                time.sleep(.35)
                self.assertEqual(commands[-1], (20, 0, 1),
                                 'No code remains to send a stop after owner death')
                driver.close()
                driver = None
        finally:
            if driver is not None:
                driver.close()
            shutdown.set()
            emulator.join(timeout=1.)
            os.close(master)


class SerialPacingTests(unittest.TestCase):
    def test_read_would_block_is_not_feedback_but_disconnect_remains_fatal(self):
        with patch('navigation.control.chassis.os.read', side_effect=BlockingIOError):
            self.assertIsNone(_read_available(5))
        with patch('navigation.control.chassis.os.read', return_value=b''):
            with self.assertRaisesRegex(RuntimeError, 'Serial input disconnected'):
                _read_available(5)
        with patch('navigation.control.chassis.os.read', return_value=b'abc'):
            self.assertEqual(_read_available(5),b'abc')

    def test_stop_and_disarm_are_due_immediately(self):
        self.assertTrue(_command_due(1., 1.05, (20, 0, 1), (0, 0, 1)))
        self.assertTrue(_command_due(1., 1.05, (0, 0, 1), (0, 0, 3)))
        self.assertFalse(_command_due(1., 1.05, (0, 0, 3), (0, 0, 3)))
        self.assertFalse(_command_due(1., 1.05, (20, 0, 1), (30, 0, 1)))
        self.assertTrue(_command_due(1.05, 1.05, (20, 0, 1), (30, 0, 1)))

    def test_fragmented_feedback_does_not_flood_serial_writes(self):
        master, slave = pty.openpty()
        tty.setraw(slave)
        port = os.ttyname(slave)
        os.close(slave)
        os.set_blocking(master, False)
        stop = threading.Event()

        def controller():
            packet = frame(mode=3)
            index = 0
            while not stop.wait(.001):
                if select.select([master], [], [], 0)[0]:
                    try:
                        os.read(master, 4096)
                    except OSError:
                        pass
                try:
                    os.write(master, packet[index:index+1])
                    index = (index+1) % len(packet)
                except OSError:
                    pass

        worker = threading.Thread(target=controller)
        worker.start()
        driver = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'chassis'
                driver = ChassisProcess(port, output, load_motion_config())
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    packet = dict(mono=now, arm=False, arm_revision=0,
                                  allowed=False, units=[0, 0], expires_mono=now+.1)
                    driver.channel.send(json.dumps(packet).encode())
                    time.sleep(.05)
                cutoff = time.monotonic()
                driver.close()
                driver = None
                rows = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
                writes = [r for r in rows if r['kind']=='wire_target']
                states = [r for r in rows if r['kind']=='state']
                # Shutdown writes use a separate fixed cleanup cadence.
                times = [r['mono'] for r in writes if r['mono'] < cutoff]
                self.assertTrue(any(r['kind']=='feedback' for r in rows))
                self.assertGreaterEqual(len(times), 15)
                self.assertLess(len(times), 65)
                self.assertTrue(all(b-a >= COMMAND_PERIOD_S-.005 for a,b in zip(times,times[1:])))
                self.assertTrue(all(not r['units'][0] and not r['units'][1] for r in writes))
                self.assertFalse(any('BlockingIOError' in str(r.get('reason')) for r in states))
        finally:
            if driver is not None:
                driver.close()
            stop.set()
            worker.join(timeout=1.)
            os.close(master)


class ChassisTelemetryTests(unittest.TestCase):
    def test_feedback_and_authorization_preserve_distinct_source_times(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ChassisTelemetry(folder)
            recorder.emit('feedback', mono=12., timestamp_source='host_receive_monotonic',
                          device_measurement_stamp_s=None, feedback=dict(velocity_mps_rps=[.03,-.02]))
            packet = dict(mono=11.99, expires_mono=12.1,
                          diagnostics=dict(clock_domain='runtime_live', pose_velocity_estimate=dict(
                              stamp_s=1790000000., source='causal_algorithm_poses_not_chassis_feedback', velocity=[.04,0.])))
            recorder.emit('authorization', mono=12.01, packet=packet)
            recorder.emit('wire_target', mono=12.02, units=[30,-20,1])
            recorder.status(dict(state='disarmed', reason='operator_disarmed', status_mono=12.03))
            result = recorder.close(timeout=1)
            self.assertTrue(result['complete'], result)
            rows = [json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['mono'],12.)
            self.assertIsNone(rows[0]['device_measurement_stamp_s'])
            self.assertEqual(rows[1]['packet'],packet)
            self.assertEqual(rows[2]['units'],[30,-20,1])
            self.assertEqual(json.loads((Path(folder)/'status.json').read_text())['reason'],'operator_disarmed')
            self.assertTrue(json.loads((Path(folder)/'telemetry_summary.json').read_text())['complete'])

    def test_blocked_disk_drops_with_bound_and_cannot_delay_stop_decision(self):
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        original = ChassisTelemetry._write_event
        def blocked(recorder, stream, row):
            entered.set()
            release.wait(timeout=3)
            original(recorder, stream, row)
        with tempfile.TemporaryDirectory() as folder, patch.object(ChassisTelemetry,'_write_event',blocked):
            recorder = ChassisTelemetry(folder, capacity=2)
            try:
                recorder.emit('feedback', mono=1.)
                self.assertTrue(entered.wait(timeout=1))
                def produce():
                    for i in range(100):
                        recorder.emit('feedback', mono=2.+i)
                    recorder.status(dict(state='disarmed', reason='safety_watchdog'))
                    submitted.set()
                producer = threading.Thread(target=produce)
                producer.start()
                self.assertTrue(submitted.wait(timeout=1), 'producer must not wait for the blocked writer')
                producer.join(timeout=1)
                state = DriveState(load_motion_config())
                state.state='armed';state.heartbeat=10.
                self.assertEqual(state.command(10.21),(0,0,3))
                self.assertEqual(state.reason,'safety_watchdog')
                self.assertLessEqual(recorder.queue.qsize(),2)
                self.assertGreater(recorder.dropped['feedback'],0)
            finally:
                release.set()
                result = recorder.close(timeout=1)
            self.assertFalse(result['complete'])
            self.assertTrue(result['finalized'])
            self.assertEqual(result['seen']['feedback'],101)
            self.assertEqual(result['written']+sum(result['dropped'].values()),101)
            self.assertEqual(json.loads((Path(folder)/'status.json').read_text())['reason'],'safety_watchdog')

    def test_write_failure_is_reported_without_entering_the_command_path(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ChassisTelemetry,'_write_event',side_effect=OSError('disk full')):
            recorder = ChassisTelemetry(folder)
            recorder.emit('feedback', mono=1.)
            result = recorder.close(timeout=1)
            self.assertFalse(result['complete'])
            self.assertIn('disk full',result['error'])
            self.assertFalse(json.loads((Path(folder)/'telemetry_summary.json').read_text())['complete'])

