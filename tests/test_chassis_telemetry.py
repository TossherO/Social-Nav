"""Synthetic chassis records, backpressure and expiry; no real serial device."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from navigation.control.chassis import DriveState
from navigation.control.telemetry import ChassisTelemetry
from navigation.planning.motion import load_motion_config


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
