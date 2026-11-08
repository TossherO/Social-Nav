"""Pure orchestration and alignment tests; no devices or ROS are opened."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parents[1] / 'navigation/commissioning'


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


program = load('video_silence')
alignment = load('video_alignment')


class VideoSilenceTests(unittest.TestCase):
    def test_default_plan_does_not_start_process(self):
        with patch.object(program.subprocess, 'Popen') as popen:
            self.assertEqual(program.main([]), 0)
        popen.assert_not_called()

    def test_marker_failure_prevents_probe(self):
        process = MagicMock()
        process.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(program.subprocess, 'Popen', return_value=process), \
             patch.object(program, 'marker_ready', return_value=False), \
             patch.object(program, '_run_segment') as probe:
            self.assertEqual(program.execute(output_parent=Path(directory)), 1)
            run = next(Path(directory).iterdir())
            summary = json.loads((run / 'summary.json').read_text())
        probe.assert_not_called()
        self.assertFalse(summary['probe_started'])
        self.assertIn('no motion sent', summary['error'])

    def test_observer_failure_prevents_probe(self):
        marker, observer = MagicMock(), MagicMock()
        marker.poll.return_value = None
        observer.poll.return_value = 1
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(program.subprocess, 'Popen', side_effect=[marker, observer]), \
             patch.object(program, 'marker_ready', return_value=True), \
             patch.object(program.helper, 'master_available', return_value=True), \
             patch.object(program.helper, 'observer_ready', return_value=False), \
             patch.object(program.helper, 'stop_process'), \
             patch.object(program, '_run_segment') as probe:
            self.assertEqual(program.execute(output_parent=Path(directory)), 1)
            run = next(Path(directory).iterdir())
            summary = json.loads((run / 'summary.json').read_text())
        probe.assert_not_called()
        self.assertFalse(summary['probe_started'])
        self.assertIn('no motion sent', summary['error'])

    def test_completed_probe_keeps_uncertainty_and_owned_cleanup(self):
        marker, observer = MagicMock(), MagicMock()
        marker.poll.return_value = observer.poll.return_value = None

        def fake_probe(command, log):
            run = Path(log).parent
            probe = run / 'probe'
            probe.mkdir()
            now = time.monotonic()
            markers = [dict(marker_id=i, draw_queued_mono=now + delta)
                       for i, delta in enumerate((-2., -1., 1., 2.))]
            (run / 'marker_events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in markers))
            events = [dict(kind='write', mono=now + delta, linear_mps=speed, angular_rps=0.)
                      for delta, speed in ((-.5, .1), (0., .1), (.5, 0.))]
            (probe / 'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
            report = dict(completed=True, stop_feedback_observed=True,
                          nonzero_candidate_feedback_observed=True, profile=program.make_plan()['profile'],
                          silence_test=dict(requested_s=program.SILENCE_S))
            (probe / 'result.json').write_text(json.dumps(report))
            return 0

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(program.subprocess, 'Popen', side_effect=[marker, observer]) as popen, \
             patch.object(program, 'marker_ready', return_value=True), \
             patch.object(program.helper, 'master_available', return_value=True), \
             patch.object(program.helper, 'observer_ready', return_value=True), \
             patch.object(program.helper, 'stop_process') as stop, \
             patch.object(program, '_run_segment', side_effect=fake_probe) as probe, \
             patch.object(program.time, 'sleep'):
            self.assertEqual(program.execute(output_parent=Path(directory)), 0)
            run = next(Path(directory).iterdir())
            summary = json.loads((run / 'summary.json').read_text())
            self.assertTrue((run / 'STOP').exists())
            self.assertTrue((run / 'MARKER_STOP').exists())
        self.assertEqual(popen.call_count, 2)  # existing ROS master, never start sensors
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(stop.call_args_list[0].args, (observer,))
        self.assertEqual(stop.call_args_list[1].args, (None,))
        self.assertTrue(summary['marker_spans_motion'])
        self.assertTrue(summary['video_recording_not_verified'])
        self.assertFalse(summary['firmware_watchdog_verified'])
        self.assertFalse(summary['vehicle_profile_updated'])


class AlignmentTests(unittest.TestCase):
    def test_estimates_time_and_rejects_invalid_marks(self):
        markers = [dict(marker_id=i, draw_request_mono=100.+i, draw_queued_mono=100.+i)
                   for i in range(3)]
        marks = [dict(marker_id=i, video_time_before_s=i+.98, video_time_after_s=i+1.02)
                 for i in range(3)]
        result = alignment.align(markers, marks)
        self.assertAlmostEqual(result['host_monotonic_per_video_second'], 1.)
        self.assertAlmostEqual(result['host_offset_s'], 99.)
        self.assertFalse(result['physical_display_latency_known'])
        with self.assertRaises(ValueError):
            alignment.align(markers, [marks[0], marks[0], marks[2]])
        with self.assertRaises(ValueError):
            alignment.align(markers, marks[:2])


if __name__ == '__main__':
    unittest.main()
