"""Local capture launchers use the project sources without touching devices."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('capture_shortcuts', ROOT/'perception/capture/capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureShortcutsTests(unittest.TestCase):
    def test_dry_runs_do_not_open_devices_or_create_recordings(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(capture.subprocess, 'Popen', side_effect=AssertionError('process started')), \
                patch.object(capture, 'master', side_effect=AssertionError('ROS contacted')), \
                contextlib.redirect_stdout(io.StringIO()):
            for args in (['sensors', '--dry-run'],
                         ['record', 'mapping', '--dry-run', '--output-root', directory],
                         ['record', 'navigation', '--dry-run', '--output-root', directory]):
                self.assertEqual(capture.main(args), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_locks_are_outside_git_visible_scripts_and_prevent_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(capture, 'ROOT', Path(directory)):
            with capture.exclusive('sensors'):
                self.assertTrue((Path(directory)/'outputs/local_capture/.sensors.lock').is_file())
                with self.assertRaises(RuntimeError):
                    with capture.exclusive('sensors'):
                        pass

    def test_finish_checks_topics_and_records_full_bag_hash(self):
        class FakeBag:
            def __init__(self, *_): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def get_type_and_topic_info(self):
                return SimpleNamespace(topics={name: SimpleNamespace(message_count=1) for name in capture.TOPICS})
            def get_end_time(self): return 2.
            def get_start_time(self): return 1.
            def get_message_count(self): return len(capture.TOPICS)

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'rosbag': SimpleNamespace(Bag=FakeBag)}), \
                contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory); bag = output/'mapping.bag'; bag.write_bytes(b'synthetic bag')
            report = {'complete': True}
            capture.finish_bag(bag, output, report, None)
            self.assertEqual(report['sha256'], hashlib.sha256(b'synthetic bag').hexdigest())
            self.assertEqual(set(report['bag_info']['topic_counts']), set(capture.TOPICS))
            self.assertTrue(json.loads((output/'capture.json').read_text())['bag_closed_and_index_readable'])
            self.assertIn(report['sha256'], (output/'SHA256SUMS').read_text())

    def test_incomplete_or_missing_topic_bag_is_not_marked_complete(self):
        class MissingTopics:
            def __init__(self, *_): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def get_type_and_topic_info(self): return SimpleNamespace(topics={})
            def get_end_time(self): return 2.
            def get_start_time(self): return 1.
            def get_message_count(self): return 0

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'rosbag': SimpleNamespace(Bag=MissingTopics)}):
            output = Path(directory); bag = output/'mapping.bag'; bag.write_bytes(b'synthetic bag')
            Path(str(bag)+'.active').touch()
            with self.assertRaisesRegex(RuntimeError, '正常关闭'):
                capture.finish_bag(bag, output, {}, None)
            Path(str(bag)+'.active').unlink()
            with self.assertRaisesRegex(RuntimeError, '缺少必要话题'):
                capture.finish_bag(bag, output, {}, None)
            self.assertFalse((output/'capture.json').exists())

    def test_sensor_launch_selects_project_file_and_requested_fps(self):
        from perception.sensors.__main__ import main
        cases = [(['--serial-front', 'TEST'], 'camera_a:=true', 'camera_b:=false', 'fps:=30'),
                 (['--serial-front', 'TEST', '--fps', '15'], 'camera_a:=true', 'camera_b:=false', 'fps:=15'),
                 (['--lidar-only'], 'camera_a:=false', 'camera_b:=false', 'fps:=30'),
                 (['--serial-front', 'TEST', '--rear', '--serial-rear', 'TEST2'],
                  'camera_a:=true', 'camera_b:=true', 'fps:=30')]
        for args, front, rear, fps in cases:
            with self.subTest(args=args), patch.object(sys, 'argv', ['perception.sensors', *args]), \
                    patch('perception.sensors.__main__.Path.is_file', return_value=False), \
                    patch('perception.sensors.imu_sampling.select_policy', return_value=None), \
                    patch('perception.sensors.__main__.os.execvp') as execute:
                main()
                binary, command = execute.call_args.args
            self.assertEqual(binary, 'roslaunch')
            self.assertEqual(command[1], str(ROOT/'perception/sensors/ros/wheelchair_sensors/launch/sensors.launch'))
            self.assertIn(front, command); self.assertIn(rear, command); self.assertIn(fps, command)


if __name__ == '__main__':
    unittest.main()
