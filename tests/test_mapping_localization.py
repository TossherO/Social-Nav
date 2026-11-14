"""Task/input contracts and map protection; no ROS master, models or hardware."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from perception.mapping_localization.__main__ import arguments, algorithm_command, wait_for_native_drain, merge_timings, wait_for_display_recording


class DisplayFinalizationTests(unittest.TestCase):
    def test_waits_for_recorder_child_after_launcher_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'display.bag';active=Path(str(path)+'.active')
            path.write_bytes(b'not_final_yet');active.write_bytes(b'complete_recording')
            writer=threading.Timer(.03,lambda:active.replace(path));writer.start()
            try:
                binding=wait_for_display_recording(path,timeout=1)
                self.assertEqual(path.read_bytes(),b'complete_recording')
                self.assertEqual(binding['size'],len(b'complete_recording'))
            finally:writer.join()

    def test_missing_recording_still_fails_with_a_bounded_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'not finalized'):
                wait_for_display_recording(Path(directory)/'missing.bag',timeout=.01)


class TimingJoinTests(unittest.TestCase):
    def test_profiles_join_by_exact_observation_and_keep_missing_measurements_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'timings.csv').write_text('stamp_ns,processing_ms\n1000000001,10\n1000000002,20\n')
            (output/'frontend_timings.csv').write_text('stamp_ns,fast_lio_processing_ms\n1000000002,5\n1000000003,6\n')
            merge_timings(output)
            self.assertEqual((output/'timings.csv').read_text().splitlines(), [
                'stamp_ns,processing_ms,fast_lio_processing_ms', '1000000001,10,', '1000000002,20,5'])
            self.assertFalse((output/'frontend_timings.csv').exists())


class NativeDrainTests(unittest.TestCase):
    def test_productive_backlog_can_exceed_twenty_seconds_then_finish(self):
        now = [0.0]
        last_output = [0.0]

        def advance(seconds):
            now[0] += seconds

        def status():
            done = min(15, int(now[0]/2))
            if now[0] < 30:
                last_output[0] = now[0]
            return dict(processed=done, received_clouds=15, received_odometry=15,
                        busy=done < 15, pending_clouds=15-done)

        with patch('perception.mapping_localization.__main__.time.monotonic', side_effect=lambda: now[0]), \
             patch('perception.mapping_localization.__main__.time.sleep', side_effect=advance):
            stats = wait_for_native_drain(status, lambda: True, last_output)
        self.assertEqual(stats['processed'], 15)
        self.assertGreater(now[0], 32)

    def test_silent_busy_or_unpaired_cloud_stalls_instead_of_early_success(self):
        for busy, pending in [(True, 0), (False, 1)]:
            now = [0.0]

            def advance(seconds):
                now[0] += seconds

            stats = dict(processed=1, received_clouds=2, received_odometry=1,
                         busy=busy, pending_clouds=pending)
            with patch('perception.mapping_localization.__main__.time.monotonic', side_effect=lambda: now[0]), \
                 patch('perception.mapping_localization.__main__.time.sleep', side_effect=advance):
                with self.assertRaisesRegex(RuntimeError, 'stalled'):
                    wait_for_native_drain(lambda: stats, lambda: True, [0.0], timeout=3)


class TaskArgumentsTests(unittest.TestCase):
    def test_joint_3d_bag_and_live_do_not_require_image_policy_or_calibration(self):
        with patch('perception.detection_tracking.inference.config.load_config',
                   side_effect=AssertionError('3D task must not load optional image policy')):
            for source in ('bag','live'):
                with self.subTest(source=source):
                    args=self.parse('odometry',source,extra=['--people-python',sys.executable])
                    self.assertFalse(args.image_association)
                    self.assertIsNone(args.decision_calibration)
            self.rejected('odometry',extra=['--image-association'])

    def test_pin_step_replay_accepts_3d_tracking_without_disabling_images_explicitly(self):
        args=self.parse('mapping',extra=['--backend','pin_slam','--backend-python',sys.executable,
                                        '--people-python',sys.executable,'--step-replay'])
        self.assertTrue(args.step_replay)
        self.assertFalse(args.image_association)
        self.rejected('mapping',extra=['--backend','pin_slam','--backend-python',sys.executable,
                                      '--people-python',sys.executable,'--step-replay','--image-association'])

    def test_environment_and_mixed_route_are_explicit_native_options(self):
        args=self.parse('localization',extra=['--environment','--allow-local-only','--scene'])
        command=algorithm_command(args,self.root/'new_run',self.map,True)
        self.assertIn('environment:=true',command)
        self.assertIn('allow_local_only:=true',command)
        self.assertIn('scene:=true',command)
        self.assertIn('scene_people:=false',command)
        self.rejected('localization',extra=['--allow-local-only'])
        self.rejected('localization',extra=['--scene'])
        self.rejected('localization',extra=['--association-coverage','missing.yaml'])
        self.rejected('odometry',extra=['--environment','--allow-local-only'])

    def test_generated_coverage_reaches_ros_localization_without_environment(self):
        args=self.parse('localization')
        coverage=self.root/'frozen.yaml'
        command=algorithm_command(args,self.root/'new_run',self.map,True,coverage_path=coverage)
        self.assertIn('association_coverage:='+str(coverage),command)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bag = self.root / 'synthetic.bag'
        self.bag.write_bytes(b'CLI_FIXTURE_NOT_A_RECORDING')
        self.map = self.root / 'prior_map'
        (self.map / 'keyframes').mkdir(parents=True)
        (self.map / 'COMPLETE').write_text('SYNTHETIC_MAP_NOT_LOADED')
        (self.map / 'keyframes/poses.txt').write_text('1 2 3 4 .1 .2 .3 0 1\n')

    def parse(self, mode, source='bag', extra=()):
        argv = ['--mode', mode, '--source', source, '--out', str(self.root / 'new_run')]
        if source == 'bag':
            argv += ['--bag', str(self.bag)]
        if mode == 'localization':
            argv += ['--localization-map', str(self.map)]
        with contextlib.redirect_stdout(io.StringIO()):
            return arguments(argv + list(extra))

    def rejected(self, mode, source='bag', extra=()):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.parse(mode, source, extra)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / 'new_run').exists())

    def test_prior_scene_requires_explicit_scene_and_fixed_map_and_forwards_package(self):
        self.rejected('mapping',extra=['--environment','--scene','--prior-map',str(self.map)])
        self.rejected('localization',extra=['--environment','--prior-map',str(self.map)])
        args = self.parse('localization',extra=['--environment','--scene','--prior-map',str(self.map)])
        command = algorithm_command(args,self.root/'new_run',self.map,True)
        self.assertIn('static_map:='+str(self.map),command)
        explicit = self.parse('localization',extra=['--environment','--scene','--static-map',str(self.map)])
        self.assertEqual(algorithm_command(explicit,self.root/'new_run',self.map,True),command)

    def test_six_combinations_select_pipeline_without_creating_outputs(self):
        for source in ('bag', 'live'):
            for mode in ('mapping', 'localization', 'odometry'):
                with self.subTest(source=source, mode=mode):
                    args = self.parse(mode, source)
                    self.assertEqual(args.backend, 'fast_lio' if mode == 'odometry' else 'roll_fast_lio')
                    self.assertEqual(args.pose, [.1, .2, .3, 2, 3, 4] if mode == 'localization' else [0]*6)
        self.assertFalse((self.root / 'new_run').exists())

    def test_display_recording_requires_environment(self):
        self.rejected('mapping',extra=['--record-display'])
        args=self.parse('localization',extra=['--environment','--record-display'])
        self.assertTrue(args.record_display)

    def test_odometry_rejects_prior_map_pose_or_incompatible_backend(self):
        for extra in (['--map', str(self.map)], ['--pose', '0', '0', '0', '0', '0', '0'],
                      ['--backend', 'roll_fast_lio'], ['--backend', 'pin_slam']):
            self.rejected('odometry', extra=extra)

    def test_live_rejects_replay_controls_and_input_recording(self):
        for extra in (['--bag', str(self.bag)], ['--rate', '.5'], ['--step-replay'], ['--start', '1']):
            self.rejected('mapping', 'live', extra)

    def test_stationary_input_is_explicit_and_bag_offset_is_validated(self):
        for source in ('bag','live'):
            args=self.parse('odometry',source,['--assume-stationary'])
            self.assertTrue(args.assume_stationary)
            command=algorithm_command(args,self.root/'new_run',None,source=='bag')
            self.assertIn('assume_stationary:=true',command)
        self.assertFalse(self.parse('odometry').assume_stationary)
        self.assertEqual(self.parse('odometry',extra=['--start','4']).start,4)
        for value in ('nan','-1'):
            self.rejected('odometry',extra=['--start',value])
        self.rejected('mapping',extra=['--backend','pin_slam','--backend-python','/usr/bin/python3',
                                       '--assume-stationary'])

    def test_rejects_missing_map_and_protects_existing_map_from_mapping(self):
        self.rejected('localization', extra=['--map', str(self.root / 'missing')])
        self.rejected('mapping', extra=['--map', str(self.map)])
        self.assertEqual((self.map / 'keyframes/poses.txt').read_text(), '1 2 3 4 .1 .2 .3 0 1\n')

    def test_nonfinite_fallback_and_invalid_budgets_fail_before_launch(self):
        (self.map / 'keyframes/poses.txt').write_text('1 nan 0 0 0 0 0 0 1\n')
        self.rejected('localization')
        for extra in (['--duration', 'nan'], ['--rate', '0'], ['--duration', '-1']):
            self.rejected('odometry', extra=extra)

    def test_loop_override_only_applies_to_mapping(self):
        args = self.parse('mapping', extra=['--loop-closure', 'off'])
        command = algorithm_command(args, self.root / 'new_run', self.root / 'new_run/map', True)
        self.assertIn('loop_closure:=false', command)
        self.rejected('localization', extra=['--loop-closure', 'on'])
        self.rejected('odometry', extra=['--loop-closure', 'off'])

    def test_explicit_calibration_is_bound_to_native_launch(self):
        calibration = self.root / 'calibration.yaml'
        calibration.write_text('common: {imu_time_offset: -0.04}\n')
        args = self.parse('odometry', extra=['--lidar-imu-calibration', str(calibration)])
        command = algorithm_command(args, self.root / 'new_run', None, True)
        self.assertIn('lidar_imu_calibration:=' + str(calibration.resolve()), command)
        self.rejected('odometry', extra=['--lidar-imu-calibration', str(self.root / 'missing.yaml')])
        self.rejected('mapping', extra=['--backend', 'pin_slam', '--backend-python', '/usr/bin/python3',
                                       '--lidar-imu-calibration', str(calibration)])

    def test_manual_initialization_gates_both_sources_without_inventing_a_pose(self):
        for source in ('bag', 'live'):
            for extra in (['--rviz'], ['--initialization', 'rviz']):
                args = self.parse('localization', source, extra)
                self.assertIsNone(args.pose)
                self.assertEqual(args.initialization, 'rviz')
                self.assertTrue(args.rviz)
                command = algorithm_command(args, self.root / 'new_run', self.map, source == 'bag')
                self.assertIn('start_frontend:=false', command)
                self.assertIn('wait_for_initial_pose:=true', command)
            for extra in (['--rviz', '--initialization', 'map-start'],
                          ['--rviz', '--pose', '0', '0', '.5', '1', '2', '3']):
                args = self.parse('localization', source, extra)
                self.assertIsNotNone(args.pose)
                self.assertNotEqual(args.initialization, 'rviz')
                command = algorithm_command(args, self.root / 'new_run', self.map, source == 'bag')
                self.assertIn('start_frontend:=true', command)
                self.assertIn('wait_for_initial_pose:=false', command)

    def test_rejects_conflicting_or_inapplicable_initialization_before_launch(self):
        self.rejected('mapping', extra=['--initialization', 'rviz'])
        self.rejected('odometry', extra=['--initialization', 'map-start'])
        self.rejected('localization', extra=['--initialization', 'rviz', '--pose', '0', '0', '0', '0', '0', '0'])
        self.rejected('localization', extra=['--backend', 'pin_slam', '--backend-python', '/usr/bin/python3',
                                              '--initialization', 'rviz'])

    def test_launch_uses_task_clock_and_optional_view_for_both_sources(self):
        for source in ('bag', 'live'):
            for mode in ('mapping', 'localization', 'odometry'):
                args = self.parse(mode, source, ['--rviz'])
                command = algorithm_command(args, self.root / 'new_run', None if mode == 'odometry' else self.map,
                                            source == 'bag')
                self.assertIn('mode:=' + mode, command)
                self.assertIn('use_sim_time:=' + str(source == 'bag').lower(), command)
                self.assertIn('rviz:=false', command)  # CLI owns the optional view separately.
                self.assertIn('visualize:=true', command)
                self.assertIn('keep_details:=false', command)
                self.assertIn('map_dir:=' + ('' if mode == 'odometry' else str(self.map)), command)

    def test_pin_mapping_and_localization_keep_the_existing_launch_contract(self):
        for mode in ('mapping', 'localization'):
            args = self.parse(mode, extra=['--backend', 'pin_slam', '--backend-python', '/usr/bin/python3',
                                           '--pose', '0', '0', '0', '0', '0', '0', '--step-replay'])
            command = algorithm_command(args, self.root / 'new_run', self.map, True)
            self.assertTrue(command[1].endswith('/pin_slam.launch'))
            self.assertIn('python_executable:=' + str(Path('/usr/bin/python3').resolve()), command)
            self.assertFalse(any(value.startswith(('run_dir:=', 'rviz:=', 'keep_details:=')) for value in command))


class FrontendBenchmarkTests(unittest.TestCase):
    def test_small_rotation_difference_is_not_masked_by_rounded_calibration(self):
        import numpy as np
        from perception.mapping_localization.benchmark import difference
        pose=np.eye(4)[None];pose[0,:3,:3]*=1.00002
        states=np.zeros((1,13));states[0,9]=np.inf
        reference=dict(poses=pose,states=states,cloud_hash=np.array(['same']))
        same=difference(reference,reference)
        self.assertTrue(same['exact_states']);self.assertEqual(same['angle_deg']['maximum'],0)
        angle=np.deg2rad(.05);rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        other={k:v.copy() for k,v in reference.items()}
        other['poses'][0,:3,:3]=rotation@pose[0,:3,:3];other['poses'][0,0,3]=.01
        other['states'][0,10]=4;other['states'][0,11]=7
        result=difference(reference,other)
        self.assertAlmostEqual(result['angle_deg']['maximum'],.05,places=6)
        self.assertAlmostEqual(result['position_m']['maximum'],.01)
        self.assertEqual(result['quality_differences'],1)
        self.assertEqual(result['max_effective_point_difference'],7)


if __name__ == '__main__':
    unittest.main()
