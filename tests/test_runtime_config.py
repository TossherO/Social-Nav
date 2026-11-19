"""Fresh-task YAML composition and the shared raw-image sensor boundary."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from perception.mapping_localization.configuration import algorithm_config,ROOT
from perception.sensors.image import rgb_array


class RuntimeConfigTests(unittest.TestCase):
    def test_invalid_command_schedule_precedes_device_setup_and_output_creation(self):
        from unittest.mock import patch
        from runtime import __main__ as entry
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / 'commands.json'
            output = Path(directory) / 'output'
            schedule.write_text('[{"offset_s": false, "kind": "cancel", "value": {}}]')
            with patch('sys.argv', ['runtime', '--bag', str(schedule), '--output', str(output),
                                    '--commands', str(schedule)]):
                with self.assertRaisesRegex(ValueError, 'offset_s'):
                    entry.main()
            self.assertFalse(output.exists())

    def test_failed_startup_closes_workers_before_their_handles(self):
        from unittest.mock import Mock, patch
        from runtime import __main__ as entry
        closed = []
        def fail_after_start(args, resources, handles):
            handles.callback(closed.append, 'socket')
            for name in ('environment', 'control'):
                resources[name] = Mock(close=lambda name=name: closed.append(name))
            raise RuntimeError('simulated startup failure')
        with patch('sys.argv', ['runtime', '--output', 'unused']), patch.object(entry, '_run', fail_after_start):
            with self.assertRaisesRegex(RuntimeError, 'simulated startup failure'):
                entry.main()
        self.assertEqual(closed, ['control', 'environment', 'socket'])

    def test_invalid_transport_settings_are_rejected_by_shared_loader(self):
        from runtime.config import runtime_settings
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'override.yaml'
            for key, value in [('control_period_s', 0), ('scene_export_hz', float('nan')),
                               ('startup_timeout_s', True), ('shared_scan_slots', 0),
                               ('shared_safety_slot_bytes', 127)]:
                with self.subTest(key=key):
                    path.write_text(yaml.safe_dump({key: value}))
                    with self.assertRaises(ValueError):
                        runtime_settings(path)

    def test_fresh_mapping_and_odometry_load_module_and_calibration_keys(self):
        calibration=yaml.safe_load((ROOT/'perception/sensors/calibration/lidar_imu.yaml').read_text())
        for mode in ('mapping','odometry'):
            with self.subTest(mode=mode):
                config,source=algorithm_config(mode=mode)
                self.assertEqual(source,{})
                self.assertEqual(config['common']['lid_topic'],'/rslidar_points')
                self.assertEqual(config['mapping']['extrinsic_R'],calibration['mapping']['extrinsic_R'])
                self.assertEqual(config['odometry']['publish_state'],mode=='odometry')
                self.assertIn('blind',config['preprocess'])
                self.assertIn('initialGuess',config['roll'])
                self.assertEqual(config['matching_backend'],'cuda')
                self.assertEqual(config['deskew_backend'],'cuda')
                self.assertEqual(config['roll']['matchingBackend'],'cuda')
                self.assertEqual(config['linearization_backend'],'cpu')
                self.assertFalse(config['matching_cuda_plane_fit'])
                self.assertFalse(config['roll']['matchingCudaPlaneFit'])

    def test_fresh_localization_needs_a_complete_map_and_explicit_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with self.assertRaises(ValueError):algorithm_config(mode='localization',map_directory=path,pose=[0.]*6)
            (path/'COMPLETE').touch()
            with self.assertRaises(ValueError):algorithm_config(mode='localization',map_directory=path)
            (path/'keyframes').mkdir()
            (path/'keyframes/poses.txt').write_text('0 0 0 0 0 0 0 0 1\n')
            (path/'manifest.yaml').write_text('frame: map\n')
            (path/'GlobalMap.pcd').write_text('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 2\nDATA ascii\n0 0 0\n4 5 0\n')
            config,_=algorithm_config(mode='localization',map_directory=path,pose=[0.]*6)
            self.assertTrue(config['roll']['localizationMode'])
            self.assertEqual(config['roll']['initialGuess'],[0.]*6)
            self.assertEqual(config['roll']['saveKeyframeMapDirectory'],str(path/'keyframes'))
            coverage=config['roll']['localization_coverage']
            self.assertEqual(coverage['source'],'automatic_map_point_xy_envelope_1m_margin')
            self.assertEqual(coverage['regions_xy'],[[[-1.,-1.],[5.,-1.],[5.,6.],[-1.,6.]]])

    def test_replay_seed_tracks_bag_contents_map_and_start(self):
        from perception.mapping_localization.results import file_fingerprint, map_fingerprint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'COMPLETE').touch()
            (root / 'keyframes').mkdir()
            (root / 'keyframes/poses.txt').write_text('0 0 0 0 0 0 0 0 1\n')
            (root / 'manifest.yaml').write_text('frame: map\n')
            (root / 'GlobalMap.pcd').write_text(
                'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 2\nDATA ascii\n0 0 0\n4 5 0\n')
            bag = root / 'A.bag'
            bag.write_bytes(b'synthetic A')
            seed = [1., 2., 3., 0., 0., 0.]
            config, _ = algorithm_config(mode='localization', map_directory=root, pose=seed)
            playback = dict(start_offset_s=3., bag_fingerprint=file_fingerprint(bag), map_fingerprint=map_fingerprint(root))
            run = dict(bag=str(bag), start=None, playback=playback, algorithm_parameters=config)
            record = root / 'run.json'
            record.write_text(json.dumps(run))
            options = dict(mode='localization', source_run=root)
            moved = root / 'copy.bag'
            moved.write_bytes(bag.read_bytes())
            for source in (bag, moved):
                cfg, _ = algorithm_config(**options, bag=source)
                self.assertEqual(cfg['roll']['initialGuess'], seed)
            with self.assertRaisesRegex(ValueError, 'start'):
                algorithm_config(**options, bag=bag, start=0.)
            moved.write_bytes(b'synthetic B')
            with self.assertRaisesRegex(ValueError, 'bag'):
                algorithm_config(**options, bag=moved)
            replacement, _ = algorithm_config(**options, bag=moved, pose=[0.] * 6)
            self.assertEqual(replacement['roll']['initialGuess'], [0.] * 6)
            (root / 'manifest.yaml').write_text('frame: changed_map\n')
            with self.assertRaisesRegex(ValueError, 'map'):
                algorithm_config(**options, bag=bag)
            # Old records with only a path still replay at their original offset.
            del run['playback']
            run['start'] = 3.
            record.write_text(json.dumps(run))
            self.assertEqual(algorithm_config(**options)[0]['roll']['initialGuess'], seed)
            with self.assertRaisesRegex(ValueError, 'bag'):
                algorithm_config(**options, bag=moved)

    def test_source_parameters_remain_bound_unless_calibration_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);run=path/'run.json'
            run.write_text(json.dumps(dict(algorithm_parameters=dict(common={'imu_time_offset':.12},mapping={'extrinsic_T':[1,2,3]}))))
            original=run.read_bytes()
            config,_=algorithm_config(mode='mapping',source_run=path)
            self.assertEqual(config['common']['imu_time_offset'],.12)
            override=path/'calibration.yaml';override.write_text('common:\n  imu_time_offset: 0.03\ncalibration:\n  status: explicit_test\n')
            changed,_=algorithm_config(mode='mapping',source_run=path,calibration=override)
            self.assertEqual(changed['common']['imu_time_offset'],.03)
            self.assertEqual(changed['mapping']['extrinsic_T'],[1,2,3])
            self.assertEqual(config['common']['imu_time_offset'],.12)
            self.assertEqual(run.read_bytes(),original)
            gpu,_=algorithm_config(mode='mapping',source_run=path,lio_backend='cuda')
            self.assertEqual(gpu['matching_backend'],'cuda')
            self.assertEqual(config['matching_backend'],'cuda')
            self.assertEqual(run.read_bytes(),original)
            accelerated,_=algorithm_config(mode='mapping',source_run=path,lio_backend='cuda',roll_backend='cuda',deskew_backend='cuda',linearization_backend='cuda')
            self.assertEqual(accelerated['roll']['matchingBackend'],'cuda')
            self.assertEqual(accelerated['roll']['matchingCudaCellSize'],.5)
            self.assertEqual(accelerated['roll']['matchingCudaThreads'],1)
            self.assertEqual(accelerated['deskew_backend'],'cuda')
            self.assertEqual(accelerated['linearization_backend'],'cuda')
            self.assertEqual(run.read_bytes(),original)
            for field in ('roll_backend','deskew_backend','linearization_backend'):
                with self.assertRaises(ValueError):algorithm_config(mode='mapping',**{field:'silent_fallback'})
            with self.assertRaises(ValueError):algorithm_config(mode='mapping',lio_backend='silent_fallback')

    def test_current_backends_override_old_execution_only_and_allow_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);run=path/'run.json'
            parameters=dict(matching_backend='cpu',deskew_backend='cpu',linearization_backend='cpu',
                mapping={'extrinsic_T':[1,2,3]},roll={'matchingBackend':'cpu','matchingThreads':4,'initialGuess':[1.]*6})
            run.write_text(json.dumps(dict(algorithm_parameters=parameters)));original=run.read_bytes()
            current,_=algorithm_config(mode='mapping',source_run=path)
            cpu,_=algorithm_config(mode='mapping',source_run=path,lio_backend='cpu',roll_backend='cpu',deskew_backend='cpu')
            for config,backend in ((current,'cuda'),(cpu,'cpu')):
                self.assertEqual(config['matching_backend'],backend)
                self.assertEqual(config['deskew_backend'],backend)
                self.assertEqual(config['roll']['matchingBackend'],backend)
                self.assertEqual(config['mapping'],parameters['mapping'])
                self.assertEqual(config['roll']['matchingThreads'],4)
                self.assertEqual(config['roll']['initialGuess'],[1.]*6)
                self.assertEqual(config['linearization_backend'],'cpu')
            self.assertEqual(run.read_bytes(),original)

    def test_ros_diagnostic_explicitly_selects_supported_cpu_numerics(self):
        launch=ET.parse(ROOT/'perception/mapping_localization/launch/native.launch').getroot()
        params={item.attrib['name']:item.attrib.get('value') for item in launch.findall('param')}
        for name in ('matching_backend','deskew_backend','linearization_backend','roll/matchingBackend'):
            self.assertEqual(params[name],'cpu')


class RawImageTests(unittest.TestCase):
    def test_shared_decoder_handles_color_order_and_padded_rows(self):
        for encoding,values in [('rgb8',[3,2,1]),('bgr8',[1,2,3]),('rgba8',[3,2,1,90]),('bgra8',[1,2,3,90])]:
            with self.subTest(encoding=encoding):
                row=bytes(values+[0,0]);image=SimpleNamespace(encoding=encoding,width=1,height=2,step=len(row),data=row*2)
                pixels=rgb_array(image)
                np.testing.assert_array_equal(pixels,np.array([[[3,2,1]],[[3,2,1]]],np.uint8))
                self.assertTrue(pixels.flags.c_contiguous)
                image.data=image.data[:-1]
                with self.assertRaises(ValueError):rgb_array(image)


if __name__=='__main__':unittest.main()
