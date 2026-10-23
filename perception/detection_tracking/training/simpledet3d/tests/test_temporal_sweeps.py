import tempfile
import unittest
from pathlib import Path

import numpy as np

from simple_det3d.datasets.coda_dataset import CodaDataset
from tools.generate_coda_info import build_sweep_infos
from tools.generate_wheelchair_info import process_category


class TemporalSweepTest(unittest.TestCase):
    def test_loader_selects_nearest_valid_sweeps(self):
        dataset = CodaDataset.__new__(CodaDataset)
        dataset.dataset_cfg = {
            'MAX_SWEEP_TIME_LAG': 0.35,
            'SWEEP_SELECT_MODE': 'near',
        }
        identity = np.eye(4, dtype=np.float32)
        info = {
            'sweeps': [
                {'time_lag': 986.0, 'transform_matrix': identity},
                {'time_lag': 0.30, 'transform_matrix': identity},
                {'time_lag': 0.10, 'transform_matrix': identity},
                {'time_lag': 0.20, 'transform_matrix': identity},
                {'time_lag': -0.10, 'transform_matrix': identity},
                {'time_lag': 0.05, 'transform_matrix': np.zeros((3, 3))},
            ]
        }

        selected = dataset._select_sweeps(info, max_sweeps=4)

        self.assertEqual([x['time_lag'] for x in selected], [0.30, 0.20, 0.10])

    def test_loader_removes_ego_points_from_current_and_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            current_path = root / 'current.bin'
            history_path = root / 'history.bin'
            np.array([
                [0.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 2.0],
            ], dtype=np.float32).tofile(current_path)
            np.array([
                [0.0, 0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0, 4.0],
            ], dtype=np.float32).tofile(history_path)

            dataset = CodaDataset.__new__(CodaDataset)
            dataset.root_path = root
            dataset.raw_data_path = root
            dataset.dataset_cfg = {
                'MAX_SWEEP_TIME_LAG': 0.35,
                'SWEEP_SELECT_MODE': 'near',
            }
            dataset.infos = [{
                'lidar_path': 'current.bin',
                'sweeps': [{
                    'lidar_path': 'history.bin',
                    'time_lag': 0.1,
                    'transform_matrix': np.eye(4, dtype=np.float32),
                }],
            }]

            points = dataset.get_lidar_with_sweeps(0, max_sweeps=2)

            np.testing.assert_allclose(points[:, 0], [2.0, 3.0])
            np.testing.assert_allclose(points[:, 4], [0.0, 0.1])

    def test_motion_prepared_history_transforms_and_missing_pose_falls_back_to_current(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            np.array([[2., 0., 0., 255.]], dtype=np.float32).tofile(root / 'current.bin')
            np.array([[3., 0., 0., 128.]], dtype=np.float32).tofile(root / 'history.bin')
            transform = np.array([[0., -1., 0., 2.], [1., 0., 0., 1.],
                                  [0., 0., 1., 0.], [0., 0., 0., 1.]])
            dataset = CodaDataset.__new__(CodaDataset)
            dataset.root_path = dataset.raw_data_path = root
            dataset.dataset_cfg = {'MAX_SWEEP_TIME_LAG': .35, 'SWEEP_SELECT_MODE': 'near',
                                   'INTENSITY_NORMALIZATION': 255.}
            dataset.infos = [dict(lidar_path='current.bin', raw_history=[{}], has_pose=True,
                                  motion_compensation_prepared=True,
                                  sweeps=[dict(lidar_path='history.bin', time_lag=.1,
                                               transform_matrix=transform)]),
                             dict(lidar_path='current.bin', raw_history=[{}], has_pose=False,
                                  motion_compensation_prepared=True, sweeps=[])]
            points = dataset.get_lidar_with_sweeps(0, max_sweeps=2)
            np.testing.assert_allclose(points[:, :3], [[2., 0., 0.], [2., 4., 0.]])
            np.testing.assert_allclose(points[:, 3:], [[1., 0.], [128. / 255., .1]])
            fallback = dataset.get_lidar_with_sweeps(1, max_sweeps=2)
            np.testing.assert_allclose(fallback, [[2., 0., 0., 1., 0.]])
            dataset.infos[1]['motion_compensation_prepared'] = False
            with self.assertRaisesRegex(ValueError, '缺少运动补偿位姿'):
                dataset.get_lidar_with_sweeps(1, max_sweeps=2)

    def test_coda_generator_uses_consecutive_frames_and_time_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            lidar_paths = {}
            for frame_id in range(5):
                path = root / f'frame_{frame_id}.bin'
                path.touch()
                lidar_paths[frame_id] = path

            poses = np.zeros((5, 8), dtype=np.float64)
            poses[:, 0] = np.arange(5) * 0.1
            poses[:, 4] = 1.0

            sweeps = build_sweep_infos(
                frame_id=4,
                current_lidar2global=np.eye(4),
                current_timestamp=0.4,
                lidar_paths=lidar_paths,
                poses=poses,
                lidar2base=np.eye(4),
                data_root=root,
                sweep_frame_len=3,
                max_sweep_time_lag=0.25,
            )

            self.assertEqual([x['frame_id'] for x in sweeps], [2, 3])
            self.assertTrue(all(0 < x['time_lag'] <= 0.25 for x in sweeps))

    def test_wheelchair_generator_drops_history_at_frame_gap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            category_dir = root / 'indoor1'
            scene_dir = category_dir / 'scene_a'
            points_dir = scene_dir / 'points'
            poses_dir = scene_dir / 'poses'
            points_dir.mkdir(parents=True)
            poses_dir.mkdir()

            for frame_id in [0, 1, 20]:
                np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32).tofile(
                    points_dir / f'{frame_id:04d}.bin'
                )
                (poses_dir / f'{frame_id:04d}.txt').write_text(
                    f'pose{frame_id}: 0 0 0 0 0 0 1\n'
                )

            train_infos, _ = process_category(
                category_dir=category_dir,
                category_name='indoor1',
                output_dir=root,
                sweep_frame_len=3,
                max_sweep_time_lag=0.35,
                val_scenes=set(),
            )

            self.assertEqual([x['frame_id'] for x in train_infos[1]['sweeps']], [0])
            self.assertEqual(train_infos[2]['sweeps'], [])

    def test_wheelchair_infos_relocate_current_and_history_with_the_raw_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            category_dir = root / 'original/indoor2'
            points_dir = category_dir / 'scene_a/points'
            poses_dir = category_dir / 'scene_a/poses'
            points_dir.mkdir(parents=True)
            poses_dir.mkdir()
            for frame_id in (0, 1):
                np.array([[2.0 + frame_id, 0.0, 0.0, 1.0]], dtype=np.float32).tofile(
                    points_dir / f'{frame_id:04d}.bin')
                (poses_dir / f'{frame_id:04d}.txt').write_text(
                    f'pose{frame_id}: 0 0 0 0 0 0 1\n')
            # A shared asset may itself be a link outside the raw root. Store
            # its logical dataset path so relocation still uses RAW_DATA_PATH.
            (points_dir / '0000.bin').rename(root / 'shared_history.bin')
            (points_dir / '0000.bin').symlink_to(root / 'shared_history.bin')
            infos, _ = process_category(category_dir, 'indoor2', root, 1, 0.2, set())
            (root / 'original').rename(root / 'relocated')
            dataset = CodaDataset.__new__(CodaDataset)
            dataset.root_path = root
            dataset.raw_data_path = root / 'relocated'
            dataset.dataset_cfg = {'MAX_SWEEP_TIME_LAG': 0.2, 'SWEEP_SELECT_MODE': 'near'}
            dataset.infos = infos
            points = dataset.get_lidar_with_sweeps(1, max_sweeps=2)
            np.testing.assert_allclose(points[:, 0], [3.0, 2.0])
            np.testing.assert_allclose(points[:, 4], [0.0, 0.1])
            self.assertTrue(infos[1]['has_pose'])  # A measured identity pose remains valid.
            self.assertEqual(infos[1]['gt_boxes'].shape, (0, 7))

    def test_stationary_wheelchair_scene_uses_identity_sweep_without_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            category_dir = root / 'indoor1'
            points_dir = category_dir / 'scene_static' / 'points'
            points_dir.mkdir(parents=True)
            for frame_id in [0, 1]:
                np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32).tofile(
                    points_dir / f'{frame_id:04d}.bin'
                )

            train_infos, _ = process_category(
                category_dir=category_dir,
                category_name='indoor1',
                output_dir=root,
                sweep_frame_len=1,
                max_sweep_time_lag=0.2,
                val_scenes=set(),
            )

            self.assertFalse(train_infos[1]['has_pose'])
            self.assertEqual(len(train_infos[1]['sweeps']), 1)
            np.testing.assert_allclose(
                train_infos[1]['sweeps'][0]['transform_matrix'], np.eye(4)
            )


if __name__ == '__main__':
    unittest.main()
