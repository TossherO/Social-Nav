import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.downsample_coda_lidar import (
    downsample_file,
    downsample_points,
    select_channel_indices,
)
from tools.generate_coda_info import generate_coda_infos


class CodaLidarDownsampleTest(unittest.TestCase):
    def test_selects_sixteen_channels_across_full_vertical_fov(self):
        indices = select_channel_indices(128, 16)

        np.testing.assert_array_equal(
            indices,
            [0, 8, 17, 25, 34, 42, 51, 59,
             68, 76, 85, 93, 102, 110, 119, 127],
        )

    def test_downsamples_by_channel_instead_of_flat_point_stride(self):
        points = np.zeros((128, 4, 4), dtype=np.float32)
        points[:, :, 0] = np.arange(128, dtype=np.float32)[:, None]

        sampled = downsample_points(
            points.reshape(-1, 4),
            input_channels=128,
            output_channels=16,
            columns=4,
        ).reshape(16, 4, 4)

        np.testing.assert_array_equal(
            sampled[:, 0, 0], select_channel_indices(128, 16),
        )
        self.assertEqual(sampled.shape, (16, 4, 4))

    def test_file_downsampling_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / 'input.bin'
            output_path = root / 'output.bin'
            points = np.arange(128 * 4 * 4, dtype=np.float32).reshape(-1, 4)
            points.tofile(input_path)

            first_status = downsample_file(
                input_path, output_path,
                input_channels=128, output_channels=16, columns=4,
            )
            second_status = downsample_file(
                input_path, output_path,
                input_channels=128, output_channels=16, columns=4,
            )

            self.assertEqual(first_status, 'written')
            self.assertEqual(second_status, 'skipped')
            self.assertEqual(
                output_path.stat().st_size,
                16 * 4 * 4 * np.dtype(np.float32).itemsize,
            )

    def test_info_generator_uses_downsampled_target_and_sweep_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            point_dir = root / '3d_comp_downsample/os1/0'
            annotation_dir = root / '3d_bbox/os1/0'
            calibration_dir = root / 'calibrations/0'
            pose_dir = root / 'poses/dense'
            metadata_dir = root / 'metadata'
            output_dir = root / 'infos'
            for path in (
                point_dir, annotation_dir, calibration_dir, pose_dir, metadata_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)

            for frame_id in (0, 1):
                np.array([[2.0, 0.0, 0.0, 1.0]], dtype=np.float32).tofile(
                    point_dir / f'3d_comp_os1_0_{frame_id}.bin'
                )

            annotation = {
                '3dbbox': [{
                    'classId': 'Pedestrian',
                    'cX': 2.0, 'cY': 0.0, 'cZ': 1.0,
                    'l': 0.5, 'w': 0.5, 'h': 1.8,
                    'r': 0.0, 'p': 0.0, 'y': 0.0,
                }],
            }
            (annotation_dir / '3d_bbox_os1_0_1.json').write_text(
                json.dumps(annotation)
            )
            (calibration_dir / 'calib_os1_to_base.yaml').write_text(
                'extrinsic_matrix:\n'
                '  rows: 4\n'
                '  cols: 4\n'
                '  data: [1, 0, 0, 0, 0, 1, 0, 0, '
                '0, 0, 1, 0, 0, 0, 0, 1]\n'
            )
            (pose_dir / '0.txt').write_text(
                '0.0 0 0 0 1 0 0 0\n'
                '0.1 0 0 0 1 0 0 0\n'
            )
            metadata = {
                'ObjectTracking': {
                    'training': ['3d_bbox/os1/0/3d_bbox_os1_0_1.json'],
                    'validation': [],
                    'testing': [],
                },
            }
            (metadata_dir / '0.json').write_text(json.dumps(metadata))

            with contextlib.redirect_stdout(io.StringIO()):
                generate_coda_infos(
                    data_root=root,
                    output_dir=output_dir,
                    sweep_frame_len=1,
                    seq_ids=[0],
                    pc_dir='3d_comp_downsample/os1',
                    max_sweep_time_lag=0.2,
                )

            with (output_dir / 'coda_infos_train.pkl').open('rb') as file:
                infos = pickle.load(file)
            self.assertEqual(
                infos[0]['lidar_path'],
                '3d_comp_downsample/os1/0/3d_comp_os1_0_1.bin',
            )
            self.assertEqual(
                infos[0]['sweeps'][0]['lidar_path'],
                '3d_comp_downsample/os1/0/3d_comp_os1_0_0.bin',
            )


if __name__ == '__main__':
    unittest.main()
