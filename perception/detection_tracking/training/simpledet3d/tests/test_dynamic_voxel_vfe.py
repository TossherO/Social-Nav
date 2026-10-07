import unittest

import torch

from simple_det3d.models.vfe import DynamicVoxelVFE, limit_voxels_per_sample


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class DynamicVoxelLimitTest(unittest.TestCase):
    def test_uncapped_fast_path_keeps_original_mapping(self):
        coords = torch.tensor([1, 2, 101], dtype=torch.long)
        inverse = torch.tensor([0, 0, 1, 2], dtype=torch.long)

        out_coords, out_inverse, point_keep = limit_voxels_per_sample(
            coords, inverse, scale_xyz=100, max_voxels=3,
        )

        self.assertIs(out_coords, coords)
        self.assertIs(out_inverse, inverse)
        self.assertIsNone(point_keep)

    def test_cap_is_per_sample_deterministic_and_remaps_points(self):
        coords = torch.tensor(
            [1, 2, 3, 4, 5, 101, 102], dtype=torch.long
        )
        inverse = torch.tensor(
            [0, 0, 1, 2, 3, 4, 5, 6, 6], dtype=torch.long
        )

        first = limit_voxels_per_sample(
            coords, inverse, scale_xyz=100, max_voxels=3,
        )
        second = limit_voxels_per_sample(
            coords, inverse, scale_xyz=100, max_voxels=3,
        )
        selected_coords, remapped_inverse, point_keep = first

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertEqual(selected_coords.numel(), 5)
        self.assertLessEqual(int((selected_coords < 100).sum()), 3)
        self.assertLessEqual(int((selected_coords >= 100).sum()), 3)
        self.assertTrue(torch.equal(
            selected_coords[remapped_inverse],
            coords[inverse][point_keep],
        ))

    def test_train_and_test_limits_follow_data_config(self):
        cfg = _AttrDict({
            'USE_NORM': True,
            'WITH_DISTANCE': False,
            'USE_ABSLOTE_XYZ': True,
            'NUM_FILTERS': [8],
            'MAX_POINTS_PER_VOXEL': 20,
        })
        vfe = DynamicVoxelVFE(
            cfg,
            num_point_features=5,
            voxel_size=[0.1, 0.1, 0.2],
            grid_size=[10, 10, 10],
            point_cloud_range=[0, 0, 0, 1, 1, 2],
            max_num_voxels={'train': 3, 'test': 5},
        )

        self.assertEqual(vfe._get_max_num_voxels(), 3)
        vfe.eval()
        self.assertEqual(vfe._get_max_num_voxels(), 5)
        vfe.max_num_voxels_mode = 'train'
        self.assertEqual(vfe._get_max_num_voxels(), 3)
        self.assertEqual(vfe.grid_size.device.type, 'cpu')
        self.assertNotIn('grid_size', vfe.state_dict())
        self.assertNotIn('voxel_size', vfe.state_dict())
        self.assertNotIn('point_cloud_range', vfe.state_dict())


if __name__ == '__main__':
    unittest.main()
