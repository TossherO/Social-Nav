"""Height coverage invariants, independent of CPU/CUDA agreement."""
import unittest
import numpy as np

from perception.environment.layers import load_config, project_evidence
from perception.environment.occupancy import OccupancyTree


def height_column(backend, *, dtype=np.float32, ground=-.87, shift=0, missing=(),
                  resolution=.1, keys=range(-8, 9), minimum=.12, maximum=1.8,
                  occupied=False, endpoint=False, supported=True, library=None):
    """Seed separate measured rays at each world voxel layer; inspect their middle."""
    config = load_config()
    config.update(resolution=resolution, min_obstacle_height_m=minimum,
                  max_height_m=maximum, voxel_ttl_s=120.)
    keys = np.asarray(keys)
    z = ((keys + shift + .5)*resolution).astype(np.float32)
    keep = ~np.isin(np.arange(len(z)), missing)
    points = np.c_[np.full(keep.sum(), 20.5*resolution),
                   np.full(keep.sum(), 1.5*resolution), z[keep]].astype(np.float32)
    origins = points.copy()
    origins[:, 0] = .5*resolution
    floor = np.full((3, 24), ground+shift*resolution, dtype)
    support = np.full(floor.shape, supported, bool)
    target = np.array([[10.5*resolution, 1.5*resolution, z[len(z)//2]]])
    tree = OccupancyTree(resolution, backend=backend, library=library)
    try:
        for stamp in np.arange(8)*.2+1:
            tree.update(points, [.5*resolution, 1.5*resolution, 0], [-4, -4, -4, 4, 4, 4],
                        float(stamp), config, ray_origins=origins, read=False)
        stamp = 2.4
        if occupied:
            stamp = 3.
            tree.update(target, [0, 0, 0], [-4, -4, -4, 4, 4, 4], stamp, config, rays=False, read=False)
        endpoints = target if endpoint else np.empty((0, 3))
        reference = project_evidence(tree.snapshot(stamp), floor, support, [0, 0], config, stamp,
                                     endpoints=endpoints)
        native = tree.project(floor, support, [0, 0], config, stamp, endpoints=endpoints)
        return reference, native
    finally:
        tree.close()


class HeightCoverageTests(unittest.TestCase):
    def test_complete_free_column_is_stable_at_bin_edges_and_world_shifts(self):
        for dtype in (np.float32, np.float64):
            for ground in (-.87, -.8701, -.87000001, -.86999999):
                for shift in (-1, 0, 1, 10):
                    with self.subTest(dtype=dtype, ground=ground, shift=shift):
                        for layers in height_column('cpu', dtype=dtype, ground=ground, shift=shift):
                            self.assertEqual(layers['coverage'][1, 10], 1.)
                            self.assertEqual(layers['occupancy'][1, 10], 0)

    def test_missing_layer_stays_unknown_including_either_end(self):
        for dtype in (np.float32, np.float64):
            for missing in ((0,), (8,), (16,), tuple(range(17))):
                with self.subTest(dtype=dtype, missing=missing):
                    for layers in height_column('cpu', dtype=dtype, missing=missing):
                        self.assertAlmostEqual(layers['coverage'][1, 10], (17-len(missing))/17, places=7)
                        self.assertEqual(layers['occupancy'][1, 10], -1)

    def test_both_band_end_centres_count_and_duplicates_cannot_fill_a_missing_layer(self):
        config = load_config()
        config.update(resolution=.15, min_obstacle_height_m=.15, max_height_m=.6)
        for dtype in (np.float32, np.float64):
            args = dict(dtype=dtype, resolution=.15, ground=.075, minimum=.15, maximum=.6, keys=range(1, 5))
            for layers in height_column('cpu', **args):
                self.assertEqual(layers['coverage'][1, 10], 1.)
                self.assertEqual(layers['occupancy'][1, 10], 0)
            for layers in height_column('cpu', missing=(3,), **args):
                self.assertEqual(layers['coverage'][1, 10], .75)
                self.assertEqual(layers['occupancy'][1, 10], -1)
            floor = np.full((3, 3), .075, dtype)
            z = np.array([.225, .375, .525, .525])
            voxels = np.c_[np.full(4, .225), np.full(4, .225), z, np.full(4, .2), np.zeros(4)]
            layers = project_evidence(voxels, floor, np.ones_like(floor, bool), [0, 0], config, 1.)
            self.assertEqual(layers['coverage'][1, 1], .75)
            self.assertEqual(layers['occupancy'][1, 1], -1)

    def test_occupied_evidence_and_ground_requirements_remain_effective(self):
        for dtype in (np.float32, np.float64):
            for mode in ('occupied', 'endpoint'):
                for layers in height_column('cpu', dtype=dtype, **{mode: True}):
                    self.assertEqual(layers['occupancy'][1, 10], 100)
            for layers in height_column('cpu', dtype=dtype, supported=False):
                self.assertEqual(layers['coverage'][1, 10], 1.)
                self.assertEqual(layers['occupancy'][1, 10], -1)
            for layers in height_column('cpu', dtype=dtype, ground=np.nan):
                self.assertEqual(layers['coverage'][1, 10], 0.)
                self.assertEqual(layers['occupancy'][1, 10], -1)

    def test_out_of_band_free_layers_do_not_replace_missing_required_evidence(self):
        # -0.85 and 0.95 are outside [-0.75, 0.93]; the middle required layer is absent.
        for dtype in (np.float32, np.float64):
            for layers in height_column('cpu', dtype=dtype, keys=range(-9, 10), missing=(9,)):
                self.assertAlmostEqual(layers['coverage'][1, 10], 16/17, places=7)
                self.assertEqual(layers['occupancy'][1, 10], -1)

    def test_band_with_no_voxel_centre_cannot_become_free(self):
        for layers in height_column('cpu', ground=0., minimum=.01, maximum=.02):
            self.assertEqual(layers['coverage'][1, 10], 0.)
            self.assertEqual(layers['occupancy'][1, 10], -1)


if __name__ == '__main__':
    unittest.main()
