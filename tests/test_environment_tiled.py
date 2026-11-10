"""Static tiled fusion must preserve global evidence and tile-boundary behavior."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

import numpy as np

from perception.environment.fusion import fuse_scans, fuse_scans_tiled
from perception.environment.layers import load_config
from perception.environment.maps import static_geometry
from perception.environment.occupancy import OccupancyTree
from perception.environment.poses import PoseTimeline
from perception.environment.scans import ScanJournal
from tests.test_environment_dynamic import observation


class TiledFusionTests(unittest.TestCase):
    def test_cli_resolution_only_changes_export_configuration(self):
        from perception.environment.__main__ import bundle_main
        with patch('perception.environment.export.export_environment', return_value={'snapshot_id': 'test'}) as export:
            with redirect_stdout(io.StringIO()):
                bundle_main(['export', '--run', 'source', '--out', 'new', '--resolution', '.05'])
        self.assertEqual(export.call_args.kwargs['config']['resolution'], .05)
        self.assertEqual(load_config()['resolution'], .1)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bundle_main(['export', '--run', 'source', '--out', 'new', '--resolution', 'nan'])

    def test_static_retention_matches_expiry_path_and_checks_assumptions(self):
        config = load_config()
        config['voxel_ttl_s'] = config['obstacle_ttl_s'] = 100.
        bounds = [-1, -1, -1, .94, 1, 1]
        regular, fast = OccupancyTree(.1), OccupancyTree(.1)
        self.addCleanup(regular.close); self.addCleanup(fast.close)
        if not fast.enable_static_retention():
            self.skipTest('Native library does not have the optional static retention optimization')
        for tree in (regular, fast):tree.enable_static_review(.5, .2)
        for stamp, points in [(1., [[.93, 0, .25], [.35, 0, .25]]),
                              (1.6, [[.35, 0, .25]]), (2.2, [[.93, 0, .25]])]:
            snapshots = [tree.update(points, [0, 0, .25], bounds, stamp, config) for tree in (regular, fast)]
            def ordered(a):return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
            np.testing.assert_allclose(ordered(snapshots[0]), ordered(snapshots[1]), rtol=0, atol=0)
            np.testing.assert_array_equal(regular.static_votes(points), fast.static_votes(points))
        with self.assertRaisesRegex(RuntimeError, 'fixed bounds'):
            fast.update([[.2, 0, .25]], [0, 0, .25], [-1, -1, -1, 1, 1, 1], 2.3, config)

    def test_static_ray_rejection_keeps_all_intersecting_voxels(self):
        config = load_config()
        config['voxel_ttl_s'] = config['obstacle_ttl_s'] = 100.
        config['max_rays'] = 10000
        bounds = [-1, -1, -.5, 1, 1, 1.]
        regular, fast = OccupancyTree(.1), OccupancyTree(.1)
        self.addCleanup(regular.close); self.addCleanup(fast.close)
        if not fast.enable_static_retention():self.skipTest('Optional static retention optimization unavailable')
        for tree in (regular, fast):tree.enable_static_review(.5, .2)
        rng = np.random.default_rng(11)
        for stamp in (1., 1.6, 2.2):
            origins = rng.uniform(-3, 3, (400, 3)).astype(np.float32)
            points = rng.uniform(-3, 3, (400, 3)).astype(np.float32)
            points[:4] = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]
            origins[:4] = [[-2, -2, 0], [-2, 2, 0], [2, -2, 0], [2, 2, 0]]
            snapshots = [tree.update(points, [0, 0, 0], bounds, stamp, config, ray_origins=origins)
                         for tree in (regular, fast)]
            def ordered(a):return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]
            np.testing.assert_allclose(ordered(snapshots[0]), ordered(snapshots[1]), rtol=0, atol=0)
            np.testing.assert_array_equal(regular.static_votes(points), fast.static_votes(points))

    def test_tile_halo_preserves_float_boundary_evidence(self):
        config = load_config()
        config['static']['tile_size_m'] = 2.
        origin = np.array([-34.6000022888, -86.8000030518])
        axis = np.arange(0, 4, .1)
        x, y = np.meshgrid(axis, axis)
        floor = np.c_[origin[0]+x.ravel(), origin[1]+y.ravel(), np.full(x.size, -1.)]
        rng = np.random.default_rng(2)
        wall = np.c_[origin[0]+np.repeat([1., 2., 3.], 50)+rng.choice([-1e-6, 0, 1e-6], 150),
                     origin[1]+rng.uniform(.1, 3.8, 150), rng.uniform(-.8, .7, 150)]
        points = np.r_[floor, wall].astype(np.float32)
        sensor = np.r_[origin+2., 0.]
        candidate = static_geometry(points, [sensor], config, map_id='map', frame='map', stamp=1.)
        times = (1., 1.6, 2.2)
        rows = [dict(stamp_s=t, stamp_ns=int(t*1e9), position=sensor.tolist(), quaternion_xyzw=[0, 0, 0, 1],
                     session_id='local', generation=0, frame_id='odom', health=0) for t in times]
        timeline = PoseTimeline(rows, config['poses'], [[1., *sensor, 0, 0, 0, 0, 0]])
        transform = np.eye(4); transform[:3, 3] = sensor
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scans.sqlite'
            writer = ScanJournal(path, config['scan_recording'], {})
            meta = dict(session_id='local', generation=0, frame_id='odom', rays_used=True,
                        ray_origin_model='pose_interpolated_origins')
            for stamp in times:
                writer.append(points-sensor, transform, int(stamp*1e9), meta, np.tile(sensor, (len(points), 1)))
            writer.close()
            reference = fuse_scans(candidate, path, timeline, config, {}, filter_mode='geometric', candidate_points=points)
            result = fuse_scans_tiled(candidate, path, timeline, config, {}, filter_mode='geometric', candidate_points=points)
            for key in reference.layers:
                with self.subTest(layer=key):
                    np.testing.assert_allclose(result.layers[key], reference.layers[key], rtol=0, atol=1e-6, equal_nan=True)

    def test_global_and_tiled_evidence_match_with_people_and_adaptive_split(self):
        config = load_config()
        axis = np.arange(-2.5, 2.5, .1)
        x, y = np.meshgrid(axis, axis)
        ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
        person = np.array([[1.45, -.05, -.5], [1.5, 0, 0], [1.55, .05, .3]])
        wall = np.c_[np.full(30, 2.), np.linspace(-1, 1, 30), np.linspace(-.8, .7, 30)]
        points = np.r_[ground, person, wall].astype(np.float32)
        candidate = static_geometry(points, [[0, 0, 0]], config, map_id='map', frame='map', stamp=1.)
        times = (1., 1.6, 2.2, 2.8)
        rows = [dict(stamp_s=t, stamp_ns=int(t*1e9), position=[0, 0, 0],
                     quaternion_xyzw=[0, 0, 0, 1], session_id='local', generation=0, frame_id='odom', health=0) for t in times]
        timeline = PoseTimeline(rows, config['poses'], [[1., 0, 0, 0, 0, 0, 0, 0, 0]])

        class People:
            reference = dict(people_sha256='synthetic')
            manifest = dict(clock_domain='synthetic')
            def at(self, stamp): return observation(stamp)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scans.sqlite'
            writer = ScanJournal(path, config['scan_recording'], {})
            meta = dict(session_id='local', generation=0, frame_id='odom', rays_used=True,
                        ray_origin_model='pose_interpolated_origins')
            for stamp in times:
                writer.append(points, np.eye(4), int(stamp*1e9), meta, np.zeros_like(points))
            writer.close()
            plain = fuse_scans(candidate, path, timeline, config, {}, filter_mode='none', candidate_points=points)
            plain_config = dict(config, static=dict(config['static'], tile_size_m=2.))
            tiled_plain = fuse_scans_tiled(candidate, path, timeline, plain_config, {},
                                           filter_mode='none', candidate_points=points)
            for key in plain.layers:
                with self.subTest(filter_mode='none', layer=key):
                    np.testing.assert_allclose(tiled_plain.layers[key], plain.layers[key], rtol=0, atol=1e-6, equal_nan=True)
            for people in (None, People()):
                reference = fuse_scans(candidate, path, timeline, config, {}, people=people,
                                       filter_mode='geometric', candidate_points=points)
                tiled_config = dict(config, static=dict(config['static'], tile_size_m=2., max_voxels=120))
                result = fuse_scans_tiled(candidate, path, timeline, tiled_config, {}, people=people,
                                          filter_mode='geometric', candidate_points=points)
                for key in reference.layers:
                    with self.subTest(people=people is not None, layer=key):
                        np.testing.assert_allclose(result.layers[key], reference.layers[key], rtol=0, atol=1e-6, equal_nan=True)
                self.assertGreater(result.metadata['fusion']['adaptive_splits'], 0)
                for key in ('evidence_scans', 'unbound_scans', 'ray_scans'):
                    self.assertEqual(result.metadata[key], reference.metadata[key])
                self.assertEqual(result.metadata['dynamic_filter']['counts'], reference.metadata['dynamic_filter']['counts'])
                self.assertEqual(result.metadata['dynamic_filter']['geometric_rejected_endpoints'],
                                 reference.metadata['dynamic_filter']['geometric_rejected_endpoints'])
                single_config = dict(config, static=dict(config['static'], tile_size_m=20., max_voxels=120))
                split = fuse_scans_tiled(candidate, path, timeline, single_config, {}, people=people,
                                        filter_mode='geometric', candidate_points=points)
                self.assertEqual(split.metadata['fusion']['initial_tiles'], 1)
                self.assertGreater(split.metadata['fusion']['adaptive_splits'], 0)
                for key in reference.layers:
                    with self.subTest(single_tile=True, people=people is not None, layer=key):
                        np.testing.assert_allclose(split.layers[key], reference.layers[key], rtol=0, atol=1e-6, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
