"""Exercise manual edits without changing the source maps."""
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from perception.environment.export import _save_package
from perception.environment.layers import GridSnapshot, load_config
from perception.environment.projection import docker_projection
from perception.environment.storage import StaticEnvironmentMap
from perception.environment import manual_edit as static_edit
from perception.mapping_localization import manual_edit as roll_edit
from perception.visualization.edit_geometry import grid_polygon_mask


class ManualMapEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def static_map(self):
        config = load_config()
        cloud = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [.01, .01, 0], [.005, .005, 0],
                          [1, 1, 0], [1.01, 1, 0], [1, 1.01, 0], [1.01, 1.01, 0], [1.005, 1.005, 0]])
        snapshot = docker_projection(cloud, [[0, 0, 1]], config, map_id='test', frame='map', stamp=10.)
        source = self.root/'static'
        _save_package(snapshot, source, {'map_fingerprint': {'test': 'original'}}, config, 'map', 0, True, [])
        return source

    def roll_map(self):
        source = self.root/'roll'
        keys = source/'keyframes'
        keys.mkdir(parents=True)
        (source/'manifest.yaml').write_text(yaml.safe_dump({'format': 'roll_keyframes_v1', 'keyframes': 2,
                                                            'frame': 'map', 'visualization_resolution': .1}))
        (source/'COMPLETE').write_text('roll_keyframes_v1\n')
        np.savetxt(keys/'poses.txt', [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [2, 0, 0, 0, 0, 0, math.pi/2, 1, 0]])
        clouds = [np.array([[0, 0, 0, 1], [1, 1, .5, 2]], np.float32),
                  np.array([[1, 0, .5, 3], [2, 0, .5, 4]], np.float32)]
        for i, cloud in enumerate(clouds):
            roll_edit.write_xyzi(keys/f'surf{i}.pcd', cloud)
            roll_edit.write_xyzi(keys/f'corner{i}.pcd', np.empty((0, 4), np.float32))
        roll_edit.write_xyzi(source/'GlobalMap.pcd', clouds[0])
        return source

    def test_static_edit_preserves_source_and_assumption_semantics(self):
        source = self.static_map()
        before = StaticEnvironmentMap(source).snapshot
        binding = static_edit.source_binding(source)
        occupancy = before.layers['occupancy'].copy()
        occupied = tuple(np.argwhere(occupancy == 100)[0])
        free = tuple(np.argwhere(occupancy == 0)[0])
        occupancy[occupied], occupancy[free] = 0, -1
        output = self.root/'static_edited'
        static_edit.save_manual_map(source, output, occupancy, expected_binding=binding)
        asset = StaticEnvironmentMap(output)
        result = asset.snapshot
        self.assertEqual(asset.manifest['source']['map_fingerprint'], {'test': 'original'})
        self.assertEqual(before.layers['occupancy'][occupied], 100)
        self.assertTrue(result.layers['manual_edit'][occupied])
        self.assertTrue(result.layers['assumed_free'][occupied])
        self.assertTrue(np.isnan(result.layers['last_observed']).all())
        self.assertFalse(result.layers['observed'][occupied])
        self.assertEqual(result.layers['occupancy'][free], -1)
        self.assertFalse(result.layers['unknown_distance'].any())
        self.assertFalse(result.layers['frontier'].any())
        self.assertTrue((output/'map.pgm').is_file())
        y, x = occupied
        xy = np.asarray(result.metadata['origin_xy']) + (np.array([x, y])+.5)*result.metadata['resolution']
        self.assertEqual(asset.query_geometry([xy], frame='map')[0]['free_source'], 'manual_edit')
        with self.assertRaises(FileExistsError):
            static_edit.save_manual_map(source, output, occupancy, expected_binding=binding)
        with self.assertRaises(ValueError):
            static_edit.save_manual_map(source, source/'nested', occupancy, expected_binding=binding)

    def test_rotated_keyframes_remove_only_selected_height_and_keep_original(self):
        source = self.roll_map()
        binding = roll_edit.source_binding(source)
        selections = [dict(vertices_xy=[[-.1, .9], [.1, .9], [.1, 1.1], [-.1, 1.1]],
                           z_min_m=.4, z_max_m=.6)]
        output = self.root/'roll_edited'
        report = roll_edit.save_manual_map(source, output, selections, expected_binding=binding)
        self.assertEqual(report['removed_points'], {'corner': 0, 'surf': 1})
        self.assertEqual(len(roll_edit.read_xyzi(source/'keyframes/surf1.pcd')), 2)
        np.testing.assert_array_equal(roll_edit.read_xyzi(output/'keyframes/surf1.pcd'), [[2, 0, .5, 4]])
        self.assertNotEqual(roll_edit.source_binding(output)['map_fingerprint'], binding['map_fingerprint'])
        self.assertTrue((output/'manual_edit.json').exists())
        with self.assertRaises(FileExistsError):
            roll_edit.save_manual_map(source, output, selections, expected_binding=binding)
        all_first = [dict(vertices_xy=[[-1, -1], [2, -1], [2, 2], [-1, 2]], z_min_m=-1, z_max_m=1)]
        with self.assertRaisesRegex(ValueError, '清空表面关键帧'):
            roll_edit.save_manual_map(source, self.root/'invalid', all_first, expected_binding=binding)
        self.assertFalse((self.root/'invalid').exists())
        with (source/'keyframes/surf0.pcd').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, '源定位地图已改变'):
            roll_edit.save_manual_map(source, self.root/'stale', selections, expected_binding=binding)

    def test_manual_free_does_not_inherit_observed_evidence(self):
        from perception.environment.maps import static_geometry
        config = load_config()
        snapshot = static_geometry(np.array([[0, 0, 0], [.1, 0, .5], [.2, 0, .6]], np.float32),
                                   [[0, 0, 1]], config, map_id='test', frame='map', stamp=10.)
        layers = {name: value.copy() for name, value in snapshot.layers.items()}
        cells = layers['occupancy']
        y, x = 0, 0
        cells[y, x] = 100
        layers['observed'][y, x] = True
        layers['last_observed'][y, x] = 9.
        layers['coverage'][y, x] = 1.
        layers['conditional_obstacle'] = np.zeros(cells.shape, bool)
        layers['conditional_obstacle'][y, x] = True
        snapshot = GridSnapshot(dict(snapshot.metadata, conditional_obstacle_cells=1), layers)
        source = self.root/'observed'
        _save_package(snapshot, source, {'map_fingerprint': {'test': 'original'}}, config, 'map', 0, True, [])
        edited = cells.copy(); edited[y, x] = 0
        output = self.root/'observed_edited'
        static_edit.save_manual_map(source, output, edited, expected_binding=static_edit.source_binding(source))
        edited_map = StaticEnvironmentMap(output)
        layers = edited_map.snapshot.layers
        self.assertTrue(layers['assumed_free'][y, x])
        self.assertFalse(layers['observed'][y, x])
        self.assertTrue(np.isnan(layers['last_observed'][y, x]))
        self.assertEqual(layers['coverage'][y, x], 0.)
        self.assertEqual(edited_map.snapshot.metadata['conditional_obstacle_cells'], 0)
        self.assertEqual(edited_map.snapshot.metadata['manual_free_cells_total'], 1)
        self.assertEqual(edited_map.manual_free_cells, 1)
        np.testing.assert_array_equal(layers['ground_support'], snapshot.layers['ground_support'])
        np.testing.assert_allclose(layers['ground_z'], snapshot.layers['ground_z'], equal_nan=True)
        second_cells = layers['occupancy'].copy(); second_cells[y, x] = -1
        second = self.root/'observed_edited_again'
        static_edit.save_manual_map(output, second, second_cells, expected_binding=static_edit.source_binding(output))
        twice = StaticEnvironmentMap(second)
        self.assertEqual(twice.manual_free_cells, 0)
        self.assertEqual(twice.snapshot.metadata['manual_free_cells_total'], 0)
        self.assertEqual(twice.manifest['limitations'].count('manual_free_not_sensor_observation'), 1)

    def test_saved_manual_map_is_not_certified_as_observed_static_path(self):
        from perception.environment.layers import finish_evidence
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.inputs import load_config as planning_config
        from navigation.planning.planner import Planner
        from tests.navigation_fixtures import request

        config = load_config()
        shape = (60, 60)
        cells = np.zeros(shape, np.int8); cells[30, 30] = 100
        ground = np.zeros(shape, np.float32); nan = np.full(shape, np.nan)
        layers = finish_evidence(cells, np.ones(shape, bool), np.full(shape, 9.), np.ones(shape, np.float32),
                                 nan.copy(), nan.copy(), ground, np.ones(shape, bool), .1)
        layers['endpoint_count'] = np.zeros(shape, np.int32)
        layers['conditional_obstacle'] = cells == 100
        metadata = dict(snapshot_id='before', map_id='map-a', frame_id='map', stamp_s=10.,
                        shape=list(shape), origin_xy=[0., 0.], resolution=.1, geometry_status='STATIC_SCAN_OBSERVED',
                        conditional_obstacle_cells=1)
        source = self.root/'planning_source'
        _save_package(GridSnapshot(metadata, layers), source, {'map_fingerprint': {'test': 'original'}},
                      config, 'map', 0, True, [])
        edited = cells.copy(); edited[30, 30] = 0
        output = self.root/'planning_edited'
        static_edit.save_manual_map(source, output, edited, expected_binding=static_edit.source_binding(source))
        asset = StaticEnvironmentMap(output)
        settings = planning_config()
        if not Path(settings['native_library']).is_file():
            self.skipTest('Native global planner is not available')
        geometry = VehicleGeometry([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]], np.eye(4), source='synthetic')
        result = Planner(settings, geometry).query(request(map_snapshot_id=asset.manifest['snapshot_id']), asset=asset)
        self.assertEqual(result['status'], 'path_found', result)
        self.assertTrue(result['manual_free_present'])
        self.assertTrue(result['unobserved_free_assumed'])
        self.assertFalse(result['current_path_valid'])
        self.assertTrue(result['global_reference_only'])
        self.assertTrue(result['needs_observation'])
        self.assertEqual(asset.summary()['manual_free_cells'], 1)
        legacy_metadata = dict(asset.snapshot.metadata)
        legacy_metadata.pop('manual_free_cells_total')
        legacy = self.root/'legacy_edited'
        _save_package(GridSnapshot(legacy_metadata, asset.snapshot.layers), legacy,
                      asset.manifest['source'], config, 'map', 0, True, [])
        legacy_asset = StaticEnvironmentMap(legacy)
        self.assertEqual(legacy_asset.manual_free_cells, 1)
        legacy_result = Planner(settings, geometry).query(
            request(map_snapshot_id=legacy_asset.manifest['snapshot_id']), asset=legacy_asset)
        self.assertTrue(legacy_result['manual_free_present'])
        self.assertFalse(legacy_result['current_path_valid'])
        from perception.environment.navigation import static_support
        local = GridSnapshot(dict(shape=list(shape), resolution=.1, origin_xy=[0., 0.], stamp_s=10.), {})
        selected = np.zeros(shape, bool); selected[30, 30:32] = True
        supported, _ = static_support(local, asset.snapshot, np.eye(4), np.zeros(3),
                                      [.12, 1.8], selected, .15)
        np.testing.assert_array_equal(supported, [False, True])

    def test_polygon_uses_grid_cell_centers(self):
        mask = grid_polygon_mask((3, 4), [0, 0], 1, [[.1, .1], [1.9, .1], [1.9, 1.9], [.1, 1.9]])
        np.testing.assert_array_equal(mask, [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
