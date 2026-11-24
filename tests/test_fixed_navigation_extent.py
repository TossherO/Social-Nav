"""Fixed navigation windows derived from immutable static map packages."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from decision.agent.tools.spatial import display_extent
from perception.environment.layers import GridSnapshot
from perception.environment.storage import StaticEnvironmentMap, content_hash


class FixedNavigationExtentTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.path = Path(workspace.name)

    def save_map(self, *, known=True, origin=(-4.3, 7.3), resolution=.3, invalid=False):
        cells = np.full((28, 36), -1, np.int8)
        if known:
            cells[8, 10] = 0
            cells[9, 14] = 100
        if invalid:
            cells[0, 0] = 50
        metadata = dict(snapshot_id='fixed:source', frame_id='map', stamp_s=5., origin_xy=list(origin),
                        resolution=resolution, shape=list(cells.shape), map_type='static_environment_map')
        grid = GridSnapshot(metadata, dict(occupancy=cells, ground_z=np.arange(cells.size).reshape(cells.shape)))
        grid.save(self.path/'layers.npz')
        manifest = dict(kind='static_environment', time_scope='offline_final_prior', snapshot_id=metadata['snapshot_id'],
                        frame_id='map', layers_sha256=content_hash(self.path/'layers.npz'))
        (self.path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
        return grid, manifest

    def test_only_known_occupancy_sets_default_extent_and_view_preserves_source(self):
        grid, manifest = self.save_map()
        manifest_hash = content_hash(self.path/'manifest.yaml')
        asset = StaticEnvironmentMap(self.path)
        view = asset.navigation_snapshot
        self.assertIs(view, asset.navigation_snapshot)
        self.assertEqual(asset.navigation_window, (6, 4, 19, 14))
        self.assertIs(asset.get_snapshot(manifest['snapshot_id']), asset.snapshot)
        self.assertEqual(asset.snapshot.metadata, grid.metadata)
        self.assertEqual(asset.manifest, manifest)
        self.assertEqual(content_hash(self.path/'layers.npz'), manifest['layers_sha256'])
        self.assertEqual(content_hash(self.path/'manifest.yaml'), manifest_hash)
        self.assertNotEqual(view.metadata['snapshot_id'], manifest['snapshot_id'])
        self.assertEqual(view.metadata['source_snapshot_id'], manifest['snapshot_id'])
        self.assertEqual(view.metadata['source_layers_sha256'], manifest['layers_sha256'])
        self.assertEqual(view.metadata['source_window_xyxy'], [6, 4, 19, 14])
        self.assertEqual(view.metadata['shape'], [10, 13])
        np.testing.assert_allclose(view.metadata['origin_xy'], [-2.5, 8.5])
        for name, layer in view.layers.items():
            np.testing.assert_array_equal(layer, grid.layers[name][4:14, 6:19])
            self.assertTrue(np.shares_memory(layer, asset.snapshot.layers[name]))
            self.assertFalse(layer.flags.writeable)

    def test_marks_expand_source_window_but_live_and_off_map_positions_keep_agent_behavior(self):
        grid, _ = self.save_map()
        annotations = dict(points=[dict(place_id='mission_start', pose=[50., 50., 0.]),
                                   dict(place_id='desk', pose=[2.4, 9.7, 0.])],
                           regions=[dict(polygon=[[-3.6, 10.3], [-3.4, 10.3], [-3.6, 10.7]])])
        asset = StaticEnvironmentMap(self.path, navigation_annotations=annotations)
        self.assertEqual(asset.navigation_window, (0, 4, 26, 15))
        annotations['points'].append(dict(place_id='off_map', pose=[20., 10.]))
        self.assertEqual(asset.navigation_window, (0, 4, 26, 15))
        clipped = StaticEnvironmentMap(self.path, navigation_annotations=annotations)
        self.assertEqual(clipped.navigation_window, (0, 4, 36, 15))
        display = display_extent(grid, annotations, extras=[(10., 25.)])
        self.assertTrue(display['annotation_expanded'])
        self.assertTrue(display['live_positions_expanded'])
        np.testing.assert_allclose(display['bounds'], [-4.6, 8.7, 21., 26.])

    def test_worker_uses_same_view_and_rejects_bad_windows(self):
        self.save_map()
        original = StaticEnvironmentMap(self.path, navigation_window=(0, 0, 21, 15))
        worker = StaticEnvironmentMap(self.path, navigation_window=list(original.navigation_window))
        self.assertEqual(worker.navigation_snapshot.metadata, original.navigation_snapshot.metadata)
        self.assertNotEqual(worker.navigation_snapshot.metadata['snapshot_id'],
                            StaticEnvironmentMap(self.path, navigation_window=(1, 1, 22, 16))
                            .navigation_snapshot.metadata['snapshot_id'])
        for window in [(0, 0, 37, 28), (6, 4, 19, 13), (6, 4, 6, 14), (False, 0, 36, 28),
                       (0, 0, 36., 28)]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                StaticEnvironmentMap(self.path, navigation_window=window)

    def test_unknown_map_keeps_its_extent_even_with_off_map_annotation(self):
        self.save_map(known=False, origin=(-4.7, -5.1), resolution=.1)
        asset = StaticEnvironmentMap(self.path, navigation_annotations=dict(points=[dict(pose=[20., 10.])]))
        self.assertEqual(asset.navigation_window, (0, 0, 36, 28))
        self.assertEqual(asset.navigation_snapshot.metadata['origin_xy'], asset.snapshot.metadata['origin_xy'])

    def test_decimal_grid_lines_do_not_add_a_cell_to_one_metre_padding(self):
        self.save_map(origin=(-.6, -.8), resolution=.1)
        asset = StaticEnvironmentMap(self.path)
        self.assertEqual(asset.navigation_window, (0, 0, 25, 20))

    def test_invalid_occupancy_cannot_be_hidden_by_cropping(self):
        self.save_map(invalid=True)
        asset = StaticEnvironmentMap(self.path)
        with self.assertRaisesRegex(ValueError, 'Invalid static map occupancy'):
            asset.navigation_snapshot


if __name__ == '__main__':
    unittest.main()
