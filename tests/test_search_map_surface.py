"""Reference-grid colours and display heights without a ROS master."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from navigation.planning.runtime import encode_display_map
from perception.environment.layers import GridSnapshot
from perception.sensors.pointcloud import pointcloud_fields
from perception.visualization.surface import SearchMapSurface


def display_snapshot(grid, stamp=7.25):
    return encode_display_map(grid, np.eye(3), 1, dict(as_of=stamp, binding='reference',
        result=dict(status='unavailable', reason='test', result_binding=None, request={})))


class SearchMapSurfaceTests(unittest.TestCase):
    def setUp(self):
        ros = SimpleNamespace(Time=SimpleNamespace(from_sec=float))
        field = type('PointField', (SimpleNamespace,), dict(FLOAT32=7, UINT32=6))
        sensor = SimpleNamespace(PointField=field, PointCloud2=lambda: SimpleNamespace(
            header=SimpleNamespace(), is_bigendian=False))
        modules = patch.dict('sys.modules', {'rospy': ros, 'sensor_msgs.msg': sensor})
        modules.start()
        self.addCleanup(modules.stop)
        self.meta = dict(shape=[2, 5], resolution=.6, origin_xy=[3., -1.], frame_id='static_map',
                         snapshot_id='source', stamp_s=1.)
        self.prior = GridSnapshot(self.meta, dict(occupancy=np.array(
            [[100, 100, -1, -1, -1], [-1, -1, -1, 0, -1]], dtype=np.int8),
            ground_z=np.array([[2., np.nan, np.nan, np.nan, np.nan],
                               [np.nan, np.nan, np.nan, np.nan, 4.]], dtype=np.float32),
            ground_support=np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=bool),
            ground_estimated=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=bool)))

    def test_search_colours_override_static_classes_and_keep_cells_without_ground(self):
        surface = SearchMapSurface(SimpleNamespace(snapshot=self.prior))
        cells = np.array([[0, 0, -1, -1, 100], [-1, -1, 100, -1, 0]], dtype=np.int8)
        conditional = np.zeros(cells.shape, bool)
        conditional[0, 1] = True
        grid = GridSnapshot(dict(self.meta, snapshot_id='search'),
                            dict(occupancy=cells, conditional_obstacle=conditional))

        cloud = surface.cloud(display_snapshot(grid))
        self.assertEqual((cloud.header.frame_id, cloud.header.stamp), ('static_map', 7.25))
        self.assertEqual((cloud.height, cloud.width, cloud.point_step, cloud.row_step), (1, 5, 16, 80))
        fields = pointcloud_fields(cloud, ('x', 'y', 'z', 'rgb'))
        np.testing.assert_array_equal(fields['rgb'], [0xD7DDE3, 0xE7A343, 0xF05848, 0xF05848, 0xD7DDE3])
        np.testing.assert_allclose(fields['x'], [3.3, 3.9, 5.7, 4.5, 5.7])
        np.testing.assert_allclose(fields['y'], [-.7, -.7, -.7, -.1, -.1])
        np.testing.assert_allclose(fields['z'], [2.005, 2.005, 4.005, 3.005, 4.005])
        self.assertEqual(self.prior.layers['occupancy'][0, 0], 100)
        self.assertTrue(np.isnan(self.prior.layers['ground_z'][0, 1]))

    def test_groundless_asset_and_all_unknown_snapshot(self):
        cells = np.full((2, 5), -1, np.int8)
        grid = GridSnapshot(self.meta, dict(occupancy=cells))
        surface = SearchMapSurface(SimpleNamespace(snapshot=grid))
        empty = surface.cloud(display_snapshot(grid))
        self.assertEqual((empty.width, empty.row_step, empty.data), (0, 0, b''))

        cells[0, 0] = 100
        cloud = surface.cloud(display_snapshot(grid, stamp=8.))
        fields = pointcloud_fields(cloud, ('z', 'rgb'))
        np.testing.assert_allclose(fields['z'], [.005])
        np.testing.assert_array_equal(fields['rgb'], [0xF05848])
        self.assertEqual(cloud.header.stamp, 8.)
        self.assertEqual(empty.data, b'')

    def test_rejects_search_in_another_map_geometry(self):
        surface = SearchMapSurface(SimpleNamespace(snapshot=self.prior))
        packet = display_snapshot(self.prior)
        for field, value in (('shape', (5, 2)), ('frame_id', 'odom'), ('resolution', .1), ('origin_xy', [0., 0.])):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'grid geometry'):
                surface.cloud(dict(packet, **{field: value}))
