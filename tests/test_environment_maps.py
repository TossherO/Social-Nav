"""Final-map and memory semantics; no ROS master, GPU, models or shared labels."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from perception.environment.layers import load_config
from perception.environment.maps import static_geometry
from perception.environment.memory import (PoseTimeline, KeyObservations, create_index,
    add_observation, query_observations, query_places, add_place)
from perception.environment.storage import EnvironmentMap, export_navigation, content_hash
from perception.mapping_localization.results import read_pcd


def pose(stamp, *, x=0., health=0, session='local'):
    return dict(stamp_s=stamp, position=[x, 0, 0], quaternion_xyzw=[0, 0, 0, 1],
                session_id=session, generation=0, frame_id='odom', health=health)


def observation(stamp, segment=1):
    transform = np.eye(4); transform[0, 3] = stamp
    return dict(stamp_s=stamp, image_stamp_ns=int(stamp*1e9), receipt_ns=int(stamp*1e9),
                source_id='bag', segment=segment, frame_id='map', world_from_lidar=transform.tolist(),
                world_from_camera=None)


class MemoryTests(unittest.TestCase):
    def test_pose_queries_are_causal_do_not_bridge_failures_and_bind_final_corrections(self):
        settings = load_config()['memory']
        rows = [pose(1), pose(1.1, x=.1), pose(1.2, health=2), pose(1.3, x=.3), pose(1.4, x=.4)]
        keys = np.array([[1, 10, 0, 0, 0, 0, 0, 0, 0], [1.4, 12, 0, 0, 0, 0, 0, 1, 0]])
        timeline = PoseTimeline(rows, settings, keys)
        self.assertIsNone(timeline.at(.99))
        chosen = timeline.at(1.15)
        self.assertAlmostEqual(chosen['pose_stamp_s'], 1.1)
        self.assertAlmostEqual(chosen['world_from_lidar'][0][3], 10.1)
        self.assertEqual(chosen['final_keyframe'], 0)
        self.assertIsNone(timeline.at(1.25))
        self.assertIsNone(timeline.at(1.35))  # New valid segment has no final anchor yet.
        self.assertEqual(timeline.at(1.4)['final_keyframe'], 1)
        self.assertIsNone(timeline.at(2.))

    def test_local_memory_does_not_need_or_invent_a_map(self):
        timeline = PoseTimeline([pose(1), pose(1.1)], load_config()['memory'])
        row = timeline.at(1.15)
        self.assertEqual(row['pose_source'], 'recorded_local_pose')
        self.assertIsNone(row['final_keyframe'])
        self.assertEqual(row['local_frame'], 'odom')
        with self.assertRaises(ValueError):
            PoseTimeline([pose(1), pose(1)], load_config()['memory'])

    def test_visual_selection_uses_motion_time_and_a_hard_budget(self):
        settings = dict(load_config()['memory'], max_observations=3)
        selector = KeyObservations(settings)
        a = dict(segment=1, world_from_lidar=np.eye(4).tolist())
        self.assertTrue(selector.accept(1, a))
        self.assertFalse(selector.accept(2, a))
        moved = np.eye(4); moved[0, 3] = 1.1
        self.assertTrue(selector.accept(2, dict(a, world_from_lidar=moved.tolist())))
        self.assertTrue(selector.accept(8, dict(a, segment=2)))
        with self.assertRaises(ValueError):
            selector.accept(20, a)

    def test_places_queries_and_transitions_keep_time_and_local_segment_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            db = create_index(Path(directory)/'observations.sqlite', dict(snapshot_id='bound'))
            self.addCleanup(db.close)
            first = add_observation(db, observation(1))
            add_observation(db, observation(2))
            add_observation(db, observation(3, segment=2))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0], 1)
            self.assertEqual([row['stamp_s'] for row in query_observations(db, as_of=1.5)], [1])
            self.assertEqual(query_observations(db, as_of=3, position=[9, 9], radius=1), [])
            place = add_place(db, 'hall', '入口', first['id'], aliases=['门口'], bounds=[-1, -1, 3, 3])
            self.assertFalse(place['navigation_verified'])
            self.assertEqual(query_places(db, as_of=.5), [])
            self.assertEqual(query_places(db, as_of=3, name='门口')[0]['id'], 'hall')
            self.assertEqual(query_places(db, as_of=3, bounds=[2, 2, 4, 4])[0]['id'], 'hall')
            self.assertEqual(query_places(db, as_of=3, bounds=[8, 8, 9, 9]), [])
            with self.assertRaises(ValueError):
                add_place(db, 'bad', 'bad', 'unknown')
            with self.assertRaises(ValueError):
                query_observations(db, as_of=float('nan'))


class StaticMapTests(unittest.TestCase):
    def geometry(self):
        config = load_config()
        axis = np.arange(-2.5, 2.5, .15)
        x, y = np.meshgrid(axis, axis)
        ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
        wall = np.c_[np.full(30, 1.5), np.linspace(-1, 1, 30), np.linspace(-.85, .7, 30)]
        points = np.r_[ground, wall].astype(np.float32)
        return static_geometry(points, [[0, 0, 0]], config, map_id='prior', frame='map', stamp=10)

    def test_endpoints_preserve_thin_obstacles_but_never_fabricate_free_space_or_times(self):
        snapshot = self.geometry()
        self.assertTrue((snapshot.layers['occupancy'] == 100).any())
        self.assertFalse((snapshot.layers['occupancy'] == 0).any())
        self.assertTrue(np.isnan(snapshot.layers['last_observed']).all())
        self.assertFalse(snapshot.metadata['free_space_supported'])
        self.assertEqual(snapshot.metadata['time_scope'], 'offline_final_prior')
        self.assertTrue((snapshot.layers['endpoint_count'] > 0).any())

    def test_budget_rejection_cannot_silently_discard_obstacles(self):
        config = load_config(); config['static']['max_cells'] = 4
        with self.assertRaises(ValueError):
            static_geometry([[0, 0, -1], [2, 2, -1]], [[0, 0, 0]], config,
                            map_id='prior', frame='map', stamp=1)
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory)/'large.pcd'
            pcd.write_text('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 9\nDATA binary\n')
            with self.assertRaises(ValueError):
                read_pcd(pcd, 4, strict=True)

    def test_standard_navigation_export_preserves_unknown_and_row_orientation(self):
        snapshot = self.geometry()
        with tempfile.TemporaryDirectory() as directory:
            export_navigation(snapshot, directory)
            path = Path(directory)
            pixels = (path/'map.pgm').read_bytes().split(b'\n', 3)[3]
            original = snapshot.layers['occupancy']
            expected = np.flipud(np.where(original == -1, 205, np.where(original == 100, 0, 254))).astype(np.uint8)
            self.assertEqual(pixels, expected.tobytes())
            self.assertEqual(yaml.safe_load((path/'map.yaml').read_text())['origin'], [*snapshot.metadata['origin_xy'], 0.])
            with self.assertRaises(FileExistsError):
                export_navigation(snapshot, directory)

    def test_package_binds_snapshot_and_index_and_rejects_wrong_frame_or_changed_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            snapshot = self.geometry(); snapshot.save(path/'layers.npz')
            binding = dict(snapshot_id=snapshot.metadata['snapshot_id'])
            db = create_index(path/'observations.sqlite', binding)
            first = add_observation(db, observation(1)); db.commit(); db.close()
            manifest = dict(snapshot_id=snapshot.metadata['snapshot_id'], frame_id='map',
                            memory_binding=binding, layers_sha256=content_hash(path/'layers.npz'),
                            time_scope='offline_final_prior', source={})
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            package = EnvironmentMap(path)
            try:
                self.assertEqual(package.summary()['observations'], 1)
                self.assertEqual(package.query_observations(as_of=.5, frame='map'), [])
                with self.assertRaises(ValueError):
                    package.query_geometry([[0, 0]], frame='odom')
                package.add_place('hall', '入口', first['id'], frame='map')
                self.assertEqual(package.query_observations(as_of=2, place='hall', frame='map')[0]['id'], first['id'])
                self.assertEqual(package.crop_map([-1, -1, 1, 1], frame='map')['snapshot_id'], manifest['snapshot_id'])
            finally:
                package.close()
            with (path/'layers.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                EnvironmentMap(path)


if __name__ == '__main__':
    unittest.main()
