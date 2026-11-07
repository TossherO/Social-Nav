"""Person masking must preserve current obstacles, real ray occlusion and source identity."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from perception.environment.dynamic import people_boxes, person_points, StaticEvidence
from perception.environment.fusion import fuse_scans
from perception.environment.layers import load_config
from perception.environment.maps import static_geometry
from perception.environment.occupancy import OccupancyTree
from perception.environment.people import PeopleObservations
from perception.environment.poses import PoseTimeline
from perception.environment.scans import ScanJournal
from perception.environment.storage import content_hash


def observation(stamp):
    return dict(stamp_s=stamp, observation_id='people:'+str(stamp), input_fingerprint='synthetic',
        frame_id='odom', health='valid', clock_domain='synthetic',
        localization_binding=dict(session_id='local', generation=0, frame_id='odom'),
        tracks=[dict(box=[1.5, 0, 0, 1.2, 1.2, 1.8, 0, .99], state='observed', confirmed=True,
                     hits=5, last_observed_stamp_s=stamp)])


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_masks_require_current_observed_confirmed_people_and_same_clock_lifecycle(self):
        select = lambda row:people_boxes(row, 10., ('local', 0), 'odom', self.config['dynamic'], clock_domain='synthetic')
        self.assertEqual(len(select(observation(10.))[0]), 1)
        for changes in (dict(stamp_s=10.1), dict(stamp_s=9.), dict(health='degraded'),
                        dict(clock_domain='other'), dict(localization_binding=dict(session_id='other'))):
            row = observation(10.); row.update(changes)
            self.assertEqual(len(select(row)[0]), 0)
        for changes in (dict(state='predicted'), dict(confirmed=False), dict(hits=1),
                        dict(box=[1.5, 0, 0, 1.2, 1.2, 1.8, 0, .1])):
            row = observation(10.); row['tracks'][0].update(changes)
            self.assertEqual(len(select(row)[0]), 0)

    def test_ground_and_disconnected_background_inside_a_weak_box_are_preserved(self):
        points = np.array([[1.45, -.05, -.5], [1.5, 0, 0], [1.55, .05, .3],
                           [1.95, -.05, -.5], [1.95, 0, 0], [1.95, .05, .3], [1.5, 0, -1.]])
        floor = np.full((60, 60), -1.)
        boxes = np.asarray([observation(10.)['tracks'][0]['box']])
        mask = person_points(points, boxes, floor, [-3, -3], self.config)
        np.testing.assert_array_equal(mask, [True, True, True, False, False, False, False])
        self.assertFalse(person_points(points, boxes, np.full_like(floor, np.nan), [-3, -3], self.config).any())

    def test_removed_person_return_keeps_only_the_visible_ray_prefix(self):
        tree = OccupancyTree(.1); self.addCleanup(tree.close)
        cloud = np.array([[1.05, 0, .25]], np.float32)
        voxels = tree.update(cloud, [0, 0, .25], [-3, -3, -3, 3, 3, 3], 1., self.config,
            transient_mask=[True], transient_ttl=.8, omit_transient=True)
        self.assertGreater(len(voxels), 0)
        self.assertFalse((voxels[:, 3] >= self.config['occupied_probability']).any())
        self.assertTrue((voxels[:, 0] < 1.05).all())
        self.assertFalse((voxels[:, 0] > 1.05).any())

    def test_local_people_stay_occupied_then_expire_without_erasing_static_hits(self):
        tree = OccupancyTree(.1); self.addCleanup(tree.close)
        bounds = [-3, -3, -3, 3, 3, 3]
        cloud = np.array([[1.05, 0, .25], [2.05, 0, .25]], np.float32)
        voxels = tree.update(cloud, [0, 0, .25], bounds, 1., self.config, rays=False,
            transient_mask=[True, False], transient_ttl=.8)
        self.assertEqual(int((voxels[:, 3] >= self.config['occupied_probability']).sum()), 2)
        voxels = tree.update(np.empty((0, 3)), [0, 0, .25], bounds, 2., self.config, rays=False,
            transient_mask=np.zeros(0, bool), transient_ttl=.8)
        self.assertEqual(len(voxels), 1)
        self.assertGreater(voxels[0, 0], 2.)

    def test_invalid_localization_withdraws_old_mask_provenance_and_occupancy(self):
        from perception.environment.ros.node import EnvironmentNode
        from tests.test_environment_consumers import snapshot
        source = snapshot(); source.metadata['dynamic_filter'] = dict(status='valid', observation_id='old')
        node = EnvironmentNode.__new__(EnvironmentNode)
        node.latest, node.status = source, 'LOCAL_INVALID'
        node.publish = lambda *args:None
        stamp = SimpleNamespace(to_sec=lambda:11., to_nsec=lambda:11000000000)
        pose = SimpleNamespace(header=SimpleNamespace(stamp=stamp, frame_id='odom'), session_id='local', generation=0)
        node.clear_display(pose)
        self.assertEqual(node.latest.metadata['dynamic_filter']['status'], 'unavailable')
        self.assertTrue((node.latest.layers['occupancy'] == -1).all())
        self.assertEqual(source.metadata['dynamic_filter']['status'], 'valid')

    def test_person_recovery_counts_independent_scans_and_preserves_unmasked_background(self):
        review = StaticEvidence(self.config)
        points = np.repeat([[.15, .15, .5]], 20, axis=0)
        floor = np.zeros((10, 10))
        self.assertTrue(review.keep(points).all())
        review.add(points, np.ones(20, bool), .5, floor, [0, 0])
        for stamp in (1., 1.1, 1.2):
            review.add(points, np.zeros(20, bool), stamp, floor, [0, 0])
        self.assertFalse(review.keep(points).any())
        for stamp in (1.6, 2.2):
            review.add(points, np.zeros(20, bool), stamp, floor, [0, 0])
        self.assertTrue(review.keep(points).all())
        review.add(points, np.ones(20, bool), 2.8, floor, [0, 0])
        self.assertFalse(review.keep(points).any())

    def test_static_free_votes_require_time_separation_and_stop_before_returns(self):
        tree = OccupancyTree(.1); self.addCleanup(tree.close)
        tree.enable_static_review(.5, .2)
        bounds = [-3, -3, -3, 3, 3, 3]
        cloud = np.repeat([[2.05, .05, .55]], 50, axis=0).astype(np.float32)
        query = np.array([[1.05, .05, .55], [1.95, .05, .55], [2.05, .05, .55], [2.55, .05, .55]])
        for stamp in (1., 1.1, 1.2):
            tree.update(cloud, [0, .05, .55], bounds, stamp, self.config, read=False)
        np.testing.assert_array_equal(tree.static_votes(query), [[0, 1], [0, 0], [1, 0], [0, 0]])
        for stamp in (1.6, 2.2):
            tree.update(cloud, [0, .05, .55], bounds, stamp, self.config, read=False)
        np.testing.assert_array_equal(tree.static_votes(query), [[0, 3], [0, 0], [3, 0], [0, 0]])

    def test_offline_ground_extrapolation_is_bounded_and_keeps_ground(self):
        floor = np.full((60, 60), -1.)
        floor[:, 45:] = np.nan
        points = np.array([[1.45, -.05, -.5], [1.5, 0, 0], [1.55, .05, .3],
                           [1.55, .05, -1.], [1.95, 0, .3]])
        boxes = np.asarray([observation(10.)['tracks'][0]['box']])
        old = person_points(points, boxes, floor, [-3, -3], self.config)
        new = person_points(points, boxes, floor, [-3, -3], self.config, ground_extrapolation=.3)
        self.assertFalse(old.any())
        np.testing.assert_array_equal(new, [True, True, True, False, False])
        self.assertFalse(person_points(points, boxes, np.full_like(floor, np.nan), [-3, -3],
            self.config, ground_extrapolation=.3).any())

    def test_free_ray_contradiction_removes_missed_person_endpoint_without_people(self):
        config = self.config
        axis = np.arange(-2.5, 2.5, .1)
        x, y = np.meshgrid(axis, axis)
        ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
        ghost = np.array([[1.05, .05, -.45]])
        wall = np.array([[2.05, .05, -.45]])
        occluded = np.array([[2.55, .05, -.45]])
        points = np.r_[ground, ghost, wall, occluded].astype(np.float32)
        candidate = static_geometry(points, [[0, .05, -.45]], config, map_id='map', frame='map', stamp=1.)
        times = (1., 1.6, 2.2, 2.8)
        rows = [dict(stamp_s=t, stamp_ns=int(t*1e9), position=[0, .05, -.45],
                     quaternion_xyzw=[0, 0, 0, 1], session_id='local', generation=0, frame_id='odom', health=0) for t in times]
        keys = [[t, 0, .05, -.45, 0, 0, 0, 0, 0] for t in times]
        timeline = PoseTimeline(rows, config['poses'], keys)
        transform = np.eye(4); transform[:3, 3] = rows[0]['position']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scans.sqlite'
            writer = ScanJournal(path, config['scan_recording'], {})
            for i, stamp in enumerate(times):
                cloud = np.r_[ground, ghost if i == 0 else wall]-transform[:3, 3]
                origins = np.tile(transform[:3, 3], (len(cloud), 1))
                writer.append(cloud, transform, int(stamp*1e9), dict(session_id='local', generation=0,
                    frame_id='odom', rays_used=True, ray_origin_model='pose_interpolated_origins'), origins)
            writer.close()
            filtered = fuse_scans(candidate, path, timeline, config, {}, filter_mode='geometric', candidate_points=points)
        def value(point):
            col,row = np.floor((point[:2]-candidate.metadata['origin_xy'])/config['resolution']).astype(int)
            return filtered.layers['occupancy'][row, col]
        # Removing a contradicted endpoint does not prove the full column free.
        self.assertEqual(value(ghost[0]), -1)
        self.assertEqual(value(wall[0]), 100)
        self.assertEqual(value(occluded[0]), 100)
        self.assertEqual(filtered.metadata['dynamic_filter']['geometric_rejected_endpoints'], 1)

    def test_people_packages_reject_wrong_source_and_modified_observations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/'people.jsonl').write_text(json.dumps(observation(1.))+'\n')
            manifest = dict(kind='environment_people', source=dict(run_fingerprint={'run':'one'},
                bag_fingerprint={'bag':'one'}), people_sha256=content_hash(path/'people.jsonl'))
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            reader = PeopleObservations(path, run_fingerprint={'run':'one'}, bag_fingerprint={'bag':'one'})
            self.assertIsNone(reader.at(.9)); self.assertEqual(reader.at(1.)['stamp_s'], 1.)
            with self.assertRaises(ValueError):
                PeopleObservations(path, run_fingerprint={'run':'one'}, bag_fingerprint={'bag':'other'})
            (path/'people.jsonl').write_text(json.dumps(observation(2.))+'\n')
            with self.assertRaises(ValueError):
                PeopleObservations(path, run_fingerprint={'run':'one'}, bag_fingerprint={'bag':'one'})

    def test_static_fusion_cannot_reintroduce_person_endpoints_and_preserves_repeated_wall(self):
        config = self.config
        axis = np.arange(-2.5, 2.5, .1)
        x, y = np.meshgrid(axis, axis)
        ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
        human = np.array([[1.45, -.05, -.5], [1.5, 0, 0], [1.55, .05, .3]])
        wall = np.c_[np.full(30, 2.), np.linspace(-1, 1, 30), np.linspace(-.8, .7, 30)]
        points = np.r_[ground, human, wall].astype(np.float32)
        candidate = static_geometry(points, [[0, 0, 0]], config, map_id='map', frame='map', stamp=1.)
        times = (1., 1.6, 2.2, 2.8)
        rows = [dict(stamp_s=t, stamp_ns=int(t*1e9), position=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1],
                     session_id='local', generation=0, frame_id='odom', health=0) for t in times]
        timeline = PoseTimeline(rows, config['poses'], [[1., 0, 0, 0, 0, 0, 0, 0, 0]])
        class People:
            reference = dict(people_sha256='synthetic')
            manifest = dict(clock_domain='synthetic')
            def at(self, stamp):return observation(stamp)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scans.sqlite'
            writer = ScanJournal(path, config['scan_recording'], {})
            meta = dict(session_id='local', generation=0, frame_id='odom', rays_used=True,
                        ray_origin_model='pose_interpolated_origins')
            for stamp in times:
                writer.append(points, np.eye(4), int(stamp*1e9), meta, np.zeros_like(points))
            writer.close()
            baseline = fuse_scans(candidate, path, timeline, config, {})
            filtered = fuse_scans(candidate, path, timeline, config, {}, people=People(),
                                  filter_mode='geometric', candidate_points=points)
        cells = lambda pts:np.floor((pts[:, :2]-candidate.metadata['origin_xy'])/config['resolution']).astype(int)
        hx, hy = cells(human).T
        wx, wy = cells(wall).T
        self.assertTrue((baseline.layers['occupancy'][hy, hx] == 100).all())
        self.assertFalse((filtered.layers['occupancy'][hy, hx] == 100).any())
        self.assertTrue((filtered.layers['occupancy'][wy, wx] == 100).all())
        self.assertGreater(filtered.metadata['dynamic_filter']['counts']['masked_points'], 0)
        self.assertNotEqual(baseline.metadata['snapshot_id'], filtered.metadata['snapshot_id'])


if __name__ == '__main__':
    unittest.main()
