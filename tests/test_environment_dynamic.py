"""Person masking, transient occupancy and restoration of observed background."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from perception.environment.dynamic import people_boxes, person_points, StaticEvidence
from perception.environment.fusion import fuse_scans
from perception.environment.layers import GridSnapshot, LocalEnvironmentMap, load_config
from perception.environment.maps import static_geometry
from perception.environment.occupancy import OccupancyTree
from perception.environment.people import PeopleObservations
from perception.environment.person_objects import point_owners
from perception.environment.poses import PoseTimeline
from perception.environment.scans import ScanJournal
from perception.environment.storage import content_hash
from perception.environment.near_field import NearFieldFreeMemory
from tests.test_environment_near_field import sample


def observation(stamp):
    return dict(stamp_s=stamp, observation_id='people:'+str(stamp), input_fingerprint='synthetic',
        frame_id='odom', health='valid', clock_domain='synthetic',
        localization_binding=dict(session_id='local', generation=0, frame_id='odom'),
        tracks=[dict(box=[1.5, 0, 0, 1.2, 1.2, 1.8, 0, .99], state='observed', confirmed=True,
                     hits=5, last_observed_stamp_s=stamp)])


def check_uniform_obstacle_confirmation(test, backend):
    config = load_config()
    config['compute']['backend'] = backend
    config['dynamic']['enabled'] = False
    config['near_field']['enabled'] = False
    config['window_m'] = 4.
    model = LocalEnvironmentMap(config)
    test.addCleanup(model.close)
    model.ground_conflict_view = None
    ground = np.zeros((40, 40), np.float32)
    support = np.ones(ground.shape, bool)
    fixed, isolated, expired_vote = [[.55, .05, .5]], [[.85, .05, .5]], [[1.15, .05, .5]]

    def scan(stamp, points, dx=0., generation=0):
        pose = np.eye(4)
        pose[0,3] = dx
        local = np.asarray(points, dtype=float).reshape(-1,3) - pose[:3,3]
        with patch.object(model, '_plane_ground', return_value=(ground,support,dict(status='synthetic'))):
            return model.update(local, pose, stamp, ('test', generation))

    def occupancy(grid, point):
        x, y = np.floor((point[0][:2]-np.asarray(grid.metadata['origin_xy']))/.1).astype(int)
        return grid.layers['occupancy'][y,x]

    first = scan(10., fixed*3+isolated*3+expired_vote*3)
    test.assertEqual(first.metadata['unconfirmed_obstacle_points'], 9)
    test.assertEqual(occupancy(first, isolated), 100)
    second = scan(10.15, fixed, dx=.2)
    test.assertEqual(second.metadata['unconfirmed_obstacle_points'], 1)
    third = scan(10.25, fixed, dx=.2)
    test.assertEqual(third.metadata['unconfirmed_obstacle_points'], 0)
    after = scan(10.5, [], dx=.2)
    test.assertEqual(occupancy(after, isolated), -1)
    test.assertEqual(occupancy(after, fixed), 100)
    late = scan(10.6, expired_vote)
    test.assertEqual(late.metadata['unconfirmed_obstacle_points'], 1)
    test.assertEqual(occupancy(late, expired_vote), 100)
    test.assertEqual(occupancy(scan(10.9, []), expired_vote), -1)
    restarted = scan(11., fixed, generation=1)
    test.assertEqual(restarted.metadata['unconfirmed_obstacle_points'], 1)


def check_long_obstacle(test, backend, *, device=None):
    from perception.environment.obstacle_memory import EXPIRY_LAYER

    config = load_config()
    config['compute']['backend'] = backend
    config['dynamic']['enabled'] = False
    config['near_field']['enabled'] = False
    config['window_m'] = 4.
    config['obstacle_ttl_s'] = .3
    model = LocalEnvironmentMap(config)
    test.addCleanup(model.close)
    model.ground_conflict_view = None
    ground = np.zeros((40,40),np.float32)
    support = np.ones(ground.shape,bool)
    point = [.55,.05,.5]
    interrupted = [.85,.05,.5]
    first,second = (20,25),(20,28)

    def scan(stamp, points):
        with patch.object(model,'_plane_ground',return_value=(ground,support,dict(status='synthetic'))):
            snapshot = model.update(np.asarray(points).reshape(-1,3),np.eye(4),stamp,('test',0),device=device)
        return snapshot if device is None else snapshot.materialize(consumer='test_long_obstacle')

    for index in range(11):
        snapshot=scan(10.+.2*index,[point,interrupted] if index!=5 else [point])
    retained=scan(13.,[])
    test.assertEqual(retained.layers['occupancy'][first],100)
    test.assertEqual(retained.layers['last_observed'][first],12.)
    test.assertEqual(retained.layers['occupancy'][second],-1)
    renewed=scan(13.2,[point])
    test.assertEqual(renewed.layers['last_observed'][first],13.2)
    test.assertEqual(scan(15.2,[]).layers['occupancy'][first],100)
    expired=scan(15.25,[])
    test.assertEqual(expired.layers['occupancy'][first],-1)
    test.assertTrue(expired.layers[EXPIRY_LAYER][first])
    test.assertEqual(expired.layers['unknown_distance'][first],0.)
    rehit=scan(15.4,[point])
    test.assertEqual(rehit.layers['occupancy'][first],100)
    test.assertFalse(rehit.layers[EXPIRY_LAYER][first])
    test.assertTrue(scan(15.7,[]).layers[EXPIRY_LAYER][first])


def check_near_short_obstacles(test, *, device=None):
    from perception.environment.obstacle_memory import ObstacleMemory

    memory = ObstacleMemory()
    shape = (2,8)
    origin = np.zeros(2)
    resolution = .6
    memory.align(origin,shape,resolution)
    y = np.zeros(shape[1],int)
    x = np.arange(shape[1])
    old_free = np.array([[0,0,-1,-1,-1,0,-1,0],[-1]*shape[1]],np.int8)
    person_y,person_x = y[[2,3]],x[[2,3]]
    memory.remember_sources(person_y,person_x,9.9,near_short=np.zeros(2,bool),
        unconfirmed=np.array([True,False]),person_cells=(person_y,person_x))
    meta = dict(origin_xy=origin.tolist(),shape=list(shape),resolution=resolution,stamp_s=9.9,
                rays_used=True,free_vertical_coverage_required=1.)
    layers = dict(occupancy=old_free,ground_support=np.ones(shape,bool),coverage=np.ones(shape,np.float32),
                  last_observed=np.full(shape,9.9),near_field_free=np.zeros(shape,bool),
                  ground_conflict=np.zeros(shape,bool))
    layers['near_field_free'][0,1] = True
    layers['ground_conflict'][0,5] = True
    previous = GridSnapshot(meta,layers)
    if device is not None:
        import torch
        from perception.environment.device import DeviceGridSnapshot
        previous = DeviceGridSnapshot.seal(meta,{k:torch.from_numpy(v).to(device) for k,v in layers.items()})

    radius = load_config()['near_obstacle_radius_m']
    test.assertEqual(radius,3.5)
    def observe(stamp,snapshot):
        short = memory.near_short_only(y,x,stamp,center=np.full(2,resolution/2),resolution=resolution,radius=radius,
            evidence_age=.5,max_gap=.35,ordinary_ttl=2.,previous=snapshot)
        provisional = memory.unconfirmed(y,x,stamp,.5)|short
        memory.observe(y,x,stamp,max_gap=.35,min_span=2.,hold=2.)
        memory.remember_sources(y,x,stamp,near_short=short,unconfirmed=provisional,person_cells=None)
        return short,provisional

    expected = [True,False,True,False,False,False,False,False]
    short,provisional = observe(10.,previous)
    np.testing.assert_array_equal(short,expected)
    test.assertTrue(provisional.all())
    observe(10.2,None)
    short,provisional = observe(10.4,None)
    np.testing.assert_array_equal(short,expected)
    np.testing.assert_array_equal(provisional,expected)
    for step in range(3,11):
        observe(10.+step*.2,None)
    test.assertTrue(memory.projection()[0][0,0])
    test.assertTrue(memory.projection()[0][0,2])


class DynamicTests(unittest.TestCase):
    def test_obstacle_confirmation_without_people_uses_distinct_scans_and_world_cells(self):
        check_uniform_obstacle_confirmation(self, 'cpu')

    def test_long_obstacle_renews_then_expires_to_unknown(self):
        check_long_obstacle(self, 'cpu')

    def test_near_short_requires_measured_free_or_confirmed_person_without_existing_obstacle(self):
        check_near_short_obstacles(self)

    def test_current_owned_returns_seed_near_short_mode_for_later_unowned_hits(self):
        config = load_config()
        config['compute']['backend'] = 'cpu'
        config['near_field']['enabled'] = False
        config['window_m'] = 4.
        model = LocalEnvironmentMap(config,lidar_from_base_xyz=np.zeros(3))
        self.addCleanup(model.close)
        model.ground_conflict_view = None
        floor = np.zeros((40,40),np.float32)
        body = np.array([[.44,.05,.5],[.45,.06,.5],[.46,.07,.5]])
        people = dict(observation(10.),session_id='tracking',generation=0)
        people['tracks'][0].update(track_id='pedestrian',box=[.45,.06,.6,.5,.5,1.2,0.,.99])

        def scan(stamp,points,persons=None):
            with patch.object(model,'_plane_ground',return_value=(floor,np.ones(floor.shape,bool),
                              dict(status='tracked',plane_world=[0.,0.,0.]))):
                return model.update(points,np.eye(4),stamp,('local',0),people=persons,clock_domain='synthetic')

        scan(10.,body,people)
        self.assertTrue((model.support_observation['person_owner']==0).all())
        for stamp in (10.1,10.2,10.3):
            local = scan(stamp,body)
            self.assertEqual(local.metadata['near_short_obstacle_cells'],1)
            self.assertEqual(local.layers['occupancy'][20,24],100)
        self.assertEqual(scan(10.6,np.empty((0,3))).layers['occupancy'][20,24],-1)

    def test_obstacle_expiry_rejects_historical_free_then_accepts_fresh_free(self):
        from perception.environment.obstacle_memory import EXPIRY_LAYER, apply_obstacle_memory
        shape=(1,3)
        layers=dict(occupancy=np.zeros(shape,np.int8),coverage=np.ones(shape,np.float32),
                    ground_support=np.ones(shape,bool),observed=np.ones(shape,bool),
                    last_observed=np.array([[9.,11.,11.]]),near_field_free=np.array([[False,True,False]]),
                    obstacle_min_z=np.full(shape,np.nan),obstacle_max_z=np.full(shape,np.nan))
        no = np.zeros(shape,bool)
        projection=(no,np.full(shape,-np.inf),~no,np.full(shape,10.),no)
        fresh_free=apply_obstacle_memory(layers,projection,1.)
        np.testing.assert_array_equal(layers['occupancy'],[[-1,-1,0]])
        np.testing.assert_array_equal(layers[EXPIRY_LAYER],[[True,True,False]])
        np.testing.assert_array_equal(fresh_free,[[False,False,True]])

    def setUp(self):
        self.config = load_config()

    def test_masks_require_current_observed_confirmed_people_and_same_clock_lifecycle(self):
        select = lambda row:people_boxes(row, 10., ('local', 0), 'odom', self.config['dynamic'], clock_domain='synthetic')
        first_hit = observation(10.)
        first_hit['tracks'][0]['hits'] = 1
        self.assertEqual(len(select(first_hit)[0]), 1)
        for changes in (dict(stamp_s=10.1), dict(stamp_s=9.), dict(health='degraded'),
                        dict(clock_domain='other'), dict(localization_binding=dict(session_id='other'))):
            row = observation(10.); row.update(changes)
            self.assertEqual(len(select(row)[0]), 0)
        for changes in (dict(state='predicted'), dict(confirmed=False), dict(hits=0),
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

    def test_person_ownership_keeps_radius_boundary_order_and_overlap_ties(self):
        self.config['dynamic']['object_radius_limit_m'] = .5
        box = np.array([0., 0., .5, 1., 1., 1., np.pi / 8, .99])
        points = np.array([[.375, 0., .5], [5., 0., .5], [.4375, 0., .5],
                           [.5, 0., .5], [.5001, 0., .5], [.375, 0., 0.]])
        boxes = np.stack((box, box))
        actual = point_owners(points, boxes, self.config, np.zeros(3), plane=[0., 0., 0.])
        np.testing.assert_array_equal(actual, [0, -1, 0, 0, -1, -1])

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



def voxel_sequence(backend,ttl,mixed=False):
    config=load_config();config['dynamic']['transient_ttl_s']=ttl
    tree=OccupancyTree(.1,backend=backend)
    query=np.array([[.55,.05,.5],[1.05,.05,.5]])
    bounds=[-4,-4,-2,4,4,3];points=query.copy();mask=[1,0]
    if mixed:points=np.r_[points,[[.56,.05,.5]]];mask=[1,0,0]
    states=[]
    try:
        for stamp,cloud,moving in ((1.,points,mask),(1.1,[],[]),(1.3,[],[]),(1.5,[],[]),(1.7,[],[]),
                                    (1.8,points,mask),(1.9,points,mask)):
            tree.update(np.asarray(cloud).reshape(-1,3),[0,0,.84],bounds,stamp,config,rays=False,
                transient_mask=moving,transient_ttl=ttl,read=False)
            values=tree.snapshot(stamp)
            keys=np.floor(query/.1).astype(int)
            occupied=np.floor(values[values[:,3]>=config['occupied_probability'],:3]/.1).astype(int)
            states.append([bool((occupied==key).all(axis=1).any()) for key in keys])
        return states
    finally:tree.close()


def overlay_sequence(backend,known):
    from perception.environment.cuda_view import native_view
    config=load_config();config['dynamic']['transient_ttl_s']=0.
    memory=NearFieldFreeMemory(config,dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
    tree=OccupancyTree(.1,backend=backend);states=[];times=[]
    try:
        for stamp,source,points,moving in ((1.,sample(1.,state=0 if known else -1),[],[]),
                (2.,sample(2.),[[.55,.05,.5]],[1]),(2.2,sample(2.2),[],[]),
                (2.4,sample(2.4),[[.55,.05,.5]],[1])):
            # Unknown occupancy can still have measured ground without free rays.
            if not known:source.layers['ground_support'][40,45]=True
            cloud=np.asarray(points).reshape(-1,3)
            if backend=='cpu':view=memory.update(source,cloud,transient_mask=moving).layers
            else:
                raw=source.layers
                view,_=native_view(tree,raw['ground_z'],raw['ground_support'],[-4,-4],config,stamp,cloud,
                    source_layers=raw,metadata=source.metadata,policy=memory,transient_mask=moving)
            states.append(int(view['occupancy'][40,45]));times.append(float(view['last_observed'][40,45]))
        return states,times
    finally:tree.close()


class PersonLifetimeTests(unittest.TestCase):
    def test_zero_config_is_explicit_and_negative_nonfinite_are_rejected(self):
        config=load_config();self.assertEqual(config['dynamic']['transient_ttl_s'],.2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'environment.yaml'
            for ttl in (0.,.2,-.1,np.nan,np.inf):
                config['dynamic']['transient_ttl_s']=ttl;path.write_text(yaml.safe_dump(config))
                if np.isfinite(ttl) and ttl>=0:self.assertEqual(load_config(path)['dynamic']['transient_ttl_s'],ttl)
                else:
                    with self.assertRaises(ValueError):load_config(path)

    def test_zero_keeps_current_and_repeated_hits_but_not_an_unobserved_person_voxel(self):
        states=voxel_sequence('cpu',0.)
        self.assertEqual([s[0] for s in states],[True,False,False,False,False,True,True])
        self.assertTrue(all(s[1] for s in states))

    def test_positive_lifetimes_retain_short_gaps_and_current_mixed_structure_is_ordinary(self):
        for ttl,expected in ((.2,[True,True,False,False,False,True,True]),
                             (.4,[True,True,True,False,False,True,True]),
                             (.6,[True,True,True,True,False,True,True])):
            with self.subTest(ttl=ttl):self.assertEqual([s[0] for s in voxel_sequence('cpu',ttl)],expected)
        self.assertTrue(all(s[0] for s in voxel_sequence('cpu',0.,mixed=True)))

    def test_zero_overlay_restores_only_existing_background_and_its_original_time(self):
        known,times=overlay_sequence('cpu',True)
        self.assertEqual(known,[0,100,0,100]);self.assertEqual(times[2],1.)
        self.assertEqual(overlay_sequence('cpu',False)[0],[-1,100,-1,100])

    def test_negative_native_wrapper_lifetime_is_rejected(self):
        tree=OccupancyTree(.1);self.addCleanup(tree.close)
        with self.assertRaises(ValueError):tree.update(np.array([[.55,.05,.5]]),[0,0,.84],[-4,-4,-2,4,4,3],
            1.,load_config(),transient_mask=[1],transient_ttl=-.1)


class PersonLifetimeCudaTests(unittest.TestCase):
    def test_zero_and_short_lifetimes_and_near_field_match_cpu(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        for ttl in (0.,.2,.4,.6):
            for mixed in (False,True):self.assertEqual(voxel_sequence('cpu',ttl,mixed),voxel_sequence('cuda',ttl,mixed))
        for known in (False,True):
            a,b=overlay_sequence('cpu',known),overlay_sequence('cuda',known)
            self.assertEqual(a[0],b[0]);np.testing.assert_equal(a[1],b[1])

if __name__ == '__main__':
    unittest.main()
