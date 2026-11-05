"""Current static-support fixtures and independent consumer checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from perception.environment.layers import GridSnapshot, load_config
from perception.environment.navigation import navigation_geometry
from perception.environment.scene import assemble_scene, SceneInputs
from navigation.planning.inputs import scene_status
from tests.test_navigation_planning import grid, asset, request


def prior_asset(cells=None):
    prior = asset(cells)
    prior.snapshot.layers['last_observed'] = np.full((60, 60), 1., np.float64)
    prior.snapshot.layers['ground_z'] = np.full((60, 60), -.85, np.float32)
    prior.snapshot.layers['ground_support'] = np.ones((60, 60), bool)
    return prior


def observation(cells=None, *, stamp=10., origin=(0., 0.), changes=None, historical=False):
    raw = grid(cells)
    raw.metadata.update(stamp_s=stamp, origin_xy=list(origin), snapshot_id='local:'+str(stamp),
        ground_tracking=dict(status='tracked', plane_world=[0., 0., -.85]),
        association=dict(valid=True, status='PRIOR_ASSOCIATED', map_id='map-a', map_frame='map',
            local_frame='odom', local_session_id='test', local_generation=0, stamp_s=stamp,
            measurement_stamp_s=stamp-.1, map_from_local=np.eye(4).tolist(),
            association_session='association', association_generation=0))
    raw.metadata['association'].update(changes or {})
    shape = raw.layers['occupancy'].shape
    raw.layers.update(ground_support=np.ones(shape, bool), coverage=np.ones(shape, np.float32),
        last_observed=np.full(shape, stamp, np.float64), near_field_free=np.full(shape, historical, bool),
        ground_z=np.full(shape,-.85,np.float32))
    return raw


def scene(snapshot):
    return assemble_scene(snapshot, clock_domain='synthetic', settings=load_config()['scene'])


class NavigationConsumerTests(unittest.TestCase):
    def setUp(self):
        from perception.environment.support import NavigationSupport
        self.support = NavigationSupport(load_config())
        self.support.ground.settings['ground_min_observations'] = 1
        self.prior = prior_asset()

    def apply(self, raw, prior):
        m = raw.metadata
        observation = dict(stamp_s=m['stamp_s'], identity=(m['session_id'],m['generation'],m['frame_id']),
            endpoints=np.empty((0,3),np.float64), transient_mask=np.empty(0,np.uint8))
        return self.support.apply(raw, prior, observation=observation)

    def unknown(self):
        cells = np.zeros((60, 60), np.int8); cells[18:23, 18:23] = -1
        return cells


    def test_cached_verification_rechecks_age_geometry_and_asset_bytes(self):
        from unittest.mock import patch
        import perception.environment.support as module
        current=scene(self.apply(observation(self.unknown()),self.prior))
        with patch.object(module,'static_support',wraps=module.static_support) as support:
            navigation_geometry(current,now=10.,asset=self.prior,check_asset=True)
            calls=support.call_count
            self.assertGreater(calls,0)
            navigation_geometry(current,now=10.01,asset=self.prior,check_asset=True)
            self.assertEqual(support.call_count,calls)
        with self.assertRaisesRegex(ValueError,'stale'):
            navigation_geometry(current,now=20.,asset=self.prior,check_asset=True)
        self.prior.snapshot.layers['occupancy'][20,20]=100
        with self.assertRaises(ValueError):navigation_geometry(current,now=10.,asset=self.prior,check_asset=True)
        current.geometry.layers['navigation_occupancy'].flags.writeable=True
        current.geometry.layers['navigation_occupancy'][20,20]=100
        with self.assertRaises(ValueError):current.verify()


    def test_loaded_layer_binding_cache_requires_physical_immutability(self):
        from perception.environment.context import layer_binding
        raw=observation()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'grid.npz';raw.save(path);loaded=GridSnapshot.load(path)
            before=layer_binding(raw);self.assertEqual(layer_binding(loaded),before)
            with self.assertRaises(ValueError):loaded.layers['occupancy'].flags.writeable=True
            loaded.metadata['stamp_s']+=1
            self.assertNotEqual(layer_binding(loaded),before)
        view=raw.layers['occupancy'].view();view.flags.writeable=False
        raw.layers['occupancy']=view;before=layer_binding(raw)
        view.base[0,0]=100
        self.assertNotEqual(layer_binding(raw),before)


    def test_full_cell_overlap_blocks_unknown_neighbor_despite_free_centre(self):
        prior = prior_asset(); prior.snapshot.layers['occupancy'][20, 21] = -1
        transform = np.eye(4); transform[0, 3] = .025
        raw = observation(self.unknown(), changes=dict(map_from_local=transform.tolist()))
        result = self.apply(raw, prior)
        self.assertEqual(prior.snapshot.layers['occupancy'][20, 20], 0)
        self.assertEqual(result.layers['navigation_occupancy'][20, 20], -1)


    def test_tilted_column_checks_height_extent_instead_of_only_cell_centre(self):
        transform = np.eye(4); angle = .12
        transform[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        self.prior.snapshot.layers['occupancy'][19, 20] = 100
        result = self.apply(observation(self.unknown(), changes=dict(map_from_local=transform.tolist())), self.prior)
        self.assertEqual(result.layers['navigation_occupancy'][20, 20], -1)


    def test_association_ages_at_consumption_and_safety_rechecks_its_asset(self):
        current = scene(self.apply(observation(self.unknown()), self.prior))
        self.assertEqual(scene_status(current, 10.35, 'synthetic', dict(max_geometry_age_s=.5)),
                         'navigation_prior_future_or_stale')
        other = prior_asset(); other.manifest['layers_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'asset mismatch'):
            navigation_geometry(current, asset=other, check_asset=True)
        other = prior_asset(); other.snapshot.layers['occupancy'][20, 20] = 100
        with self.assertRaisesRegex(ValueError, 'floor support absent'):
            navigation_geometry(current, asset=other, check_asset=True)


    def test_persistence_ipc_and_corrupted_priority(self):
        current = scene(self.apply(observation(self.unknown()), self.prior))
        with tempfile.TemporaryDirectory() as directory:
            current.save(directory); loaded = SceneInputs.load(directory)
            np.testing.assert_array_equal(loaded.geometry.layers['navigation_occupancy'], current.geometry.layers['navigation_occupancy'])
        from runtime.scenes import ScenePool, read_scene
        pool = ScenePool(1, 1048576); reader = pool.pool.reader()
        try:
            description = pool.publish(current, 'reference'); loaded = read_scene(reader, description)
            navigation_geometry(loaded, asset=self.prior, check_asset=True)
            del loaded; pool.release(description, 'reference')
        finally:
            reader.close(); pool.close()
        bad = copy.deepcopy(current.geometry)
        bad.layers['navigation_source'].flags.writeable = True
        bad.layers['navigation_source'][20, 20] = 1
        with self.assertRaisesRegex(ValueError, 'obstacle/source policy'):
            scene(bad).verify()


    def test_planner_reference_dwa_and_cpu_safety_share_the_fused_geometry(self):
        from navigation.planning.inputs import load_config as planning_config
        from navigation.planning.motion import load_motion_config
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope
        from navigation.planning.reference import ReferenceRuntime
        from navigation.control.dwa import DwaController
        from navigation.safety.guard import LocalSafety
        from tests.test_navigation_runtime import FakeWorker
        cfg = planning_config(); cfg['algorithm'] = 'hybrid_astar'
        cfg['local_replanning']['enabled'] = False
        motion = load_motion_config()
        geometry = VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]], np.eye(4), source='synthetic')
        current = scene(self.apply(observation(self.unknown()), self.prior))
        req = request('local', goal=[4., 2., 0.], goal_behavior='stop')
        feedback = dict(status='valid', source='synthetic_actual_plant', pose=[2.,2.,0.], velocity=[0.,0.],
            stamp_s=10., clock_domain='synthetic', local_identity=['test',0,'odom'])
        result = Planner(cfg, geometry).query(req, scene=current, asset=self.prior, now=10., clock_domain='synthetic')
        self.assertEqual(result['status'], 'path_found', result)
        reference = reference_envelope(result, req, current, self.prior)
        runtime = ReferenceRuntime(cfg, motion, geometry, worker_factory=FakeWorker)
        try:
            self.assertTrue(runtime.update(req, current, None, self.prior, now=10.))
            self.assertFalse(runtime.blocked)
        finally: runtime.close()
        controller = DwaController(cfg, motion, geometry, asset=self.prior)
        proposal = controller.command(req, current, feedback, reference, now=10., clock_domain='synthetic')
        self.assertEqual(proposal['status'], 'candidate', proposal)
        self.assertGreater(proposal['velocity'][0], 0.)
        safety = LocalSafety(geometry, motion, cfg['safety_margin_m'], people_settings=cfg['people'], shadow=True, asset=self.prior)
        approved = safety.gate(proposal, req, current, feedback, now=10., clock_domain='synthetic', dt=.1)
        self.assertEqual(approved['state'], 'allowed', approved)
        # Removing the asset cannot silently preserve permission from the producer.
        safety.asset = None
        self.assertEqual(safety.gate(proposal, req, current, feedback, now=10., clock_domain='synthetic', dt=.1)['state'], 'stop_request')


    def test_shadow_rejoins_fused_geometry_and_new_contact_still_stops(self):
        from tests.test_navigation_dwa import DwaTests
        from navigation.planning.planner import Planner
        from navigation.planning.continuity import reference_envelope
        from navigation.control.shadow import DwaShadowController
        from navigation.safety.shadow import ShadowSafety
        fixture = DwaTests(); fixture.setUp()
        current = scene(self.apply(observation(self.unknown()), self.prior))
        result = Planner(fixture.cfg, fixture.geometry).query(fixture.req, scene=current,
            asset=self.prior, now=10., clock_domain='synthetic')
        reference = reference_envelope(result, fixture.req, current, self.prior)
        rows = [dict(stamp_s=t, health=0, session_id='test', generation=0, frame_id='odom',
            position=[2.,2.,0.], quaternion_xyzw=[0.,0.,0.,1.]) for t in (9.7,9.8,9.9,10.)]
        control = DwaShadowController(fixture.cfg, fixture.motion, fixture.geometry, asset=self.prior)
        packet = control.step(fixture.req, current, rows, reference, now=10., clock_domain='synthetic', mono=0.)
        safety = ShadowSafety(fixture.cfg, fixture.motion, fixture.geometry, clock_domain='synthetic', asset=self.prior)
        safety.command('goal', fixture.req); safety.push_grid(current.geometry, mono=0.)
        safety.push_record(current.record)
        for row in rows: safety.push_state(row)
        safety.push_packet(packet, mono=0.)
        self.assertEqual(safety.tick(10., mono=0.)['state'], 'allowed')
        hit = self.unknown(); hit[20,20] = 100
        newer = self.apply(observation(hit, stamp=10.05), self.prior)
        safety.push_grid(newer, mono=.05); safety.push_state(dict(rows[-1], stamp_s=10.05))
        result = safety.tick(10.05, mono=.05)
        self.assertEqual(result['state'], 'stop_request', result)
        self.assertEqual(result['reason'], 'stopping_sweep_occupied')


    def test_ros_wire_preserves_derived_states_source_and_historical_time(self):
        import io
        import json
        try:
            from social_nav_environment.msg import EnvironmentGrid
        except ImportError: self.skipTest('Generated ROS messages unavailable')
        from perception.environment.scene import assign_grid_layers, snapshot_from_message
        raw = observation(self.unknown())
        for name in ('ground_z','obstacle_min_z','obstacle_max_z','clearance','unknown_distance'):
            raw.layers[name] = np.full((60,60), np.nan, np.float32)
        raw.layers['observed'] = raw.layers['occupancy'] != -1
        raw.layers['frontier'] = np.zeros((60,60), bool)
        fused = self.apply(raw, self.prior); meta = fused.metadata
        message = EnvironmentGrid(); message.snapshot_id = meta['snapshot_id']
        message.session_id = meta['session_id']; message.generation = meta['generation']
        message.header.frame_id = meta['frame_id']; message.header.stamp.secs = 10
        message.info.height, message.info.width = meta['shape']; message.info.resolution = meta['resolution']
        message.info.origin.position.x, message.info.origin.position.y = meta['origin_xy']
        message.metadata_json = json.dumps(meta); assign_grid_layers(message, fused.layers)
        stream = io.BytesIO(); message.serialize_numpy(stream, np)
        restored = snapshot_from_message(EnvironmentGrid().deserialize_numpy(stream.getvalue(), np))
        current = scene(restored); current.verify()
        navigation_geometry(current, asset=self.prior, check_asset=True)
        self.assertEqual(restored.layers['occupancy'][20,20], -1)
        self.assertEqual(restored.layers['navigation_occupancy'][20,20], 0)
        self.assertEqual(restored.layers['navigation_source_stamp'][20,20], 1.)


    def test_retired_derivation_is_rejected_and_raw_layers_remain_readable(self):
        raw = observation()
        raw.metadata['navigation_fusion'] = dict(method='persistent_background_with_voxel_clearing')
        with self.assertRaisesRegex(ValueError, 'Unsupported navigation derivation'):
            navigation_geometry(scene(raw))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'grid.npz'
            raw.save(path)
            loaded = GridSnapshot.load(path)
            np.testing.assert_equal(loaded.layers['occupancy'], raw.layers['occupancy'])

    def test_retired_config_cannot_silently_choose_a_different_algorithm(self):
        import yaml
        from perception.environment.layers import LocalEnvironmentMap
        for name in ('terrain', 'navigation_fusion'):
            cfg = load_config(); cfg[name] = dict(enabled=True)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
                with self.assertRaisesRegex(ValueError, 'Unsupported environment policy'):
                    load_config(path)
            with self.assertRaisesRegex(ValueError, 'Unsupported environment policy'):
                LocalEnvironmentMap(cfg)
