import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.reference import ReferenceRuntime
from navigation.planning.runtime import ReferenceQuery, Worker
from perception.environment.layers import GridSnapshot, load_config as environment_config
from perception.environment.scene import assemble_scene
from runtime.control import ControlLoop
from tests.navigation_fixtures import FakeWorker, asset, bundle, request


def reference_fixture():
    cfg=load_config();cfg['algorithm']='hybrid_astar';cfg['local_replanning']['enabled']=False
    cfg['pose_search']['native_library']=str(Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
    geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
    feedback=dict(status='valid',source='synthetic_actual_plant',pose=[2.,2.,0.],velocity=[0.,0.],stamp_s=10.,
        clock_domain='synthetic',local_identity=['test',0,'odom'])
    return SimpleNamespace(cfg=cfg,motion=load_motion_config(),geometry=geometry,scene=bundle(),
        req=request('local',goal=[4.,2.,0.],goal_behavior='stop'),feedback=feedback)


class SearchSceneTests(unittest.TestCase):
    def test_fused_scene_reuses_host_evidence_with_or_without_expiry(self):
        from perception.environment.free_memory import ConfirmedFreeMemory, remembered_free
        from perception.environment.navigation import safety_capsule
        from perception.environment.obstacle_memory import EXPIRY_LAYER
        from perception.environment.support import NavigationSupport
        from tests.test_environment_navigation import observation, prior_asset
        from tests.test_environment_support import points

        config = environment_config()
        asset = prior_asset()
        reference_fields = ('coverage', 'ground_support', 'last_observed', 'near_field_free')
        for expiry, person_background in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(expiry=expiry, person_background=person_background):
                support = NavigationSupport(config)
                memory = ConfirmedFreeMemory()
                for stamp, value in ((10., 0), (10.1, -1)):
                    raw = observation(np.full((60, 60), value, np.int8), stamp=stamp)
                    if expiry and value == -1:
                        raw.layers[EXPIRY_LAYER] = np.zeros((60, 60), bool)
                        raw.layers[EXPIRY_LAYER][25, 25] = True
                    if person_background:
                        raw.layers['person_background_free'] = np.zeros((60, 60), bool)
                    fused = support.apply(raw, asset, observation=points(raw))
                    capsule = safety_capsule(fused, extra_fields=reference_fields)
                    for layer in capsule.layers.values():
                        layer.flags.writeable = False
                    host = memory.update(capsule, asset)
                scene = assemble_scene(host, clock_domain='synthetic', settings=config['scene'], stamp_ns=10100000000)
                device = SimpleNamespace(metadata=fused.metadata, _layers=fused.layers, materialize=Mock())
                search = ControlLoop._search_scene(SimpleNamespace(device=device), scene)
                self.assertIs(search.verify(), scene)
                device.materialize.assert_not_called()
                dates = remembered_free(search, asset=asset)
                self.assertEqual(np.count_nonzero(np.isfinite(dates)), 3600 - expiry)
                np.testing.assert_array_equal(np.unique(dates[np.isfinite(dates)]), [10.])
                self.assertEqual(search.record['stamp_ns'], 10100000000)
                self.assertEqual(search.record['as_of'], 10.1)
                self.assertEqual(EXPIRY_LAYER in search.geometry.layers, expiry)
                self.assertTrue(all(not layer.flags.writeable for layer in search.geometry.layers.values()))

                partial = GridSnapshot(host.metadata, {k:v for k,v in host.layers.items() if k != 'coverage'})
                bound = assemble_scene(partial, clock_domain='synthetic', settings=config['scene'], stamp_ns=10100000000)
                device.materialize.return_value = GridSnapshot(fused.metadata, {'coverage': fused.layers['coverage']})
                extended = ControlLoop._search_scene(SimpleNamespace(device=device, metrics=None), bound).verify()
                device.materialize.assert_called_once_with(consumer='reference_search', fields=['coverage'], metrics=None)
                self.assertEqual(extended.record, search.record)
                self.assertEqual(extended.geometry.layers.keys(), host.layers.keys())

    def test_unfused_scene_downloads_only_missing_search_fields(self):
        from perception.environment.obstacle_memory import EXPIRY_LAYER
        from tests.test_environment_navigation import observation
        from tests.test_environment_scene import person, rgb

        raw = observation()
        raw.layers.update(person_background_free=np.zeros((60, 60), bool),
                          current_obstacle_hits=np.zeros((60, 60), bool),
                          **{EXPIRY_LAYER: np.zeros((60, 60), bool)})
        for layer in raw.layers.values():
            layer.flags.writeable = False
        fields = ('occupancy', 'person_background_free', EXPIRY_LAYER, 'last_observed', 'near_field_free',
                  'current_obstacle_hits')
        host = GridSnapshot(raw.metadata, {name: raw.layers[name] for name in fields})
        detection = dict(person(), localization_binding=dict(session_id='test', generation=0, frame_id='odom'))
        scene = assemble_scene(host, clock_domain='synthetic', settings=environment_config()['scene'],
                               stamp_ns=10000000000, people=detection, rgb=rgb())
        extra = {name: raw.layers[name] for name in ('ground_support', 'coverage')}
        materialize = Mock(return_value=GridSnapshot(raw.metadata, extra))
        device = SimpleNamespace(metadata=dict(raw.metadata), _layers=raw.layers, materialize=materialize)
        loop = SimpleNamespace(device=device, metrics=None)
        search = ControlLoop._search_scene(loop, scene).verify()
        materialize.assert_called_once_with(consumer='reference_search', fields=list(extra), metrics=None)
        self.assertEqual(search.record['health'], 'valid')
        self.assertEqual(search.record['as_of'], scene.record['as_of'])
        self.assertEqual(search.record['stamp_ns'], scene.record['stamp_ns'])
        self.assertEqual(search.record['clock_domain'], scene.record['clock_domain'])
        self.assertEqual(search.record['people'], scene.record['people'])
        self.assertIs(search.rgb, scene.rgb)
        self.assertNotEqual(search.record['input_fingerprint'], scene.record['input_fingerprint'])
        for name in fields:
            self.assertIs(search.geometry.layers[name], host.layers[name])
        self.assertTrue(all(not layer.flags.writeable for layer in search.geometry.layers.values()))
        self.assertIs(ControlLoop._search_scene(loop, search), search)
        self.assertEqual(materialize.call_count, 1)
        for name, value in (('stamp_s', 11.), ('generation', 1)):
            with self.subTest(name=name):
                device.metadata = dict(raw.metadata, **{name: value})
                with self.assertRaisesRegex(ValueError, 'Reference/device observation mismatch'):
                    ControlLoop._search_scene(loop, scene)
        self.assertEqual(materialize.call_count, 1)
        device.metadata = dict(raw.metadata)
        scene.record['stamp_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'Scene metadata changed after observation binding'):
            ControlLoop._search_scene(loop, scene)


class SharedReferenceTests(unittest.TestCase):
    def test_standalone_reference_runtime_uses_shared_worker(self):
        fixture=reference_fixture()
        runtime=ReferenceRuntime(fixture.cfg,fixture.motion,fixture.geometry)
        try:
            deadline=time.monotonic()+10
            while not runtime.worker.ready:
                runtime.worker.poll()
                if time.monotonic()>deadline:self.fail('CPU reference worker startup exceeded its budget')
                time.sleep(.01)
            self.assertTrue(runtime.update(fixture.req,fixture.scene,fixture.feedback,now=10.,clock_rate=.001))
            deadline=time.monotonic()+5
            while runtime.current is None:
                status=runtime.tick(now=10.,clock_domain='synthetic')
                if time.monotonic()>deadline:self.fail('Standalone reference did not become ready: '+str(status))
                time.sleep(.01)
            self.assertEqual(runtime.current['result']['status'],'path_found')
            self.assertFalse(runtime.worker.scenes.pool.leases)
        finally:runtime.close()

    def test_reference_records_cpu_and_releases_scene_after_epoch_change(self):
        fixture=reference_fixture()
        measurements=[];adapted=[]
        metrics=SimpleNamespace(add=lambda name,value:measurements.append((name,value)))
        worker=Worker(fixture.cfg,fixture.geometry,metrics=metrics,
            scene_factory=lambda scene:adapted.append(scene) or scene)
        try:
            deadline=time.monotonic()+10
            while not worker.ready:
                worker.poll()
                if time.monotonic()>deadline:self.fail('CPU reference worker startup exceeded its budget')
                time.sleep(.01)
            paths=[]
            for epoch in (0,1):
                worker.send(dict(job_id=str(epoch),request=fixture.req,scene=fixture.scene,asset=None,
                    epoch=epoch,now=10.,sent_mono=time.monotonic(),clock_rate=.001))
                deadline=time.monotonic()+5;reply=None
                while reply is None:
                    reply=worker.poll()
                    if time.monotonic()>deadline:self.fail('CPU reference query timed out')
                    time.sleep(.01)
                self.assertIsNone(reply[1]['error'])
                result=reply[1]['result']['result'];self.assertEqual(result['status'],'path_found')
                paths.append(result['path']);self.assertFalse(worker.scenes.pool.leases)
            self.assertEqual(paths[0],paths[1]);self.assertEqual(len(adapted),2)
            self.assertEqual([name for name,_ in measurements],['reference_query_cpu']*2)
        finally:
            worker.stop()

    def test_real_cpu_query_releases_read_lease_and_recovers_after_process_exit(self):
        fixture=reference_fixture()
        worker=Worker(fixture.cfg,fixture.geometry)
        def ready():
            deadline=time.monotonic()+10
            while not worker.ready:
                worker.poll()
                if time.monotonic()>deadline:self.fail('CPU reference worker startup exceeded its budget')
                time.sleep(.01)
        try:
            ready()
            worker.send(dict(job_id='reference-test',request=fixture.req,scene=fixture.scene,asset=None,
                epoch=0,now=10.,sent_mono=time.monotonic()))
            deadline=time.monotonic()+5;result=None
            while result is None:
                result=worker.poll()
                if time.monotonic()>deadline:self.fail('CPU reference query timed out')
                time.sleep(.01)
            self.assertIsNone(result[1]['error']);self.assertEqual(result[1]['result']['result']['status'],'path_found')
            self.assertFalse(worker.scenes.pool.leases)
            worker.process.terminate();worker.process.join(timeout=1)
            worker.restart();ready();self.assertTrue(worker.process.is_alive())
        finally:worker.stop()

    def test_frozen_planner_source_clock_rate(self):
        from navigation.planning.runtime import query_time
        for rate in (.5,1.,2.):self.assertAlmostEqual(query_time(dict(now=100.,sent_mono=10.,clock_rate=rate),mono=10.4),100.+.4*rate)
        with self.assertRaises(ValueError):query_time(dict(now=100.,sent_mono=10.,clock_rate=0),mono=10.4)

    def test_local_approach_rechecks_active_and_inflight_paths(self):
        fixture = reference_fixture()
        transform = np.array([[0., -1., 0., 5.], [1., 0., 0., .4], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        req = request('mapped', use_local_approach=True, goal=[3., 4.6, np.pi/2], goal_behavior='stop')

        def scene_at(stamp, hit=None, *, map_dx=0.):
            cells = np.zeros((60, 60), np.int8)
            if hit is not None:
                cells[hit] = 100
            local = bundle(cells).geometry
            association = transform.copy()
            association[0, 3] += map_dx
            local.metadata.update(stamp_s=stamp, association=dict(
                local.metadata['association'], stamp_s=stamp, measurement_stamp_s=stamp-.1,
                map_from_local=association.tolist()))
            return assemble_scene(local, clock_domain='synthetic', settings=environment_config()['scene'])

        first = scene_at(10.)
        for mode in ('retained', 'local_only', 'off_path', 'mapped', 'no_map'):
            for inflight in (False, True):
                with self.subTest(mode=mode, inflight=inflight):
                    static = np.zeros((60, 60), np.int8)
                    if mode in ('local_only', 'mapped'):
                        # The new local hit lies over unknown static cells.
                        static[35:39, 28:32] = -1
                    source = None if mode == 'no_map' else asset(static)
                    goal = req
                    if mode == 'no_map':
                        goal = request('local', goal=[4.2, 2., 0.], goal_behavior='stop')
                    elif mode == 'mapped':
                        goal = dict(req)
                        goal.pop('use_local_approach')
                    with ReferenceRuntime(fixture.cfg, fixture.motion, fixture.geometry,
                                          worker_factory=FakeWorker) as runtime:
                        query = ReferenceQuery(runtime.planning, fixture.geometry)

                        def observe(scene):
                            stamp = scene.record['as_of']
                            runtime.observe(scene, source, now=stamp)
                            self.assertTrue(runtime.update(goal, scene, None, source, now=stamp, clock_rate=.001))
                            return runtime.tick(now=stamp, clock_domain='synthetic')

                        observe(first)
                        job = runtime.worker.job
                        envelope = query.query(job, first, source)
                        result = envelope['result']
                        self.assertEqual('path_found', result['status'], result['reason'])
                        if goal.get('use_local_approach'):
                            self.assertEqual(('odom', 'map'), (result['reference_search_frame'], result['frame_id']))
                            np.testing.assert_allclose(result['path'][-1], req['goal'])
                        if not inflight:
                            runtime.worker.reply(envelope)
                            clear = runtime.tick(now=10., clock_domain='synthetic')
                            self.assertEqual('clear', clear['planning_observation']['status'])
                        if mode == 'retained':
                            runtime.observe(scene_at(10.1, hit=(20, 32)), source, now=10.1)
                            if not inflight:
                                self.assertIsNone(runtime.current)
                            second = scene_at(10.2)
                        elif mode == 'local_only':
                            second = scene_at(10.2, hit=(25, 32), map_dx=.5)
                        else:
                            second = scene_at(10.2, hit=(40, 40) if mode == 'off_path' else (20, 32))
                        if inflight:
                            runtime.worker.reply(envelope)
                        status = observe(second)
                        if mode in ('local_only', 'mapped', 'no_map'):
                            self.assertEqual(job['reference_obstacle_version'], runtime.obstacles.version)
                        elif mode == 'retained':
                            self.assertGreater(runtime.obstacles.version, job['reference_obstacle_version'])
                        if mode in ('retained', 'local_only'):
                            self.assertIsNone(status['result'])
                            self.assertIsNone(status['planning_observation'])
                            self.assertEqual('reference_obstructed_by_new_observation', status['reason'])
                        else:
                            self.assertEqual(envelope, status['result'])

