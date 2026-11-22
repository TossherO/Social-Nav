import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

from navigation.planning.reference import ReferenceRuntime
from navigation.planning.runtime import ReferenceQuery, decode_display_map
from perception.environment.storage import StaticEnvironmentMap, content_hash
from tests.navigation_fixtures import FakeWorker, asset, bundle, request
from tests.test_runtime_reference import reference_fixture


class ReferenceDisplayMapTests(unittest.TestCase):
    def setUp(self):
        self.fixture = reference_fixture()
        cells = np.zeros((60, 60), np.int8)
        cells[45, 44:46] = 100
        cells[46, 45] = -1
        cells[20, 43] = 100
        self.asset = asset(cells)
        conditional = np.zeros(cells.shape, bool)
        conditional[45, 44:46] = True
        self.asset.snapshot.layers['conditional_obstacle'] = conditional
        local = np.zeros_like(cells)
        local[45, 44] = 100
        self.scene = bundle(local)
        self.req = request('mapped', goal=[3.65, 2., 0.], goal_behavior='stop', valid_until_s=12.)

    def test_query_captures_effective_occupancy_and_conditional_layer(self):
        query = ReferenceQuery(self.fixture.cfg, self.fixture.geometry)
        job = dict(job_id=3, epoch=0, request=self.req, now=10., sent_mono=time.monotonic(), clock_rate=.001,
                   reference_obstacles=(40 * 60 + 40,), display_enabled=True)
        result = query.query(job, self.scene, self.asset)['result']
        self.assertEqual('path_found', result['status'])
        snapshot = query.display_map
        self.assertEqual((3, 10., 'map', (60, 60), .1, (0., 0.)),
                         tuple(snapshot[k] for k in ('job_id', 'stamp_s', 'frame_id', 'shape', 'resolution', 'origin_xy')))
        self.assertEqual(result['grid_snapshot_id'], snapshot['snapshot_id'])
        self.assertEqual(result['result_binding'], snapshot['result_binding'])
        self.assertNotIn('cells', result)
        cells, conditional = decode_display_map(snapshot)
        expected_cells = self.asset.snapshot.layers['occupancy'].copy()
        expected_cells[45, 45] = expected_cells[46, 45] = 0
        expected_cells[40, 40] = 100
        expected_conditional = self.asset.snapshot.layers['conditional_obstacle'].copy()
        expected_conditional[45, 44] = False
        np.testing.assert_array_equal(expected_cells, cells)
        np.testing.assert_array_equal(expected_conditional, conditional)
        self.assertFalse(cells.flags.writeable)
        self.assertFalse(conditional.flags.writeable)

        failure = dict(self.req, goal=[4.35, 2., 0.])
        result = query.query(dict(job, job_id=4, request=failure), self.scene, self.asset)['result']
        self.assertEqual('goal_footprint_occupied', result['reason'])
        self.assertEqual(result['reason'], query.display_map['reason'])
        self.assertEqual(4, query.display_map['job_id'])
        self.assertTrue(result['reference_work_reused'])

        unavailable = query.query(dict(job, job_id=5), bundle(associated=False), self.asset)['result']
        self.assertEqual('association_unavailable', unavailable['reason'])
        self.assertIsNone(query.display_map)
        query.query(dict(job, display_enabled=False), self.scene, self.asset)
        self.assertIsNone(query.display_map)

    def test_runtime_uses_the_map_for_the_reference_it_keeps_displaying(self):
        runtime = ReferenceRuntime(self.fixture.cfg, self.fixture.motion, self.fixture.geometry, worker_factory=FakeWorker)
        self.assertFalse(runtime.display_enabled)
        runtime.display_enabled = True
        runtime.diagnostics = True
        query = ReferenceQuery(runtime.planning, self.fixture.geometry)

        def update(goal, *, locked=False):
            self.assertTrue(runtime.update(goal, self.scene, None, self.asset, now=10., clock_rate=.001,
                                           follow_key='target', locked_follow_goal=locked))
            runtime.last_sent = -np.inf
            return runtime.tick(now=10., clock_domain='synthetic')

        def reply():
            job = runtime.worker.job
            self.assertTrue(job['display_enabled'])
            runtime.worker.reply(query.query(job, self.scene, self.asset))
            runtime.worker.response['display_map'] = query.display_map
            status = runtime.tick(now=10., clock_domain='synthetic')
            self.assertNotIn('display_map', runtime.last_reply[1])
            self.assertNotIn('display_map', status)
            return status

        update(self.req)
        active = reply()['result']
        self.assertEqual(1, runtime.display_map['job_id'])
        self.assertEqual(active['result']['result_binding'], runtime.display_map['result_binding'])
        blocked = dict(self.req, goal_revision=1, goal=[4.35, 2., 0.])
        update(blocked)
        self.assertEqual(active, reply()['result'])
        self.assertEqual(1, runtime.display_map['job_id'])

        self.assertIsNone(update(blocked, locked=True)['result'])
        self.assertEqual(2, runtime.display_map['job_id'])
        self.assertEqual('goal_footprint_occupied', runtime.display_map['reason'])

        update(dict(self.req, goal_revision=2, goal=[3.8, 2., 0.]))
        self.assertIsNone(runtime.display_map)
        runtime.cancel()
        self.assertIsNone(runtime.display_map)

    def test_worker_sends_map_separately_from_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.asset.snapshot.save(path/'layers.npz')
            (path/'manifest.yaml').write_text(yaml.safe_dump(dict(kind='static_environment',
                time_scope='offline_final_prior', snapshot_id='static-a', frame_id='map',
                layers_sha256=content_hash(path/'layers.npz'))))
            saved = StaticEnvironmentMap(path)
            with ReferenceRuntime(self.fixture.cfg, self.fixture.motion, self.fixture.geometry) as runtime:
                runtime.display_enabled = True
                deadline = time.monotonic() + 10.
                while not runtime.worker.ready:
                    runtime.worker.poll()
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                self.assertTrue(runtime.update(self.req, self.scene, None, saved, now=10., clock_rate=.001))
                deadline = time.monotonic() + 5.
                while runtime.current is None:
                    status = runtime.tick(now=10., clock_domain='synthetic')
                    self.assertLess(time.monotonic(), deadline, str(status))
                    time.sleep(.01)
                self.assertEqual(runtime.current['result']['result_binding'], runtime.display_map['result_binding'])
                cells, conditional = decode_display_map(runtime.display_map)
                self.assertEqual(100, cells[45, 44])
                self.assertTrue(conditional[45, 45])
                self.assertFalse(runtime.worker.scenes.pool.leases)
