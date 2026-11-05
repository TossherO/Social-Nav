import time
import unittest
from pathlib import Path
from runtime.reference import SharedReferenceWorker


class SharedReferenceTests(unittest.TestCase):
    def test_real_cpu_query_releases_read_lease_and_recovers_after_process_exit(self):
        from tests.test_navigation_dwa import DwaTests
        fixture=DwaTests();fixture.setUp()
        fixture.cfg['pose_search']['native_library']=str(Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
        worker=SharedReferenceWorker('reference',fixture.cfg,fixture.motion,fixture.geometry)
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

