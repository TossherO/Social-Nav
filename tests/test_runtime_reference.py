import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.runtime import Worker
from tests.navigation_fixtures import bundle, request


def reference_fixture():
    cfg=load_config();cfg['algorithm']='hybrid_astar';cfg['local_replanning']['enabled']=False
    cfg['pose_search']['native_library']=str(Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
    geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
    feedback=dict(status='valid',source='synthetic_actual_plant',pose=[2.,2.,0.],velocity=[0.,0.],stamp_s=10.,
        clock_domain='synthetic',local_identity=['test',0,'odom'])
    return SimpleNamespace(cfg=cfg,motion=load_motion_config(),geometry=geometry,scene=bundle(),
        req=request('local',goal=[4.,2.,0.],goal_behavior='stop'),feedback=feedback)


class SharedReferenceTests(unittest.TestCase):
    def test_standalone_reference_runtime_uses_shared_worker(self):
        from navigation.planning.reference import ReferenceRuntime
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

