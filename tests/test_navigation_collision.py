"""Batched sweep certificates retain occupied/unknown/boundary semantics."""
import os
import time
import unittest
from unittest.mock import patch
import numpy as np
from navigation.planning.collision import SweepChecker,CertificationBudget
from navigation.planning.footprint import VehicleGeometry,swept_check,polygon_status
from perception.environment.context import layer_binding
from tests.navigation_fixtures import grid


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.geometry=VehicleGeometry([[-.25,-.2],[.35,-.2],[.35,.2],[-.25,.2]],np.eye(4),source='synthetic')

    def scenarios(self):
        paths=[np.array([[2.,2.,0.],[2.5,2.,0.],[3.,2.,0.]]),
            np.array([[2.,2.,0.],[2.3,2.1,.2],[2.6,2.2,.4]]),
            np.array([[.1,.1,0.],[.2,.2,.1]])]
        for value in [0,-1,100]:
            cells=np.zeros((60,60),np.int8);cells[19:22,25]=value
            for path in paths:yield grid(cells),path
        rng=np.random.default_rng(42)
        for _ in range(24):
            cells=np.zeros((60,60),np.int8)
            y,x=rng.integers(10,45,(2,12));cells[y,x]=rng.choice([-1,100],12)
            yaw=rng.uniform(-np.pi,np.pi);t=np.linspace(0,1,12)
            path=np.c_[2.+t*np.cos(yaw),2.+t*np.sin(yaw),yaw+.15*t]
            yield grid(cells),path

    def compare(self,backend):
        checker=SweepChecker(backend);self.addCleanup(checker.close)
        for snapshot,path in self.scenarios():
            expected=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False)
            actual=checker.check(snapshot,self.geometry,path,.05,2000,binding=layer_binding(snapshot))
            self.assertEqual(actual['status'],expected['status'],(backend,path.tolist(),actual,expected))

    def test_cpu_batch_matches_adaptive_reference(self):self.compare('cpu')

    def test_rebound_same_grid_object_invalidates_cached_free_certificate(self):
        snapshot=grid();checker=SweepChecker();path=np.array([[2.,2.,0.],[3.,2.,0.]])
        self.assertEqual(checker.check(snapshot,self.geometry,path,.05,2000,binding=layer_binding(snapshot))['status'],'free')
        snapshot.layers['occupancy'][20,25]=100
        self.assertEqual(checker.check(snapshot,self.geometry,path,.05,2000,binding=layer_binding(snapshot))['status'],'occupied')

    def test_partial_enclosures_reuse_free_prefix_without_hiding_a_late_unknown(self):
        cells=np.zeros((60,60),np.int8);cells[19:22,42]=-1;snapshot=grid(cells)
        t=np.linspace(0,1,160);path=np.c_[2.+2*t,2.+.05*t,.1*t]
        expected=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False)
        checker=SweepChecker();self.addCleanup(checker.close)
        result=checker.check(snapshot,self.geometry,path,.05,2000,binding=layer_binding(snapshot))
        self.assertEqual(result['status'],expected['status']);self.assertEqual(result['status'],'unknown')
        self.assertGreater(result['reused_segments'],100)
        self.assertLess(result['checked_samples'],expected['checked_samples']//2)
        self.assertGreater(result['timing_ms']['cpu_fallback'],0.)

    def test_deadline_rejects_before_starting_any_geometry(self):
        checker=SweepChecker();snapshot=grid();path=np.array([[2.,2.,0.],[3.,2.,0.]])
        budget=CertificationBudget(time.perf_counter()-1,2000)
        with patch.object(checker,'_grid',side_effect=AssertionError('expired work must not start')):
            result=checker.check(snapshot,self.geometry,path,.05,2000,binding='test',budget=budget)
        self.assertEqual(result['status'],'budget_exceeded')
        self.assertEqual(budget.reason,'certification_time_budget_exceeded')

    def test_shared_work_budget_cannot_be_reset_for_a_second_sweep(self):
        cells=np.zeros((60,60),np.int8);cells[20,25]=100;snapshot=grid(cells)
        path=np.array([[2.,2.,0.],[3.,2.,0.]])
        budget=CertificationBudget(time.perf_counter()+10,2)
        first=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False,budget=budget)
        second=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False,budget=budget)
        self.assertEqual(first['status'],'occupied');self.assertEqual(second['status'],'budget_exceeded')
        self.assertEqual(budget.checked,2);self.assertEqual(budget.reason,'certification_work_budget_exceeded')

    def test_adaptive_loop_checks_deadline_internally(self):
        from navigation.planning import footprint
        clock=[0.];original=footprint.footprint_status
        def slow_query(*args):
            result=original(*args);clock[0]+=.02;return result
        # An intentionally unprovable enclosure needs subdivision even though
        # each exact pose is free; expiry must stop within this single sweep.
        with patch('navigation.planning.collision.time.perf_counter',side_effect=lambda:clock[0]), \
                patch.object(footprint,'footprint_status',side_effect=slow_query):
            budget=CertificationBudget(.05,2000)
            snapshot=grid();snapshot.layers['occupancy'][26,25]=-1
            path=np.array([[2.,2.,0.],[3.,2.,.1]])
            result=swept_check(snapshot,self.geometry,path,.3,2000,prefix=False,budget=budget)
        self.assertEqual(result['status'],'budget_exceeded')
        self.assertEqual(budget.reason,'certification_time_budget_exceeded')
        self.assertLessEqual(budget.checked,3)

    def test_single_pose_cannot_return_free_after_its_query_exceeds_deadline(self):
        clock=[0.]
        def late(*args):clock[0]=.1;return 'free'
        with patch('navigation.planning.collision.time.perf_counter',side_effect=lambda:clock[0]), \
                patch('navigation.planning.footprint.footprint_status',side_effect=late):
            budget=CertificationBudget(.05,2000)
            result=swept_check(grid(),self.geometry,np.array([[2.,2.,0.]]),.05,2000,budget=budget)
        self.assertEqual(result['status'],'budget_exceeded')

    def test_gpu_refinement_requires_all_covering_leaves_to_be_free(self):
        checker=SweepChecker();path=np.array([[2.,2.,0.],[3.,2.,.2],[3.5,2.1,.3]])
        for value in (0,-1,100):
            cells=np.zeros((60,60),np.int8);cells[20,27]=value;snapshot=grid(cells)
            certified=np.array([True,False,False])
            def oracle(snapshot,polygons,margins):
                return np.array([0 if polygon_status(snapshot,p,float(m))=='free' else 1 for p,m in zip(polygons,margins)])
            with patch.object(checker,'_cuda_batch',side_effect=oracle):
                batches,count=checker._refine(snapshot,self.geometry,path,.05,certified,2000,None)
            self.assertLessEqual(batches,4);self.assertLessEqual(count,4*2000)
            if value==0:self.assertTrue(certified.all())
            else:self.assertFalse(certified.all())
            actual=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False,certified=certified)
            expected=swept_check(snapshot,self.geometry,path,.05,2000,prefix=False)
            self.assertEqual(actual['status'],expected['status'])

    def test_gpu_refinement_checks_deadline_between_bounded_batches(self):
        checker=SweepChecker();clock=[0.];certified=np.array([True,False])
        def unresolved(snapshot,polygons,margins):
            clock[0]+=.02;return np.ones(len(polygons),np.int32)
        with patch('navigation.planning.collision.time.perf_counter',side_effect=lambda:clock[0]), \
                patch.object(checker,'_cuda_batch',side_effect=unresolved):
            budget=CertificationBudget(.025,2000)
            batches,_=checker._refine(grid(),self.geometry,np.array([[2.,2.,0.],[3.,2.,.2]]),.05,certified,2000,budget)
        self.assertEqual(batches,2);self.assertFalse(certified.all())
        self.assertEqual(budget.reason,'certification_time_budget_exceeded')

    @unittest.skipUnless(os.environ.get('SOCIAL_NAV_TEST_CUDA')=='1','Explicit CUDA test required')
    def test_cuda_batch_matches_adaptive_reference(self):self.compare('cuda')


if __name__=='__main__':unittest.main()
