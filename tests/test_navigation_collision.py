"""Batched sweep certificates retain occupied/unknown/boundary semantics."""
import os
import unittest
import numpy as np
from navigation.planning.collision import SweepChecker
from navigation.planning.footprint import VehicleGeometry,swept_check
from perception.environment.context import layer_binding
from tests.test_navigation_planning import grid


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

    @unittest.skipUnless(os.environ.get('SOCIAL_NAV_TEST_CUDA')=='1','Explicit CUDA test required')
    def test_cuda_batch_matches_adaptive_reference(self):self.compare('cuda')


if __name__=='__main__':unittest.main()
