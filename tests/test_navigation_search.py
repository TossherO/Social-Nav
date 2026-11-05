"""Native search reuse must preserve input ownership and no-path semantics."""
from pathlib import Path
import unittest
import weakref

import numpy as np

from navigation.planning.backend import FootprintSearch
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from perception.environment.layers import GridSnapshot


class SearchCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        if not cls.library.is_file():raise unittest.SkipTest('Build the standalone native pose-search library')

    def setUp(self):
        self.config=load_config()
        self.config['pose_search'].update(max_expansions=10000,max_search_s=.08)
        self.cells=np.zeros((64,64),np.int8);self.cells[:,32]=100
        self.costs=np.where(self.cells==0,0,254).astype(np.uint8)
        self.clearance=np.zeros(self.cells.shape,np.float32)
        self.grid=GridSnapshot(dict(shape=[64,64],origin_xy=[0.,0.],resolution=.1),dict(occupancy=self.cells))
        self.geometry=VehicleGeometry([[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]],np.eye(4),source='synthetic')
        self.backend=FootprintSearch(self.library);self.addCleanup(self.backend.close)

    def search(self,goal=(4.8,3.2)):
        return self.backend.search(self.grid,self.geometry,self.config,self.clearance,self.costs,[1.6,3.2,0.],goal)

    def test_complete_field_proves_disconnection_and_reuse_checks_actual_inputs(self):
        path,first=self.search()
        self.assertIsNone(path);self.assertEqual(first['reason'],'disconnected_search_region')
        self.assertEqual(first['expanded'],0);self.assertFalse(first['heuristic_cache_reused'])
        self.assertTrue(self.search()[1]['heuristic_cache_reused'])
        self.cells[2,2]=-1
        self.assertFalse(self.search()[1]['heuristic_cache_reused'])
        self.assertTrue(self.search()[1]['heuristic_cache_reused'])
        self.costs[4,4]=35
        self.assertFalse(self.search()[1]['heuristic_cache_reused'])
        self.assertFalse(self.search(goal=(5.,3.2))[1]['heuristic_cache_reused'])
        self.grid.metadata['origin_xy']=[.1,0.]
        self.assertFalse(self.search(goal=(5.,3.2))[1]['heuristic_cache_reused'])

    def test_partial_wavefront_is_neither_cached_nor_a_disconnection_certificate(self):
        self.config['pose_search']['max_search_s']=1e-9
        for _ in range(2):
            path,detail=self.search()
            self.assertIsNone(path)
            self.assertEqual(detail['reason'],'curvature_search_time_budget')
            self.assertFalse(detail['heuristic_cache_reused'])

    def test_disconnection_shortcut_requires_base_centre_inside_footprint(self):
        self.geometry=VehicleGeometry(self.geometry.footprint+[.7,0.],np.eye(4),source='offset synthetic body')
        self.config['pose_search']['max_expansions']=1
        path,detail=self.search()
        self.assertIsNone(path);self.assertNotEqual(detail['reason'],'disconnected_search_region')
        self.assertEqual(detail['expanded'],1)

    def test_native_cache_keeps_owned_data_without_retaining_the_snapshot(self):
        reference=weakref.ref(self.grid)
        self.search();self.grid=None
        self.assertIsNone(reference())
        self.grid=GridSnapshot(dict(shape=[64,64],origin_xy=[0.,0.],resolution=.1),dict(occupancy=self.cells.copy()))
        self.assertTrue(self.search()[1]['heuristic_cache_reused'])
