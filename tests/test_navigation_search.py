"""Native search reuse must preserve input ownership and no-path semantics."""
from pathlib import Path
import unittest
import weakref

import numpy as np

from navigation.planning.backend import FootprintSearch
from navigation.planning.footprint import VehicleGeometry, swept_check, wrap
from navigation.planning.inputs import load_config
from navigation.planning.planner import Planner
from navigation.planning.reference import planning_config
from perception.environment.layers import GridSnapshot
from tests.navigation_fixtures import asset, request


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


class ReferenceTangentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so'
        if not cls.library.is_file():raise unittest.SkipTest('Build the standalone native pose-search library')

    def setUp(self):
        self.config=planning_config(load_config())
        self.config['pose_search']['native_library']=str(self.library)
        self.config['motion_constraints'].update(allow_in_place_rotation=False,allow_reverse=False)
        self.geometry=VehicleGeometry([[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]],np.eye(4),source='synthetic')
        self.start=[2.,2.,0.]
        self.planner=Planner(self.config,self.geometry,require_actual_start=True)

    def query(self, snapshot, goal):
        result=self.planner.query(request(start=self.start,goal=goal),asset=snapshot)
        self.assertEqual(result['status'],'path_found',result)
        self.assertEqual(result['footprint_check']['status'],'free')
        path=np.asarray(result['path'])
        np.testing.assert_allclose(path[0],self.start)
        np.testing.assert_allclose(path[-1,:len(goal)],goal,atol=1e-6)
        return path,result['search_attempts'][-1]

    def test_xy_reference_finishes_with_a_tangent_straight(self):
        for side in (-1,1):
            with self.subTest(side=side):
                path,detail=self.query(asset(),[5.,2.+side*.35])
                turns=side*wrap(np.diff(path[:,2]))
                self.assertGreater(turns.sum(),.05)
                self.assertTrue(np.all(turns>=-1e-8))
                self.assertGreater(np.linalg.norm(path[-1,:2]-path[-2,:2]),2.)
                self.assertAlmostEqual(turns[-1],0.)
                self.assertEqual(detail['expanded'],0)

    def test_fixed_goal_yaw_keeps_the_required_final_turn(self):
        path,_=self.query(asset(),[5.,2.35,0.])
        turns=wrap(np.diff(path[:,2]))
        self.assertGreater(turns.max(),.01)
        self.assertLess(turns.min(),-.01)

    def test_coincident_goal_and_short_forward_goal(self):
        for goal in ([2.,2.],[2.005,2.000001]):
            with self.subTest(goal=goal):
                path,_=self.query(asset(),goal)
                self.assertTrue(np.isfinite(path).all())
                self.assertLess(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum(),.006)

    def test_original_connector_remains_available_under_output_capacity(self):
        goal=[3.166,2.507]
        roomy,_=self.query(asset(),goal)
        self.config['max_path_samples']=27
        self.planner=Planner(self.config,self.geometry,require_actual_start=True)
        bounded,_=self.query(asset(),goal)
        self.assertGreater(roomy[-1,2]-bounded[-1,2],.1)

    def test_obstructed_tangent_keeps_a_certified_detour(self):
        clear,_=self.query(asset(),[5.,2.35])
        cells=np.zeros((60,60),np.int8)
        cells[21,34]=100
        blocked=asset(cells)
        self.assertEqual(swept_check(blocked.snapshot,self.geometry,clear,
            self.config['safety_margin_m'],self.config['max_path_samples'],prefix=False)['status'],'occupied')
        self.query(blocked,[5.,2.35])
        self.query(blocked,[5.,2.35,0.])

    def test_sealed_boundary_and_search_deadline(self):
        cells=np.zeros((60,60),np.int8)
        cells[:,34]=-1
        for seconds,expected in ((.8,'disconnected_search_region'),(1e-9,'curvature_search_time_budget')):
            with self.subTest(timeout=seconds):
                config=planning_config(load_config())
                config['pose_search'].update(native_library=str(self.library),max_search_s=seconds)
                planner=Planner(config,self.geometry,require_actual_start=True)
                result=planner.query(request(start=self.start,goal=[5.,2.35]),asset=asset(cells))
                self.assertEqual(result['search_attempts'][-1]['reason'],expected,result)
                self.assertEqual(result['path'],[])
