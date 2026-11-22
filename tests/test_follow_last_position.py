"""Mapped loss recovery uses the exact endpoint and the normal curvature planner."""
from pathlib import Path
import unittest

import numpy as np

from navigation.planning.footprint import VehicleGeometry, swept_check
from navigation.planning.inputs import load_config
from navigation.planning.planner import Planner
from navigation.search import check_last_position, follow_candidates
from perception.environment.context import fingerprint
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from tests.navigation_fixtures import asset, bundle, request


class LastPositionPlanningTests(unittest.TestCase):
    def setUp(self):
        self.planning = load_config()
        self.planning['algorithm'] = 'hybrid_astar'
        self.planning['pose_search']['native_library'] = str(
            Path(__file__).resolve().parents[1]/'outputs/runtime/build/planning/libsocial_nav_pose_search.so')
        self.geometry = VehicleGeometry([[-.15,-.15],[.15,-.15],[.15,.15],[-.15,.15]], np.eye(4), source='synthetic')
        self.goal = request('mapped', goal=[4.5, 2.], goal_behavior='stop', completion_policy='goal',
                            follow_last_evidence_binding='confirmed-visual-source')

    def plan(self, scene, static):
        return Planner(self.planning,self.geometry,require_actual_start=True).query(
            self.goal, scene=scene, asset=static, now=10., clock_domain='synthetic')

    def test_fixed_goal_beyond_current_local_window_has_one_complete_mapped_route(self):
        scene = bundle(np.zeros((60,35), dtype=np.int8))
        static = asset()
        result = self.plan(scene, static)
        self.assertEqual(result['status'], 'path_found', result['reason'])
        self.assertGreater(result['path_length_m'], 1.5)
        np.testing.assert_allclose(result['path'][-1][:2], self.goal['goal'])
        self.assertEqual(result['footprint_check']['status'], 'free')
        self.assertTrue(result['current_local_path']['valid'])
        self.assertFalse(result['current_local_path']['complete'])

    def test_wall_beyond_previous_step_requires_a_certified_detour(self):
        cells = np.zeros((60,60), dtype=np.int8)
        cells[16:28,32] = 100
        static = asset(cells)
        result = self.plan(bundle(), static)
        self.assertEqual(result['status'], 'path_found', result.get('search_attempts', result['reason']))
        np.testing.assert_allclose(result['path'][-1][:2], self.goal['goal'])
        self.assertEqual(swept_check(static.snapshot,self.geometry,result['path'],
            self.planning['safety_margin_m'],self.planning['max_path_samples'],prefix=False)['status'], 'free')
        self.assertGreater(result['path_length_m'], 2.5)

    def test_full_path_cannot_cross_unknown_and_occupied_endpoint_has_no_substitute(self):
        cells = np.zeros((60,60), dtype=np.int8)
        cells[:,38] = -1
        unknown = self.plan(bundle(np.zeros((60,35), dtype=np.int8)), asset(cells))
        self.assertNotEqual(unknown['status'], 'path_found')
        cells = np.zeros((60,60), dtype=np.int8)
        cells[20,45] = 100
        occupied = self.plan(bundle(cells), asset())
        self.assertEqual(occupied['reason'], 'goal_occupied')
        self.assertEqual(occupied['path'], [])
        self.assertNotIn('follow_goal_adjustment', occupied)

    def test_corner_heading_is_transformed_and_required_only_when_present(self):
        scene = bundle()
        grid = scene.geometry
        mapping = np.array([[0.,-1.,0.,5.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        grid.metadata['association']['map_from_local'] = mapping.tolist()
        scene = assemble_scene(grid,clock_domain='synthetic',settings=environment_config()['scene'])
        static = asset()
        last = dict(stamp_s=9.9,clock_domain='synthetic',local_identity=['test',0,'odom'],position_xy_m=[4.,2.])
        last['binding'] = fingerprint(last)
        plain = follow_candidates(scene,static,self.geometry,last,planning=self.planning,local_reid=True)
        turned = follow_candidates(scene,static,self.geometry,last,planning=self.planning,local_reid=True,
                                   loss=dict(departure_yaw_rad=0.))
        np.testing.assert_allclose(plain['candidates'][1]['pose'], [3.,4.])
        np.testing.assert_allclose(turned['candidates'][1]['pose'], [3.,4.,np.pi/2])
        check_last_position(scene,static,self.geometry,[3.,2.,0.],self.planning,now=10.,clock_domain='synthetic')
        with self.assertRaisesRegex(ValueError, 'follow_last_position_reached'):
            check_last_position(scene,static,self.geometry,[3.,2.,np.pi/2],self.planning,now=10.,clock_domain='synthetic')
