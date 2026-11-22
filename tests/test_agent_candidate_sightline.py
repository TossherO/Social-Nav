"""Static sightlines are diagnostic evidence for explicit camera targets."""
import unittest
from unittest.mock import patch

import numpy as np

from decision.agent.tools.spatial import static_sightline
from navigation.planning.footprint import VehicleGeometry
from perception.environment.layers import GridSnapshot
from tests import test_agent_visual_context as fixtures


class StaticSightlineTests(unittest.TestCase):
    def test_raycast_reports_intervening_occupancy_and_target_cell_separately(self):
        cells = np.zeros((3, 7), np.int8)
        cells[1, [2, 4, 5]] = [-1, 100, 100]
        grid = GridSnapshot(dict(origin_xy=[10., 20.], resolution=.25), dict(occupancy=cells))
        start = [10.125, 20.375]

        result = static_sightline(grid, start, [11.375, 20.375])
        self.assertEqual(result, dict(status='occupied', endpoint_status='occupied',
            nearest_obstruction_distance_m=.875, first_unknown_distance_m=.375))

        result = static_sightline(grid, [10.875, 20.375], [11.125, 20.375])
        self.assertEqual(result, dict(status='clear', endpoint_status='occupied',
            nearest_obstruction_distance_m=None, first_unknown_distance_m=None))

        result = static_sightline(grid, start, [10.875, 20.375])
        self.assertEqual(result, dict(status='unknown', endpoint_status='free',
            nearest_obstruction_distance_m=None, first_unknown_distance_m=.375))

    def test_raycast_reports_coverage_and_ignores_corner_only_contacts(self):
        cells = np.zeros((4, 4), np.int8)
        cells[0, 1] = 100
        cells[1, 0] = -1
        grid = GridSnapshot(dict(origin_xy=[0., 0.], resolution=1.), dict(occupancy=cells))

        result = static_sightline(grid, [.5, .5], [2.5, 2.5])
        self.assertEqual(result['status'], 'clear')
        result = static_sightline(grid, [.5, 2.5], [4.5, 2.5])
        self.assertEqual(result, dict(status='unknown', endpoint_status='out_of_map',
            nearest_obstruction_distance_m=None, first_unknown_distance_m=3.5))
        result = static_sightline(grid, [-.5, 2.5], [2.5, 2.5])
        self.assertEqual(result, dict(status='unknown', endpoint_status='free',
            nearest_obstruction_distance_m=None, first_unknown_distance_m=0.))


class CandidateSightlineTests(unittest.TestCase):
    def visual_dispatch(self, stage='region', priority='required', with_intrinsics=True):
        fixture = fixtures.VisualContextTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        geometry = VehicleGeometry([[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]],
                                   np.eye(4), source='fixture', assumed=False)
        fixture.nav.geometry = geometry
        camera = dict(position_m=[0.,0.,0.])
        if with_intrinsics:
            camera['view'] = dict(status='approximate_pinhole', edge_yaw_rad=[.5,-.5])
        spatial_inputs = fixture.nav.spatial_inputs
        fixture.nav.spatial_inputs = lambda: dict(spatial_inputs(),geometry=geometry,
                                                  sensors=dict(status='configured',camera=camera))
        target = dict(position=[7,4]) if stage=='point' else dict(id='lobby_area')
        task = dict(goal_id='g',stage=stage,target=target,description='停车看电梯',
                    conditions=dict(visual=[dict(description='看清电梯口',priority=priority)]))
        self.assertEqual(fixture.session.call('record_task',goals=[task])['status'],'recorded')
        fixture.session.call('query_map')
        def submit(yaw):
            return fixture.session.call('navigate',goal_id='g',pose=[7.,4.,yaw],point_ids=['elevator'],
                                        view_id=fixture.session.spatial_view.record['view_id'],
                                        orientation_source='semantic')
        return fixture,submit

    def test_checked_submission_reports_outside_landmark_then_accepts_new_heading(self):
        fixture,submit = self.visual_dispatch()
        refused = submit(0.)
        self.assertEqual(refused['reason'],'candidate_points_outside_horizontal_fov')
        self.assertEqual(refused['outside_point_ids'],['elevator'])
        self.assertIsNone(fixture.nav.state['handle'])
        self.assertEqual(fixture.session.candidate_checks[-1]['pose'],[7.,4.,0.])
        accepted = submit(np.pi)
        self.assertEqual(accepted['status'],'accepted')
        self.assertTrue(accepted['camera_view']['points'][0]['in_horizontal_fov'])
        self.assertFalse(accepted['camera_view']['actual_visibility_verified'])

    def test_first_stop_and_preferences_keep_the_observed_camera_result(self):
        for kwargs in [dict(stage='point'),dict(priority='preferred'),dict(with_intrinsics=False)]:
            with self.subTest(**kwargs):
                _,submit = self.visual_dispatch(**kwargs)
                accepted = submit(0.)
                self.assertEqual(accepted['status'],'accepted')
                expected = None if kwargs.get('with_intrinsics') is False else False
                self.assertIs(accepted['camera_view']['points'][0]['in_horizontal_fov'],expected)

    def test_explicit_point_uses_camera_origin_and_preserves_candidate_footprints(self):
        fixture = fixtures.VisualContextTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        cells = np.zeros((100, 140), np.int8)
        cells[40, 47] = 100
        grid = GridSnapshot(fixture.grid.metadata, dict(occupancy=cells))
        geometry = VehicleGeometry([[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]],
                                   np.eye(4), source='fixture', assumed=False)
        sensors = dict(status='configured', camera=dict(position_m=[.4,-.45,0.],
            view=dict(status='approximate_pinhole', edge_yaw_rad=[.5,-.5])))
        inputs = dict(global_grid=grid, scene=None, pose=[4.,4.5,0.], stamp_s=10.,
                      clock_domain=fixture.nav.clock_domain, geometry=geometry, sensors=sensors)
        fixture.nav.spatial_inputs = lambda: inputs
        session = fixture.session
        session.call('query_map')
        view_id = session.spatial_view.record['view_id']
        args = dict(pose=inputs['pose'], view_id=view_id)

        with patch('decision.agent.core.session.static_sightline', wraps=static_sightline) as raycast:
            ordinary = session.call('check_candidate', **args)
            raycast.assert_not_called()
            self.assertFalse(ordinary['camera_view']['static_sightline_checked'])
            result = session.call('check_candidate', **args, point_ids=['elevator'])
            raycast.assert_called_once()

        self.assertEqual(result['static_footprint'], ordinary['static_footprint'])
        self.assertEqual(result['local_footprint'], ordinary['local_footprint'])
        camera = result['camera_view']
        np.testing.assert_allclose(camera['camera_xy_m'], [4.4,4.05])
        self.assertTrue(camera['static_sightline_checked'])
        self.assertFalse(camera['actual_visibility_verified'])
        point, = camera['points']
        self.assertTrue(point['in_horizontal_fov'])
        self.assertEqual(point['static_sightline']['status'], 'occupied')
        self.assertAlmostEqual(point['static_sightline']['nearest_obstruction_distance_m'],
                               np.hypot(.3,.025))
        remembered = session.candidate_checks[-1]
        self.assertTrue(remembered['static_sightline_checked'])
        self.assertFalse(remembered['actual_visibility_verified'])
        self.assertEqual(remembered['points'], camera['points'])

        sensors['camera']['view'] = dict(status='unavailable', reason='matching_camera_info_required')
        session.call('query_map')
        args = dict(pose=inputs['pose'], view_id=session.spatial_view.record['view_id'])
        ordinary = session.call('check_candidate', **args)['camera_view']
        self.assertEqual(ordinary['reason'], 'matching_camera_geometry_unavailable')
        self.assertFalse(ordinary['static_sightline_checked'])
        self.assertEqual(ordinary['points'], [])
        result = session.call('check_candidate', **args, point_ids=['elevator'])
        camera = result['camera_view']
        self.assertEqual(camera['status'], 'unavailable')
        self.assertTrue(camera['static_sightline_checked'])
        self.assertIsNone(camera['points'][0]['in_horizontal_fov'])
        self.assertEqual(camera['points'][0]['static_sightline'], point['static_sightline'])

        inputs['sensors'] = None
        session.call('query_map')
        result = session.call('check_candidate', pose=inputs['pose'],
                              view_id=session.spatial_view.record['view_id'], point_ids=['elevator'])
        self.assertEqual(result['camera_view']['status'], 'unavailable')
        self.assertFalse(result['camera_view']['static_sightline_checked'])
