"""Unconfirmed live scans are display-only and confirmation is freshness-bound."""
import math
import types
import unittest

import numpy as np

from runtime.initial_preview import PreviewState, PreviewGroundEstimator, preview_points
from runtime.interactive import selected_initial_height
from perception.mapping_localization.initial_pose import map_ground_initial_height


def cloud(stamp=10., frame='rslidar'):
    points=np.array([[1.,0.,0.],[0.,1.,0.]],dtype='<f4')
    return types.SimpleNamespace(header=types.SimpleNamespace(frame_id=frame,
        stamp=types.SimpleNamespace(to_sec=lambda:stamp)),fields=[types.SimpleNamespace(
            name=axis,datatype=7,count=1,offset=i*4) for i,axis in enumerate('xyz')],
        is_bigendian=False,width=2,height=1,point_step=12,row_step=24,data=points.tobytes())


class InitialPreviewTests(unittest.TestCase):
    def test_candidate_applies_lidar_pose_only_to_display_points(self):
        actual=preview_points(cloud(),[0.,0.,math.pi/2,2.,3.,1.],'rslidar')
        np.testing.assert_allclose(actual,[[2.,4.,1.],[1.,3.,1.]],atol=1e-6)
        with self.assertRaisesRegex(ValueError,'frame'):
            preview_points(cloud(frame='other'),[0.,0.,0.,0.,0.,0.],'rslidar')

    def test_confirmation_requires_recent_render_of_latest_candidate(self):
        p=PreviewState(max_age_s=1.)
        p.scan(cloud(),5.);p.select([0.,0.,0.,1.,2.,3.],10.)
        self.assertFalse(p.ready(5.1))
        p.rendered=(p.revision,p.source_stamp,p.received_mono)
        self.assertTrue(p.ready(5.1))
        self.assertFalse(p.ready(6.1))
        self.assertFalse(p.scan(cloud(),6.2))  # A repeated source stamp does not refresh it.
        self.assertFalse(p.ready(6.2))
        p.scan(cloud(stamp=11.),6.3)
        self.assertFalse(p.ready(6.3))  # The last rendered source has expired.
        p.rendered=(p.revision,p.source_stamp,p.received_mono)
        self.assertTrue(p.ready(6.4))
        p.select([0.,0.,0.,4.,5.,6.],12.)
        self.assertFalse(p.ready(6.4))  # Reselection cannot confirm the old preview.
        with self.assertRaisesRegex(ValueError,'measurement time'):
            p.scan(cloud(stamp=12.),6.5,now_source=14.)

    def test_selected_ground_height_tracks_map_elevation_and_keeps_reference_provenance(self):
        ground=np.full((20,40),-1.,np.float32);ground[:,20:]=-.4
        grid=types.SimpleNamespace(metadata=dict(frame_id='map',origin_xy=[0.,0.],resolution=.1,snapshot_id='static'),
            layers=dict(ground_z=ground,ground_support=np.ones_like(ground,dtype=bool)))
        hint=dict(map=dict(frame_id='map'),pose=dict(position_m=[.55,.55,-.2]),
                  source=dict(kind='historical_seed'))
        seed=selected_initial_height(grid,hint,[2.55,.55])
        self.assertEqual(seed['method'],'static_ground_relative_to_map_bound_hint')
        self.assertAlmostEqual(seed['lidar_z_m'],.4,places=6)
        self.assertAlmostEqual(seed['reference_offset_m'],.8,places=6)
        self.assertEqual(seed['reference_source']['kind'],'historical_seed')
        self.assertEqual(seed['static_snapshot_id'],'static')
        self.assertIn('not_measured',seed['offset_scope'])
        self.assertAlmostEqual(map_ground_initial_height(grid,hint,[.55,.55])['lidar_z_m'],-.2)
        state=PreviewState();state.select([0.,0.,0.,2.55,.55,seed['lidar_z_m']],10.,height_seed=seed)
        seed['lidar_z_m']=999.
        self.assertAlmostEqual(state.height_seed['lidar_z_m'],.4,places=6)

    def test_missing_or_discontinuous_ground_falls_back_without_inventing_elevation(self):
        ground=np.full((20,40),-1.,np.float32)
        support=np.ones_like(ground,dtype=bool)
        grid=types.SimpleNamespace(metadata=dict(frame_id='map',origin_xy=[0.,0.],resolution=.1,snapshot_id='static'),
            layers=dict(ground_z=ground,ground_support=support))
        hint=dict(map=dict(frame_id='map'),pose=dict(position_m=[.55,.55,-.2]))
        support[5,25]=False
        fallback=selected_initial_height(grid,hint,[2.55,.55])
        self.assertEqual(fallback['status'],'fallback')
        self.assertEqual(fallback['reason'],'target_selected_cell_has_no_ground_support')
        self.assertEqual(fallback['lidar_z_m'],-.2)
        support[5,25]=True
        ground[:,25:]=-.5
        fallback=selected_initial_height(grid,hint,[2.55,.55])
        self.assertEqual(fallback['reason'],'target_nearby_ground_height_discontinuous')
        self.assertEqual(selected_initial_height(grid,hint,[40.,.5])['reason'],'target_outside_static_map')
        self.assertEqual(selected_initial_height(grid,hint,[2.55,.55],initial_z=-.395),
                         dict(status='available',method='explicit_initial_z',lidar_z_m=-.395))

    def test_fresh_ground_measurement_takes_precedence_over_historical_offset(self):
        from perception.environment.layers import load_config
        config=load_config()
        estimator=PreviewGroundEstimator(0.,0.,config)
        x,y=np.meshgrid(np.linspace(-2.,2.,25),np.linspace(-2.,2.,25))
        points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-.95)]).astype('<f4')
        def scan(stamp):
            result=cloud(stamp)
            result.width=len(points);result.row_step=12*len(points);result.data=points.tobytes()
            return result
        for i in range(5):
            self.assertTrue(estimator.scan(scan(10.+i*.25),5.+i*.25,sensor_frame='rslidar'))
        observed=estimator.current(6.1)
        self.assertIsNotNone(observed)
        self.assertAlmostEqual(observed['height_m'],.95,places=3)
        self.assertEqual(observed['source_stamp_s'],11.)
        self.assertIsNone(estimator.current(7.6))
        grid=types.SimpleNamespace(metadata=dict(frame_id='map',origin_xy=[0.,0.],resolution=.1,snapshot_id='static'),
            layers=dict(ground_z=np.full((20,40),-1.,np.float32),ground_support=np.ones((20,40),dtype=bool)))
        hint=dict(map=dict(frame_id='map'),pose=dict(position_m=[.55,.55,-.2]))
        seed=selected_initial_height(grid,hint,[2.55,.55],observed_height=observed)
        self.assertEqual(seed['method'],'static_ground_plus_current_lidar_ground')
        self.assertAlmostEqual(seed['lidar_z_m'],-.05,places=3)
        self.assertEqual(seed['observed_height']['source_stamp_s'],11.)


if __name__ == '__main__': unittest.main()
