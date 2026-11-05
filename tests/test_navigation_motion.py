"""Causal motion estimation and shared SE(2) geometry, without a trajectory solver."""
import unittest
import numpy as np
from navigation.planning.motion import load_motion_config,estimate_motion,twists,interpolate_se2
from navigation.planning.footprint import VehicleGeometry
from tests.test_navigation_planning import bundle


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.motion=load_motion_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')

    def test_velocity_uses_past_valid_algorithm_poses_and_stops_at_failure(self):
        rows=[dict(stamp_s=t,health=0,session_id='test',generation=0,frame_id='odom',
            position=[t/2,0,0],quaternion_xyzw=[0,0,0,1]) for t in [9.7,9.8,9.9,10.,10.1]]
        rows[-1]['position']=[100,0,0]
        s=estimate_motion(rows,bundle(),self.geometry,self.motion)
        np.testing.assert_allclose(s['velocity'],[.5,0,0],atol=1e-9)
        self.assertEqual(s['window_s'][-1],10.)
        rows[2]['health']=2
        self.assertEqual(estimate_motion(rows,bundle(),self.geometry,self.motion)['status'],'unavailable')

    def test_se2_arc_has_zero_lateral_twist_and_endpoint_preserved(self):
        poses=np.array([[0.,0.,0.],[1.,1.,np.pi/2]])
        rates=twists(poses,np.array([2.]));self.assertAlmostEqual(rates[0,1],0.,places=10)
        dense,_=interpolate_se2(poses,.1,500)
        np.testing.assert_allclose(dense[-1],poses[-1],atol=1e-10)
        self.assertGreater(len(dense),20)

    def test_circle_connections_preserve_both_endpoints_and_motion_in_both_directions(self):
        from navigation.planning.curvature import connections,motion_valid
        start=np.array([0.,0.,0.]);count=0
        for goal in ([3.,1.,.4],[0.,2.,np.pi/2],[0.,0.,1.]):
            for reverse in (False,True):
                policy=dict(self.motion,allow_reverse=reverse)
                for poses,length in connections(start,goal,policy['min_turning_radius_m'],reverse=reverse):
                    np.testing.assert_allclose(poses[0],start,atol=1e-10)
                    np.testing.assert_allclose(poses[-1,:2],goal[:2],atol=1e-8)
                    self.assertLess(abs((poses[-1,2]-goal[2]+np.pi)%(2*np.pi)-np.pi),1e-8)
                    self.assertTrue(motion_valid(poses,policy));count+=1
        self.assertGreater(count,20)
