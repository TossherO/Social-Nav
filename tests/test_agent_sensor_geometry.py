"""Sensor projection and task geometry; no device, network, or robot output."""
import copy
import math
import unittest
import numpy as np
from decision.agent import CONFIG_DIR
from decision.agent.core.tasks import TaskGoals,centroid,distance_to_region
from decision.agent.tools.places import Catalog
from decision.agent.tools.sensors import sensor_context
from navigation.planning.footprint import VehicleGeometry


class SensorGeometryTests(unittest.TestCase):
    def setUp(self):
        transform=np.eye(4);transform[:3,3]=[.67,.33,-.2]
        self.geometry=VehicleGeometry([[-.28,-.33],[.82,-.33],[.82,.33],[-.28,.33]],transform,
                                      source={'status':'configured','vehicle_id':'wheelchair_tl_local'},assumed=False)
        self.info=dict(width=640,height=480,K=[600.,0.,320.,0.,600.,240.,0.,0.,1.],
                       frame_id='camera_color_optical_frame',stamp_ns=10_000_000_000,distortion_model='plumb_bob')

    def test_camera_and_imu_come_from_calibration_chain(self):
        sensors=sensor_context(self.geometry,self.info,rgb_shape=[480,640,3],
                               rgb_frame=self.info['frame_id'],rgb_stamp=10.1)
        np.testing.assert_allclose(sensors['camera']['position_m'],[.66065772,.3644858,-.34620628],atol=1e-7)
        self.assertEqual(sensors['camera']['view']['status'],'approximate_pinhole')
        self.assertGreater(sensors['camera']['view']['edge_yaw_rad'][0],sensors['camera']['view']['edge_yaw_rad'][1])
        self.assertEqual(len(sensors['imu']['position_m']),3)
        self.assertIn('not_occlusion',sensors['camera']['view']['interpretation'])

    def test_mismatched_or_future_intrinsics_never_create_fov(self):
        for changes in (dict(rgb_shape=[240,320,3]),dict(rgb_frame='wrong'),dict(rgb_stamp=9.)):
            args=dict(rgb_shape=[480,640,3],rgb_frame=self.info['frame_id'],rgb_stamp=10.1)
            args.update(changes)
            view=sensor_context(self.geometry,self.info,**args)['camera']['view']
            self.assertEqual(view['status'],'unavailable')

    def test_assumed_geometry_is_not_presented_as_current_sensor_mount(self):
        geometry=VehicleGeometry(self.geometry.footprint,self.geometry.base_from_lidar,source='historical',assumed=True)
        self.assertEqual(sensor_context(geometry,self.info)['status'],'unavailable')

    def test_other_vehicle_never_inherits_this_mount(self):
        geometry=VehicleGeometry(self.geometry.footprint,self.geometry.base_from_lidar,
                                 source={'vehicle_id':'other'},assumed=False)
        self.assertEqual(sensor_context(geometry,self.info)['reason'],'sensor_calibration_vehicle_identity_mismatch')

    def test_region_center_and_reference_distance_are_checked(self):
        catalog=Catalog.load(CONFIG_DIR/'simulation.json')
        region=catalog.regions['lobby_area']; middle=centroid(region['polygon'])
        self.assertEqual(distance_to_region(middle,region['polygon']),0.)
        task=TaskGoals(catalog)
        task.record([dict(goal_id='r',kind='region',target_id='lobby_area',description='center',position_rule='region_centroid')])
        self.assertTrue(task.geometry(task.goals[0]['requirement'],[*middle,0.])['position_met'])
        shifted=np.array(middle)+[1.,0.]
        self.assertFalse(task.geometry(task.goals[0]['requirement'],[*shifted,0.])['position_met'])
        nearby=TaskGoals(catalog)
        nearby.record([dict(goal_id='n',kind='reference',target_id='lobby_area',description='near',max_distance_m=.5)])
        self.assertTrue(nearby.geometry(nearby.goals[0]['requirement'],[*middle,0.])['position_met'])
        self.assertFalse(nearby.geometry(nearby.goals[0]['requirement'],[100.,100.,0.])['position_met'])

    def test_facing_region_does_not_derive_heading_from_centroid(self):
        catalog=Catalog.load(CONFIG_DIR/'simulation.json');middle=centroid(catalog.regions['lobby_area']['polygon'])
        task=TaskGoals(catalog)
        task.record([dict(goal_id='p',kind='point',target_id='',position=[middle[0]-1,middle[1]],description='face',facing='lobby_area')])
        req=task.goals[0]['requirement']
        self.assertIsNone(task.geometry(req,[middle[0]-1,middle[1],0.])['facing_met'])
        self.assertIsNone(task.geometry(req,[middle[0]-1,middle[1],math.pi])['facing_met'])

