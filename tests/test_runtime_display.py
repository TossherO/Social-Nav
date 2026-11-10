"""Display sampling is independent, bounded and never refreshes observation time."""
from collections import deque, OrderedDict
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from perception.visualization.runtime import DisplayBuffer,RuntimeDisplay
from runtime.config import ROOT,runtime_settings


class RuntimeDisplayTests(unittest.TestCase):
    def setUp(self):self.settings=runtime_settings(ROOT/'configs/runtime.yaml')['display']

    def test_independent_limits_latest_items_and_no_copy_for_disabled_group(self):
        buffer=DisplayBuffer(self.settings);made=[]
        for group in ('scan','people','map','rgb','navigation'):
            self.assertTrue(buffer.offer(group,0,lambda group=group:group,now=0.))
        self.assertTrue(buffer.offer('scan',1,lambda:'new scan',now=.1))
        self.assertFalse(buffer.offer('rgb',1,lambda:made.append('unexpected'),now=.1))
        self.assertTrue(buffer.offer('map',1,lambda:'new map',now=.2))
        self.assertEqual(len(buffer.pending),5);self.assertEqual(buffer.pending['scan'],'new scan')
        self.assertEqual(buffer.superseded['scan'],1)
        self.assertFalse(buffer.offer('map',1,lambda:made.append('duplicate'),now=.8))
        buffer.settings['rgb_hz']=0
        self.assertFalse(buffer.offer('rgb',2,lambda:made.append('disabled'),now=2.))
        self.assertFalse(made)
        buffer.close()
        for _ in range(5):self.assertIsNotNone(buffer.take())
        self.assertIsNone(buffer.take())

    def test_fixed_sampling_phase_avoids_half_rate_after_small_input_jitter(self):
        buffer=DisplayBuffer(self.settings)
        accepted=[buffer.offer('scan',i,lambda:i,now=t) for i,t in enumerate([0]+[i*.1-.0001 for i in range(1,20)])]
        self.assertGreaterEqual(sum(accepted),18)

    def test_pending_points_and_images_are_owned_before_source_release(self):
        display=object.__new__(RuntimeDisplay);display.buffer=DisplayBuffer(self.settings)
        display.navigation_lock=threading.Lock();display.navigation_result=None;display.submit_ms=deque()
        points=np.ones((20,4),np.float32);rgb=np.ones((2,3,3),np.uint8)
        row=dict(session_id='s',generation=0,frame_id='odom',stamp_s=10.,health=0)
        people=dict(rgb_stamp_s=9.99,stamp_s=10.,tracks=[])
        grid=SimpleNamespace(metadata=dict(snapshot_id='g',stamp_s=9.95))
        display.submit(points,row,people,rgb=rgb,rgb_frame='camera',grid=grid)
        points[:]=0;rgb[:]=0;people['tracks'].append('changed')
        np.testing.assert_array_equal(display.buffer.pending['scan'][0],np.ones((20,3)))
        self.assertTrue(display.buffer.pending['rgb'][0].all())
        self.assertEqual(display.buffer.pending['rgb'][1],9.99)
        self.assertFalse(display.buffer.pending['people'][0]['tracks'])
        self.assertIs(display.buffer.pending['map'],grid)

    def test_partial_configuration_inherits_and_frequency_validation_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runtime.yaml';path.write_text('display:\n  rgb_hz: 0\n')
            cfg=runtime_settings(path)
            self.assertEqual(cfg['display']['scan_hz'],10.)
            self.assertEqual(cfg['display']['rgb_hz'],0)
            for invalid in (-1,61,float('nan'),True):
                path.write_text(yaml.safe_dump(dict(display=dict(scan_hz=invalid))))
                with self.assertRaises(ValueError):runtime_settings(path)
            for values in (dict(height_mode='invalid'),dict(ground_plane_radius_m=-1),dict(ground_plane_radius_m=float('nan'))):
                path.write_text(yaml.safe_dump(dict(display=values)))
                with self.assertRaises(ValueError):runtime_settings(path)

    def test_rgb_keeps_original_timestamp(self):
        class Message:
            def __init__(self):self.header=SimpleNamespace();self.poses=[]
        packets={name:[] for name in ('rgb','path','reference','trajectory')}
        display=object.__new__(RuntimeDisplay)
        display.rospy=SimpleNamespace(Time=SimpleNamespace(from_sec=lambda value:value))
        display.Image=Message
        display.publishers={name:SimpleNamespace(publish=rows.append) for name,rows in packets.items()}
        display._publish('rgb',(np.zeros((2,3,3),np.uint8),11.5,'camera'))
        self.assertEqual(packets['rgb'][0].header.stamp,11.5)

    def test_idle_timeout_allows_expiry_without_new_sensor_input(self):
        buffer=DisplayBuffer(self.settings)
        self.assertEqual(buffer.take(timeout=0),('idle',None))
        buffer.close();self.assertIsNone(buffer.take(timeout=0))

    def test_navigation_transform_and_empty_paths_use_valid_ros_messages(self):
        try:
            import rospy
            from visualization_msgs.msg import Marker
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.inputs import load_config
        from perception.visualization.navigation import NavigationView
        geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        row=dict(session_id='test',generation=0,frame_id='odom',health=0,stamp_s=10.1,
            position=[2.,2.,0.],quaternion_xyzw=[0,0,0,1],has_map_pose=True,
            map_frame='map',map_position=[5.,6.,0.],map_quaternion_xyzw=[0,0,0,1])
        view=NavigationView(geometry,load_config(),height_mode='plane').update(row,None,mono=0.)
        transforms=[];markers=[];display=object.__new__(RuntimeDisplay);display.rospy=rospy
        display.map_tf=SimpleNamespace(publish=transforms.append)
        display._map_tf_seen=OrderedDict()
        display.publishers={name:SimpleNamespace(get_num_connections=lambda:0)
            for name in ('path','lidar_path','reference','trajectory')}
        display.publishers['navigation']=SimpleNamespace(get_num_connections=lambda:1,publish=markers.append)
        display._publish_navigation(view)
        transform=transforms[0].transforms[0]
        self.assertEqual(transform.header.stamp,rospy.Time.from_sec(row['stamp_s']))
        self.assertEqual((transform.header.frame_id,transform.child_frame_id),('map','odom'))
        self.assertEqual((transform.transform.translation.x,transform.transform.translation.y),(3.,4.))
        clear=markers[0].markers[0]
        self.assertEqual(clear.action,Marker.DELETEALL)
        self.assertEqual(clear.header.frame_id,'odom')
        self.assertEqual(clear.header.stamp,rospy.Time.from_sec(row['stamp_s']))
        added=[m for m in markers[0].markers if m.action==Marker.ADD]
        self.assertTrue(added)
        self.assertFalse(any(m.ns in ('history','reference','trajectory') for m in added))
        self.assertTrue(all(m.points for m in added if m.type==Marker.LINE_LIST))
        display._publish_map_transform(dict(row,has_map_pose=False))
        display._publish_map_transform(dict(row,health=1))
        self.assertEqual(len(transforms),1)
        # Scan, navigation and a changed localization session must not publish
        # the same TF frame pair at the same ROS timestamp twice.
        display._publish_map_transform(row)
        display._publish_map_transform(dict(row,session_id='next',generation=1,map_position=[6.,6.,0.]))
        self.assertEqual(len(transforms),1)
        # A different, even out-of-order measurement keeps its source stamp.
        display._publish_map_transform(dict(row,stamp_s=10.05))
        display._publish_map_transform(dict(row,stamp_s=10.2))
        display._publish_map_transform(dict(row,stamp_s=10.05))
        self.assertEqual(len(transforms),3)
        self.assertEqual([t.transforms[0].header.stamp for t in transforms],
            [rospy.Time.from_sec(x) for x in (10.1,10.05,10.2)])
        display._publish_map_transform(dict(row,stamp_s=10.2,map_frame='other_map'))
        self.assertEqual(len(transforms),4)

    def test_map_topics_use_navigation_cells_and_keep_raw_diagnostic(self):
        from perception.visualization.surface import DisplayGrids
        from tests.test_navigation_display import navigation_display_fixture
        _,grid,asset=navigation_display_fixture()
        def message():
            return SimpleNamespace(header=SimpleNamespace(),info=SimpleNamespace(origin=SimpleNamespace(
                position=SimpleNamespace(),orientation=SimpleNamespace())))
        received={name:[] for name in ('occupancy','raw_occupancy','ground_map')}
        display=object.__new__(RuntimeDisplay);display.grids=DisplayGrids(asset=asset)
        display.rospy=SimpleNamespace(Time=SimpleNamespace(from_sec=lambda value:value))
        display.OccupancyGrid=message
        display.publishers={name:SimpleNamespace(publish=rows.append,get_num_connections=lambda:1) for name,rows in received.items()}
        with patch('perception.visualization.surface.ground_cloud',side_effect=lambda view:view):
            display._publish('map',grid)
        for topic,field in [('occupancy','navigation_occupancy'),('raw_occupancy','occupancy')]:
            output=received[topic][0]
            np.testing.assert_array_equal(np.asarray(output.data).reshape(grid.metadata['shape']),grid.layers[field])
            self.assertEqual(output.header.stamp,grid.metadata['stamp_s'])
            self.assertEqual(output.header.frame_id,grid.metadata['frame_id'])
        np.testing.assert_array_equal(received['ground_map'][0].layers['occupancy'],grid.layers['navigation_occupancy'])

    def test_control_display_keeps_original_clock_anchor_and_owns_metadata(self):
        display=object.__new__(RuntimeDisplay);display.buffer=DisplayBuffer(self.settings)
        display.navigation_lock=threading.Lock();display.navigation_result=None
        row=dict(session_id='s',generation=0,frame_id='odom',stamp_s=10.,health=0)
        display.navigation(None,None,None,None,row,dict(proposal=None),'no_active_goal',
            now=10.,clock_domain='synthetic',clock_rate=1.,clock_mono=2.)
        row['health']=1
        _,packet,_=display.buffer.pending['navigation']
        self.assertEqual(packet['mono'],2.);self.assertEqual(packet['now_s'],10.)
        self.assertEqual(packet['row']['health'],0)
