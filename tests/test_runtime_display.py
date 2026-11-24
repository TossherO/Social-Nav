"""Display sampling is independent, bounded and never refreshes observation time."""
from collections import deque, OrderedDict
import json
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

    def test_interactive_and_standalone_views_default_to_navigation_over_static_map(self):
        directory=ROOT/'perception/visualization/configs'
        for name in ('interactive','runtime'):
            view=yaml.safe_load((directory/(name+'.rviz')).read_text())
            manager=view['Visualization Manager']
            self.assertIn('social_nav_rviz/SessionPanel',[panel['Class'] for panel in view['Panels']])
            by_topic={layer['Topic']:layer for layer in manager['Displays'] if 'Topic' in layer}
            main=by_topic['/social_nav_runtime_view/navigation_basemap']
            static_topic='/social_nav_interactive/static_basemap' if name=='interactive' else '/social_nav_runtime_view/static_basemap'
            static=by_topic[static_topic]
            self.assertEqual((main['Name'],static['Name']),('Navigation basemap','Static map'))
            self.assertTrue(main['Enabled'] and static['Enabled'])
            self.assertEqual((main['Color Transformer'],main['Decay Time']),('RGB8',0))
            search=by_topic['/social_nav_runtime_view/reference_search_map']
            self.assertEqual((search['Name'],search['Class']),('Reference search snapshot','rviz/PointCloud2'))
            self.assertFalse(search['Enabled'])
            window=next(layer for layer in manager['Displays'] if
                        layer.get('Marker Topic')=='/social_nav_runtime_view/navigation_window')
            self.assertEqual((window['Name'],window['Class']),('Navigation window','rviz/Marker'))
            self.assertTrue(window['Enabled'])
            self.assertFalse(by_topic['/social_nav_runtime_view/ground_map']['Enabled'])
            if name=='interactive':
                tools=[tool for tool in manager['Tools'] if tool['Class'] in
                       ('social_nav_rviz/GroundInitialPose','social_nav_rviz/GroundGoal')]
                self.assertEqual(len(tools),2)
                self.assertTrue(all('/social_nav_interactive/ground_map' in tool.values() for tool in tools))

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

    def test_environment_clear_supersedes_a_queued_map_without_waiting_for_display_rate(self):
        buffer=DisplayBuffer(self.settings)
        self.assertTrue(buffer.offer('map','old',lambda:'old',now=1.))
        buffer.clear_environment('odom',10.)
        self.assertEqual(buffer.take(),('map_clear',('odom',10.)))
        self.assertFalse(buffer.offer('map','new',lambda:'new',now=1.01))
        self.assertTrue(buffer.offer('map','new',lambda:'new',now=2.))
        self.assertEqual(buffer.take(),('map','new'))

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
        display._show_basemap=lambda:None
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

    def basemap_display(self, asset):
        try:
            import rospy
        except ImportError:
            self.skipTest('source perception/scripts/ros1_env.sh to check ROS display messages')
        display=object.__new__(RuntimeDisplay)
        display.rospy=rospy
        display.buffer=DisplayBuffer(self.settings)
        display.clock_rate=2.;display.max_grid_age=.5;display.max_association_age=.55
        display.local_key=display.local_frame=display.search_key=display.basemap_status=None
        display.local_surface=display.search_surface=None
        display.navigation_view=SimpleNamespace(reference_max_age_s=5.)
        display.frozen=False;display.frozen_snapshot=display.frozen_row=None
        display.frozen_association=None
        display.last_navigation=display.last_view=None
        display.last_environment_grid=None
        display.error=None
        display.grids=SimpleNamespace(asset=asset)
        display.String=SimpleNamespace
        received={name:[] for name in ('navigation_basemap','navigation_window','static_basemap','basemap_source','reference_search_map')}
        display.publishers={name:SimpleNamespace(publish=messages.append) for name,messages in received.items()}
        return display,received

    def test_static_background_publishes_once_without_map_localization(self):
        from perception.environment.layers import GridSnapshot
        from perception.sensors.pointcloud import pointcloud_fields
        from tests.test_environment_navigation import prior_asset
        asset=prior_asset()
        source=asset.snapshot
        asset.navigation_snapshot=GridSnapshot(dict(source.metadata,origin_xy=[1.,2.],shape=[4,3]),
            {name:values[20:24,10:13] for name,values in source.layers.items()})
        display,received=self.basemap_display(asset)
        display.publish_static=True
        display.buffer.take=lambda **kwargs: next(entries)
        entries=iter([('idle',None),('idle',None),None])
        display._work()
        self.assertIsNone(display.error)
        self.assertEqual(len(received['static_basemap']),1)
        self.assertEqual(received['static_basemap'][0].header.frame_id,'map')
        self.assertEqual(received['static_basemap'][0].header.stamp.to_sec(),0.)
        fields=pointcloud_fields(received['static_basemap'][0],('x','y'))
        self.assertEqual(len(fields['x']),12)
        np.testing.assert_allclose(np.unique(fields['x']),[1.05,1.15,1.25])
        np.testing.assert_allclose(np.unique(fields['y']),[2.05,2.15,2.25,2.35])
        self.assertFalse(received['navigation_basemap'])
        self.assertFalse(received['reference_search_map'])
        self.assertEqual([json.loads(m.data) for m in received['basemap_source']],
            [dict(static_available=True,static_frame='map',local_available=False,frozen=False,search_query_id=None)])

    def test_overlay_uses_grid_geometry_and_valid_current_floor_association(self):
        from perception.environment.layers import GridSnapshot
        from tests.test_environment_navigation import observation,prior_asset
        asset=prior_asset()
        source=asset.snapshot
        asset.navigation_snapshot=GridSnapshot(dict(source.metadata,origin_xy=[1.5,1.4],shape=[4,3]),
            {name:values[14:18,15:18] for name,values in source.layers.items()})
        display,received=self.basemap_display(asset)
        from visualization_msgs.msg import Marker
        from perception.visualization.surface import LOCAL_FREE_COLOR,LOCAL_OCCUPIED_COLOR
        cells=np.array([[0,-1,100],[-1,-1,-1]],dtype=np.int8)
        grid=observation(cells,origin=(.2,.4))
        transform=np.array([[0.,-1,0,2],[1,0,0,1],[0,0,1,-.3],[0,0,0,1]])
        association=dict(grid.metadata['association'],map_revision=7,map_from_local=transform.tolist())
        grid.metadata['association']=association
        row=dict(session_id='test',generation=0,frame_id='odom',stamp_s=10.,health=0,
                 has_map_pose=True,map_frame='map',map_revision=7)
        display.last_environment_grid=grid
        display.buffer.observe(row,(10.,0.),association)
        display._show_basemap(mono=.1)
        display._show_basemap(mono=.15)
        cloud,=received['navigation_basemap']
        window,=received['navigation_window']
        self.assertEqual((cloud.header.frame_id,cloud.header.stamp.to_sec()),('map',10.))
        self.assertEqual(window.header,cloud.header)
        self.assertEqual((window.type,window.action),(Marker.LINE_STRIP,Marker.ADD))
        self.assertEqual(cloud.width,2)
        points=np.frombuffer(cloud.data,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
        np.testing.assert_allclose(np.c_[points['x'],points['y']],[[1.55,1.25],[1.55,1.45]])
        # The lower point is outside the fixed static window and keeps local Z.
        np.testing.assert_allclose(points['z'],[-1.135,-.835])
        np.testing.assert_array_equal(points['rgb'],[LOCAL_FREE_COLOR,LOCAL_OCCUPIED_COLOR])
        np.testing.assert_allclose([[p.x,p.y] for p in window.points],[[1.6,1.2],[1.6,1.5],[1.4,1.5],[1.4,1.2],[1.6,1.2]])
        self.assertTrue(json.loads(received['basemap_source'][-1].data)['local_available'])

        # Same odometry generation can regain localization on a new association.
        replacement=dict(association,association_generation=1)
        display.buffer.observe(row,(10.,0.),replacement)
        display._show_basemap(mono=.15)
        self.assertEqual(received['navigation_basemap'][-1].width,0)
        self.assertEqual(received['navigation_window'][-1].action,Marker.DELETE)
        self.assertEqual(received['navigation_window'][-1].header,received['navigation_basemap'][-1].header)
        self.assertTrue(json.loads(received['basemap_source'][-1].data)['static_available'])
        self.assertFalse(json.loads(received['basemap_source'][-1].data)['local_available'])
        self.assertFalse(received['static_basemap'])

        for invalid in (dict(association,map_id='other-floor'),dict(association,valid=False),
                        dict(association,local_generation=2),dict(association,measurement_stamp_s=9.7),
                        dict(association,measurement_stamp_s=10.1)):
            display.buffer.observe(row,(10.,0.),invalid)
            display._show_basemap(mono=.15)
        self.assertEqual(len(received['navigation_basemap']),2)
        display.buffer.observe(row,(10.,0.),association)
        display._show_basemap(mono=.15)
        display._show_basemap(mono=.24)  # source association expires before the local grid
        self.assertEqual([c.width for c in received['navigation_basemap']],[2,0,2,0])

    def test_search_publication_is_independent_of_local_overlay_and_pause(self):
        from tests.test_environment_navigation import observation,prior_asset
        from perception.visualization.surface import coloured_cloud
        display,received=self.basemap_display(prior_asset())
        grid=observation(np.zeros((2,2),np.int8))
        association=dict(grid.metadata['association'],map_revision=7)
        grid.metadata['association']=association
        row=dict(session_id='test',generation=0,frame_id='odom',stamp_s=10.,health=0,
                 has_map_pose=True,map_frame='map',map_revision=7)
        packet=dict(row=row,mono=0.,now_s=10.,clock_rate=2.,scene=dict(as_of=10.,prior=association))
        snapshot=dict(reference_binding='reference',job_id=4,frame_id='map',stamp_s=10.)
        display.last_environment_grid=grid
        display.buffer.observe(row,(10.,0.),association)
        display.search_surface=SimpleNamespace(cloud=lambda s:coloured_cloud([0],[0],[0],[0],'map',s['stamp_s']))
        display.last_navigation=(row,packet);display.last_view=dict(search_map=snapshot)
        display._show_basemap(mono=.1)
        display._show_basemap(mono=.15)
        self.assertEqual(len(received['navigation_basemap']),1)
        self.assertEqual(len(received['reference_search_map']),1)
        self.assertEqual(received['reference_search_map'][0].header.stamp.to_sec(),10.)
        self.assertEqual(json.loads(received['basemap_source'][-1].data)['search_query_id'],4)
        display.last_view['search_map']=None  # canceled or unavailable during recomputation
        display._show_basemap(mono=.15)
        self.assertEqual(len(received['navigation_basemap']),1)
        self.assertEqual(received['reference_search_map'][-1].width,0)
        self.assertIsNone(json.loads(received['basemap_source'][-1].data)['search_query_id'])
        display.last_view['search_map']=dict(snapshot,job_id=5)
        display._show_basemap(mono=.15)
        self.assertEqual(json.loads(received['basemap_source'][-1].data)['search_query_id'],5)
        display.last_navigation=None
        display.frozen_basemap(True,snapshot,row=row)
        display._publish(*display.buffer.take())
        later=dict(row,stamp_s=10.2)
        display.buffer.observe(later,(10.2,99.),dict(association,valid=False,stamp_s=10.2))
        display._show_basemap(mono=100.)
        self.assertTrue(json.loads(received['basemap_source'][-1].data)['frozen'])
        self.assertEqual(json.loads(received['basemap_source'][-1].data)['search_query_id'],4)
        self.assertEqual(len(received['navigation_basemap']),1)
        display.frozen_basemap(True,row=row)
        display._publish(*display.buffer.take())
        self.assertEqual(received['reference_search_map'][-1].width,0)
        self.assertEqual(len(received['navigation_basemap']),1)
        display.frozen_basemap(False)
        display._publish(*display.buffer.take())
        self.assertEqual(received['navigation_basemap'][-1].width,0)
        self.assertFalse(json.loads(received['basemap_source'][-1].data)['frozen'])

    def test_mapless_overlay_uses_local_frame_and_clears_on_lifecycle_change(self):
        from tests.test_environment_navigation import observation
        display,received=self.basemap_display(None)
        row=dict(session_id='test',generation=0,frame_id='odom',stamp_s=10.,health=0)
        grid=observation(np.zeros((2,2),np.int8))
        grid.metadata.pop('association')
        display.last_environment_grid=grid
        display.buffer.observe(row,(10.,0.))
        display._show_basemap(mono=.1)
        self.assertEqual(received['navigation_basemap'][-1].header.frame_id,'odom')
        self.assertEqual(json.loads(received['basemap_source'][-1].data),
            dict(static_available=False,static_frame=None,local_available=True,frozen=False,search_query_id=None))
        display.buffer.observe(dict(row,generation=1),(10.,0.))
        display._show_basemap(mono=.1)
        self.assertEqual(received['navigation_basemap'][-1].width,0)
        display.buffer.observe(row,(10.,0.))
        display._show_basemap(mono=.1)
        display._show_basemap(stopped=True,mono=.1)
        self.assertEqual([message.width for message in received['navigation_basemap']],[4,0,4,0])

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
