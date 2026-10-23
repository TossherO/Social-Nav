"""Static priors cannot replace local evidence or survive a changed association."""
from pathlib import Path
import tempfile
import threading
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from decision.agent.environment_tools import query_scene
from decision.vlm.environment_inputs import prepare_scene_context, render_environment_context
from perception.environment.layers import GridSnapshot, load_config
from perception.environment.memory import create_index
from perception.environment.prior import PriorMap, map_digest, live_prior_status
from perception.environment.scene import SceneInputs, assemble_scene
from perception.environment.storage import content_hash
from tests.test_environment_consumers import snapshot
from tests.test_environment_scene import rgb, person


class PriorTests(unittest.TestCase):
    def test_scene_and_visualization_launch_nodes_can_coexist(self):
        root = Path(__file__).resolve().parents[1]
        scene = ET.parse(root/'perception/mapping_localization/launch/native.launch')
        display = ET.parse(root/'perception/detection_tracking/ros/launch/visualization.launch')
        scene_names = {node.attrib['name'] for node in scene.findall('.//node')}
        display_names = {node.attrib['name'] for node in display.findall('.//node')}
        self.assertTrue(scene_names.isdisjoint(display_names))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.settings = load_config()['scene']
        current = snapshot()
        meta = dict(current.metadata,frame_id='map',snapshot_id='static',time_scope='offline_final_prior',
                    stamp_s=1.,geometry_status='STATIC_SCAN_OBSERVED')
        GridSnapshot(meta,current.layers).save(self.path/'layers.npz')
        binding = dict(snapshot_id='static')
        create_index(self.path/'observations.sqlite',binding).close()
        self.source = {'geometry':{'sha256':'synthetic_map'}}
        manifest = dict(kind='static_environment',snapshot_id='static',frame_id='map',
            memory_binding=binding,layers_sha256=content_hash(self.path/'layers.npz'),
            time_scope='offline_final_prior',source=dict(map_fingerprint=self.source))
        (self.path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
        self.prior = PriorMap(self.path,expected_map=self.source)

    def scene(self, changes=None, local=None):
        current = local or snapshot()
        association = dict(valid=True,status='PRIOR_ASSOCIATED',stamp_s=10.,measurement_stamp_s=9.9,
            local_session_id='local',local_generation=0,local_frame='odom',map_frame='map',
            asset_fingerprint=map_digest(self.source),map_id='roll:'+map_digest(self.source),
            association_session='association',association_generation=0,map_from_local=np.eye(4).tolist())
        association.update(changes or {})
        current.metadata['association'] = association
        return assemble_scene(current,clock_domain='synthetic',settings=self.settings,rgb=rgb(),people=person())

    def test_package_identity_and_query_frame_cannot_be_substituted(self):
        with self.assertRaises(ValueError):PriorMap(self.path,expected_map={'other':True})
        with self.assertRaises(ValueError):self.prior.attach(self.scene(dict(asset_fingerprint='other')),self.settings)
        with self.assertRaises(ValueError):self.prior.attach(self.scene(dict(map_frame='odom')),self.settings)
        bound = self.prior.attach(self.scene(),self.settings)
        request = dict(operation='query_prior_geometry',observation_id=bound.record['observation_id'],
                       frame='map',points=[[-.5,-.5],[100,100]])
        result = query_scene(bound,request)['data']['data']
        self.assertEqual(result['cells'][0]['state'],'unknown')
        self.assertEqual(result['cells'][1]['state'],'outside_coverage')
        self.assertFalse(result['current_free_space'])
        with self.assertRaises(ValueError):query_scene(bound,dict(request,frame='odom'))

    def test_invalid_stale_and_mismatched_lifecycle_keep_only_local_geometry(self):
        for changes in (dict(valid=False),dict(measurement_stamp_s=9.),dict(local_generation=1),
                        dict(local_frame='other'),dict(measurement_stamp_s=10.1)):
            source = self.scene(changes)
            bound = self.prior.attach(source,self.settings)
            self.assertIsNone(bound.prior_geometry)
            self.assertIs(bound.geometry,source.geometry)
            self.assertEqual(source.record.get('prior_geometry'),None)
            self.assertEqual(bound.record['people']['status'],'valid')
            bound.verify()

    def test_crop_uses_full_se3_and_clips_without_inventing_cells(self):
        current = snapshot()
        current.metadata['world_from_lidar'][2][3] = 2.
        angle = .3
        transform = np.array([[1,0,0,0],[0,np.cos(angle),-np.sin(angle),0],
                              [0,np.sin(angle),np.cos(angle),0],[0,0,0,1]])
        source = self.scene(dict(map_from_local=transform.tolist()),local=current)
        bound = self.prior.attach(source,self.settings)
        self.assertAlmostEqual(bound.record['prior_geometry']['map_from_lidar'][1][3],-2*np.sin(angle))
        np.testing.assert_array_equal(bound.prior_geometry.layers['occupancy'],self.prior.snapshot.layers['occupancy'])
        far = transform.copy();far[0,3] = 1000
        outside = self.prior.attach(self.scene(dict(map_from_local=far.tolist())),self.settings)
        self.assertEqual(outside.record['prior_geometry']['reason'],'outside_prior_raster')
        with self.assertRaises(ValueError):self.prior.attach(source,dict(self.settings,prior_max_cells=1))

    def test_reassociation_withdraws_old_live_view_without_resetting_local_identity(self):
        bound = self.prior.attach(self.scene(),self.settings)
        pose = dict(session_id='local',generation=0,frame_id='odom',health=0)
        association = bound.record['prior']
        status = lambda b=bound,p=pose,a=association,t=10.1:live_prior_status(b,p,a,t,.5,.4)['ready']
        self.assertTrue(status())
        self.assertFalse(status(a=dict(association,valid=False)))
        self.assertFalse(status(a=dict(association,association_generation=1)))
        self.assertFalse(status(p=dict(pose,health=2)))
        self.assertFalse(status(p=dict(pose,generation=1)))
        self.assertFalse(status(t=11.))
        self.assertFalse(status(a=dict(association,stamp_s=11.)))
        recovered = self.prior.attach(self.scene(dict(association_generation=1)),self.settings)
        self.assertTrue(status(b=recovered,a=recovered.record['prior']))
        self.assertEqual(bound.record['generation'],recovered.record['generation'])
        self.assertNotEqual(bound.record['input_fingerprint'],recovered.record['input_fingerprint'])

    def test_persistence_rendering_and_queries_bind_the_actual_prior_arrays(self):
        source = self.scene();old_fingerprint = source.record['input_fingerprint']
        bound = self.prior.attach(source,self.settings)
        self.assertEqual(old_fingerprint,source.record['input_fingerprint'])
        bound.save(self.path/'scene')
        loaded = SceneInputs.load(self.path/'scene')
        context = prepare_scene_context(loaded)
        preview = render_environment_context(loaded.geometry,context,loaded.rgb,prior_geometry=loaded.prior_geometry)
        self.assertEqual(preview.shape[2],3)
        with self.assertRaises(ValueError):render_environment_context(loaded.geometry,context,loaded.rgb)
        request = dict(operation='crop_prior_map',observation_id=bound.record['observation_id'],
                       bounds=[-1,-1,2,2],frame='map')
        with self.assertRaises(ValueError):query_scene(bound,request,settings={'tool_crop_cells':1})
        self.assertEqual(query_scene(bound,dict(request,operation='get_prior'))['data']['data']['status'],'valid')
        changed = {k:v.copy() for k,v in bound.prior_geometry.layers.items()}
        changed['occupancy'][0,0] = 0
        bad = SceneInputs(bound.record,bound.geometry,bound.rgb,GridSnapshot(bound.prior_geometry.metadata,changed))
        with self.assertRaises(ValueError):bad.verify()
        without = self.prior.attach(self.scene(dict(valid=False)),self.settings)
        with self.assertRaises(ValueError):query_scene(without,dict(request,operation='query_prior_geometry',points=[[0,0]]))
        # The new optional layer does not break loading old scenes.
        source.save(self.path/'old_scene')
        self.assertIsNone(SceneInputs.load(self.path/'old_scene').prior_geometry)

    def test_ros_display_publishes_full_transform_and_clears_on_loss_reentry_and_expiry(self):
        try:
            from nav_msgs.msg import OccupancyGrid
            from std_msgs.msg import String
            from tf.transformations import quaternion_matrix
        except ImportError:
            self.skipTest('ROS message/TF modules unavailable; source ros1_env.sh')
        from perception.environment.ros.scene_node import SceneNode
        angle = .3
        transform = np.array([[1,0,0,4],[0,np.cos(angle),-np.sin(angle),2],
                              [0,np.sin(angle),np.cos(angle),0],[0,0,0,1.]])
        bound = self.prior.attach(self.scene(dict(map_from_local=transform.tolist())),self.settings)
        node = SceneNode.__new__(SceneNode)
        now = [10.1]
        class Stamp:
            def __init__(self, value):self.value = value
            def to_sec(self):return self.value
        messages,states = [],[]
        node.rospy = SimpleNamespace(Time=SimpleNamespace(now=lambda:Stamp(now[0])))
        node.String,node.OccupancyGrid = String,OccupancyGrid
        node.condition = threading.RLock()
        node.config,node.settings = load_config(),self.settings
        node.latest,node.association = bound,bound.record['prior']
        node.poses = SimpleNamespace(rows=[dict(session_id='local',generation=0,frame_id='odom',health=0)])
        node.counts,node.queue,node.busy,node.error = dict(prior_withdrawals=0),[],False,None
        node.prior_pub,node.state_pub = SimpleNamespace(publish=messages.append),SimpleNamespace(publish=states.append)
        node.displayed_prior = None
        node.publish_status()
        message = messages[-1]
        self.assertEqual(message.header.frame_id,'odom')
        self.assertEqual(message.info.width,bound.prior_geometry.metadata['shape'][1])
        q = message.info.origin.orientation
        np.testing.assert_allclose(quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3],np.linalg.inv(transform)[:3,:3],atol=1e-6)
        node.association = dict(node.association,valid=False)
        node.publish_status()
        self.assertEqual(messages[-1].data,[-1])
        self.assertEqual(messages[-1].info.width,1)
        node.association = dict(bound.record['prior'],association_generation=1)
        node.publish_status()
        self.assertEqual(len(messages),2)
        node.latest = self.prior.attach(self.scene(dict(map_from_local=transform.tolist(),association_generation=1)),self.settings)
        node.publish_status()
        self.assertGreater(messages[-1].info.width,1)
        now[0] = 11.
        node.publish_status()
        self.assertEqual(messages[-1].data,[-1])
        # Late valid packets cannot resurrect a newer explicit failure/session.
        node.retired_associations = deque(maxlen=32)
        node.association = dict(bound.record['prior'],valid=False,stamp_s=10.2)
        with patch('perception.mapping_localization.ros.context_node.association_row',
                   return_value=dict(bound.record['prior'],stamp_s=10.1)):
            node.associate(None)
        self.assertFalse(node.association['valid'])
        with patch('perception.mapping_localization.ros.context_node.association_row',
                   return_value=dict(bound.record['prior'],stamp_s=10.3,association_session='new')):
            node.associate(None)
        with patch('perception.mapping_localization.ros.context_node.association_row',
                   return_value=dict(bound.record['prior'],stamp_s=10.4)):
            node.associate(None)
        self.assertEqual(node.association['association_session'],'new')
        # A restarted node clears a previous RViz image before any new scene.
        node.latest,node.displayed_prior = None,'uninitialized'
        node.publish_status()
        self.assertEqual(messages[-1].data,[-1])


if __name__ == '__main__':
    unittest.main()
