"""Scene causality/lifecycle/resource binding and mixed-prior reentry boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from decision.vlm.environment_inputs import prepare_scene_context, render_environment_context
from perception.environment.layers import GridSnapshot, load_config
from perception.environment.scene import SceneInputs, assemble_scene, latest_past, live_scene_status
from perception.environment.context import query_scene
from perception.mapping_localization.coverage import PriorCoverage
from perception.mapping_localization.context import LocalizationContext
from perception.mapping_localization.experiment import compare_events
from tests.test_environment import pose, match
from tests.test_environment_consumers import snapshot


def person(stamp=9.9):
    return dict(stamp_s=stamp,clock_domain='synthetic',health='valid',frame_id='odom',
        localization_binding=dict(session_id='local',generation=0,frame_id='odom'),
        observation_id='detector/9.9',session_id='tracker_session',generation=7,sequence_id='replay',
        pose_stamp_s=stamp-.02,world_from_lidar=np.eye(4).tolist(),input_fingerprint='cloud_binding',
        detection_frame_id='rslidar',detections=[],tracks=[],velocity_semantics='world_motion',covariance_order=[])


def rgb(stamp=9.95):
    return dict(stamp_s=stamp,frame_id='camera_optical',pixels=np.zeros((20,30,3),np.uint8),
                reference=dict(kind='synthetic',topic='/camera/color/image_raw'))


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.settings=load_config()['scene']

    def assemble(self, **kwargs):
        return assemble_scene(snapshot(),clock_domain='synthetic',settings=self.settings,**kwargs)

    def test_latest_past_does_not_choose_future_or_hide_a_recent_failure(self):
        good=person(9.8);bad=dict(person(9.9),health='invalid')
        self.assertIs(latest_past([good,bad,person(10.1)],10.),bad)
        result=self.assemble(rgb=rgb(),people=bad).record
        self.assertEqual(result['health'],'degraded')
        self.assertEqual(result['people']['reason'],'people_localization_unavailable')
        self.assertNotIn('data',result['people'])

    def test_valid_empty_people_remain_distinct_from_missing_and_unknown_geometry(self):
        scene=self.assemble(rgb=rgb(),people=person()).record
        self.assertEqual(scene['health'],'valid')
        self.assertEqual(scene['people']['data']['generation'],7)
        self.assertEqual(scene['generation'],0)
        self.assertEqual(scene['people']['data']['tracks'],[])
        self.assertFalse(scene['navigation_reachability_verified'])
        self.assertEqual(self.assemble(rgb=rgb()).record['people']['status'],'missing')

    def test_late_complete_scene_cannot_override_new_invalid_reset_or_expired_live_state(self):
        bundle=self.assemble(rgb=rgb(),people=person())
        current=dict(session_id='local',generation=0,frame_id='odom',health=0)
        self.assertTrue(live_scene_status(bundle,current,10.1,.5)['ready'])
        self.assertFalse(live_scene_status(bundle,dict(current,health=2),10.1,.5)['ready'])
        self.assertFalse(live_scene_status(bundle,dict(current,generation=1),10.1,.5)['ready'])
        self.assertFalse(live_scene_status(bundle,current,11.,.5)['ready'])

    def test_actual_ros_wire_uint8_arrays_roundtrip_to_bound_layers(self):
        import io
        try:
            from social_nav_environment.msg import EnvironmentGrid
        except ImportError:
            self.skipTest('Generated ROS messages unavailable; source ros1_env.sh')
        from perception.environment.scene import snapshot_from_message
        message=EnvironmentGrid()
        current=snapshot();meta=current.metadata
        message.snapshot_id=meta['snapshot_id'];message.session_id=meta['session_id'];message.generation=0
        message.header.frame_id='odom';message.header.stamp.secs=10
        message.info.height,message.info.width=meta['shape'];message.info.resolution=meta['resolution']
        message.info.origin.position.x,message.info.origin.position.y=meta['origin_xy']
        message.metadata_json=json.dumps(meta)
        for name in ('occupancy','ground_z','ground_support','observed','last_observed','coverage',
                     'obstacle_min_z','obstacle_max_z','clearance','unknown_distance','frontier'):
            values=current.layers.get(name,np.zeros(meta['shape']))
            if name in ('ground_support','observed','frontier'):
                values=values.astype(np.uint8)
            setattr(message,name,values.ravel().tolist())
        buffer=io.BytesIO();message.serialize(buffer)
        received=EnvironmentGrid().deserialize(buffer.getvalue())
        self.assertIsInstance(received.observed,bytes)
        restored=snapshot_from_message(received)
        np.testing.assert_array_equal(restored.layers['occupancy'],current.layers['occupancy'])
        self.assertEqual(restored.layers['observed'].dtype,bool)

    def test_actual_ros_wire_preserves_history_masks_and_measurement_times(self):
        import io
        try:
            from social_nav_environment.msg import EnvironmentGrid
        except ImportError:
            self.skipTest('Generated ROS messages unavailable; source ros1_env.sh')
        from perception.environment.near_field import NearFieldFreeMemory
        from perception.environment.scene import snapshot_from_message
        from perception.environment.context import layer_binding
        from tests.test_environment_near_field import sample
        cfg=dict(load_config(),hit_invalidates_free=True)
        memory=NearFieldFreeMemory(cfg,dict(remember_outside_ring=True,temporary_obstacle_overlay=True))
        memory.update(sample(1,state=0),np.empty((0,3)))
        current=memory.update(sample(10),np.empty((0,3)))
        current.layers['person_background_free']=np.zeros(current.layers['occupancy'].shape,bool)
        current.layers['ground_conflict']=np.zeros(current.layers['occupancy'].shape,bool)
        current.layers['ground_conflict'][0,0]=True
        current.metadata['dynamic_navigation']=dict(policy='exclusive_person_on_certified_historical_free')
        message=EnvironmentGrid();meta=current.metadata
        message.snapshot_id=meta['snapshot_id'];message.session_id=meta['session_id'];message.generation=meta['generation']
        message.header.frame_id=meta['frame_id'];message.header.stamp.secs=10
        message.info.height,message.info.width=meta['shape'];message.info.resolution=meta['resolution']
        message.info.origin.position.x,message.info.origin.position.y=meta['origin_xy']
        message.metadata_json=json.dumps(meta)
        for name,values in current.layers.items():setattr(message,name,values.ravel().tolist())
        stream=io.BytesIO();message.serialize(stream)
        from rospy.numpy_msg import numpy_msg
        for message_type in (EnvironmentGrid,numpy_msg(EnvironmentGrid)):
            received=message_type().deserialize(stream.getvalue());restored=snapshot_from_message(received)
            self.assertEqual(layer_binding(restored),layer_binding(current))
            self.assertEqual(restored.layers['last_observed'][40,45],1.)
            self.assertIn('person_background_free',restored.layers)
            self.assertTrue(all(not a.flags.writeable for a in restored.layers.values()))
            encoded=io.BytesIO();received.serialize(encoded)
            self.assertEqual(stream.getvalue(),encoded.getvalue())
        received.person_background_free=b''
        with self.assertRaisesRegex(ValueError,'dynamic person provenance'):snapshot_from_message(received)
        received.person_background_free=bytes(current.layers['person_background_free'].size)
        received.near_field_free=b''
        with self.assertRaisesRegex(ValueError,'history provenance'):snapshot_from_message(received)

    def test_future_stale_clocks_and_local_lifecycles_are_not_merged(self):
        for change,reason in [(dict(stamp_s=10.1),'future_or_stale_people'),
                              (dict(stamp_s=8.),'future_or_stale_people'),
                              (dict(clock_domain='other'),'people_clock_mismatch'),
                              (dict(localization_binding=dict(session_id='other',generation=0,frame_id='odom')),
                               'people_localization_lifecycle_mismatch'),
                              (dict(pose_stamp_s=10.),'people_pose_time_invalid')]:
            scene=self.assemble(rgb=rgb(),people=dict(person(),**change)).record
            self.assertEqual(scene['people']['reason'],reason)
        for stamp in (10.1,9.):
            self.assertEqual(self.assemble(rgb=rgb(stamp)).record['rgb']['status'],'missing')

    def test_prior_is_bound_to_the_same_local_cutoff_without_visual_memory(self):
        current=snapshot()
        current.metadata['association']=dict(valid=True,status='PRIOR_ASSOCIATED',stamp_s=10.,
            local_session_id='local',local_generation=0,measurement_stamp_s=9.9,
            map_from_local=np.eye(4).tolist(),map_id='map',association_generation=2)
        scene=assemble_scene(current,clock_domain='synthetic',settings=self.settings,rgb=rgb(),people=person())
        self.assertNotIn('memory',scene.record)
        self.assertTrue(scene.record['prior']['valid'])
        current.metadata['association']['measurement_stamp_s']=9.
        stale=assemble_scene(current,clock_domain='synthetic',settings=self.settings).record['prior']
        self.assertFalse(stale['valid']);self.assertIsNone(stale['map_from_local'])

    def test_invalid_local_geometry_clears_people_even_with_previous_valid_input(self):
        current=snapshot();current.metadata.update(geometry_status='LOCAL_INVALID',world_from_lidar=None)
        scene=assemble_scene(current,clock_domain='synthetic',settings=self.settings,rgb=rgb(),people=person())
        self.assertEqual(scene.record['health'],'invalid')
        self.assertNotIn('data',scene.record['people'])

    def test_persisted_scene_and_queries_keep_the_exact_images_layers_and_identity(self):
        import cv2
        bundle=self.assemble(rgb=rgb(),people=person())
        wrong_shape=SceneInputs(bundle.record,bundle.geometry,bundle.rgb.reshape(10,60,3))
        with self.assertRaises(ValueError):wrong_shape.verify()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);bundle.geometry.save(path/'layers.npz')
            (path/'scene.json').write_text(json.dumps(bundle.record))
            cv2.imwrite(str(path/'rgb.png'),bundle.rgb[:,:,::-1])
            loaded=SceneInputs.load(path)
            req=dict(operation='get_rgb',observation_id=bundle.record['observation_id'])
            self.assertEqual(query_scene(loaded,req,load_config()['consumer'])['data']['pixels_sha256'],
                             bundle.record['rgb']['sha256'])
            from decision.agent.tools.environment import query_scene as agent_query
            fact=agent_query(loaded,dict(req,operation='get_scene'))
            self.assertEqual(fact['data']['observation_id'],loaded.record['observation_id'])
            self.assertFalse(fact['navigation_reachability_verified'])
            with self.assertRaises(ValueError):query_scene(loaded,dict(req,input_fingerprint='other'),{})
            with self.assertRaises(ValueError):query_scene(loaded,dict(req,observation_id='other'),{})
            with self.assertRaises(ValueError):query_scene(loaded,dict(req,operation='drive'),{})
            loaded.record['as_of']=11.
            with self.assertRaises(ValueError):loaded.verify()

    def test_scene_model_adapter_retains_people_states_and_fingerprint_without_rel_protocol_changes(self):
        people=person()
        people['tracks']=[dict(track_id='track',lifecycle_generation=7,box=[.5,.5,0,.5,.5,1.7,0,.8],
            state='predicted',last_observed_stamp_s=9.8,confirmed=True,velocity_xyz_mps=None)]
        bundle=self.assemble(rgb=rgb(),people=people)
        context=prepare_scene_context(bundle)
        self.assertEqual(context['protocol'],'environment_scene_context')
        self.assertEqual(context['scene_input_fingerprint'],bundle.record['input_fingerprint'])
        self.assertEqual(context['person_mapping'][0]['label'],'P1')
        preview=render_environment_context(bundle.geometry,context,bundle.rgb)
        self.assertEqual(preview.shape[2],3)
        context['people']['data']['tracks'][0]['state']='observed'
        with self.assertRaises(ValueError):render_environment_context(bundle.geometry,context,bundle.rgb)


class MixedCoverageTests(unittest.TestCase):
    def coverage(self):
        return PriorCoverage(dict(frame_id='map',map_asset_fingerprint='asset',source='synthetic_interface',
            regions_xy=[[[20,-2],[22,-2],[22,2],[20,2]],[[24,-2],[30,-2],[30,2],[24,2]]]),asset_fingerprint='asset')

    def test_loss_and_reentry_require_fresh_matches_and_independent_association_generation(self):
        events=[]
        for index,x in enumerate([1,1.1,1.2,2.5,2.6,4,4.1,4.2,4.3]):
            stamp=round(1+index*.1,2)
            front=pose(stamp);front['position']=[x,0,0]
            measured=match(stamp);measured.update(position=[x,0,0],map_position=[20+x,0,0])
            events.extend([dict(kind='frontend',row=front),dict(kind='match',row=measured)])
        result=compare_events(events,coverage=self.coverage(),source='roll_fast_lio',max_map_age=.4,confirm_matches=2)
        self.assertTrue(result['local_pose_and_lifecycle_unchanged'])
        rows=result['records']
        self.assertEqual(rows[2]['masked'],'PRIOR_ASSOCIATED')
        self.assertEqual(rows[3]['coverage'],'OUTSIDE_COVERAGE')
        self.assertEqual(rows[6]['masked'],'ASSOCIATING')
        self.assertEqual(rows[7]['masked'],'PRIOR_ASSOCIATED')
        self.assertEqual(rows[7]['association_generation'],1)
        self.assertTrue(all(r['health']==0 and r['session_id']=='front' and r['generation']==0 for r in rows))

    def test_stale_matches_do_not_reuse_pre_loss_confirmation(self):
        context=LocalizationContext('roll_fast_lio',allow_local_only=True)
        for stamp in (1,1.1):
            context.update(pose(stamp));context.add_match(match(stamp))
        self.assertTrue(context.update(pose(1.2))[1]['valid'])
        self.assertFalse(context.update(pose(1.8))[1]['valid'])
        context.add_match(match(1.8))
        self.assertFalse(context.update(pose(1.9))[1]['valid'])
        context.add_match(match(1.9))
        self.assertTrue(context.update(pose(2.))[1]['valid'])

    def test_coverage_does_not_claim_free_space_and_rejects_asset_frame_mismatch(self):
        coverage=self.coverage()
        self.assertTrue(coverage.contains([21,0,0],'map'))
        self.assertFalse(coverage.contains([23,0,0],'map'))
        with self.assertRaises(ValueError):coverage.contains([21,0,0],'odom')
        with self.assertRaises(ValueError):PriorCoverage(coverage.record,asset_fingerprint='changed')

    def test_disabled_matcher_still_reports_exit_from_last_trusted_anchor(self):
        context=LocalizationContext('roll_fast_lio',allow_local_only=True,coverage=self.coverage())
        for stamp in (1,1.1,1.2):
            front=pose(stamp); front['position']=[1,0,0]
            context.update(front)
            measured=match(stamp); measured.update(position=[1,0,0],map_position=[21,0,0])
            context.add_match(measured)
        front=pose(1.3);front['position']=[3,0,0]
        result,association=context.update(front)
        self.assertEqual(association['coverage_status'],'OUTSIDE_COVERAGE')
        self.assertFalse(association['valid'])
        self.assertEqual(result['health'],0)
        measured=match(1.3); measured.update(position=[3,0,0],map_position=[21,0,0])
        context.add_match(measured)
        front=pose(1.4);front['position']=[3,0,0]
        self.assertEqual(context.update(front)[1]['coverage_status'],'OUTSIDE_COVERAGE')
        front=pose(2.);front['position']=[3,0,0]
        self.assertEqual(context.update(front)[1]['coverage_status'],'OUTSIDE_COVERAGE')


if __name__ == '__main__':
    unittest.main()
