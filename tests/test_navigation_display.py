"""Display coordinates, source times and invalidation without ROS or a GPU."""
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from navigation.control.dwa import DwaController
from navigation.planning.continuity import reference_envelope
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.planner import Planner,se2,transform_poses
from perception.visualization.navigation import NavigationView,display_packet
from perception.visualization.surface import DisplayGrids,DisplaySurface,line_segments,navigation_grid
from tests.navigation_fixtures import asset,bundle,request,grid


def ground_grid(z=-1.,stamp=10.):
    snapshot=grid();snapshot.metadata['stamp_s']=stamp
    snapshot.layers.update(ground_z=np.broadcast_to(np.asarray(z,np.float32),snapshot.metadata['shape']).copy(),
        ground_support=np.ones(snapshot.metadata['shape'],bool))
    return snapshot


def navigation_display_fixture():
    from perception.environment.layers import load_config as environment_config
    from perception.environment.support import NavigationSupport
    from tests.test_environment_navigation import observation,prior_asset
    from tests.test_environment_support import points
    static=prior_asset();static.snapshot.layers['ground_z'][:]=-.7
    raw=observation(np.full((60,60),-1,np.int8))
    raw.metadata['association']['map_from_local'][2][3]=.15
    raw.layers['ground_z'][:]=np.nan;raw.layers['ground_support'][:]=False
    raw.layers['occupancy'][25,25]=100
    support=NavigationSupport(environment_config())
    derived=support.apply(raw,static,observation=points(raw,[[2.05,2.05,-.3]],[1]))
    return raw,derived,static


class NavigationDisplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config();self.cfg['algorithm']='hybrid_astar';self.cfg['local_replanning']['enabled']=False
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.scene=bundle();self.req=request('local',goal=[4.,2.,0.],goal_behavior='stop')
        self.row=dict(session_id='test',generation=0,frame_id='odom',health=0,stamp_s=10.,
            position=[2.,2.,-2.],quaternion_xyzw=[0,0,0,1])

    def packet(self,scene=None,req=None,static=None,geometry=None):
        scene=self.scene if scene is None else scene;req=self.req if req is None else req
        geometry=self.geometry if geometry is None else geometry
        result=Planner(self.cfg,geometry).query(req,scene=scene,asset=static,now=10.,clock_domain='synthetic')
        self.assertEqual(result['status'],'path_found',result)
        reference=reference_envelope(result,req,scene,static)
        feedback=dict(status='valid',source='synthetic',pose=geometry.pose(np.asarray(scene.record['localization']['world_from_lidar'])).tolist(),
            velocity=[0.,0.],stamp_s=10.,clock_domain='synthetic',local_identity=['test',0,'odom'])
        controller=DwaController(self.cfg,load_motion_config(),geometry,asset=static)
        proposal=controller.command(req,scene,feedback,reference,now=10.,clock_domain='synthetic')
        self.assertEqual(proposal['status'],'candidate',proposal)
        packet=display_packet(controller.result,reference,req,scene,self.row,dict(proposal=proposal),None,
            now=10.,clock_domain='synthetic',clock_rate=1.,clock_mono=0.)
        return packet

    def test_body_history_uses_full_extrinsic_and_raw_history_keeps_3d(self):
        transform=np.eye(4);transform[:3,3]=[.67,.33,-.2]
        geometry=VehicleGeometry(self.geometry.footprint,transform,source='synthetic historical offset')
        row=dict(self.row,quaternion_xyzw=Rotation.from_euler('xyz',[.2,.1,.6]).as_quat().tolist())
        view=NavigationView(geometry,self.cfg,path_z_m=.07,height_mode='plane').update(row,None,mono=0.)
        world=np.eye(4);world[:3,:3]=Rotation.from_quat(row['quaternion_xyzw']).as_matrix();world[:3,3]=row['position']
        np.testing.assert_allclose(view['history'][0][1],geometry.pose(world))
        self.assertEqual(view['z'],.07);self.assertEqual(view['raw_history'][0][1][2],-2.)
        self.assertGreater(np.linalg.norm(view['pose'][:2]-row['position'][:2]),.7)

    def test_reference_and_dwa_keep_source_stamps_and_input_data(self):
        packet=self.packet();before=copy.deepcopy(packet)
        renderer=NavigationView(self.geometry,self.cfg)
        view=renderer.update(dict(self.row,stamp_s=10.1),packet,mono=.1)
        self.assertEqual(view['reference']['stamp_s'],10.)
        self.assertEqual(view['trajectory']['stamp_s'],10.)
        self.assertEqual(view['stamp_s'],10.1)
        self.assertEqual(packet,before)
        np.testing.assert_allclose(view['reference']['points'][0],[2,2,0],atol=1e-7)
        self.assertIn('candidate',view['text'])

    def test_mapped_reference_uses_canonical_planar_conversion(self):
        current=bundle();static=asset()
        # Rotate/translate the local origin while keeping the mapped start inside the static map.
        mapped=np.eye(4);mapped[:2,:2]=se2([0.,0.,np.pi/2])[:2,:2];mapped[:2,3]=[4.,0.]
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config as environment_config
        current.geometry.metadata['association']['map_from_local']=mapped.tolist()
        current=assemble_scene(current.geometry,clock_domain='synthetic',settings=environment_config()['scene'])
        req=request('mapped',goal=[2.,4.,np.pi/2],goal_behavior='stop')
        packet=self.packet(current,req,static)
        view=NavigationView(self.geometry,self.cfg,asset=static).update(self.row,packet,mono=0.)
        self.assertIsNotNone(view['reference'],view['text'])
        expected=transform_poses(np.asarray(packet['reference']['result']['path']),np.linalg.inv(se2([4.,0.,np.pi/2])))
        np.testing.assert_allclose(view['reference']['points'],expected,atol=1e-7)
        np.testing.assert_allclose(view['reference']['points'][0],[2.,2.,0.],atol=1e-7)
        np.testing.assert_allclose(view['goal'],[4.,2.,0.],atol=1e-7)
        packet['scene']['prior']['valid']=False
        invalid=NavigationView(self.geometry,self.cfg,asset=static).update(self.row,packet,mono=0.)
        self.assertIsNone(invalid['reference']);self.assertIsNone(invalid['trajectory'])
        self.assertIsNone(invalid['goal']);self.assertIn('association_unavailable',invalid['text'])

    def test_stale_cancel_changed_goal_and_reset_clear_predictions(self):
        packet=self.packet();renderer=NavigationView(self.geometry,self.cfg)
        self.assertIsNotNone(renderer.update(self.row,packet,mono=0.)['trajectory'])
        stale=renderer.update(self.row,packet,mono=1.)
        self.assertIsNone(stale['reference']);self.assertIsNone(stale['trajectory']);self.assertTrue(stale['history'])
        self.assertIn('geometry_future_or_stale',stale['text'])
        cancelled=dict(packet,request=None)
        out=renderer.update(self.row,cancelled,mono=0.)
        self.assertIsNone(out['reference']);self.assertIsNone(out['trajectory']);self.assertIsNone(out['goal'])
        changed=copy.deepcopy(packet);changed['request']['goal_revision']+=1
        out=renderer.update(self.row,changed,mono=0.)
        self.assertIsNone(out['reference']);self.assertIsNone(out['trajectory'])
        out=renderer.update(dict(self.row,generation=1),packet,mono=0.)
        self.assertEqual(len(out['history']),1);self.assertIsNone(out['reference'])
        invalid=renderer.update(dict(self.row,health=1),packet,mono=0.)
        self.assertFalse(invalid['history']);self.assertIsNone(invalid['reference'])
        stopped=renderer.update(self.row,packet,mono=0.,stopped=True)
        self.assertIsNone(stopped['trajectory']);self.assertIn('stopped',stopped['text'])

    def test_newer_measured_control_pose_is_used_without_predicting_history(self):
        packet=self.packet();packet['row'].update(stamp_s=10.1,position=[2.1,2.,-2.])
        renderer=NavigationView(self.geometry,self.cfg)
        view=renderer.update(self.row,packet,mono=.1)
        self.assertEqual(view['stamp_s'],10.1)
        self.assertEqual(view['localization_row']['stamp_s'],10.1)
        self.assertEqual(view['localization_row']['position'],[2.1,2.,-2.])
        np.testing.assert_allclose(view['pose'],[2.1,2.,0.])
        self.assertEqual(view['history'][0][0],10.1)
        renderer.update(self.row,packet,mono=.2)
        self.assertEqual(len(renderer.history),1)

    def test_expired_or_rejected_dwa_does_not_hide_a_valid_reference(self):
        packet=self.packet();renderer=NavigationView(self.geometry,self.cfg,trajectory_max_age_s=.1)
        view=renderer.update(self.row,packet,mono=.2)
        self.assertIsNotNone(view['reference']);self.assertIsNone(view['trajectory'])
        self.assertIn('trajectory_future_or_stale',view['text'])
        rejected=copy.deepcopy(packet);rejected['proposal'].update(status='unavailable',reason='no_admissible_velocity')
        view=renderer.update(self.row,rejected,mono=0.)
        self.assertIsNotNone(view['reference']);self.assertIsNone(view['trajectory'])
        self.assertIn('no_admissible_velocity',view['text'])

    def test_clock_mismatch_clears_both_curves(self):
        packet=self.packet();packet['clock_domain']='another_clock'
        view=NavigationView(self.geometry,self.cfg).update(self.row,packet,mono=0.)
        self.assertIsNone(view['reference']);self.assertIsNone(view['trajectory'])
        self.assertIn('clock_domain_mismatch',view['text'])

    def test_lower_ground_changes_current_height_and_preserves_previous_heights(self):
        renderer=NavigationView(self.geometry,self.cfg)
        first=renderer.update(self.row,None,grid=ground_grid(-1.),mono=0.)
        self.assertAlmostEqual(first['z'],-.97)
        row=dict(self.row,stamp_s=10.1,position=[2.1,2.,-3.])
        second=renderer.update(row,None,grid=ground_grid(-2.,10.1),mono=.1)
        self.assertAlmostEqual(second['z'],-1.97)
        np.testing.assert_allclose([p[2] for p in second['history']],[-.97,-1.97])
        self.assertTrue(np.allclose(first['footprint'][:,2],-.93))
        self.assertTrue(np.allclose(second['footprint'][:,2],-1.93))
        self.assertEqual(second['raw_history'][1][1][2],-3.)

    def test_slope_and_holes_are_sampled_between_sparse_reference_endpoints(self):
        packet=self.packet();snapshot=ground_grid(np.arange(60)[None,:]*.1-3.)
        snapshot.layers['ground_support'][:,28:31]=False
        view=NavigationView(self.geometry,self.cfg).update(self.row,packet,grid=snapshot,mono=0.)
        path=view['reference'];self.assertIsNotNone(path)
        self.assertGreater(len(path['display_poses']),len(path['points']))
        self.assertTrue(np.isnan(path['heights']).any())
        indices=np.floor(path['display_poses'][:,0]/.1).astype(int)
        valid=np.isfinite(path['heights'])
        np.testing.assert_allclose(path['heights'][valid],indices[valid]*.1-3.+.03,atol=1e-6)
        segments=line_segments(np.c_[path['display_poses'][:,:2],path['heights']]).reshape(-1,2,3)
        self.assertTrue(np.isfinite(segments).all())
        self.assertFalse(np.any((segments[:,0,0]<2.8)&(segments[:,1,0]>=3.1)))

    def test_ground_unknown_stale_future_or_reset_never_becomes_zero_height(self):
        renderer=NavigationView(self.geometry,self.cfg)
        for snapshot in (None,ground_grid(stamp=9.),ground_grid(stamp=10.1),ground_grid()):
            if snapshot is not None and snapshot.metadata['stamp_s']==10.:snapshot.metadata['generation']=1
            view=renderer.update(self.row,None,grid=snapshot,mono=0.)
            self.assertTrue(np.isnan(view['z']));self.assertIn('GROUND: unknown',view['text'])
            self.assertTrue(np.isnan(view['footprint'][:,2]).all())

    def test_near_vehicle_fit_is_explicit_bounded_and_does_not_fill_map(self):
        snapshot=ground_grid();snapshot.layers['ground_support'][:]=False
        snapshot.layers['ground_z'][:]=np.nan
        snapshot.metadata['ground_tracking']=dict(status='tracked',plane_world=[.1,0.,-2.])
        surface=DisplaySurface(snapshot,self.row,plane_radius=1.5)
        heights,sources=surface.sample([[2.,2.],[4.,2.]])
        self.assertAlmostEqual(heights[0],-1.77);self.assertEqual(sources[0],2)
        self.assertTrue(np.isnan(heights[1]));self.assertEqual(sources[1],0)
        self.assertTrue(np.isnan(snapshot.layers['ground_z']).all())
        self.assertFalse(snapshot.layers['ground_support'].any())
        snapshot.metadata['ground_tracking']['status']='warming'
        self.assertTrue(np.isnan(DisplaySurface(snapshot,self.row).sample([[2.,2.]])[0][0]))

    def test_display_grid_download_is_named_selective_and_cached(self):
        calls=[];host=ground_grid()
        class Device:
            metadata=host.metadata
            _layers=dict(occupancy=None,ground_z=None,ground_support=None,coverage=None)
            def materialize(self,**kwargs):calls.append(kwargs);return host
        cache=DisplayGrids();device=Device()
        self.assertIs(cache.get(device),host);self.assertIs(cache.get(device),host)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['fields'],['occupancy','ground_z','ground_support'])
        self.assertEqual(calls[0]['consumer'],'optional_ros_display')
        self.assertIs(cache.get(host),host)


class FrozenDisplayTests(unittest.TestCase):
    def test_accepted_goal_marker_keeps_xy_yaw_and_rejects_unknown_height(self):
        from perception.visualization.interactive import goal_geometry
        grid=ground_grid(-2.)
        request=dict(frame_id='odom',goal=[2.,2.,.7])
        view=goal_geometry(request,grid)
        np.testing.assert_allclose(view['position'],[2.,2.,-1.9]);self.assertEqual(view['yaw'],.7)
        self.assertEqual(request['goal'],[2.,2.,.7])
        self.assertIsNone(goal_geometry(None,grid))
        self.assertIsNone(goal_geometry(dict(request,frame_id='map'),grid))
        request['goal']=[2.,2.]
        self.assertIsNone(goal_geometry(request,grid)['yaw'])
        grid.layers['ground_support'][:]=False
        self.assertIsNone(goal_geometry(request,grid))

    def setUp(self):
        from perception.visualization.interactive import preview_geometry
        self.project=preview_geometry;self.grid=ground_grid(-2.)
        self.asset=asset();self.asset.snapshot=ground_grid(-1.)
        self.asset.snapshot.metadata['frame_id']='map';self.asset.manifest['frame_id']='map'
        self.settings=dict(path_z_m=.03,height_mode='ground',ground_plane_radius_m=0.)
        self.preview=dict(as_of=10.,reference=dict(result=dict(frame_id='map',start=[2.,2.,0.],path=[[2.,2.,0.],[4.,2.,0.]])),
            trajectory=dict(frame_id='odom',points=[[0.,2.,2.,0.],[1.,3.,2.,0.]]))

    def test_mapped_reference_and_local_prediction_use_own_terrain_frames(self):
        import json
        original=copy.deepcopy(self.preview)
        view=self.project(self.preview,self.grid,self.asset,self.settings)
        ref=view['curves']['reference'];dwa=view['curves']['trajectory']
        self.assertEqual(ref['frame_id'],'map');self.assertEqual(dwa['frame_id'],'odom')
        np.testing.assert_allclose(np.asarray(ref['segments'])[:,2],-.97)
        np.testing.assert_allclose(np.asarray(dwa['segments'])[:,2],-1.95)
        self.assertEqual(ref['stamp_s'],10.);self.assertEqual(dwa['stamp_s'],10.)
        self.assertAlmostEqual(view['label_position'][2],1.23)
        self.assertEqual(self.preview,original)
        json.dumps(view,allow_nan=False)

    def test_unknown_static_and_local_heights_leave_gaps_without_zero_fill(self):
        self.asset.snapshot.layers['ground_support'][:,28:31]=False
        self.grid.layers['ground_support'][:]=False
        view=self.project(self.preview,self.grid,self.asset,self.settings)
        ref=view['curves']['reference'];dwa=view['curves']['trajectory']
        self.assertLess(ref['height_count'],ref['point_count'])
        segments=np.asarray(ref['segments']).reshape(-1,2,3)
        self.assertFalse(np.any((segments[:,0,0]<2.8)&(segments[:,1,0]>=3.1)))
        self.assertLessEqual(ref['prefix'][-1][0],2.8)
        self.assertEqual(dwa['segments'],[]);self.assertEqual(dwa['prefix'],[])
        np.testing.assert_allclose(self.grid.layers['ground_z'],-2.)

    def test_failed_reference_anchors_status_in_measured_local_frame(self):
        self.preview['reference']['result']=dict(reason='association_unavailable')
        self.preview['trajectory']=None
        view=self.project(self.preview,self.grid,self.asset,self.settings)
        self.assertEqual(view['label_frame'],'odom')
        self.assertEqual(view['curves']['reference']['segments'],[])
        self.assertAlmostEqual(view['label_position'][2],.23)

    def test_stale_local_height_is_not_used_for_frozen_prediction(self):
        self.preview['as_of']=11.
        view=self.project(self.preview,self.grid,self.asset,self.settings)
        self.assertEqual(view['curves']['trajectory']['segments'],[])
        self.assertTrue(view['curves']['reference']['segments'])


class NavigationMapDisplayTests(unittest.TestCase):
    def setUp(self):
        self.raw,self.derived,self.asset=navigation_display_fixture()

    def test_display_cells_equal_independent_navigation_and_preserve_raw_inputs(self):
        from perception.environment.context import layer_binding
        from perception.environment.navigation import navigation_geometry
        from tests.test_environment_navigation import scene
        before=layer_binding(self.derived);view=navigation_grid(self.derived,self.asset)
        expected=navigation_geometry(scene(self.derived),asset=self.asset,check_asset=True)
        np.testing.assert_array_equal(view.layers['occupancy'],expected.layers['occupancy'])
        self.assertEqual(view.layers['occupancy'][20,21],0)
        self.assertEqual(view.layers['occupancy'][20,20],100)
        self.assertEqual(view.layers['occupancy'][25,25],100)
        self.assertEqual(view.layers['occupancy'][55,55],-1)
        self.assertEqual(layer_binding(self.derived),before)
        self.assertTrue(np.isnan(self.raw.layers['ground_z']).all())
        self.assertFalse(self.raw.layers['ground_support'].any())
        self.assertEqual(view.metadata['stamp_s'],self.derived.metadata['stamp_s'])
        self.assertTrue(view.metadata['display_only'])

    def test_static_height_uses_map_transform_only_for_supported_cells(self):
        view=navigation_grid(self.derived,self.asset)
        np.testing.assert_allclose(view.layers['ground_z'][20,20:22],-.85,atol=1e-6)
        np.testing.assert_array_equal(view.layers['display_ground_source'][20,20:22],[4,4])
        self.assertTrue(np.isnan(view.layers['ground_z'][55,55]))
        row={k:self.raw.metadata[k] for k in ('session_id','generation','frame_id','stamp_s')}
        surface=DisplaySurface(view,row,plane_radius=0.)
        heights,sources=surface.sample([[2.05,2.05],[2.15,2.05],[5.55,5.55]])
        np.testing.assert_allclose(heights[:2],[-.82,-.82],atol=1e-6)
        np.testing.assert_array_equal(sources,[4,4,0])
        self.assertTrue(np.isnan(heights[2]))

    def test_missing_mismatched_or_invalid_prior_does_not_invent_height(self):
        invalid=copy.deepcopy(self.derived);invalid.metadata['association']['valid']=False
        wrong=copy.deepcopy(self.asset);wrong.manifest['snapshot_id']='another-map'
        for raw,asset in [(self.derived,None),(invalid,self.asset),(self.derived,wrong)]:
            with self.subTest(asset=asset is not None,valid=raw.metadata['association']['valid']):
                view=navigation_grid(raw,asset)
                np.testing.assert_array_equal(view.layers['occupancy'],raw.layers['navigation_occupancy'])
                self.assertFalse(view.layers['ground_support'].any())

    def test_device_download_and_projection_are_shared_with_raw_diagnostic(self):
        calls=[];host=self.derived
        class Device:
            metadata=host.metadata
            _layers=host.layers
            def materialize(self,**kwargs):calls.append(kwargs);return host
        cache=DisplayGrids(asset=self.asset);device=Device()
        raw=cache.get(device,navigation=False);view=cache.get(device)
        self.assertIs(raw,host);self.assertIs(cache.get(device),view)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['fields'],['occupancy','ground_z','ground_support','navigation_occupancy','navigation_source'])
        self.assertEqual(calls[0]['consumer'],'optional_ros_display')
        self.assertIs(cache.get(self.raw),self.raw)
        unbound=copy.deepcopy(self.derived);del unbound.metadata['navigation_fusion']
        with self.assertRaisesRegex(ValueError,'provenance'):cache.get(unbound)


if __name__=='__main__':unittest.main()
