"""Person delegation preserves background, current contact and missing-data fallback."""
import unittest
from unittest.mock import patch
import numpy as np

from perception.environment.layers import load_config,LocalEnvironmentMap
from perception.environment.person_objects import point_owners,observed_objects
from perception.environment.navigation import navigation_geometry
from perception.environment.scene import assemble_scene,people_observation
from perception.environment.support import NavigationSupport
from navigation.planning.inputs import load_config as planning_config
from navigation.planning.people import PeopleModel
from tests.test_environment_navigation import observation,prior_asset,scene
from tests.test_environment_support import points
from tests.test_navigation_prediction import tracked_scene,track


def object_scene(tracks):
    initial=tracked_scene(tracks);data=people_observation(initial);raw=initial.geometry
    raw.metadata['person_objects']=dict(method='tracked_person_circles',stamp_s=10.,
        local_identity=['test',0,'odom'],people_source=data['input_fingerprint'],objects=[
            dict(identity=[data['session_id'],data['generation'],str(t['track_id'])],xy=t['box'][:2],radius_m=.42,points=3)
            for t in tracks])
    return assemble_scene(raw,clock_domain='synthetic',settings=load_config()['scene'],people=data)


class PersonObjectTests(unittest.TestCase):
    def test_stale_cuda_library_is_rejected_before_kernel_launch(self):
        from perception.environment.support_cuda import library
        with patch('perception.environment.support_cuda.C.CDLL',return_value=object()):
            with self.assertRaisesRegex(RuntimeError,'Rebuild environment CUDA'):
                library('synthetic-old-library')

    def test_sparse_vertical_body_and_disconnected_wall_without_cell_ground(self):
        cfg=load_config();box=np.array([[1.5,0,0,1.2,1.2,1.8,0,.99]])
        cloud=np.array([[1.45,-.05,-.5],[1.5,0,0],[1.55,.05,.3],
                        [1.95,-.05,-.5],[1.95,0,0],[1.95,.05,.3],[1.5,0,-1.]])
        for plane in (None,np.array([0,0,-1.])):
            actual=point_owners(cloud,box,cfg,np.zeros(3),plane)
            np.testing.assert_equal(actual,[0,0,0,-1,-1,-1,-1])

    def test_ownership_never_changes_preexisting_unknown_or_ordinary_background(self):
        current=object_scene([track(3,2)])
        current.geometry.layers['occupancy'][20,30]=-1
        current.geometry.layers['occupancy'][20,31]=100
        data=people_observation(object_scene([track(3,2)]))
        current=assemble_scene(current.geometry,clock_domain='synthetic',settings=load_config()['scene'],people=data)
        grid,_,report=PeopleModel(current,planning_config()['people']).navigation_grid(current)
        self.assertEqual(grid.layers['occupancy'][20,30],-1)
        self.assertEqual(grid.layers['occupancy'][20,31],100)
        self.assertEqual(report['delegated_tracks'],1)

    def test_unknown_velocity_is_delegated_individually_and_radius_covers_returns(self):
        current=object_scene([track(3,2,None,'a'),track(4,2,(0,1,0),'b')])
        model=PeopleModel(current,planning_config()['people']);grid,_,report=model.navigation_grid(current)
        self.assertEqual(report['delegated_tracks'],2)
        self.assertFalse(report['fallback_circles'])
        np.testing.assert_equal(model.known,[False,True]);np.testing.assert_equal(model.radii,[.42,.42])
        self.assertEqual(grid.layers['occupancy'][20,30],0)
        self.assertEqual(navigation_geometry(current).layers['occupancy'][20,30],100)

    def test_missing_or_mismatched_people_retains_current_circles(self):
        current=object_scene([track(3,2)])
        missing=assemble_scene(current.geometry,clock_domain='synthetic',settings=load_config()['scene'])
        model=PeopleModel(missing,planning_config()['people']);grid,_,report=model.navigation_grid(missing)
        self.assertEqual(report['delegated_tracks'],0)
        self.assertEqual(grid.layers['occupancy'][20,30],100)
        data=people_observation(current);data['input_fingerprint']='different'
        changed=assemble_scene(current.geometry,clock_domain='synthetic',settings=load_config()['scene'],people=data)
        self.assertEqual(PeopleModel(changed,planning_config()['people']).navigation_grid(changed)[0].layers['occupancy'][20,30],100)

    def test_one_missing_track_does_not_block_delegation_of_another(self):
        current=object_scene([track(3,2,id='a'),track(4,2,id='b')]);data=people_observation(current)
        data['tracks']=data['tracks'][:1]
        partial=assemble_scene(current.geometry,clock_domain='synthetic',settings=load_config()['scene'],people=data)
        grid,_,report=PeopleModel(partial,planning_config()['people']).navigation_grid(partial)
        self.assertEqual(report['delegated_tracks'],1)
        self.assertEqual(grid.layers['occupancy'][20,30],0)
        self.assertEqual(grid.layers['occupancy'][20,40],100)

    def test_wrong_frame_or_future_object_geometry_is_rejected(self):
        for changes in (dict(stamp_s=11.),dict(local_identity=['other',0,'odom'])):
            current=object_scene([track()]);current.geometry.metadata['person_objects'].update(changes)
            with self.assertRaises(ValueError):observed_objects(current.geometry)

    def test_prior_starts_without_local_floor_and_does_not_reset_on_local_jump(self):
        cfg=load_config();support=NavigationSupport(cfg);asset=prior_asset()
        for stamp,status in ((10.,dict(status='unavailable')),(10.2,dict(status='warming',clear_ground_history=True))):
            raw=observation(np.full((60,60),-1,np.int8),stamp=stamp);raw.metadata['ground_tracking']=status
            result=support.apply(raw,asset,observation=points(raw))
            self.assertEqual(navigation_geometry(scene(result),asset=asset,check_asset=True).layers['occupancy'][20,20],0)

    def test_local_obstacle_height_is_independent_of_association_z(self):
        cfg=load_config();support=NavigationSupport(cfg);asset=prior_asset()
        raw=observation(np.full((60,60),-1,np.int8));raw.metadata['association']['map_from_local'][2][3]=-.25
        # 20 cm above the observed local floor, but below the static height after the erroneous association.
        result=support.apply(raw,asset,observation=points(raw,[[2.05,2.05,-.65]],[0]))
        self.assertEqual(result.layers['navigation_occupancy'][20,20],100)

    def test_owned_returns_do_not_enter_background_occupancy(self):
        cfg=load_config();cfg['compute']['backend']='cpu'
        model=LocalEnvironmentMap(cfg);self.addCleanup(model.close)
        data=dict(stamp_s=10.,observation_id='p',input_fingerprint='p',session_id='people',generation=0,
            frame_id='odom',health='valid',clock_domain='synthetic',localization_binding=dict(session_id='test',generation=0,frame_id='odom'),
            tracks=[dict(track_id='a',box=[1.5,0,0,1.2,1.2,1.8,0,.99],state='observed',confirmed=True,hits=4,last_observed_stamp_s=10.)])
        cloud=np.array([[1.45,-.05,-.5],[1.5,0,0],[1.55,.05,.3],[1.95,-.05,-.5],[1.95,0,0],[1.95,.05,.3]])
        n=int(cfg['window_m']/cfg['resolution'])
        with patch.object(model,'_plane_ground',return_value=(np.full((n,n),-1.,np.float32),np.ones((n,n),bool),
                dict(status='tracked',plane_world=[0,0,-1.]))):
            raw=model.update(cloud,np.eye(4),10.,('test',0),frame='odom',people=data,clock_domain='synthetic')
        xy=np.floor((cloud[:,:2]-raw.metadata['origin_xy'])/cfg['resolution']).astype(int)
        self.assertFalse((raw.layers['occupancy'][xy[:3,1],xy[:3,0]]==100).any())
        self.assertTrue((raw.layers['occupancy'][xy[3:,1],xy[3:,0]]==100).any())
        self.assertEqual(len(observed_objects(raw)),1)
        np.testing.assert_equal(model.support_observation['person_owner'],[0,0,0,-1,-1,-1])


if __name__=='__main__':unittest.main()
