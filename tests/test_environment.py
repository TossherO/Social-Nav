"""Small, risk-focused geometry and localization association regressions."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from perception.mapping_localization.context import LocalizationContext
from perception.environment.layers import GridSnapshot, LocalEnvironment, load_config
from perception.environment.context import crop_map, query_geometry
from perception.environment.geometry import nearest_support, shift_array, interpolate_origins
from perception.environment.occupancy import OccupancyTree
from perception.visualization.environment import add_environment_display


def pose(stamp, *, health=0, session='front', generation=0):
    return dict(stamp_s=stamp,frame_id='odom',session_id=session,generation=generation,
                source='fast_lio',health=health,reason='' if health == 0 else 'invalid',
                position=[stamp,0,0],quaternion_xyzw=[0,0,0,1],map_revision=0)


def match(stamp, *, health=0, generation=0):
    return dict(pose(stamp,health=health,session='map',generation=generation),
                has_map_pose=health == 0,map_frame='map',map_position=[stamp+20,0,0],
                map_quaternion_xyzw=[0,0,0,1],map_revision=4)


class AssociationTests(unittest.TestCase):
    def test_map_loss_retains_continuous_pose_and_reassociation_does_not_reset_local_identity(self):
        context = LocalizationContext('roll_fast_lio',requires_map=True,allow_local_only=True)
        for stamp in [1,1.1]:
            context.update(pose(stamp))
            context.add_match(match(stamp))
        row,binding = context.update(pose(1.2))
        self.assertTrue(row['has_map_pose'])
        self.assertAlmostEqual(binding['measurement_stamp_s'],1.1)
        self.assertAlmostEqual(row['map_matrix'][0,3],21.2)
        context.add_match(match(1.2,health=2))
        row,binding = context.update(pose(1.3))
        self.assertEqual((row['health'],binding['status'],row['has_map_pose']),(0,'LOCAL_ONLY',False))
        for stamp in [1.3,1.4]:
            if stamp == 1.4:
                context.update(pose(stamp))
            context.add_match(match(stamp,generation=1))
        row,binding = context.update(pose(1.5))
        self.assertEqual((row['session_id'],row['generation']),('front',0))
        self.assertTrue(binding['valid'])
        self.assertEqual(binding['association_generation'],1)

    def test_invalid_frontend_strict_prior_and_stale_or_future_results_do_not_become_valid(self):
        strict = LocalizationContext('roll_fast_lio',requires_map=True)
        self.assertEqual(strict.update(pose(1))[0]['health'],2)
        mixed = LocalizationContext('roll_fast_lio',requires_map=True,allow_local_only=True)
        mixed.update(pose(1))
        mixed.add_match(match(2))  # Future result cannot be attached to an earlier pose.
        self.assertFalse(mixed.update(pose(1.1))[0]['has_map_pose'])
        invalid,association=mixed.update(pose(1.2,health=2))
        self.assertEqual((invalid['status'],association['status']),('LOCAL_INVALID','LOCAL_INVALID'))
        self.assertFalse(mixed.update(pose(3))[0]['has_map_pose'])

    def test_retired_sessions_duplicates_and_mismatched_pose_binding(self):
        context = LocalizationContext('roll_fast_lio',allow_local_only=True,confirm_matches=1)
        context.update(pose(1))
        wrong = match(1); wrong['position']=[99,0,0]
        context.add_match(wrong)
        self.assertFalse(context.update(pose(1.1))[1]['valid'])
        self.assertIsNone(context.update(pose(1.1)))
        context.update(pose(2,session='new'))
        self.assertIsNone(context.update(pose(2.1)))
        context.add_match(match(2))
        self.assertFalse(context.update(pose(2.1,session='new'))[1]['valid'])


class GeometryTests(unittest.TestCase):
    def test_moving_ray_origins_are_causal_and_never_extrapolated_across_a_gap(self):
        times=np.array([0,50,100.])
        samples=[(1.,[0,0,0]),(1.1,[.1,0,0])]
        origins=interpolate_origins(times,1.1,samples,.2)
        np.testing.assert_allclose(origins[:,0],[0,.05,.1],atol=1e-6)
        # RS organized firings can precede point zero. A reference shift must
        # not change physical ray origins or discard the whole scan.
        np.testing.assert_allclose(interpolate_origins(times-.53,1.1,samples,.2),origins,atol=1e-6)
        self.assertIsNone(interpolate_origins(times,1.,samples,.2))
        self.assertIsNone(interpolate_origins(times,1.3,[(1.,[0,0,0]),(1.3,[.3,0,0])],.2))
        self.assertIsNone(interpolate_origins([0,0],1.1,samples,.2))

    def test_support_is_an_actual_seed_and_rolling_shift_does_not_wrap_cells(self):
        mask=np.zeros((8,9),bool);mask[1,2]=True;mask[6,7]=True
        distances,indices=nearest_support(mask)
        self.assertTrue(mask[indices].all())
        self.assertEqual(distances[1,2],0)
        old=np.arange(12).reshape(3,4)
        new=shift_array(old,1,-1,-1)
        self.assertTrue((new[0,1:]==old[1,:3]).all())
        self.assertTrue((new[:,0]==-1).all())
        self.assertTrue((new[-1]==-1).all())

    def test_environment_display_keeps_current_deskewed_scan_and_unknown_map(self):
        config={'Visualization Manager':{'Displays':[{'Name':'Current scan (observation-bound)','Topic':'old'}]}}
        result=add_environment_display(config,'/site/environment')['Visualization Manager']['Displays']
        self.assertEqual(result[0]['Topic'],'/site/environment/scan')
        self.assertEqual(result[1]['Topic'],'/site/environment/occupancy')
        following={'Visualization Manager':{'Displays':[],'Global Options':{'Fixed Frame':'odom'}}}
        add_environment_display(following,target_frame='lidar_sensor')
        self.assertEqual(following['Visualization Manager']['Views']['Current']['Target Frame'],'lidar_sensor')
        self.assertEqual(following['Visualization Manager']['Global Options']['Fixed Frame'],'odom')

    def test_native_hit_wins_over_free_ray_and_expiration_returns_unknown(self):
        config=load_config()
        tree=OccupancyTree(.15)
        self.addCleanup(tree.close)
        bounds=np.array([-3,-3,-3,3,3,3],float)
        cloud=np.array([[1.05,0,0],[2.05,0,0]],np.float32)
        voxels=tree.update(cloud,[0,0,0],bounds,1,config,rays=True)
        closest=np.argmin(np.linalg.norm(voxels[:,:3]-cloud[0],axis=1))
        self.assertGreater(voxels[closest,3],.65)
        empty=tree.update(np.empty((0,3)),[0,0,0],bounds,10,config,rays=False)
        self.assertEqual(len(empty),0)

    def test_local_geometry_unknown_visibility_and_snapshot_queries_survive_roundtrip(self):
        config=load_config(); config.update(window_m=6.,resolution=.15,ground_fit_radius_m=4.,max_range_m=5.)
        engine=LocalEnvironment(config);self.addCleanup(engine.close)
        axis=np.arange(-2.7,2.7,.15)
        xx,yy=np.meshgrid(axis,axis)
        floor=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,-1.)]
        wall=np.c_[np.full(40,1.5),np.linspace(-1,1,40),np.linspace(-.8,.7,40)]
        points=np.r_[floor,wall].astype(np.float32)
        first=engine.update(points,np.eye(4),1,('session',0),scan_duration=.1)
        self.assertFalse(first.metadata['rays_used'])
        self.assertFalse((first.layers['occupancy']==0).any())
        self.assertTrue((first.layers['occupancy']==100).any())
        snapshot=engine.update(points,np.eye(4),1.2,('session',0),scan_duration=.1)
        self.assertTrue(snapshot.metadata['rays_used'])
        self.assertTrue((snapshot.layers['occupancy']==-1).any())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'layers.npz'
            snapshot.save(path)
            restored=GridSnapshot.load(path)
            np.testing.assert_array_equal(restored.layers['occupancy'],snapshot.layers['occupancy'])
            self.assertEqual(query_geometry(restored,[[100,100]],frame='odom')[0]['state'],'outside_coverage')
            with self.assertRaises(ValueError):
                query_geometry(restored,[[0,0]],frame='map')
            crop=crop_map(restored,[-1,-1,1,1],frame='odom')
            self.assertEqual(crop['snapshot_id'],snapshot.metadata['snapshot_id'])
        reset=engine.update(points,np.eye(4),2,('new',0),scan_duration=.1)
        self.assertFalse(reset.metadata['rays_used'])
        with self.assertRaises(KeyError):
            engine.get_snapshot(snapshot.metadata['snapshot_id'])


if __name__ == '__main__':
    unittest.main()
