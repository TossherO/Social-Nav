"""Rigid sensor floor tracking rejects raised furniture without inventing free cells."""
import unittest
import numpy as np
from perception.environment.ground import GroundTracker
from perception.environment.layers import load_config,LocalEnvironmentMap
from perception.environment.poses import PoseTimeline


def scalar_ground_plane(points,sensor,config):
    """Frozen scalar oracle for sampling, numeric thresholds and first-max ties."""
    delta=points-sensor
    samples=points[(np.linalg.norm(delta[:,:2],axis=1)<config['ground_fit_radius_m']) &
        (delta[:,2]<-config['ground_min_below_sensor_m']) & (delta[:,2]>-config['ground_max_below_sensor_m'])]
    if len(samples)<config['ground_min_points']:return None
    samples=samples[::max(1,int(np.ceil(len(samples)/3000)))]
    xy=np.c_[samples[:,:2],np.ones(len(samples))];random=np.random.default_rng(0);best=None
    for _ in range(40):
        index=random.choice(len(samples),3,replace=False)
        if abs(np.linalg.det(xy[index]))<1e-6:continue
        plane=np.linalg.solve(xy[index],samples[index,2])
        if np.linalg.norm(plane[:2])>np.tan(config['ground_max_slope_rad']):continue
        mask=np.abs(xy@plane-samples[:,2])<=config['ground_tolerance_m']
        if best is None or mask.sum()>best[0]:best=(int(mask.sum()),mask)
    if best is None or best[0]<max(config['ground_min_points'],len(samples)*config['ground_min_ratio']):return None
    plane=np.linalg.lstsq(xy[best[1]],samples[best[1],2],rcond=None)[0]
    return None if np.linalg.norm(plane[:2])>np.tan(config['ground_max_slope_rad']) else plane


class GroundBatchTests(unittest.TestCase):
    def test_spatial_keys_preserve_signed_cell_order_and_first_low_point(self):
        from perception.environment.ground import spatial_floor_points
        config=load_config();settings=GroundTracker.DEFAULTS
        points=np.array([[.1,.1,-.85],[-.1,.1,-.84],[-.1,-.1,-.83],[.1,-.1,-.82],
            [-.11,-.11,-.83],[.11,.11,-.50]])
        support,seeds=spatial_floor_points(points,np.zeros(3),config,settings)
        # Positive-positive cell is a vertical column; duplicate lows keep scan order.
        np.testing.assert_equal(support,points[[1,2,3,4]])
        np.testing.assert_equal(seeds,points[[2,1,3]])

    def test_batch_preserves_scalar_hypotheses_inliers_and_refit(self):
        from perception.environment.geometry import ground_plane
        config=load_config();rng=np.random.default_rng(30)
        for count in (29,30,60,301,3000,3001,9001):
            for dtype in (np.float32,np.float64):
                sensor=np.array([17.2,-11.8,.3],dtype=dtype)
                points=np.c_[rng.uniform(-7,7,(count,2)),rng.normal(-.85,.015,count)]
                points[:,2]+=.02*points[:,0];points[::7,2]+=.3
                points=points.astype(dtype)+sensor
                np.testing.assert_equal(ground_plane(points,sensor,config),scalar_ground_plane(points,sensor,config))

    def test_batch_preserves_degenerate_and_threshold_cases(self):
        from perception.environment.geometry import ground_plane
        config=load_config();sensor=np.zeros(3)
        x,y=np.meshgrid(np.linspace(-2,2,30),np.linspace(-2,2,30))
        for scale in (0.,1.-1e-13,1.,1.+1e-13):
            z=-.85+x.ravel()*np.tan(config['ground_max_slope_rad'])*scale
            points=np.c_[x.ravel(),y.ravel(),z]
            for error in (0.,config['ground_tolerance_m']-1e-14,config['ground_tolerance_m']+1e-14):
                cloud=points.copy();cloud[::3,2]+=error
                np.testing.assert_equal(ground_plane(cloud,sensor,config),scalar_ground_plane(cloud,sensor,config))
        line=np.c_[np.linspace(-2,2,100),np.zeros(100),np.full(100,-.85)]
        self.assertIsNone(ground_plane(line,sensor,config))

    def test_identical_groups_share_fit_without_changing_source_priority(self):
        from unittest.mock import patch
        import perception.environment.ground as ground
        config=load_config();tracker=GroundTracker(config['ground_tracking'])
        x,y=np.meshgrid(np.linspace(-3,3,30),np.linspace(-3,3,30));points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        with patch.object(ground,'fit_ground_plane',wraps=ground.fit_ground_plane) as fit:
            for _ in range(6):plane,status=tracker.fit(points,np.zeros(3),config)
            self.assertEqual(fit.call_count,6)
        self.assertIsNotNone(plane);self.assertEqual(status['fit_source'],'spatial_low')


class FloorTests(unittest.TestCase):
    def test_dense_wall_returns_do_not_tilt_the_visible_floor(self):
        config=load_config();tracker=GroundTracker(config['ground_tracking'])
        x,y=np.meshgrid(np.linspace(-2,2,40),np.linspace(2.5,5.5,35))
        floor=np.c_[x.ravel(),y.ravel(),-.85+.02*x.ravel()]
        y,z=np.meshgrid(np.linspace(.5,5.5,100),np.linspace(-.75,-.21,30))
        walls=np.r_[np.c_[np.full(y.size,-1.3),y.ravel(),z.ravel()],
                    np.c_[np.full(y.size,1.3),y.ravel(),z.ravel()]]
        for _ in range(5):plane,status=tracker.fit(np.r_[floor,walls],np.zeros(3),config)
        self.assertIsNotNone(plane);self.assertAlmostEqual(tracker.reference,.85,places=2)
        self.assertAlmostEqual(plane[0],.02,places=2)

    def test_far_narrow_floor_cannot_lock_an_extrapolated_sensor_height(self):
        config=load_config();tracker=GroundTracker(config['ground_tracking'])
        x,y=np.meshgrid(np.linspace(-2,2,50),np.linspace(4.9,5.7,30))
        floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        a,b=np.meshgrid(np.linspace(-1,1,20),np.linspace(1,2,20))
        foreground=np.c_[a.ravel(),b.ravel(),np.full(a.size,-.4)]
        for _ in range(8):plane,status=tracker.fit(np.r_[floor,foreground],np.zeros(3),config)
        self.assertIsNone(plane);self.assertIsNone(tracker.reference)
        self.assertEqual(status['reason'],'ground_fit_or_extrapolation_uncertain')
        self.assertAlmostEqual(status['measured_height_m'],.85,places=2)

    def test_wrong_high_floor_reference_recovers_after_lower_ground_confirmation(self):
        config=load_config();tracker=GroundTracker(config['ground_tracking']);tracker.reference=.54
        x,y=np.meshgrid(np.linspace(-3,3,25),np.linspace(-3,3,25))
        floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        for _ in range(config['ground_tracking']['reacquire_observations']-1):
            plane,status=tracker.fit(floor,np.zeros(3),config)
            self.assertIsNone(plane);self.assertAlmostEqual(tracker.reference,.54)
            self.assertTrue(status['clear_ground_history'])
        plane,status=tracker.fit(floor,np.zeros(3),config)
        self.assertIsNotNone(plane);self.assertAlmostEqual(tracker.reference,.85,places=2)
        self.assertTrue(status['reference_changed'])
        table=floor.copy();table[:,2]=-.3
        for _ in range(12):
            plane,status=tracker.fit(table,np.zeros(3),config)
            self.assertIsNone(plane);self.assertAlmostEqual(tracker.reference,.85)

    def test_reference_conflict_withdraws_ground_and_near_field_history(self):
        config=load_config();config['compute']['backend']='cpu'
        model=LocalEnvironmentMap(config);self.addCleanup(model.close)
        x,y=np.meshgrid(np.linspace(-3,3,40),np.linspace(-3,3,40))
        floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        for i in range(5):view=model.update(floor,np.eye(4),1+i*.1,('test',0),scan_duration=.1)
        self.assertTrue(view.layers['ground_support'].any())
        model.ground_tracker.reference=.54  # Explicit fault injection, not sensor feedback.
        view=model.update(floor,np.eye(4),1.6,('test',0),scan_duration=.1)
        self.assertFalse(view.layers['ground_support'].any())
        self.assertFalse(view.layers['near_field_free'].any())

    def test_floor_bootstrap_ignores_larger_table_then_tracks_partial_floor(self):
        config=load_config();tracker=GroundTracker(config['ground_tracking'])
        rng=np.random.default_rng(3)
        floor=np.c_[rng.uniform(-4,4,(900,2)),rng.normal(-.84,.003,900)]
        table=np.c_[rng.uniform(-3,3,(3000,2)),rng.normal(-.30,.003,3000)]
        for i in range(5):plane,status=tracker.fit(np.r_[floor,table],np.zeros(3),config)
        self.assertIsNotNone(plane);self.assertAlmostEqual(tracker.reference,.84,places=2)
        self.assertEqual(status['status'],'tracked')
        plane,status=tracker.fit(table,np.zeros(3),config)
        self.assertIsNone(plane);self.assertTrue(status['no_missing_cells_filled'])
        self.assertAlmostEqual(tracker.reference,.84,places=2)
        plane,status=tracker.fit(floor[:100],np.zeros(3),config)
        self.assertIsNotNone(plane)
        tilted=floor.copy();tilted[:,2]+=.15*tilted[:,0]
        plane,status=tracker.fit(tilted,np.zeros(3),config)
        self.assertIsNotNone(plane);self.assertAlmostEqual(float(plane[0]),.15,places=2)

    def test_initialization_requires_support_and_does_not_publish_warming_plane(self):
        config=load_config();tracker=GroundTracker(config['ground_tracking'])
        x,y=np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20));points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.85)]
        for _ in range(4):
            plane,status=tracker.fit(points,np.zeros(3),config);self.assertIsNone(plane)
        self.assertEqual(status['status'],'warming')
        tracker.fit(np.empty((0,3)),np.zeros(3),config)
        self.assertEqual(len(tracker.heights),0)
        self.assertIsNone(tracker.reference)

class RegisteredEvidenceTests(unittest.TestCase):
    def test_registered_pose_requires_same_observation_and_valid_map_pose(self):
        row=dict(stamp_s=1.,stamp_ns=10**9,session_id='local',generation=0,frame_id='odom',
                 child_frame_id='lidar',health=0,position=[1,0,0],quaternion_xyzw=[0,0,0,1])
        target=dict(row,has_map_pose=True,map_position=[10,3,0],map_quaternion_xyzw=[0,0,0,1])
        timeline=PoseTimeline([row],load_config()['poses'],registered_rows=[target])
        result=timeline.at(1.,exact_ns=10**9)
        self.assertEqual(result['world_from_lidar'][0][3],10)
        self.assertEqual(result['local_from_lidar'][0][3],1)
        self.assertEqual(result['pose_source'],'same_observation_registered_map_pose')
        target['has_map_pose']=False;self.assertIsNone(timeline.at(1.,exact_ns=10**9))
        target['has_map_pose']=True;target['child_frame_id']='other'
        with self.assertRaises(ValueError):timeline.at(1.,exact_ns=10**9)
        with self.assertRaises(ValueError):PoseTimeline([row],load_config()['poses'],registered_rows=[target,target])

    def test_registered_scan_endpoints_and_moving_origins_receive_same_correction(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock,patch
        from perception.environment.scans import ScanJournal
        from perception.environment.maps import static_geometry
        from perception.environment.fusion import fuse_scans
        config=load_config()
        row=dict(stamp_s=1.,stamp_ns=10**9,session_id='local',generation=0,frame_id='odom',
                 child_frame_id='lidar',health=0,position=[1,0,0],quaternion_xyzw=[0,0,0,1])
        target=dict(row,has_map_pose=True,map_position=[10,3,0],map_quaternion_xyzw=[0,0,2**-.5,2**-.5])
        timeline=PoseTimeline([row],config['poses'],registered_rows=[target])
        x,y=np.meshgrid(np.arange(8,12,.1),np.arange(1,5,.1));floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1)]
        candidate=static_geometry(floor,[[10,3,0]],config,map_id='target',frame='map',stamp=1.)
        tree=Mock();tree.snapshot.return_value=np.empty((0,5),np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite';writer=ScanJournal(path,config['scan_recording'],{})
            transform=np.eye(4);transform[0,3]=1.
            writer.append([[2,0,-1]],transform,10**9,dict(session_id='local',generation=0,frame_id='odom',
                rays_used=True,ray_origin_model='pose_interpolated_origins'),[[1.1,0,0]])
            writer.close()
            with patch('perception.environment.fusion.OccupancyTree',return_value=tree):
                result=fuse_scans(candidate,path,timeline,config,{},filter_mode='none')
        call=tree.update.call_args
        np.testing.assert_allclose(call.args[0],[[10,5,-1]],atol=1e-6)
        np.testing.assert_allclose(call.kwargs['ray_origins'],[[10,3.1,0]],atol=1e-6)
        self.assertEqual(result.metadata['ray_scans'],1)
