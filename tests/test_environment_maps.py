"""Final-map geometry and pose semantics; no ROS master, GPU, models or shared labels."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from perception.environment.layers import GridSnapshot, load_config
from perception.environment.maps import static_geometry
from perception.environment.poses import PoseTimeline
from perception.environment.export import export_environment, export_endpoint_environment
from perception.environment.storage import StaticEnvironmentMap, export_map_server_grid, content_hash
from perception.mapping_localization.results import read_pcd, file_fingerprint, map_fingerprint


def pose(stamp, *, x=0., health=0, session='local'):
    return dict(stamp_s=stamp, position=[x, 0, 0], quaternion_xyzw=[0, 0, 0, 1],
                session_id=session, generation=0, frame_id='odom', health=health)




class PoseTimelineTests(unittest.TestCase):
    def test_pose_queries_are_causal_do_not_bridge_failures_and_bind_final_corrections(self):
        settings = load_config()['poses']
        rows = [pose(1), pose(1.1, x=.1), pose(1.2, health=2), pose(1.3, x=.3), pose(1.4, x=.4)]
        keys = np.array([[1, 10, 0, 0, 0, 0, 0, 0, 0], [1.4, 12, 0, 0, 0, 0, 0, 1, 0]])
        timeline = PoseTimeline(rows, settings, keys)
        self.assertIsNone(timeline.at(.99))
        chosen = timeline.at(1.15)
        self.assertAlmostEqual(chosen['pose_stamp_s'], 1.1)
        self.assertAlmostEqual(chosen['world_from_lidar'][0][3], 10.1)
        self.assertEqual(chosen['final_keyframe'], 0)
        self.assertIsNone(timeline.at(1.25))
        self.assertIsNone(timeline.at(1.35))  # New valid segment has no final anchor yet.
        self.assertEqual(timeline.at(1.4)['final_keyframe'], 1)
        self.assertIsNone(timeline.at(2.))

    def test_local_pose_lookup_does_not_invent_a_map(self):
        timeline = PoseTimeline([pose(1), pose(1.1)], load_config()['poses'])
        row = timeline.at(1.15)
        self.assertEqual(row['pose_source'], 'recorded_local_pose')
        self.assertIsNone(row['final_keyframe'])
        self.assertEqual(row['local_frame'], 'odom')
        with self.assertRaises(ValueError):
            PoseTimeline([pose(1), pose(1)], load_config()['poses'])




class StaticMapTests(unittest.TestCase):
    def test_ground_relative_defaults_preserve_sparse_clusters_without_changing_online_support(self):
        from perception.environment.projection import ground_relative_projection
        config=load_config()
        cloud=np.array([[.05,.05,.5],[.23,.05,.5],[.24,.05,.5],[1.35,.05,.5]],np.float32)
        floor=np.zeros((1,15),np.float32)
        base=GridSnapshot(dict(shape=[1,15],origin_xy=[0.,0.],resolution=.1,snapshot_id='base'),
            dict(ground_z=floor,ground_support=np.ones_like(floor,dtype=bool),
                 obstacle_min_z=np.full_like(floor,np.nan),obstacle_max_z=np.full_like(floor,np.nan)))
        with patch('perception.environment.projection.static_geometry',return_value=base) as geometry:
            result=ground_relative_projection(cloud,[[0,0,1]],config,map_id='test',frame='map',stamp=1.)
        self.assertEqual(result.metadata['projection_counts']['radius_filtered_points'],3)
        self.assertEqual(result.metadata['projection_settings']['min_neighbors'],2)
        self.assertEqual(result.metadata['projection_settings']['radius_m'],.2)
        self.assertEqual(result.metadata['projection_settings']['ground_support_radius_m'],.75)
        self.assertEqual(geometry.call_args.args[2]['ground_support_radius_m'],.75)
        self.assertEqual(config['ground_support_radius_m'],.55)
        self.assertEqual(result.layers['occupancy'][0,13],0)
        self.assertTrue(result.layers['assumed_free'][0,13])
        self.assertTrue(np.isnan(result.layers['last_observed']).all())

    def test_ground_relative_projection_filters_height_and_keeps_unsupported_points(self):
        from perception.environment.projection import ground_relative_projection
        config = load_config()
        config.update(resolution=.1, min_obstacle_height_m=.12, max_height_m=1.8)
        # Each pair survives the 3-D neighbor filter. The last column has no
        # reliable ground estimate and must retain its endpoint as occupied.
        cloud = np.array([[.05,.05,0], [.051,.05,0], [.15,.05,.5], [.151,.05,.5],
                          [.25,.05,2.8], [.251,.05,2.8], [.35,.05,2.8], [.351,.05,2.8]], np.float32)
        floor = np.array([[0., 0., 0., np.nan, 0.]], np.float32)
        base = GridSnapshot(dict(shape=[1,5], origin_xy=[0.,0.], resolution=.1, snapshot_id='base'),
            dict(ground_z=floor, ground_support=np.isfinite(floor),
                 obstacle_min_z=np.full((1,5),np.nan), obstacle_max_z=np.full((1,5),np.nan)))
        with patch('perception.environment.projection.static_geometry', return_value=base):
            result = ground_relative_projection(cloud, [[0,0,1]], config, map_id='test', frame='map',
                                                stamp=1., radius=.02, min_neighbors=1)
        np.testing.assert_array_equal(result.layers['occupancy'], [[0,100,0,100,0]])
        np.testing.assert_array_equal(result.layers['assumed_free'], result.layers['occupancy']==0)
        self.assertEqual(result.metadata['projection_counts']['missing_ground_points'], 2)
        self.assertEqual(result.metadata['projection_counts']['height_band_points'], 2)
        self.assertTrue(np.isnan(result.layers['last_observed']).all())
        self.assertFalse(result.metadata['free_space_supported'])
        with self.assertRaises(ValueError):
            ground_relative_projection(cloud, [[0,0,1]], config, map_id='test', frame='map',
                                       stamp=1., height_min=.9, height_max=.2)

    def test_docker_projection_assumes_free_but_preserves_no_ray_evidence(self):
        from perception.environment.projection import docker_projection
        config=load_config()
        cloud=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0],[.005,.005,0],
                        [1,1,0],[1.01,1,0],[1,1.01,0],[1.01,1.01,0],[1.005,1.005,0],
                        [.5,.5,0],[0,0,20]],dtype=np.float32)
        snapshot=docker_projection(cloud,[[0,0,1]],config,map_id='test',frame='map',stamp=10.)
        cells=snapshot.layers['occupancy'];assumed=snapshot.layers['assumed_free']
        self.assertEqual(snapshot.metadata['projection_counts']['radius_filtered_points'],10)
        self.assertEqual(snapshot.metadata['projection_strategy'],'docker_assume_free')
        self.assertEqual(snapshot.metadata['free_assumption'],'no_filtered_endpoint_in_cell')
        self.assertFalse(snapshot.metadata['free_space_supported'])
        self.assertTrue((cells==0).any());self.assertTrue((cells==100).any())
        np.testing.assert_array_equal(assumed,cells==0)
        edge=np.floor((cloud[8,:2]-snapshot.metadata['origin_xy'])/snapshot.metadata['resolution']).astype(int)
        self.assertEqual(cells[edge[1],edge[0]],100)  # The final boundary is not silently dropped.
        self.assertTrue(np.isnan(snapshot.layers['last_observed']).all())
        self.assertFalse(snapshot.layers['coverage'].any())
        from perception.environment.context import query_geometry
        y,x=np.argwhere(assumed)[0]
        xy=np.asarray(snapshot.metadata['origin_xy'])+(np.array([x,y])+.5)*snapshot.metadata['resolution']
        result=query_geometry(snapshot,[xy],frame='map')[0]
        self.assertEqual(result['state'],'free')
        self.assertTrue(result['assumed_free'])
        self.assertEqual(result['free_source'],'assumed_no_endpoint')
        self.assertIsNone(result['last_observed_s'])
        with self.assertRaises(ValueError):
            docker_projection(cloud,[[0,0,1]],config,map_id='test',frame='map',stamp=10.,radius=0.)

    def test_ground_cell_uses_unique_point_median_instead_of_last_point(self):
        config=load_config();axis=np.arange(-2.,2.,.15);x,y=np.meshgrid(axis,axis)
        floor=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.)]
        samples=np.array([[.021,.021,-1.04],[.025,.025,-1.],[.029,.029,-.96]])
        cloud=np.r_[floor,samples].astype(np.float32)
        a=static_geometry(cloud,[[0,0,0],[.1,0,0]],config,map_id='prior',frame='map',stamp=10.)
        b=static_geometry(cloud[::-1],[[.1,0,0],[0,0,0]],config,map_id='prior',frame='map',stamp=10.)
        def height(s):
            ix,iy=np.floor((samples[0,:2]-s.metadata['origin_xy'])/.1).astype(int)
            return s.layers['ground_z'][iy,ix]
        self.assertAlmostEqual(float(height(a)),-1.,places=5)
        self.assertAlmostEqual(float(height(a)),float(height(b)),places=5)
        self.assertEqual(a.metadata['ground_estimator'],'per_cell_median_of_unique_ground_points')

    def geometry(self):
        config = load_config()
        axis = np.arange(-2.5, 2.5, .15)
        x, y = np.meshgrid(axis, axis)
        ground = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.)]
        wall = np.c_[np.full(30, 1.5), np.linspace(-1, 1, 30), np.linspace(-.85, .7, 30)]
        points = np.r_[ground, wall].astype(np.float32)
        return static_geometry(points, [[0, 0, 0]], config, map_id='prior', frame='map', stamp=10)

    def test_endpoints_preserve_thin_obstacles_but_never_fabricate_free_space_or_times(self):
        snapshot = self.geometry()
        self.assertEqual((snapshot.metadata['entity_type'],snapshot.metadata['map_type']),('map','static_environment_map'))
        self.assertTrue((snapshot.layers['occupancy'] == 100).any())
        self.assertFalse((snapshot.layers['occupancy'] == 0).any())
        self.assertTrue(np.isnan(snapshot.layers['last_observed']).all())
        self.assertFalse(snapshot.metadata['free_space_supported'])
        self.assertEqual(snapshot.metadata['time_scope'], 'offline_final_prior')
        self.assertTrue((snapshot.layers['endpoint_count'] > 0).any())

    def test_budget_rejection_cannot_silently_discard_obstacles(self):
        config = load_config(); config['static']['max_cells'] = 4
        with self.assertRaises(ValueError):
            static_geometry([[0, 0, -1], [2, 2, -1]], [[0, 0, 0]], config,
                            map_id='prior', frame='map', stamp=1)
        with tempfile.TemporaryDirectory() as directory:
            pcd = Path(directory)/'large.pcd'
            pcd.write_text('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 9\nDATA binary\n')
            with self.assertRaises(ValueError):
                read_pcd(pcd, 4, strict=True)

    def test_standard_navigation_export_preserves_unknown_and_row_orientation(self):
        snapshot = self.geometry()
        with tempfile.TemporaryDirectory() as directory:
            export_map_server_grid(snapshot, directory)
            path = Path(directory)
            pixels = (path/'map.pgm').read_bytes().split(b'\n', 3)[3]
            original = snapshot.layers['occupancy']
            expected = np.flipud(np.where(original == -1, 205, np.where(original == 100, 0, 254))).astype(np.uint8)
            self.assertEqual(pixels, expected.tobytes())
            self.assertEqual(yaml.safe_load((path/'map.yaml').read_text())['origin'], [*snapshot.metadata['origin_xy'], 0.])
            with self.assertRaises(FileExistsError):
                export_map_server_grid(snapshot, directory)

    def test_geometry_package_needs_no_visual_index_and_rejects_changed_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            snapshot=self.geometry();snapshot.save(path/'layers.npz')
            manifest=dict(kind='static_environment',snapshot_id=snapshot.metadata['snapshot_id'],frame_id='map',
                layers_sha256=content_hash(path/'layers.npz'),time_scope='offline_final_prior',source={})
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            package=StaticEnvironmentMap(path)
            self.assertEqual(package.summary()['map_type'],'static_environment_map')
            self.assertEqual(package.crop_map([-1,-1,1,1],frame='map')['snapshot_id'],manifest['snapshot_id'])
            with self.assertRaises(ValueError):package.query_geometry([[0,0]],frame='odom')
            package.close()
            manifest['kind']='local_environment_map'
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            with self.assertRaises(ValueError):StaticEnvironmentMap(path)
            manifest['kind']='static_environment'
            (path/'manifest.yaml').write_text(yaml.safe_dump(manifest))
            # Extra files in frozen old packages are not opened as current features.
            (path/'observations.sqlite').write_bytes(b'NOT_AN_INDEX')
            StaticEnvironmentMap(path).close()
            with (path/'layers.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):StaticEnvironmentMap(path)

    def test_static_export_uses_geometry_without_rgb_or_camera_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);run=path/'run';mapping=run/'map'
            (mapping/'keyframes').mkdir(parents=True)
            (mapping/'COMPLETE').touch()
            (mapping/'manifest.yaml').write_text('frame: map\n')
            (mapping/'keyframes/poses.txt').write_text('1 0 0 0 0 0 0 0 0\n1.1 0 0 0 0 0 0 1 0\n')
            axis=np.arange(-2.5,2.5,.15);x,y=np.meshgrid(axis,axis)
            points=np.c_[x.ravel(),y.ravel(),np.full(x.size,-1.)]
            with (mapping/'GlobalMap.pcd').open('w') as stream:
                stream.write('FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS %d\nDATA ascii\n'%len(points))
                np.savetxt(stream,points)
            bag=path/'source.bag';bag.write_bytes(b'BOUND_SOURCE_NO_IMAGE_DECODING')
            (run/'localization.jsonl').write_text(''.join(json.dumps(pose(t))+'\n' for t in (1.,1.1)))
            record=dict(success=True,source='bag',mode='mapping',bag=str(bag),
                playback=dict(bag_fingerprint=file_fingerprint(bag),map_fingerprint=map_fingerprint(mapping)))
            (run/'run.json').write_text(json.dumps(record))
            out=path/'export'
            with self.assertRaisesRegex(ValueError,'sealed scan journal'):
                export_environment(run,out,export_pgm=True)
            manifest=export_environment(run,out,export_pgm=True,filter_mode='none')
            self.assertEqual({p.name for p in out.iterdir()},{'manifest.yaml','layers.npz','map.pgm','map.yaml'})
            self.assertEqual(manifest['kind'],'static_environment')
            self.assertEqual(manifest['map_type'],'static_environment_map')
            self.assertEqual(manifest['configuration']['compute']['backend'],'cpu')
            self.assertNotIn('images',manifest)
            self.assertNotIn('calibration',manifest)
            self.assertEqual(StaticEnvironmentMap(out).summary()['frame_id'],'map')
            # An older map can be imported independently, with missing rays explicitly retained as unknown.
            old=path/'old_map_export'
            imported=export_endpoint_environment(mapping,old,export_pgm=True)
            package=StaticEnvironmentMap(old)
            self.assertEqual(imported['source']['map_fingerprint'],map_fingerprint(mapping))
            self.assertEqual(imported['source']['input_kind'],'localization_map_endpoints')
            self.assertEqual(imported['configuration']['compute']['backend'],'cpu')
            self.assertFalse((package.snapshot.layers['occupancy']==0).any())
            self.assertIn('no_recorded_rays_or_free_space',imported['limitations'])
            docker=path/'docker_projection'
            projected=export_endpoint_environment(mapping,docker,export_pgm=True,
                projection_strategy='docker_assume_free',
                projection_settings=dict(radius=.25,min_neighbors=2))
            loaded=StaticEnvironmentMap(docker).snapshot
            self.assertIn('unobserved_cells_assumed_free_not_measured',projected['limitations'])
            np.testing.assert_array_equal(loaded.layers['assumed_free'],loaded.layers['occupancy']==0)
            self.assertTrue(np.isnan(loaded.layers['last_observed']).all())
            self.assertTrue((loaded.layers['occupancy']==0).any())
            relative=path/'ground_relative_projection'
            relative_manifest=export_endpoint_environment(mapping,relative,export_pgm=True,
                projection_strategy='ground_relative_assume_free',
                projection_settings=dict(radius=.25,min_neighbors=2))
            relative_loaded=StaticEnvironmentMap(relative).snapshot
            self.assertEqual(relative_loaded.metadata['projection_strategy'],'ground_relative_assume_free')
            self.assertIn('unobserved_cells_assumed_free_not_measured',relative_manifest['limitations'])
            np.testing.assert_array_equal(relative_loaded.layers['assumed_free'],relative_loaded.layers['occupancy']==0)
            self.assertTrue(np.isnan(relative_loaded.layers['last_observed']).all())
            bad_relative=path/'bad_relative_projection';bad_relative.mkdir()
            bad_relative_layers={name:value.copy() for name,value in relative_loaded.layers.items()}
            bad_relative_layers['last_observed'][bad_relative_layers['assumed_free']]=1.
            GridSnapshot(dict(relative_loaded.metadata),bad_relative_layers).save(bad_relative/'layers.npz')
            bad_relative_manifest=dict(relative_manifest,layers_sha256=content_hash(bad_relative/'layers.npz'))
            (bad_relative/'manifest.yaml').write_text(yaml.safe_dump(bad_relative_manifest))
            with self.assertRaisesRegex(ValueError,'absence of observation times'):
                StaticEnvironmentMap(bad_relative)
            # A matching package hash cannot turn assumed space into measured time.
            bad=path/'bad_projection';bad.mkdir()
            bad_layers={name:value.copy() for name,value in loaded.layers.items()}
            bad_layers['last_observed'][bad_layers['assumed_free']]=1.
            GridSnapshot(dict(loaded.metadata),bad_layers).save(bad/'layers.npz')
            bad_manifest=dict(projected,layers_sha256=content_hash(bad/'layers.npz'))
            (bad/'manifest.yaml').write_text(yaml.safe_dump(bad_manifest))
            with self.assertRaisesRegex(ValueError,'absence of observation times'):
                StaticEnvironmentMap(bad)


if __name__ == '__main__':
    unittest.main()
