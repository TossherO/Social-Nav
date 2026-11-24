"""Synthetic multi-run update; never touches real maps or sensors."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from perception.environment.layers import load_config
from perception.environment.occupancy import VoxelBudgetExceeded
from perception.environment.storage import StaticEnvironmentMap, content_hash
from perception.mapping_localization.manual_edit import read_xyzi
from perception.mapping_localization.results import map_fingerprint,read_pcd
from perception.mapping_localization.update_map import update
from tests.mapping_fixtures import registered_map_inputs


class LocalMapUpdateTests(unittest.TestCase):
    def test_pose_innovation_is_measured_at_current_lidar_position(self):
        from scipy.spatial.transform import Rotation
        from perception.mapping_localization.update_map import _pose_innovation
        local0=np.eye(4);local0[0,3]=30.
        local1=np.eye(4);local1[0,3]=30.1
        map0=local0.copy();map1=local1.copy()
        map1[:3,:3]=Rotation.from_euler('z',.04).as_matrix()
        correction0=map0@np.linalg.inv(local0);correction1=map1@np.linalg.inv(local1)
        self.assertGreater(np.linalg.norm((np.linalg.inv(correction0)@correction1)[:3,3]),.5)
        distance,angle=_pose_innovation(map0,local0,map1,local1)
        self.assertAlmostEqual(distance,0.,places=8)
        self.assertAlmostEqual(angle,.04,places=8)
        map1[0,3]+=.7
        self.assertGreater(_pose_innovation(map0,local0,map1,local1)[0],.5)

    def test_nearby_stable_hit_vetoes_free_vote_deletion(self):
        from scipy.spatial import cKDTree
        from perception.mapping_localization.update_map import _masks
        class Votes:
            def static_votes(self,points):return np.tile([0,5],(len(points),1))
        mask=_masks(Votes(),np.array([[0.,0.,0.],[2.,0.,0.]]),load_config(),
                    hit_tree=cKDTree([[.1,0.,0.]]))
        np.testing.assert_array_equal(mask,[False,True])

    def test_update_retains_unobserved_old_points_and_produces_a_new_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            before=map_fingerprint(old);report=update(old,mapping,reference,root/'candidate')
            self.assertEqual(map_fingerprint(old),before)
            self.assertEqual(report['status'],'candidate_not_accepted')
            self.assertEqual(report['evidence_policy']['max_voxels'],load_config()['static']['max_voxels'])
            self.assertGreater(report['counts']['old_removed'],0)
            self.assertGreater(report['counts']['new_kept'],0)
            cloud=read_xyzi(root/'candidate/map/keyframes/surf0.pcd')
            self.assertFalse(np.any(np.all(np.isclose(cloud[:,:3],[1,0,0]),axis=1)))
            self.assertTrue(np.any(np.all(np.isclose(cloud[:,:3],[1,1,0]),axis=1)))
            self.assertTrue((root/'candidate/map/COMPLETE').is_file())
            self.assertTrue((root/'candidate/comparison.png').is_file())
            with np.load(root/'candidate/changes.npz',allow_pickle=False) as changes:
                self.assertEqual(len(changes['removed_map_xyz']),report['counts']['old_removed'])
                self.assertEqual(len(changes['added_map_xyz']),report['counts']['new_kept'])
            self.assertTrue((root/'candidate/map/update_source.json').is_file())

    def test_custom_3d_voxel_budget_bounds_evidence_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            output=root/'candidate'
            with self.assertRaisesRegex(ValueError,'positive integer'):
                update(old,mapping,reference,output,max_voxels=0)
            with self.assertRaisesRegex(VoxelBudgetExceeded,'3D update voxel limit: 1'):
                update(old,mapping,reference,output,max_voxels=1)
            self.assertFalse(output.exists())
            result=update(old,mapping,reference,output,max_voxels=1000)
            self.assertGreater(result['evidence']['retained_voxels'],1)
            self.assertLessEqual(result['evidence']['retained_voxels'],1000)
            for name in ('report.json','map/update_source.json'):
                report=json.loads((output/name).read_text())
                self.assertEqual(report['evidence_policy']['max_voxels'],1000)

    def test_different_bag_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            target=reference/'run.json';value=json.loads(target.read_text())
            value['playback']['bag_fingerprint']={'wrong':True};target.write_text(json.dumps(value))
            with self.assertRaises(ValueError):update(old,mapping,reference,root/'candidate')
            self.assertFalse((root/'candidate').exists())
            with self.assertRaises(FileExistsError):update(old,mapping,reference,old)

    def test_static_candidate_uses_final_map_binding_and_publishes_together(self):
        from perception.environment.export import _save_package
        from perception.environment.maps import static_geometry

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            static=root/'old_static';output=root/'candidate';config=load_config()
            geometry=static_geometry(read_pcd(old/'GlobalMap.pcd'),[[0.,0.,0.]],config,
                                     map_id='parent',frame='map',stamp=101.)
            _save_package(geometry,static,dict(map_fingerprint=map_fingerprint(old)),config,'map',2,False,[])
            source_hashes={name:content_hash(static/name) for name in ('manifest.yaml','layers.npz')}
            with patch('perception.mapping_localization.update_map._masks',
                       side_effect=lambda tree,world,*args,**kwargs: np.zeros(len(world),bool)):
                with self.assertRaisesRegex(ValueError,'inside an input'):
                    update(old,mapping,reference,static/'bad-output',old_static=static)
                with patch('perception.environment.incremental.export_incremental_environment',
                           side_effect=ValueError('failed static fusion')):
                    with self.assertRaisesRegex(ValueError,'failed static fusion'):
                        update(old,mapping,reference,output,old_static=static)
                self.assertFalse(output.exists())
                result=update(old,mapping,reference,output,old_static=static)
            self.assertEqual(result['counts']['old_removed'],0)
            self.assertEqual(result['counts']['new_kept'],0)
            self.assertEqual(content_hash(output/'map/GlobalMap.pcd'),content_hash(old/'GlobalMap.pcd'))
            self.assertTrue((output/result['static_map']['report']).is_file())
            candidate=StaticEnvironmentMap(output/'static_map')
            self.assertEqual(candidate.manifest['source']['map']['path'],str(output/'map'))
            self.assertEqual(candidate.manifest['source']['map_fingerprint'],map_fingerprint(output/'map'))
            self.assertEqual(candidate.snapshot.metadata['frame_id'],'map')
            for name,digest in source_hashes.items():
                self.assertEqual(content_hash(static/name),digest)

    def test_registration_jump_and_overlapping_session_time_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            path=reference/'localization.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[3]['map_position'][0]+=2.
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'correction jump'):
                update(old,mapping,reference,root/'jump')
            self.assertFalse((root/'jump').exists())
            pose_path=old/'keyframes/poses.txt';poses=np.loadtxt(pose_path,ndmin=2)
            poses[:,0]=[200.,201.];np.savetxt(pose_path,poses,fmt='%.17g')
            run=reference/'run.json';data=json.loads(run.read_text())
            data['playback']['map_fingerprint']=map_fingerprint(old);run.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'no timestamps are rewritten'):
                update(old,mapping,reference,root/'overlap')
            self.assertFalse((root/'overlap').exists())

    def test_mismatched_sensor_timing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=registered_map_inputs(root)
            path=reference/'run.json';run=json.loads(path.read_text())
            run['algorithm_parameters']['common']['imu_time_offset']=.1
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError,'different sensor extrinsics or IMU timing'):
                update(old,mapping,reference,root/'candidate')
            self.assertFalse((root/'candidate').exists())
