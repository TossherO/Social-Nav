"""Synthetic multi-run update; never touches real maps or sensors."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from perception.environment.layers import load_config
from perception.environment.scans import ScanJournal
from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.manual_edit import read_xyzi,write_xyzi
from perception.mapping_localization.results import file_fingerprint,map_fingerprint
from perception.mapping_localization.update_map import update


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

    def write_map(self,path,rows,points):
        (path/'keyframes').mkdir(parents=True)
        for index,cloud in enumerate(points):
            write_xyzi(path/'keyframes'/f'corner{index}.pcd',np.empty((0,4),np.float32))
            write_xyzi(path/'keyframes'/f'surf{index}.pcd',np.asarray(cloud,np.float32))
        np.savetxt(path/'keyframes/poses.txt',np.asarray(rows),fmt='%.17g')
        write_xyzi(path/'GlobalMap.pcd',np.concatenate([np.asarray(p,np.float32) for p in points]))
        (path/'manifest.yaml').write_text(yaml.safe_dump(dict(format='roll_keyframes_v1',frame='map',
            units='metres_radians',keyframes=len(rows),visualization_resolution=.1)))
        (path/'COMPLETE').write_text('roll_keyframes_v1\n')

    def fixture(self,root):
        root=Path(root);old=root/'old';mapping=root/'mapping';reference=root/'reference'
        mapping.mkdir();reference.mkdir();bag=root/'input.bag';bag.write_bytes(b'synthetic bag')
        self.write_map(old,[[100,0,0,0,0,0,0,0,1],[101,0,0,0,0,0,0,1,1]],
                       [[[1,0,0,1],[1,1,0,1],[4,4,1,1]],[[1,0,0,1],[1,1,0,1]]])
        self.write_map(mapping/'map',[[200,0,0,0,0,0,0,0,1],[201.2,.2,0,0,0,0,0,1,1]],
                       [[[3,0,0,1]],[[2.8,0,0,1]]])
        settings=load_config()['scan_recording'];journal=ScanJournal(mapping/'environment/scans.sqlite',settings,{})
        rows=[];targets=[]
        for i in range(6):
            stamp=200+i*.6;x=i*.1;ns=round(stamp*1e9)
            row=dict(stamp_s=stamp,stamp_ns=ns,frame_id='odom',child_frame_id='rslidar',
                position=[x,0,0],quaternion_xyzw=[0,0,0,1],health=0,session_id='synthetic',generation=0)
            rows.append(row);targets.append(dict(row,has_map_pose=True,map_frame='map',map_position=[x,0,0],
                                                 map_quaternion_xyzw=[0,0,0,1]))
            points=np.array([[3-x,0,0]],np.float32);origins=np.array([[x,0,0]],np.float32)
            journal.append(points,pose_matrix(row['position'],row['quaternion_xyzw']),ns,
                dict(session_id='synthetic',generation=0,frame_id='odom',sensor_frame='rslidar',
                     rays_used=True,ray_origin_model='pose_interpolated_origins'),origins,np.array([0],np.float32))
        journal.close()
        for folder,data in ((mapping,rows),(reference,targets)):
            (folder/'localization.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
        digest=file_fingerprint(bag)
        def run(mode,map_path):
            return dict(success=True,source='bag',mode=mode,bag=str(bag),
                        algorithm_parameters=dict(mapping=dict(extrinsic_R=np.eye(3).reshape(-1).tolist(),
                            extrinsic_T=[0.,0.,0.],extrinsic_est_en=False),
                            common=dict(imu_time_offset=0.,time_sync_en=False)),
                        playback=dict(bag_fingerprint=digest,map_fingerprint=map_fingerprint(map_path)))
        a=run('mapping',mapping/'map');a['environment_scans_fingerprint']=file_fingerprint(mapping/'environment/scans.sqlite')
        (mapping/'run.json').write_text(json.dumps(a));(reference/'run.json').write_text(json.dumps(run('localization',old)))
        return old,mapping,reference

    def test_update_retains_unobserved_old_points_and_produces_a_new_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=self.fixture(root)
            before=map_fingerprint(old);report=update(old,mapping,reference,root/'candidate')
            self.assertEqual(map_fingerprint(old),before)
            self.assertEqual(report['status'],'candidate_not_accepted')
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

    def test_different_bag_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=self.fixture(root)
            target=reference/'run.json';value=json.loads(target.read_text())
            value['playback']['bag_fingerprint']={'wrong':True};target.write_text(json.dumps(value))
            with self.assertRaises(ValueError):update(old,mapping,reference,root/'candidate')
            self.assertFalse((root/'candidate').exists())
            with self.assertRaises(FileExistsError):update(old,mapping,reference,old)

    def test_registration_jump_and_overlapping_session_time_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old,mapping,reference=self.fixture(root)
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
            root=Path(directory);old,mapping,reference=self.fixture(root)
            path=reference/'run.json';run=json.loads(path.read_text())
            run['algorithm_parameters']['common']['imu_time_offset']=.1
            path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError,'different sensor extrinsics or IMU timing'):
                update(old,mapping,reference,root/'candidate')
            self.assertFalse((root/'candidate').exists())
