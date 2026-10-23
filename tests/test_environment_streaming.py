"""Moving-origin fusion and live visual memory, without a ROS master or GPU."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from perception.environment.layers import load_config, GridSnapshot, project_evidence
from perception.environment.live_memory import PoseBuffer, VisualWriter, query_live_memory
from perception.environment.memory import PoseTimeline
from perception.environment.scans import ScanJournal, read_scans, sealed_scan_summary
from perception.environment.fusion import fuse_scans
from perception.environment.storage import MemoryReader
from perception.mapping_localization.__main__ import arguments, replay_topics, algorithm_command


class Stamp:
    def __init__(self, ns): self.ns = ns
    def to_nsec(self): return self.ns
    def to_sec(self): return self.ns/1e9


def image(ns):
    pixels = np.array([[[120,30,10],[10,150,60]],[[255,255,255],[0,0,0]]], np.uint8)
    return SimpleNamespace(header=SimpleNamespace(stamp=Stamp(ns), frame_id='camera_optical'),
        encoding='rgb8', width=2, height=2, step=6, data=pixels.tobytes())


def row(ns, *, health=0, session='local', generation=0):
    return dict(stamp_ns=ns, stamp_s=ns/1e9, position=[0,0,0], quaternion_xyzw=[0,0,0,1],
                session_id=session, generation=generation, frame_id='odom', child_frame_id='rslidar', health=health)


class StreamingTests(unittest.TestCase):
    def test_world_voxel_band_alignment_and_epoch_observation_time(self):
        config=load_config()
        floor=np.zeros((3,3));support=np.ones((3,3),bool)
        # [0.12, 1.8] contains these 11 world-grid centres, not ceil(1.68/.15)=12.
        z=.225+np.arange(11)*.15
        voxels=np.column_stack([np.full(11,.225),np.full(11,.225),z,np.full(11,.2),np.full(11,.125)]).astype(np.float32)
        stamp=1792488000.12345
        complete=project_evidence(voxels,floor,support,[0,0],config,stamp)
        self.assertEqual(complete['occupancy'][1,1],0)
        self.assertEqual(complete['coverage'][1,1],1.)
        self.assertAlmostEqual(complete['last_observed'][1,1],stamp-.125,places=6)
        missing=project_evidence(voxels[:-1],floor,support,[0,0],config,stamp)
        self.assertEqual(missing['occupancy'][1,1],-1)
        self.assertLess(missing['coverage'][1,1],1.)

    def test_free_column_needs_all_heights_and_cannot_erase_a_low_hit(self):
        config=load_config(); config.update(resolution=.15, min_obstacle_height_m=.15,max_height_m=.6)
        floor=np.zeros((3,3)); support=np.ones((3,3),bool)
        voxels=np.array([[.225,.225,.225,.2,0],[.225,.225,.375,.2,0],[.225,.225,.525,.2,0]])
        empty=project_evidence(voxels[:1],floor,support,[0,0],config,1)
        self.assertEqual(empty['occupancy'][1,1],-1)
        free=project_evidence(voxels,floor,support,[0,0],config,1)
        self.assertEqual(free['occupancy'][1,1],0)
        occupied=project_evidence(np.r_[voxels,[[.225,.225,.225,.8,0]]],floor,support,[0,0],config,1)
        self.assertEqual(occupied['occupancy'][1,1],100)

    def test_journal_keeps_exact_origins_and_refuses_unsealed_or_over_budget_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            settings=dict(max_records=1,max_bytes=100000)
            writer=ScanJournal(path,settings,dict(source='test'))
            meta=dict(session_id='local',generation=0,frame_id='odom',rays_used=True,ray_origin_model='pose_interpolated_origins')
            points=np.array([[1,0,.5],[2,0,1]],np.float32)
            origins=np.array([[0,0,0],[.1,0,0]],np.float32)
            writer.append(points,np.eye(4),10**9,meta,origins,[-.5,99.5])
            with self.assertRaises(ValueError):list(read_scans(path))
            with self.assertRaises(ValueError):writer.append(points,np.eye(4),2*10**9,meta,origins,[0,100])
            writer.close()
            self.assertEqual(sealed_scan_summary(path)['records'],1)
            record,data=next(read_scans(path))
            self.assertEqual(record['stamp_ns'],10**9)
            np.testing.assert_array_equal(data['origins'],origins)
            np.testing.assert_array_equal(data['offset_ms'],[-.5,99.5])

    def test_final_fusion_uses_the_same_recorded_pose_and_preserves_unknown(self):
        config=load_config(); config.update(resolution=.15)
        floor=np.zeros((40,40),np.float32)
        empty=project_evidence(np.empty((0,5)),floor,np.ones_like(floor,bool),[-3,-3],config,1)
        empty['endpoint_count']=np.zeros_like(floor,np.int32)
        metadata=dict(map_id='map',snapshot_id='prior',origin_xy=[-3,-3],shape=[40,40],frame_id='map')
        candidate=GridSnapshot(metadata,empty)
        timeline=PoseTimeline([row(10**9),row(1100000000)],config['memory'],[[1,0,0,0,0,0,0,0,0]])
        points=np.array([[1,0,.5],[2,0,1.5]],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scans.sqlite'
            writer=ScanJournal(path,config['scan_recording'],{})
            meta=dict(session_id='local',generation=0,frame_id='odom',rays_used=True,ray_origin_model='pose_interpolated_origins')
            writer.append(points,np.eye(4),1100000000,meta,np.zeros_like(points),[0,100]);writer.close()
            result=fuse_scans(candidate,path,timeline,config,{'sha256':'test'})
            self.assertEqual(result.metadata['ray_scans'],1)
            self.assertTrue(result.metadata['free_space_supported'])
            self.assertTrue((result.layers['occupancy']==-1).any())
            wrong=PoseTimeline([dict(row(10**9),position=[1,0,0]),dict(row(1100000000),position=[1,0,0])],
                               config['memory'],[[1,0,0,0,0,0,0,0,0]])
            with self.assertRaises(ValueError):fuse_scans(candidate,path,wrong,config,{'sha256':'test'})

    def test_live_pose_history_is_causal_breaks_on_failure_and_retires_old_sessions(self):
        poses=PoseBuffer(load_config()['memory'])
        poses.add(row(10**9));poses.add(row(1100000000,health=2));poses.add(row(1200000000))
        self.assertEqual(poses.at(1050000000)['segment'],1)
        self.assertIsNone(poses.at(1150000000))
        self.assertEqual(poses.at(1250000000)['segment'],2)
        self.assertIsNone(poses.at(2*10**9))
        poses.add(row(2*10**9,session='new'))
        self.assertFalse(poses.add(row(2100000000)))
        self.assertIsNone(poses.at(1250000000))

    def test_live_png_roundtrip_and_queries_bind_memory_identity_time_and_local_session(self):
        settings=load_config()['memory']
        binding=dict(memory_id='memory',source_id='live',bag=None,rgb_topic='/rgb',calibration_sha256=None,time_scope='causal_observation')
        with tempfile.TemporaryDirectory() as directory:
            poses=PoseBuffer(settings);poses.add(row(10**9))
            writer=VisualWriter(Path(directory)/'observations.sqlite',binding,settings)
            first=writer.append(image(1050000000),poses.at(1050000000))
            poses.add(row(2*10**9,generation=1))
            second=writer.append(image(2050000000),poses.at(2050000000))
            writer.close()
            reader=MemoryReader(directory)
            try:
                request=dict(operation='query_observations',memory_id='memory',session_id='local',generation=0,frame='odom',as_of=3.)
                self.assertEqual([r['id'] for r in reader.query(request)['observations']],[first['id']])
                self.assertEqual(reader.query(dict(request,as_of=.5))['observations'],[])
                with self.assertRaises(ValueError):reader.query(dict(request,memory_id='other'))
                np.testing.assert_array_equal(reader.read_image(first['id']),np.frombuffer(image(1).data,np.uint8).reshape(2,2,3))
                self.assertEqual(reader.db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0],0)
                self.assertIsNone(second['receipt_ns'])
            finally:reader.close()

    def test_bag_replay_memory_only_references_the_source_image(self):
        settings=load_config()['memory']
        binding=dict(memory_id='memory',source_id='live',bag={'path':'test'},rgb_topic='/rgb',calibration_sha256=None,time_scope='causal_observation')
        with tempfile.TemporaryDirectory() as directory:
            poses=PoseBuffer(settings);poses.add(row(10**9))
            writer=VisualWriter(Path(directory)/'observations.sqlite',binding,settings)
            saved=writer.append(image(1050000000),poses.at(1050000000))
            self.assertEqual(saved['image_storage'],'bag_reference')
            self.assertEqual(writer.db.execute('SELECT COUNT(*) FROM image_assets').fetchone()[0],0)
            writer.close()

    def test_environment_flags_and_rgb_transport_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            bag=Path(directory)/'example.bag';bag.touch()
            output=Path(directory)/'run'
            args=arguments(['--mode','mapping','--bag',str(bag),'--out',str(output),
                            '--environment','--record-environment-scans','--visual-memory'])
            topics=replay_topics(args,{},environment_config=load_config())
            self.assertIn('/camera/color/image_raw',topics)
            self.assertIn('/camera/color/camera_info',topics)
            command=algorithm_command(args,output,output/'map',True)
            self.assertIn('record_environment_scans:=true',command)
            self.assertIn('visual_memory:=true',command)
            scene=arguments(['--mode','odometry','--bag',str(bag),'--out',str(output),
                             '--environment','--scene'])
            topics=replay_topics(scene,{},environment_config=load_config())
            self.assertIn('/camera/color/image_raw',topics)
            self.assertNotIn('/camera/color/camera_info',topics)


if __name__=='__main__':unittest.main()
