"""Moving-origin fusion and causal geometric poses, without a ROS master or GPU."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from perception.environment.layers import load_config, GridSnapshot, project_evidence
from perception.environment.poses import PoseBuffer, PoseTimeline
from perception.environment.scans import ScanJournal, read_scans, sealed_scan_summary
from perception.environment.fusion import fuse_scans
from perception.mapping_localization.__main__ import arguments, replay_topics, algorithm_command






def row(ns, *, health=0, session='local', generation=0):
    return dict(stamp_ns=ns, stamp_s=ns/1e9, position=[0,0,0], quaternion_xyzw=[0,0,0,1],
                session_id=session, generation=generation, frame_id='odom', child_frame_id='rslidar', health=health)


class StreamingTests(unittest.TestCase):
    def test_throttled_scans_skip_people_wait_but_kept_scans_and_resets_do_not(self):
        import threading,time
        from types import SimpleNamespace as NS
        from collections import OrderedDict,deque
        from perception.environment.ros.node import EnvironmentNode
        node=EnvironmentNode.__new__(EnvironmentNode);waits=[];processed=[]
        class Condition:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def wait(self,delay):
                waits.append(delay);node.people.append(dict(stamp_s=1.2))
        node.condition=Condition();node.engine_lock=threading.RLock()
        node.config=dict(update_hz=5,pose_wait_s=.2,dynamic=dict(wait_s=.12))
        node.rate_identity=('local',0);node.rate_anchor=1.;node.rate_tick=1
        node.engine=NS(last_stamp=1.);node.done=False;node.people_enabled=True
        node.people=deque();node.scan_durations={};node.counts=dict(rate_limited=0);node.queue_timings=[]
        node.clouds=OrderedDict((stamp,(NS(header=NS(stamp=NS(to_sec=lambda s=stamp:s/1e9))),time.monotonic()))
                               for stamp in (1100000000,1200000000))
        pose=NS(health=0,VALID=0,session_id='local',generation=0)
        node.poses={stamp:pose for stamp in node.clouds}
        node.associations={stamp:dict(local_session_id='local',local_generation=0) for stamp in node.clouds}
        def process(stamp,cloud,pose,duration,association,people):processed.append((stamp,people));node.done=True
        node.process=process;node.work()
        self.assertEqual(node.counts['rate_limited'],1);self.assertEqual(waits,[.01])
        self.assertEqual(processed,[(1200000000,dict(stamp_s=1.2))])
        self.assertFalse(node.rate_limited(1100000000,NS(health=2,VALID=0,session_id='local',generation=0)))
        self.assertFalse(node.rate_limited(1100000000,NS(health=0,VALID=0,session_id='new',generation=0)))
        node.engine.last_stamp=None;self.assertFalse(node.rate_limited(1100000000,pose))

    def test_phase_locked_sampling_does_not_accumulate_timestamp_jitter(self):
        from perception.environment.ros.node import sampling_slot
        random=np.random.default_rng(9)
        stamps=1792725575.+np.arange(600)*.1+random.uniform(-.0003,.0003,600)
        tick=0;chosen=[]
        for stamp in stamps:
            slot=sampling_slot(stamp,stamps[0],5)
            if slot>=tick:chosen.append(stamp);tick=slot+1
        self.assertLessEqual(abs(len(chosen)-300),1)
        self.assertTrue(np.all(np.diff(chosen)>0))
        self.assertEqual(sampling_slot(stamps[0]+10,stamps[0],5),50)

    def test_numpy_ros_grid_serialization_preserves_wire_bytes(self):
        import io
        try:
            from rospy.numpy_msg import numpy_msg
            from social_nav_environment.msg import EnvironmentGrid
        except ImportError:
            self.skipTest('System ROS and generated environment messages are required')
        layers=project_evidence(np.empty((0,5)),np.zeros((3,3),np.float32),
            np.ones((3,3),bool),[0,0],load_config(),1.)
        plain=EnvironmentGrid();vectorized=numpy_msg(EnvironmentGrid)()
        for message in (plain,vectorized):
            message.info.height,message.info.width=3,3
        from perception.environment.scene import assign_grid_layers
        assign_grid_layers(plain,layers);assign_grid_layers(vectorized,layers)
        before,after=io.BytesIO(),io.BytesIO()
        plain.serialize(before);vectorized.serialize(after)
        self.assertEqual(before.getvalue(),after.getvalue())

    def test_world_voxel_band_alignment_and_epoch_observation_time(self):
        config=load_config(); config.update(resolution=.15)
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
        timeline=PoseTimeline([row(10**9),row(1100000000)],config['poses'],[[1,0,0,0,0,0,0,0,0]])
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
                               config['poses'],[[1,0,0,0,0,0,0,0,0]])
            with self.assertRaises(ValueError):fuse_scans(candidate,path,wrong,config,{'sha256':'test'})

    def test_live_pose_history_is_causal_breaks_on_failure_and_retires_old_sessions(self):
        poses=PoseBuffer(load_config()['poses'])
        poses.add(row(10**9));poses.add(row(1100000000,health=2));poses.add(row(1200000000))
        self.assertEqual(poses.at(1050000000)['segment'],1)
        self.assertIsNone(poses.at(1150000000))
        self.assertEqual(poses.at(1250000000)['segment'],2)
        self.assertIsNone(poses.at(2*10**9))
        poses.add(row(2*10**9,session='new'))
        self.assertFalse(poses.add(row(2100000000)))
        self.assertIsNone(poses.at(1250000000))



    def test_scene_rgb_transport_is_independent_of_removed_visual_indexing(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            bag=Path(directory)/'example.bag';bag.touch()
            output=Path(directory)/'run'
            base=['--mode','mapping','--bag',str(bag),'--out',str(output),'--environment']
            args=arguments(base+['--record-environment-scans'])
            self.assertNotIn('/camera/color/image_raw',replay_topics(args,{},environment_config=load_config()))
            self.assertIn('record_environment_scans:=true',algorithm_command(args,output,output/'map',True))
            configured=arguments(base+['--environment-backend','cuda','--environment-config',
                str(Path('perception/environment/configs/environment.yaml').resolve())])
            command=algorithm_command(configured,output,output/'map',True)
            self.assertIn('environment_backend:=cuda',command)
            self.assertTrue(any(value.startswith('environment_config:=') for value in command))
            scene=arguments(base+['--scene'])
            topics=replay_topics(scene,{},environment_config=load_config())
            self.assertIn('/camera/color/image_raw',topics)
            self.assertNotIn('/camera/color/camera_info',topics)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                arguments(base+['--visual-memory'])


if __name__=='__main__':unittest.main()
