"""Mocked observation-only floor handoff; never starts a real runtime."""
import tempfile
import json
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch

import numpy as np
from decision.agent.core.elevator_transition import ElevatorTransition
from decision.agent.workflows.floor_handoff import FloorHandoff,ElevatorRide,follow_runtime_options


class FloorHandoffTests(unittest.TestCase):
    def test_real_handoff_accepts_standard_seed_and_rejects_mismatched_inputs(self):
        from decision.agent.core.elevator_links import standard_destination_seed
        transition=ElevatorTransition(task_id='t',goal_id='lift',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        transition.observe(camera_record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
            frame_id='camera',stamp_s=10.,sequence=1),image=b'\xff\xd8test\xff\xd9',judgment='completed',evidence='已到标准点')
        transition.clear_observations()
        standard=dict(map_id='m',frame_id='map',reference='rslidar',pose_rpy_xyz=[.01,-.02,.3,1.,2.,-.14])
        scan=dict(source='local_lidar_returns',clock_domain='ros_time',frame_id='rslidar',stamp_s=11.)
        seed=standard_destination_seed(transition,standard,scan)
        session=MagicMock();session.skill.record={'handle':None};session.state={'task_id':'t'}
        session.catalog=SimpleNamespace(value={'floor_id':'floor2'})
        session.navigation.value={'local_skills':{}}
        session.task_goals.current.return_value=dict(requirement=dict(stage_type='elevator',floor_id='floor2',to_floor='floor1'))
        target=SimpleNamespace(value={'floor_id':'floor1','localization_map_id':'m'})
        with tempfile.TemporaryDirectory() as root,patch('decision.agent.workflows.floor_handoff.observation_owner',return_value=None):
            def make(value):return FloorHandoff(session=session,transition=transition,target_catalog=target,
                config_path='x.yaml',seed=value,output=root,source_socket='source')
            handoff=make(seed)
            self.assertEqual(json.loads(handoff.seed_path.read_text()),seed)
            self.assertEqual(handoff.status,'stopping_source')
            process=MagicMock();process.poll.return_value=None
            with patch.object(handoff,'_source_stopped',return_value=True), \
                    patch('decision.agent.workflows.floor_handoff.subprocess.Popen',return_value=process) as start:
                self.assertEqual(handoff.tick()['status'],'starting_destination')
                command=start.call_args.args[0]
                self.assertIn('--handoff-seed',command)
                self.assertIn(str(handoff.seed_path),command)
                handoff.close()
            for change in (dict(source='unrecognized'),dict(task_id='other'),dict(scan_stamp_s=10.),
                           dict(destination=dict(from_floor='floor2',to_floor='floor1',to_map_id='other')),
                           dict(lidar_pose_rpy_xyz=[0.]),dict(lidar_xy_yaw=[9.,2.,.3]),dict(scan_frame='other')):
                invalid=copy.deepcopy(seed);invalid.update(change)
                with self.assertRaises(ValueError):make(invalid)

    def test_standard_reference_and_at_most_two_retained_images_fit_model_limit(self):
        ride=ElevatorRide({},'unused','unused')
        ride.transition=SimpleNamespace(status='monitoring_ride')
        ride.standard_reference=({'rgb':{'stamp_s':1},'lidar':{'stamp_s':1}},b'jpeg-reference',b'png-reference')
        ride.retained=[dict(record={'sequence':i},image=b'history') for i in range(4)]
        ride.frame=dict(record={'sequence':5},image=b'current',scan_record={'sequence':5},scan_image=b'current-scan')
        items=ride.attachments()
        self.assertEqual(len(items),6)
        self.assertEqual([x[0] for x in items],['historical_elevator_rgb','historical_elevator_rgb',
            'current_elevator_rgb','current_local_lidar','standard_start_reference_rgb','standard_start_reference_lidar'])
        self.assertEqual([x[1]['sequence'] for x in items[:2]],[2,3])

    def test_standard_failure_opens_picker_and_picker_start_failure_allows_operator_retry(self):
        from decision.agent.core.elevator_transition import ElevatorTransition
        transition=ElevatorTransition(task_id='t',goal_id='lift',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        transition.observe(camera_record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
            frame_id='camera',stamp_s=10.,sequence=1),image=b'\xff\xd8test\xff\xd9',judgment='completed',evidence='已到标准点')
        transition.clear_observations()
        standard=dict(map_id='m',frame_id='map',reference='rslidar',pose_rpy_xyz=[.01,-.02,.3,1.,2.,-.14])
        scan=dict(source='local_lidar_returns',clock_domain='ros_time',frame_id='rslidar',stamp_s=11.)
        sample=dict(status='available',record=dict(stamp_s=11.1),scan_record=scan)
        session=MagicMock();session.state={'status':'waiting_monitor'}
        session.task_goals.floor_catalogs={'floor1':SimpleNamespace(value={'localization_map_id':'m'})}
        ride=ElevatorRide({'floor1':'unused'},'source','/tmp');ride.transition=transition;ride.standard=standard
        ride.camera=MagicMock();ride.camera.snapshot.return_value=sample
        with patch('decision.agent.workflows.floor_handoff.FloorHandoff') as factory:
            factory.return_value.tick.return_value=dict(status='failed',error='registration_failed')
            ride.poll(session)
            seed=factory.call_args.kwargs['seed']
            self.assertEqual(seed['lidar_pose_rpy_xyz'],standard['pose_rpy_xyz'])
            self.assertEqual(seed['scan_stamp_s'],11.)
            self.assertEqual(seed['source'],'standard_destination_start')
            self.assertIsNone(ride.poll(session))
            self.assertEqual(ride.wait_reason,'registration_failed')
            self.assertIsNone(factory.call_args.kwargs['seed'])
            factory.return_value.tick.return_value=dict(status='failed',error='destination_observation_exited',
                                                       manual_pose_confirmed=False)
            self.assertIsNone(ride.poll(session))
            self.assertIsNone(ride.poll(session))
            self.assertEqual(factory.call_count,2)
            self.assertIsNone(ride.handoff)
            ride.reopen_manual_selection(session)
            self.assertEqual(factory.call_count,3)
            self.assertIsNone(factory.call_args.kwargs['seed'])
            self.assertEqual(session.state['status'],'waiting_monitor')

    def test_floor_change_preserves_explicit_t2_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'run').mkdir();socket=root/'run/commands.sock'
            (root/'command.json').write_text(json.dumps(['python','-m','runtime','--t2-backend','dspark',
                '--t2-jpeg-quality','80','--t2-raw-rgb']))
            self.assertEqual(follow_runtime_options(socket),['--t2-backend','dspark','--t2-jpeg-quality','80','--t2-raw-rgb'])
            (root/'command.json').write_text(json.dumps(['--t2-backend','dspark','--t2-backend','hosted']))
            with self.assertRaises(ValueError):follow_runtime_options(socket)
            (root/'command.json').write_text(json.dumps(['--agent-follow-candidates']))
            with self.assertRaisesRegex(ValueError,'backend was not recorded'):follow_runtime_options(socket)

    def test_ride_starts_without_stopped_or_valid_source_localization(self):
        session=MagicMock()
        session.navigation.scope='manual'
        session.navigation.snapshot.return_value=dict(stopped=False,input_stale=True,localization_status='invalid')
        session.navigation.geometry.base_from_lidar=np.eye(4)
        session.skill.record={'handle':None}
        session.state={'task_id':'t'}
        session.task_goals.current.return_value=dict(requirement=dict(stage_type='elevator',floor_id='floor2',to_floor='floor1'))
        target=SimpleNamespace(value={'localization_map_id':'m'},places={
            'p-01-'+str(i):{'pose':[float(i),0.]} for i in range(1,5)})
        session.task_goals.floor_catalogs={'floor1':target}
        with tempfile.TemporaryDirectory() as root, \
             patch('decision.agent.workflows.floor_handoff.load_standard_start',return_value=(None,None)), \
             patch('decision.agent.workflows.floor_handoff.observation_owner',return_value=123), \
             patch('decision.agent.workflows.floor_handoff.os.kill') as stop, \
             patch('decision.agent.adapters.elevator_camera.ElevatorCamera') as camera, \
             patch('decision.agent.adapters.shadow.load_catalog_map'), \
             patch('decision.agent.tools.spatial.render',return_value=b'crop'):
            ride=ElevatorRide({'floor1':'unused'},'source',root)
            result=ride.start(session,'lift')
            self.assertEqual(result['status'],'accepted')
            self.assertEqual(ride.transition.status,'monitoring_ride')
            camera.return_value.start.assert_called_once()
            stop.assert_called_once()
            ride.close()

    def test_ride_refuses_active_navigation_or_nonmanual_scope(self):
        session=MagicMock();session.navigation.scope='manual';session.skill.record={'handle':{'goal_id':'g'}}
        session.task_goals.current.return_value=dict(requirement=dict(stage_type='elevator'))
        ride=ElevatorRide({},'unused','unused')
        with self.assertRaises(ValueError):ride.start(session,'lift')
        session.skill.record={'handle':None};session.navigation.scope='live'
        with self.assertRaises(ValueError):ride.start(session,'lift')

    def test_owned_observer_is_replaced_and_task_rebound_after_localization(self):
        transition=ElevatorTransition(task_id='t',goal_id='lift',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        for i,judgment in enumerate(('not_completed','completed')):
            transition.observe(camera_record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
                frame_id='camera',stamp_s=10.+i,sequence=i),image=b'\xff\xd8test\xff\xd9',
                judgment=judgment,evidence='可见现场')
        transition.clear_observations()
        session=MagicMock();session.skill.record={'handle':None};session.state={'task_id':'t'}
        session.navigation.value={'local_skills':{'follow':True}}
        session.catalog=SimpleNamespace(value={'floor_id':'floor2'})
        session.task_goals.current.return_value=dict(requirement=dict(stage_type='elevator',floor_id='floor2',to_floor='floor1'))
        catalog=SimpleNamespace(value={'floor_id':'floor1','localization_map_id':'m'})
        seed=dict(verified_localization=False,floor_id='floor1',task_id='t',goal_id='lift',
                  source='agent_destination_scan_and_map',exit_stamp_s=11.,scan_stamp_s=11.5,
                  destination=dict(from_floor='floor2',to_floor='floor1',to_map_id='m'))
        process=MagicMock();process.poll.return_value=None
        binding=dict(valid=True,map_id='m',map_frame='map',association_session='assoc',association_generation=0)
        state=dict(input_stale=False,origin_pose=[1.,2.,0.],localization_status='valid',map_binding=binding,
                   session_id='s',generation=0,frame_id='odom',stamp_s=12.,clock_domain='runtime_live')
        navigation=MagicMock();navigation.tick.return_value=state
        navigation.value=dict(control=dict(agent_mapped_pose=dict(source='current_localization_and_scene_association',association_stamp_s=12.)))
        with tempfile.TemporaryDirectory() as root,\
             patch('decision.agent.workflows.floor_handoff.observation_owner',return_value=123),\
             patch('decision.agent.workflows.floor_handoff.os.kill') as stop,\
             patch('decision.agent.workflows.floor_handoff.subprocess.Popen',return_value=process) as start,\
             patch('decision.agent.workflows.floor_handoff.ManualRuntimeClient'),\
             patch('decision.agent.workflows.floor_handoff.ManualNavigation',return_value=navigation),\
             patch.object(FloorHandoff,'_source_stopped',return_value=True):
            source=Path(root)/'source';(source/'run').mkdir(parents=True)
            (source/'command.json').write_text(json.dumps(['--t2-backend','dspark','--t2-jpeg-quality','80','--t2-raw-rgb']))
            handoff=FloorHandoff(session=session,transition=transition,target_catalog=catalog,config_path='x.yaml',
                seed=seed,output=root,source_socket=source/'run/commands.sock')
            self.assertEqual(stop.call_count,1)
            self.assertEqual(handoff.tick()['status'],'starting_destination')
            self.assertEqual(start.call_count,1)
            command=start.call_args.args[0]
            self.assertEqual(command[command.index('--t2-backend')+1],'dspark')
            self.assertEqual(command[command.index('--t2-jpeg-quality')+1],'80')
            self.assertIn('--t2-raw-rgb',command)
            socket=Path(handoff.run_dir)/'run/commands.sock';socket.parent.mkdir(parents=True);socket.touch()
            self.assertEqual(handoff.tick()['status'],'complete')
            session.rebind_manual_floor.assert_called_once()
            target,adapter,accepted=session.rebind_manual_floor.call_args.args
            self.assertEqual((target,adapter),(catalog,navigation))
            self.assertEqual(accepted.snapshot(),transition.snapshot())
            self.assertEqual(transition.status,'completed')
            handoff.close()
