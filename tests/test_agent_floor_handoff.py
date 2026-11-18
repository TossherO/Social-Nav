"""Mocked observation-only floor handoff; never starts a real runtime."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch

from decision.agent.core.elevator_transition import ElevatorTransition
from decision.agent.workflows.floor_handoff import FloorHandoff


class FloorHandoffTests(unittest.TestCase):
    def test_owned_observer_is_replaced_and_task_rebound_after_localization(self):
        transition=ElevatorTransition(task_id='t',goal_id='lift',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        for i,judgment in enumerate(('inside','outside')):
            transition.observe(camera_record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
                frame_id='camera',stamp_s=10.+i,sequence=i),image=b'\xff\xd8test\xff\xd9',
                judgment=judgment,evidence='可见现场')
        transition.clear_observations()
        session=MagicMock();session.skill.record={'handle':None};session.state={'task_id':'t'}
        session.catalog=SimpleNamespace(value={'floor_id':'floor2'})
        session.task_goals.current.return_value=dict(requirement=dict(stage_type='elevator',floor_id='floor2',to_floor='floor1'))
        catalog=SimpleNamespace(value={'floor_id':'floor1','localization_map_id':'m'})
        seed=dict(verified_localization=False,floor_id='floor1',floor_transform=dict(from_floor='floor2',to_floor='floor1',to_map_id='m'))
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
            handoff=FloorHandoff(session=session,transition=transition,target_catalog=catalog,config_path='x.yaml',
                seed=seed,output=root,source_socket='unused')
            self.assertEqual(stop.call_count,1)
            self.assertEqual(handoff.tick()['status'],'starting_destination')
            self.assertEqual(start.call_count,1)
            socket=Path(handoff.run_dir)/'run/commands.sock';socket.parent.mkdir(parents=True);socket.touch()
            self.assertEqual(handoff.tick()['status'],'complete')
            session.rebind_manual_floor.assert_called_once_with(catalog,navigation,transition)
            self.assertEqual(transition.status,'completed')
            handoff.close()
