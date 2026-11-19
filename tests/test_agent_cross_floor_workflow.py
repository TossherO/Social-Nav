"""Cross-floor workflow composition with mock camera, runtime and model."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch

import numpy as np
from decision.agent import CONFIG_DIR
from decision.agent.tools.places import Catalog
from decision.agent.workflows import manual


class Navigation:
    scope='manual'
    def __init__(self,catalog,map_id,session,stamp):
        self.catalog=catalog;self.handle=None;self.status='idle';self.route=None
        self.geometry=SimpleNamespace(base_from_lidar=np.eye(4))
        self.binding=dict(valid=True,map_id=map_id,map_frame='sim_odom',association_session='assoc-'+session,association_generation=0)
        self.state=dict(scope='manual',source='runtime_manual_observer',clock_domain='runtime_live',stamp_s=stamp,
            session_id=session,generation=0,frame_id='odom',localization_status='valid',input_stale=False,
            origin_pose=[4.,4.,0.],origin_stamp_s=stamp,origin_frame_id='sim_odom',map_binding=self.binding,stopped=True)
        self.value=dict(control=dict(agent_mapped_pose=dict(source='current_localization_and_scene_association',
            association_stamp_s=stamp,local_identity=[session,0,'odom'],map_binding=self.binding)))
    def tick(self):return copy.deepcopy(self.state)
    def snapshot(self):return self.tick()
    def reference_grid(self):return object()


class Camera:
    def __init__(self):self.sequence=-1
    def start(self,extrinsic):pass
    def close(self):pass
    def snapshot(self):
        self.sequence+=1
        return dict(status='available',record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
            frame_id='camera',stamp_s=10.+self.sequence,sequence=self.sequence),image=b'\xff\xd8test\xff\xd9',
            scan_image=b'scan',base_xy_dm=[[10,20]],
            scan_record=dict(source='local_lidar_returns',frame_id='lidar',clock_domain='ros_time',stamp_s=10.+self.sequence))


class CrossFloorWorkflowTests(unittest.TestCase):
    def test_single_loop_reselects_after_failed_registration(self):
        from decision.agent.core.loop import AgentLoop
        from decision.agent.core.session import AgentSession
        from decision.agent.core.store import Store
        from decision.agent.tools.framework import Reply
        from decision.agent.workflows.floor_handoff import ElevatorRide

        base=Catalog.load(CONFIG_DIR/'simulation.json')
        floors={}
        for floor in ('floor1','floor2'):
            value=copy.deepcopy(base.value)
            value.update(floor_id=floor,localization_map_id=floor+'-map')
            for idx in range(1,5):
                value['places'].append(dict(value['places'][0],place_id='p-01-'+str(idx)))
            floors[floor]=Catalog(value)
        nav1=Navigation(floors['floor2'],'floor2-map','s1',10.)
        nav2=Navigation(floors['floor1'],'floor1-map','s2',100.)
        nav1.state.update(status='idle')
        nav2.state.update(status='idle')
        requirements=[
            dict(goal_id='lift',kind='skill',stage_type='elevator',target_id='',description='乘电梯',
                 floor_id='floor2',to_floor='floor1'),
            dict(goal_id='next',kind='point',target_id='elevator',description='前往目标',floor_id='floor1')]
        seeds=[];requests=[]
        class Handoff:
            def __init__(self,*,session,transition,target_catalog,seed,**kwargs):
                self.session,self.transition,self.target=session,transition,target_catalog
                self.run_dir=Path('/tmp/mock-observation')
                seeds.append(seed)
            def tick(self):
                if len(seeds)==1:return dict(status='failed',error='registration_failed')
                association=dict(nav2.binding,local_session_id='s2',local_generation=0,
                                 local_frame='odom',measurement_stamp_s=100.)
                self.transition.accept_localization(association=association,now_s=100.,max_age_s=1.,clock_domain='runtime_live')
                self.session.rebind_manual_floor(self.target,nav2,self.transition)
                return dict(status='complete')
            def close(self):pass

        def work(payload,observation,spatial):
            state=payload['current_state']
            requests.append(payload)
            if not state['requirements_registered']:
                return Reply.fixture(dict(tool='record_task',arguments=dict(goals=requirements)))
            if not state.get('ride'):
                if state['completion_evidence'].get('destination_association'):
                    return Reply.fixture(dict(tool='complete_subtask',arguments=dict(goal_id='lift',evidence='目标楼层关联成立，车辆已停稳。')))
                return Reply.fixture(dict(tool='pause_map_localization',arguments=dict(goal_id='lift')))
            phase=state['ride']['status']
            if phase=='awaiting_destination_localization':
                return Reply.fixture(dict(tool='restart_map_localization',arguments=dict(
                    pose=[4.+len(seeds),5.,1.],evidence='matched corner')))
            return Reply.fixture(dict(tool='record_elevator_observation',arguments=dict(
                judgment='inside' if phase=='awaiting_entry' else 'outside',evidence='ephemeral',retain_observation=True)))
        with tempfile.TemporaryDirectory() as root, patch('decision.agent.workflows.floor_handoff.os.kill'), \
             patch('decision.agent.workflows.floor_handoff.observation_owner',return_value=123), \
             patch('decision.agent.adapters.elevator_camera.ElevatorCamera',Camera), \
             patch('decision.agent.adapters.shadow.load_catalog_map'),patch('decision.agent.tools.spatial.render',return_value=b'crop'), \
             patch('decision.agent.workflows.floor_handoff.scan_alignment_preview',return_value=b'overlay'), \
             patch('decision.agent.core.session.robot_context',return_value={}):
            store=Store(Path(root)/'session.sqlite')
            session=AgentSession('下楼',floors['floor2'],nav1,store,floor_catalogs=floors)
            session.elevator=ElevatorRide({'floor1':'unused'},'source',root)
            actor=AgentLoop(session,work)
            try:
                with patch.object(session,'prepare_reference'),patch('decision.agent.workflows.floor_handoff.FloorHandoff',Handoff):
                    operator_reviewed=False
                    for _ in range(25):
                        if session.state.get('elevator_active') and not operator_reviewed:
                            actor.event('review')
                            self.assertEqual(session.state['status'],'waiting_monitor')
                            operator_reviewed=True
                        actor.tick()
                        if actor.model.thread:actor.model.thread.join(1)
                        if session.task_goals.goals and session.task_goals.goals[0]['status']=='completed':break
                self.assertEqual([s['base_pose'] for s in seeds],[[4.,5.,1.],[5.,5.,1.]])
                self.assertTrue(operator_reviewed)
                self.assertEqual(session.task_goals.goals[0]['status'],'completed')
                self.assertEqual(session.task_goals.goals[1]['status'],'pending')
                self.assertFalse(session.state['elevator_active'])
                self.assertIs(session.navigation,nav2)
                rides=[p for p in requests if p.get('ride_inputs')]
                self.assertTrue(all(p['reference_map'] is None for p in rides))
                roles=[[row[0] for row in p['ride_inputs']] for p in rides]
                self.assertIn('historical_elevator_rgb',roles[1])
                self.assertNotIn('destination_waiting_area',roles[0])
                self.assertEqual(roles[-1],['current_elevator_rgb','current_local_lidar','destination_waiting_area','failed_pose_scan_overlay'])
                self.assertEqual(rides[-1]['current_state']['ride']['last_attempt']['result'],'registration_failed')
                self.assertNotIn('ephemeral',str(store.db.execute('SELECT value FROM events').fetchall()))
            finally:
                actor.close();session.elevator.close();store.close()
