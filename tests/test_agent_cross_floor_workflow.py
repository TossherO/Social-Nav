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
from decision.agent.core.elevator_monitor import ElevatorMonitor
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
    def start(self):pass
    def close(self):pass
    def snapshot(self):
        self.sequence+=1
        return dict(status='available',record=dict(source='ros_camera_independent_of_localization',clock_domain='ros_time',
            frame_id='camera',stamp_s=10.+self.sequence,sequence=self.sequence),image=b'\xff\xd8test\xff\xd9')


class CrossFloorWorkflowTests(unittest.TestCase):
    def test_single_task_advances_after_visual_ride_and_destination_rebind(self):
        base=Catalog.load(CONFIG_DIR/'simulation.json')
        a=copy.deepcopy(base.value);a.update(floor_id='floor2',localization_map_id='old-map')
        b=copy.deepcopy(base.value);b.update(floor_id='floor1',localization_map_id='new-map')
        source,target=Catalog(a),Catalog(b)
        nav1=Navigation(source,'old-map','s1',10.);nav2=Navigation(target,'new-map','s2',20.)
        monitors=[];emitted=[]
        class Handoff:
            def __init__(self,*,session,transition,target_catalog,**_):
                self.session,self.transition,self.catalog=session,transition,target_catalog
                self.run_dir=Path('/tmp/mock-observation')
            def tick(self):
                a=dict(nav2.binding,local_session_id='s2',local_generation=0,local_frame='odom',measurement_stamp_s=20.)
                self.transition.accept_localization(association=a,now_s=20.,max_age_s=1.,clock_domain='runtime_live')
                self.session.rebind_manual_floor(self.catalog,nav2,self.transition)
                return dict(status='complete')
            def close(self):pass
        class Actor:
            def __init__(self,session):
                self.session=session;self.model=MagicMock();self.exit_reason='synthetic';self.requests=0
            def run(self,*,on_tick,**_):
                self.session.call('record_task',goals=[
                    dict(goal_id='lift',kind='skill',target_id='',description='乘电梯到一楼',
                         stage_type='elevator',floor_id='floor2',to_floor='floor1'),
                    dict(goal_id='next',kind='point',target_id='elevator',description='到一楼电梯口',floor_id='floor1')])
                for _ in range(30):
                    on_tick(self)
                    if monitors and monitors[0].model.thread:monitors[0].model.thread.join(1.)
                    if self.session.catalog.value['floor_id']=='floor1':break
                    time.sleep(.005)
                self.assert_finished()
            def assert_finished(self):
                if self.session.catalog.value['floor_id']!='floor1':raise AssertionError('Handoff did not complete')
                if self.session.task_goals.goals[0]['status']!='completed':raise AssertionError('Elevator not completed')
                if self.session.task_goals.goals[1]['status']!='pending':raise AssertionError('Next goal was lost')
            def close(self):pass
        def monitor(*args,**kwargs):
            kwargs['interval_s']=.001
            result=ElevatorMonitor(*args,**kwargs);monitors.append(result);return result
        def decide(_config,_session,context,_history,_frame,**_):
            return dict(judgment='inside' if context['status']=='awaiting_entry' else 'outside',
                        evidence='模拟观察',retain_observation=True)
        fit=dict(from_floor='floor2',to_floor='floor1',to_map_id='new-map',matrix=np.eye(3).tolist())
        with tempfile.TemporaryDirectory() as root,\
             patch.object(manual.Catalog,'load',return_value=source),\
             patch.object(manual,'floor_catalogs_for_manual',return_value=({'floor2':source,'floor1':target},
                 {'floor1':Path('floor1.yaml'),'floor2':Path('floor2.yaml')})),\
             patch.object(manual,'ManualRuntimeClient'),patch.object(manual,'ManualNavigation',return_value=nav1),\
             patch.object(manual,'ManualReference'),patch.object(manual,'RouteDisplay'),\
             patch.object(manual,'make_loop',side_effect=lambda _a,s,_p:Actor(s)),\
             patch.object(manual,'shared_budget') as budget,patch.object(manual,'emit',side_effect=emitted.append),\
             patch('decision.agent.workflows.floor_handoff.observation_owner'),\
             patch('decision.agent.workflows.floor_handoff.FloorHandoff',Handoff),\
             patch('decision.agent.adapters.elevator_camera.ElevatorCamera',Camera),\
             patch('decision.agent.core.elevator_monitor.ElevatorMonitor',side_effect=monitor),\
             patch('decision.agent.core.elevator_model.decide_elevator',side_effect=decide),\
             patch('decision.agent.core.elevator_links.fit_floor_transform',return_value=fit):
            budget.return_value.summary.return_value={}
            args=SimpleNamespace(duration=10,interactive=True,catalog='unused',map_condition=None,cross_floor=True,
                actions=None,runtime_socket='/tmp/source/run/commands.sock',output=str(Path(root)/'run'),
                instruction='乘电梯到一楼',no_memory=False,gui_commands=False,api_config='unused',
                cache_mode='implicit',thinking_mode='off')
            result=manual.run(args)
            self.assertEqual(result['task']['task_goals'][0]['status'],'completed')
            self.assertTrue(any(x.get('event')=='elevator_stage_completed' for x in emitted))
            self.assertEqual(monitors[0].history,[])
