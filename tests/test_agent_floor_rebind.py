"""Manual task handoff across independently localized map contexts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import httpx

from decision.agent import CONFIG_DIR
from decision.agent.core.elevator_transition import ElevatorTransition
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store,Budget
from decision.agent.core.model import Model
from decision.agent.core.tasks import TaskGoals
from decision.agent.tools.places import Catalog
from tests.test_agent import ScriptedFeedback,point_goal,reply
from tests.test_agent_conversation import wire


class FloorRebindTests(unittest.TestCase):
    def test_registration_model_receives_both_floors_and_correct_mode(self):
        original=Catalog.load(CONFIG_DIR/'simulation.json')
        a=copy.deepcopy(original.value);a['floor_id']='floor2'
        b=copy.deepcopy(original.value);b['floor_id']='floor1'
        source,target=Catalog(a),Catalog(b)
        nav=ScriptedFeedback();nav.scope='manual'
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);store=Store(root/'session.sqlite');budget=Budget(root/'budget.sqlite')
            private=root/'private.json';private.write_text(json.dumps(dict(base_url='https://unit.invalid/v1',api_key='fixture')))
            requests=[]
            def handle(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200,text=wire(reply('lookup_places',name='',floor_id='floor1').message['tool_calls']))
            try:
                session=AgentSession('乘电梯到一楼',source,nav,store,floor_catalogs={'floor2':source,'floor1':target})
                result=Model(private,budget,store,transport=httpx.MockTransport(handle)).decide(session.prompt())
                self.assertEqual(result.action['arguments']['floor_id'],'floor1')
                self.assertIn('人工驾驶跨楼层任务',requests[0]['messages'][0]['content'])
                self.assertNotIn('只支持当前楼层的地图',requests[0]['messages'][0]['content'])
                self.assertIn('floor1',json.dumps(requests[0]['messages'],ensure_ascii=False))
            finally:budget.close();store.close()

    def test_ordered_requirements_are_validated_against_their_own_floor(self):
        original=Catalog.load(CONFIG_DIR/'simulation.json')
        a=copy.deepcopy(original.value);a['floor_id']='floor2'
        b=copy.deepcopy(original.value);b['floor_id']='floor1'
        b['places'].append(dict(place_id='floor1_only',name='一楼地点',pose=[1.,1.,0.],reference='base_link',status='simulation'))
        source,target=Catalog(a),Catalog(b)
        goals=TaskGoals(source,floor_catalogs={'floor2':source,'floor1':target})
        items=[dict(point_goal('elevator'),floor_id='floor2'),
               dict(goal_id='lift',kind='skill',target_id='',description='乘电梯到一楼',stage_type='elevator',
                    floor_id='floor2',to_floor='floor1'),
               dict(goal_id='after',kind='point',target_id='floor1_only',description='到一楼地点',floor_id='floor1')]
        self.assertEqual(len(goals.record(items)),3)
        wrong=TaskGoals(source,floor_catalogs={'floor2':source,'floor1':target})
        with self.assertRaisesRegex(ValueError,'current floor'):
            wrong.record([items[0],items[2]])
        single=TaskGoals(source)
        with self.assertRaisesRegex(ValueError,'destination floor'):
            single.record([items[1]])

    def test_task_and_requirements_survive_only_after_matching_destination_localization(self):
        original=Catalog.load(CONFIG_DIR/'simulation.json')
        source_value=copy.deepcopy(original.value);source_value.update(floor_id='floor2',localization_map_id='old-map')
        target_value=copy.deepcopy(original.value);target_value.update(floor_id='floor1',localization_map_id='new-map')
        source,target=Catalog(source_value),Catalog(target_value)
        nav=ScriptedFeedback();nav.scope='manual'
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                session=AgentSession('先乘电梯再到指定地点',source,nav,store,
                                     floor_catalogs={'floor2':source,'floor1':target})
                session.call('record_task',goals=[dict(goal_id='lift',kind='skill',target_id='',
                    description='乘电梯',stage_type='elevator',floor_id='floor2',to_floor='floor1'),
                    dict(point_goal('elevator'),floor_id='floor1')])
                transition=ElevatorTransition(task_id=session.state['task_id'],goal_id='lift',from_floor='floor2',
                                              to_floor='floor1',destination_map_id='new-map')
                for seq,judgment in enumerate(('not_completed','completed')):
                    transition.observe(camera_record=dict(source='ros_camera_independent_of_localization',
                        clock_domain='ros_time',frame_id='camera',stamp_s=10.+seq,sequence=seq),
                        image=b'\xff\xd8test\xff\xd9',judgment=judgment,evidence='可见场景')
                transition.clear_observations()
                association=dict(valid=True,map_id='new-map',map_frame='sim_odom',local_session_id='s2',
                                 local_generation=2,local_frame='odom',association_session='assoc',
                                 association_generation=3,measurement_stamp_s=12.)
                transition.accept_localization(association=association,now_s=12.,max_age_s=1.,clock_domain='ros_time')
                binding=dict(valid=True,map_id='new-map',map_frame='sim_odom',association_session='assoc',association_generation=3)
                state=dict(scope='manual',source='runtime_manual_observer',clock_domain='runtime_live',stamp_s=12.,
                    session_id='s2',generation=2,frame_id='odom',localization_status='valid',input_stale=False,
                    origin_pose=[1.,2.,0.],origin_frame_id='sim_odom',map_binding=binding)
                next_nav=MagicMock();next_nav.scope='manual';next_nav.handle=None
                next_nav.tick.return_value=next_nav.snapshot.return_value=state
                next_nav.value=dict(control=dict(agent_mapped_pose=dict(source='current_localization_and_scene_association',
                    association_stamp_s=12.,local_identity=['s2',2,'odom'],map_binding=binding)))
                bad=copy.deepcopy(transition);bad.association['map_id']='other'
                with self.assertRaises(ValueError):
                    session.rebind_manual_floor(target,next_nav,bad)
                task_id=session.state['task_id'];next_goal=copy.deepcopy(session.task_goals.goals[1]['requirement'])
                result=session.rebind_manual_floor(target,next_nav,transition)
                self.assertEqual(result['task_id'],task_id)
                self.assertEqual(session.task_goals.goals[0]['status'],'pending')
                self.assertEqual(session.task_goals.goals[0]['destination_localization']['association'],transition.association)
                self.assertEqual(session.task_goals.goals[1]['requirement'],next_goal)
                self.assertEqual(session.catalog.value['floor_id'],'floor1')
                self.assertIsNone(session.spatial_view)
                self.assertEqual(session.skill.context['local_identity'],['s2',2,'odom'])
            finally:store.close()
