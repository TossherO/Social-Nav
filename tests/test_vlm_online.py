"""In-memory T1/T3 interfaces with mocked transport, never a robot or service."""
import copy
import unittest

import jsonschema
import numpy as np

from decision.vlm.t1.online import GroupsDecision
from decision.vlm.t3.online import SeatingDecision
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from navigation.planning.people import PeopleModel, social_groups
from navigation.tasks import TaskPolicy
from perception.environment.layers import load_config as environment_config
from perception.environment.scene import assemble_scene
from tests.decision_fixtures import candidate_set, dspark_client, observation, two_person_window


class OnlineVlmTests(unittest.TestCase):
    def setUp(self):
        self.planning=load_config()
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')

    def test_social_window_bound_to_current_identities_and_expires(self):
        scene,window,images=two_person_window()
        policy=TaskPolicy('mission','social',self.planning,self.geometry)
        online=GroupsDecision(policy)
        prepared=online.social(scene,window,images,now=11.1)
        factory,calls=dspark_client('{"walking":[[0,1]]}')
        answer=online.infer(prepared,client_factory=factory)
        self.assertEqual(answer['status'],'complete')
        result=online.accept(prepared,answer,scene=scene,now=11.1)
        self.assertEqual(result['operation'],'social_context')
        self.assertEqual(result['social']['groups'],[[['tracker_session',7,'a'],['tracker_session',7,'b']]])
        groups,status=social_groups(dict(social=result['social']),scene,PeopleModel(scene,self.planning['people']))
        self.assertEqual(status['status'],'valid');self.assertEqual(len(groups),1)
        _,_,_,later=observation(13.2)
        groups,status=social_groups(dict(social=result['social']),later,PeopleModel(later,self.planning['people']))
        self.assertEqual(groups,[]);self.assertEqual(status['status'],'expired')
        self.assertEqual(sum(call.method=='POST' for call in calls),1)
        with self.assertRaises(ValueError):online.accept(prepared,answer,scene=scene,now=11.1)

    def test_social_rejects_changed_images_and_unready_service(self):
        scene,window,images=two_person_window()
        online=GroupsDecision(TaskPolicy('mission','social',self.planning,self.geometry))
        with self.assertRaisesRegex(ValueError,'image differs'):
            online.social(scene,window,(images[0],images[1],images[2]+b'x'),now=11.1)
        prepared=online.social(scene,window,images,now=11.1)
        factory,calls=dspark_client('{}',ready=False)
        answer=online.infer(prepared,client_factory=factory)
        self.assertEqual(answer['reason'],'remote_dspark_unavailable')
        self.assertFalse(any(call.method=='POST' for call in calls))
        with self.assertRaisesRegex(ValueError,'already attempted'):
            online.infer(prepared,client_factory=factory)
        with self.assertRaises(ValueError):online.accept(prepared,answer,scene=scene,now=11.1)
        self.assertEqual(online.discard(prepared,answer)['reason'],'remote_dspark_unavailable')
        self.assertIsNone(online.gate.pending)
        online.cancel()

    def test_social_invalid_json_is_retained_and_never_applied(self):
        scene,window,images=two_person_window()
        for raw in ('{"walking":[[0,0]]}', '{"walking":[[97,98]],"walking":[[0,1]]}'):
            with self.subTest(raw=raw):
                online=GroupsDecision(TaskPolicy('mission','social',self.planning,self.geometry))
                prepared=online.social(scene,window,images,now=11.1)
                factory,_=dspark_client(raw)
                answer=online.infer(prepared,client_factory=factory)
                with self.assertRaises((ValueError,jsonschema.ValidationError)):
                    online.accept(prepared,answer,scene=scene,now=11.1)
                self.assertEqual(online.gate.last_response['raw_response'],raw)
                self.assertIsNone(online.policy.groups)

    def test_interrupted_or_invalid_completion_never_applies(self):
        for options in (dict(complete=False),dict(model='wrong'),dict(finish='length'),dict(reasoning=True)):
            with self.subTest(options=options):
                scene,window,images=two_person_window()
                online=GroupsDecision(TaskPolicy('mission','social',self.planning,self.geometry))
                prepared=online.social(scene,window,images,now=11.1)
                factory,calls=dspark_client('{}',**options)
                answer=online.infer(prepared,client_factory=factory)
                self.assertEqual(answer['status'],'unavailable')
                self.assertEqual(answer['raw_content'],'{}')
                self.assertEqual(sum(call.method=='POST' for call in calls),1)
                self.assertFalse(online.discard(prepared,answer)['accepted'])

    def test_seating_selection_remains_a_geometry_unverified_hypothesis(self):
        from decision.agent.tools.observations import Observation
        row,arrays,_,scene=observation(10.)
        # The online scene normally carries RGB separately in the Agent observation.
        visual=Observation.from_scene(scene,rgb=dict(pixels=arrays['rgb'],stamp_s=9.98,frame_id='camera'))
        candidates=candidate_set(scene)
        online=SeatingDecision(TaskPolicy('seat','seating',self.planning,self.geometry))
        prepared=online.seating(scene,candidates,now=10.,observation=visual)
        self.assertEqual(prepared['rgb_observation_binding'],visual.record['input_fingerprint'])
        factory,_=dspark_client('{"p":0,"s":"v"}')
        answer=online.infer(prepared,client_factory=factory)
        result=online.accept(prepared,answer,scene=scene,now=10.)
        self.assertEqual(result['operation'],'plan')
        self.assertTrue(result['requires_geometry_verification'])
        self.assertFalse(result['executable']);self.assertFalse(result['task_completed'])
        self.assertEqual(result['request']['candidate_id'],0)

    def test_seating_rejects_history_and_late_or_modified_results(self):
        row,arrays,_,base=observation(10.)
        scene=assemble_scene(base.geometry,clock_domain='synthetic',people=row,
            rgb=dict(pixels=arrays['rgb'],stamp_s=9.98,frame_id='camera',reference=dict(source='test')),
            settings=environment_config()['scene'])
        candidates=candidate_set(scene)
        online=SeatingDecision(TaskPolicy('seat','seating',self.planning,self.geometry))
        with self.assertRaisesRegex(ValueError,'historical context'):
            online.seating(scene,dict(candidates,context={}),now=10.)
        prepared=online.seating(scene,candidates,now=10.)
        changed=copy.deepcopy(prepared);changed['body']['temperature']=.5
        with self.assertRaisesRegex(ValueError,'request changed'):
            online.infer(changed)
        factory,_=dspark_client('{"p":0,"s":"v"}')
        response=online.infer(prepared,client_factory=factory)
        with self.assertRaisesRegex(ValueError,'deadline'):
            online.accept(prepared,response,scene=scene,now=12.1)
        self.assertIsNone(online.policy.request)

    def test_seating_rejects_rgb_from_another_scene(self):
        from decision.agent.tools.observations import Observation
        _,arrays,_,scene=observation(10.)
        _,_,_,other=observation(10.1)
        visual=Observation.from_scene(other,rgb=dict(pixels=arrays['rgb'],stamp_s=10.08,frame_id='camera'))
        online=SeatingDecision(TaskPolicy('seat','seating',self.planning,self.geometry))
        with self.assertRaisesRegex(ValueError,'does not bind'):
            online.seating(scene,candidate_set(scene),now=10.,observation=visual)
        self.assertIsNone(online.gate.pending)


if __name__=='__main__':unittest.main()
