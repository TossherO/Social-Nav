"""Continuous T2 worker drives a bound follow owner through the control loop."""
import copy
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from runtime.control import ControlLoop
from runtime.decision import DecisionBuffer
from runtime.follow import FollowRunner
from runtime.follow_decision import FollowDecisionWorker
from tests.test_decision_runtime import observation
from tests.test_follow_decision_worker import factory,search


class ControlT2Tests(unittest.TestCase):
    def setUp(self):
        self.events=[];self.follow_states=[];self.goals=[];self.pauses=[]
        self.geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
        self.handle=dict(task_id='mission',plan_revision=1,goal_id='g',goal_revision=1,call_id='call')
        self.choice='P0'
        def infer(request):
            return request.infer(client_factory=factory(self.choice))
        self.worker=FollowDecisionWorker(search,inference=infer,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768))
        self.worker.begin(self.handle,'red jacket')
        control=object.__new__(ControlLoop);self.control=control
        control.lock=threading.Lock();control.follow=FollowRunner(load_config(),self.geometry)
        control.follow.handle=copy.deepcopy(self.handle)
        control.follow_pending=dict(handle=self.handle,intent=dict(target_description='red jacket'))
        control.follow_visual_pending=[];control.follow_visual_review=None
        control.follow_decision=self.worker;control.follow_search_anchor=control.follow_search_choice=None
        control.request=None;control.cancelled_request=None;control.asset=None;control.clock_domain='synthetic'
        control.record=lambda name,event:self.events.append((name,event))
        control.reference=SimpleNamespace(planning=load_config(),invalidate=lambda *_:None,cancel=lambda *_:None)
        control.controller=SimpleNamespace(controller=SimpleNamespace(geometry=self.geometry,clear=lambda:None))
        def follow_goal(handle,request):
            self.goals.append(copy.deepcopy(request));return True
        def pause(handle):
            self.pauses.append(copy.deepcopy(handle))
            return dict(task_id='follow:call',goal_revision=self.goals[-1]['goal_revision'] if self.goals else 0)
        control.commands=SimpleNamespace(follow_goal=follow_goal,pause_follow=pause,follow_owner_valid=lambda _:True,
            follow_failed=lambda *_args,**_kw:dict(target=dict(task_id='follow:call',goal_revision=0),retained=False),
            agent_state=SimpleNamespace(publish_follow=lambda value:self.follow_states.append(copy.deepcopy(value))))

    def tearDown(self):
        self.worker.close()

    def feed(self,start,end):
        for i in range(start,end):
            stamp=round(10+i/10,1)
            row,arrays,association,scene=observation(stamp)
            self.control.follow.observe(scene);self.control.scene=scene
            self.worker.submit(row,arrays,scene=scene,image_metadata=None,association=association,
                               context=self.control.follow_context())
            if i>=10:self.control._update_follow(stamp)
            time.sleep(.001)

    def drain(self, stamp):
        deadline=time.monotonic()+6
        while self.worker.snapshot()['inflight'] and time.monotonic()<deadline:time.sleep(.01)
        self.control._update_follow(stamp)

    def test_initial_selection_search_and_reacquisition_keep_same_owner(self):
        self.feed(0,12);self.drain(11.1)
        self.assertEqual(self.follow_states[-1]['status'],'running')
        self.assertEqual(self.follow_states[-1]['target_identity'],['tracker_session',7,'a'])
        self.assertEqual(self.goals[-1]['request_id'],'call')
        self.assertTrue(self.worker.snapshot()['reference_fixed'])
        first_revision=self.goals[-1]['goal_revision']
        self.choice='A'
        self.feed(12,34);self.drain(13.3)
        self.assertEqual(self.follow_states[-1]['status'],'running')
        self.feed(34,82);self.drain(18.1)
        self.assertEqual(self.follow_states[-1]['status'],'searching')
        self.assertTrue(self.pauses)
        self.assertIsNone(self.control.request)
        self.choice='P0'
        self.feed(82,105);self.drain(20.4)
        self.assertEqual(self.follow_states[-1]['status'],'running')
        self.assertGreater(self.goals[-1]['goal_revision'],first_revision)
        self.assertEqual(self.goals[-1]['request_id'],'call')

    def test_search_pose_is_checked_again_at_current_map_and_deadline(self):
        from tests.navigation_fixtures import asset,bundle
        from perception.environment.layers import GridSnapshot,load_config as env_config
        from perception.environment.scene import assemble_scene
        self.control.asset=asset();self.control.scene=bundle()
        self.control.follow_search_anchor=[2.,2.]
        choice=dict(kind='approach',pose=[2.5,2.,0.])
        request=self.control._follow_search_request(self.handle,choice,'source',10.,12.)
        self.assertIsNotNone(request)
        self.assertEqual(request['search_bounds']['radius_m'],1.5)
        self.assertEqual(request['valid_until_s'],12.)
        cells=np.zeros((60,60),np.int8);cells[20,25]=-1
        scene=self.control.scene
        changed=GridSnapshot(copy.deepcopy(scene.geometry.metadata),dict(occupancy=cells))
        self.control.scene=assemble_scene(changed,clock_domain='synthetic',settings=env_config()['scene'])
        self.assertIsNone(self.control._follow_search_request(self.handle,choice,'source',10.,12.))
        self.assertIsNone(self.control._follow_search_request(self.handle,choice,'source',13.,12.))

    def test_waiting_owner_loss_stops_worker(self):
        self.control.commands.follow_owner_valid=lambda _:False
        self.control._update_follow(10.)
        self.assertEqual(self.follow_states[-1]['status'],'failed')
        self.assertEqual(self.follow_states[-1]['reason'],'follow_owner_arming_or_agent_heartbeat_lost')
        self.assertFalse(self.worker.snapshot()['active'])


if __name__=='__main__':unittest.main()
