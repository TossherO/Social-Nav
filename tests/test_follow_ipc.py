"""Exercise real local IPC ownership without a robot or model service."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from decision.agent.adapters.runtime import LiveRuntimeClient
from navigation.planning.footprint import VehicleGeometry
from navigation.planning.inputs import load_config
from perception.environment.context import fingerprint
from runtime.command import CommandInbox
from runtime.control import ControlLoop
from runtime.follow import FollowRunner
from tests.test_follow_continuity import reference,scene


class FollowIpcTests(unittest.TestCase):
    def test_navigation_follow_navigation_requires_confirmed_holds(self):
        events=[];chassis=dict(state='armed',status_fresh=True,arm_revision=1)
        session=SimpleNamespace(live=True,execute=True,status=lambda:dict(chassis=chassis),note_command=lambda *_:None)
        with tempfile.TemporaryDirectory() as folder:
            inbox=CommandInbox(Path(folder)/'commands.sock',SimpleNamespace(command=lambda *v:events.append(v)),
                               session=session,agent_execution=True,follow_enabled=True)
            try:
                client=LiveRuntimeClient(inbox.path)
                packet=dict(request=None,stamp_s=10.,clock_domain='synthetic',feedback={})
                inbox.agent_state.publish_control(packet,map_binding=dict(valid=True))
                goal=dict(request_id='nav1',task_id='f',goal_revision=1,mode='mapped',frame_id='map',
                    target_reference='base_link',goal=[1.,2.,0.],map_snapshot_id='static')
                self.assertTrue(client.exchange('goal',goal)['accepted'])
                packet.update(request=goal)
                inbox.agent_state.goal_evidence=SimpleNamespace(update=lambda *_:dict(status='criteria_held',task_id='f',goal_revision=1))
                inbox.agent_state.publish_control(packet,map_binding=dict(valid=True))
                self.assertTrue(client.exchange('hold',dict(task_id='f',goal_revision=1))['accepted'])
                handle=dict(task_id='f',goal_id='g',plan_revision=1,goal_revision=2,call_id='follow')
                initial=scene(10.);ref=reference(initial);ref.update(task_revision=1,goal_id='g')
                ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
                with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
                    client.exchange('follow_start',dict(handle=handle,reference=ref))
                packet.update(request=None)
                localization=dict(health=0,session_id='s',generation=0,frame_id='odom')
                idle=dict(status='valid',source='causal_algorithm_poses_not_chassis_feedback',
                          local_identity=['s',0,'odom'],stamp_s=10.,velocity=[0.,0.])
                inbox.agent_state.publish_control(packet,localization,map_binding=dict(valid=True),
                    cancelled_request=dict(task_id='f',goal_revision=1),idle_feedback=idle)
                self.assertTrue(client.exchange('follow_start',dict(handle=handle,reference=ref))['accepted'])
                self.assertTrue(client.exchange('follow_finish',dict(call_id='follow'))['accepted'])
                next_goal=dict(goal,request_id='nav2',goal_revision=3)
                with self.assertRaisesRegex(ValueError,'previous_agent_goal_not_completed'):
                    client.exchange('goal',next_goal)
                inbox.agent_state.publish_control(packet,localization,map_binding=dict(valid=True),
                    cancelled_request=dict(task_id='follow:follow',goal_revision=0),idle_feedback=idle)
                self.assertTrue(client.exchange('goal',next_goal)['accepted'])
                self.assertEqual([kind for kind,_ in events],['goal','goal_hold','goal_hold','goal'])
            finally:inbox.close()

    def test_lost_retains_zero_hold_and_cancel_releases_owner(self):
        events=[]
        chassis=dict(state='armed',status_fresh=True,arm_revision=1)
        session=SimpleNamespace(live=True,execute=True,status=lambda:dict(chassis=chassis),note_command=lambda *_:None)
        with tempfile.TemporaryDirectory() as folder:
            inbox=CommandInbox(Path(folder)/'commands.sock',SimpleNamespace(command=lambda *v:events.append(v)),
                               session=session,agent_execution=True,follow_enabled=True)
            try:
                inbox.agent_state.publish_control(dict(request=None,stamp_s=10.,clock_domain='synthetic',feedback={}),
                                                  map_binding=dict(valid=True))
                client=LiveRuntimeClient(inbox.path)
                handle=dict(task_id='f',goal_id='g',plan_revision=1,goal_revision=1,call_id='call')
                initial=scene(10.);ref=reference(initial);ref.update(task_revision=1,goal_id='g')
                ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
                self.assertTrue(client.exchange('follow_start',dict(handle=handle,reference=ref))['accepted'])
                inbox.drain()
                self.assertTrue(client.exchange('follow_lost',dict(call_id='call',reason='follow_visual_lost'))['accepted'])
                self.assertEqual(events[-1][0],'goal_hold')
                self.assertIsNotNone(inbox.agent_owner)
                self.assertEqual(inbox.drain()[0][0],'follow_stop')
                self.assertTrue(client.exchange('follow_cancel',dict(call_id='call'))['accepted'])
                self.assertEqual(events[-1][0],'cancel')
            finally:inbox.close()

    def test_start_renew_and_finish_use_real_socket(self):
        events=[]
        chassis=dict(state='armed',status_fresh=True,arm_revision=1)
        session=SimpleNamespace(live=True,execute=True,status=lambda:dict(chassis=chassis),note_command=lambda *_:None)
        with tempfile.TemporaryDirectory() as folder:
            inbox=CommandInbox(Path(folder)/'commands.sock',SimpleNamespace(command=lambda *v:events.append(v)),
                               session=session,agent_execution=True,follow_enabled=True)
            try:
                packet=dict(request=None,stamp_s=10.,clock_domain='synthetic',feedback={})
                inbox.agent_state.publish_control(packet,map_binding=dict(valid=True))
                client=LiveRuntimeClient(inbox.path)
                handle=dict(task_id='f',goal_id='g',plan_revision=1,goal_revision=1,call_id='call')
                initial=scene(10.);ref=reference(initial);ref.update(task_revision=1,goal_id='g')
                ref['reference_id']=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})
                self.assertTrue(client.exchange('follow_start',dict(handle=handle,reference=ref))['accepted'])
                inbox.agent_owner['heartbeat_mono']=time.monotonic()-4.
                client.snapshot()
                self.assertGreater(time.monotonic()-inbox.agent_owner['heartbeat_mono'],3.)
                client.snapshot(follow_call_id='other')
                self.assertGreater(time.monotonic()-inbox.agent_owner['heartbeat_mono'],3.)
                client.snapshot(follow_call_id='call')
                self.assertLess(time.monotonic()-inbox.agent_owner['heartbeat_mono'],1.)
                kind,value=inbox.drain()[0];self.assertEqual(kind,'follow_start')
                geometry=VehicleGeometry([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],np.eye(4),source='synthetic')
                control=object.__new__(ControlLoop)
                control.lock=threading.Lock();control.follow=FollowRunner(load_config(),geometry)
                control.follow_pending=value;control.follow_visual_pending=[];control.follow_visual_review=None
                control.scene=initial;control.follow.observe(initial);control.request=None;control.cancelled_request=None
                control.commands=inbox;control.record=lambda *_:None
                control.reference=SimpleNamespace(invalidate=lambda *_:None,cancel=lambda *_:None)
                control.controller=SimpleNamespace(controller=SimpleNamespace(clear=lambda:None))
                control._update_follow(10.)
                self.assertEqual(client.snapshot()['follow']['status'],'running')
                self.assertEqual(events[-1][0],'goal')
                self.assertTrue(client.exchange('follow_finish',dict(call_id='call'))['accepted'])
                self.assertEqual(events[-1][0],'goal_hold')
                self.assertFalse(inbox.follow_goal(handle,dict(control.request,goal_revision=2)))
                self.assertEqual(inbox.drain()[0][0],'follow_stop')
            finally:inbox.close()


if __name__=='__main__':unittest.main()
