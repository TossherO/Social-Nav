"""Manual reference previews remain separate from chassis and ordinary goals."""
import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch

from runtime.command import CommandInbox
from decision.agent.workflows.manual_reference import ManualReference,reference_request,request_preview


def goal(owner='call',index=0):
    return dict(request_id=owner+':'+str(index),task_id='preview-task',goal_revision=index,
        mode='mapped',frame_id='map',target_reference='base_link',goal=[1.,2.],
        map_snapshot_id='map-id',goal_behavior='pass_through')


class ManualReferenceTests(unittest.TestCase):
    def inbox(self,*,execute=False,live=True):
        inbox=object.__new__(CommandInbox)
        inbox.session=SimpleNamespace(live=live,execute=execute)
        inbox.lock=threading.Lock();inbox.manual_preview_owner=None
        inbox.safety=SimpleNamespace(command=lambda *_:self.fail('preview entered safety command'))
        return inbox

    def test_only_observation_session_accepts_preview_and_lease_expires(self):
        inbox=self.inbox()
        with patch('runtime.command.time.monotonic',return_value=10.):
            self.assertTrue(inbox.manual_preview(dict(owner='call',request=goal()))['accepted'])
            self.assertEqual(inbox.manual_preview_snapshot()['request']['goal'],[1.,2.])
            with self.assertRaisesRegex(ValueError,'another session'):
                inbox.manual_preview(dict(owner='other',request=goal('other')))
            with self.assertRaisesRegex(ValueError,'bound mapped'):
                inbox.manual_preview(dict(owner='call',request=goal('different')))
        with patch('runtime.command.time.monotonic',return_value=13.1):
            self.assertIsNone(inbox.manual_preview_snapshot())
        for session in (self.inbox(execute=True),self.inbox(live=False)):
            with self.assertRaisesRegex(ValueError,'without hardware output'):
                session.manual_preview(dict(owner='call',request=goal()))

    def test_route_preview_tracks_current_waypoint_and_final_yaw(self):
        catalog=SimpleNamespace(value=dict(frame_id='map',map_snapshot_id='map-id'))
        handle=dict(call_id='call',task_id='task',goal_revision=2)
        route=dict(waypoints=[[1.,2.],[3.,4.]],final_yaw=.4)
        first=reference_request(handle,[3.,4.,.4],route,0,catalog)
        final=reference_request(handle,[3.,4.,.4],route,1,catalog,require_yaw=True)
        self.assertEqual(first['goal'],[1.,2.]);self.assertEqual(first['goal_behavior'],'pass_through')
        self.assertEqual(final['goal'],[3.,4.,.4]);self.assertEqual(final['goal_behavior'],'stop')
        self.assertNotEqual(first['request_id'],final['request_id'])
        self.assertEqual(reference_request(handle,[3.,4.,.4],route,1,catalog)['goal'],[3.,4.])
        self.assertEqual(reference_request(handle,[3.,4.,.4],None,None,catalog)['goal'],[3.,4.])

    def test_worker_changes_and_clears_lease_without_chassis_commands(self):
        calls=[];first=threading.Event();cleared=threading.Event()
        def send(_path,owner,request):
            calls.append((owner,copy.deepcopy(request)))
            if request is None:cleared.set()
            else:first.set()
            return dict(accepted=True)
        with patch('decision.agent.workflows.manual_reference.request_preview',side_effect=send):
            worker=ManualReference('/unused')
            try:
                worker.set('call',goal());self.assertTrue(first.wait(2))
                worker.set(None,None);self.assertTrue(cleared.wait(2))
            finally:worker.close()
        self.assertEqual(calls[0][0],'call');self.assertIsNone(calls[-1][1])

    def test_preview_ipc_does_not_submit_control(self):
        with tempfile.TemporaryDirectory() as root:
            commands=[]
            session=SimpleNamespace(live=True,execute=False,note_command=lambda *_:None)
            inbox=CommandInbox(Path(root)/'commands.sock',SimpleNamespace(command=lambda *v:commands.append(v)),session=session)
            try:
                result=request_preview(inbox.path,'call',goal())
                self.assertTrue(result['accepted']);self.assertEqual(commands,[]);self.assertEqual(inbox.drain(),[])
                self.assertEqual(inbox.manual_preview_snapshot()['owner'],'call')
                request_preview(inbox.path,'call',None)
                self.assertIsNone(inbox.manual_preview_snapshot());self.assertEqual(commands,[])
            finally:inbox.close()

    def test_control_tick_plans_preview_but_keeps_controller_idle(self):
        from runtime.control import ControlLoop
        loop=object.__new__(ControlLoop);loop.done=threading.Event();loop.error=None
        loop.pose_socket=SimpleNamespace(recv=MagicMock(side_effect=BlockingIOError))
        loop.clock=dict(clock_s=10.,clock_mono=time.monotonic(),clock_rate=1.)
        loop.rows=[dict(stamp_s=10.,session_id='s',generation=0,frame_id='odom')]
        scene=SimpleNamespace(record=dict(prior=dict(valid=True,map_id='map'),as_of=10.))
        loop.scene=None;loop.pending=(scene,object());loop.lock=threading.Lock();loop.offered_mono=None
        loop.session=SimpleNamespace(epoch=0,pause_requested=False,paused=False,publish_observation=lambda *a,**k:None)
        loop.session_epoch=0;loop.follow=None;loop.route_replay=None;loop.request=None;loop.manual_preview_key=None
        loop.settings=dict(control_period_s=.1);loop.metrics=SimpleNamespace(trace_path=None,add=lambda *a:None,span=lambda *_:nullcontext())
        loop.reference=SimpleNamespace(invalidate=MagicMock(),update=MagicMock(),tick=MagicMock(return_value=dict(
            status='reference_ready',reason=None,waiting={},start_evidence=None)),current=dict(as_of=9.9),
            reference=object(),reason=None,blocked=False,note_control=MagicMock(),drain_events=lambda:[])
        packet=dict(request=None,stamp_s=10.,clock_domain='runtime_live',feedback={},sequence_id=1,reason='no_active_goal')
        controller=SimpleNamespace(result=None,clear=lambda:None)
        loop.controller=SimpleNamespace(controller=controller,geometry=object(),motion={'state_max_age_s':.1},step=MagicMock(return_value=packet))
        preview=dict(owner='call',request=goal(),expires_mono=time.monotonic()+3.)
        state=SimpleNamespace(goal_evidence=object(),publish_control=MagicMock())
        loop.commands=SimpleNamespace(drain=lambda:[],manual_preview_snapshot=lambda:preview,agent_state=state)
        loop.safety=SimpleNamespace(packet=MagicMock());loop.record=lambda *a:None
        loop.batch=SimpleNamespace(bind=lambda *_:None);loop.planning_recorder=None;loop.cancelled_request=None
        def publish(*args,**kwargs):loop.done.set()
        loop.display=SimpleNamespace(navigation=MagicMock(side_effect=publish));loop.asset=None;loop.count=0
        loop.clock_domain='runtime_live';loop.device=None
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(set_device=lambda *_:None,Stream=lambda **_:object(),stream=lambda *_:nullcontext()))
        idle=dict(status='valid',velocity=[0.,0.])
        with patch.dict('sys.modules',{'torch':fake_torch}),patch('navigation.control.shadow.observed_feedback',return_value=idle):
            loop._work()
        self.assertIsNone(loop.error)
        self.assertEqual(loop.reference.update.call_args.args[0],preview['request'])
        self.assertIsNone(loop.controller.step.call_args.args[0])
        self.assertIsNone(loop.controller.step.call_args.args[3])
        self.assertIsNone(loop.safety.packet.call_args.args[0]['request'])
        self.assertEqual(state.publish_control.call_args.kwargs['idle_feedback'],idle)
        self.assertEqual(state.publish_control.call_args.kwargs['manual_preview']['status'],'reference_ready')
        self.assertEqual(loop.display.navigation.call_args.args[2],preview['request'])

    @unittest.skipUnless(Path('outputs/runtime/build/planning/libsocial_nav_pose_search.so').is_file(),
                         'Local pose search native library unavailable')
    def test_display_keeps_real_reference_without_control_proposal(self):
        from tests.test_navigation_display import NavigationDisplayTests
        from tests.navigation_fixtures import asset,bundle,request
        from perception.visualization.navigation import NavigationView
        fixture=NavigationDisplayTests();fixture.setUp()
        fixture.cfg['pose_search']['native_library']=str(Path('outputs/runtime/build/planning/libsocial_nav_pose_search.so').resolve())
        scene=bundle();static=asset();req=request('mapped',goal=[4.,2.,0.],goal_behavior='stop')
        packet=fixture.packet(scene,req,static)
        packet.update(result=None,proposal=None,reason='no_active_goal')
        view=NavigationView(fixture.geometry,fixture.cfg,asset=static).update(fixture.row,packet,mono=0.)
        self.assertIsNotNone(view['reference'],view['text'])
        self.assertIsNone(view['trajectory'])


if __name__=='__main__':unittest.main()
