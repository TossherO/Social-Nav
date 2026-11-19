"""Task recording lifecycle with a simulated recorder and no ROS or hardware."""
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from decision.agent.core.loop import AgentLoop
from decision.agent.core.session import AgentSession
from decision.agent.core.store import Store
from decision.agent.tools.places import Catalog
from decision.agent import ROOT
from decision.agent.workflows.task_bag import TaskBagRecorder
from tests.test_agent import ScriptedFeedback,reply


class Process:
    def __init__(self):self.code=None;self.signals=[]
    def poll(self):return self.code
    def send_signal(self,number):self.signals.append(number)


class TaskBagTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.output=Path(self.temp.name)
        self.session=SimpleNamespace(state=dict(revision=1,status='ready'),
            task_goals=SimpleNamespace(goals=[]),navigation=SimpleNamespace(ros_master_uri='http://127.0.0.1:12345'),
            store=SimpleNamespace(event=Mock()),stop=Mock())
        self.events=[];self.proc=Process();self.popen=Mock(return_value=self.proc)
        self.recorder=TaskBagRecorder(self.output,self.session,self.events.append,output_root=str(self.output),popen=self.popen)

    def tearDown(self):self.temp.cleanup()

    def status(self,status,**fields):
        self.recorder.status_file.write_text(json.dumps(dict(status=status,**fields)))

    def test_waits_for_registered_plan_and_recorded_data_before_first_execution(self):
        self.assertTrue(self.recorder.before_request(None));self.popen.assert_not_called()
        self.session.task_goals.goals=[{'goal':'first'}]
        self.assertFalse(self.recorder.before_request(None))
        self.assertFalse(self.recorder.before_request(None))
        self.assertEqual(self.popen.call_count,1)
        command=self.popen.call_args.args[0]
        self.assertEqual(command[command.index('--parent-pid')+1],str(__import__('os').getpid()))
        self.assertEqual(self.popen.call_args.kwargs['env']['SOCIAL_NAV_RECORD_ROS_MASTER_URI'],'http://127.0.0.1:12345')
        self.status('starting',bag='/tmp/test.bag')
        self.assertFalse(self.recorder.before_request(None))
        self.status('recording',bag='/tmp/test.bag')
        self.assertTrue(self.recorder.before_request(None))
        self.session.state['status']='succeeded';self.recorder.update()
        self.assertEqual(self.proc.signals,[signal.SIGINT])
        self.recorder.close();self.assertEqual(self.proc.signals,[signal.SIGINT])

    def test_failure_before_start_stops_task_without_retry(self):
        self.session.task_goals.goals=[{'goal':'first'}]
        self.assertFalse(self.recorder.before_request(None))
        self.status('failed',error='missing sensor')
        self.proc.code=1
        self.assertFalse(self.recorder.before_request(None))
        self.session.stop.assert_called_with('needs_clarification','agent_bag_recording_unavailable')
        self.assertFalse(self.recorder.before_request(None))
        self.assertEqual(self.popen.call_count,1)

    def test_missing_master_blocks_execution_with_clear_status(self):
        self.session.navigation.ros_master_uri=None
        self.session.task_goals.goals=[{'goal':'first'}]
        self.assertFalse(self.recorder.before_request(None))
        self.popen.assert_not_called()
        self.session.stop.assert_called_once_with('needs_clarification','agent_bag_recording_unavailable')
        status=json.loads(self.recorder.status_file.read_text())
        self.assertEqual(status['status'],'failed')
        self.assertIn('ROS master',status['error'])

    def test_unexpected_mid_task_failure_is_visible_and_not_silently_retried(self):
        self.session.task_goals.goals=[{'goal':'first'}]
        self.recorder.before_request(None);self.status('recording',bag='/tmp/test.bag')
        self.assertTrue(self.recorder.before_request(None))
        self.status('failed',error='disk error',bag='/tmp/test.bag');self.proc.code=1
        self.assertTrue(self.recorder.before_request(None))
        self.assertTrue(any(x['status']=='failed' for x in self.events))
        self.assertEqual(self.popen.call_count,1)

    def test_new_task_waits_until_previous_bag_has_finished(self):
        self.session.task_goals.goals=[{'goal':'first'}]
        self.recorder.before_request(None);self.status('recording',bag='/tmp/first.bag')
        self.assertTrue(self.recorder.before_request(None))
        self.session.state['revision']=2
        self.assertFalse(self.recorder.before_request(None))
        self.assertEqual(self.proc.signals,[signal.SIGINT])
        self.status('saved',bag='/tmp/first.bag');self.proc.code=0
        self.popen.return_value=Process()
        self.assertFalse(self.recorder.before_request(None))
        self.assertEqual(self.popen.call_count,2)
        self.assertEqual(self.recorder.revision,2)

    def test_planning_can_finish_but_next_model_request_waits_for_recorder(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            session=AgentSession('到电梯',Catalog.load(ROOT/'decision/agent/configs/simulation.json'),ScriptedFeedback(),store)
            actor=AgentLoop(session,lambda *_:reply('record_task',goals=[dict(goal_id='g',stage='point',
                target=dict(id='elevator'),description='到电梯')]))
            release=[False]
            actor.before_request=lambda _:not session.task_goals.goals or release[0]
            try:
                actor.tick();actor.model.thread.join(2);actor.tick()
                self.assertEqual(actor.requests,1)
                self.assertTrue(session.task_goals.goals)
                actor.tick();self.assertEqual(actor.requests,1)
                release[0]=True;actor.tick();self.assertEqual(actor.requests,2)
                actor.model.thread.join(2)
            finally:actor.close();store.close()


if __name__=='__main__':unittest.main()
