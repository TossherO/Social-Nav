"""Online T2 scheduling uses sealed causal observations and task epochs."""
import json
import threading
import time
import unittest

import httpx

from perception.environment.context import fingerprint
from runtime.decision import DecisionBuffer
from runtime.follow_decision import FollowDecisionWorker
from tests.test_decision_runtime import observation


def search(scene, context):
    value=dict(clock_domain=scene.record['clock_domain'],source_stamp_s=scene.record['as_of'],
        scene_binding=scene.record['input_fingerprint'],candidates=[dict(id=0,kind='wait',pose=None)])
    value['binding']=fingerprint(value)
    return value


def factory(raw, gate=None):
    def handle(request):
        if request.url.path=='/v1/models':
            return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b-fp8'}]})
        if request.url.path=='/get_server_info':
            return httpx.Response(200,json={'server_args':{'speculative_algorithm':'DSPARK'}})
        if gate is not None:gate.wait(3)
        value=dict(target=raw if raw.startswith('P') else None,observation=None if raw.startswith('P') else raw)
        event=dict(model='qwen3.8-27b-fp8',choices=[dict(index=0,delta=dict(content=json.dumps(value)),finish_reason='stop')])
        return httpx.Response(200,text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
    transport=httpx.MockTransport(handle)
    return lambda **kw:httpx.Client(transport=transport,**kw)


class FollowDecisionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.gate=None
        self.handle=dict(task_id='f',goal_id='g',plan_revision=1,goal_revision=1,call_id='call')

    def worker(self, client):
        return FollowDecisionWorker(search,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                                    inference=lambda request:request.infer(client_factory=client))

    def feed(self,worker):
        for i in range(12):
            row,arrays,association,scene=observation(round(10+i/10,1))
            worker.submit(row,arrays,scene=scene,image_metadata=None,association=association)

    def wait_result(self,worker):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            result=worker.poll()
            if result is not None:return result
            time.sleep(.01)
        self.fail('T2 worker did not finish')

    def test_real_window_selection_and_fixed_crop(self):
        worker=self.worker(factory('P0'))
        try:
            worker.begin(self.handle,'red jacket');self.feed(worker)
            outcome=self.wait_result(worker)
            self.assertEqual(outcome['result']['choice']['kind'],'person')
            self.assertEqual(outcome['prepared']['task'],self.handle)
            self.assertEqual(outcome['window']['fields']['dt'],[-1.,-.5,0.])
            self.assertTrue(worker.adopt_reference(self.handle,outcome['reference_proposal']))
            self.assertFalse(worker.adopt_reference(self.handle,outcome['reference_proposal']))
            self.assertTrue(worker.snapshot()['reference_fixed'])
        finally:worker.close()

    def test_cancelled_result_is_discarded_and_cannot_adopt_reference(self):
        gate=threading.Event();worker=self.worker(factory('P0',gate))
        try:
            worker.begin(self.handle,'red jacket');self.feed(worker)
            deadline=time.monotonic()+3
            while not worker.snapshot()['inflight'] and time.monotonic()<deadline:time.sleep(.01)
            worker.stop();gate.set()
            deadline=time.monotonic()+3
            while worker.snapshot()['inflight'] and time.monotonic()<deadline:time.sleep(.01)
            self.assertIsNone(worker.poll())
            self.assertFalse(worker.snapshot()['active'])
        finally:gate.set();worker.close()

    def test_single_bad_observation_does_not_disable_later_frames(self):
        calls=[]
        def flaky(scene,context):
            calls.append(scene.record['as_of'])
            if len(calls)==1:raise ValueError('one invalid input')
            return search(scene,context)
        worker=FollowDecisionWorker(flaky,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                                    inference=lambda request:request.infer(client_factory=factory('P0')))
        try:
            worker.begin(self.handle,'red jacket');self.feed(worker)
            result=self.wait_result(worker)
            self.assertTrue(result['transient'])
            self.assertIsNone(worker.snapshot()['error'])
            worker.request_fresh()
            row,arrays,association,scene=observation(11.2)
            worker.submit(row,arrays,scene=scene,image_metadata=None,association=association)
            self.assertEqual(self.wait_result(worker)['result']['choice']['kind'],'person')
            self.assertEqual(len(calls),2)
        finally:worker.close()


if __name__=='__main__':unittest.main()
