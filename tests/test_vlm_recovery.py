"""Mocked DSPARK transport; no model service or external images are contacted."""
import json
import unittest

import httpx

from decision.vlm.follow_target import reference
from decision.vlm.recovery import run
from tests.test_follow_recovery import candidate_observation


class RecoveryVlmTests(unittest.TestCase):
    def setUp(self):
        self.original=candidate_observation(10.)
        self.ref,_,self.crop=reference(self.original,0,'红衣人物',task_id='f',revision=1,goal_id='g')
        self.current=candidate_observation(10.5,'b')

    def client(self, raw, *, ready=True, complete=True):
        calls=[]
        def handle(request):
            calls.append(request)
            if request.url.path=='/v1/models':
                return httpx.Response(200 if ready else 503,json={'data':[{'id':'qwen3.8-27b-fp8'}]})
            if request.url.path=='/get_server_info':
                return httpx.Response(200,json={'server_args':{'speculative_algorithm':'DSPARK'}})
            reply=dict(model='qwen3.8-27b-fp8',choices=[dict(index=0,delta=dict(content=raw),finish_reason='stop')])
            return httpx.Response(200,text='data: '+json.dumps(reply)+'\n\n'+('data: [DONE]\n\n' if complete else ''))
        transport=httpx.MockTransport(handle)
        return (lambda **kwargs:httpx.Client(transport=transport,**kwargs)),calls

    def test_unique_result_keeps_raw_response_and_bound_inputs(self):
        factory,calls=self.client('{"id":0,"match":"unique"}')
        result=run(self.ref,self.crop,self.current,client_factory=factory)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['raw_content'],'{"id":0,"match":"unique"}')
        self.assertEqual(result['response'],dict(id=0,match='unique'))
        self.assertEqual(result['candidate_binding'],self.current.record['follow_candidates']['binding'])
        body=json.loads(calls[-1].content)
        self.assertEqual(sum(part['type']=='image_url' for part in body['messages'][1]['content']),2)
        self.assertEqual(body['response_format']['type'],'json_schema')

    def test_invalid_response_is_not_repaired_or_retried(self):
        factory,calls=self.client('{"id":8,"match":"unique"}')
        result=run(self.ref,self.crop,self.current,client_factory=factory)
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['raw_content'],'{"id":8,"match":"unique"}')
        self.assertEqual(sum(r.method=='POST' for r in calls),1)

    def test_unavailable_remote_never_sends_inference_request(self):
        factory,calls=self.client('',ready=False)
        result=run(self.ref,self.crop,self.current,client_factory=factory)
        self.assertEqual(result['reason'],'remote_dspark_unavailable')
        self.assertFalse(any(r.method=='POST' for r in calls))

    def test_interrupted_stream_retains_received_text(self):
        factory,calls=self.client('{"id":0',complete=False)
        result=run(self.ref,self.crop,self.current,client_factory=factory)
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['raw_content'],'{"id":0')
        self.assertEqual(sum(r.method=='POST' for r in calls),1)


if __name__=='__main__':unittest.main()
