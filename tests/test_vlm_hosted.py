"""Hosted comparisons keep frozen pixels, causal state and bounded API accounting."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from decision.agent.core.store import Budget
from decision.vlm import hosted_evaluation as hosted
from decision.vlm.follow import FollowHistory
from decision.vlm.vlm_api_common import groups_schema, save_json


class HostedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config=self.root/'private.json'
        save_json(self.config,dict(base_url='https://provider.invalid/v1',api_key='test-private-key'))
        self.budget=Budget(self.root/'budget.sqlite');self.addCleanup(self.budget.close)
        self.env=patch.dict(os.environ,{'SOCIAL_NAV_AGENT_API_KEY':''});self.env.start();self.addCleanup(self.env.stop)

    def fixture(self):
        images=[]
        for n in range(3):
            path=self.root/f'{n}.png';path.write_bytes(b'original-png-pixels'+bytes([n]))
            images.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        item=dict(fields=dict(v=[0,1]),images=images,response_schema=groups_schema([0,1]),
                  stamp_ns=10_000_000_000,source='bag',window=dict(mapping={'track':1},frames=['now'],trajectories={}))
        path=self.root/'input.json';save_json(path,item)
        case=dict(id='case',input=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),images=images)
        batch=dict(name='test',task='groups',max_tokens=256,prompt=dict(text='frozen prompt',sha256=hashlib.sha256(b'frozen prompt').hexdigest()),cases=[case])
        return batch,case

    def test_original_images_and_causal_history_are_used(self):
        batch,case=self.fixture();item,body,_=hosted.prepare_case(batch,case)
        self.assertFalse(body['enable_thinking']);self.assertNotIn('chat_template_kwargs',body)
        self.assertEqual(body['response_format']['json_schema']['schema'],item['response_schema'])
        images=[p for p in body['messages'][1]['content'] if p['type']=='image_url']
        for part,spec in zip(images,case['images']):
            self.assertEqual(base64.b64decode(part['image_url']['url'].split(',')[1]),Path(spec['path']).read_bytes())
        batch['task']='follow';history=FollowHistory(2.5)
        _,body,context=hosted.prepare_case(batch,case,history)
        self.assertFalse(context['confirmed_before'])
        history.accept(dict(id=1,state='visible'),item,item['window'])
        _,body,context=hosted.prepare_case(batch,case,history)
        self.assertTrue(context['confirmed_before']);self.assertEqual(context['current_pid'],1)
        self.assertEqual(json.loads(body['messages'][1]['content'][-1]['text'])['last_target'],context)
        Path(case['images'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):hosted.prepare_case(batch,case,history)

    def test_response_is_preserved_even_when_semantically_invalid(self):
        batch,case=self.fixture();item,body,_=hosted.prepare_case(batch,case)
        raw='{"walking":[[0,0]]}'
        def handler(request):
            self.assertEqual(json.loads(request.content),body)
            self.assertEqual(request.headers['Authorization'],'Bearer test-private-key')
            event=dict(model=hosted.Model.model,choices=[dict(index=0,delta=dict(content=raw),finish_reason='stop')],
                       usage=dict(prompt_tokens=3000,completion_tokens=12))
            return httpx.Response(200,text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
        client=hosted.HostedClient(self.config,self.budget,transport=httpx.MockTransport(handler));row={}
        value=client.request(body,row)
        with self.assertRaises(Exception):hosted.validate_response('groups',value,item,body)
        self.assertEqual(row['raw_content'],raw);self.assertFalse(row['valid'])
        self.assertEqual(self.budget.summary()['calls'],1)
        self.assertLess(self.budget.summary()['accounted_rmb'],.01)
        self.assertGreaterEqual(row['timing']['full_return_s'],row['timing']['first_output_text_s'])

    def test_http_failure_is_not_retried_and_keeps_reservation(self):
        batch,case=self.fixture();_,body,_=hosted.prepare_case(batch,case);calls=[]
        def handler(request):
            calls.append(request)
            return httpx.Response(400,text='test-private-key https://provider.invalid/v1/chat/completions')
        client=hosted.HostedClient(self.config,self.budget,transport=httpx.MockTransport(handler));row={}
        self.assertIsNone(client.request(body,row));self.assertEqual(len(calls),1)
        self.assertNotIn('test-private-key',str(row));self.assertNotIn('provider.invalid',str(row))
        self.assertGreater(self.budget.summary()['accounted_rmb'],.1)

    def test_portable_schema_keeps_strict_local_validation(self):
        batch,case=self.fixture();batch.update(task='follow',schema_mode='portable')
        item,body,_=hosted.prepare_case(batch,case,FollowHistory(2.5))
        self.assertNotIn('oneOf',body['response_format']['json_schema']['schema'])
        # Each field is individually allowed, but the pair is forbidden.
        with self.assertRaises(Exception):hosted.validate_response('follow',dict(id=1,state='not_found'),item,body)
        hosted.validate_response('follow',dict(id=-1,state='not_found'),item,body)
        batch['task']='groups';item,body,_=hosted.prepare_case(batch,case)
        self.assertNotIn('uniqueItems',json.dumps(body['response_format']))
        with self.assertRaises(Exception):hosted.validate_response('groups',dict(walking=[[0,0]]),item,body)


if __name__=='__main__':unittest.main()
