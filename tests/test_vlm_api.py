"""Current wheelchair transport and service checks; synthetic assets and HTTP only."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from decision.vlm import server,vlm_api_common as common
from decision.vlm.follow import appearance_schema,run,file_hash,write_json


def request_fixture(directory):
    directory.mkdir()
    images=[]
    for i in range(3):
        path=directory/('image_%d.png'%i)
        Image.new('RGB',(8,8),(20*i,60,90)).save(path)
        images.append(path)
    fields=dict(v=[0,1],dt=[-1,-.5,0],ts={},e=[-5,15,-10,10],o=[],d=[],target_description='dark coat')
    case=dict(paths=images,model=dict(fields=fields,response_schema=appearance_schema(fields['v']),
              user_prompt=json.dumps(fields,separators=(',',':'))))
    return common.make_body('Match the described person.',case,max_tokens=128)


def sse(content):
    event=dict(model='qwen3.8-27b-fp8',choices=[dict(index=0,delta=dict(content=content),finish_reason='stop')],
               usage=dict(prompt_tokens=20,completion_tokens=10))
    return 'data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n'


class TransportTests(unittest.TestCase):
    def test_messages_preserve_image_order_and_dynamic_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            body=request_fixture(Path(folder)/'input')
        content=body['messages'][1]['content']
        self.assertEqual([content[2*i]['text'] for i in range(3)],['IMAGE_0','IMAGE_1','IMAGE_2'])
        self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(body['temperature'],0)
        self.assertEqual(body['response_format']['type'],'json_schema')

    def test_stream_preserves_invalid_json_without_repair(self):
        response=httpx.Response(200,text=sse('{bad JSON'))
        result=common.read_stream(response,time.perf_counter(),{})
        self.assertEqual(result['raw_content'],'{bad JSON')
        with self.assertRaises(ValueError):
            common.read_stream(httpx.Response(200,text='data: {}\n'),time.perf_counter(),{})

    def test_server_metadata_is_get_only_and_whitelisted(self):
        calls=[]
        def handler(request):
            calls.append(request.method)
            return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b-fp8'}]} if request.url.path.endswith('/models') else
                {'server_args':{'speculative_algorithm':'DSPARK','secret':'DO_NOT_RETAIN'},'version':'test'})
        config=dict(base_url='http://localhost/v1',api_key='',model='qwen3.8-27b-fp8')
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result=server.check_server(client,config,info=True)
        self.assertEqual(calls,['GET','GET'])
        self.assertEqual(result['status'],'ready')
        self.assertNotIn('DO_NOT_RETAIN',str(result))

    def test_unsafe_urls_and_ambiguous_model_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            for url in ['http://user:password@localhost/v1','http://localhost/v1?token=x','http://public.example/v1']:
                path.write_text(json.dumps(dict(base_url=url,model='m')))
                with self.assertRaises(ValueError):server.load_config(path)
        with self.assertRaises(ValueError):server.choose_model({'data':[{'id':'other'}]},'expected')

    def test_follow_reuses_frozen_assets_and_writes_separate_results(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'frozen';body=request_fixture(base)
            fields=json.loads(body['messages'][1]['content'][-1]['text'])
            (base/'prompt.txt').write_text('Match the described person.')
            (base/'input.json').write_text('{}')
            write_json(base/'reference.json',{})
            item=dict(case_id='F01',input_fingerprint='input',fields=fields,
                response_schema=appearance_schema(fields['v']),
                images=[dict(path='image_%d.png'%i,sha256=file_hash(base/('image_%d.png'%i))) for i in range(3)],
                source_input=dict(path='input.json',sha256=file_hash(base/'input.json')))
            write_json(base/'inputs.json',[item])
            write_json(base/'settings.json',dict(inputs_sha256=file_hash(base/'inputs.json'),
                reference_sha256=file_hash(base/'reference.json'),prompt_sha256=file_hash(base/'prompt.txt'),max_tokens=128))
            posts=[]
            def handler(request):
                if request.method=='POST':
                    posts.append(json.loads(request.content))
                    return httpx.Response(200,text=sse('{"id":0,"match":"unique"}'))
                return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b-fp8'}]} if request.url.path.endswith('/models') else
                                      {'server_args':{'speculative_algorithm':'DSPARK'}})
            client=httpx.Client(transport=httpx.MockTransport(handler))
            result=Path(folder)/'result'
            with patch('httpx.Client',return_value=client),patch('sys.stdout',new=io.StringIO()):
                run(base,'http://localhost/v1',result_directory=result)
            self.assertEqual(len(posts),2) # One explicit warmup and one task, no retry.
            self.assertFalse((base/'requests.jsonl').exists())
            self.assertTrue(json.loads((result/'requests.jsonl').read_text())['valid'])
            self.assertEqual(posts[0],posts[1])


if __name__=='__main__':unittest.main()
