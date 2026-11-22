"""Current wheelchair transport and service checks; synthetic assets and HTTP only."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx
from PIL import Image

from decision.vlm import server,vlm_api_common as common
from decision.vlm.t1.schema import groups_schema


def request_fixture(directory):
    directory.mkdir()
    images=[]
    for i in range(3):
        path=directory/('image_%d.png'%i)
        Image.new('RGB',(8,8),(20*i,60,90)).save(path)
        images.append(path)
    fields=dict(v=[0,1],dt=[-1,-.5,0],ts={},e=[-5,15,-10,10],o=[],d=[])
    case=dict(paths=images,model=dict(fields=fields,response_schema=groups_schema(fields['v']),
              user_prompt=json.dumps(fields,separators=(',',':'))))
    return common.make_body('Identify social groups.',case,max_tokens=128)


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



if __name__=='__main__':unittest.main()
