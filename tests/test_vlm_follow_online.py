import copy
import json
import re
import unittest

import httpx

from decision.vlm.follow_online import FollowRequest, prepare, selected_reference
from decision.vlm.scene_inputs import model_eligible
from perception.environment.context import fingerprint
from tests.test_vlm_online import OnlineVlmTests


class FollowOnlineTests(unittest.TestCase):
    def setUp(self):
        self.scene,self.window,self.images=OnlineVlmTests().window()
        self.search=dict(clock_domain=self.window['clock_domain'],source_stamp_s=self.window['stamp_s'],
            candidates=[dict(id=0,kind='wait',pose=None)])
        self.search['binding']=fingerprint(self.search)

    def client(self, raw, *, finish='stop', tokens=None):
        calls=[]
        def handle(request):
            calls.append(request)
            if request.url.path=='/v1/models':
                return httpx.Response(200,json={'data':[{'id':'qwen3.8-27b-fp8'}]})
            if request.url.path=='/get_server_info':
                return httpx.Response(200,json={'server_args':{'speculative_algorithm':'DSPARK'}})
            event=dict(model='qwen3.8-27b-fp8',choices=[dict(index=0,delta=dict(content=raw),finish_reason=finish)])
            if tokens is not None:event['usage']=dict(completion_tokens=tokens)
            return httpx.Response(200,text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
        transport=httpx.MockTransport(handle)
        return lambda **kw:httpx.Client(transport=transport,**kw),calls

    def test_frozen_mapping_and_structured_response(self):
        prepared=prepare(self.window,self.images,'red jacket',self.search)
        self.assertEqual(prepared['choices']['P0']['kind'],'person')
        self.assertEqual(prepared['choices']['A']['candidate']['kind'],'wait')
        factory,calls=self.client('{"target":null,"observation":"A"}')
        result=FollowRequest(prepared).infer(client_factory=factory)
        self.assertEqual(result['choice']['candidate']['kind'],'wait')
        body=json.loads(next(c.content for c in calls if c.method=='POST'))
        self.assertTrue(re.fullmatch(body['regex'],'{"target":"P0","observation":null}'))
        self.assertTrue(re.fullmatch(body['regex'],'{"target":null,"observation":"A"}'))
        self.assertFalse(re.fullmatch(body['regex'],'{"target":"P0","observation":"A"}'))
        self.assertEqual(body['max_tokens'],24)
        factory,_=self.client('C')
        result=FollowRequest(prepared).infer(client_factory=factory)
        self.assertEqual(result['reason'],'invalid_t2_response')
        self.assertEqual(result['raw_content'],'C')

    def test_truncated_or_invalid_structured_completion_is_not_accepted(self):
        prepared=prepare(self.window,self.images,'red jacket',self.search)
        for raw,finish in [('{"target":"P0","observation":null}','length'),
                           ('{"target":"P9","observation":null}','stop'),
                           ('{"target":"P0","observation":"A"}','stop'),
                           ('{"target":null,"observation":null}','stop'),
                           ('{"target":"A","observation":null}','stop'),
                           ('{"target":null,"observation":"P0"}','stop'),
                           ('{"target":"P0","target":"P1","observation":null}','stop'),
                           ('{"target":"P0","observation":null,"other":1}','stop')]:
            with self.subTest(raw=raw,finish=finish):
                factory,_=self.client(raw,finish=finish,tokens=5)
                self.assertEqual(FollowRequest(prepared).infer(client_factory=factory)['status'],'unavailable')

    def test_changed_inputs_and_source_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.window,(self.images[0]+b'x',*self.images[1:]),'red jacket',self.search)
        changed=copy.deepcopy(self.search);changed['source_stamp_s']-=1
        changed['binding']=fingerprint({k:v for k,v in changed.items() if k!='binding'})
        with self.assertRaises(ValueError):
            prepare(self.window,self.images,'red jacket',changed)
        prepared=prepare(self.window,self.images,'red jacket',self.search)
        prepared['choices']['P0']['identity']=['wrong']
        with self.assertRaises(ValueError):FollowRequest(prepared).infer()

    def test_prediction_is_explicitly_opt_in(self):
        person=dict(kind='track',box=[0,0,0,1,1,1,0],confirmed=True,observed=False)
        self.assertFalse(model_eligible(person))
        self.assertTrue(model_eligible(person,include_predicted=True))
        person['confirmed']=False
        self.assertFalse(model_eligible(person,include_predicted=True))

    def test_search_marker_uses_planar_map_pose_when_map_is_tilted(self):
        import io
        import math
        import numpy as np
        from PIL import Image
        from decision.vlm.follow_online import _draw_search
        from decision.vlm.scene_inputs import bev_xy
        angle=math.pi/3
        transform=np.array([[math.cos(angle),0,math.sin(angle),0],
                            [0,1,0,0],[-math.sin(angle),0,math.cos(angle),2],
                            [0,0,0,1]])
        window=dict(scene=dict(localization=dict(world_from_lidar=transform.tolist()),
                               prior=dict(map_from_local=np.eye(4).tolist())),
                    fields=dict(e=[-4,4,-4,4]))
        out=io.BytesIO();Image.new('RGB',(1280,576),'white').save(out,format='PNG')
        choices={'A':dict(kind='observation',candidate=dict(kind='approach',pose=[2,0,0]))}
        rendered=Image.open(io.BytesIO(_draw_search(out.getvalue(),window,choices)))
        x,y=bev_xy(2,0,window['fields']['e'])
        self.assertNotEqual(rendered.getpixel((round(x)-8,round(y)+8)),(255,255,255))
        old_x,old_y=bev_xy(1+2*math.sin(angle),0,window['fields']['e'])
        self.assertEqual(rendered.getpixel((round(old_x),round(old_y))),(255,255,255))

    def test_single_request_and_source_bound_reference(self):
        prepared=prepare(self.window,self.images,'red jacket',self.search)
        request=FollowRequest(prepared)
        factory,_=self.client('{"target":"P0","observation":null}')
        result=request.infer(client_factory=factory)
        ref=selected_reference(prepared,result,self.window,task_id='mission',revision=1,goal_id='g')
        self.assertEqual(ref['track']['identity'],prepared['choices']['P0']['identity'])
        self.assertEqual(request.accept(result),prepared['choices']['P0'])
        with self.assertRaises(ValueError):request.accept(result)
        with self.assertRaises(ValueError):request.infer(client_factory=factory)

    def test_render_window_includes_predicted_only_for_follow(self):
        from decision.vlm.observations import render_window
        from runtime.decision import DecisionBuffer
        from tests.test_decision_runtime import observation
        from perception.environment.scene import assemble_scene
        from perception.environment.layers import load_config
        buffer=DecisionBuffer(max_frames=16,slot_bytes=32768)
        try:
            for i in range(12):
                stamp=10+i/10
                row,arrays,association,scene=observation(stamp)
                if i>=8:
                    row['tracks'][0]['state']='predicted'
                    row['tracks'][0]['last_observed_stamp_s']=10.7
                    row['input_fingerprint']=fingerprint({k:v for k,v in row.items() if k!='input_fingerprint'})
                    association['perception_input_fingerprint']=row['input_fingerprint']
                    scene=assemble_scene(scene.geometry,clock_domain='synthetic',settings=load_config()['scene'],people=row)
                buffer.push(row,arrays,association,scene=scene)
            lease=buffer.acquire()
            with buffer.consume(lease) as read:
                record,_=render_window(lease['entries'],lease['history'],read,
                    extent=[-4,4,-4,4],view_binding={},include_predicted=True)
            self.assertTrue(record['predicted_tracks_included'])
            self.assertEqual(record['fields']['v'],[0])
            trajectories=record['trajectories'][lease['entries'][-1]['id']]['people']
            self.assertTrue(any(not part['observed'] for parts in trajectories.values() for part in parts))
        finally:
            buffer.close()


if __name__=='__main__':
    unittest.main()
