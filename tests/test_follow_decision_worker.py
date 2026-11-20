"""Online T2 scheduling uses sealed causal observations and task epochs."""
import json
import hashlib
import io
from pathlib import Path
import tempfile
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
        from PIL import Image
        out=io.BytesIO();Image.new('RGB',(15,35),'red').save(out,format='PNG');self.crop=out.getvalue()
        self.reference=dict(description='red jacket',clock_domain='synthetic',rgb_stamp_s=9.,
                            crop_sha256=hashlib.sha256(self.crop).hexdigest())
        self.reference['reference_id']=fingerprint(self.reference)

    def begin(self,worker):
        worker.begin(self.handle,'red jacket',reference=self.reference,crop=self.crop)

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
            self.begin(worker);self.feed(worker)
            outcome=self.wait_result(worker)
            self.assertEqual(outcome['result']['choice']['kind'],'person')
            self.assertEqual(outcome['prepared']['task'],self.handle)
            self.assertEqual(outcome['window']['fields']['dt'],[-1.,-.5,0.])
            self.assertIsNone(outcome['reference_proposal'])
            self.assertEqual(outcome['prepared']['reference_id'],self.reference['reference_id'])
            self.assertTrue(worker.snapshot()['reference_fixed'])
        finally:worker.close()

    def test_next_inference_uses_next_frame_without_two_second_period(self):
        worker=self.worker(factory('P0'))
        try:
            self.begin(worker);self.feed(worker)
            first=self.wait_result(worker)
            row,arrays,association,scene=observation(11.2)
            self.assertTrue(worker.submit(row,arrays,scene=scene,image_metadata=None,association=association))
            second=self.wait_result(worker)
            self.assertLess(second['prepared']['source_stamp_s']-first['prepared']['source_stamp_s'],2.)
        finally:worker.close()

    def test_idle_candidate_preview_does_not_start_model_request(self):
        published=[]
        worker=FollowDecisionWorker(search,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
            inference=lambda _:self.fail('preview must not call the model'),
            publish_agent_frame=lambda *args:published.append(args))
        try:
            self.feed(worker)
            deadline=time.monotonic()+3
            while not published and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(published);self.assertIsNone(published[0][0])
            self.assertIsNone(worker.poll())
        finally:worker.close()

    def test_yolo_warms_before_a_three_frame_window_and_reuses_worker(self):
        warmed=threading.Event()
        created=[]
        detector=object()
        def factory():
            created.append(detector)
            warmed.set()
            return detector
        worker=FollowDecisionWorker(search,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
            image_preparer_factory=factory,publish_agent_frame=lambda *_:None,
            inference=lambda _:self.fail('warmup must not call the model'))
        try:
            row,arrays,association,scene=observation(10.)
            self.assertTrue(worker.submit(row,arrays,scene=scene,association=None,
                image_metadata=dict(rgb_stamp_ns=association['image_input']['rgb_stamp_ns'])))
            self.assertTrue(warmed.wait(3.))
            self.assertEqual(created,[detector])
            self.assertIsNone(worker.poll())
        finally:worker.close()

    def test_recorded_request_matches_input_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=FollowDecisionWorker(search,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                inference=lambda request:request.infer(client_factory=factory('P0')),
                record_directory=Path(directory)/'t2')
            try:
                self.begin(worker);self.feed(worker)
                outcome=self.wait_result(worker)
            finally:worker.close()
            paths=list((Path(directory)/'t2').glob('*.request.json'))
            results=list((Path(directory)/'t2').glob('*.result.json'))
            self.assertEqual(len(paths),1);self.assertEqual(len(results),1)
            request=json.loads(paths[0].read_text())
            self.assertEqual(request['prepared'],outcome['prepared'])
            self.assertEqual(request['transport_body'],outcome['prepared']['body'])
            self.assertEqual(len(list((Path(directory)/'t2').glob('*.png'))),4)
            self.assertEqual(json.loads(results[0].read_text()),outcome['result'])
            self.assertEqual(json.loads((Path(directory)/'t2/recording_summary.json').read_text()),
                             dict(dropped=0,error=None))

    def test_hosted_jpeg_records_exact_transport_images(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=FollowDecisionWorker(search,backend='hosted',jpeg_quality=70,
                buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                inference=lambda request:request.infer(client_factory=factory('P0')),
                record_directory=Path(directory)/'t2')
            try:
                self.begin(worker);self.feed(worker)
                outcome=self.wait_result(worker)
                self.assertEqual(outcome['result']['status'],'complete')
            finally:worker.close()
            self.assertEqual(len(list((Path(directory)/'t2').glob('*.jpg'))),4)
            self.assertEqual(json.loads((Path(directory)/'t2/recording_summary.json').read_text()),
                             dict(dropped=0,error=None))

    def test_dspark_jpeg_80_preserves_reference_and_exact_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=FollowDecisionWorker(search,backend='dspark',jpeg_quality=80,
                buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                inference=lambda request:request.infer(client_factory=factory('P0')),
                record_directory=Path(directory)/'t2')
            try:
                self.begin(worker);self.feed(worker)
                outcome=self.wait_result(worker)
                self.assertEqual(outcome['result']['status'],'complete')
                self.assertEqual(outcome['prepared']['reference_id'],self.reference['reference_id'])
            finally:worker.close()
            request=json.loads(next((Path(directory)/'t2').glob('*.request.json')).read_text())
            pictures=[p['image_url']['url'] for m in request['transport_body']['messages']
                      if isinstance(m['content'],list) for p in m['content'] if p.get('type')=='image_url']
            self.assertEqual(len(pictures),4)
            self.assertTrue(all(p.startswith('data:image/jpeg;base64,') for p in pictures))
            self.assertEqual(len(list((Path(directory)/'t2').glob('*.jpg'))),4)

    def test_raw_follow_uses_no_detector_and_preserves_3d_track_identity(self):
        def unavailable_detector():self.fail('T2 raw mode must not load YOLO')
        worker=FollowDecisionWorker(search,backend='dspark',jpeg_quality=80,raw_rgb=True,
            image_preparer_factory=unavailable_detector,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
            inference=lambda request:request.infer(client_factory=factory('P0')))
        try:
            self.begin(worker)
            for i in range(12):
                row,arrays,association,scene=observation(round(10+i/10,1))
                worker.submit(row,arrays,scene=scene,association=None,
                    image_metadata=dict(rgb_stamp_ns=association['image_input']['rgb_stamp_ns'],rgb_frame='camera'))
            outcome=self.wait_result(worker)
            self.assertEqual(outcome['result']['status'],'complete',outcome)
            self.assertEqual(outcome['result']['selection'],'P0')
            self.assertEqual(outcome['window']['image_mode'],'raw_rgb_with_separate_bev')
            self.assertEqual(len(outcome['window']['images']),6)
            self.assertTrue(all(s['association_sha256'] is None for s in outcome['window']['sources']))
            self.assertEqual(outcome['prepared']['choices']['P0']['kind'],'person')
        finally:worker.close()

    def test_raw_mode_previews_only_when_agent_requests_selection(self):
        requested=[False];published=[]
        worker=FollowDecisionWorker(search,raw_rgb=True,preview_requested=lambda:requested[0],
            buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),publish_agent_frame=lambda *args:published.append(args))
        try:
            self.feed(worker)
            self.assertFalse(published)
            requested[0]=True
            row,arrays,association,scene=observation(11.2)
            self.assertTrue(worker.submit(row,arrays,scene=scene,image_metadata=None,association=association))
            deadline=time.monotonic()+3
            while not published and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue(published)
        finally:worker.close()

    def test_outer_agent_receives_same_t2_candidate_and_returns_to_ordinary_rgb_after_stop(self):
        from decision.agent.adapters.runtime import follow_observation
        from runtime.command import AgentState
        state=AgentState()
        state.control=dict(clock_domain='synthetic',stamp_s=11.2)
        state.publish_observation({'source':'ordinary'},b'ordinary')
        state.publish_follow(dict(status='starting',handle=self.handle))
        def publish(handle,scene,pixels,association,camera_info):
            frame=follow_observation(scene,pixels,association,None,camera_info)
            state.publish_observation(frame.record,frame.image,follow_handle=handle)
        worker=FollowDecisionWorker(search,buffer=DecisionBuffer(max_frames=16,slot_bytes=32768),
                                    inference=lambda req:req.infer(client_factory=factory('P0')),publish_agent_frame=publish)
        try:
            self.begin(worker)
            for i in range(12):
                row,pixels,assoc,scene=observation(round(10+i/10,1))
                assoc['image_input']['camera_frame']='camera'
                assoc.update(detector_identity='test',detections=[dict(box=[20.,5.,35.,40.],score=.9,class_id=0)])
                assoc['result']['tracks']['a/g7']['detection_index']=0
                worker.submit(row,pixels,association=assoc,scene=scene,image_metadata=None)
            self.assertEqual(self.wait_result(worker)['result']['choice']['kind'],'person')
            current=state.snapshot(with_image=True)['observation']
            self.assertEqual(current['follow_candidates']['candidates'][0]['track']['identity'],['tracker_session',7,'a'])
            self.assertEqual(current['follow_candidates']['rgb_stamp_ns'],current['rgb']['stamp_ns'])
            state.publish_follow(dict(status='stopped',handle=self.handle))
            state.publish_observation(current,b'late',follow_handle=self.handle)
            self.assertEqual(state.snapshot()['observation'],{'source':'ordinary'})
        finally:worker.close()

    def test_cancelled_result_is_discarded_and_cannot_adopt_reference(self):
        gate=threading.Event();worker=self.worker(factory('P0',gate))
        try:
            self.begin(worker);self.feed(worker)
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
            self.begin(worker);self.feed(worker)
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
