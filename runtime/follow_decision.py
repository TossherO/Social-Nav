"""Optional causal T2 observation preparation and single-flight model worker."""
import copy
import hashlib
import io
import queue
import threading

from PIL import Image

from decision.vlm.annotation.store import perception_view
from decision.vlm.follow_online import FollowRequest, prepare
from decision.vlm.observations import render_window
from decision.vlm.project_config import load_config
from perception.environment.context import fingerprint
from .decision import DecisionBuffer


class FollowDecisionWorker:
    """Prepare from sealed frames off the perception thread; never command motion."""
    def __init__(self, search_factory, *, metrics=None, image_policy=None, camera_calibration=None,
                 image_preparer_factory=None, inference=None, buffer=None):
        self.search_factory=search_factory
        self.buffer=buffer or DecisionBuffer(metrics=metrics)
        self.extent,self.view=perception_view()
        self.period=load_config()['inference']['period_s']
        if image_preparer_factory is None:
            from decision.vlm.image_inputs import ImagePreparer
            image_preparer_factory=lambda:ImagePreparer(image_policy,camera_calibration)
        self.preparer_factory=image_preparer_factory;self.preparer=None
        self.inference=inference or (lambda request:request.infer())
        self.lock=threading.Lock();self.queue=queue.Queue(1)
        self.epoch=0;self.task=None;self.result=None;self.inflight=False
        self.next_stamp=None;self.last_stamp=None;self.error=None;self.reference=None;self.crop=None
        self.closed=False
        self.thread=threading.Thread(target=self._work,name='follow-t2',daemon=True);self.thread.start()

    def begin(self, handle, description):
        if not isinstance(handle,dict) or not isinstance(description,str) or not description.strip():
            raise ValueError('Follow task and target description required')
        with self.lock:
            if self.task is not None:raise ValueError('Follow T2 already active')
            self.epoch+=1;self.task=dict(handle=copy.deepcopy(handle),description=description.strip())
            self.result=None;self.next_stamp=None;self.error=None;self.reference=self.crop=None
        self.buffer.reset()

    def stop(self):
        with self.lock:
            self.epoch+=1;self.task=None;self.result=None;self.next_stamp=None
            self.reference=self.crop=None

    def request_fresh(self):
        with self.lock:
            self.next_stamp=None

    def adopt_reference(self, handle, proposal):
        """Fix the first accepted image; later identity updates cannot replace it."""
        with self.lock:
            if self.task is None or self.task['handle']!=handle or self.reference is not None:
                return False
            if (not isinstance(proposal,dict) or not isinstance(proposal.get('crop'),bytes) or
                    proposal['record'].get('crop_sha256')!=hashlib.sha256(proposal['crop']).hexdigest() or
                    proposal['record'].get('reference_id')!=fingerprint({k:v for k,v in proposal['record'].items() if k!='reference_id'})):
                raise ValueError('Invalid visual reference')
            self.reference=copy.deepcopy(proposal['record']);self.crop=proposal['crop']
            return True

    def submit(self, people, arrays, *, scene, image_metadata, context=None, association=None):
        if self.closed:return False
        with self.lock:
            if self.task is None:return False
        previous=self.buffer.scope
        pushed=self.buffer.push(people,arrays,association,scene=scene,image_metadata=image_metadata)
        stamp=people['stamp_s']
        with self.lock:
            if previous!=self.buffer.scope or self.last_stamp is not None and stamp<=self.last_stamp:
                self.next_stamp=None
            self.last_stamp=stamp
            if (not pushed or scene is None or self.task is None or self.error is not None or self.inflight or
                    self.result is not None or self.next_stamp is not None and stamp<self.next_stamp):
                return False
            lease=self.buffer.acquire()
            if lease is None:return False
            self.inflight=True;self.next_stamp=stamp+self.period
            value=dict(epoch=self.epoch,task=copy.deepcopy(self.task),reference=copy.deepcopy(self.reference),
                crop=self.crop,scene=scene,context=copy.deepcopy(context),lease=lease)
            self.queue.put_nowait(value)
            return True

    def poll(self):
        with self.lock:
            value,self.result=self.result,None
            return value

    def snapshot(self):
        with self.lock:
            return dict(active=self.task is not None,inflight=self.inflight,error=self.error,
                        reference_fixed=self.reference is not None)

    def _work(self):
        while True:
            item=self.queue.get()
            if item is None:return
            lease=item['lease'];outcome=None;stage='initialization'
            try:
                with self.buffer.consume(lease) as read:
                    entries=lease['entries']
                    if any(e['association'] is None for e in entries):
                        if self.preparer is None:self.preparer=self.preparer_factory()
                        stage='input'
                        entries=self.preparer.prepare(entries,lease['history'],read)
                    stage='input'
                    window,images=render_window(entries,lease['history'],read,extent=self.extent,
                        view_binding=self.view,image_preparation=None if self.preparer is None else self.preparer.metadata,
                        include_predicted=True)
                    search=self.search_factory(item['scene'],item['context'])
                    prepared=prepare(window,images,item['task']['description'],search,
                        reference_image=item['crop'],reference=item['reference'],task=item['task']['handle'])
                    # Crop source pixels while their lease is held; inference must
                    # not retain shared views or block the observation history.
                    pixels=read(entries[-1]['buffer'])['rgb']
                    association=entries[-1]['association']['result']['tracks']
                    inverse={v:k for k,v in window['mapping'].items()};crops={}
                    for letter,choice in prepared['choices'].items():
                        if choice['kind']!='person':continue
                        box=association[inverse[choice['pid']]].get('rgb_box')
                        if box is None:continue
                        import math
                        picture=Image.fromarray(pixels).crop((math.floor(box[0]),math.floor(box[1]),
                                                              math.ceil(box[2]),math.ceil(box[3])))
                        out=io.BytesIO();picture.save(out,format='PNG');crops[letter]=out.getvalue()
                stage='inference'
                request=FollowRequest(prepared);result=self.inference(request)
                if result.get('status')=='complete':
                    request.accept(result)
                proposal=None
                if (result.get('status')=='complete' and result.get('choice',{}).get('kind')=='person' and
                        item['reference'] is None):
                    crop=crops.get(result['selection'])
                    if crop is None:raise ValueError('Selected visual crop unavailable')
                    record=dict(description=item['task']['description'],clock_domain=window['clock_domain'],
                        rgb_stamp_s=window['sources'][-1]['rgb_input']['rgb_stamp_ns']/1e9,
                        scene_binding=prepared['scene_binding'],window_binding=prepared['window_binding'],
                        crop_sha256=hashlib.sha256(crop).hexdigest())
                    record['reference_id']=fingerprint(record)
                    proposal=dict(record=record,crop=crop)
                outcome=dict(handle=item['task']['handle'],prepared=prepared,window=window,result=result,
                             reference_proposal=proposal)
            except Exception as error:
                outcome=dict(handle=item['task']['handle'],error=type(error).__name__,
                             transient=stage=='input' and isinstance(error,(ValueError,KeyError,TypeError)))
            with self.lock:
                self.inflight=False
                if item['epoch']!=self.epoch or self.task is None:continue
                self.result=outcome
                if not outcome.get('transient') and (outcome.get('error') or outcome.get('result',{}).get('status')!='complete'):
                    self.error=outcome.get('error') or outcome['result'].get('reason','t2_request_failed')

    def close(self):
        self.closed=True;self.stop()
        self.queue.put(None);self.thread.join(timeout=20)
        if self.thread.is_alive():raise RuntimeError('T2 worker did not stop')
        self.buffer.close()
