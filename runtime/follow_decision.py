"""Optional causal T2 observation preparation and single-flight model worker."""
import copy
import base64
import hashlib
import json
import queue
import threading
from pathlib import Path

from PIL import Image

from decision.vlm.annotation.store import perception_view
from decision.vlm.follow_online import FollowRequest, prepare
from decision.vlm.observations import render_window
from perception.environment.context import fingerprint
from .decision import DecisionBuffer
from .export import drain_worker


class FollowRequestRecorder:
    """Bounded asynchronous storage of exact T2 inputs and returned results."""
    def __init__(self, directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.queue=queue.Queue(4);self.dropped=0;self.error=None;self.sequence=0
        self.thread=threading.Thread(target=self._work,name='t2-request-recorder',daemon=True);self.thread.start()

    def submit(self, binding, kind, value):
        self.sequence+=1
        try:self.queue.put_nowait((self.sequence,binding,kind,value))
        except queue.Full:self.dropped+=1

    def _work(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:return
                sequence,binding,kind,value=item
                with (self.directory/(f'{sequence:06d}.'+binding+'.'+kind+'.json')).open('x',encoding='utf-8') as stream:
                    json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
                if kind=='request':
                    index=0
                    for message in value['transport_body']['messages']:
                        content=message.get('content')
                        if not isinstance(content,list):continue
                        for part in content:
                            if not isinstance(part,dict) or part.get('type')!='image_url':continue
                            url=part['image_url']['url']
                            formats={'data:image/png;base64,':'.png','data:image/jpeg;base64,':'.jpg'}
                            prefix=next((p for p in formats if url.startswith(p)),None)
                            if prefix is None:raise ValueError('Unexpected recorded image format')
                            index+=1
                            (self.directory/(f'{sequence:06d}.'+binding+f'.image{index}'+formats[prefix])).write_bytes(
                                base64.b64decode(url[len(prefix):],validate=True))
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        drain_worker(self.thread,self.queue)
        (self.directory/'recording_summary.json').write_text(json.dumps(dict(dropped=self.dropped,error=self.error),ensure_ascii=False)+'\n')


class FollowDecisionWorker:
    """Prepare from sealed frames off the perception thread; never command motion."""
    def __init__(self, search_factory, *, metrics=None, image_policy=None, camera_calibration=None,
                 image_preparer_factory=None, inference=None, buffer=None, publish_agent_frame=None,backend='dspark',record_directory=None,
                 jpeg_quality=None,raw_rgb=False,preview_requested=None):
        self.search_factory=search_factory
        self.buffer=buffer or DecisionBuffer(metrics=metrics)
        self.extent,self.view=perception_view()
        if image_preparer_factory is None:
            from decision.vlm.image_inputs import ImagePreparer
            image_preparer_factory=lambda:ImagePreparer(image_policy,camera_calibration)
        self.preparer_factory=image_preparer_factory;self.preparer=None
        if backend not in ('dspark','hosted'):raise ValueError('Unknown T2 backend')
        if jpeg_quality is not None and (type(jpeg_quality) is not int or not 1<=jpeg_quality<=95):
            raise ValueError('T2 JPEG quality must be an integer in 1..95')
        self.jpeg_quality=jpeg_quality
        self.raw_rgb=bool(raw_rgb)
        self.preview_requested=preview_requested
        self.backend=backend
        self.inference=inference or (lambda request:request.infer(backend=backend))
        self.lock=threading.Lock();self.queue=queue.Queue(1)
        self.epoch=0;self.task=None;self.result=None;self.inflight=False
        self.error=None;self.reference=None;self.crop=None
        self.next_preview_stamp=None
        self.closed=False
        self.publish_agent_frame=publish_agent_frame
        self.recorder=FollowRequestRecorder(record_directory) if record_directory is not None else None
        self.agent_frame_error=None
        self.thread=threading.Thread(target=self._work,name='follow-t2',daemon=True);self.thread.start()

    def begin(self, handle, description, *, reference=None, crop=None):
        if not isinstance(handle,dict) or not isinstance(description,str) or not description.strip():
            raise ValueError('Follow task and target description required')
        if (not isinstance(reference,dict) or not isinstance(crop,bytes) or
                reference.get('description')!=description.strip() or
                reference.get('reference_id')!=fingerprint({k:v for k,v in reference.items() if k!='reference_id'}) or
                reference.get('crop_sha256')!=hashlib.sha256(crop).hexdigest()):
            raise ValueError('Agent-confirmed follow reference required')
        with self.lock:
            if self.task is not None:raise ValueError('Follow T2 already active')
            self.epoch+=1;self.task=dict(handle=copy.deepcopy(handle),description=description.strip())
            self.result=None;self.error=None;self.reference=copy.deepcopy(reference);self.crop=crop
        self.buffer.reset()

    def stop(self):
        with self.lock:
            self.epoch+=1;self.task=None;self.result=None;self.error=None
            self.reference=self.crop=None

    def request_fresh(self):
        """The next available observation is scheduled after result consumption."""
        return None

    def submit(self, people, arrays, *, scene, image_metadata, context=None, association=None):
        if self.closed or not isinstance(people,dict) or scene is None:return False
        with self.lock:
            if self.task is None and self.publish_agent_frame is None:return False
        previous=self.buffer.scope
        pushed=self.buffer.push(people,arrays,association,scene=scene,image_metadata=image_metadata)
        stamp=people['stamp_s']
        with self.lock:
            preview=self.task is None
            if preview and self.preview_requested is not None and not self.preview_requested():return False
            if previous!=self.buffer.scope:self.next_preview_stamp=None
            if (not pushed or scene is None or self.error is not None or self.inflight or
                    self.result is not None):
                return False
            if preview and self.next_preview_stamp is not None and stamp<self.next_preview_stamp:return False
            lease=self.buffer.acquire()
            if lease is None:return False
            self.inflight=True
            if preview:self.next_preview_stamp=stamp+1.
            value=dict(epoch=self.epoch,preview=preview,task=copy.deepcopy(self.task),reference=copy.deepcopy(self.reference),
                crop=self.crop,scene=scene,context=copy.deepcopy(context),lease=lease)
            self.queue.put_nowait(value)
            return True

    def poll(self):
        with self.lock:
            value,self.result=self.result,None
            return value

    def snapshot(self):
        with self.lock:
            return dict(backend=self.backend,active=self.task is not None,inflight=self.inflight,error=self.error,
                        reference_fixed=self.reference is not None,agent_frame_error=self.agent_frame_error)

    def _work(self):
        while True:
            item=self.queue.get()
            if item is None:return
            lease=item['lease'];outcome=None;stage='initialization'
            if item['preview']:
                error=None
                try:
                    with self.buffer.consume(lease) as read:
                        entry=lease['entries'][-1]
                        if entry['association'] is None:
                            if self.preparer is None:self.preparer=self.preparer_factory()
                            entry=self.preparer.prepare([entry],lease['history'],read)[0]
                        with self.lock:active=item['epoch']==self.epoch and self.task is None
                        if active:
                            args=(None,item['scene'],read(entry['buffer'])['rgb'],entry['association'],
                                  (entry['image_metadata'] or {}).get('camera_info'))
                            self.publish_agent_frame(*args)
                except (ValueError,KeyError,TypeError) as caught:error=type(caught).__name__
                except Exception as caught:error=type(caught).__name__
                with self.lock:
                    self.inflight=False;self.agent_frame_error=error
                continue
            try:
                with self.buffer.consume(lease) as read:
                    entries=lease['entries']
                    if not self.raw_rgb and any(e['association'] is None for e in entries):
                        if self.preparer is None:self.preparer=self.preparer_factory()
                        stage='input'
                        entries=self.preparer.prepare(entries,lease['history'],read)
                    stage='input'
                    window,images=render_window(entries,lease['history'],read,extent=self.extent,
                        view_binding=self.view,image_preparation=None if self.preparer is None else self.preparer.metadata,
                        include_predicted=True,raw_rgb=self.raw_rgb)
                    search=self.search_factory(item['scene'],item['context'])
                    prepared=prepare(window,images,item['task']['description'],search,
                        reference_image=item['crop'],reference=item['reference'],task=item['task']['handle'],
                        jpeg_quality=self.jpeg_quality)
                    # Crop source pixels while their lease is held; inference must
                    # not retain shared views or block the observation history.
                    pixels=read(entries[-1]['buffer'])['rgb']
                    if self.publish_agent_frame is not None:
                        try:
                            args=(item['task']['handle'],item['scene'],pixels,entries[-1]['association'],
                                  (entries[-1]['image_metadata'] or {}).get('camera_info'))
                            if self.raw_rgb:self.publish_agent_frame(*args,image_metadata=entries[-1]['image_metadata'])
                            else:self.publish_agent_frame(*args)
                            self.agent_frame_error=None
                        except (ValueError,KeyError,TypeError) as error:
                            self.agent_frame_error=type(error).__name__
                stage='inference'
                if self.recorder is not None:
                    if self.backend=='hosted':
                        from decision.vlm.follow_hosted import hosted_body
                        transport_body=hosted_body(prepared['body'])
                    else:transport_body=prepared['body']
                    self.recorder.submit(prepared['binding'],'request',dict(backend=self.backend,
                        prepared=prepared,transport_body=transport_body))
                request=FollowRequest(prepared);result=self.inference(request)
                if self.recorder is not None:self.recorder.submit(prepared['binding'],'result',result)
                if result.get('status')=='complete':
                    request.accept(result)
                outcome=dict(handle=item['task']['handle'],prepared=prepared,window=window,result=result,
                             reference_proposal=None)
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
        drain_worker(self.thread, self.queue)
        self.buffer.close()
        if self.recorder is not None:self.recorder.close()
