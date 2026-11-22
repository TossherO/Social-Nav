"""Optional causal T2 observation preparation and single-flight model worker."""
import copy
import base64
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

from PIL import Image

from decision.vlm.annotation.store import perception_view
from decision.vlm.t2.online import FollowRequest, prepare
from decision.vlm.observations import render_window
from perception.environment.context import fingerprint
from .decision import DecisionBuffer
from .export import drain_worker


class FollowRequestRecorder:
    """Bounded asynchronous storage of VLM requests or local follow evidence."""
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
                 jpeg_quality=None,raw_rgb=False,preview_requested=None,camera_client=None,
                 reid_weights=None,reid_device=None,reid_factory=None):
        if backend not in ('dspark','hosted','reid'):raise ValueError('Unknown T2 backend')
        if backend=='reid' and (raw_rgb or jpeg_quality is not None or inference is not None):
            raise ValueError('Local ReID requires associated detections and does not use VLM request settings')
        if backend=='reid' and reid_factory is None and not reid_weights:
            raise ValueError('Local ReID requires explicit weights')
        self.search_factory=search_factory
        self.camera_client=camera_client
        self.buffer=buffer or DecisionBuffer(metrics=metrics,camera_lookup=camera_client.lookup if camera_client else None)
        self.metrics=metrics
        if backend!='reid':self.extent,self.view=perception_view()
        if image_preparer_factory is None:
            from decision.vlm.image_inputs import ImagePreparer
            image_preparer_factory=lambda:ImagePreparer(image_policy,camera_calibration)
        self.preparer_factory=image_preparer_factory;self.preparer=None
        self.warmup_attempted=False
        if jpeg_quality is not None and (type(jpeg_quality) is not int or not 1<=jpeg_quality<=95):
            raise ValueError('T2 JPEG quality must be an integer in 1..95')
        self.jpeg_quality=jpeg_quality
        self.raw_rgb=bool(raw_rgb)
        self.preview_requested=preview_requested
        self.backend=backend
        self.inference=inference or (lambda request:request.infer(backend=backend))
        self.reid_weights=reid_weights
        self.reid_device=reid_device
        self.reid_factory=reid_factory
        self.reid_encoder=self.reid=self.reid_search=None
        self.reid_model=None
        self.reid_epoch=None
        self.reid_interrupted=False
        if backend=='reid':
            from navigation.planning.inputs import load_config
            self.reid_planning=load_config()
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
        if self.backend=='reid' and (reference.get('task_id')!=handle.get('task_id') or
                reference.get('task_revision')!=handle.get('plan_revision') or reference.get('goal_id')!=handle.get('goal_id')):
            raise ValueError('Local ReID reference does not belong to this follow task')
        with self.lock:
            if self.task is not None:raise ValueError('Follow T2 already active')
            self.epoch+=1;self.task=dict(handle=copy.deepcopy(handle),description=description.strip())
            self.result=None;self.error=None;self.reference=copy.deepcopy(reference);self.crop=crop
            self.reid_interrupted=False
            self.buffer.reset()

    def stop(self):
        with self.lock:
            self.epoch+=1;self.task=None;self.result=None;self.error=None
            self.reference=self.crop=None
            self.reid_interrupted=False

    def request_fresh(self):
        """The next available observation is scheduled after result consumption."""
        return None

    def interrupt_reid(self):
        """Require fresh consecutive confirmations after a rejected or missing source."""
        with self.lock:
            if self.backend=='reid':
                self.reid_interrupted=True
                self.result=None

    def submit(self, people, arrays, *, scene, image_metadata, context=None, association=None):
        if self.closed or not isinstance(people,dict):return False
        with self.lock:
            if self.task is None and self.publish_agent_frame is None:return False
            epoch=self.epoch
        previous=self.buffer.scope
        pushed=self.buffer.push(people,arrays,association,scene=scene,image_metadata=image_metadata,
                                keep_association_history=True)
        stamp=people['stamp_s']
        with self.lock:
            if epoch!=self.epoch:return False
            preview=self.task is None
            if preview and self.preview_requested is not None and not self.preview_requested():return False
            if previous!=self.buffer.scope:self.next_preview_stamp=None
            if (not pushed or scene is None or self.error is not None or self.inflight or
                    self.result is not None):
                return False
            if (preview and association is None and self.preparer is None and
                    not self.warmup_attempted and scene.record['localization']['status']=='valid'):
                self.warmup_attempted=True
                self.inflight=True
                self.queue.put_nowait(dict(warmup=True))
                return True
            if preview and self.next_preview_stamp is not None and stamp<self.next_preview_stamp:return False
            lease=self.buffer.acquire(latest_only=self.backend=='reid')
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
            status=dict(backend=self.backend,active=self.task is not None,inflight=self.inflight,error=self.error,
                        reference_fixed=self.reference is not None,agent_frame_error=self.agent_frame_error,
                        camera_pair_error=self.camera_client.error if self.camera_client else None)
            if self.backend=='reid':
                status.update(reid_model=copy.deepcopy(self.reid_model),
                              reid_model_sha256=self.reid_model['sha256'] if self.reid_model else None)
            return status

    def _publish_agent_observation(self, item, entry, pixels, **options):
        try:
            with self.lock:
                if item['epoch'] != self.epoch or item['preview'] != (self.task is None):
                    return
                handle = None if item['preview'] else item['task']['handle']
                self.publish_agent_frame(handle,item['scene'],pixels,entry['association'],
                                         (entry['image_metadata'] or {}).get('camera_info'),**options)
                self.agent_frame_error = None
        except (ValueError,KeyError,TypeError) as error:
            self.agent_frame_error = type(error).__name__

    def _work_reid(self,item):
        from .follow_reid import current_candidates, current_track_positions, search_action
        from navigation.planning.people import PeopleModel, stable_identity

        handle=item['task']['handle']
        scene=item['scene']
        lease=item['lease']
        stamp=scene.record['as_of']
        started=time.monotonic()
        stage='input'
        try:
            with self.buffer.consume(lease,latest_only=True) as read:
                entry=lease['entries'][-1]
                if entry['association'] is None:
                    if 'rgb' not in dict(entry['buffer'].arrays):raise ValueError('follow_reid_rgb_unavailable')
                    if self.preparer is None:
                        stage='initialization'
                        self.preparer=self.preparer_factory()
                        stage='input'
                    entry=self.preparer.prepare([entry],lease['association_history'],read)[0]
                pixels=read(entry['buffer'])['rgb'].copy()
            source=current_candidates(scene,entry,pixels)
            people=PeopleModel(scene,self.reid_planning['people'])
            positions=current_track_positions(source,people)
            observed=None
            if people.status=='valid' and people.origin==source['stamp_s']:
                data=scene.record['people']['data']
                observed={(stable_identity(data,track),track['lifecycle_generation']) for track in data['tracks']
                          if track['confirmed'] and track['state']=='observed' and
                          track['last_observed_stamp_s']==people.origin}
            localization=scene.record['localization']
            pose_xy=None
            if localization['status']=='valid':
                matrix=localization['world_from_lidar']
                pose_xy=(matrix[0][3],matrix[1][3])
            with self.lock:
                if item['epoch']!=self.epoch or self.task is None:return None
                interrupted=self.reid_interrupted
                self.reid_interrupted=False
            if self.publish_agent_frame is not None:
                self._publish_agent_observation(item,entry,pixels,bound_candidates=source)
            stage='matcher'
            if self.reid_epoch!=item['epoch']:
                from decision.osnet.search import FollowSearch
                if self.reid_factory is None:
                    from decision.osnet.reid import FollowReid, OsnetEmbedder
                    if self.reid_encoder is None:
                        self.reid_encoder=OsnetEmbedder(self.reid_weights,device=self.reid_device)
                        encoder=self.reid_encoder
                        self.reid_model=dict(backend=encoder.backend,path=str(encoder.path),sha256=encoder.sha256,
                                             onnx_sha256=encoder.onnx_sha256,device=encoder.device,
                                             tensorrt_version=encoder.tensorrt_version)
                        if self.metrics is not None:
                            self.metrics.trace('t2_reid_model_loaded',reid_model=self.reid_model)
                    self.reid=FollowReid(self.reid_encoder,item['crop'],item['reference'])
                else:self.reid=self.reid_factory(item['crop'],item['reference'])
                self.reid_search=FollowSearch()
                self.reid_epoch=item['epoch']
            if interrupted:
                self.reid.interrupt();self.reid_search.defer()
            decision=self.reid.step(pixels,source,track_positions=positions)
            stage='input'
            context=item['context'] or {}
            last=context.get('last')
            loss=self.reid_search.observe(source,decision,last=last,positions=positions,pose_xy=pose_xy,
                observed=observed,association_tracks=entry['association']['result']['tracks'],policy=self.reid.policy)
            decision['search_eligible']=loss['search_ready']
            search=selected=continuation=None
            if decision['search_eligible']:
                search=self.search_factory(scene,dict(context,local_reid=True,loss=loss,
                                                      continuation=self.reid_search.continuation))
                selected=self.reid_search.choose(search,last=last,scene=scene)
                search_action(search,source,selected)
                if selected is not None and selected['kind']!='wait':
                    continuation=copy.deepcopy(self.reid_search.continuation)
            evidence=dict(task=handle,reference_id=item['reference']['reference_id'],source=source,
                          decision=decision,loss=loss,search=search,selected=selected,continuation=continuation,
                          reid_model=self.reid_model,
                          reid_model_sha256=self.reid_model['sha256'] if self.reid_model else None)
            evidence['binding']=fingerprint(evidence)
            elapsed=time.monotonic()-started
            if self.metrics is not None:
                self.metrics.trace('t2_reid_done',measurement_stamp_s=stamp,rgb_stamp_ns=source['rgb_stamp_ns'],
                                   elapsed_s=elapsed,status=decision['status'],reason=decision['reason'],loss=loss)
            if self.recorder is not None:
                self.recorder.submit(evidence['binding'],'reid_observation',dict(evidence,elapsed_s=elapsed))
            return dict(handle=handle,local=evidence,timing=dict(elapsed_s=elapsed))
        except Exception as error:
            if self.reid_epoch==item['epoch']:
                self.reid.interrupt();self.reid_search.defer()
            return dict(handle=handle,error=type(error).__name__,
                        reason=str(error),transient=stage=='input' and isinstance(error,(ValueError,KeyError,TypeError)))

    def _work(self):
        while True:
            item=self.queue.get()
            if item is None:return
            if item.get('warmup'):
                error=None
                if self.metrics is not None:self.metrics.trace('t2_yolo_warmup_start')
                try:self.preparer=self.preparer_factory()
                except Exception as caught:error=type(caught).__name__
                if self.metrics is not None:self.metrics.trace('t2_yolo_warmup_done',error=error)
                with self.lock:
                    self.inflight=False;self.agent_frame_error=error
                continue
            lease=item['lease'];outcome=None
            if item['preview']:
                error=None
                try:
                    with self.buffer.consume(lease,latest_only=True) as read:
                        entry=lease['entries'][-1]
                        if entry['association'] is None:
                            if self.preparer is None:self.preparer=self.preparer_factory()
                            entry=self.preparer.prepare([entry],lease['association_history'],read)[0]
                        pixels=read(entry['buffer'])['rgb'].copy()
                        if self.metrics is not None:
                            self.metrics.transfer('host_copy',pixels.nbytes,'agent_follow_rgb')
                        self._publish_agent_observation(item,entry,pixels)
                except (ValueError,KeyError,TypeError) as caught:error=type(caught).__name__
                except Exception as caught:error=type(caught).__name__
                with self.lock:
                    self.inflight=False
                    if error is not None:self.agent_frame_error=error
                continue
            if self.backend=='reid':
                outcome=self._work_reid(item)
                with self.lock:
                    self.inflight=False
                    if (outcome is not None and item['epoch']==self.epoch and self.task is not None and
                            not self.reid_interrupted):
                        self.result=outcome
                        if outcome.get('error') and not outcome.get('transient'):
                            self.error=outcome['error']
                continue
            try:
                stage='input'
                with self.buffer.consume(lease) as read:
                    entries=lease['entries']
                    stamp=entries[-1]['people']['stamp_s']
                    if self.metrics is not None:self.metrics.trace('t2_prepare_start',measurement_stamp_s=stamp)
                    if not self.raw_rgb and any(e['association'] is None for e in entries):
                        if self.preparer is None:
                            stage='initialization'
                            self.preparer=self.preparer_factory()
                            stage='input'
                        entries=self.preparer.prepare(entries,lease['association_history'],read)
                        if self.metrics is not None and self.metrics.trace_path is not None:
                            association=entries[-1]['association']
                            result=association['result']
                            self.metrics.trace('t2_rgb_association',measurement_stamp_s=stamp,
                                rgb_stamp_ns=association['image_input']['rgb_stamp_ns'],
                                detections=len(association['detections']),matches=result['matches'],
                                tracks={identity:{key:track[key] for key in
                                         ('status','compensation','geometry_stamp_s','pose_age_s') if key in track}
                                        for identity,track in result['tracks'].items()})
                    window,images=render_window(entries,lease['history'],read,extent=self.extent,
                        view_binding=self.view,image_preparation=None if self.preparer is None else self.preparer.metadata,
                        include_predicted=True,raw_rgb=self.raw_rgb,in_memory=True)
                    if self.metrics is not None:self.metrics.trace('t2_render_done',measurement_stamp_s=stamp)
                    search=self.search_factory(item['scene'],item['context'])
                    prepared=prepare(window,images,item['task']['description'],search,
                        reference_image=item['crop'],reference=item['reference'],task=item['task']['handle'],
                        jpeg_quality=self.jpeg_quality)
                    if self.publish_agent_frame is not None:
                        # The Agent retains only its owned current frame after the lease is released.
                        entry=entries[-1]
                        pixels=read(entry['buffer'])['rgb'].copy()
                        if self.metrics is not None:
                            self.metrics.transfer('host_copy',pixels.nbytes,'agent_follow_rgb')
                        options=dict(image_metadata=entry['image_metadata']) if self.raw_rgb else {}
                        self._publish_agent_observation(item,entry,pixels,**options)
                stage='inference'
                if self.recorder is not None:
                    if self.backend=='hosted':
                        from decision.vlm.t2.hosted import hosted_body
                        transport_body=hosted_body(prepared['body'])
                    else:transport_body=prepared['body']
                    self.recorder.submit(prepared['binding'],'request',dict(backend=self.backend,
                        prepared=prepared,transport_body=transport_body))
                request=FollowRequest(prepared)
                if self.metrics is not None:self.metrics.trace('t2_request_start',measurement_stamp_s=stamp)
                result=self.inference(request)
                if self.metrics is not None:self.metrics.trace('t2_request_done',measurement_stamp_s=stamp)
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
        if self.camera_client is not None:self.camera_client.close()
        if self.recorder is not None:self.recorder.close()
