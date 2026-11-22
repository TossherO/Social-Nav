"""Bounded decision observation leases and optional image recording off the hot path."""
from collections import deque
from contextlib import contextmanager
import copy
import math
import queue
import threading
from pathlib import Path
import uuid

from interfaces.shared import SharedArrayPool, BufferFull
from interfaces.runtime import source_stamp_ns
from decision.vlm.trajectory_inputs import local_scope, pose, HISTORY_S
from perception.sensors.image import matching_camera_info
from .export import drain_worker


ASSOCIATION_POSE_FIELDS = ('sequence_id','session_id','generation','clock_domain','frame_id','pose_source',
                           'localization_binding','health','pose_health','stamp_s','pose_stamp_s','world_from_lidar')
ASSOCIATION_TRACK_FIELDS = ('track_id','lifecycle_generation','confirmed','box','velocity_xyz_mps')


class DecisionBuffer:
    def __init__(self,*,max_frames=48,slot_bytes=4*1024*1024,metrics=None,camera_lookup=None):
        self.camera_lookup=camera_lookup
        self.pool=SharedArrayPool(max_frames+(6 if camera_lookup else 3),slot_bytes);self.reader=self.pool.reader()
        self.frames=deque();self.lock=threading.RLock();self.scope=None;self.metrics=metrics
        self.max_frames=max_frames;self.active=None;self.dropped=0
        self.association_history=deque(maxlen=max_frames)

    def _clear(self):
        while self.frames:self.pool.release(self.frames.popleft()['buffer'],'history')
        self.association_history.clear()

    def reset(self):
        """Discard indexed history while existing consumer leases remain valid."""
        with self.lock:
            self._clear();self.scope=None

    def push(self,people,arrays,association,*,scene=None,image_metadata=None,keep_association_history=False):
        with self.lock:
            scope=local_scope(people);stamp=people['stamp_s']
            if not math.isfinite(stamp):raise ValueError('Finite decision observation time required')
            stamp_ns=source_stamp_ns(stamp,people.get('stamp_ns'))
            valid_pose=pose(people) is not None
            latest=self.association_history if self.association_history else self.frames
            if self.scope!=scope or not valid_pose or (latest and stamp_ns<=latest[-1]['stamp_ns']):
                self._clear();self.scope=scope
            if not valid_pose:return False
            if keep_association_history:
                while (self.association_history and
                        stamp-self.association_history[0]['result']['stamp_s']>HISTORY_S+1.15):
                    self.association_history.popleft()
                compact={key:people.get(key) for key in ASSOCIATION_POSE_FIELDS}
                compact['tracks']=[{key:track.get(key) for key in ASSOCIATION_TRACK_FIELDS}
                                   for track in people.get('tracks',[])]
                self.association_history.append(dict(stamp_ns=stamp_ns,result=copy.deepcopy(compact)))
                # A scan without a same-source environment scene supplies pose
                # history only. Never copy its RGB or point cloud into T2.
                if scene is None:return False
            if 'raw' not in arrays:return False
            while self.frames and (stamp-self.frames[0]['people']['stamp_s']>HISTORY_S+1.15 or len(self.frames)>=self.max_frames):
                self.pool.release(self.frames.popleft()['buffer'],'history')
            values={k:arrays[k] for k in ('raw','rgb') if k in arrays}
            try:ref=self.pool.publish(values,readers=['history'])
            except BufferFull:self.dropped+=1;return False
            if self.metrics:self.metrics.transfer('host_copy',sum(a.nbytes for a in values.values()),'decision_observation_history')
            record=None if scene is None else scene.record
            if record is not None and (abs(record['as_of']-stamp)>1e-6 or
                    (record.get('stamp_ns') is not None and people.get('stamp_ns') is not None and record['stamp_ns']!=stamp_ns) or
                    record['clock_domain']!=people['clock_domain'] or
                    record.get('people',{}).get('data',{}).get('input_fingerprint')!=people['input_fingerprint']):
                self.pool.release(ref,'history');raise ValueError('Decision scene is not this people observation')
            ident=f"{people['session_id']}:{people['generation']}:{people['observation_id']}"
            self.frames.append(dict(id=ident,stamp_ns=stamp_ns,people=copy.deepcopy(people),association=copy.deepcopy(association),
                image_metadata=copy.deepcopy(image_metadata),scene=copy.deepcopy(record),buffer=ref))
            return True

    def acquire(self, *, latest_only=False):
        """One in-flight request; no queue of outdated observations."""
        with self.lock:
            if self.active is not None or not self.frames:return None
            if latest_only:
                token=uuid.uuid4().hex
                entry=self.frames[-1]
                self.pool.retain(entry['buffer'],token)
                self.active=token
                return dict(token=token,entries=(entry,),association_history=tuple(self.association_history))
            end=self.frames[-1]['stamp_ns'];selected=[]
            for delta in (-1_000_000_000,-500_000_000,0):
                past=[e for e in self.frames if e['stamp_ns']<=end+delta]
                if not past or end+delta-past[-1]['stamp_ns']>150_000_000:return None
                entry=past[-1]
                image=(entry['association'].get('image_input') or {}) if entry['association'] is not None else (entry['image_metadata'] or {})
                rgb_ns=image.get('rgb_stamp_ns')
                if entry['association'] is not None or self.camera_lookup is None:
                    if ('rgb' not in dict(entry['buffer'].arrays) or type(rgb_ns) is not int or
                            not 0<=entry['stamp_ns']-rgb_ns<=150_000_000):return None
                if entry['association'] is not None and not image.get('raw_rgb_sha256'):return None
                selected.append(entry)
            if len({e['id'] for e in selected})!=3:return None
            token=uuid.uuid4().hex
            for entry in selected:self.pool.retain(entry['buffer'],token)
            history=[dict(id=e['id'],stamp_ns=e['stamp_ns'],result=e['people']) for e in self.frames]
            self.active=token
            return dict(token=token,entries=tuple(selected),history=history,
                        association_history=tuple(self.association_history))

    def _pair_follow_frame(self, entry, token):
        from .follow_camera import PAIR_AGE_NS
        if entry['association'] is not None:
            return entry
        meta = entry['image_metadata'] or {}
        rgb = dict(entry['buffer'].arrays).get('rgb')
        rgb_ns = meta.get('rgb_stamp_ns')
        stamp = entry['stamp_ns']
        age = stamp - rgb_ns if rgb is not None and type(rgb_ns) is int else None
        if (age is not None and 0 <= age <= PAIR_AGE_NS and
                matching_camera_info(meta.get('camera_info'), meta.get('rgb_frame'), rgb_ns, rgb.shape)):
            return entry
        # A scene that already attests specific pixels must keep that evidence.
        bound_rgb = ((entry['scene'] or {}).get('rgb') or {}).get('status') == 'valid'
        pair = None if bound_rgb else self.camera_lookup(stamp)
        if pair is not None:
            pixels, metadata = pair
            raw = self.reader.arrays(entry['buffer'])['raw']
            try:
                with self.lock:
                    ref = self.pool.publish(dict(raw=raw, rgb=pixels), readers=[token])
                return dict(entry, buffer=ref, image_metadata=metadata)
            except BufferFull:
                pass
        if age is not None and 0 <= age <= 150_000_000:
            return entry
        raise ValueError('T2 camera observation unavailable')

    @contextmanager
    def consume(self,lease,*,latest_only=False):
        """Array views must not escape this scope. Release also on render failure."""
        if lease['token']!=self.active:raise ValueError('Unknown decision lease')
        original=lease['entries']
        extra=[]
        try:
            if self.camera_lookup is not None:
                entries=list(original)
                for i in range(len(entries)-1 if latest_only else 0, len(entries)):
                    entries[i] = self._pair_follow_frame(original[i], lease['token'])
                    if entries[i] is not original[i]:
                        extra.append(entries[i]['buffer'])
                lease['entries']=tuple(entries)
            yield self.reader.arrays
        finally:
            with self.lock:
                lease['entries']=original
                for ref in extra:self.pool.release(ref,lease['token'])
                for entry in original:self.pool.release(entry['buffer'],lease['token'])
                self.active=None

    def close(self):
        with self.lock:
            if self.active is not None:raise RuntimeError('Decision consumer still holds a lease')
            self._clear();self.reader.close();self.pool.close()


class DecisionRecorder:
    def __init__(self,directory,*,metrics=None,image_policy=None,camera_calibration=None,image_preparer_factory=None):
        from decision.vlm.project_config import load_config
        from decision.vlm.annotation.store import perception_view
        self.directory=Path(directory);self.buffer=DecisionBuffer(metrics=metrics)
        self.extent,self.view=perception_view();self.period=load_config()['inference']['period_s']
        self.next_stamp=None;self.last_stamp=None;self.written=0;self.skipped=0;self.error=None
        self.image_preparer=None
        if image_preparer_factory is None:
            from decision.vlm.image_inputs import ImagePreparer
            image_preparer_factory=lambda:ImagePreparer(image_policy,camera_calibration)
        self.image_preparer_factory=image_preparer_factory
        self.queue=queue.Queue(1);self.worker=threading.Thread(target=self._work,name='decision-inputs',daemon=True);self.worker.start()

    def submit(self,people,arrays,association=None,*,scene=None,image_metadata=None):
        if self.error:self.skipped+=1;return False
        previous=self.buffer.scope
        pushed=self.buffer.push(people,arrays,association,scene=scene,image_metadata=image_metadata)
        stamp=people['stamp_s']
        if previous!=self.buffer.scope or self.last_stamp is not None and stamp<=self.last_stamp:self.next_stamp=None
        self.last_stamp=stamp
        if not pushed:self.skipped+=1;return
        if scene is None:return
        if self.next_stamp is not None and stamp<self.next_stamp:return
        lease=self.buffer.acquire()
        if lease is None:self.skipped+=1;return
        self.next_stamp=stamp+self.period;self.queue.put_nowait(lease)

    def _work(self):
        try:
            from decision.vlm.observations import freeze_window
            while True:
                lease=self.queue.get()
                if lease is None:return
                entry=lease['entries'][-1];name=str(entry['stamp_ns'])+'_'+lease['token'][:8]
                with self.buffer.consume(lease) as read:
                    entries=lease['entries'];preparation=None
                    if any(e['association'] is None for e in entries):
                        if self.image_preparer is None:self.image_preparer=self.image_preparer_factory()
                        entries=self.image_preparer.prepare(entries,lease['history'],read)
                        preparation=self.image_preparer.metadata
                    freeze_window(entries,lease['history'],read,self.directory/name,extent=self.extent,
                        view_binding=self.view,image_preparation=preparation)
                self.written+=1
        except Exception as error:
            import sys
            self.error=type(error).__name__+': '+str(error)
            print('可选VLM输入准备已停止，感知导航继续：'+self.error,file=sys.stderr,flush=True)

    def close(self):
        drain_worker(self.worker, self.queue)
        self.buffer.close()
        return dict(written=self.written,unready_or_busy=self.skipped,buffer_dropped=self.buffer.dropped,
                    complete=self.error is None,error=self.error,model_requests=0,
                    image_detector_calls=0 if self.image_preparer is None else self.image_preparer.calls,
                    image_preparation=None if self.image_preparer is None else self.image_preparer.metadata,
                    scope='optional_vlm_input_preparation_not_navigation')
