"""Optional bounded RGB preparation and explicit nonexecuting runtime client."""
import base64
import copy
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
import numpy as np
from runtime.export import drain_worker
from runtime.transport import exchange_request
from interfaces.runtime import rgb_age_s
from ..tools.observations import Observation


def follow_observation(scene,pixels,association,geometry,camera_info,*,image_metadata=None,
                       bound_candidates=None,fast_png=False):
    """Expose the exact frame and associations already prepared by the T2 worker."""
    from decision.vlm.t2.target import candidates,raw_candidates
    from ..tools.sensors import sensor_context
    image=association['image_input'] if association is not None else image_metadata
    stamp_ns=image['rgb_stamp_ns']
    stamp=stamp_ns/1_000_000_000
    frame=image.get('camera_frame',image.get('rgb_frame'))
    people=scene.record['people'].get('data') or {}
    if bound_candidates is not None:
        detected=bound_candidates
    elif association is None:
        detected=raw_candidates(scene,pixels,stamp,rgb_stamp_ns=stamp_ns)
    else:
        if association['perception_input_fingerprint']!=people.get('input_fingerprint'):
            raise ValueError('T2 associations do not refer to this scene')
        detected=candidates(scene,pixels,stamp,association['detections'],association['result'],
                            detector=dict(identity=association['detector_identity']),rgb_stamp_ns=stamp_ns)
        if detected['rgb_pixel_sha256']!=image['raw_rgb_sha256']:
            raise ValueError('T2 associations do not refer to these RGB pixels')
    return Observation.from_scene(scene,geometry=geometry,
        rgb=dict(pixels=pixels,stamp_s=stamp,stamp_ns=stamp_ns,frame_id=frame,source=dict(kind='t2_window_current_rgb')),
        sensors=sensor_context(geometry,camera_info,rgb_shape=pixels.shape,rgb_frame=frame,rgb_stamp=stamp,rgb_stamp_ns=stamp_ns),
        include_local_grid=True,follow_candidates=detected,fast_png=fast_png)


class FollowAgentFrame:
    """One owned RGB frame and its sealed T2 inputs, encoded only on demand."""
    def __init__(self,scene,pixels,association,geometry,camera_info,*,image_metadata=None,
                 bound_candidates=None,metrics=None):
        if not pixels.flags.owndata:
            raise ValueError('Agent follow RGB needs its own storage')
        pixels.setflags(write=False)
        image = association['image_input'] if association is not None else image_metadata
        record = scene.record
        stamp_ns = image['rgb_stamp_ns']
        if type(stamp_ns) is not int or not 0 <= rgb_age_s(
                record['as_of'],stamp_ns/1e9,record.get('stamp_ns'),stamp_ns) <= .15:
            raise ValueError('Agent follow RGB is missing or stale')
        self.source = dict(observation_id=record['observation_id'],stamp_s=record['as_of'],
            stamp_ns=record.get('stamp_ns'),clock_domain=record['clock_domain'],
            scene_binding=record['input_fingerprint'],
            local_identity=[record[k] for k in ('session_id','generation','frame_id')],
            map_binding={k:copy.deepcopy(record['prior'].get(k)) for k in
                ('valid','map_id','map_frame','map_revision','association_session','association_generation')},
            rgb_stamp_ns=stamp_ns,rgb_stamp_s=stamp_ns/1e9,
            rgb_frame=image.get('camera_frame',image.get('rgb_frame')))
        # The sealed scene arrays are shared, but verification's mutable
        # navigation cache belongs to this reader.
        self.scene = replace(scene,_navigation_checks={})
        self.pixels,self.geometry = pixels,geometry
        if bound_candidates is not None:
            association = dict(image_input=image)
        self.association,self.camera_info = copy.deepcopy(association),copy.deepcopy(camera_info)
        self.image_metadata = copy.deepcopy(image_metadata)
        self.bound_candidates = copy.deepcopy(bound_candidates)
        self.metrics = metrics
        self.lock = threading.Lock()
        self.encoded = None

    def materialize(self):
        with self.lock:
            if self.encoded is None:
                started=time.monotonic()
                result=follow_observation(self.scene,self.pixels,self.association,self.geometry,self.camera_info,
                    image_metadata=self.image_metadata,bound_candidates=self.bound_candidates,fast_png=True)
                encoded=base64.b64encode(result.image).decode('ascii')
                if len(encoded)>1500000:
                    raise ValueError('Agent follow observation exceeds IPC budget')
                self.encoded=(result.record,encoded)
                if self.metrics is not None:
                    self.metrics.trace('agent_follow_frame_materialized',measurement_stamp_s=self.source['stamp_s'],
                        rgb_stamp_ns=self.source['rgb_stamp_ns'],scene_binding=self.source['scene_binding'],
                        elapsed_s=time.monotonic()-started,png_bytes=len(result.image))
            return self.encoded


class ObservationWorker:
    def __init__(self, publish, *, metrics=None, geometry=None, seating=None):
        self.publish, self.metrics, self.geometry = publish, metrics, geometry
        self.queue = queue.Queue(1)
        self.next_stamp = -float('inf')
        self.identity = None
        self.error = None
        self.closed = False
        self.seating=seating
        self.thread = threading.Thread(target=self._work, name='agent-observation', daemon=True)
        self.thread.start()

    def submit(self, scene, pixels, rgb_stamp, rgb_frame, camera_info=None, rgb_stamp_ns=None, *, points=None,clock=None):
        identity = [scene.record[k] for k in ('session_id', 'generation', 'frame_id')]
        stamp = scene.record['as_of']
        if self.error or self.closed or self.queue.full() or identity == self.identity and stamp < self.next_stamp:
            return False
        if (pixels is None or rgb_stamp is None or
                not 0 <= rgb_age_s(stamp,rgb_stamp,scene.record.get('stamp_ns'),rgb_stamp_ns) <= .15):
            return False
        # Producer slots are released after submit. Keep RGB and, for seating,
        # the corresponding scan. The sealed scene retains its CPU safety grid.
        image = np.array(pixels, dtype=np.uint8, copy=True)
        if self.metrics:
            self.metrics.transfer('host_copy', image.nbytes, 'agent_rgb_observation')
        scan=np.array(points[:,:3],copy=True) if self.seating is not None and points is not None else None
        if scan is not None and self.metrics:self.metrics.transfer('host_copy',scan.nbytes,'seating_observation')
        self.queue.put_nowait((scene, image, rgb_stamp, rgb_frame, copy.deepcopy(camera_info), rgb_stamp_ns,scan,clock))
        self.next_stamp, self.identity = stamp+1., identity
        return True

    def _work(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                scene, image, stamp, frame, camera_info, stamp_ns,scan,clock = item
                from ..tools.sensors import sensor_context
                sensors = sensor_context(self.geometry,camera_info,rgb_shape=image.shape,rgb_frame=frame,rgb_stamp=stamp,rgb_stamp_ns=stamp_ns)
                observation = Observation.from_scene(scene, geometry=self.geometry, rgb=dict(pixels=image, stamp_s=stamp, stamp_ns=stamp_ns, frame_id=frame,
                    source=dict(kind='runtime_shared_host_rgb_copy', observation_id=scene.record['observation_id'])), sensors=sensors,
                    include_local_grid=True)
                if self.seating is not None:
                    self.seating.observe(scene,observation,image,scan,camera_info,clock)
                self.publish(observation.record, observation.image)
        except Exception as error:
            self.error = type(error).__name__

    def close(self):
        self.closed = True
        drain_worker(self.thread, self.queue, timeout=3.)
        return dict(error=self.error)


class RuntimeClient:
    follow_channel = 'shadow'
    state_scope = 'shadow_recommendation_not_execution'
    def __init__(self, path):
        self.path = Path(path)

    def exchange(self, kind, value):
        if kind not in ('agent_state','agent_seating_select','agent_seating_cancel', 'goal', 'cancel','follow_start','follow_finish','follow_cancel','follow_lost'):
            raise ValueError('Agent runtime client cannot issue override or safety-management commands')
        if kind.startswith('follow_'):
            wire_kind = 'agent_'+self.follow_channel+'_'+kind
        elif kind in ('goal','cancel'):
            wire_kind = 'agent_shadow_'+kind
        else:
            wire_kind = kind
        result = exchange_request(self.path,wire_kind,value,timeout=.5)
        if result.get('executable') is not False or result.get('execution_feedback') is True:
            raise ValueError('Agent client only supports a nonexecuting shadow runtime')
        return result

    def snapshot(self, *, with_image=False, follow_call_id=None, with_follow_candidates=False):
        value=dict(with_image=with_image)
        if follow_call_id is not None:value['follow_call_id']=follow_call_id
        if with_follow_candidates:value['with_follow_candidates']=True
        result = self.exchange('agent_state', value)
        if result.get('scope') != self.state_scope:
            raise ValueError('Unexpected runtime scope')
        return result

    @staticmethod
    def observation(value):
        record, image = value.get('observation'), value.get('image_base64')
        if record is None or record['rgb']['status'] == 'valid' and image is None:
            return None
        return Observation(copy.deepcopy(record), None if image is None else base64.b64decode(image, validate=True)).verify()

    @staticmethod
    def local_grid(value):
        record = value.get('observation')
        return None if record is None else Observation.local_grid_from_record(record)


class LiveRuntimeClient(RuntimeClient):
    """Explicit live intent channel; it cannot arm or manage safety."""
    state_scope = 'live_observation'

    def exchange(self, kind, value):
        if kind not in ('agent_state','agent_seating_select','agent_seating_cancel', 'goal', 'cancel', 'hold','retarget','cancel_owned','follow_start','follow_finish','follow_cancel','follow_lost'):
            raise ValueError('Live Agent cannot arm or manage safety')
        if kind == 'agent_state':
            wire_kind = 'agent_live_state'
        elif kind in ('agent_seating_select','agent_seating_cancel'):
            wire_kind = kind
        else:
            wire_kind = 'agent_live_'+kind
        result = exchange_request(self.path,wire_kind,value,timeout=.5)
        if result.get('accepted') is False:
            from ..tools.navigation import IntentRejected
            raise IntentRejected(result.get('error', 'Live intent rejected'))
        if result.get('execution_feedback') is True:
            raise ValueError('IPC acknowledgements are not execution evidence')
        return result

class ManualRuntimeClient(RuntimeClient):
    """Observation and display-only follow channel."""
    follow_channel = 'manual'
    def exchange(self, kind, value):
        if kind not in ('agent_state','agent_seating_select','agent_seating_cancel','follow_start','follow_cancel','follow_finish','follow_lost'):
            raise ValueError('Manual Agent client is read-only')
        return super().exchange(kind,value)

    def require_observation_only(self):
        status=exchange_request(self.path,'session_status',{},timeout=.5)
        if status.get('live') is not True or status.get('hardware_enabled') is not False:
            raise ValueError('Manual Agent requires a live observation session without chassis output')
        return status
