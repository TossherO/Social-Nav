"""Latest-only leased reference previews; no goal or chassis control channel."""
import copy
import math
import socket
import threading
import time
from pathlib import Path
from runtime.transport import Connection


def request_preview(path, owner, request):
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(.5)
    try:
        sock.connect(str(Path(path).resolve()))
        wire=Connection(sock)
        wire.send(dict(kind='agent_manual_preview',value=dict(owner=owner,request=request)))
        result=wire.recv()
    finally:sock.close()
    if result.get('accepted') is not True:
        raise ValueError(result.get('error','manual_preview_refused'))
    return result


def reference_request(handle, pose, route, route_index, catalog, *, require_yaw=False):
    index=0 if route is None else route_index
    if index is None:return None
    target=(pose if require_yaw else pose[:2]) if route is None else route['waypoints'][index]
    final=route is None or index==len(route['waypoints'])-1
    if route is not None and final and require_yaw:target=[*target,route['final_yaw']]
    return dict(request_id=handle['call_id']+':'+str(index),
        task_id='manual-preview:'+handle['task_id']+':'+handle['call_id'],
        goal_revision=handle['goal_revision']*100+index,mode='mapped',
        frame_id=catalog.value['frame_id'],target_reference='base_link',goal=copy.deepcopy(target),
        goal_behavior='stop' if final else 'pass_through',map_snapshot_id=catalog.value['map_snapshot_id'])


def region_check(preview, route, route_index, handle, frame, now):
    """Check the matching, fresh planner output, never a line between waypoints."""
    if not route or not route.get('region_polygon') or route_index is None or route_index<=0:return 'not_applicable'
    if (preview.get('owner')!=handle['call_id'] or
            preview.get('request_id')!=handle['call_id']+':'+str(route_index) or
            preview.get('reference_frame')!=frame or preview.get('status')!='reference_ready'):
        return 'unavailable'
    stamp=preview.get('reference_stamp_s')
    if (not isinstance(stamp,(int,float)) or not isinstance(now,(int,float)) or
            not math.isfinite(stamp) or not math.isfinite(now) or not 0<=now-stamp<=3.):return 'unavailable'
    path=preview.get('reference_path')
    if not isinstance(path,list) or len(path)<2:return 'unavailable'
    from ..core.route import path_outside_region
    return 'clear' if path_outside_region(path,route['region_polygon']) else 'crosses'


class ManualReference:
    def __init__(self,path):
        self.path=path;self.lock=threading.Lock();self.wakeup=threading.Event()
        self.desired=None;self.version=0;self.error=None;self.closed=False
        self.thread=threading.Thread(target=self._work,name='manual-reference-preview',daemon=True)
        self.thread.start()

    def set(self,owner,request):
        with self.lock:
            desired=None if owner is None else (owner,copy.deepcopy(request))
            if desired!=self.desired:
                self.desired=desired;self.version+=1;self.wakeup.set()

    def status(self):
        with self.lock:return self.error

    def _work(self):
        previous=None
        while True:
            with self.lock:
                desired=copy.deepcopy(self.desired);version=self.version;closed=self.closed
            try:
                if previous is not None and (desired is None or desired[0]!=previous):
                    request_preview(self.path,previous,None);previous=None
                if desired is not None:
                    request_preview(self.path,*desired);previous=desired[0]
                with self.lock:
                    if version==self.version:self.error=None
            except (OSError,EOFError,ValueError) as error:
                with self.lock:
                    if version==self.version:self.error=type(error).__name__+': '+str(error)[:160]
            if closed:return
            self.wakeup.wait(.8);self.wakeup.clear()

    def close(self):
        with self.lock:self.desired=None;self.version+=1;self.closed=True
        self.wakeup.set();self.thread.join(timeout=2.)
