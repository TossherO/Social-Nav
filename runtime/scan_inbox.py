"""Drain sensor descriptors promptly while retaining one pending GPU scan lease."""
from collections import deque
import socket
import threading


class LatestScanInbox:
    def __init__(self,wire,observe,invalidate,drain_timeout):
        self.wire,self.observe,self.invalidate=wire,observe,invalidate
        self.drain_timeout=drain_timeout;self.condition=threading.Condition();self.send_lock=threading.Lock()
        self.pending=None;self.messages=deque();self.error=None;self.stopping=False
        self.epoch=0;self.identity=None;self.received=0;self.replaced=0
        self.thread=threading.Thread(target=self._run,name='runtime-scan-inbox',daemon=True);self.thread.start()

    def release(self,packet):
        ref=packet['buffer']
        with self.send_lock:self.wire.send(dict(slot=ref['slot'],epoch=ref['epoch']))

    def current(self,packet):
        with self.condition:return packet['_input_epoch']==self.epoch

    def submit_if_current(self,packet,submit):
        with self.condition:
            if packet['_input_epoch']==self.epoch:submit()

    def _run(self):
        try:
            while True:
                packet=self.wire.recv()
                with self.condition:
                    if packet['kind']=='scan':
                        row=packet['state'];identity=tuple(row[k] for k in ('session_id','generation','frame_id'))
                        if row['health']!=0 or self.identity is not None and identity!=self.identity:
                            self.epoch+=1;self.invalidate()
                        self.identity=identity;packet['_input_epoch']=self.epoch
                        self.observe(packet);self.received+=1
                        if self.pending is not None:
                            self.release(self.pending);self.replaced+=1
                        self.pending=packet
                    else:
                        if packet['kind']=='draining':self.wire.socket.settimeout(self.drain_timeout)
                        self.messages.append(packet)
                    self.condition.notify_all()
                if packet['kind'] in ('end','error'):return
        except Exception as error:
            with self.condition:
                if not self.stopping:self.error=type(error).__name__+': '+str(error)
                self.condition.notify_all()

    def recv(self):
        with self.condition:
            self.condition.wait_for(lambda:self.pending is not None or self.messages or self.error)
            if self.error:raise RuntimeError('Sensor inbox failed: '+self.error)
            if self.pending is not None:
                packet=self.pending;self.pending=None;return packet
            return self.messages.popleft()

    def close(self):
        with self.condition:
            self.stopping=True
            if self.pending is not None:self.release(self.pending);self.pending=None
        if self.thread.is_alive():self.wire.socket.shutdown(socket.SHUT_RD)
        self.thread.join(timeout=2.)
        if self.thread.is_alive():raise TimeoutError('Sensor inbox failed to stop')
        return dict(policy='one_active_one_latest_pending',received=self.received,replaced=self.replaced)
