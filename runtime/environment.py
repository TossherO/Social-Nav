"""One map owner with at most one pending observation; source times stay intact."""
import threading
import time


class LatestMapWorker:
    def __init__(self,process,publish,reset):
        self.process,self.publish,self.reset=process,publish,reset
        self.condition=threading.Condition();self.pending=None;self.result=None
        self.token=0;self.identity=None;self.active=False;self.stopping=False;self.error=None
        self.counts=dict(submitted=0,replaced=0,published=0,invalidated=0,discarded=0)
        self.thread=threading.Thread(target=self._run,name='runtime-environment',daemon=True)
        self.thread.start()

    def submit(self,observation,*,identity):
        with self.condition:
            self._check()
            if self.stopping:raise RuntimeError('Map worker is closed')
            if self.identity is not None and identity!=self.identity:
                self._invalidate()
            self.identity=identity
            self.counts['submitted']+=1
            if self.pending is not None:self.counts['replaced']+=1
            self.pending=(self.token,observation);self.condition.notify_all()

    def invalidate(self):
        with self.condition:
            self._check();self._invalidate()

    def _invalidate(self):
        self.token+=1;self.counts['invalidated']+=1
        if self.pending is not None:self.counts['discarded']+=1
        self.pending=None;self.result=None
        # Serialized with publication, so an older in-flight map cannot
        # restore geometry after a failed localization observation.
        self.publish(None);self.condition.notify_all()

    def latest(self):
        with self.condition:
            self._check();return self.result

    def _check(self):
        if self.error is not None:raise RuntimeError('Map worker failed: '+self.error)

    def _run(self):
        processed_token=None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.stopping)
                    if self.pending is None:return
                    token,observation=self.pending;self.pending=None;self.active=True
                if token!=processed_token:self.reset();processed_token=token
                result=self.process(observation)
                with self.condition:
                    if token==self.token:
                        self.publish(result);self.result=result;self.counts['published']+=1
                    else:self.counts['discarded']+=1
                    self.active=False;self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error=type(error).__name__+': '+str(error)
                self.active=False;self.condition.notify_all()

    def flush(self,timeout=50.):
        deadline=time.monotonic()+timeout
        with self.condition:
            while self.pending is not None or self.active:
                self._check()
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Map worker failed to drain')
                self.condition.wait(min(remaining,1.))
            self._check()

    def close(self):
        with self.condition:self.stopping=True;self.condition.notify_all()
        self.thread.join(timeout=50.)
        if self.thread.is_alive():raise TimeoutError('Map worker failed to stop')
        self._check()
        return dict(self.counts,policy='one_active_one_latest_pending')
