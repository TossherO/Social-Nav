"""Periodic asynchronous observation with explicit ephemeral-history policy."""
import copy
import math
import time

from .async_model import AsyncModel


class PeriodicObservation:
    """Schedule one model request at a time; no source timestamp is refreshed."""
    def __init__(self, camera, decide, *, interval_s, history_limit, history_policy):
        if (not math.isfinite(interval_s) or interval_s<=0 or type(history_limit) is not int or
                not 0<=history_limit<=64 or history_policy not in ('none','bounded','agent')):
            raise ValueError('Invalid periodic observation settings')
        self.camera,self.decide=camera,decide
        self.interval_s,self.history_limit,self.history_policy=interval_s,history_limit,history_policy
        self.history=[];self.last_sequence=-1;self.last_request=None;self.model=AsyncModel()

    def poll(self, context):
        result=self.model.poll()
        if result is None:return None
        if result['stale'] or result['context']['context']!=context:
            return dict(status='discarded',reason='context_changed')
        if not result['ok']:
            return dict(status='failed',reason='model_'+result['error_type'])
        return dict(status='available',value=result['value'],frame=result['context']['frame'])

    def submit_if_due(self, context, *, monotonic_s=None):
        now=time.monotonic() if monotonic_s is None else monotonic_s
        if (self.model.busy or self.last_request is not None and now-self.last_request<self.interval_s):return False
        frame=self.camera.snapshot()
        if frame.get('status')!='available':return False
        sequence=frame['record']['sequence']
        if sequence<=self.last_sequence:return False
        frame=copy.deepcopy(frame);history=copy.deepcopy(self.history);context=copy.deepcopy(context)
        self.model.submit('periodic_observation',dict(context=context,frame=frame),
                          lambda:self.decide(context,history,frame))
        self.last_sequence=sequence;self.last_request=now
        return True

    def remember(self, observation, image, decision):
        if self.history_policy=='agent' and type(decision.get('retain_observation')) is not bool:
            raise ValueError('Agent must explicitly decide whether to retain this observation')
        keep=(self.history_policy=='bounded' or self.history_policy=='agent' and decision['retain_observation'])
        if keep and self.history_limit:
            self.history.append(dict(record=copy.deepcopy(observation),image=image))
            self.history=self.history[-self.history_limit:]

    def clear(self):
        self.model.invalidate()
        self.history.clear()

    def close(self):
        self.clear()
        return self.model.close()
