"""Single-flight acceptance of decision responses; no service or motion startup."""
import copy
import json
import math

from .annotation.rules import digest
from .project_config import load_config


class DecisionRequest:
    """One task's original observation stays bound while newer scenes arrive."""
    def __init__(self,policy,*,deadline_s=None):
        self.policy=policy;self.pending=None;self.last_response=None
        self.deadline=load_config()['inference']['deadline_s'] if deadline_s is None else deadline_s
        if not math.isfinite(self.deadline) or self.deadline<=0:raise ValueError('Positive response deadline required')

    def begin(self,scene,*,window=None,candidates=None,now):
        if self.policy.cancelled:raise ValueError('Task is cancelled')
        if self.pending is not None:raise ValueError('A request is already in flight; do not queue old frames')
        scene.verify();r=scene.record
        if not 0<=now-r['as_of']<=self.deadline:raise ValueError('Request observation is already expired')
        mapping={}
        if self.policy.kind!='seating':
            if window is None or digest({k:v for k,v in window.items() if k!='input_fingerprint'})!=window['input_fingerprint']:
                raise ValueError('A frozen task window is required')
            if window.get('scene')!=r or window['clock_domain']!=r['clock_domain'] or abs(window['stamp_s']-r['as_of'])>1e-6:
                raise ValueError('Window must bind the original navigation scene')
            mapping={int(k):v for k,v in window['pid_to_track'].items()}
            if sorted(mapping)!=window['fields']['v']:raise ValueError('Rendered PIDs and navigation identities differ')
        ticket=self.policy.observe(scene,pid_to_track=mapping,candidates=candidates)
        if self.policy.kind=='social':
            from .vlm_api_common import groups_schema
            schema=groups_schema(sorted(mapping))
        elif self.policy.kind=='follow':
            from .follow import follow_schema
            schema=follow_schema(sorted(mapping),ticket['history'])
        else:
            from .seating.inference import selection_schema
            schema=selection_schema(candidates)
        envelope=dict(ticket=ticket,response_schema=schema,deadline_s=ticket['stamp_s']+self.deadline,
            input_fingerprint=None if window is None else window['input_fingerprint'])
        envelope['binding']=digest(envelope);self.pending=copy.deepcopy(envelope)
        return envelope

    def accept(self,envelope,raw_response,*,scene,now):
        # Retain exactly what arrived, including malformed or late content.
        self.last_response=dict(envelope=copy.deepcopy(envelope),raw_response=copy.deepcopy(raw_response),accepted=False)
        if self.pending is None or envelope!=self.pending:raise ValueError('Unknown or cancelled request')
        self.pending=None
        if not envelope['ticket']['stamp_s']<=now<=envelope['deadline_s']:
            raise ValueError('Decision response deadline exceeded')
        response=json.loads(raw_response) if isinstance(raw_response,str) else raw_response
        result=self.policy.respond(envelope['ticket'],response,scene=scene,now=now)
        self.last_response.update(accepted=True,result=copy.deepcopy(result))
        return result

    def expire(self,now):
        if self.pending is not None and now>self.pending['deadline_s']:
            self.last_response=dict(envelope=self.pending,raw_response=None,accepted=False,reason='deadline')
            self.pending=None
            return True
        return False

    def cancel(self):
        self.pending=None
        return self.policy.cancel()
