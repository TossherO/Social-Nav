"""One bounded asynchronous curvature reference; never a timed control plan."""
import copy
from collections import deque
from pathlib import Path
import time
import numpy as np
import yaml
from perception.environment.context import fingerprint
from .continuity import context_binding,check_reference
from .inputs import scene_status,request_status
from .footprint import footprint_status
from perception.environment.navigation import navigation_geometry
from .runtime import Worker


def load_config(path=None):
    settings=yaml.safe_load(Path(path or Path(__file__).with_name('configs')/'reference.yaml').read_text())
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in settings.values()):raise ValueError('Invalid reference budgets')
    if settings['revalidate_s']>=settings['max_age_s']:raise ValueError('Reference refresh must precede expiry')
    return settings


def planning_config(planning,settings=None):
    value=copy.deepcopy(planning);settings=load_config() if settings is None else settings
    value['algorithm']='hybrid_astar';value['local_replanning']['enabled']=False
    value['pose_search']['goal_approach_straight_m']=settings['terminal_straight_m']
    return value


class ReferenceRuntime:
    retain_reference_on_observation=True
    def __init__(self,planning,motion,geometry,settings=None,*,worker_factory=Worker):
        self.motion=motion;self.geometry=geometry;self.settings=copy.deepcopy(settings) if settings is not None else load_config()
        self.planning=planning_config(planning,self.settings)
        self.worker=worker_factory('reference',self.planning,motion,geometry)
        self.events=deque(maxlen=512);self.sequence=0;self.epoch=0;self.context=None
        self.latest=None;self.current=None;self.reference=None;self.reason='waiting_for_observation'
        self.cancelled={};self.last_sent=-np.inf;self.next_retry=-np.inf;self.blocked=False;self.need_replan=False
        self.diagnostics=False;self.last_reply=None;self.start_evidence=None
        self.failure=None
        self.retry_after_observation=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.worker.stop()
    def _event(self,event,**fields):self.events.append(dict(event=event,epoch=self.epoch,monotonic_s=time.monotonic(),**fields))
    def drain_events(self):
        rows=list(self.events);self.events.clear();return rows

    def invalidate(self,reason):
        if reason=='localization_reset' and self.latest is not None:
            req=self.latest['request'];self.cancelled[req['task_id']]=req['goal_revision']
        self.epoch+=1;self.latest=self.current=self.reference=None;self.context=None
        self.reason=reason;self.blocked=False;self.need_replan=False;self.last_sent=self.next_retry=-np.inf
        self.failure=None
        self.retry_after_observation=None
        if self.worker.job is not None:self.worker.restart()
        self._event('invalidated',reason=reason)

    def cancel(self,request=None):
        req=request or (None if self.latest is None else self.latest['request'])
        if req is not None:self.cancelled[req['task_id']]=req['goal_revision']
        self.invalidate('cancel')

    def update(self,request,scene,state,asset=None,*,now,clock_rate=1.,**unused):
        self.start_evidence=None
        try:
            if not np.isfinite(clock_rate) or clock_rate<=0:raise ValueError('Invalid planning source clock rate')
            if request.get('algorithm','hybrid_astar')!='hybrid_astar':raise ValueError('direct_control_requires_curvature_reference')
            if request['goal_revision']<=self.cancelled.get(request['task_id'],-1):raise ValueError('cancelled_goal_revision')
            reason=scene_status(scene,now,scene.record['clock_domain'],self.planning,check_navigation=False) or request_status(request,scene,now,scene.record['clock_domain'])
            if reason:raise ValueError(reason)
            scene.verify()
            context=context_binding(request,scene,asset)
            if self.context is not None and context!=self.context:self.invalidate('context_changed')
            self.context=context
            self.latest=dict(request=copy.deepcopy(request),scene=scene,asset=asset,context=context,received_mono=time.monotonic(),clock_rate=clock_rate)
            pose=self.geometry.pose(scene.record['localization']['world_from_lidar'])
            grid=navigation_geometry(scene,now=now,asset=asset,check_asset=True)
            self.blocked=footprint_status(grid,self.geometry,pose,self.planning['safety_margin_m'])!='free'
            if self.diagnostics:
                from .footprint import footprint_details
                self.start_evidence=footprint_details(grid,self.geometry,pose,self.planning['safety_margin_m'])
                self.start_evidence.update(status=footprint_status(grid,self.geometry,pose,self.planning['safety_margin_m']),
                    pose=pose.tolist(),scene_binding=scene.record['input_fingerprint'])
            self.reason='start_footprint_not_free' if self.blocked else None
            self._event('observation',as_of=scene.record['as_of'],scene=scene.record['input_fingerprint'])
            return True
        except (ValueError,KeyError,TypeError) as error:
            self.current=None;self.reason=str(error);return False

    def request_replan(self):self.need_replan=True

    def tick(self,*,now,clock_domain):
        mono=time.monotonic();response=self.worker.poll()
        if response is not None:
            job,reply=response
            transient_input=False
            if self.diagnostics:self.last_reply=response
            fields=dict(role='reference',job_id=job['job_id'],as_of=job['scene'].record['as_of'],
                elapsed_ms=(mono-job['sent_mono'])*1000,compute_ms=reply['compute_s']*1000)
            envelope=reply.get('result')
            result=envelope.get('result') if isinstance(envelope,dict) else None
            if isinstance(result,dict):
                fields['result_status']=result.get('status');fields['result_reason']=result.get('reason')
                fields['planning_timing_ms']=result.get('timing_ms',{})
                fields['search_attempts']=result.get('search_attempts',[])
            try:
                if reply['error']:raise ValueError(reply['error'])
                if self.latest is None or job['epoch']!=self.epoch or job['context']!=self.context:raise ValueError('superseded_context')
                check_reference(reply['result'],self.latest['request'],self.latest['scene'],self.latest['asset'],self.settings['max_age_s'],require_path=False)
                if result['status']!='path_found' or len(result.get('path',[]))<2:
                    if result.get('reason') in ('navigation_prior_future_or_stale','geometry_future_or_stale'):
                        transient_input=True;self.failure=None
                        raise ValueError(result['reason'])
                    self.failure=dict(request_binding=fingerprint(job['request']),reason=result['reason'],
                        as_of=job['scene'].record['as_of'],job_id=job['job_id'])
                    raise ValueError('reference_has_no_path')
                self.reference=self.current=reply['result'];self.need_replan=False;self.failure=None;self.retry_after_observation=None
                self._event('reference_ready',**fields)
            except (ValueError,KeyError,TypeError) as error:
                self.reason=str(error)
                self.retry_after_observation=job['scene'].record['as_of'] if transient_input else None
                self.next_retry=mono if transient_input else mono+self.settings['failure_retry_s']
                self._event('discarded',reason=str(error),transient_input=transient_input,
                    retry_after_s=max(0.,max(self.next_retry,self.last_sent+self.settings['retry_s'])-mono),**fields)
        if self.worker.job is not None and mono-self.worker.job['sent_mono']>self.settings['timeout_s']:
            self.worker.restart();self._event('timeout',role='reference');self.reason='reference_timeout'
            self.next_retry=mono+self.settings['failure_retry_s']
        if not self.worker.ready and mono-self.worker.started>self.settings['startup_timeout_s']:
            self.worker.restart();self.reason='reference_worker_startup_timeout'
        if self.latest is None:return self.status(now=now,clock_domain=clock_domain)
        item=self.latest;scene=item['scene'];r=scene.record
        reason=scene_status(scene,now,clock_domain,self.planning)
        if reason or mono-item['received_mono']>self.planning['max_geometry_age_s']:
            self.current=None;self.reason=reason or 'observation_wall_stale'
            return self.status(now=now,clock_domain=clock_domain)
        # A transient input outage must not permanently hide a retained valid
        # reference. Restore only after checking the original source time and
        # current task/map identity; DWA still certifies the current geometry.
        if self.reference is not None:
            try:
                check_reference(self.reference,item['request'],scene,item['asset'],self.settings['max_age_s'])
                self.current=self.reference
            except (ValueError,KeyError,TypeError) as error:
                self.current=None;self.reason=str(error)
        expired=self.current is None or r['as_of']-self.current['as_of']>=self.settings['revalidate_s']
        if (not self.blocked and self.worker.ready and self.worker.job is None and (expired or self.need_replan)
                and mono-self.last_sent>=self.settings['retry_s'] and mono>=self.next_retry
                and (self.retry_after_observation is None or r['as_of']>self.retry_after_observation)):
            self.sequence+=1
            job=dict(job_id=self.sequence,epoch=self.epoch,context=self.context,request=item['request'],scene=scene,
                asset=item['asset'],sent_mono=mono,now=now,clock_rate=item['clock_rate'])
            self.worker.send(job);self.last_sent=mono;self.retry_after_observation=None
            self._event('submitted',role='reference',job_id=self.sequence,as_of=r['as_of'])
        return self.status(now=now,clock_domain=clock_domain)

    def status(self,*,now,clock_domain):
        return dict(status='reference_ready' if self.current is not None else 'unavailable',reason=None if self.current is not None else self.reason or 'waiting_for_background_reference',
            result=self.current,executable=False,task_completed=False,event_overflow=0,start_evidence=self.start_evidence,
            waiting=dict(worker_ready=self.worker.ready,busy=self.worker.job is not None,blocked=self.blocked,
                retry_remaining_s=max(0.,self.next_retry-time.monotonic()),epoch=self.epoch))
