"""One bounded asynchronous curvature reference; never a timed control plan."""
import copy
from collections import deque
from pathlib import Path
import time
import numpy as np
import yaml
from perception.environment.context import fingerprint
from .continuity import context_binding,check_reference,goal_binding,local_goal
from .inputs import association_status,scene_status,request_status
from .footprint import footprint_status,swept_check
from perception.environment.navigation import navigation_geometry
from .overlay import ReferenceObstacles,retain_obstacles
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
    def __init__(self,planning,motion,geometry,settings=None,*,worker_factory=Worker):
        self.motion=motion;self.geometry=geometry;self.settings=copy.deepcopy(settings) if settings is not None else load_config()
        self.planning=planning_config(planning,self.settings)
        self.worker=worker_factory(self.planning,geometry)
        self.events=deque(maxlen=512);self.sequence=0;self.epoch=0;self.context=None
        self.latest=None;self.current=None;self.reference=None;self.reason='waiting_for_observation'
        self.cancelled={};self.last_sent=-np.inf;self.next_retry=-np.inf;self.blocked=False;self.need_replan=False
        self.diagnostics=False;self.last_reply=None;self.start_evidence=None
        self.failure=None
        self.retry_after_observation=None
        self.control_blocked_since=None
        self.obstacles=ReferenceObstacles()
        self.follow_key=None
        self.follow_revision=-1

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.worker.stop()
    def _event(self,event,**fields):self.events.append(dict(event=event,epoch=self.epoch,monotonic_s=time.monotonic(),**fields))
    def drain_events(self):
        rows=list(self.events);self.events.clear();return rows

    def invalidate(self,reason):
        if reason in ('seek','localization_reset'):
            self.obstacles.reset()
        if reason=='localization_reset' and self.latest is not None:
            req=self.latest['request'];self.cancelled[req['task_id']]=req['goal_revision']
        self.epoch+=1;self.latest=self.current=self.reference=None;self.context=None
        self.follow_key=None;self.follow_revision=-1
        self.reason=reason;self.blocked=False;self.need_replan=False;self.last_sent=self.next_retry=-np.inf
        self.failure=None
        self.retry_after_observation=None
        self.control_blocked_since=None
        if self.worker.job is not None:self.worker.restart()
        self.start_evidence=None
        self._event('invalidated',reason=reason)

    @staticmethod
    def _obstacle_key(scene,asset):
        r=scene.record;a=r['prior']
        return (asset.manifest['snapshot_id'],asset.manifest['layers_sha256'],r['clock_domain'],
                r['session_id'],r['generation'],a.get('association_session'),a.get('association_generation'),
                a.get('map_revision'))

    def _retained_obstacles(self,scene,asset):
        if asset is None or self.obstacles.key!=self._obstacle_key(scene,asset):
            return ()
        return self.obstacles.blocked

    def observe(self,scene,asset,*,now):
        """Accumulate reference evidence as observations arrive, even without a goal."""
        if asset is None or scene is None:return
        r=scene.record
        a=r['prior']
        key=self._obstacle_key(scene,asset)
        if key==self.obstacles.key and r['as_of']<=self.obstacles.last:return
        if (scene_status(scene,now,r['clock_domain'],self.planning,check_navigation=False) or
                association_status(scene,asset,self.planning)):
            return
        matrix=np.asarray(a['map_from_local'])
        if matrix[2,2]<np.cos(.15):return
        try:
            scene.verify()
        except ValueError:
            return
        from .planner import se2
        world=np.asarray(r['localization']['world_from_lidar'])
        local_from_map=se2(self.geometry.pose(world))@np.linalg.inv(se2(self.geometry.pose(matrix@world)))
        version=self.obstacles.version
        added=self.obstacles.update(scene,asset.snapshot,local_from_map,key)
        if self.obstacles.version!=version:
            self.need_replan=True
            self.next_retry=-np.inf
            if added and self.reference is not None and not self._avoids_obstacles(self.reference,scene,asset):
                self.reference=self.current=None
                self.reason='reference_obstructed_by_new_observation'

    def _avoids_obstacles(self,envelope,scene,asset):
        blocked=self._retained_obstacles(scene,asset)
        if not blocked:return True
        result=envelope['result']
        if result.get('reference_search_frame',result['frame_id'])!=asset.snapshot.metadata['frame_id']:
            return True
        # Recheck just the new reference exclusions. The original result and the
        # controller still perform their own complete geometry checks.
        from perception.environment.layers import GridSnapshot
        from .planner import se2,transform_poses
        poses=np.asarray(result['path'])
        if result['frame_id']!=asset.snapshot.metadata['frame_id']:
            r=scene.record
            world=np.asarray(r['localization']['world_from_lidar'])
            matrix=np.asarray(r['prior']['map_from_local'])
            poses=transform_poses(poses,se2(self.geometry.pose(matrix@world))@
                                  np.linalg.inv(se2(self.geometry.pose(world))))
        empty=GridSnapshot(asset.snapshot.metadata,dict(occupancy=np.zeros(
            asset.snapshot.metadata['shape'],np.int8)))
        excluded=retain_obstacles(empty,blocked)
        check=swept_check(excluded,self.geometry,poses,self.planning['safety_margin_m'],
                          self.planning['max_path_samples'],prefix=False)
        return check['status']=='free'

    def cancel(self,request=None):
        req=request or (None if self.latest is None else self.latest['request'])
        if req is not None:self.cancelled[req['task_id']]=req['goal_revision']
        self.invalidate('cancel')

    def update(self,request,scene,state,asset=None,*,now,clock_rate=1.,follow_key=None,**unused):
        self.start_evidence=None
        try:
            if not np.isfinite(clock_rate) or clock_rate<=0:raise ValueError('Invalid planning source clock rate')
            if request.get('algorithm','hybrid_astar')!='hybrid_astar':raise ValueError('direct_control_requires_curvature_reference')
            if request['goal_revision']<=self.cancelled.get(request['task_id'],-1):raise ValueError('cancelled_goal_revision')
            clock=scene.record['clock_domain']
            reason=request_status(request,scene,now,clock) or scene_status(scene,now,clock,self.planning,check_navigation=False)
            if (reason=='geometry_future_or_stale' and self.latest is not None and
                    self.latest['scene'] is scene and self.latest['request']==request):
                # The control loop can poll the same scan between sensor updates.
                # tick checks its age; keep the goal identity and cached path for recovery.
                self.current=None
                self.reason=reason
                return False
            if reason:raise ValueError(reason)
            scene.verify()
            # Follow revisions are a replaceable pending goal within a single
            # owner/track and map context. Results still bind to exact goals.
            scope=dict(request,goal=[],goal_revision=0) if follow_key is not None else request
            context=context_binding(scope,scene,asset)
            if self.context is not None and (context!=self.context or follow_key!=self.follow_key):
                self.invalidate('context_changed')
            self.context=context
            self.follow_key=follow_key
            if follow_key is not None and (self.latest is None or
                    goal_binding(request)!=goal_binding(self.latest['request'])):
                if self.failure is not None:
                    self.failure=None;self.next_retry=-np.inf
                self._event('follow_goal_pending',goal_revision=request['goal_revision'],
                            goal=request['goal'],as_of=scene.record['as_of'])
            self.latest=dict(request=copy.deepcopy(request),scene=scene,asset=asset,context=context,received_mono=time.monotonic(),clock_rate=clock_rate)
            pose=self.geometry.pose(scene.record['localization']['world_from_lidar'])
            grid=navigation_geometry(scene,now=now,asset=asset,check_asset=True)
            start_status=footprint_status(grid,self.geometry,pose,self.planning['safety_margin_m'])
            can_use_static=(request['mode']=='mapped' or request.get('completion_policy')=='continuous') and (
                not association_status(scene,asset,self.planning))
            # The worker still checks the whole start footprint on the mapped
            # reference grid. Only the local execution checks require fresh
            # local free evidence under the entire vehicle.
            self.blocked=start_status!='free' and not (can_use_static and start_status=='unknown')
            if self.diagnostics:
                from .footprint import footprint_details
                self.start_evidence=footprint_details(grid,self.geometry,pose,self.planning['safety_margin_m'])
                self.start_evidence.update(status=start_status,pose=pose.tolist(),scene_binding=scene.record['input_fingerprint'])
            self.reason='start_footprint_not_free' if self.blocked else None
            self._event('observation',as_of=scene.record['as_of'],scene=scene.record['input_fingerprint'])
            return True
        except (ValueError,KeyError,TypeError) as error:
            if self.context is not None:
                self.invalidate(str(error))
            else:
                self.reason=str(error)
            return False

    def note_control(self,packet,*,monotonic_s=None):
        """Request an early refresh after sustained lack of local progress.

        This is only a planning hint: DWA and independent safety continue to
        check every candidate, without waiting for the background worker.
        """
        proposal=packet.get('proposal') or {};velocity=proposal.get('velocity') or [0.,0.]
        goal=packet.get('recorded_goal_diagnostic') or {}
        stopped=(proposal.get('status')=='candidate' and len(velocity)>=2 and
            abs(velocity[0])<.03 and abs(velocity[1])<.05 and not goal.get('reached_xy',False))
        blocked=proposal.get('reason')=='no_admissible_velocity' or stopped
        if not blocked:
            self.control_blocked_since=None;return
        now=time.monotonic() if monotonic_s is None else monotonic_s
        if self.control_blocked_since is None:self.control_blocked_since=now
        elif now-self.control_blocked_since>=1.:self.need_replan=True

    def tick(self,*,now,clock_domain):
        mono=time.monotonic()
        if self.latest is not None:
            reason=request_status(self.latest['request'],self.latest['scene'],now,clock_domain)
            if reason:
                self.invalidate(reason)
        response=self.worker.poll()
        if response is not None:
            job,reply=response
            transient_input=False
            if self.diagnostics:self.last_reply=response
            fields=dict(role='reference',job_id=job['job_id'],as_of=job['scene'].record['as_of'],
                goal_revision=job['request']['goal_revision'],
                elapsed_ms=(mono-job['sent_mono'])*1000,compute_ms=reply['compute_s']*1000)
            envelope=reply.get('result')
            result=envelope.get('result') if isinstance(envelope,dict) else None
            if isinstance(result,dict):
                fields['result_status']=result.get('status');fields['result_reason']=result.get('reason')
                fields['planning_timing_ms']=result.get('timing_ms',{})
                fields['search_attempts']=result.get('search_attempts',[])
                if 'follow_goal_adjustment' in result:
                    fields['follow_goal_adjustment']=result['follow_goal_adjustment']
                if 'reference_search_frame' in result:
                    fields['reference_search_frame']=result['reference_search_frame']
                    fields['retained_occupied_cells']=result['reference_overlay']['retained_occupied_cells']
            try:
                if reply['error']:raise ValueError(reply['error'])
                if self.latest is None or job['epoch']!=self.epoch or job['context']!=self.context:raise ValueError('superseded_context')
                requested=self.latest['request'] if self.follow_key is None else job['request']
                check_reference(envelope,requested,self.latest['scene'],self.latest['asset'],self.settings['max_age_s'],require_path=False)
                superseded=goal_binding(requested)!=goal_binding(self.latest['request'])
                if result['status']!='path_found' or len(result.get('path',[]))<2:
                    if superseded:
                        transient_input=True
                        raise ValueError('pending_follow_goal_changed')
                    if job.get('reference_obstacle_version',self.obstacles.version)!=self.obstacles.version:
                        transient_input=True
                        raise ValueError('reference_observation_changed')
                    if result.get('reason') in ('navigation_prior_future_or_stale','geometry_future_or_stale'):
                        transient_input=True;self.failure=None
                        raise ValueError(result['reason'])
                    # A negative result withdraws the path for that goal. A
                    # compatible active path may still work up to its endpoint
                    # while a different pending follow goal is unreachable.
                    active=self.reference
                    if result['status']=='no_path' and (self.follow_key is None or active is None or
                            goal_binding(active['result']['request'])==goal_binding(requested)):
                        self.reference=self.current=None
                    self.failure=dict(request_binding=fingerprint(job['request']),reason=result['reason'],
                        as_of=job['scene'].record['as_of'],job_id=job['job_id'])
                    raise ValueError('reference_has_no_path')
                if (job.get('reference_obstacle_version',self.obstacles.version)!=self.obstacles.version and
                        not self._avoids_obstacles(envelope,self.latest['scene'],self.latest['asset'])):
                    transient_input=True
                    raise ValueError('reference_obstructed_by_new_observation')
                if self.follow_key is not None:
                    if not self._compatible_follow(requested,self.latest):
                        transient_input=True
                        raise ValueError('pending_follow_goal_changed')
                    self.follow_revision=requested['goal_revision']
                self.reference=self.current=reply['result'];self.need_replan=False;self.failure=None;self.retry_after_observation=None
                self._event('reference_ready',**fields)
            except (ValueError,KeyError,TypeError) as error:
                self.reason=str(error)
                self.retry_after_observation=job['scene'].record['as_of'] if transient_input else None
                self.next_retry=mono if transient_input else mono+self.settings['failure_retry_s']
                self._event('discarded',reason=str(error),transient_input=transient_input,
                    retry_after_s=max(0.,max(self.next_retry,self.last_sent+self._retry_period())-mono),**fields)
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
                accepted=self.reference['result']['request'] if self.follow_key is not None else item['request']
                if self.follow_key is not None and not self._compatible_follow(accepted,item):
                    self.reference=None
                    raise ValueError('reference_follow_goal_changed')
                check_reference(self.reference,accepted,scene,item['asset'],self.settings['max_age_s'])
                self.current=self.reference
            except (ValueError,KeyError,TypeError) as error:
                self.current=None;self.reason=str(error)
        expired=self.current is None or r['as_of']-self.current['as_of']>=self.settings['revalidate_s']
        new_follow_goal=(self.follow_key is not None and self.current is not None and
                         goal_binding(self.current['result']['request'])!=goal_binding(item['request']))
        if (not self.blocked and self.worker.ready and self.worker.job is None and (expired or self.need_replan or new_follow_goal)
                and mono-self.last_sent>=self._retry_period() and mono>=self.next_retry
                and (self.retry_after_observation is None or r['as_of']>self.retry_after_observation)):
            self.sequence+=1
            job=dict(job_id=self.sequence,epoch=self.epoch,context=self.context,request=item['request'],scene=scene,
                asset=item['asset'],sent_mono=mono,now=now,clock_rate=item['clock_rate'],
                reference_obstacles=self._retained_obstacles(scene,item['asset']),
                reference_obstacle_version=self.obstacles.version)
            self.worker.send(job);self.last_sent=mono;self.retry_after_observation=None
            self._event('submitted',role='reference',job_id=self.sequence,as_of=r['as_of'],
                        goal_revision=item['request']['goal_revision'])
        return self.status(now=now,clock_domain=clock_domain)

    def _retry_period(self):
        return self.settings['retry_s'] if self.follow_key is None else self.settings['follow_retry_s']

    def _compatible_follow(self,request,item):
        latest=item['request']
        if request['goal_revision']<self.follow_revision:return False
        if goal_binding(request)==goal_binding(latest):return True
        scene,asset=item['scene'],item['asset']
        a=local_goal(request,scene,self.geometry,asset,self.planning)[:2]
        b=local_goal(latest,scene,self.geometry,asset,self.planning)[:2]
        base=self.geometry.pose(scene.record['localization']['world_from_lidar'])[:2]
        return bool(np.linalg.norm(a-b)<=self.settings['follow_goal_delta_m'] and (a-base)@(b-base)>=0.)

    def status(self,*,now,clock_domain):
        request=None
        if self.current is not None and self.follow_key is not None:
            # Refresh only the tracking lease from the current verified track.
            # The original path, goal and observation timestamps stay intact.
            request=dict(self.current['result']['request'],valid_until_s=self.latest['request']['valid_until_s'])
        return dict(status='reference_ready' if self.current is not None else 'unavailable',reason=None if self.current is not None else self.reason or 'waiting_for_background_reference',
            result=self.current,request=request,executable=False,task_completed=False,event_overflow=0,start_evidence=self.start_evidence,
            waiting=dict(worker_ready=self.worker.ready,busy=self.worker.job is not None,blocked=self.blocked,
                retry_remaining_s=max(0.,self.next_retry-time.monotonic()),epoch=self.epoch))
