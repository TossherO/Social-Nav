"""One bounded asynchronous curvature reference; never a timed control plan."""
import copy
from collections import deque
from pathlib import Path
import time
import numpy as np
import yaml
from perception.environment.context import fingerprint
from perception.environment.layers import GridSnapshot
from .continuity import context_binding,check_reference,goal_binding,local_goal
from .inputs import association_status,scene_status,request_status,navigation_grid,navigation_binding
from .footprint import footprint_status,swept_check
from perception.environment.navigation import navigation_geometry
from .overlay import ReferenceObstacles,retain_obstacles
from .runtime import Worker


def planning_observation(request, record, *, source, status, reason, max_age_s, **details):
    """A compact outcome tied to the source geometry and exact planning request."""
    prior=record['prior']
    return dict(source=source,status=status,reason=reason,request_id=request['request_id'],
        task_id=request['task_id'],goal_revision=request['goal_revision'],goal_binding=goal_binding(request),
        target=copy.deepcopy(request['goal']),frame_id=request['frame_id'],
        final_endpoint=request.get('goal_behavior','pass_through')=='stop',
        segment_id=request.get('route_segment_id'),source_stamp_s=record['as_of'],
        source_stamp_ns=record.get('stamp_ns'),valid_until_s=record['as_of']+max_age_s,
        scene_binding=record['input_fingerprint'],observation_id=record['observation_id'],
        clock_domain=record['clock_domain'],local_identity=[record[k] for k in ('session_id','generation','frame_id')],
        map_binding={k:prior.get(k) for k in ('valid','map_id','map_frame','map_revision',
            'association_session','association_generation')},**details)


def blockage_kind(result, request):
    """Recognize completed negative searches and explicit occupied footprints."""
    if result['status']!='no_path':
        return None
    reason=result['reason']
    if reason in ('goal_occupied','goal_footprint_occupied'):
        return 'destination_occupied' if request.get('goal_behavior')=='stop' else 'route_blocked'
    if reason in ('start_footprint_occupied','returned_path_occupied'):
        return 'route_blocked'
    if reason=='returned_path_no_curvature_route':
        attempts=result.get('search_attempts') or []
        if attempts and attempts[-1].get('reason') in ('no_curvature_route','disconnected_search_region'):
            return 'route_blocked'
    return None


def blocked_position(result, request):
    reason=result['reason']
    if reason in ('goal_occupied','goal_footprint_occupied'):
        return list(request['goal'][:2])
    if (reason=='returned_path_occupied' and
            result.get('reference_search_frame',result.get('frame_id'))==request['frame_id']):
        pose=(result.get('footprint_check') or {}).get('failed_pose')
        if pose is not None:
            return list(pose[:2])
    return None


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
        self.obstacles=ReferenceObstacles(confirm_scans=self.settings.get('person_obstacle_confirm_scans',2),
                                          confirm_window_s=self.settings.get('person_obstacle_confirm_window_s',.5),
                                          person_margin_m=self.settings.get('person_obstacle_margin_m',.2))
        self.worker=worker_factory(self.planning,geometry)
        self.events=deque(maxlen=512);self.sequence=0;self.epoch=0;self.context=None
        self.latest=None;self.current=None;self.reference=None;self.reason='waiting_for_observation'
        self.cancelled={};self.last_sent=-np.inf;self.next_retry=-np.inf;self.blocked=False;self.need_replan=False
        self.diagnostics=False;self.last_reply=None;self.start_evidence=None
        self.failure=None
        self.planning_observation=None
        self.start_status=None
        self.start_occupied_stamp_s=None
        self.display_enabled=False
        self.display_subscribed=None
        self._display_reference_map=None
        self._display_failure_map=None
        self.retry_after_observation=None
        self.control_blocked_since=None
        self.follow_key=None
        self.follow_revision=-1

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.worker.stop()
    def _event(self,event,**fields):self.events.append(dict(event=event,epoch=self.epoch,monotonic_s=time.monotonic(),**fields))
    def drain_events(self):
        rows=list(self.events);self.events.clear();return rows

    @property
    def display_map(self):
        """Snapshot for the current displayed reference or the exact pending goal failure."""
        if not self.display_enabled or self.latest is None:return None
        if self.current is not None:
            snapshot=self._display_reference_map
        elif (self._display_failure_map is not None and self.failure is not None and
                self.failure['goal_binding']==goal_binding(self.latest['request']) and
                0<=self.latest['scene'].record['as_of']-self.failure['as_of']<=self.settings['max_age_s']):
            snapshot=self._display_failure_map
        else:
            snapshot=None
        return None if snapshot is None else dict(snapshot)

    def invalidate(self,reason):
        if reason in ('seek','localization_reset'):
            self.obstacles.reset()
        if reason=='localization_reset' and self.latest is not None:
            req=self.latest['request'];self.cancelled[req['task_id']]=req['goal_revision']
        self.epoch+=1;self.latest=self.current=self.reference=None;self.context=None
        self.follow_key=None;self.follow_revision=-1
        self.reason=reason;self.blocked=False;self.need_replan=False;self.last_sent=self.next_retry=-np.inf
        self.failure=None
        self.planning_observation=None
        self.start_status=None
        self.start_occupied_stamp_s=None
        self._display_reference_map=self._display_failure_map=None
        self.retry_after_observation=None
        self.control_blocked_since=None
        if self.worker.job is not None:self.worker.restart()
        self.start_evidence=None
        self._event('invalidated',reason=reason)

    @staticmethod
    def _obstacle_key(scene,asset):
        r=scene.record;a=r['prior']
        return (str(getattr(asset,'directory','')),asset.manifest['snapshot_id'],asset.manifest['layers_sha256'],
                fingerprint(navigation_binding(asset)),r['clock_domain'],
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
        added=self.obstacles.update(scene,navigation_grid(asset),local_from_map,key)
        if self.obstacles.version!=version:
            self.need_replan=True
            self.next_retry=-np.inf
            self._display_failure_map=None
            self.planning_observation=None
            if added and self.reference is not None and not self._avoids_obstacles(self.reference,scene,asset):
                self.reference=self.current=None
                self._display_reference_map=None
                self.reason='reference_obstructed_by_new_observation'

    def _avoids_obstacles(self,envelope,scene,asset):
        blocked=self._retained_obstacles(scene,asset)
        if not blocked:return True
        result=envelope['result']
        work_grid=navigation_grid(asset)
        frame=work_grid.metadata['frame_id']
        local_approach=result['request'].get('use_local_approach') is True
        if result.get('reference_search_frame',result['frame_id'])!=frame and not local_approach:
            return True
        # Recheck just the new reference exclusions. The original result and the
        # controller still perform their own complete geometry checks.
        poses=self._path_in_frame(result,scene,frame)
        empty=GridSnapshot(work_grid.metadata,dict(occupancy=np.zeros(work_grid.metadata['shape'],np.int8)))
        excluded=retain_obstacles(empty,blocked)
        if local_approach:
            return self._avoids_grid_occupancy(poses,excluded)
        check=swept_check(excluded,self.geometry,poses,self.planning['safety_margin_m'],
                          self.planning['max_path_samples'],prefix=False)
        return check['status']=='free'

    def _path_in_frame(self,result,scene,frame):
        poses=np.asarray(result['path'])
        if result['frame_id']==frame:return poses
        from .planner import se2,transform_poses
        record=scene.record
        world=np.asarray(record['localization']['world_from_lidar'])
        matrix=np.asarray(record['prior']['map_from_local'])
        map_from_local=se2(self.geometry.pose(matrix@world))@np.linalg.inv(se2(self.geometry.pose(world)))
        transform=np.linalg.inv(map_from_local) if frame==record['frame_id'] else map_from_local
        return transform_poses(poses,transform)

    def _avoids_local_occupancy(self,envelope,scene,asset,now,*,grid=None):
        result=envelope['result']
        if (result['request'].get('use_local_approach') is not True or
                scene.record['as_of']<=envelope['as_of']):
            return True
        if grid is None:
            grid=navigation_geometry(scene,now=now,asset=asset,check_asset=True)
        return self._avoids_grid_occupancy(self._path_in_frame(result,scene,grid.metadata['frame_id']),grid)

    def _reference_inside_work_extent(self,envelope,scene,asset):
        result=envelope['result']
        # A continuous local-frame reference is reprojected by later map associations.
        if (result['request']['mode']!='local' or getattr(asset,'navigation_window',None) is None or
                result.get('reference_search_frame')!=asset.snapshot.metadata['frame_id']):
            return True
        from .planner import path_within_work_extent
        grid=navigation_grid(asset)
        poses=self._path_in_frame(result,scene,grid.metadata['frame_id'])
        return path_within_work_extent(grid,self.geometry,poses,self.planning['safety_margin_m'],
                                       self.planning['max_path_samples'])

    def _avoids_grid_occupancy(self,poses,grid):
        cells=grid.layers['occupancy']
        margin=self.planning['safety_margin_m']
        bounds=self.geometry.radius+margin
        resolution=grid.metadata['resolution']
        origin=np.asarray(grid.metadata['origin_xy'])
        lower=np.floor((poses[:,:2].min(axis=0)-bounds-origin)/resolution).astype(int)-1
        upper=np.floor((poses[:,:2].max(axis=0)+bounds-origin)/resolution).astype(int)+2
        lo=np.maximum(lower,0)
        hi=np.minimum(upper,cells.shape[::-1])
        if (lo>=hi).any():return True
        occupied=cells[lo[1]:hi[1],lo[0]:hi[0]]==100
        if not occupied.any():return True
        # An exclusion-only recheck can ignore areas outside this evidence raster.
        shape=upper-lower
        if np.prod(shape,dtype=np.float64)>self.planning['max_cells']:return False
        exclusions=np.zeros(shape[::-1],np.int8)
        x,y=lo-lower
        exclusions[y:y+occupied.shape[0],x:x+occupied.shape[1]]=100*occupied
        snapshot=GridSnapshot(dict(grid.metadata,shape=list(exclusions.shape),
            origin_xy=(origin+lower*resolution).tolist()),dict(occupancy=exclusions))
        check=swept_check(snapshot,self.geometry,poses,margin,self.planning['max_path_samples'],prefix=False)
        return check['status']=='free'

    def cancel(self,request=None):
        req=request or (None if self.latest is None else self.latest['request'])
        if req is not None:self.cancelled[req['task_id']]=req['goal_revision']
        self.invalidate('cancel')

    def update(self,request,scene,state,asset=None,*,now,clock_rate=1.,follow_key=None,locked_follow_goal=False,**unused):
        self.start_evidence=None
        self.start_status=None
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
                self._display_failure_map=None
                self.planning_observation=None
                self.reason=reason
                return False
            if reason:raise ValueError(reason)
            scene.verify()
            if (request['mode']=='mapped' or request.get('completion_policy')=='continuous') and (
                    not association_status(scene,asset,self.planning)):
                local_goal(request,scene,self.geometry,asset,self.planning)
            # Follow revisions are a replaceable pending goal within a single
            # owner/track and map context. Results still bind to exact goals.
            scope=dict(request,goal=[],goal_revision=0) if follow_key is not None else request
            context=context_binding(scope,scene,asset)
            new_request_id=self.latest is not None and request['request_id']!=self.latest['request']['request_id']
            if self.context is not None and (context!=self.context or follow_key!=self.follow_key or new_request_id):
                self.invalidate('context_changed')
            self.context=context
            self.follow_key=follow_key
            if follow_key is not None and (self.latest is None or
                    goal_binding(request)!=goal_binding(self.latest['request'])):
                if self.failure is not None:
                    self.failure=None;self.next_retry=-np.inf
                    self._display_failure_map=None
                self.planning_observation=None
                self._event('follow_goal_pending',goal_revision=request['goal_revision'],
                            goal=request['goal'],as_of=scene.record['as_of'])
            new_scene=self.latest is None or self.latest['scene'] is not scene
            self.latest=dict(request=copy.deepcopy(request),scene=scene,asset=asset,context=context,
                received_mono=time.monotonic(),clock_rate=clock_rate,locked_follow_goal=locked_follow_goal)
            pose=self.geometry.pose(scene.record['localization']['world_from_lidar'])
            grid=navigation_geometry(scene,now=now,asset=asset,check_asset=True,include_confirmed_free=True)
            start_status=footprint_status(grid,self.geometry,pose,self.planning['safety_margin_m'])
            self.start_status=start_status
            can_use_static=(request['mode']=='mapped' or request.get('completion_policy')=='continuous') and (
                not association_status(scene,asset,self.planning))
            # The worker still checks the whole start footprint on the mapped
            # reference grid. Only the local execution checks require fresh
            # local free evidence under the entire vehicle.
            self.blocked=start_status!='free' and not (can_use_static and start_status=='unknown')
            if start_status=='occupied':
                self.start_occupied_stamp_s=scene.record['as_of']
            if self.diagnostics or start_status=='occupied':
                from .footprint import footprint_details
                self.start_evidence=footprint_details(grid,self.geometry,pose,self.planning['safety_margin_m'],
                    source_snapshot=scene.geometry if start_status=='occupied' else None)
                self.start_evidence.update(status=start_status,pose=pose.tolist(),scene_binding=scene.record['input_fingerprint'],
                    frame_id=grid.metadata['frame_id'],source_stamp_s=scene.record['as_of'])
            if self.blocked:
                self.reason='start_footprint_not_free'
            elif self.reason!='reference_obstructed_by_new_observation':
                self.reason=None
            if (new_scene and self.reference is not None and
                    not self._avoids_local_occupancy(self.reference,scene,asset,now,grid=grid)):
                self.reference=self.current=self.planning_observation=None
                self._display_reference_map=None
                self.reason='reference_obstructed_by_new_observation'
                self.need_replan=True
                self.next_retry=-np.inf
            if new_scene and self.reference is not None and not self._reference_inside_work_extent(self.reference,scene,asset):
                self.reference=self.current=self.planning_observation=None
                self._display_reference_map=None
                self.reason='reference_outside_work_extent'
                self.need_replan=True
                self.next_retry=-np.inf
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
            if self.diagnostics:
                self.last_reply=(job,{key:value for key,value in reply.items() if key!='display_map'})
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
                    if 'reference_overlay' in result:
                        fields['retained_occupied_cells']=result['reference_overlay']['retained_occupied_cells']
            try:
                same_context=(self.latest is not None and job['epoch']==self.epoch and job['context']==self.context)
                if same_context and job['request']['request_id']==self.latest['request']['request_id']:
                    self.planning_observation=None
                if (self._display_failure_map is not None and same_context and
                        goal_binding(job['request'])==goal_binding(self.latest['request'])):
                    self._display_failure_map=None
                if reply['error']:raise ValueError(reply['error'])
                if not same_context:raise ValueError('superseded_context')
                if (self.start_occupied_stamp_s is not None and
                        job['scene'].record['as_of']<=self.start_occupied_stamp_s):
                    transient_input=True
                    raise ValueError('reference_predates_occupied_start')
                requested=self.latest['request'] if self.follow_key is None else job['request']
                check_reference(envelope,requested,self.latest['scene'],self.latest['asset'],self.settings['max_age_s'],require_path=False)
                requested_goal=goal_binding(requested)
                superseded=requested_goal!=goal_binding(self.latest['request'])
                if (job['request']['request_id']!=requested['request_id'] or
                        result['request'].get('request_id')!=requested['request_id'] or
                        result['request_binding']!=fingerprint(job['request']) or
                        result['input_binding']['scene']!=job['scene'].record['input_fingerprint'] or
                        envelope['as_of']!=job['scene'].record['as_of']):
                    raise ValueError('reference_source_mismatch')
                snapshot=reply.get('display_map') if self.display_enabled and job.get('display_enabled') else None
                if snapshot is not None and (not isinstance(snapshot,dict) or
                        snapshot.get('job_id')!=job['job_id'] or snapshot.get('reference_binding')!=envelope['binding'] or
                        snapshot.get('goal_binding')!=requested_goal or snapshot.get('stamp_s')!=envelope['as_of'] or
                        snapshot.get('frame_id')!=result.get('reference_search_frame')):
                    snapshot=None
                if result['status']!='path_found' or len(result.get('path',[]))<2:
                    if superseded:
                        transient_input=True
                        raise ValueError('pending_follow_goal_changed')
                    if job.get('reference_obstacle_version',self.obstacles.version)!=self.obstacles.version:
                        transient_input=True
                        raise ValueError('reference_observation_changed')
                    if result.get('reason') in ('navigation_prior_future_or_stale','geometry_future_or_stale'):
                        transient_input=True;self.failure=None;self.planning_observation=None
                        raise ValueError(result['reason'])
                    # A negative result withdraws the path for that goal. A
                    # compatible active path may still work up to its endpoint
                    # while a different pending follow goal is unreachable.
                    active=self.reference
                    if (result['status']=='no_path' or 'follow_last_evidence_binding' in requested) and (self.follow_key is None or active is None or
                            goal_binding(active['result']['request'])==goal_binding(requested)):
                        self.reference=self.current=None
                        self._display_reference_map=None
                    self.failure=dict(request_binding=fingerprint(job['request']),reason=result['reason'],
                        goal_binding=goal_binding(job['request']),
                        as_of=job['scene'].record['as_of'],job_id=job['job_id'])
                    kind=blockage_kind(result,requested)
                    self.planning_observation=planning_observation(requested,job['scene'].record,
                        source='reference',status='blocked' if kind else 'unknown',reason=result['reason'],
                        max_age_s=self.settings['max_age_s'],kind=kind,result_binding=result['result_binding'])
                    position=blocked_position(result,requested) if kind else None
                    if position is not None:
                        self.planning_observation['blocked_position']=position
                    self._display_failure_map=snapshot
                    raise ValueError('reference_has_no_path')
                if (job.get('reference_obstacle_version',self.obstacles.version)!=self.obstacles.version and
                        not self._avoids_obstacles(envelope,self.latest['scene'],self.latest['asset'])):
                    transient_input=True
                    raise ValueError('reference_obstructed_by_new_observation')
                if not self._avoids_local_occupancy(envelope,self.latest['scene'],self.latest['asset'],now):
                    transient_input=True
                    raise ValueError('reference_obstructed_by_new_observation')
                if not self._reference_inside_work_extent(envelope,self.latest['scene'],self.latest['asset']):
                    transient_input=True
                    raise ValueError('reference_outside_work_extent')
                if self.follow_key is not None:
                    if not self._compatible_follow(requested,self.latest):
                        transient_input=True
                        raise ValueError('pending_follow_goal_changed')
                    self.follow_revision=requested['goal_revision']
                self.reference=self.current=reply['result'];self.need_replan=False;self.failure=None;self.retry_after_observation=None
                self._display_reference_map=snapshot;self._display_failure_map=None
                self.planning_observation=planning_observation(requested,job['scene'].record,
                    source='reference',status='clear',reason=None,max_age_s=self.settings['max_age_s'],
                    result_binding=result['result_binding'])
                self._event('reference_ready',**fields)
            except (ValueError,KeyError,TypeError) as error:
                self.reason=str(error)
                self.retry_after_observation=job['scene'].record['as_of'] if transient_input else None
                self.next_retry=mono if transient_input else mono+self.settings['failure_retry_s']
                self._event('discarded',reason=str(error),transient_input=transient_input,
                    retry_after_s=max(0.,max(self.next_retry,self.last_sent+self._retry_period())-mono),**fields)
        if self.worker.job is not None and mono-self.worker.job['sent_mono']>self.settings['timeout_s']:
            self.worker.restart();self._event('timeout',role='reference');self.reason='reference_timeout'
            self._display_failure_map=None
            self.planning_observation=None
            self.next_retry=mono+self.settings['failure_retry_s']
        if not self.worker.ready and mono-self.worker.started>self.settings['startup_timeout_s']:
            self.worker.restart();self.reason='reference_worker_startup_timeout'
        if self.latest is None:return self.status(now=now,clock_domain=clock_domain)
        item=self.latest;scene=item['scene'];r=scene.record
        reason=scene_status(scene,now,clock_domain,self.planning)
        if reason or mono-item['received_mono']>self.planning['max_geometry_age_s']:
            self.current=None;self.reason=reason or 'observation_wall_stale'
            self._display_failure_map=None
            self.planning_observation=None
            self.start_status=None
            return self.status(now=now,clock_domain=clock_domain)
        if self.blocked and self.start_status=='occupied':
            self.reference=self.current=self.planning_observation=None
            self._display_reference_map=self._display_failure_map=None
            self.reason='start_footprint_not_free'
            return self.status(now=now,clock_domain=clock_domain)
        # A transient input outage must not permanently hide a retained valid
        # reference. Restore only after checking the original source time and
        # current task/map identity; DWA still certifies the current geometry.
        if self.reference is not None:
            try:
                accepted=self.reference['result']['request'] if self.follow_key is not None else item['request']
                if self.follow_key is not None and not self._compatible_follow(accepted,item):
                    self.reference=None
                    self._display_reference_map=None
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
            if self.display_enabled and (self.display_subscribed is None or self.display_subscribed()):
                job['display_enabled']=True
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
        if item.get('locked_follow_goal'):return False
        scene,asset=item['scene'],item['asset']
        a=local_goal(request,scene,self.geometry,asset,self.planning)[:2]
        b=local_goal(latest,scene,self.geometry,asset,self.planning)[:2]
        base=self.geometry.pose(scene.record['localization']['world_from_lidar'])[:2]
        return bool(np.linalg.norm(a-b)<=self.settings['follow_goal_delta_m'] and (a-base)@(b-base)>=0.)

    def status(self,*,now,clock_domain):
        request=None
        evidence=None
        if self.latest is not None:
            active=self.latest['request'];record=self.latest['scene'].record
            observed=self.planning_observation
            if (record['clock_domain']==clock_domain and
                    self.reason not in ('geometry_future_or_stale','observation_wall_stale') and not self.blocked):
                if (observed is not None and observed['request_id']==active['request_id'] and
                        observed['goal_binding']==goal_binding(active) and
                        observed['source_stamp_s']<=now<=observed['valid_until_s'] and
                        (observed['status']!='clear' or self.current is not None)):
                    evidence=observed
            elif (self.start_status=='occupied' and self.reason=='start_footprint_not_free' and
                    record['clock_domain']==clock_domain and
                    0<=now-record['as_of']<=self.planning['max_geometry_age_s']):
                evidence=planning_observation(active,record,source='reference',status='blocked',
                    reason='start_footprint_occupied',max_age_s=self.planning['max_geometry_age_s'],
                    kind='route_blocked')
        if self.current is not None and self.follow_key is not None:
            # Refresh only the tracking lease from the current verified track.
            # The original path, goal and observation timestamps stay intact.
            request=dict(self.current['result']['request'],valid_until_s=self.latest['request']['valid_until_s'])
        return dict(status='reference_ready' if self.current is not None else 'unavailable',reason=None if self.current is not None else self.reason or 'waiting_for_background_reference',
            result=self.current,request=request,executable=False,task_completed=False,event_overflow=0,start_evidence=self.start_evidence,
            planning_observation=evidence,
            waiting=dict(worker_ready=self.worker.ready,busy=self.worker.job is not None,blocked=self.blocked,
                retry_remaining_s=max(0.,self.next_retry-time.monotonic()),epoch=self.epoch))
