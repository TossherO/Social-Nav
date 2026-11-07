"""Source-bound demonstration goals for bag shadow replay, never execution feedback."""
import json
import copy
from pathlib import Path

import numpy as np

from perception.mapping_localization.results import file_fingerprint
from perception.environment.context import fingerprint
from perception.environment.navigation import navigation_geometry
from .continuity import local_goal
from .footprint import goal_obstruction
from .inputs import scene_status
from .route import RoutePlan, _identity


class RouteReplay:
    """Select recorded time windows; match progress only against current observations."""
    def __init__(self,path,bag,floor,geometry,config,settings,asset):
        self.path=Path(path).resolve();record=json.loads(self.path.read_text())
        self.route=RoutePlan(record);source=record['source']
        if (not isinstance(source,dict) or source.get('bag_fingerprint')!=file_fingerprint(bag) or
                source.get('not_blind_evaluation') is not True):
            raise ValueError('Route replay needs the same bag and an explicit development reference')
        self.origin=float(source['source_start_s'])
        if not np.isfinite(self.origin):raise ValueError('Invalid route source clock')
        self.legs=[s for s in self.route.segments.values() if s.get('floor_id')==floor]
        if not self.legs:raise ValueError('No geometric route legs for the selected floor')
        end=-np.inf
        for leg in self.legs:
            a,b=(leg['source'].get(k) for k in ('start_offset_s','end_offset_s'))
            if (type(a) not in (int,float) or type(b) not in (int,float) or
                    not np.isfinite([a,b]).all() or not end<=a<b):
                raise ValueError('Route replay requires ordered nonoverlapping recorded windows')
            end=b
            if leg['mode']=='mapped' and (asset is None or asset.snapshot.metadata['map_id']!=leg['map_id']):
                raise ValueError('Select the static environment map for this route floor')
        self.floor=floor;self.geometry=geometry;self.config=config;self.settings=dict(settings);self.asset=asset
        self.task_id=record['route_id']+':'+floor;self.revision=-1;self.current=None
        self.identity=None;self.clock=None;self.disabled=None;self.segment=None;self.progress=0.
        self.anchors={};self.last_stamp=-np.inf;self.last_scene=None;self.last_submit=-np.inf
        self.events=[];self.last_status=None;self.goals=0;self.cancels=0
        self.pending_goals=[];self.search_remaining=0.
        self.has_transition=any(s['mode']=='transition' and s['from_floor']==floor for s in self.route.segments.values())

    def _event(self,status,offset,**fields):
        key=(self.segment,status)
        if key!=self.last_status or fields:
            self.events.append(dict(event='route_replay',status=status,segment_id=self.segment,
                source_offset_s=offset,task_completed=False,execution_feedback=False,executable=False,**fields))
        self.last_status=key

    def disable(self,reason):
        self.disabled=reason

    def _pause(self,reason,offset):
        self._event(reason,offset)
        if self.current is None:return None
        # A short measurement-age failure pauses proposals through their own
        # input checks. It does not revoke the still-identical route task.
        if (reason in ('navigation_prior_future_or_stale','geometry_future_or_stale') and
                self.current['route_segment_id']==self.segment):return None
        value={k:self.current[k] for k in ('task_id','goal_revision')}
        self.current=None;self.last_submit=-np.inf;self.cancels+=1;self.pending_goals=[]
        return 'cancel',value

    def _publish_next(self,scene,stamp,now,clock_domain):
        reason=scene_status(scene,now,clock_domain,self.config)
        if reason:raise ValueError(reason)
        grid=navigation_geometry(scene,now=now,asset=self.asset,check_asset=True)
        offset=stamp-self.origin;selected=None;rejected=[]
        original=self.pending_goals[0] if self.pending_goals else None
        while self.pending_goals:
            candidate=self.pending_goals.pop(0)
            goal=local_goal(candidate,scene,self.geometry,self.asset,self.config)
            obstruction=goal_obstruction(grid,self.geometry,goal,self.config['safety_margin_m'])
            if obstruction is None:
                selected=candidate;break
            rejected.append(dict(goal=candidate['goal'],reason='goal_footprint_'+obstruction))
        if selected is None and rejected:
            # Preserve an explicit failed goal when every candidate is blocked;
            # do not cancel the task to manufacture better availability counts.
            if self.current is not None and self.current['route_segment_id']==self.segment:
                self._event('goal_candidates_blocked',offset,rejected=rejected);return None
            selected=original
        if selected is None:return None
        candidate=copy.deepcopy(selected);self.revision+=1
        candidate['goal_revision']=self.revision
        candidate['request_id']='%s:%s:%d'%(self.route.record['route_id'],self.segment,self.revision)
        if candidate['goal_behavior']!='stop':
            budget=self.search_remaining/(1+len(self.pending_goals));self.search_remaining-=budget
            target=candidate['route_target'];settings=self.config['route_goals']
            pose=self.geometry.pose(scene.record['localization']['world_from_lidar'])
            distance=max(np.linalg.norm(local_goal(candidate,scene,self.geometry,self.asset,self.config)[:2]-pose[:2]),
                target['target_progress_m']-target['progress_m'])
            candidate['route_goal_selection']=dict(search_budget_s=budget,max_path_length_m=None if target['terminal'] else
                float(settings['max_detour_ratio']*distance+settings['max_detour_extra_m']))
        self.current=candidate;self.last_submit=stamp;self.goals+=1
        anchor_id=self.route.segments[self.segment].get('anchor_id')
        self._event('goal',offset,request=candidate,progress_m=self.progress,rejected_candidates=rejected,
            anchor=None if anchor_id is None else dict(observation=self.anchors[anchor_id].observation_binding,
                stamp_s=self.anchors[anchor_id].stamp_s,local_from_anchor=self.anchors[anchor_id].local_from_anchor))
        return 'goal',candidate

    def step(self,scene,state,*,now,clock_domain,failure=None):
        stamp=float(state['stamp_s']);offset=stamp-self.origin
        if not np.isfinite(stamp):raise ValueError('Invalid recorded observation time')
        if stamp<self.last_stamp:self.disabled='recorded_clock_reversed'
        self.last_stamp=stamp
        if self.disabled:return self._pause(self.disabled,offset)
        identity=tuple(state[k] for k in ('session_id','generation','frame_id'))
        if self.identity is not None and (identity!=self.identity or self.clock!=clock_domain):
            self.anchors.clear();self.disabled='localization_reset_requires_route_restart'
            return self._pause(self.disabled,offset)
        leg=next((s for s in self.legs if s['source']['start_offset_s']<=offset<s['source']['end_offset_s']),None)
        if leg is None:
            reason=('waiting_for_external_confirmation' if self.has_transition else 'route_window_finished') if offset>=self.legs[-1]['source']['end_offset_s'] else 'waiting_for_route_window'
            return self._pause(reason,offset)
        changed=self.segment!=leg['segment_id']
        if changed:self.segment=leg['segment_id'];self.progress=0.;self.last_scene=None;self.pending_goals=[]
        if scene is None or state['health']!=0:return self._pause('route_observation_unavailable',offset)
        if (scene.record['as_of']<self.origin+leg['source']['start_offset_s'] or
                scene.record['as_of']>stamp+1e-6 or _identity(scene)!=identity):
            return self._pause('waiting_for_route_observation',offset)
        self.identity=identity;self.clock=clock_domain
        recoverable=('goal_footprint_unknown','goal_footprint_occupied','goal_footprint_outside',
            'goal_unknown','goal_occupied','returned_path_no_curvature_route',
            'returned_path_search_budget_exceeded','route_candidate_detour')
        failed=(not changed and failure is not None and self.current is not None and
            failure['request_binding']==fingerprint(self.current) and failure['reason'] in recoverable and
            failure['as_of']<=scene.record['as_of'])
        # Exhausted alternatives keep the current goal. ReferenceRuntime owns
        # retries against fresh observations; map changes never restart this list.
        if failed and self.pending_goals:
            try:
                action=self._publish_next(scene,stamp,now,clock_domain)
                if action is not None:return action
            except ValueError as error:return self._pause(str(error),offset)
        if scene.record['input_fingerprint']==self.last_scene:return None
        self.last_scene=scene.record['input_fingerprint']
        # Bind only the next local leg, using a fresh pre-exit map association.
        upcoming=next((s for s in self.legs if s['mode']=='local' and s['source']['start_offset_s']>=offset),None)
        anchor_id=leg.get('anchor_id')
        target=None
        if leg['mode']=='mapped' and upcoming:target=upcoming['anchor_id']
        elif anchor_id and anchor_id not in self.anchors:target=anchor_id
        if target:
            try:self.anchors[target]=self.route.bind_anchor(target,scene,now=now,clock_domain=clock_domain,config=self.config)
            except ValueError:pass  # An unavailable association cannot create an anchor.
        try:
            result=self.route.request(self.segment,scene,self.geometry,task_id=self.task_id,
                goal_revision=self.revision+1,now=now,clock_domain=clock_domain,config=self.config,
                progress_m=self.progress,lookahead_m=self.settings['lookahead_m'],asset=self.asset,
                anchor=self.anchors.get(anchor_id),alternatives=True)
        except ValueError as error:return self._pause(str(error),offset)
        self.progress=result['progress_m'];candidate=result['request']
        if self.current is not None and self.current['route_segment_id']==self.segment:
            shift=np.linalg.norm(np.asarray(candidate['goal'])-self.current['goal'])
            target=self.current.get('route_target',{})
            passed=(not target.get('terminal',True) and result['cross_track_m']<=self.config['route_goals']['step_m'] and
                self.progress>target['target_progress_m']+1e-3)
            if not passed and (stamp-self.last_submit<self.settings['min_update_s'] or
                    shift<self.settings['goal_shift_m'] or
                    self.progress-target['progress_m']<self.settings['goal_shift_m']):return None
        self.pending_goals=copy.deepcopy(result['candidates'])
        self.search_remaining=self.config['pose_search']['max_search_s']
        try:return self._publish_next(scene,stamp,now,clock_domain)
        except ValueError as error:return self._pause(str(error),offset)

    def drain_events(self):
        events=self.events;self.events=[];return events

    def summary(self):
        return dict(route=str(self.path),route_binding=self.route.binding,floor_id=self.floor,
            source=self.route.record['source'],settings=self.settings,goals=self.goals,cancels=self.cancels,
            disabled=self.disabled,status=self.last_status,task_completed=False,execution_feedback=False,
            scope='recorded_demonstration_shadow_replay_not_online_route_execution')
