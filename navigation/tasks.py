"""Observation-bound navigation policies for validated decision responses.

No model calls or agent loop. Outputs are requests to the existing planner, or
stop requests; neither a selected candidate nor a reached pose completes T3.
Only continuous local-frame candidates are accepted (not frozen camera/LiDAR
coordinates). The caller supplies the PID mapping used to render that input.
"""
import copy
from collections import deque
import numpy as np
from perception.environment.context import fingerprint
from navigation.planning.people import PeopleModel,social_groups


def identity(scene):
    return [scene.record[k] for k in ('session_id','generation','frame_id')]


class FollowTrackHistory:
    """Bounded evidence connecting a visual selection to subsequent 3D tracks."""
    def __init__(self, planning, *, seconds=30., capacity=600):
        self.config=planning;self.seconds=seconds;self.rows=deque(maxlen=capacity)

    def add(self, scene):
        scene.verify();r=scene.record;model=PeopleModel(scene,self.config['people'])
        from navigation.planning.inputs import scene_status
        try:valid_scene=scene_status(scene,r['as_of'],r['clock_domain'],self.config,check_navigation=False) is None
        except (ValueError,KeyError,TypeError):valid_scene=False
        if self.rows and (r['clock_domain']!=self.rows[-1]['clock_domain'] or
                identity(scene)!=self.rows[-1]['local_identity'] or r['as_of']<self.rows[-1]['stamp_s']):
            self.rows.clear()
        if self.rows and r['as_of']==self.rows[-1]['stamp_s']:
            if r['input_fingerprint']!=self.rows[-1]['scene_binding']:self.rows.clear()
            else:return
        data=(r.get('people') or {}).get('data') or {}
        tracks={}
        if model.status=='valid' and valid_scene:
            for raw in data.get('tracks',[]):
                from navigation.planning.people import stable_identity
                key=stable_identity(data,raw)
                if key in model.ids and raw.get('confirmed') and type(raw.get('lifecycle_generation')) is int:
                    tracks[key]=(raw['lifecycle_generation'],raw.get('state'),raw.get('last_observed_stamp_s'))
        self.rows.append(dict(stamp_s=r['as_of'],clock_domain=r['clock_domain'],local_identity=identity(scene),
            scene_binding=r['input_fingerprint'],people_fingerprint=model.source,people_stamp_s=model.origin,tracks=tracks))
        while self.rows and r['as_of']-self.rows[0]['stamp_s']>self.seconds:self.rows.popleft()

    def validate(self, reference, scene, *, now):
        if (not isinstance(reference,dict) or reference.get('reference_id')!=
                fingerprint({k:v for k,v in reference.items() if k!='reference_id'})):
            raise ValueError('follow_reference_invalid')
        track=reference.get('track')
        if not isinstance(track,dict):raise ValueError('follow_reference_has_no_3d_track')
        key=tuple(track.get('identity') or ());generation=track.get('lifecycle_generation')
        rows=list(self.rows);start=next((i for i,row in enumerate(rows) if
            row['scene_binding']==reference.get('scene_binding') and row['stamp_s']==reference.get('stamp_s')),None)
        if start is None:raise ValueError('follow_reference_history_unavailable')
        end=next((i for i,row in enumerate(rows) if row['scene_binding']==scene.record['input_fingerprint']),None)
        if end is None or end<start:raise ValueError('follow_reference_history_unavailable')
        rows=rows[start:end+1];r=scene.record
        rgb=reference.get('rgb_stamp_s')
        if (type(rgb) not in (int,float) or not np.isfinite(rgb) or not 0<=rows[0]['stamp_s']-rgb<=.15 or
                rows[0]['people_fingerprint']!=track.get('people_fingerprint') or
                rows[0]['people_stamp_s']!=track.get('people_stamp_s') or
                reference.get('local_identity')!=identity(scene) or reference.get('clock_domain')!=r['clock_domain'] or
                not rows or rows[-1]['scene_binding']!=r['input_fingerprint'] or
                not 0<=now-r['as_of']<=self.config['max_geometry_age_s']):
            raise ValueError('follow_reference_context_or_scene_invalid')
        source=rows[0]['tracks'].get(key)
        if (source is None or source[1]!='observed' or source[2]!=track.get('support_stamp_s')):
            raise ValueError('follow_reference_support_mismatch')
        self._check(rows,key,generation)
        return key,generation

    def continue_from(self, binding, key, generation, scene):
        rows=list(self.rows)
        start=next((i for i,row in enumerate(rows) if row['scene_binding']==binding),None)
        end=next((i for i,row in enumerate(rows) if row['scene_binding']==scene.record['input_fingerprint']),None)
        if start is None or end is None or end<start:
            raise ValueError('follow_track_history_unavailable')
        self._check(rows[start:end+1],key,generation)

    def _check(self, rows, key, generation):
        previous=None;last_support=None
        for row in rows:
            value=row['tracks'].get(key)
            if (value is None or value[0]!=generation or value[1] not in ('observed','predicted') or
                    type(value[2]) not in (int,float) or not np.isfinite(value[2]) or
                    not 0<=row['stamp_s']-value[2]<=self.config['people']['max_track_age_s'] or
                    last_support is not None and value[2]<last_support or
                    previous is not None and row['stamp_s']-previous>self.config['people']['max_observation_age_s']):
                raise ValueError('follow_track_continuity_unavailable')
            previous=row['stamp_s'];last_support=value[2]


class TaskPolicy:
    def __init__(self,task_id,kind,planning,geometry,*,follow_distance_m=1.5,goal_update_m=.25,occlusion_hold_s=None):
        if occlusion_hold_s is None:
            from decision.vlm.project_config import load_config
            occlusion_hold_s=load_config()['tasks']['follow']['visual_memory_s']
        if not isinstance(task_id,str) or not task_id or kind not in ('social','follow','seating'):
            raise ValueError('Explicit task and supported policy required')
        if any(not np.isfinite(v) or v<=0 for v in (follow_distance_m,goal_update_m,occlusion_hold_s)):
            raise ValueError('Invalid task policy setting')
        self.task_id,self.kind,self.config,self.geometry=task_id,kind,planning,geometry
        self.distance,self.update_distance,self.hold_s=follow_distance_m,goal_update_m,occlusion_hold_s
        self.latest=None;self.sequence=0;self.target=None;self.last_visual=None;self.response_stamp=None
        self.revision=0;self.request=None;self.raw=None;self.groups=None;self.cancelled=False
        self.consumed=None
        self.last_update=None;self.response_binding=None
        self.tracked_generation=None;self.track_scene_binding=None

    def start_continuous_follow(self, reference, scene, history, *, now):
        """Start only after proving continuity from the original selected frame."""
        if self.kind!='follow' or self.cancelled or self.target is not None:
            raise ValueError('follow_policy_not_available')
        if not isinstance(reference,dict) or reference.get('task_id')!=self.task_id:
            raise ValueError('follow_reference_task_mismatch')
        target,generation=history.validate(reference,scene,now=now)
        self.observe(scene)
        self.target=target;self.tracked_generation=generation
        self.last_visual=reference['rgb_stamp_s'];self.response_stamp=reference['stamp_s']
        self.response_binding=reference['reference_id'];self.raw=None;self.track_scene_binding=scene.record['input_fingerprint']
        return self.update(scene,now=now,history=history)

    def observe(self,scene,*,pid_to_track=None,candidates=None):
        scene.verify();r=scene.record
        if self.latest is not None and identity(scene)!=self.latest['local_identity']:
            self.target=self.last_visual=self.response_stamp=self.request=self.groups=None
            self.last_update=self.response_binding=None
        if self.latest is not None and (r['clock_domain']!=self.latest['clock_domain'] or
                r['as_of']<self.latest['stamp_s']):raise ValueError('Task observation clock changed or moved backwards')
        model=PeopleModel(scene,self.config['people']);mapping=pid_to_track or {}
        if any(type(k) is not int or k<0 or tuple(v) not in model.ids for k,v in mapping.items()):
            raise ValueError('PID mapping must bind current stable tracker identities')
        if len(set(tuple(v) for v in mapping.values()))!=len(mapping):raise ValueError('Duplicate PID identity')
        if candidates is not None:
            from decision.vlm.seating.inference import validate_candidate_set
            validate_candidate_set(candidates)
            if (candidates['observation_id']!=r['observation_id'] or candidates['stamp_s']!=r['as_of'] or
                    candidates['clock_domain']!=r['clock_domain'] or
                    [candidates[k] for k in ('session_id','generation','frame_id')]!=identity(scene)):
                raise ValueError('Candidates must use this observation and continuous local frame')
        self.sequence+=1
        ticket=dict(task_id=self.task_id,kind=self.kind,sequence=self.sequence,scene_binding=r['input_fingerprint'],
            observation_id=r['observation_id'],stamp_s=r['as_of'],clock_domain=r['clock_domain'],local_identity=identity(scene),
            people_source=model.source,pid_to_track={str(k):list(v) for k,v in mapping.items()},candidates=copy.deepcopy(candidates))
        if self.kind=='follow':
            pid=next((k for k,v in mapping.items() if tuple(v)==self.target),None)
            ticket['history']=dict(confirmed_before=self.last_visual is not None,current_pid=pid,
                hold_allowed=bool(pid is not None and self.last_visual is not None and
                                  0<=r['as_of']-self.last_visual<=self.hold_s))
        ticket['binding']=fingerprint(ticket);self.latest=copy.deepcopy(ticket)
        return ticket

    def _bound(self,ticket,scene,now):
        scene.verify();r=scene.record
        if self.cancelled or ticket!=self.latest or self.consumed==ticket['binding'] or ticket['binding']!=fingerprint({k:v for k,v in ticket.items() if k!='binding'}):
            raise ValueError('Late, cancelled or modified decision ticket')
        if (ticket['local_identity']!=identity(scene) or ticket['clock_domain']!=r['clock_domain'] or
                not ticket['stamp_s']<=r['as_of']<=now or
                now-ticket['stamp_s']>self.config['people']['decision_max_age_s']):raise ValueError('Decision observation expired or reset')

    def cancel(self):
        self.cancelled=True;self.request=None;self.target=None;self.groups=None
        return self._stop('task_cancelled')

    def _stop(self,reason):
        self.request=None
        return dict(operation='stop',reason=reason,task_id=self.task_id,task_completed=False,executable=False)

    def respond(self,ticket,response,*,scene,now):
        if self.kind=='follow' and self.tracked_generation is not None:
            raise ValueError('continuous_follow_requires_separate_visual_revalidation')
        self._bound(ticket,scene,now);mapping=ticket['pid_to_track'];pids=[int(k) for k in mapping]
        # A malformed response is retained by the caller; it is not repaired or
        # silently resubmitted. A ticket permits one response attempt.
        self.consumed=ticket['binding'];self.raw=copy.deepcopy(response)
        if self.kind=='social':
            from decision.vlm.vlm_api_common import validate_groups
            validate_groups(response,pids)
            groups=[[mapping[str(pid)] for pid in group] for group_list in response.values() for group in group_list]
            value=dict(source=ticket['binding'],stamp_s=ticket['stamp_s'],clock_domain=ticket['clock_domain'],
                local_identity=ticket['local_identity'],groups=groups)
            value['binding']=fingerprint(value);self.groups=value;self.raw=copy.deepcopy(response)
            return dict(operation='social_context',social=value,raw_response=self.raw,task_completed=False,executable=False)
        if self.kind=='follow':
            from decision.vlm.follow import follow_schema,validate_decision
            # Validate against the history actually sent, then apply current
            # geometry/track/age gates below. Receipt never refreshes source time.
            validate_decision(response,follow_schema(pids,ticket['history']));self.raw=copy.deepcopy(response)
            if response['state'] in ('uncertain','lost','not_found'):
                self.target=None;return self._stop('follow_'+response['state'])
            selected=tuple(mapping[str(response['id'])])
            if response['state']=='visible':
                self.target=selected;self.last_visual=ticket['stamp_s'];self.last_update=None
            elif selected!=self.target:raise ValueError('Occluded response cannot change target')
            self.response_stamp=ticket['stamp_s']
            self.response_binding=ticket['binding']
            return self.update(scene,now=now)
        from decision.vlm.seating.inference import validate_selection,binding
        candidates=ticket['candidates']
        if candidates is None:raise ValueError('Missing seating candidate set')
        validate_selection(response,candidates);self.raw=copy.deepcopy(response)
        if response['s']!='v':return self._stop('no_suitable_seating_candidate')
        chosen=next(c for c in candidates['candidates'] if c['id']==response['p'])
        if chosen['geometry']=='blocked':return self._stop('selected_candidate_blocked')
        if chosen['motion']=='wait':return self._stop('selected_candidate_wait')
        request=self._goal(chosen['pose'],scene,source=ticket['binding'],stop=True)
        request.update(candidate_binding=binding(candidates),candidate_id=chosen['id'],
            motion_intent='reverse_required' if chosen['motion']=='reverse' else 'forward_preferred')
        self.request=request
        return dict(operation='plan',request=copy.deepcopy(request),requires_geometry_verification=True,
            raw_response=self.raw,task_completed=False,executable=False)

    def _goal(self,goal,scene,*,source,stop):
        r=scene.record
        changed=(self.kind!='follow' or self.request is None or np.linalg.norm(np.asarray(self.request['goal'])[:2]-goal[:2])>=self.update_distance or
            len(goal)!=len(self.request['goal']) or len(goal)==3 and abs(np.arctan2(np.sin(goal[2]-self.request['goal'][2]),np.cos(goal[2]-self.request['goal'][2])))>.1)
        if changed:self.revision+=1
        else:goal=self.request['goal']
        return dict(task_id=self.task_id,request_id=self.task_id+':'+str(self.revision),goal_revision=self.revision,
            mode='local',frame_id=r['frame_id'],target_reference='base_link',goal=list(map(float,goal)),
            goal_behavior='stop' if stop else 'pass_through',decision_source=source,
            completion_policy='continuous' if self.kind=='follow' else 'goal',
            local_identity=identity(scene),clock_domain=r['clock_domain'])

    def update(self,scene,*,now,history=None):
        """Re-evaluate a follow/hold on every input; this never calls the VLM."""
        scene.verify()
        if self.cancelled:return self._stop('task_cancelled')
        if self.latest is None or identity(scene)!=self.latest['local_identity'] or scene.record['clock_domain']!=self.latest['clock_domain']:
            if self.tracked_generation is not None:self.target=None
            return self._stop('task_localization_changed')
        if self.kind=='social':
            if self.groups is None:return self._stop('no_social_context')
            _,status=social_groups(dict(social=self.groups),scene,PeopleModel(scene,self.config['people']))
            return dict(operation='social_context',social=self.groups if status['status']=='valid' else None,status=status,executable=False)
        if self.kind!='follow':raise ValueError('Only follow and social have per-observation policy updates')
        if self.target is None:return self._stop('follow_target_unavailable')
        if self.tracked_generation is None and (self.response_stamp is None or not 0<=now-self.response_stamp<=self.hold_s):
            return self._stop('follow_decision_expired')
        model=PeopleModel(scene,self.config['people'])
        if self.target not in model.ids:
            self.target=None
            return self._stop('follow_track_lost')
        track=model.tracks[model.ids.index(self.target)]
        if self.tracked_generation is None and now-self.last_visual>self.hold_s:return self._stop('follow_visual_confirmation_expired')
        if not 0<=now-scene.record['as_of']<=self.config['max_geometry_age_s']:
            if self.tracked_generation is not None:self.target=None
            return self._stop('follow_scene_future_or_stale')
        stamp=scene.record['as_of']
        if self.tracked_generation is not None:
            from navigation.planning.inputs import scene_status
            try:invalid=scene_status(scene,now,scene.record['clock_domain'],self.config,check_navigation=False)
            except (ValueError,KeyError,TypeError):invalid='localization_or_geometry_invalid'
            if invalid:
                self.target=None
                return self._stop('follow_'+invalid)
            try:
                if history is None:raise ValueError('follow_track_history_unavailable')
                history.continue_from(self.track_scene_binding,self.target,self.tracked_generation,scene)
            except ValueError:
                self.target=None
                return self._stop('follow_track_continuity_lost')
            data=scene.record['people']['data']
            from navigation.planning.people import stable_identity
            raw=next((v for v in data['tracks'] if stable_identity(data,v)==self.target),None)
            if (raw is None or raw.get('lifecycle_generation')!=self.tracked_generation or not raw.get('confirmed') or
                    raw.get('state') not in ('observed','predicted') or
                    not 0<=now-track['last_observed']<=self.config['people']['max_track_age_s']):
                self.target=None
                return self._stop('follow_track_continuity_lost')
            self.track_scene_binding=scene.record['input_fingerprint']
        if self.last_update is not None and (stamp<self.last_update or
                stamp-self.last_update>self.config['people']['max_observation_age_s']):
            self.target=None
            return self._stop('follow_continuity_lost')
        self.last_update=stamp
        position=model.positions([scene.record['as_of']])[0][0,model.ids.index(self.target)]
        base=self.geometry.pose(scene.record['localization']['world_from_lidar'])
        direction=track['velocity']
        if direction is None or np.linalg.norm(direction)<.15:direction=position-base[:2]
        direction=np.asarray(direction)/max(np.linalg.norm(direction),1e-9)
        goal=position-self.distance*direction
        request=self._goal(goal,scene,source=self.response_binding,stop=True);self.request=request
        request['valid_until_s']=(min(stamp+self.config['max_geometry_age_s'],
            track['last_observed']+self.config['people']['max_track_age_s']) if self.tracked_generation is not None else
            min(self.response_stamp+self.hold_s,self.last_visual+self.hold_s))
        return dict(operation='plan',request=copy.deepcopy(request),target_identity=list(self.target),
            raw_response=self.raw,task_completed=False,executable=False)
