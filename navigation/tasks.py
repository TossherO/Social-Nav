"""Observation-bound navigation policies for validated decision responses.

No model calls or agent loop. Outputs are requests to the existing planner, or
stop requests; neither a selected candidate nor a reached pose completes T3.
Only continuous local-frame candidates are accepted (not frozen camera/LiDAR
coordinates). The caller supplies the PID mapping used to render that input.
"""
import copy
import numpy as np
from perception.environment.context import fingerprint
from navigation.planning.people import PeopleModel,social_groups


def identity(scene):
    return [scene.record[k] for k in ('session_id','generation','frame_id')]


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

    def update(self,scene,*,now):
        """Re-evaluate a follow/hold on every input; this never calls the VLM."""
        scene.verify()
        if self.cancelled:return self._stop('task_cancelled')
        if self.latest is None or identity(scene)!=self.latest['local_identity'] or scene.record['clock_domain']!=self.latest['clock_domain']:
            return self._stop('task_localization_changed')
        if self.kind=='social':
            if self.groups is None:return self._stop('no_social_context')
            _,status=social_groups(dict(social=self.groups),scene,PeopleModel(scene,self.config['people']))
            return dict(operation='social_context',social=self.groups if status['status']=='valid' else None,status=status,executable=False)
        if self.kind!='follow':raise ValueError('Only follow and social have per-observation policy updates')
        if self.target is None:return self._stop('follow_target_unavailable')
        if self.response_stamp is None or not 0<=now-self.response_stamp<=self.hold_s:
            return self._stop('follow_decision_expired')
        model=PeopleModel(scene,self.config['people'])
        if self.target not in model.ids:
            self.target=None
            return self._stop('follow_track_lost')
        track=model.tracks[model.ids.index(self.target)]
        if now-self.last_visual>self.hold_s:return self._stop('follow_visual_confirmation_expired')
        if not 0<=now-scene.record['as_of']<=self.config['max_geometry_age_s']:return self._stop('follow_scene_future_or_stale')
        stamp=scene.record['as_of']
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
        request['valid_until_s']=min(self.response_stamp+self.hold_s,self.last_visual+self.hold_s)
        return dict(operation='plan',request=copy.deepcopy(request),target_identity=list(self.target),
            raw_response=self.raw,task_completed=False,executable=False)
