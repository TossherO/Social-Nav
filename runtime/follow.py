"""Control-thread follow orchestration; it never commands a chassis directly."""
import copy
from collections import deque

from navigation.tasks import FollowTrackHistory, TaskPolicy


class FollowRunner:
    def __init__(self, planning, geometry):
        self.planning,self.geometry=planning,geometry
        self.history=FollowTrackHistory(planning)
        self.policy=self.handle=self.reference=self.request=None
        self.occluded_since=None
        self.last_evidence=None
        self.t2_active=False
        self.revision_floor=0
        self.trace=deque(maxlen=30)

    def observe(self, scene):
        if scene is None:
            self.history.rows.clear()
        else:
            self.history.add(scene)

    def start(self, handle, reference, scene, *, now):
        if self.policy is not None:raise ValueError('follow_already_active')
        if (not isinstance(handle,dict) or not isinstance(reference,dict) or
                handle.get('task_id')!=reference.get('task_id') or
                handle.get('plan_revision')!=reference.get('task_revision') or
                handle.get('goal_id')!=reference.get('goal_id') or
                not isinstance(handle.get('call_id'),str) or not handle['call_id']):
            raise ValueError('follow_handle_reference_mismatch')
        if scene is None:raise ValueError('follow_scene_unavailable')
        policy=TaskPolicy(handle['task_id'],'follow',self.planning,self.geometry)
        policy.revision=self.revision_floor
        result=policy.start_continuous_follow(reference,scene,self.history,now=now)
        if result['operation']!='plan':raise ValueError(result['reason'])
        self.policy,self.handle,self.reference=policy,copy.deepcopy(handle),copy.deepcopy(reference)
        return self._translate(result,scene)

    def update(self, scene, *, now):
        if self.policy is None:return None
        if self.t2_active and not 0<=now-self.policy.last_visual<=self.policy.hold_s:
            self.policy.cancel()
            return dict(operation='stop',reason='follow_visual_confirmation_expired')
        if self.occluded_since is not None and now-self.occluded_since>self.policy.hold_s:
            self.policy.cancel()
            return dict(operation='stop',reason='follow_visual_occlusion_expired')
        if scene is None:
            self.policy.cancel()
            return dict(operation='stop',reason='follow_scene_unavailable')
        try:result=self.policy.update(scene,now=now,history=self.history)
        except (ValueError,KeyError,TypeError):
            self.policy.cancel()
            return dict(operation='stop',reason='follow_input_invalid')
        return self._translate(result,scene)

    def visual(self, value, scene, *, now):
        if self.policy is None or scene is None:raise ValueError('follow_not_active')
        if (value.get('identity')!=list(self.policy.target or ()) or value.get('state') not in ('visible','occluded')):
            raise ValueError('follow_visual_identity_mismatch')
        rows=list(self.history.rows)
        source=next((row for row in rows if row['scene_binding']==value.get('scene_binding')),None)
        rgb=value.get('rgb_stamp_s')
        if (source is None or type(rgb) not in (int,float) or
                not 0<=source['stamp_s']-rgb<=.15 or not 0<=now-rgb<=30.):
            raise ValueError('follow_visual_source_unavailable')
        self.history.continue_from(source['scene_binding'],self.policy.target,self.policy.tracked_generation,scene)
        if rgb<=self.policy.last_visual:raise ValueError('follow_visual_result_out_of_order')
        current=next((row for row in reversed(rows)
                      if row['scene_binding']==scene.record['input_fingerprint']),None)
        current_track=None if current is None else current['tracks'].get(self.policy.target)
        if (value['state']=='occluded' and current_track is not None and
                current_track[0]==self.policy.tracked_generation and current_track[1]=='observed' and
                current_track[2]>source['stamp_s'] and
                0<=now-self.policy.last_visual<=self.policy.hold_s):
            # This is not a fresh visual confirmation. Keep the prior deadline;
            # request a new visual judgment against the current image.
            return 'superseded'
        if value['state']=='visible':
            if self.occluded_since is not None and rgb<=self.occluded_since:raise ValueError('follow_visual_result_out_of_order')
            self.occluded_since=None;self.policy.last_visual=rgb
        elif self.occluded_since is None:
            self.occluded_since=rgb
        return 'accepted'

    def retarget(self, reference, scene, *, now, max_age_s=2.):
        """Switch only to a source-bound, continuously supported visual selection."""
        if self.policy is None or scene is None:raise ValueError('follow_not_active')
        if (not isinstance(reference,dict) or reference.get('task_id')!=self.handle['task_id'] or
                reference.get('task_revision')!=self.handle['plan_revision'] or
                reference.get('goal_id')!=self.handle['goal_id']):
            raise ValueError('follow_handle_reference_mismatch')
        rgb=reference.get('rgb_stamp_s')
        if (type(rgb) not in (int,float) or not 0<=now-rgb<=max_age_s or
                rgb<=self.policy.last_visual):
            raise ValueError('follow_visual_result_expired_or_out_of_order')
        key,generation=self.history.validate(reference,scene,now=now)
        # Check the entire transition before changing the active policy.
        if key!=self.policy.target:
            # A different person needs a fresh planning revision even when the
            # two predicted follow poses happen to coincide.
            self.policy.request=None
        self.policy.target=key
        self.policy.tracked_generation=generation
        self.policy.last_visual=rgb
        self.policy.response_stamp=reference['stamp_s']
        self.policy.response_binding=reference['reference_id']
        self.policy.track_scene_binding=scene.record['input_fingerprint']
        self.policy.last_update=None
        self.reference=copy.deepcopy(reference)
        self.occluded_since=None
        self.trace.clear()

    def apply_t2(self, handle, prepared, result, window, scene, *, now, deadline_s=None):
        """Apply one bound T2 choice; caller owns scheduling and checked search motion."""
        from decision.vlm.follow_online import selected_reference,parse_selection
        from decision.vlm.project_config import load_config
        from perception.environment.context import fingerprint
        from decision.vlm.annotation.rules import digest
        if deadline_s is None:deadline_s=load_config()['tasks']['follow']['decision_deadline_s']
        if (scene is None or not isinstance(handle,dict) or prepared.get('task')!=handle or
                prepared.get('binding')!=fingerprint({k:v for k,v in prepared.items() if k!='binding'}) or
                result.get('status')!='complete' or result.get('request_binding')!=prepared['binding'] or
                parse_selection(result.get('raw_content'),prepared.get('choices',{}))!=result.get('selection') or
                result.get('choice')!=prepared.get('choices',{}).get(result.get('selection')) or
                window.get('input_fingerprint')!=prepared.get('window_binding') or
                digest({k:v for k,v in window.items() if k!='input_fingerprint'})!=window.get('input_fingerprint') or
                window.get('scene',{}).get('input_fingerprint')!=prepared.get('scene_binding') or
                window.get('clock_domain')!=scene.record['clock_domain'] or
                not 0<=now-prepared.get('source_stamp_s',float('inf'))<=deadline_s):
            raise ValueError('follow_t2_result_invalid_or_expired')
        if self.handle is not None and self.handle!=handle:
            raise ValueError('follow_t2_handle_changed')
        choice=result['choice']
        if choice['kind']=='person':
            ref=selected_reference(prepared,result,window,task_id=handle['task_id'],
                                   revision=handle['plan_revision'],goal_id=handle['goal_id'])
            if self.policy is None:
                action=self.start(handle,ref,scene,now=now)
            else:
                self.retarget(ref,scene,now=now,max_age_s=deadline_s)
                action=self.update(scene,now=now)
            self.t2_active=True
            return dict(action,source_stamp_s=prepared['source_stamp_s'],t2_choice=result['selection'])
        if choice['kind']!='observation':raise ValueError('follow_t2_choice_invalid')
        if self.policy is not None:
            rows=list(self.history.rows)
            source=next((row for row in rows if row['scene_binding']==prepared['scene_binding']),None)
            current=next((row for row in reversed(rows) if row['scene_binding']==scene.record['input_fingerprint']),None)
            key=self.policy.target
            old=None if source is None else source['tracks'].get(key)
            new=None if current is None else current['tracks'].get(key)
            if (old is not None and new is not None and
                    new[0]==self.policy.tracked_generation and new[1]=='observed' and
                    new[2]>source['stamp_s'] and 0<=now-self.policy.last_visual<=self.policy.hold_s):
                self.history.continue_from(source['scene_binding'],key,self.policy.tracked_generation,scene)
                return dict(operation='continue',reason='follow_t2_occlusion_superseded',
                            request_fresh_observation=True,source_stamp_s=prepared['source_stamp_s'])
            self.policy.cancel();self.policy=None;self.request=None
        return dict(operation='search',candidate=copy.deepcopy(choice['candidate']),
                    search_binding=prepared['search_binding'],source_stamp_s=prepared['source_stamp_s'],
                    task_completed=False,executable=False)

    def _translate(self, result, scene):
        if result['operation']!='plan':return result
        from navigation.planning.people import PeopleModel,stable_identity
        from perception.environment.context import fingerprint
        model=PeopleModel(scene,self.planning['people']);data=scene.record['people']['data']
        key=tuple(result['target_identity'])
        raw=next((t for t in data['tracks'] if stable_identity(data,t)==key),None)
        if (raw is not None and raw.get('state')=='observed' and raw.get('confirmed') and key in model.ids and
                abs(data['stamp_s']-raw['last_observed_stamp_s'])<=1e-6):
            track=model.tracks[model.ids.index(key)]
            if not self.trace or self.trace[-1]['stamp_s']!=raw['last_observed_stamp_s']:
                self.trace.append(dict(stamp_s=raw['last_observed_stamp_s'],position_xy_m=track['xy'].tolist()))
            evidence=dict(identity=list(key),lifecycle_generation=raw['lifecycle_generation'],
                stamp_s=raw['last_observed_stamp_s'],clock_domain=scene.record['clock_domain'],
                local_identity=[scene.record[k] for k in ('session_id','generation','frame_id')],
                scene_binding=scene.record['input_fingerprint'],people_fingerprint=data['input_fingerprint'],
                position_xy_m=track['xy'].tolist(),velocity_xy_mps=None if track['velocity'] is None else track['velocity'].tolist(),
                recent_trace=copy.deepcopy(list(self.trace)))
            evidence['binding']=fingerprint(evidence);self.last_evidence=evidence
        request=copy.deepcopy(result['request'])
        # The changing local goal owns a separate internal revision namespace;
        # the agent's stable handle remains valid for cancellation and feedback.
        request['task_id']='follow:'+self.handle['call_id']
        request['request_id']=self.handle['call_id']
        self.request=request
        self.revision_floor=max(self.revision_floor,request['goal_revision'])
        return dict(operation='plan',request=copy.deepcopy(request),target_identity=result['target_identity'],
                    target_generation=self.policy.tracked_generation)

    def clear(self):
        if self.policy is not None:self.policy.cancel()
        self.policy=self.handle=self.reference=self.request=self.occluded_since=self.last_evidence=None
        self.t2_active=False
        self.revision_floor=0
        self.trace.clear()
