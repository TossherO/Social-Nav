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
        self.reid_reference_id=self.reid_stamp_ns=None
        self.reid_last_evidence=None
        self.revision_floor=0
        self.trace=deque(maxlen=30)

    def observe(self, scene):
        if scene is None:
            self.history.rows.clear()
        else:
            self.history.add(scene)

    def start(self, handle, reference, scene, *, now, visual_hold_s=None):
        if self.policy is not None:raise ValueError('follow_already_active')
        if (not isinstance(handle,dict) or not isinstance(reference,dict) or
                handle.get('task_id')!=reference.get('task_id') or
                handle.get('plan_revision')!=reference.get('task_revision') or
                handle.get('goal_id')!=reference.get('goal_id') or
                not isinstance(handle.get('call_id'),str) or not handle['call_id']):
            raise ValueError('follow_handle_reference_mismatch')
        if scene is None:raise ValueError('follow_scene_unavailable')
        policy=TaskPolicy(handle['task_id'],'follow',self.planning,self.geometry,occlusion_hold_s=visual_hold_s)
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
        self.policy.apply_validated_follow_reference(reference,key,generation,scene,self.history)
        self.reference=copy.deepcopy(reference)
        self.occluded_since=None
        self.trace.clear()

    def apply_t2(self, handle, prepared, result, window, scene, *, now, deadline_s=None):
        """Apply one bound T2 choice; caller owns scheduling and checked search motion."""
        from decision.vlm.t2.online import selected_reference
        from decision.vlm.t2.schema import parse_selection
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
                            source_stamp_s=prepared['source_stamp_s'])
            self.policy.cancel();self.policy=None;self.request=None
        return dict(operation='search',candidate=copy.deepcopy(choice['candidate']),
                    search_binding=prepared['search_binding'],source_stamp_s=prepared['source_stamp_s'],
                    task_completed=False,executable=False)

    def apply_reid(self, handle, evidence, scene, *, now):
        """Apply local evidence through the same 3D continuity and task policy checks."""
        from perception.environment.context import fingerprint
        from .follow_reid import bound, checked_departure, fresh_source, search_action, SEARCH_LEASE_S, VISUAL_HOLD_S

        reference_id=evidence['reference_id']
        if (scene is None or not bound(evidence) or evidence['task']!=handle or
                self.handle!=handle or not isinstance(reference_id,str) or self.reid_reference_id!=reference_id):
            raise ValueError('follow_reid_task_evidence_invalid')
        source=evidence['source'];decision=evidence['decision']
        fresh_source(source,scene,self.history,now)
        stamp_ns=source['rgb_stamp_ns']
        if decision['rgb_stamp_ns']!=stamp_ns:
            raise ValueError('follow_reid_decision_source_mismatch')
        if self.reid_stamp_ns is not None and stamp_ns<=self.reid_stamp_ns:
            return dict(operation='continue',reason='follow_reid_source_not_new')
        state=decision['status'];search=evidence['search'];selected=evidence['selected']
        loss=evidence['loss']
        continuation=evidence['continuation']
        if selected is not None and selected['kind']!='wait':
            last=self.reid_last_evidence
            if (selected['kind']!='approach' or last is None or
                    loss['last_evidence_binding']!=last['binding'] or
                    selected['last_evidence_binding']!=last['binding'] or
                    not isinstance(continuation,dict) or
                    set(continuation)!={'intent_id','kind','target','expires_s'} or
                    not isinstance(continuation['intent_id'],str) or not continuation['intent_id'] or
                    continuation['kind']!=selected['kind'] or continuation['target']!=selected['pose'] or
                    not now<continuation['expires_s']<=last['stamp_s']+45.):
                raise ValueError('follow_reid_search_intent_invalid')
            if len(selected['pose'])!=(3 if loss.get('departure_yaw_rad') is not None else 2):
                raise ValueError('follow_reid_search_pose_invalid')
            if len(selected['pose'])==3 and 'continuation_id' not in selected:
                checked_departure(loss,last,self.history)
        elif continuation is not None:
            raise ValueError('follow_reid_search_intent_unexpected')
        if state in ('same','recovered'):
            item=next((item for item in source['candidates'] if item['id']==decision['candidate_id']),None)
            track=None if item is None else item['track']
            if (loss['state']!='confirmed' or track is None or source['association_status']!='complete' or track['identity']!=decision['identity'] or
                    track['lifecycle_generation']!=decision['lifecycle_generation'] or
                    search is not None or selected is not None):
                raise ValueError('follow_reid_person_evidence_invalid')
            reference=dict(task_id=handle['task_id'],task_revision=handle['plan_revision'],goal_id=handle['goal_id'],
                track=track,reid_evidence_binding=evidence['binding'],
                **{key:source[key] for key in ('stamp_s','rgb_stamp_s','rgb_stamp_ns','clock_domain','local_identity','scene_binding')})
            reference['reference_id']=fingerprint(reference)
            self.history.validate(reference,scene,now=now)
            row=next(row for row in self.history.rows if row['scene_binding']==source['scene_binding'])
            position,velocity=row['observations'][tuple(track['identity'])]
            if self.policy is None:
                action=self.start(handle,reference,scene,now=now,visual_hold_s=VISUAL_HOLD_S)
            else:
                self.retarget(reference,scene,now=now,max_age_s=VISUAL_HOLD_S)
                action=self.update(scene,now=now)
            self.t2_active=True
            if action['operation']=='plan':
                last=dict(identity=track['identity'],lifecycle_generation=track['lifecycle_generation'],
                    stamp_s=track['support_stamp_s'],rgb_stamp_s=source['rgb_stamp_s'],
                    rgb_stamp_ns=stamp_ns,clock_domain=source['clock_domain'],local_identity=source['local_identity'],
                    scene_binding=source['scene_binding'],people_fingerprint=track['people_fingerprint'],
                    position_xy_m=position,velocity_xy_mps=velocity)
                last['binding']=fingerprint(last)
                self.reid_last_evidence=copy.deepcopy(last)
        elif state=='unconfirmed':
            if decision['reason']=='source_not_new':return dict(operation='continue',reason='follow_reid_source_not_new')
            if loss['state']=='holding' and (self.reid_last_evidence is None or
                    loss['last_evidence_binding']!=self.reid_last_evidence['binding'] or
                    source['association_status']!='complete'):
                raise ValueError('follow_reid_hold_evidence_invalid')
            if (selected is not None and search is None or search is not None and
                    (loss['state']!='lost' or not decision['search_eligible'] or source['association_status']!='complete')):
                raise ValueError('follow_reid_search_evidence_invalid')
            if search is not None:search_action(search,source,selected)
            if selected is not None and selected['kind']=='approach':
                valid_until=source['stamp_s']+SEARCH_LEASE_S
                if continuation is not None:valid_until=min(valid_until,continuation['expires_s'])
                action=dict(operation='search',candidate=copy.deepcopy(selected),
                            search_binding=search['binding'],valid_until_s=valid_until,
                            continuation=copy.deepcopy(continuation),reason=decision['reason'])
            else:
                action=dict(operation='hold' if loss['state']=='holding' else 'wait',
                            reason=loss['reason'])
        else:
            raise ValueError('follow_reid_decision_invalid')
        self.reid_stamp_ns=stamp_ns
        return action

    def hold_reid(self, request, scene, *, now):
        """Refresh the lease for the already issued goal using fresh 3D support."""
        from navigation.planning.inputs import scene_status
        from .follow_reid import VISUAL_HOLD_S

        last=self.reid_last_evidence
        if self.policy is None or request is None or last is None or scene is None:
            raise ValueError('follow_reid_no_issued_goal')
        if self.policy.target!=tuple(last['identity']) or self.policy.tracked_generation!=last['lifecycle_generation']:
            raise ValueError('follow_reid_issued_identity_changed')
        limit=last['rgb_stamp_s']+VISUAL_HOLD_S
        if not last['rgb_stamp_s']<=now<limit:
            raise ValueError('follow_visual_confirmation_expired')
        clock=last['clock_domain']
        reason=scene_status(scene,now,clock,self.planning,check_navigation=False)
        if reason:
            raise ValueError(reason)
        row=self.history.continue_from(last['scene_binding'],self.policy.target,
            self.policy.tracked_generation,scene,observed_only=True)
        valid_until=min(limit,row['stamp_s']+self.planning['max_geometry_age_s'],
            row['tracks'][self.policy.target][2]+self.planning['people']['max_track_age_s'])
        if now>=valid_until or now>request['valid_until_s']:
            raise ValueError('follow_reid_issued_goal_expired')
        return dict(request,valid_until_s=valid_until)

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
        self.reid_reference_id=self.reid_stamp_ns=None
        self.reid_last_evidence=None
        self.revision_floor=0
        self.trace.clear()
