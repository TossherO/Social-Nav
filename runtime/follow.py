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
        result=policy.start_continuous_follow(reference,scene,self.history,now=now)
        if result['operation']!='plan':raise ValueError(result['reason'])
        self.policy,self.handle,self.reference=policy,copy.deepcopy(handle),copy.deepcopy(reference)
        return self._translate(result,scene)

    def update(self, scene, *, now):
        if self.policy is None:return None
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
        if value['state']=='visible':
            if self.occluded_since is not None and rgb<=self.occluded_since:raise ValueError('follow_visual_result_out_of_order')
            self.occluded_since=None;self.policy.last_visual=rgb
        elif self.occluded_since is None:
            self.occluded_since=rgb

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
        return dict(operation='plan',request=copy.deepcopy(request),target_identity=result['target_identity'])

    def clear(self):
        if self.policy is not None:self.policy.cancel()
        self.policy=self.handle=self.reference=self.request=self.occluded_since=self.last_evidence=None
        self.trace.clear()
