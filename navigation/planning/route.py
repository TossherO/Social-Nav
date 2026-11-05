"""Explicit route references across mapped, local and externally confirmed floor legs.

This adapter only prepares geometric planning requests. It never completes a task,
identifies a floor, infers an unknown corridor or produces a motion command.
"""
import copy
from dataclasses import dataclass

import numpy as np

from perception.environment.context import fingerprint
from .inputs import scene_status
from .planner import se2


def _identity(scene):
    r=scene.record
    return r['session_id'],r['generation'],r['frame_id']


def _association(scene, map_id, frame, now, clock, config):
    reason=scene_status(scene,now,clock,config)
    if reason:raise ValueError(reason)
    r=scene.record;a=r['prior'];stamp=a.get('measurement_stamp_s')
    if not a.get('valid') or a.get('map_id')!=map_id or a.get('map_frame')!=frame:
        raise ValueError('Route requires a fresh association to the specified map')
    if (a.get('local_session_id'),a.get('local_generation'),a.get('local_frame'))!=_identity(scene):
        raise ValueError('Route association lifecycle mismatch')
    if (stamp is None or a.get('stamp_s')!=r['as_of'] or
            not np.isfinite(stamp) or not 0<=r['as_of']-stamp<=config['max_association_age_s']):
        raise ValueError('Route association is future or stale')
    from perception.detection_tracking.inference.geometry import validate_transform
    return validate_transform(a['map_from_local'])


def _transform(points, matrix):
    return np.asarray(points)@matrix[:2,:2].T+matrix[:2,2]


def _lookahead(points, position, progress, lookahead, advance, max_cross_track):
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    arc=np.r_[0.,np.cumsum(lengths)]
    if not np.isfinite(progress) or not 0<=progress<=arc[-1]:
        raise ValueError('Route progress is outside this segment')
    # Restrict matching to the next arc interval. A loop or crossing cannot jump
    # to a distant later part solely because it is geometrically nearby.
    end=min(arc[-1],progress+advance)
    candidates=[]
    for i,length in enumerate(lengths):
        low,high=max(progress,arc[i]),min(end,arc[i+1])
        if low>high:continue
        fraction=np.clip((position-points[i])@(points[i+1]-points[i])/length**2,
                         (low-arc[i])/length,(high-arc[i])/length)
        projection=points[i]+fraction*(points[i+1]-points[i])
        candidates.append((np.linalg.norm(position-projection),arc[i]+fraction*length))
    distance,matched=min(candidates)
    if distance>max_cross_track:raise ValueError('Position is outside the explicit route corridor')
    target=min(arc[-1],matched+lookahead)
    goal=np.array([np.interp(target,arc,points[:,k]) for k in range(2)])
    return goal,float(matched),float(arc[-1]),float(distance)


@dataclass(frozen=True)
class RouteAnchor:
    route_binding: str
    anchor_id: str
    local_identity: tuple
    clock_domain: str
    stamp_s: float
    observation_binding: str
    local_from_anchor: tuple


class RoutePlan:
    def __init__(self, record):
        self.record=copy.deepcopy(record)
        for key in ('route_id','source'):
            if not record.get(key):raise ValueError('Route needs '+key)
        self.anchors=self.record.get('anchors',{})
        for anchor in self.anchors.values():
            matrix=np.asarray(anchor['map_from_anchor'],dtype=float)
            if (matrix.shape!=(3,3) or not np.isfinite(matrix).all() or
                    not np.allclose(matrix[2],[0,0,1]) or
                    not np.allclose(matrix[:2,:2].T@matrix[:2,:2],np.eye(2),atol=1e-6) or
                    not np.isclose(np.linalg.det(matrix[:2,:2]),1) or
                    not anchor.get('map_id') or not anchor.get('map_frame') or not anchor.get('source')):
                raise ValueError('Invalid explicit route anchor')
        self.segments={}
        for segment in record['segments']:
            ident=segment['segment_id'];kind=segment['mode']
            if not ident or ident in self.segments:raise ValueError('Duplicate or empty route segment')
            if not segment.get('source'):raise ValueError('Each route leg needs its own source')
            if kind=='transition':
                if not all(segment.get(k) for k in ('from_floor','to_floor','confirmation_source')):
                    raise ValueError('Floor transition needs explicit endpoints and confirmation source')
            elif kind in ('mapped','local'):
                if segment.get('goal_behavior','pass_through') not in ('pass_through','stop'):
                    raise ValueError('Unknown route goal behavior')
                points=np.asarray(segment['points_xy'],dtype=float)
                if (points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all() or
                        (np.linalg.norm(np.diff(points,axis=0),axis=1)<=1e-6).any() or not segment.get('floor_id')):
                    raise ValueError('Route leg needs a finite nondegenerate ordered polyline and floor')
                if kind=='local' and segment['anchor_id'] not in self.anchors:
                    raise ValueError('Unknown route anchor')
                if kind=='mapped' and not all(segment.get(k) for k in ('map_id','frame_id')):
                    raise ValueError('Mapped route needs map and frame identities')
            else:raise ValueError('Unknown route mode')
            self.segments[ident]=copy.deepcopy(segment)
        if not self.segments:raise ValueError('Empty route')
        self.binding=fingerprint(self.record)

    def bind_anchor(self, anchor_id, scene, *, now, clock_domain, config):
        scene.verify();anchor=self.anchors[anchor_id]
        transform=_association(scene,anchor['map_id'],anchor['map_frame'],now,clock_domain,config)
        map_from_local=se2([*transform[:2,3],np.arctan2(transform[1,0],transform[0,0])])
        local_from_anchor=np.linalg.inv(map_from_local)@np.asarray(anchor['map_from_anchor'])
        return RouteAnchor(self.binding,anchor_id,_identity(scene),clock_domain,scene.record['as_of'],
            scene.record['input_fingerprint'],tuple(map(tuple,local_from_anchor)))

    def request(self, segment_id, scene, geometry, *, task_id, goal_revision, now, clock_domain,
                config, progress_m=0., lookahead_m=2., max_advance_m=8., max_cross_track_m=2.,
                asset=None, anchor=None):
        """Explicit leg selection; returned progress is geometric, never execution feedback.

        Persist progress per task/leg/local lifecycle. On reset rebind/reselect it;
        invoking this method or reaching a polyline end cannot advance a floor.
        """
        segment=self.segments[segment_id]
        if segment['mode']=='transition':
            return dict(status='waiting_for_external_confirmation',executable=False,
                segment_id=segment_id,route_binding=self.binding,
                confirmation_source=segment['confirmation_source'],request=None)
        scene.verify()
        reason=scene_status(scene,now,clock_domain,config)
        if reason:raise ValueError(reason)
        if not task_id or type(goal_revision) is not int or goal_revision<0:
            raise ValueError('Route request needs task identity and nonnegative goal revision')
        if not np.isfinite([lookahead_m,max_advance_m,max_cross_track_m]).all() or min(lookahead_m,max_advance_m,max_cross_track_m)<=0:
            raise ValueError('Route lookahead and matching bounds must be positive')
        points=np.asarray(segment['points_xy']);mode=segment['mode']
        pose=geometry.pose(scene.record['localization']['world_from_lidar'])
        if mode=='mapped':
            transform=_association(scene,segment['map_id'],segment['frame_id'],now,clock_domain,config)
            if (asset is None or asset.snapshot.metadata.get('map_id')!=segment['map_id'] or
                    asset.snapshot.metadata['frame_id']!=segment['frame_id']):
                raise ValueError('Route static environment map mismatch')
            pose=geometry.pose(transform@np.asarray(scene.record['localization']['world_from_lidar']))
            frame=segment['frame_id']
        else:
            if (not isinstance(anchor,RouteAnchor) or anchor.route_binding!=self.binding or
                    anchor.anchor_id!=segment['anchor_id'] or anchor.local_identity!=_identity(scene) or
                    anchor.clock_domain!=clock_domain or scene.record['as_of']<anchor.stamp_s):
                raise ValueError('Local route anchor missing, from the future or from another lifecycle')
            points=_transform(points,np.asarray(anchor.local_from_anchor));frame=scene.record['frame_id']
        goal,progress,total,cross_track=_lookahead(points,pose[:2],progress_m,lookahead_m,max_advance_m,max_cross_track_m)
        terminal=np.linalg.norm(goal-points[-1])<1e-7
        request=dict(request_id='%s:%s:%d'%(self.record['route_id'],segment_id,goal_revision),
            task_id=task_id,goal_revision=goal_revision,mode=mode,frame_id=frame,
            target_reference='base_link',goal=goal.tolist(),
            goal_behavior=segment.get('goal_behavior','pass_through') if terminal else 'pass_through',route_binding=self.binding,
            route_segment_id=segment_id,floor_id=segment['floor_id'],route_source=segment['source'])
        if mode=='mapped':request['map_snapshot_id']=asset.manifest['snapshot_id']
        else:request['route_anchor_observation']=anchor.observation_binding
        return dict(status='reference_ready',executable=False,request=request,progress_m=progress,
                    segment_length_m=total,cross_track_m=cross_track,at_reference_end=bool(total-progress<.1),
                    task_completed=False)


class RouteSession:
    """Explicit task/leg owner. Geometric progress never completes or advances a leg."""
    def __init__(self,route,task_id):
        if not task_id:raise ValueError('Route task needs an identity')
        self.route=route;self.task_id=task_id;self.segment_id=None;self.revision=-1
        self.anchors={};self.progress=0.;self.confirmation=None

    def select(self,segment_id,goal_revision):
        if segment_id not in self.route.segments or type(goal_revision) is not int or goal_revision<=self.revision:
            raise ValueError('Explicit route selection needs a new goal revision')
        if self.segment_id is not None and self.route.segments[self.segment_id]['mode']=='transition' and self.confirmation is None:
            raise ValueError('Selected floor transition still requires external confirmation')
        self.segment_id=segment_id;self.revision=goal_revision;self.progress=0.;self.confirmation=None;self.last_goal=None
        self.local_identity=None

    def bind_anchor(self,anchor_id,scene,*,now,clock_domain,config):
        self.anchors[anchor_id]=self.route.bind_anchor(anchor_id,scene,now=now,clock_domain=clock_domain,config=config)

    def confirm_transition(self,feedback):
        if self.segment_id is None:raise ValueError('No selected route leg')
        leg=self.route.segments[self.segment_id]
        if leg['mode']!='transition':raise ValueError('Selected route leg is not a floor transition')
        expected=dict(task_id=self.task_id,segment_id=self.segment_id,goal_revision=self.revision,
            from_floor=leg['from_floor'],to_floor=leg['to_floor'],source=leg['confirmation_source'],
            event='floor_transition_confirmed')
        stamp=feedback.get('stamp_s')
        if (any(feedback.get(k)!=v for k,v in expected.items()) or not feedback.get('source_event_id') or
                not feedback.get('clock_domain') or stamp is None or not np.isfinite(stamp)):
            raise ValueError('External floor confirmation does not match the selected task/leg/source')
        self.confirmation=copy.deepcopy(feedback)
        return dict(status='transition_confirmed',source=copy.deepcopy(feedback),task_completed=False,executable=False)

    def request(self,scene,geometry,*,now,clock_domain,config,asset=None,lookahead_m=2.):
        if self.segment_id is None:raise ValueError('Select a route leg explicitly')
        leg=self.route.segments[self.segment_id]
        if scene is not None:
            current=(_identity(scene),scene.record['clock_domain'])
            if self.local_identity is not None and self.local_identity!=current:
                raise ValueError('Localization reset requires explicit route reselection and anchor rebinding')
            self.local_identity=current
        anchor=self.anchors.get(leg.get('anchor_id'))
        result=self.route.request(self.segment_id,scene,geometry,task_id=self.task_id,goal_revision=self.revision,
            now=now,clock_domain=clock_domain,config=config,progress_m=self.progress,lookahead_m=lookahead_m,
            asset=asset,anchor=anchor)
        if result['request'] is not None:
            goal=result['request']['goal']
            previous=getattr(self,'last_goal',None)
            if previous is not None and fingerprint(previous)!=fingerprint(goal):self.revision+=1
            result['request']['goal_revision']=self.revision;self.last_goal=copy.deepcopy(goal)
            self.progress=result['progress_m']
        return result
