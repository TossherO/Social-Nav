"""Navigation display with terrain heights, independent of control decisions."""
from collections import deque
from types import SimpleNamespace
import copy
import time

import numpy as np

from navigation.planning.continuity import check_reference,content_valid,context_binding,goal_binding,local_goal
from navigation.planning.inputs import request_status,scene_status
from navigation.planning.planner import se2,transform_poses
from perception.mapping_localization.context import pose_matrix
from .surface import DisplaySurface,sample_poses


IDENTITY=('session_id','generation','frame_id')


def display_packet(result,reference,request,scene,row,control,reason,*,now,clock_domain,clock_rate,clock_mono=None):
    """Copy only small metadata; finalized results remain immutable shared objects."""
    clock_mono=time.monotonic() if clock_mono is None else clock_mono
    record=None if scene is None else copy.deepcopy({k:scene.record[k] for k in (
        *IDENTITY,'as_of','clock_domain','localization','geometry','prior','input_fingerprint')})
    return dict(result=result,reference=reference,request=copy.deepcopy(request),scene=record,
        row=copy.deepcopy(row),proposal=copy.deepcopy(control.get('proposal')),reason=control.get('reason'),
        reference_reason=reason,now_s=now,clock_domain=clock_domain,clock_rate=clock_rate,mono=clock_mono)


class NavigationView:
    def __init__(self,geometry,planning,*,asset=None,path_z_m=.03,height_mode='ground',ground_plane_radius_m=1.5,reference_max_age_s=5.,trajectory_max_age_s=.5):
        self.geometry,self.planning,self.asset=geometry,planning,asset
        self.z=path_z_m;self.height_mode=height_mode;self.plane_radius=ground_plane_radius_m
        self.reference_max_age_s=reference_max_age_s;self.trajectory_max_age_s=trajectory_max_age_s
        self.history=deque(maxlen=2000);self.raw_history=deque(maxlen=2000);self.identity=None

    def update(self,row,packet,*,grid=None,mono=None,stopped=False):
        mono=time.monotonic() if mono is None else mono
        identity=tuple(row[k] for k in IDENTITY)
        # Independent control may have a newer measured pose than perception.
        same=packet is not None and tuple(packet['row'][k] for k in IDENTITY)==identity
        if same and packet['row']['stamp_s']>row['stamp_s']:row=packet['row']
        valid=row['health']==0
        if identity!=self.identity or not valid or (self.history and row['stamp_s']<self.history[-1][0]):
            self.history.clear();self.raw_history.clear();self.identity=identity
        surface=DisplaySurface(grid,row,offset=self.z,mode=self.height_mode,plane_radius=self.plane_radius,
            max_age=self.planning['max_geometry_age_s'])
        pose=None;z=np.nan;ground_source=0
        if valid:
            pose=self.geometry.pose(pose_matrix(row['position'],row['quaternion_xyzw']))
            heights,sources=surface.sample([pose[:2]]);z=float(heights[0]);ground_source=int(sources[0])
            if not self.history or row['stamp_s']>self.history[-1][0]:
                self.history.append((row['stamp_s'],pose.copy(),z))
                self.raw_history.append((row['stamp_s'],list(row['position']),list(row['quaternion_xyzw'])))
        out=dict(frame_id=row['frame_id'],stamp_s=row['stamp_s'],localization_row=row,history=list(self.history),
            raw_history=list(self.raw_history),pose=pose,reference=None,trajectory=None,goal=None,
            text='SHADOW | no control observation',level='waiting',z=z,ground_source=ground_source,
            text_z=z+1.6 if np.isfinite(z) else float(row['position'][2])+.8)
        if stopped:
            out.update(text='SHADOW stopped | history only',level='waiting');return self._project(out,surface)
        if not valid:
            out.update(text='SHADOW | localization invalid',level='error');return self._project(out,surface)
        if not same:
            out['text']='SHADOW | waiting for matching control observation';return self._project(out,surface)
        now=packet['now_s']+max(0.,mono-packet['mono'])*packet['clock_rate']
        request=packet['request'];record=packet['scene'];proposal=packet['proposal'] or {}
        ref_reason=packet['reference_reason'] or 'waiting_for_background_reference'
        dwa_reason=packet['reason'] or proposal.get('reason') or 'waiting_for_control'
        common=None;scene=None
        if request is None:common='no_active_goal'
        elif record is None:common='observation_unavailable'
        elif tuple(record[k] for k in IDENTITY)!=identity:common='observation_lifecycle_mismatch'
        else:
            scene=SimpleNamespace(record=record)
            try:
                # Display packets carry source metadata, not a collision map.
                common=scene_status(scene,now,packet['clock_domain'],self.planning,check_navigation=False) or request_status(request,scene,now,packet['clock_domain'])
                if common is None:out['goal']=local_goal(request,scene,self.geometry,self.asset,self.planning)
            except ValueError as error:common=str(error)
            except (KeyError,TypeError):common='display_context_invalid'
        reference=packet['reference']
        if common is None and reference is not None:
            try:
                check_reference(reference,request,scene,self.asset,self.reference_max_age_s)
                if not 0<=now-reference['as_of']<=self.reference_max_age_s:raise ValueError('reference_future_or_stale')
                points=np.asarray(reference['result']['path'],float)
                if request['mode']=='mapped':
                    # Exactly the same planar map-to-local convention as DWA.
                    world=np.asarray(record['localization']['world_from_lidar'])
                    mapped=self.geometry.pose(np.asarray(record['prior']['map_from_local'])@world)
                    transform=se2(self.geometry.pose(world))@np.linalg.inv(se2(mapped))
                    points=transform_poses(points,transform)
                out['reference']=dict(points=points,stamp_s=reference['as_of'])
                ref_reason='ready (age %.2fs)'%(now-reference['as_of'])
            except (ValueError,KeyError,TypeError) as error:ref_reason=str(error)
        result=packet['result']
        if common is None and result is not None and proposal.get('status')=='candidate':
            try:
                source=result['input_binding'];age=now-source['as_of']
                if (result.get('status')!='control_ready' or not content_valid(result) or
                        result['context_binding']!=context_binding(request,scene,self.asset) or
                        goal_binding(result['request'])!=goal_binding(request) or
                        proposal.get('trajectory_binding')!=result['result_binding']):raise ValueError('trajectory_context_mismatch')
                if (source['clock_domain']!=packet['clock_domain'] or source['frame_id']!=out['frame_id'] or
                        result['trajectory']['frame_id']!=out['frame_id']):raise ValueError('trajectory_frame_or_clock_mismatch')
                if not 0<=age<=self.trajectory_max_age_s:raise ValueError('trajectory_future_or_stale')
                if out['reference'] is None or result['reference_binding']!=reference['binding']:
                    raise ValueError('trajectory_reference_unavailable')
                out['trajectory']=dict(points=np.asarray(result['trajectory']['points'],float)[:,1:],stamp_s=source['as_of'])
                velocity=proposal['velocity'];dwa_reason='candidate v=%.2f w=%.2f (age %.2fs)'%(*velocity,age)
            except (ValueError,KeyError,TypeError) as error:dwa_reason=str(error)
        if common:ref_reason=dwa_reason=common
        goal_text='no goal' if request is None else '%s | revision %s'%(request['mode'],request['goal_revision'])
        out['text']='SHADOW | %s\nREF: %s\nDWA: %s'%(goal_text,ref_reason,dwa_reason)
        out['level']='ready' if out['trajectory'] is not None else 'waiting'
        return self._project(out,surface)

    def _project(self,out,surface):
        pose=out['pose'];out['footprint']=out['heading']=np.empty((0,3));out['goal_z']=np.nan
        if pose is not None:
            polygon=self.geometry.polygon(pose);polygon=np.vstack([polygon,polygon[0]])
            polygon=sample_poses(np.c_[polygon,np.zeros(len(polygon))],surface.resolution)[:,:2]
            heights,_=surface.sample(polygon);out['footprint']=np.c_[polygon,heights+.04]
            heading=np.vstack([pose[:2],pose[:2]+.8*np.array([np.cos(pose[2]),np.sin(pose[2])])])
            heights,_=surface.sample(heading);out['heading']=np.c_[heading,heights+.05]
        if out['goal'] is not None:out['goal_z']=float(surface.sample([out['goal'][:2]])[0][0])
        status={0:'unknown (markers omitted)',1:'supported cells',2:'fitted near vehicle',3:'fixed XY diagnostic',4:'static supported cells'}[out['ground_source']]
        for key in ('reference','trajectory'):
            data=out[key]
            if data is not None:
                data['display_poses']=sample_poses(data['points'],surface.resolution)
                data['heights'],data['height_sources']=surface.sample(data['display_poses'][:,:2])
                if key=='trajectory':data['heights']+=.02
                status+=' | %s height %d/%d'%(key,int(np.isfinite(data['heights']).sum()),len(data['heights']))
        out['text']+='\nGROUND: '+status
        return out
