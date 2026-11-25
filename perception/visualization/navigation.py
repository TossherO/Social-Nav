"""Navigation display with terrain heights, independent of control decisions."""
from collections import deque
from types import SimpleNamespace
import copy
import time

import numpy as np

from navigation.planning.continuity import check_reference,content_valid,context_binding,goal_binding,local_goal
from navigation.planning.inputs import association_status,navigation_grid,request_status,scene_status
from navigation.planning.planner import se2,transform_poses
from perception.mapping_localization.context import pose_matrix
from .surface import (DisplaySurface, NavigationGridSurface, complete_display_heights,
                      sample_ground, sample_poses)


IDENTITY=('session_id','generation','frame_id')


def display_packet(result,reference,request,scene,row,control,reason,*,now,clock_domain,clock_rate,clock_mono=None,goal_hint=None,search_map=None):
    """Copy only small metadata; finalized results remain immutable shared objects."""
    clock_mono=time.monotonic() if clock_mono is None else clock_mono
    record=None if scene is None else copy.deepcopy({k:scene.record[k] for k in (
        *IDENTITY,'as_of','clock_domain','localization','geometry','prior','input_fingerprint')})
    return dict(result=result,reference=reference,request=copy.deepcopy(request),goal_hint=copy.deepcopy(goal_hint),scene=record,search_map=search_map,
        row=copy.deepcopy(row),proposal=copy.deepcopy(control.get('proposal')),reason=control.get('reason'),
        reference_reason=reason,now_s=now,clock_domain=clock_domain,clock_rate=clock_rate,mono=clock_mono)


class NavigationView:
    def __init__(self,geometry,planning,*,asset=None,path_z_m=.03,vehicle_clearance_m=.12,
                 height_mode='ground',ground_plane_radius_m=1.5,reference_max_age_s=5.,trajectory_max_age_s=.5):
        self.geometry,self.planning,self.asset=geometry,planning,asset
        self.z=path_z_m;self.height_mode=height_mode;self.plane_radius=ground_plane_radius_m
        self.vehicle_clearance_m=vehicle_clearance_m
        self.static_surface=None
        self.reference_max_age_s=reference_max_age_s;self.trajectory_max_age_s=trajectory_max_age_s
        self.history=deque(maxlen=2000);self.raw_history=deque(maxlen=2000);self.identity=None

    def display_surface(self,row,grid,*,mode=None):
        if self.static_surface is None and self.asset is not None:
            self.static_surface=NavigationGridSurface(self.asset)
        return DisplaySurface(grid,row,offset=self.z,mode=mode or self.height_mode,
            plane_radius=self.plane_radius,max_age=self.planning['max_geometry_age_s'],terrain=self.static_surface)

    def update(self,row,packet,*,grid=None,mono=None,stopped=False):
        mono=time.monotonic() if mono is None else mono
        identity=tuple(row[k] for k in IDENTITY)
        # Independent control may have a newer measured pose than perception.
        same=packet is not None and tuple(packet['row'][k] for k in IDENTITY)==identity
        if same and packet['row']['stamp_s']>row['stamp_s']:row=packet['row']
        surface=self.display_surface(row,grid)
        valid=row['health']==0
        pose,z,ground_source=self.observe(row,surface)
        out=dict(frame_id=row['frame_id'],stamp_s=row['stamp_s'],localization_row=row,history=list(self.history),
            raw_history=list(self.raw_history),pose=pose,reference=None,trajectory=None,goal=None,search_map=None,
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
                if common is None:
                    hint=packet.get('goal_hint')
                    goal=hint if hint is not None and not request_status(hint,scene,now,packet['clock_domain']) else request
                    out['goal']=local_goal(goal,scene,self.geometry,self.asset,self.planning)
            except ValueError as error:common=str(error)
            except (KeyError,TypeError):common='display_context_invalid'
        reference=packet['reference']
        if common is None and reference is not None:
            try:
                check_reference(reference,request,scene,self.asset,self.reference_max_age_s)
                if not 0<=now-reference['as_of']<=self.reference_max_age_s:raise ValueError('reference_future_or_stale')
                result_frame=reference['result']['frame_id']
                data=dict(points=np.asarray(reference['result']['path'],float),frame_id=result_frame,
                          stamp_s=reference['as_of'],display_stamp_s=reference['as_of'])
                if request['mode']=='mapped':
                    # Use the planner's planar transform only to sample local terrain.
                    world=np.asarray(record['localization']['world_from_lidar'])
                    map_from_local=np.asarray(record['prior']['map_from_local'])
                    mapped=self.geometry.pose(map_from_local@world)
                    data.update(local_from_map=se2(self.geometry.pose(world))@np.linalg.inv(se2(mapped)),
                                map_from_local=map_from_local,display_stamp_s=row['stamp_s'])
                out['reference']=data
                ref_reason='ready (age %.2fs)'%(now-reference['as_of'])
            except (ValueError,KeyError,TypeError) as error:ref_reason=str(error)
        snapshot=packet.get('search_map')
        if (snapshot is not None and common is None and self.asset is not None and
                snapshot['goal_binding']==goal_binding(request) and
                0<=now-snapshot['stamp_s']<=self.reference_max_age_s and
                association_status(scene,self.asset,self.planning) is None):
            if reference is None and snapshot['status']!='path_found':
                out['search_map']=snapshot
                ref_reason=snapshot['reason'] or ref_reason
            elif (out['reference'] is not None and
                    snapshot['reference_binding']==reference['binding']):
                out['search_map']=snapshot
                path=np.asarray(reference['result']['path'],float)
                if reference['result']['frame_id']!=snapshot['frame_id']:
                    path=transform_poses(path,np.linalg.inv(snapshot['local_from_map']))
                out['reference']['search_xy']=sample_poses(path,snapshot['resolution'])[:,:2]
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

    def observe(self,row,surface=None):
        identity=tuple(row[k] for k in IDENTITY)
        valid=row['health']==0
        stamp=row['stamp_s']
        if identity!=self.identity or not valid or (self.history and stamp<self.history[-1][0]):
            self.history.clear();self.raw_history.clear();self.identity=identity
        pose=None;z=np.nan;ground_source=0
        if valid:
            pose=self.geometry.pose(pose_matrix(row['position'],row['quaternion_xyzw']))
            if surface is not None:
                heights,sources=surface.sample([pose[:2]])
                z,ground_source=float(heights[0]),int(sources[0])
            elif self.height_mode=='plane':
                z,ground_source=self.z,3
            if not self.history or stamp>self.history[-1][0]:
                self.history.append((stamp,pose.copy(),z))
                self.raw_history.append((stamp,list(row['position']),list(row['quaternion_xyzw'])))
            elif np.isfinite(z) and not np.isfinite(self.history[-1][2]):
                self.history[-1]=(stamp,pose.copy(),z)
        return pose,z,ground_source

    def _project(self,out,surface):
        snapshot=out.get('search_map')
        if snapshot is not None:
            out['text']+='\nReference search #%s'%snapshot['job_id']
        pose=out['pose'];out['footprint']=out['heading']=np.empty((0,3));out['goal_z']=np.nan
        if pose is not None:
            polygon=self.geometry.polygon(pose);polygon=np.vstack([polygon,polygon[0]])
            polygon=sample_poses(np.c_[polygon,np.zeros(len(polygon))],surface.resolution)[:,:2]
            heading=np.vstack([pose[:2],pose[:2]+.8*np.array([np.cos(pose[2]),np.sin(pose[2])])])
            body=self._vehicle_points(np.vstack((polygon,heading)),surface)
            out['footprint'],out['heading']=body[:-len(heading)],body[-len(heading):]
        history=out.get('history')
        if history and surface.mode=='ground':
            draw=surface.above([[p[0],p[1],z] for _,p,z in history])
            out['history']=[(stamp,p,height) for (stamp,p,_),height in zip(history,draw[:,2])]
        if out['goal'] is not None:out['goal_z']=float(surface.sample([out['goal'][:2]])[0][0])
        status={0:'unknown (markers omitted)',1:'supported cells',2:'fitted near vehicle',3:'fixed XY diagnostic',4:'static supported cells',5:'estimated static ground'}[out['ground_source']]
        for key in ('reference','trajectory'):
            data=out[key]
            if data is not None:
                data['display_poses']=sample_poses(data['points'],surface.resolution)
                xy=data['display_poses'][:,:2]
                transform=data.get('map_from_local')
                mapped=transform is not None and surface.mode=='ground'
                local_xy=transform_poses(data['display_poses'],data['local_from_map'])[:,:2] if mapped else xy
                data['heights'],data['height_sources']=surface.sample(local_xy)
                if key=='reference':
                    if mapped:
                        # Convert the static fallback into local Z at the sampled XY.
                        z_offset=local_xy@transform[2,:2]+transform[2,3]
                        heights,sources=sample_ground(navigation_grid(self.asset),xy,self.z)
                        missing=~np.isfinite(data['heights']) & np.isfinite(heights)
                        data['heights'][missing]=(heights[missing]-z_offset[missing])/transform[2,2]
                        data['height_sources'][missing]=np.where(sources[missing]==5,5,4)
                    fallback=out['localization_row']['position'][2]+self.z
                    data['heights'],approx=complete_display_heights(data['heights'],fallback)
                    data['height_sources'][approx]=5
                    if approx.any():status+=' | reference approximate display heights %d'%int(approx.sum())
                if key=='trajectory':data['heights']+=.02
                data['heights']=surface.above(np.c_[local_xy,data['heights']],
                    clearance=.06 if key=='trajectory' else .04)[:,2]
                if mapped:
                    data['heights']=np.fmax(data['heights']*transform[2,2]+z_offset,heights)
                status+=' | %s height %d/%d'%(key,int(np.isfinite(data['heights']).sum()),len(data['heights']))
        out['text']+='\nGROUND: '+status
        return out

    def _vehicle_points(self, xy, surface):
        z, _ = surface.sample(xy)
        if surface.mode == 'plane':
            return np.c_[xy, z+self.vehicle_clearance_m]
        return surface.above(np.c_[xy,z-surface.offset+self.vehicle_clearance_m],
            clearance=self.vehicle_clearance_m)
