"""Observation-bound free memory inside the nearest ground scan ring.

This local view preserves a previously certified column, never
creates free space from a presumed floor or clears current obstacles. Ray/ground
layers retain their historical values and times; additional layers identify reuse.
"""
import numpy as np

from .context import fingerprint
from .geometry import shift_array


HISTORY_DTYPES=dict(near_field_free=bool,near_field_free_classified_at=np.float64,
    near_field_obstacle=bool,near_field_obstacle_stamp=np.float64,near_field_obstacle_expires_at=np.float64)
DYNAMIC_DTYPES=dict(person_background_free=bool)
GROUND_DTYPES=dict(ground_conflict=bool)


def ground_blind_zone(snapshot, plane, settings):
    """Whole cells inside a bounded lowest-beam/plane intersection, not a FOV proof."""
    m=snapshot.metadata;r=m['resolution'];shape=tuple(m['shape'])
    inside=np.zeros(shape,bool);transform=np.asarray(m['world_from_lidar']);sensor=transform[:3,3]
    if plane is None:return inside,dict(status='ground_ring_unavailable')
    rotation=transform[:3,:3]
    if np.arccos(np.clip(rotation[2,2],-1,1))>settings['max_sensor_tilt_rad']:
        return inside,dict(status='sensor_tilt_outside_ring_model')
    angle=np.arange(360)*np.pi/180;beta=np.deg2rad(settings['lowest_beam_deg'])
    directions=np.c_[np.cos(angle)*np.cos(beta),np.sin(angle)*np.cos(beta),np.full(360,np.sin(beta))]@rotation.T
    denominator=directions[:,2]-directions[:,:2]@plane[:2]
    clearance=sensor[2]-np.r_[sensor[:2],1.]@plane
    if not np.isfinite(clearance) or clearance<=0:return inside,dict(status='invalid_ring_clearance')
    distance=np.zeros(360)
    valid=denominator<-1e-6
    distance[valid]=clearance/-denominator[valid]*np.linalg.norm(directions[valid,:2],axis=1)
    radius=np.maximum(0,np.minimum(settings['max_radius_m'],distance)-settings['boundary_margin_m'])
    azimuth=np.arctan2(directions[:,1],directions[:,0]);order=np.argsort(azimuth)
    azimuth,radius=azimuth[order],radius[order]
    azimuth=np.r_[azimuth[-1]-2*np.pi,azimuth,azimuth[0]+2*np.pi]
    radius=np.r_[radius[-1],radius,radius[0]]
    origin=np.asarray(m['origin_xy']);limit=settings['max_radius_m']
    lo=np.maximum(0,np.floor((sensor[:2]-limit-origin)/r).astype(int))
    hi=np.minimum(np.asarray(shape)[::-1],np.ceil((sensor[:2]+limit-origin)/r).astype(int))
    if np.any(hi<=lo):return inside,dict(status='ring_outside_grid')
    y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    local=np.stack((x+.5,y+.5),axis=-1)*r+origin-sensor[:2]
    keep=np.ones(x.shape,bool)
    for corner in [(-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)]:
        delta=local+np.asarray(corner)*r
        reach=np.interp(np.arctan2(delta[...,1],delta[...,0]),azimuth,radius)
        keep&=np.linalg.norm(delta,axis=-1)<=reach
    inside[lo[1]:hi[1],lo[0]:hi[0]]=keep
    return inside,dict(status='available',clearance_m=float(clearance),
        radius_min_m=float(radius.min()),radius_max_m=float(radius.max()),
        method='lowest_beam_intersection_with_tracked_plane_whole_cell_inside')


class NearFieldFreeMemory:
    """Continuous local lifecycle; optional bounded candidates beyond the blind ring."""
    DEFAULTS=dict(lowest_beam_deg=-15.,max_radius_m=3.5,boundary_margin_m=.15,
                  max_sensor_tilt_rad=.35,ground_change_m=.15,remember_outside_ring=False,
                  temporary_obstacle_overlay=False)

    def __init__(self, config, settings=None):
        if config.get('hit_invalidates_free') is not True:
            raise ValueError('Near-field memory requires explicit hit_invalidates_free')
        self.config=config;self.settings=dict(self.DEFAULTS,**(settings or {}))
        if (not -45<=self.settings['lowest_beam_deg']<0 or
                any(not np.isfinite(v) or v<=0 for k,v in self.settings.items()
                    if k not in ('lowest_beam_deg','remember_outside_ring','temporary_obstacle_overlay')) or
                type(self.settings['remember_outside_ring']) is not bool or
                type(self.settings['temporary_obstacle_overlay']) is not bool or
                self.settings['boundary_margin_m']>=self.settings['max_radius_m']):
            raise ValueError('Invalid near-field memory geometry')
        self.binding=fingerprint(self.settings)
        self.reset()

    def reset(self):
        self.key=None;self.origin=None;self.stamp=None;self.plane=None;self.fields={}
        self.confirmed=None
        self.hit_stamp=self.hit_min=self.hit_max=None
        self.ground_cache=None
        self.hit_expires=None
        self.ordinary_stamp=None
        self.history_suspended=False

    def prepare_plane(self, ground, transform):
        """Small policy state only; both backends use the same causal plane choice."""
        self.history_suspended=ground.get('history_suspended',False)
        plane=ground.get('plane_world')
        if plane is not None:self.plane=np.asarray(plane,float)
        elif self.plane is not None and ground.get('reference_height_m') is not None:
            sensor=np.asarray(transform)[:3,3];self.plane=self.plane.copy()
            self.plane[2]=sensor[2]-ground['reference_height_m']-sensor[:2]@self.plane[:2]
        return self.plane

    def describe(self, source_id, counts, ring, backend):
        cached,invalidated,conflicts,hits,restored,active=map(int,counts)
        return dict(source_snapshot_id=source_id,settings=self.settings,ring=ring,
            history_suspended=self.history_suspended,
            restored_cells=restored,cached_cells=cached,invalidated_hit_cells=invalidated,
            hit_conflict_cells=hits,ground_conflict_cells=conflicts,temporary_obstacle_cells=active,
            obstacle_overlay_lifetime_s=self.config['obstacle_ttl_s'],
            hit_policy='temporary_overlay_on_historical_prior' if self.settings['temporary_obstacle_overlay'] else 'invalidate_candidate',
            current_obstacles_preserved=True,time_semantics='reused_original_observation_times',
            compute_backend=backend,scope='local_history_view_not_fresh_ray_measurement')

    def update(self, snapshot, endpoints, *, transient_mask=None,person_only=None):
        from .layers import GridSnapshot, finish_evidence
        m=snapshot.metadata;raw=snapshot.layers;stamp=float(m['stamp_s'])
        key=tuple(m[k] for k in ('session_id','generation','frame_id','resolution'))+tuple(m['shape'])
        if key!=self.key or (self.stamp is not None and stamp<=self.stamp):self.reset()
        shape=tuple(m['shape']);origin=np.asarray(m['origin_xy']);r=m['resolution']
        if self.key is None:
            self.key=key;self.confirmed=np.full(shape,np.nan)
            self.hit_stamp=np.full(shape,np.nan);self.hit_min=np.full(shape,np.nan);self.hit_max=np.full(shape,np.nan)
            self.hit_expires=np.full(shape,np.nan)
            self.ordinary_stamp=np.full(shape,-np.inf)
            self.ground_cache=np.full(shape,np.nan,dtype=np.float32)
            self.fields={k:np.full(shape,np.nan,dtype=raw[k].dtype) for k in
                         ('ground_z','coverage','last_observed')}
        else:
            delta=(self.origin-origin)/r
            if not np.allclose(delta,np.rint(delta),atol=1e-6,rtol=0):
                raise ValueError('Near-field memory grid moved off its source lattice')
            dx,dy=np.rint(delta).astype(int)
            if dx or dy:
                self.confirmed=shift_array(self.confirmed,dx,dy,np.nan)
                self.fields={k:shift_array(v,dx,dy,np.nan) for k,v in self.fields.items()}
                self.hit_stamp=shift_array(self.hit_stamp,dx,dy,np.nan)
                self.hit_min=shift_array(self.hit_min,dx,dy,np.nan);self.hit_max=shift_array(self.hit_max,dx,dy,np.nan)
                self.ground_cache=shift_array(self.ground_cache,dx,dy,np.nan)
                self.hit_expires=shift_array(self.hit_expires,dx,dy,np.nan)
                self.ordinary_stamp=shift_array(self.ordinary_stamp,dx,dy,-np.inf)
        self.origin=origin.copy();self.stamp=stamp
        self.prepare_plane(m.get('ground_tracking',{}),m['world_from_lidar'])
        paused=np.asarray(raw.get('ground_conflict',np.zeros(shape,bool)))|self.history_suspended
        measured_ground=raw['ground_support']&np.isfinite(raw['ground_z'])&~paused
        self.ground_cache[measured_ground]=raw['ground_z'][measured_ground]
        blind,ring=ground_blind_zone(snapshot,self.plane,self.settings)
        valid=np.isfinite(self.confirmed)
        comparable=valid&measured_ground
        conflict=np.zeros(shape,bool)
        conflict[comparable]=np.abs(raw['ground_z'][comparable]-self.fields['ground_z'][comparable])>self.settings['ground_change_m']
        hit=raw['occupancy']==100
        points=np.asarray(endpoints,float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Near-field memory needs finite world endpoints')
        moving=None
        if transient_mask is not None:
            moving=np.asarray(transient_mask)
            if moving.shape!=(len(points),) or not np.isin(moving,[0,1]).all():
                raise ValueError('Transient mask must match near-field endpoints')
        if len(points):
            cells=np.floor((points[:,:2]-origin)/r).astype(int)
            keep=(cells>=0).all(1)&(cells[:,0]<shape[1])&(cells[:,1]<shape[0])
            cells,points=cells[keep],points[keep];x,y=cells.T
            if moving is not None:moving=moving[keep]
            reference=self.ground_cache if self.settings['temporary_obstacle_overlay'] else self.fields['ground_z']
            heights=points[:,2]-reference[y,x]
            eligible=np.isfinite(heights) if self.settings['temporary_obstacle_overlay'] else valid[y,x]
            seen=eligible&np.isfinite(heights)
            selected=np.flatnonzero(seen)
            seen[selected]=(heights[selected]>=self.config['min_obstacle_height_m'])&(heights[selected]<=self.config['max_height_m'])
            hit=hit.copy();hit[y[seen],x[seen]]=True
            if self.settings['temporary_obstacle_overlay'] and seen.any():
                lower=np.full(shape,np.inf);upper=np.full(shape,-np.inf)
                np.minimum.at(lower,(y[seen],x[seen]),points[seen,2])
                np.maximum.at(upper,(y[seen],x[seen]),points[seen,2])
                touched=np.isfinite(lower)
                self.hit_stamp[touched]=stamp;self.hit_min[touched]=lower[touched];self.hit_max[touched]=upper[touched]
                ttl=np.full(shape,self.config['obstacle_ttl_s'])
                ordinary=touched.copy()
                if moving is not None:
                    ordinary=np.zeros(shape,bool);ordinary[y[seen&~moving.astype(bool)],x[seen&~moving.astype(bool)]]=True
                    ttl[touched&~ordinary]=min(self.config['obstacle_ttl_s'],self.config['dynamic']['transient_ttl_s'])
                self.ordinary_stamp[ordinary]=stamp
                self.hit_expires[touched]=np.maximum(stamp+ttl[touched],self.ordinary_stamp[touched]+self.config['obstacle_ttl_s'])
        invalidated=int((valid&hit).sum())
        certified=(raw['occupancy']==0)&raw['ground_support']&(
            raw['coverage']>=self.config['free_vertical_coverage'])&~hit&~paused
        retained=valid if self.settings['remember_outside_ring'] else valid&blind
        blocked=np.zeros(shape,bool) if self.settings['temporary_obstacle_overlay'] else hit
        valid=(retained|certified)&~blocked&~conflict
        self.confirmed[~valid]=np.nan
        certified&=~conflict
        self.confirmed[certified]=stamp
        for k in self.fields:self.fields[k][certified]=raw[k][certified]
        # Newly certified columns have post-hit free rays under the guarded model.
        self.hit_stamp[certified]=np.nan
        self.hit_expires[certified]=np.nan
        self.ordinary_stamp[certified]=-np.inf
        possible=np.isfinite(self.ground_cache)&blind&np.isfinite(self.hit_stamp)&np.isfinite(self.hit_expires)
        active=np.zeros(shape,bool);active[possible]=stamp<=self.hit_expires[possible]
        restored=valid&blind&(raw['occupancy']==-1)&~active&~paused
        layers=dict(raw)
        if restored.any() or active.any() or paused.any():
            layers={k:v.copy() for k,v in raw.items()}
            layers['occupancy'][paused&(layers['occupancy']==0)]=-1
            layers['ground_support'][paused]=False
            for k,v in self.fields.items():layers[k][restored]=v[restored]
            layers['occupancy'][restored]=0
            layers['ground_support'][restored]=True
            layers['observed'][restored]=True
            layers['obstacle_min_z'][restored]=np.nan;layers['obstacle_max_z'][restored]=np.nan
            layers['occupancy'][active]=100
            layers['ground_z'][active]=self.ground_cache[active]
            layers['ground_support'][active]=True
            layers['coverage'][active]=0.
            layers['observed'][active]=True
            layers['last_observed'][active]=np.fmax(layers['last_observed'][active],self.hit_stamp[active])
            layers['obstacle_min_z'][active]=self.hit_min[active]-self.ground_cache[active]
            layers['obstacle_max_z'][active]=self.hit_max[active]-self.ground_cache[active]
            layers.update(finish_evidence(layers['occupancy'],layers['observed'],layers['last_observed'],
                layers['coverage'],layers['obstacle_min_z'],layers['obstacle_max_z'],layers['ground_z'],
                layers['ground_support'],r))
        layers['near_field_free']=restored
        layers['near_field_free_classified_at']=np.where(restored,self.confirmed,np.nan)
        if self.settings['temporary_obstacle_overlay']:
            layers['near_field_obstacle']=active
            layers['near_field_obstacle_stamp']=np.where(blind,self.hit_stamp,np.nan)
            layers['near_field_obstacle_expires_at']=np.where(blind,self.hit_expires,np.nan)
        if person_only is not None:
            person_only=np.asarray(person_only)
            if person_only.dtype!=bool or person_only.shape!=shape:raise ValueError('Invalid exclusive person columns')
            layers['person_background_free']=valid&person_only&(layers['occupancy']==100)&~paused
        metadata=dict(m,snapshot_id=m['snapshot_id']+':near_field:'+self.binding,
            near_field_memory=self.describe(m['snapshot_id'],[valid.sum(),
                0 if self.settings['temporary_obstacle_overlay'] else invalidated,conflict.sum(),
                invalidated,restored.sum(),active.sum()],ring,'cpu'))
        for array in layers.values():array.flags.writeable=False
        return GridSnapshot(metadata,layers)
