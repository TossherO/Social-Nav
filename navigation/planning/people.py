"""Causal deterministic CV prediction with a fixed physical body radius.

Unknown velocity retains a current-occupancy proxy, never a measured zero speed.
This is a development model, not a guarantee about human motion.
"""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from perception.environment.context import fingerprint
from navigation.kinematics import sample_trajectory
from .footprint import wrap


def stable_identity(data,track):
    return (data['session_id'],data['generation'],str(track['track_id']))


class PeopleModel:
    def __init__(self,scene,config):
        self.config=config;self.as_of=scene.record['as_of'];self.status='missing';self.reason='people_unavailable'
        if config.get('prediction_model')!='deterministic':raise ValueError('Unsupported people prediction model')
        self.tracks=[];self.ids=[];self.origin=self.as_of;self.source=None
        self._navigation_grid=None
        from perception.environment.person_objects import observed_objects
        self.observed_objects=observed_objects(scene.geometry)
        self.delegated=set()
        people=scene.record.get('people',{})
        if people.get('status')!='valid':return
        data=people.get('data',{});r=scene.record
        try:
            age=self.as_of-data['stamp_s'];binding=data.get('localization_binding') or {}
            if (not 0<=age<=config['max_observation_age_s'] or data['frame_id']!=r['frame_id'] or
                    [binding.get(k) for k in ('session_id','generation','frame_id')]!=[r[k] for k in ('session_id','generation','frame_id')]):
                raise ValueError('people_frame_lifecycle_or_age_invalid')
            self.origin=data['stamp_s'];self.source=data['input_fingerprint']
            if len(data['tracks'])>config['max_tracks']:raise ValueError('people_prediction_budget_exceeded')
            for track in data['tracks']:
                last=track['last_observed_stamp_s']
                if type(last) not in (int,float) or not np.isfinite(last):raise ValueError('invalid_person_support_time')
                if last>self.origin:raise ValueError('future_person_support')
                if self.as_of-last>config['max_track_age_s']:continue
                box=np.asarray(track['box'],float)
                if box.shape not in ((7,),(8,)) or not np.isfinite(box).all() or (box[3:6]<=0).any():
                    raise ValueError('invalid_person_box')
                key=stable_identity(data,track)
                if key in self.ids:raise ValueError('duplicate_person_identity')
                velocity=track.get('velocity_xyz_mps')
                known=(data.get('velocity_semantics')=='world_motion' and bool(track.get('confirmed')) and
                    velocity is not None and np.shape(velocity)==(3,) and np.isfinite(velocity).all())
                covariance=track.get('covariance')
                if covariance is not None:
                    if data.get('covariance_order')!=['x','y','z','dx','dy','dz','yaw','vx','vy','vz']:
                        raise ValueError('unknown_person_covariance_order')
                    covariance=np.asarray(covariance,float)
                    if covariance.size!=100 or not np.isfinite(covariance).all():raise ValueError('invalid_person_covariance')
                    covariance=covariance.reshape(10,10)[np.ix_([0,1,7,8],[0,1,7,8])]
                    if (not np.allclose(covariance,covariance.T,atol=1e-7) or
                            np.linalg.eigvalsh(covariance).min()<-1e-7):raise ValueError('invalid_person_covariance')
                self.ids.append(key)
                self.tracks.append(dict(identity=key,xy=box[:2],velocity=np.asarray(velocity)[:2] if known else None,
                    radius=config['body_radius_m'],covariance=covariance,
                    covariance_source='tracker' if track.get('covariance') is not None else 'missing',last_observed=last))
            if self.tracks:
                self.xy=np.array([t['xy'] for t in self.tracks]);self.known=np.array([t['velocity'] is not None for t in self.tracks])
                self.velocity=np.array([t['velocity'] if t['velocity'] is not None else [0.,0.] for t in self.tracks])
                self.radii=np.array([t['radius'] for t in self.tracks])
                ownership=scene.geometry.metadata.get('person_objects',{})
                if ownership.get('people_source')==self.source:
                    by_id={key:i for i,key in enumerate(self.ids)}
                    for obj in self.observed_objects:
                        key=tuple(obj['identity']);i=by_id.get(key)
                        if i is None or not np.allclose(self.xy[i],obj['xy'],rtol=0,atol=1e-8):continue
                        # Each track is independent. Unknown velocity uses its
                        # current circle; another person's velocity cannot gate it.
                        self.radii[i]=max(self.radii[i],obj['radius_m'])
                        self.tracks[i]['radius']=self.radii[i]
                        self.delegated.add(key)
            self.group_cache=None
            self.status='valid';self.reason=None
        except (KeyError,ValueError,TypeError) as error:
            self.status='invalid';self.reason=str(error);self.tracks=[];self.ids=[]

    def summary(self):
        return dict(status=self.status,reason=self.reason,source=self.source,source_stamp_s=self.origin,
            as_of=self.as_of,tracks=len(self.tracks),unknown_velocities=sum(t['velocity'] is None for t in self.tracks),
            horizon_s=self.config['horizon_s'],model='deterministic_world_CV',
            unknown_velocity_policy='current_occupancy_hold_not_measured_motion',
            uncertainty_applied=False,unknown_velocity_uncertainty='disabled',current_map_unchanged=True)

    def positions(self,times,*,components=False):
        tau=np.atleast_1d(times)-self.origin
        # The zero displacement for unknown motion is an occupancy proxy only.
        centre=self.xy[None]+tau[:,None,None]*self.velocity[None]
        radius=np.broadcast_to(self.radii,centre.shape[:2])
        if components:
            zeros=np.zeros_like(radius)
            return centre,radius,dict(covariance=zeros,acceleration=zeros,unknown_velocity=zeros)
        return centre,radius

    def navigation_grid(self,scene):
        """A separate collision view; the bound perception map stays unchanged.

        Only exclusive person hits over previously certified free background
        may be represented by time-dependent people instead of frozen cells.
        Missing/unknown motion or mismatched mask sources retain raw geometry.
        """
        if self._navigation_grid is not None:return self._navigation_grid
        from perception.environment.context import layer_binding
        from perception.environment.layers import GridSnapshot
        from perception.environment.navigation import navigation_geometry
        raw=scene.geometry;mask=raw.layers.get('person_background_free')
        status=raw.metadata.get('dynamic_filter',{})
        report=dict(status='raw_geometry',restored_cells=0,source_scene=scene.record['input_fingerprint'],
            source_layers=scene.record['geometry']['layer_binding'],people_source=self.source)
        grid=navigation_geometry(scene,include_people=False)
        binding=scene.record['geometry']['layer_binding'] if grid is raw else layer_binding(grid)
        if grid is not raw:
            report.update(status='static_prior_fusion',prior_fusion=raw.metadata['navigation_fusion'])
        if self.observed_objects:
            from perception.environment.person_objects import circle_view
            fallback=[obj for obj in self.observed_objects if tuple(obj['identity']) not in self.delegated]
            grid=circle_view(grid,fallback);binding=layer_binding(grid)
            report.update(status='tracked_person_objects',delegated_tracks=len(self.delegated),
                fallback_circles=fallback,scope='background_geometry_and_timed_people')
            self._navigation_grid=grid,binding,report
            return self._navigation_grid
        if (self.status=='valid' and self.tracks and self.known.all()
                and mask is not None and status.get('status')=='valid' and status.get('input_fingerprint')==self.source):
            if mask.dtype!=bool or mask.shape!=raw.layers['occupancy'].shape:
                raise ValueError('Invalid person background provenance')
            selected=mask&(raw.layers['occupancy']==100)
            if selected.any():
                cells=grid.layers['occupancy'].copy();cells[selected]=0;cells.flags.writeable=False
                grid=GridSnapshot(dict(raw.metadata,navigation_view='dynamic_person_background_not_new_observation'),dict(occupancy=cells))
                binding=layer_binding(grid)
                report.update(status='people_prediction_over_historical_free',restored_cells=int(selected.sum()),
                    derived_layers=binding,projection_margin_m=raw.metadata['resolution']/np.sqrt(2),
                    scope='collision_view_only_current_people_checked_separately')
        self._navigation_grid=grid,binding,report
        return self._navigation_grid

    def check(self,points,geometry,margin=0.,*,start_time=None,details=False,hard_until=None):
        result=dict(self.summary(),valid=True,scope='timed_people_prediction',minimum_clearance_m=None,spatial_margin_m=float(margin))
        if self.status=='invalid' or self.status=='missing' and self.config['required']:
            return dict(result,valid=False,reason=self.reason)
        if not self.tracks:return dict(result,scope='geometry_only' if self.status=='missing' else 'observed_no_tracks')
        points=np.asarray(points,float);start_time=self.as_of if start_time is None else start_time
        result.update(prediction_start_s=float(start_time),prediction_end_s=float(start_time+points[-1,0]))
        tau=np.array([start_time,start_time+points[-1,0]])-self.origin
        if tau.min()<-1e-8 or tau.max()>self.config['horizon_s']:
            return dict(result,valid=False,reason='people_prediction_horizon_exceeded')
        cutoff=float(points[-1,0]) if hard_until is None else min(float(hard_until),float(points[-1,0]))
        if not np.isfinite(cutoff) or cutoff<=0:raise ValueError('Invalid hard people horizon')
        times=np.unique(np.r_[points[:,0],cutoff,np.arange(0.,points[-1,0],self.config['sample_s'])])
        poses,_=sample_trajectory(points,times)
        # Enclose each entire relative motion interval, including body rotation.
        mid=(times[1:]+times[:-1])/2;body,_=sample_trajectory(points,mid)
        person,radius=self.positions(start_time+mid)
        endpoints,rad,parts=self.positions(start_time+times,components=True)
        delta=person-body[:,None,:2];c,s=np.cos(body[:,2]),np.sin(body[:,2])
        local=np.stack([c[:,None]*delta[:,:,0]+s[:,None]*delta[:,:,1],-s[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1]],axis=-1)
        polygon=geometry.footprint;edge=np.roll(polygon,-1,axis=0)-polygon
        projection=np.clip(np.sum((local[:,:,None]-polygon)*edge,axis=-1)/np.sum(edge**2,axis=1),0,1)
        nearest=polygon+projection[...,None]*edge
        distance=np.linalg.norm(local[:,:,None]-nearest,axis=-1).min(axis=-1)
        cross=edge[:,0]*(local[:,:,None,1]-polygon[:,1])-edge[:,1]*(local[:,:,None,0]-polygon[:,0])
        inside=np.all(cross>=-1e-10,axis=-1)|np.all(cross<=1e-10,axis=-1)
        distance[inside]=0.
        relative=np.diff(poses[:,:2],axis=0)[:,None]-np.diff(endpoints,axis=0)
        turn=abs(wrap(np.diff(poses[:,2])))
        enclosure=(np.linalg.norm(relative,axis=-1)+geometry.radius*turn[:,None])/2
        chord=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)*turn/8
        clearance=distance-np.maximum(rad[:-1],rad[1:])-enclosure-chord[:,None]-margin
        hard=times[:-1]<cutoff-1e-9
        i,j=np.unravel_index(np.argmin(np.where(hard[:,None],clearance,np.inf)),clearance.shape)
        minimum=float(clearance[i,j])
        nominal=distance-self.radii[None]-enclosure-chord[:,None]-margin
        conflicts=np.flatnonzero(np.any(clearance<=0,axis=1)&hard)
        first=None if not len(conflicts) else int(conflicts[0])
        result.update(valid=minimum>0.,reason=None if minimum>0 else 'predicted_people_collision',
            minimum_clearance_m=minimum,minimum_nominal_clearance_m=float(nominal.min()),
            hard_horizon_s=cutoff,minimum_forecast_clearance_m=float(clearance.min()),
            full_forecast_clear=bool((clearance>0).all()),
            future_conflict_duration_s=float(np.diff(times)[(~hard)&np.any(clearance<=0,axis=1)].sum()),
            critical_identity=list(self.ids[j]),critical_relative_time_s=float(mid[i]),
            first_conflict_interval_s=None if first is None else [float(times[first]),float(times[first+1])],
            critical_envelope=dict(velocity_known=bool(self.known[j]),shape_radius_m=float(self.radii[j]),
                covariance_radius_m=float(max(parts['covariance'][i,j],parts['covariance'][i+1,j])),
                acceleration_radius_m=float(max(parts['acceleration'][i,j],parts['acceleration'][i+1,j])),
                unknown_velocity_radius_m=float(max(parts['unknown_velocity'][i,j],parts['unknown_velocity'][i+1,j]))))
        if details:
            result['diagnostic']=dict(times=times.tolist(),body_poses=poses.tolist(),person_centres=endpoints.tolist(),
                person_radii=rad.tolist(),identities=[list(k) for k in self.ids],mid_times=mid.tolist(),
                clearance=clearance.tolist(),nominal_clearance=nominal.tolist())
        return result

    def social_cost(self,points,groups=None):
        if not self.tracks:return 0.
        points=np.asarray(points);times=(points[1:,0]+points[:-1,0])/2;dt=np.diff(points[:,0])
        poses,_=sample_trajectory(points,times);people,_=self.positions(self.as_of+times)
        delta=poses[:,None,:2]-people
        velocity=self.velocity.copy()
        moving=np.linalg.norm(velocity,axis=1)>.15
        velocity[~moving]=[1.,0.]
        direction=velocity/np.maximum(np.linalg.norm(velocity,axis=1)[:,None],1e-9)
        along=np.sum(delta*direction[None],axis=-1);across=delta[:,:,0]*direction[None,:,1]-delta[:,:,1]*direction[None,:,0]
        front=np.where(along>=0,self.config['personal_front_m'],self.config['personal_back_m'])
        front[:,~moving]=self.config['personal_side_m']
        value=np.exp(-.5*((along/front)**2+(across/self.config['personal_side_m'])**2)).sum(axis=1)*self.config['individual_cost']
        pairs=self.group_pairs(groups)
        if pairs:
            a,b=np.asarray(pairs).T
            v=people[:,b]-people[:,a];u=poses[:,None,:2]-people[:,a]
            t=np.clip(np.sum(u*v,axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-9),0,1)
            d=np.linalg.norm(u-t[...,None]*v,axis=-1)
            value+=self.config['group_cost']*np.exp(-.5*(d/self.config['group_distance_m'])**2).sum(axis=1)
        return float(np.sum(value*dt))

    def group_pairs(self,groups):
        # A short spanning tree connects a group without an O(N^2) clique or
        # quadratic cost inflation for larger groups. Cache its topology once
        # per observation; evaluate links at the predicted positions in time.
        key=tuple(tuple(tuple(member) for member in group) for group in groups or [])
        if self.group_cache is None or self.group_cache[0]!=key:
            lookup={tuple(key):i for i,key in enumerate(self.ids)};pairs=[]
            for group in key:
                members=np.array([lookup[member] for member in group if member in lookup],int)
                if len(members)<2:continue
                xy=self.xy[members];distance=np.linalg.norm(xy[:,None]-xy[None],axis=-1)
                distance+=1e-6;np.fill_diagonal(distance,0.)
                a,b=minimum_spanning_tree(distance).nonzero();pairs.extend(zip(members[a],members[b]))
            self.group_cache=(key,pairs)
        return self.group_cache[1]


def social_groups(request,scene,model):
    context=request.get('social')
    if context is None:return [],dict(status='absent')
    required={'source','stamp_s','clock_domain','local_identity','groups','binding'}
    if set(context)!=required or context['binding']!=fingerprint({k:v for k,v in context.items() if k!='binding'}):
        raise ValueError('invalid_social_binding')
    r=scene.record
    if (context['clock_domain']!=r['clock_domain'] or context['local_identity']!=[r[k] for k in ('session_id','generation','frame_id')]):
        raise ValueError('social_lifecycle_mismatch')
    age=r['as_of']-context['stamp_s']
    if age<0:raise ValueError('future_social_context')
    if age>model.config['decision_max_age_s']:return [],dict(status='expired',source=context['source'])
    groups=context['groups']
    if (not isinstance(groups,list) or len(groups)>model.config['max_tracks']//2 or
            any(not isinstance(group,list) or not 2<=len(group)<=model.config['max_tracks'] for group in groups)):
        raise ValueError('invalid_social_groups')
    keys=[key for group in groups for key in group]
    if (len(keys)>model.config['max_tracks'] or any(not isinstance(key,list) or len(key)!=3 or
            not isinstance(key[0],str) or type(key[1]) is not int or not isinstance(key[2],str) for key in keys) or
            len(set(tuple(key) for key in keys))!=len(keys)):raise ValueError('invalid_social_identity')
    return context['groups'],dict(status='valid',source=context['source'],age_s=age)
