"""Bounded 2D static support over raw maps, with a small endpoint overlay.

One stable floor reference per association; no inferred holes, voxel probes or
person ownership history. Independent consumers recompute static-cell overlaps.
"""
import copy
import hashlib
import numpy as np

from .context import fingerprint
from .geometry import shift_array
from .layers import GridSnapshot
from .navigation import (NavigationGround, GROUND_DEFAULTS, association_reason,
                         static_support, prior_floor, transform_plane, NAVIGATION_DTYPES)

METHOD = 'bounded_static_support'


def region(metadata, radius):
    y,x=np.indices(metadata['shape']);r=metadata['resolution']
    xy=np.stack((x+.5,y+.5),axis=-1)*r+metadata['origin_xy']
    delta=np.abs(xy-np.asarray(metadata['world_from_lidar'])[:2,3])+r/2
    return (delta**2).sum(axis=-1)<=radius**2


def confirmation_candidates(snapshot, prior, transform, plane, selected, tolerance):
    """Cheap rejection only; accepted candidates still require full SAT overlap."""
    m,pm=snapshot.metadata,prior.metadata;y,x=np.nonzero(selected)
    xy=np.c_[x+.5,y+.5]*m['resolution']+m['origin_xy']
    mapped=np.c_[xy,xy@plane[:2]+plane[2]]@transform[:3,:3].T+transform[:3,3]
    ij=np.floor((mapped[:,:2]-pm['origin_xy'])/pm['resolution']).astype(int)
    inside=((ij>=0)&(ij<np.asarray(pm['shape'][::-1]))).all(axis=1)
    rows=np.flatnonzero(inside);xx,yy=ij[rows].T;l=prior.layers
    valid=((l['occupancy'][yy,xx]==0)&l['ground_support'][yy,xx]&
           (np.abs(l['ground_z'][yy,xx]-mapped[rows,2])<=tolerance)&
           np.isfinite(l['last_observed'][yy,xx])&(l['last_observed'][yy,xx]<=m['stamp_s']))
    result=np.zeros(m['shape'],bool);rows=rows[valid];result[y[rows],x[rows]]=True
    return result


class NavigationSupport:
    def __init__(self, config, *, library=None):
        self.config=config
        self.library=library
        self.settings=dict(enabled=False,radius_m=3.5)
        self.settings.update(config.get('navigation_support',{}))
        if (type(self.settings['enabled']) is not bool or
                not np.isfinite(self.settings['radius_m']) or not 0<self.settings['radius_m']<=9):
            raise ValueError('Invalid navigation support region')
        self.ground=NavigationGround(dict(GROUND_DEFAULTS))
        self.reset()

    def reset(self):
        self.key=None;self.stamp=None;self.origin=None;self.reference=None
        self.hits=None;self.cached_asset=None;self.ready=None;self.ground.reset()

    def _prepare(self, snapshot, asset):
        m=snapshot.metadata;stamp=m['stamp_s'];a=m.get('association',{})
        reason=association_reason(snapshot,asset,self.config['association_max_age_s'])
        if reason:
            self.reset();return None,reason
        key=(m['session_id'],m['generation'],m['frame_id'],m['resolution'],tuple(m['shape']),
             a['map_id'],a['map_frame'],a.get('association_session'),a.get('association_generation'),
             a.get('map_revision'),
             asset.manifest['snapshot_id'],asset.manifest['layers_sha256'])
        if key!=self.key or (self.stamp is not None and stamp<=self.stamp):
            self.reset();self.key=key
        self.stamp=stamp
        transform=np.asarray(a['map_from_local'],float)
        ground=m.get('ground_tracking',{})
        if ground.get('clear_ground_history'):
            self.reference=None;self.hits=None;self.ground.reset()
        if self.reference is None:
            if ground.get('status')!='tracked' or ground.get('plane_world') is None:
                return None,'floor_unavailable'
            plane,model=self.ground.update(snapshot,transform,self.settings['radius_m'])
            if plane is None:return None,'floor_warming'
            selected=region(m,self.settings['radius_m'])
            selected=confirmation_candidates(snapshot,asset.snapshot,transform,plane,selected,GROUND_DEFAULTS['ground_tolerance_m'])
            minimum=max(1,int(np.ceil(.25/m['resolution']**2)))
            if selected.sum()<minimum:return None,'floor_prior_unconfirmed'
            allowed,_=static_support(snapshot,asset.snapshot,transform,plane,
                [self.config['min_obstacle_height_m'],self.config['max_height_m']],selected,
                GROUND_DEFAULTS['ground_tolerance_m'])
            indices=np.flatnonzero(selected)[allowed]
            if len(indices)<minimum:return None,'floor_prior_unconfirmed'
            confirmation={k:copy.deepcopy(m[k]) for k in ('stamp_s','resolution','shape','origin_xy','world_from_lidar')}
            confirmation.update(map_from_local=transform.tolist(),cells=indices[:minimum].tolist())
            self.reference=dict(map_plane=model['map_plane'],stamp_s=stamp,model=model,
                confirmation=confirmation,
                association_identity=[a['map_id'],a['map_frame'],a.get('association_session'),a.get('association_generation'),a.get('map_revision')])
        return transform_plane(np.asarray(self.reference['map_plane']),np.linalg.inv(transform)),None

    def apply(self, snapshot, asset=None, *, observation=None, metrics=None):
        if not self.settings['enabled']:return snapshot
        if 'navigation_fusion' in snapshot.metadata:raise ValueError('Support requires the raw snapshot')
        plane,reason=self._prepare(snapshot,asset)
        m=snapshot.metadata
        if plane is None:
            metadata=dict(m,navigation_support_status=reason)
            if hasattr(snapshot,'consume'):
                from .device import DeviceGridSnapshot
                return DeviceGridSnapshot.seal(metadata,snapshot.consume())
            return GridSnapshot(metadata,snapshot.layers)
        proof=dict(method=METHOD,source_snapshot_id=m['snapshot_id'],as_of=m['stamp_s'],
            association_binding=fingerprint(m.get('association',{})),max_association_age_s=self.config['association_max_age_s'],
            radius_m=self.settings['radius_m'],height_band=[self.config['min_obstacle_height_m'],self.config['max_height_m']],
            ground_tolerance_m=GROUND_DEFAULTS['ground_tolerance_m'],floor_reference=copy.deepcopy(self.reference),
            obstacle_retention_s=self.config['obstacle_ttl_s'],person_retention_s=min(self.config['obstacle_ttl_s'],self.config['dynamic']['transient_ttl_s']),
            reason=reason,static_map=None if asset is None else {k:asset.manifest[k] for k in ('snapshot_id','layers_sha256')})
        if plane is not None and (observation is None or observation['stamp_s']!=m['stamp_s'] or
                observation['identity']!=(m['session_id'],m['generation'],m['frame_id'])):
            raise ValueError('Static support needs endpoints from this same observation')
        if plane is not None:
            points=observation['endpoints'];moving=observation['transient_mask']
            if (points.dtype!=np.float64 or points.ndim!=2 or points.shape[1]!=3 or not points.flags.c_contiguous or
                    not np.isfinite(points).all() or moving.dtype!=np.uint8 or moving.shape!=(len(points),) or
                    not moving.flags.c_contiguous or not np.isin(moving,[0,1]).all()):
                raise ValueError('Invalid static support endpoints or transient mask')
        if hasattr(snapshot,'consume'):
            result=self._device(snapshot,asset,plane,proof,observation,metrics)
            self.ready=result.ready
        else:
            result=self._host(snapshot,asset,proof,observation)
        self.origin=np.asarray(m['origin_xy'])
        return result

    def _host(self,snapshot,asset,proof,observation):
        m=snapshot.metadata;l=snapshot.layers;cells=l['occupancy'];shape=cells.shape
        nav=cells.copy();source=np.where(cells==0,1,np.where(cells==100,2,0)).astype(np.uint8)
        times=np.where(cells!=-1,l['last_observed'],np.nan)
        selected=region(m,proof['radius_m'])
        transform=np.asarray(m['association']['map_from_local'])
        floor=prior_floor(snapshot,asset.snapshot,transform,self.reference['map_plane'],selected=selected)
        selected&=np.isfinite(floor)
        free,seen=static_support(snapshot,asset.snapshot,transform,np.zeros(3),
            proof['height_band'],selected,proof['ground_tolerance_m'],floor=floor)
        supported=np.zeros(shape,bool);supported[selected]=free
        historical=supported&(cells==-1);nav[historical]=0;source[historical]=3
        prior_times=np.full(shape,np.nan);prior_times[selected]=seen;times[historical]=prior_times[historical]
        if self.hits is None:self.hits=np.full((2,*shape),-np.inf)
        else:
            delta=(self.origin-np.asarray(m['origin_xy']))/m['resolution']
            if not np.allclose(delta,np.rint(delta),rtol=0,atol=1e-6):raise ValueError('Support grid lattice changed')
            dx,dy=np.rint(delta).astype(int)
            self.hits=np.stack([shift_array(v,dx,dy,-np.inf) for v in self.hits])
        points=observation['endpoints'];ij=np.floor((points[:,:2]-m['origin_xy'])/m['resolution']).astype(int)
        inside=((ij>=0)&(ij<np.asarray(shape[::-1]))).all(axis=1)
        height=np.full(len(points),np.nan);xx,yy=ij[inside].T
        height[inside]=points[inside,2]-floor[yy,xx]
        keep=inside&(height>=proof['height_band'][0])&(height<=proof['height_band'][1])
        x,y=ij[keep].T;moving=observation['transient_mask'][keep].astype(int)
        self.hits[moving,y,x]=m['stamp_s']
        active=supported&((m['stamp_s']-self.hits[0]<=proof['obstacle_retention_s'])|
                          (m['stamp_s']-self.hits[1]<=proof['person_retention_s']))
        extra=active&(cells!=100);nav[extra]=100;source[extra]=10;times[extra]=self.hits.max(axis=0)[extra]
        for value in (nav,source,times):value.flags.writeable=False
        result=GridSnapshot(dict(m,navigation_fusion=proof),dict(l,**dict(zip(NAVIGATION_DTYPES,(nav,source,times)))))
        return finalize(result)

    def _device(self,snapshot,asset,plane,proof,observation,metrics):
        import torch
        from .device import DeviceGridSnapshot
        layers=snapshot.consume();device=layers['occupancy'].device;m=snapshot.metadata
        if self.ready is not None:torch.cuda.current_stream(device).wait_event(self.ready)
        from .support_cuda import project
        if self.cached_asset is None:
            names=('occupancy','ground_z','ground_support','last_observed')
            values=np.stack([asset.snapshot.layers[k] for k in names]).astype(np.float64)
            self.cached_asset=torch.from_numpy(values).to(device)
            if metrics is not None:metrics.transfer('h2d',values.nbytes,'navigation_static_asset_once')
        nav,source,times,self.hits=project(snapshot,layers,asset,self.cached_asset,plane,proof,
            observation,self.hits,self.origin,metrics,library_path=self.library)
        layers.update(zip(NAVIGATION_DTYPES,(nav,source,times)))
        return DeviceGridSnapshot.seal(dict(m,navigation_fusion=proof),layers)


def finalize(snapshot):
    """Bind the raw bytes after the single shared host download."""
    proof=snapshot.metadata.get('navigation_fusion',{})
    if proof.get('method')!=METHOD:return snapshot
    metadata=copy.deepcopy(snapshot.metadata)
    metadata['navigation_fusion']['raw_occupancy_sha256']=hashlib.sha256(snapshot.layers['occupancy'].tobytes()).hexdigest()
    return GridSnapshot(metadata,snapshot.layers)


def verify(scene, *, now=None, asset=None, check_asset=False):
    """Independent CPU recheck, including overlap, original times and floor ID."""
    raw=scene.geometry;m=raw.metadata;p=m['navigation_fusion'];l=raw.layers;cells=l['occupancy'];stamp=scene.record['as_of']
    if (any(k not in l or l[k].dtype!=dtype or l[k].shape!=cells.shape for k,dtype in NAVIGATION_DTYPES.items()) or
            p['source_snapshot_id']!=m['snapshot_id'] or p['as_of']!=stamp or
            p['association_binding']!=fingerprint(m.get('association',{})) or
            p['raw_occupancy_sha256']!=hashlib.sha256(cells.tobytes()).hexdigest()):
        raise ValueError('Navigation support source mismatch')
    nav,source,times=(l[k] for k in NAVIGATION_DTYPES)
    extra=source==10;historical=source==3
    if (not np.isin(source,[0,1,2,3,10]).all() or not np.isin(nav,[-1,0,100]).all() or
            np.any((cells==100)&((nav!=100)|(source!=2))) or
            np.any((source==0)&((nav!=-1)|(cells!=-1))) or
            np.any((source==1)&((nav!=0)|(cells!=0))) or
            np.any((source==2)&((nav!=100)|(cells!=100))) or
            np.any(historical&((nav!=0)|(cells!=-1))) or np.any(extra&(nav!=100)) or
            np.any((source!=0)&(~np.isfinite(times)|(times>stamp))) or
            np.any(extra&(stamp-times>p['obstacle_retention_s']))):
        raise ValueError('Navigation support violates obstacle/source policy')
    selected=historical|extra
    if selected.any():
        a=scene.record['prior'];measured=a.get('measurement_stamp_s')
        if (fingerprint(a)!=p['association_binding'] or not a.get('valid') or a.get('status')!='PRIOR_ASSOCIATED' or
                measured is None or not np.isfinite(measured) or not 0< p['max_association_age_s']<=1. or
                not 0< p['radius_m']<=9 or not 0<= (stamp if now is None else now)-measured<=p['max_association_age_s'] or
                np.any(selected&~region(m,p['radius_m']))):
            raise ValueError('navigation_prior_future_or_stale')
        reference=p['floor_reference'];model=reference['model'];samples=model['samples']
        from .navigation import ground_model
        computed,_=ground_model(samples,np.asarray(model['anchor_map_xy']),model['radius_m'])
        if (reference['association_identity']!=[a['map_id'],a['map_frame'],a.get('association_session'),a.get('association_generation'),a.get('map_revision')] or
                not samples or samples[-1]['stamp_s']!=reference['stamp_s'] or reference['stamp_s']>stamp or
                not np.allclose(computed,reference['map_plane'],rtol=0,atol=1e-10)):
            raise ValueError('Navigation support floor mismatch')
        if check_asset:
            if (association_reason(raw,asset,p['max_association_age_s'],now=now) or
                    p['static_map']!={k:asset.manifest[k] for k in ('snapshot_id','layers_sha256')}):
                raise ValueError('Navigation support asset mismatch')
            transform=np.asarray(a['map_from_local']);plane=transform_plane(np.asarray(reference['map_plane']),np.linalg.inv(transform))
            confirmation=reference['confirmation'];cm=GridSnapshot(confirmation,{})
            mask=np.zeros(confirmation['shape'],bool);indices=np.asarray(confirmation['cells'],int)
            minimum=max(1,int(np.ceil(.25/confirmation['resolution']**2)))
            if (len(np.unique(indices))<minimum or not ((indices>=0)&(indices<mask.size)).all() or
                    confirmation['stamp_s']!=reference['stamp_s']):raise ValueError('Navigation floor confirmation invalid')
            mask.ravel()[indices]=True
            if np.any(mask&~region(confirmation,p['radius_m'])):raise ValueError('Navigation floor confirmation outside region')
            ct=np.asarray(confirmation['map_from_local'])
            cp=transform_plane(np.asarray(reference['map_plane']),np.linalg.inv(ct))
            confirmed,_=static_support(cm,asset.snapshot,ct,cp,p['height_band'],mask,p['ground_tolerance_m'])
            if not confirmed.all():raise ValueError('Navigation floor confirmation mismatch')
            floor=prior_floor(raw,asset.snapshot,transform,reference['map_plane'],selected=selected)
            if not np.isfinite(floor[selected]).all():raise ValueError('Navigation floor support absent')
            allowed,seen=static_support(raw,asset.snapshot,transform,np.zeros(3),p['height_band'],selected,p['ground_tolerance_m'],floor=floor)
            if not allowed.all() or not np.array_equal(times[historical],seen[historical[selected]]):
                raise ValueError('Navigation support prior mismatch')
    return GridSnapshot(dict(m,navigation_view='historical_support_not_new_measurement',snapshot_id=m['snapshot_id']+':navigation'),dict(occupancy=nav))
