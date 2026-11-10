"""Revisable historical priors; temporary returns never become permanent walls."""
import hashlib
import numpy as np
import cv2

from .context import fingerprint
from .layers import GridSnapshot, finish_evidence
from .geometry import distance_to_zero


def cell_evidence(points, rotation, target_resolution, local):
    """Exact SAT overlap of target squares with the local axis-aligned raster."""
    m, layers=local.metadata,local.layers
    cells=layers['occupancy'];res=m['resolution'];a=target_resolution/2;b=res/2
    extent=a*np.abs(rotation).sum(axis=0)
    lower=np.floor((points-extent-m['origin_xy'])/res-1e-9).astype(int)
    upper=np.floor((points+extent-m['origin_xy'])/res+1e-9).astype(int)
    span=(upper-lower).max(axis=0)+1
    if int(np.prod(span))>81:raise ValueError('Prior/local cell scale ratio exceeds the overlap budget')
    free=np.ones(len(points),bool);hit=np.zeros(len(points),bool);seen=np.full(len(points),np.nan)
    supported=layers.get('ground_support',np.zeros_like(cells,bool))
    valid_free=(cells==0)&supported&(layers['coverage']>=1.-1e-6)
    # Reused background is useful locally but cannot become new negative
    # evidence that permanently clears a separate historical map asset.
    if 'near_field_free' in layers:valid_free&=~layers['near_field_free']
    for dy in range(span[1]):
        for dx in range(span[0]):
            ij=lower+[dx,dy];centres=(ij+.5)*res+m['origin_xy'];delta=centres-points
            overlap=((np.abs(delta)<=extent+b+1e-9).all(axis=1)&
                     (np.abs(delta@rotation.T)<=a+b*np.abs(rotation).sum(axis=1)+1e-9).all(axis=1))
            inside=((ij>=0)&(ij<np.asarray(cells.shape[::-1]))).all(axis=1)
            free[overlap & ~inside]=False
            rows=np.flatnonzero(overlap & inside);x,y=ij[rows].T
            free[rows]&=valid_free[y,x];hit[rows]|=cells[y,x]==100
            seen[rows]=np.fmax(seen[rows],layers['last_observed'][y,x])
    free &= np.isfinite(seen)&(seen<=m['stamp_s']+1e-6)
    return free,hit,seen


def conditional_obstacles(layers, config):
    """A geometric route hypothesis, deliberately not a semantic movable label."""
    occupied = layers['occupancy'] == 100
    n, labels, stats, _ = cv2.connectedComponentsWithStats(occupied.astype(np.uint8), 8)
    settings, resolution = config['static'], config['resolution']
    selected = np.zeros(n, bool)
    for label in range(1, n):
        x, y, width, height, area = stats[label]
        selected[label] = (area*resolution**2 <= settings['conditional_max_area_m2'] and
            max(width, height)*resolution <= settings['conditional_max_extent_m'] and
            x > 0 and y > 0 and x+width < occupied.shape[1] and y+height < occupied.shape[0])
    return selected[labels] & layers['ground_support']


class PriorRevisions:
    """Session-scoped free corrections; only current occupied overlays expire.

    Historical assets stay immutable. Only confirmed, supported full-height local
    free space can clear their occupied cells. Occlusion/absence cannot do so.
    A new asset, localization/association lifecycle or backward seek discards the
    ledger. Call reset on an explicit seek, including seeks to the same timestamp.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = self.cleared = self.stamps = self.last = self.source = None

    def apply(self, prior, local, association, *, scene_binding, bounds=None, overlay_current=True, geometry_only=False):
        from perception.detection_tracking.inference.geometry import validate_transform
        if bounds is not None:
            bounds=np.asarray(bounds,dtype=float)
            if bounds.shape!=(4,) or not np.isfinite(bounds).all() or (bounds[2:]<=bounds[:2]).any():
                raise ValueError('Revision bounds must be finite ordered XY limits')
        transform = validate_transform(association['map_from_local'])
        # Navigation is planar. Reject a transform outside that approximation
        # rather than clearing walls using a strongly tilted projection.
        tilt = np.arccos(np.clip(transform[2, 2], -1., 1.))
        required={'occupancy','observed','last_observed','coverage','ground_z','ground_support',
                  'obstacle_min_z','obstacle_max_z'}
        reason=('map_local_tilt_exceeds_planar_limit' if tilt>.15 else
                'evidence_layers_unavailable' if not required.issubset(prior.layers) or
                not {'ground_support','coverage','last_observed'}.issubset(local.layers) else None)
        if reason:
            return GridSnapshot(dict(prior.metadata,prior_revision=dict(status='unavailable',reason=reason,
                source_snapshot_id=prior.metadata['snapshot_id'],scene_binding=scene_binding)),prior.layers)
        stamp = float(local.metadata['stamp_s'])
        key = (prior.metadata['snapshot_id'], local.metadata['session_id'], local.metadata['generation'],
               association.get('association_session'), association.get('association_generation'))
        if prior is not self.source or key != self.key or (self.last is not None and stamp < self.last):
            self.reset(); self.key = key; self.source = prior
            self.cleared = np.zeros_like(prior.layers['occupancy'], bool)
            self.stamps = np.full(self.cleared.shape, np.nan)
        self.last = stamp
        cells = local.layers['occupancy']; lm, pm = local.metadata, prior.metadata
        # Only local cells intersecting this raster need projection. Bounding
        # target cells by a half diagonal covers arbitrary sub-cell alignment.
        yy, xx = np.indices(cells.shape)
        local_xy = np.stack((xx.ravel()+.5, yy.ravel()+.5), axis=1)*lm['resolution']+lm['origin_xy']
        yaw = np.arctan2(transform[1, 0], transform[0, 0]); c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c,-s],[s,c]])
        local_pose = np.asarray(lm['world_from_lidar'])[:2,3]
        map_pose = (transform @ np.asarray(lm['world_from_lidar']))[:2,3]
        mapped = (local_xy-local_pose) @ rotation.T+map_pose
        low = np.maximum(0,np.floor((mapped.min(axis=0)-pm['resolution']-pm['origin_xy'])/pm['resolution']).astype(int))
        high = np.minimum(pm['shape'][::-1],np.ceil((mapped.max(axis=0)+pm['resolution']-pm['origin_xy'])/pm['resolution']).astype(int))
        current_free = np.zeros(self.cleared.shape, bool)
        current_hit = np.zeros(self.cleared.shape, bool)
        if (high>low).all():
            y,x = np.mgrid[low[1]:high[1],low[0]:high[0]]
            xy = np.stack((x.ravel()+.5,y.ravel()+.5),axis=1)*pm['resolution']+pm['origin_xy']
            sample = (xy-map_pose) @ rotation+local_pose
            certified,occupied,seen=cell_evidence(sample,rotation,pm['resolution'],local)
            current_free[y.ravel()[certified],x.ravel()[certified]]=True
            current_hit[y.ravel()[occupied],x.ravel()[occupied]]=True
        current_free &= ~current_hit
        newly_free = current_free & (prior.layers['occupancy']==100)
        self.cleared |= newly_free
        if (high>low).all():
            selected=newly_free[y.ravel(),x.ravel()]
            self.stamps[y.ravel()[selected],x.ravel()[selected]]=seen[selected]
        begin,end=np.zeros(2,int),np.asarray(pm['shape'][::-1])
        if bounds is not None:
            begin=np.maximum(begin,np.floor((np.asarray(bounds[:2])-pm['origin_xy'])/pm['resolution']).astype(int))
            end=np.minimum(end,np.ceil((np.asarray(bounds[2:])-pm['origin_xy'])/pm['resolution']).astype(int))
            if (end<=begin).any():raise ValueError('Revision crop outside prior raster')
        region=np.s_[begin[1]:end[1],begin[0]:end[0]]
        layers={k:v[region].copy() for k,v in prior.layers.items() if not geometry_only or k in ('occupancy','conditional_obstacle')}
        cleared,stamps=self.cleared[region],self.stamps[region]
        current_hit=current_hit[region]
        if not overlay_current:current_hit=np.zeros_like(current_hit)
        conditional=layers.setdefault('conditional_obstacle',np.zeros(cleared.shape,bool))
        layers['occupancy'][cleared] = 0
        layers['occupancy'][current_hit] = 100
        conditional[cleared|current_hit] = False
        if 'assumed_free' in layers:
            layers['assumed_free'][cleared|current_hit] = False
        if 'manual_edit' in layers:
            layers['manual_edit'][cleared|current_hit] = False
        # No temporary occupied observation is written into the persistent ledger.
        # Navigation uses only occupancy/risk and builds its own safety distance
        # field. Do not copy/recompute display and terrain layers it never reads.
        if not geometry_only:
            layers['last_observed'][cleared] = stamps[cleared]
            layers['coverage'][cleared] = 1.
            layers['observed'][cleared] = True
            layers['observed'][current_hit] = True
            layers['last_observed'][current_hit] = stamp
            for name in ('obstacle_min_z','obstacle_max_z'):
                layers[name][cleared|current_hit] = np.nan
            derived = finish_evidence(layers['occupancy'],layers['observed'],layers['last_observed'],layers['coverage'],
                layers['obstacle_min_z'],layers['obstacle_max_z'],layers['ground_z'],layers['ground_support'],pm['resolution'])
            layers.update(derived)
            layers['revised_free_stamp'] = stamps.copy()
        revision = dict(status='valid',source_snapshot_id=pm['snapshot_id'],scene_binding=scene_binding,as_of=stamp,
            local_session_id=key[1],local_generation=key[2],association_session=key[3],association_generation=key[4],
            ledger_sha256=hashlib.sha256(self.stamps.tobytes()).hexdigest(),
            output_cells=[begin.tolist(),end.tolist()],
            free_corrections=int(self.cleared.sum()),current_free_corrections=int(newly_free.sum()),
            occupied_overlay_cells=int(current_hit.sum()),projection='exact_SE2_square_overlap',
            occupied_overlay_enabled=overlay_current,
            occupied_overlay_persistent=False,source_occupancy_immutable=True)
        metadata = dict(pm,prior_revision=revision,snapshot_id='revised_prior:'+fingerprint(revision),
            origin_xy=(np.asarray(pm['origin_xy'])+begin*pm['resolution']).tolist(),shape=list(cleared.shape))
        if geometry_only:metadata['layer_scope']='planning_occupancy_and_risk'
        for v in layers.values(): v.flags.writeable = False
        return GridSnapshot(metadata,layers)


def reference_grid(prior,*,geometry_only=False):
    """Optimistic global-reference search; never use for a current local check."""
    conditional = prior.layers.get('conditional_obstacle')
    if conditional is None or not conditional.any(): return prior
    layers = dict(prior.layers)
    cells = layers['occupancy'].copy()
    cells[conditional] = 0
    layers['occupancy'] = cells
    # Clearance must follow this search grid, not the immutable historical one.
    if not geometry_only:
        distance = (distance_to_zero(cells!=100)*prior.metadata['resolution'] if (cells==100).any()
                    else np.full(cells.shape,np.nan))
        distance[cells==-1] = np.nan
        layers['clearance'] = distance.astype(np.float32)
    else:
        layers={k:v for k,v in layers.items() if k in ('occupancy','conditional_obstacle')}
    for v in layers.values(): v.flags.writeable = False
    return GridSnapshot(dict(prior.metadata,reference_policy='conditional_obstacles_need_reobservation',
        snapshot_id=prior.metadata['snapshot_id']+':conditional_reference'),layers)
