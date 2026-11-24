"""Revisable historical priors; temporary returns never become permanent walls."""
import hashlib
import numpy as np
import cv2

from .context import fingerprint
from .layers import GridSnapshot, finish_evidence
from .geometry import MAX_OVERLAP_CELLS, distance_to_zero, square_overlaps


def cell_evidence(points, rotation, target_resolution, local, *, confirmed_free=None):
    """Exact SAT overlap of target squares with the local axis-aligned raster."""
    m, layers=local.metadata,local.layers
    cells=layers['occupancy']
    free=np.ones(len(points),bool);hit=np.zeros(len(points),bool);seen=np.full(len(points),np.nan)
    occupied_overlap=np.zeros(len(points),bool)
    if confirmed_free is None:
        supported=layers.get('ground_support',np.zeros_like(cells,bool))
        valid_free=(cells==0)&supported&(layers['coverage']>=1.-1e-6)
        if 'near_field_free' in layers:valid_free&=~layers['near_field_free']
        observed=layers['last_observed']
    else:
        valid_free=np.isfinite(confirmed_free)&(cells!=100)
        observed=confirmed_free
    for rows,y,x,outside,interior in square_overlaps(points,rotation,target_resolution,local,include_contacts=True):
        free[outside]=False
        occupied=cells[y,x]==100
        free[rows]&=valid_free[y,x];hit[rows]|=occupied
        occupied_overlap[rows]|=occupied&interior
        seen[rows]=np.fmin(seen[rows],observed[y,x])
    free &= np.isfinite(seen)&(seen<=m['stamp_s']+1e-6)
    return free,hit,seen,occupied_overlap


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
        self.has_confirmed_free = False

    def apply(self, prior, local, association, *, scene_binding, bounds=None, overlay_current=True,
              geometry_only=False, confirmed_free=None):
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
               association.get('association_session'), association.get('association_generation'),
               association.get('map_revision'),local.metadata['frame_id'])
        if prior is not self.source or key != self.key or (self.last is not None and stamp < self.last):
            self.reset(); self.key = key; self.source = prior
            self.cleared = np.zeros_like(prior.layers['occupancy'], bool)
            self.stamps = np.full(self.cleared.shape, np.nan)
        self.last = stamp
        self.has_confirmed_free |= confirmed_free is not None
        cells = local.layers['occupancy']; lm, pm = local.metadata, prior.metadata
        # Only local cells intersecting this raster need projection. Bounding
        # target cells by a half diagonal covers arbitrary sub-cell alignment.
        h,w=cells.shape
        local_xy = (np.array([[0,0],[0,h-1],[w-1,0],[w-1,h-1]])+.5)*lm['resolution']+lm['origin_xy']
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
            y,x = y.ravel(),x.ravel()
            xy = (np.column_stack((x,y))+.5)*pm['resolution']+pm['origin_xy']
            sample = (xy-map_pose) @ rotation+local_pose
            # Near the scale limit, use the full set so the exact budget check
            # still sees every alignment, including regions with no corrections.
            span_bound = np.ceil(pm['resolution']/lm['resolution']*np.abs(rotation).sum(axis=0)+2e-9)+1
            if not overlay_current and span_bound.prod() <= MAX_OVERLAP_CELLS:
                # This consumer only needs updates to historical free cells. A
                # new one must cover its centre with locally certified free
                # space. Existing corrections always get the full revoke test.
                prior_cells = prior.layers['occupancy'][y,x]
                eligible = prior_cells!=0 if confirmed_free is not None else prior_cells==100
                ij = np.floor((sample-lm['origin_xy'])/lm['resolution']).astype(np.intp)
                cx,cy = ij.T
                inside = (cx>=0)&(cy>=0)&(cx<w)&(cy<h)
                possible = self.cleared[y,x]
                rows = np.flatnonzero(eligible&inside)
                sy,sx = cy[rows],cx[rows]
                if confirmed_free is not None:
                    possible[rows] |= np.isfinite(confirmed_free[sy,sx]) & (cells[sy,sx]!=100)
                elif not self.has_confirmed_free:
                    possible[rows] |= cells[sy,sx]==0
                sample,y,x = sample[possible],y[possible],x[possible]
            certified,occupied,seen,occupied_overlap=cell_evidence(
                sample,rotation,pm['resolution'],local,confirmed_free=confirmed_free)
            if confirmed_free is not None or not self.has_confirmed_free:
                current_free[y[certified],x[certified]]=True
            current_hit[y[occupied],x[occupied]]=True
            # A partial view can revoke a correction when occupied cells
            # intersect its interior. Unknown space requires full coverage.
            forget=occupied_overlap if self.has_confirmed_free else np.zeros(len(sample),bool)
            if confirmed_free is not None:
                extent=pm['resolution']/2*np.abs(rotation).sum(axis=0)
                origin=np.asarray(lm['origin_xy'])
                upper=origin+np.asarray(lm['shape'][::-1])*lm['resolution']
                observable=((sample-extent>=origin)&(sample+extent<=upper)).all(axis=1)
                forget|=observable&~certified
            self.cleared[y[forget],x[forget]]=False
            self.stamps[y[forget],x[forget]]=np.nan
        current_free &= ~current_hit
        eligible = prior.layers['occupancy']!=0 if confirmed_free is not None else prior.layers['occupancy']==100
        newly_free = current_free & eligible
        self.cleared |= newly_free
        if (high>low).all():
            selected=newly_free[y,x]
            self.stamps[y[selected],x[selected]]=seen[selected]
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
