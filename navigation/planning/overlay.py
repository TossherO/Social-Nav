"""Current local evidence in a temporary global reference grid, without a ledger."""
import numpy as np

from perception.environment.context import fingerprint
from perception.environment.layers import GridSnapshot


def overlap_states(points, rotation, target_resolution, local, raw):
    """A target cell is free only when all positive-area local overlaps are free.

    Only raw local FREE can supply missing prior evidence. Derived static support
    and predicted person background cannot recursively clear the static unknowns.
    """
    m=local.metadata;cells=local.layers['occupancy'];r=m['resolution']
    a=target_resolution/2;b=r/2;extent=a*np.abs(rotation).sum(axis=0)
    low=np.floor((points-extent-m['origin_xy'])/r+1e-9).astype(int)
    high=np.floor((points+extent-m['origin_xy'])/r-1e-9).astype(int)
    span=(high-low).max(axis=0)+1
    if np.prod(span)>81:raise ValueError('Reference/local projection exceeds overlap budget')
    free=np.ones(len(points),bool);hit=np.zeros(len(points),bool)
    for dy in range(span[1]):
        for dx in range(span[0]):
            ij=low+[dx,dy];delta=(ij+.5)*r+m['origin_xy']-points
            overlap=((np.abs(delta)<extent+b-1e-10).all(axis=1)&
                (np.abs(delta@rotation.T)<a+b*np.abs(rotation).sum(axis=1)-1e-10).all(axis=1))
            inside=((ij>=0)&(ij<np.asarray(cells.shape)[::-1])).all(axis=1)
            free[overlap&~inside]=False
            rows=np.flatnonzero(overlap&inside);x,y=ij[rows].T
            free[rows]&=(cells[y,x]==0)&(raw[y,x]==0)
            hit[rows]|=cells[y,x]==100
    return free&~hit,hit


def reference_overlay(prior, local, raw, local_from_map, scene):
    """Caller checks scene/association/asset first; reuse the planner's base SE2."""
    m,pm=local.metadata,prior.metadata;transform=np.asarray(local_from_map)
    report=dict(source_scene=scene.record['input_fingerprint'],source_as_of=scene.record['as_of'],
        source_layers=scene.record['geometry']['layer_binding'],source_prior=pm['snapshot_id'],
        local_from_map=transform.tolist(),projection='positive_area_SE2_overlap_at_current_base',
        unknown_to_free=0,occupied_overlaps_rejected=0,persistent=False,static_occupied_preserved=True,
        free_source='raw_local_free_and_verified_navigation_free')
    association=np.asarray(scene.record['prior']['map_from_local'])
    if np.arccos(np.clip(association[2,2],-1.,1.))>.15:
        return prior,dict(report,status='unavailable',reason='map_local_tilt_exceeds_planar_limit')
    # Map only the bounded local raster. No additional device download is needed.
    origin=np.asarray(m['origin_xy']);size=np.asarray(m['shape'])[::-1]*m['resolution']
    corners=origin+np.array([[0,0],[1,0],[1,1],[0,1]])*size
    rotation=transform[:2,:2]
    mapped=(corners-transform[:2,2])@rotation
    lo=np.maximum(0,np.floor((mapped.min(0)-pm['origin_xy'])/pm['resolution']).astype(int))
    hi=np.minimum(np.asarray(pm['shape'])[::-1],np.ceil((mapped.max(0)-pm['origin_xy'])/pm['resolution']).astype(int))
    if (hi<=lo).any():return prior,dict(report,status='outside_prior')
    y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];x=x.ravel();y=y.ravel()
    unknown=prior.layers['occupancy'][y,x]==-1;x,y=x[unknown],y[unknown]
    xy=np.c_[x+.5,y+.5]*pm['resolution']+pm['origin_xy']
    points=xy@rotation.T+transform[:2,2]
    # Free centre is only a cheap necessary condition. Every accepted target
    # still goes through the complete overlap check below.
    ij=np.floor((points-m['origin_xy'])/m['resolution']).astype(int)
    inside=((ij>=0)&(ij<np.asarray(m['shape'])[::-1])).all(axis=1)
    rows=np.flatnonzero(inside);xx,yy=ij[rows].T
    rows=rows[(local.layers['occupancy'][yy,xx]==0)&(raw[yy,xx]==0)]
    if not len(rows):return prior,dict(report,status='valid')
    x,y,points=x[rows],y[rows],points[rows]
    free,hit=overlap_states(points,rotation.T,pm['resolution'],local,raw)
    report.update(status='valid',unknown_to_free=int(free.sum()),occupied_overlaps_rejected=int(hit.sum()))
    if not free.any():return prior,report
    cells=prior.layers['occupancy'].copy();cells[y[free],x[free]]=0
    cells.flags.writeable=False;layers=dict(occupancy=cells)
    if 'conditional_obstacle' in prior.layers:
        layers['conditional_obstacle']=prior.layers['conditional_obstacle']
    return GridSnapshot(dict(pm,snapshot_id='reference_overlay:'+fingerprint(report),
        reference_overlay=report,layer_scope='temporary_reference_occupancy_and_risk'),layers),report
