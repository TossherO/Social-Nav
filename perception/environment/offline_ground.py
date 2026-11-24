"""Optional scan-supported ground seeds for an offline static-map candidate."""
import hashlib
import numpy as np

from .context import encoded
from .fusion import _scan_geometry
from .geometry import distance_to_zero, ground_cell_statistics, nearest_support
from .ground import GroundTracker
from .layers import GridSnapshot
from .maps import endpoint_layers
from .scans import read_scans, sampled_scan_stamps


def scan_ground(candidate, scan_path, timeline, config, scan_fingerprint, *, interval_s=.2,
                min_scans=2, max_height_spread_m=.2, progress=None, allow_empty=False):
    """Build a new ground candidate from source-bound, consecutive deskewed scans.

    Each scan contributes at most one median per cell. Conflicting heights
    withdraw support; a missing fit never borrows the legacy global-map ground.
    For a partial survey, allow_empty returns empty measured support if needed
    and exposes the reliable direct-sample mask for comparison with older ground.
    """
    if interval_s <= 0 or min_scans < 2 or max_height_spread_m <= 0:
        raise ValueError('Invalid offline scan-ground settings')
    meta=candidate.metadata;shape=tuple(meta['shape']);origin=np.asarray(meta['origin_xy']);res=float(meta['resolution'])
    tracker=GroundTracker(config['ground_tracking']);identity=None;previous_stamp=None
    # Report the archive record count even though only sampled payloads are decoded.
    count, sampled = sampled_scan_stamps(scan_path, interval_s)
    keys=[];heights=[];stats=dict(scans_read=count,scans_sampled=0,scans_unbound=0,tracked_scans=0,
        tracker_resets=0,cell_scan_samples=0)
    for record, data in read_scans(scan_path, stamps=sampled):
        stamp=record['stamp_ns']/1e9
        stats['scans_sampled']+=1
        if progress and stats['scans_sampled']%500==0:
            progress('地面扫描：%d 已采样，%d 已跟踪。'%(stats['scans_sampled'],stats['tracked_scans']))
        item=_scan_geometry(record,data,timeline)
        if item is None:
            stats['scans_unbound']+=1;tracker=GroundTracker(config['ground_tracking']);identity=None;previous_stamp=None
            continue
        _,pose,_,final,cloud,_=item
        current=(pose['session_id'],pose['generation'],pose['segment'])
        if current!=identity or (previous_stamp is not None and stamp-previous_stamp>1.):
            tracker=GroundTracker(config['ground_tracking']);stats['tracker_resets']+=1
        identity=current;previous_stamp=stamp
        _,info=tracker.fit(cloud,final[:3,3],config)
        points=tracker.observed_ground
        if info.get('status')!='tracked' or points is None or not len(points):continue
        stats['tracked_scans']+=1
        cells=np.floor((points[:,:2]-origin)/res).astype(np.int64)
        inside=((cells>=0)&(cells<np.asarray(shape[::-1]))).all(axis=1)
        if not inside.any():continue
        cell,median,_,_=ground_cell_statistics(points[inside],origin,res,shape[1])
        keys.append(cell);heights.append(median);stats['cell_scan_samples']+=len(cell)
    if not keys and not allow_empty:raise ValueError('No tracked scan ground is available for this candidate')
    flat=np.concatenate(keys) if keys else np.empty(0,np.int64)
    values=np.concatenate(heights) if heights else np.empty(0,np.float32)
    order=np.lexsort((values,flat));flat=flat[order];values=values[order]
    cells,start,count=np.unique(flat,return_index=True,return_counts=True)
    median=(values[start+(count-1)//2]+values[start+count//2])/2
    # Percentiles across scan medians reveal repeated incompatible surfaces.
    low=values[start+np.floor((count-1)*.1).astype(int)]
    high=values[start+np.ceil((count-1)*.9).astype(int)]
    reliable=(count>=min_scans)&((high-low)<=max_height_spread_m)
    seed=np.zeros(shape,bool);seed.ravel()[cells[reliable]]=True
    if not seed.any() and not allow_empty:raise ValueError('No scan-ground cells satisfy the evidence requirements')
    ground=np.full(shape,np.nan,np.float32);ground.ravel()[cells[reliable]]=median[reliable]
    support=seed.copy()
    if seed.any():
        distance,nearest=nearest_support(seed)
        support=distance*res<=config['ground_support_radius_m']
        ground=ground[nearest].copy();ground[~support]=np.nan
    undersampled=np.zeros(shape,bool);undersampled.ravel()[cells[count<min_scans]]=True
    conflict=np.zeros(shape,bool)
    conflict.ravel()[cells[(count>=min_scans)&((high-low)>max_height_spread_m)]]=True
    # A sparse observation may use nearby well-supported ground only if it
    # does not contradict that ground; it is never itself promoted to a seed.
    single=(count<min_scans)&np.isfinite(ground.ravel()[cells])
    delta=np.zeros(median.shape,dtype=median.dtype)
    np.subtract(median,ground.ravel()[cells],out=delta,where=single)
    single &= np.abs(delta)>max_height_spread_m
    conflict.ravel()[cells[single]]=True
    # Conflicting observations cannot become supported through a neighbour.
    support[conflict]=False;ground[conflict]=np.nan
    discontinuous=np.zeros(shape,bool)
    for first,second in ((np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])):
        paired=support[first]&support[second]
        difference=np.zeros(paired.shape,dtype=ground.dtype)
        np.subtract(ground[first],ground[second],out=difference,where=paired)
        jump=paired&(np.abs(difference)>max_height_spread_m)
        discontinuous[first]|=jump;discontinuous[second]|=jump
    support[discontinuous]=False;ground[discontinuous]=np.nan
    conflict |= discontinuous
    if not support.any() and not allow_empty:raise ValueError('No scan ground remains after discontinuity checks')
    stats.update(seed_cells=int(seed.sum()),undersampled_cells=int((count<min_scans).sum()),
                 conflicting_cells=int(((count>=min_scans)&((high-low)>max_height_spread_m)).sum()),
                 contradictory_sparse_cells=int(single.sum()),
                 discontinuous_cells=int(discontinuous.sum()),
                 supported_cells=int(support.sum()),interval_s=float(interval_s),min_scans=int(min_scans),
                 max_height_spread_m=float(max_height_spread_m),
                 support_radius_m=float(config['ground_support_radius_m']),
                 support_note='Nearby reliable ground can be propagated within the support radius; it is not a direct measurement at every supported cell')
    signature=hashlib.sha256(encoded(dict(method='tracked_scan_ground',scan_fingerprint=scan_fingerprint,
        original=meta['snapshot_id'],settings=stats)).encode()).hexdigest()[:20]
    updated=dict(meta,snapshot_id=meta['snapshot_id']+':tracked_ground:'+signature,
        ground_estimator='consecutive_scan_tracker_cell_median',ground_scan_diagnostics=stats)
    layers={key:value.copy() for key,value in candidate.layers.items()}
    layers.update(ground_z=ground,ground_support=support,ground_scan_unreliable=conflict,
                  ground_scan_undersampled=undersampled)
    if allow_empty:
        layers['ground_scan_column_observed']=seed & support
    for value in layers.values():value.flags.writeable=False
    return GridSnapshot(updated,layers)


def reproject_endpoints(candidate, points, config):
    """Recompute endpoint classification against the candidate's new ground."""
    layers={key:value.copy() for key,value in candidate.layers.items()}
    layers.update(endpoint_layers(points,layers['ground_z'],np.asarray(candidate.metadata['origin_xy']),config))
    occupancy=layers['occupancy'];res=config['resolution']
    clearance=(distance_to_zero(occupancy!=100)*res if (occupancy==100).any() else np.full(occupancy.shape,np.nan))
    clearance[occupancy==-1]=np.nan;layers['clearance']=clearance.astype(np.float32)
    for value in layers.values():value.flags.writeable=False
    return GridSnapshot(candidate.metadata,layers)
