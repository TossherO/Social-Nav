"""Observed floor candidates with spatial support and bounded reference recovery."""
from collections import deque
import numpy as np

from .geometry import ground_plane, fit_ground_plane


def spatial_floor_points(points,sensor,config,settings):
    """Balance XY support; wall columns cannot outvote a sparse visible floor."""
    delta=points-sensor
    points=points[(np.linalg.norm(delta[:,:2],axis=1)<config['ground_fit_radius_m']) &
        (delta[:,2]<-config['ground_min_below_sensor_m']) &
        (delta[:,2]>-config['ground_max_below_sensor_m'])]
    if not len(points):return points,points
    cells=np.floor((points[:,:2]-sensor[:2])/settings['support_cell_m']).astype(np.int32)
    # A scalar key avoids row-wise sorting and preserves signed XY lexicographic order.
    ordered=(cells.astype(np.int64)+(1<<31)).astype(np.uint64)
    keys=(ordered[:,0]<<np.uint64(32))|ordered[:,1]
    _,inverse=np.unique(keys,return_inverse=True)
    count=int(inverse.max())+1;low=np.full(count,np.inf);high=np.full(count,-np.inf)
    np.minimum.at(low,inverse,points[:,2]);np.maximum.at(high,inverse,points[:,2])
    valid=high-low<=settings['max_cell_vertical_span_m']
    first=np.full(count,len(points),np.int64)
    indices=np.flatnonzero(points[:,2]==low[inverse])
    np.minimum.at(first,inverse[indices],indices)
    return points[valid[inverse]],points[first[valid]]


def floor_candidate(plane,support,seeds,low,sensor,config,settings,coordinates):
    if plane is None:return None
    support_xy,seeds_xy,low_xy,sensor_xy=coordinates
    tolerance=config['ground_tolerance_m']
    residual=np.abs(seeds_xy@plane-seeds[:,2])
    inliers=seeds[residual<=tolerance]
    raw_error=np.abs(support_xy@plane-support[:,2])
    raw_count=int((raw_error<=tolerance).sum())
    if len(inliers)<3 or raw_count<settings['min_inliers']:return None
    xy=inliers[:,:2]-sensor[:2];covariance=np.cov(xy,rowvar=False,bias=True)
    if np.linalg.eigvalsh(covariance).min()<(settings['min_span_m']/4)**2:return None
    if np.ptp(xy,axis=0).min()<settings['min_span_m']:return None
    # Correlated scan rings are not N independent height measurements.
    leverage=float(np.sqrt(1+xy.mean(0)@np.linalg.solve(covariance,xy.mean(0))))
    error=max(tolerance/3,float(np.median(residual[residual<=tolerance])))
    uncertainty=leverage*error
    height=float(sensor[2]-sensor_xy@plane)
    if not config['ground_min_below_sensor_m']<=height<=config['ground_max_below_sensor_m']:return None
    low_error=low_xy@plane-low[:,2]
    loss=float(np.mean(np.abs(low_error))+np.mean(np.maximum(low_error-tolerance,0)))
    return dict(plane=plane,height=height,loss=loss,inlier_points=raw_count,
        support_cells=len(inliers),height_uncertainty_m=uncertainty)


class GroundTracker:
    DEFAULTS=dict(support_cell_m=.25,max_cell_vertical_span_m=.30,
                  max_extrapolation_error_m=.15,reacquire_observations=5)

    def __init__(self, settings):
        self.settings=dict(self.DEFAULTS,**settings)
        self.heights=deque(maxlen=max(25,settings.get('min_observations',1)))
        self.pending=deque(maxlen=self.settings['reacquire_observations'])
        self.reference=None;self.slope=np.zeros(2)
        self.history_suspended=False

    def fit(self, points, sensor, config):
        settings=self.settings
        if not settings.get('enabled',True):
            return ground_plane(points,sensor,config),dict(status='disabled')
        info=dict(status='unavailable',source='causal_measured_floor',reference_height_m=self.reference,
                  no_missing_cells_filled=True,clear_ground_history=False,
                  history_suspended=self.history_suspended)
        support,seeds=spatial_floor_points(points,sensor,config,settings)
        if len(seeds)<config['ground_min_points']:
            self.pending.clear()
            if self.reference is None:self.heights.clear()
            return None,dict(info,reason='insufficient_spatial_ground_support')
        tolerance=config['ground_tolerance_m']
        quantiles=np.quantile(seeds[:,2],[.05,.2])+tolerance
        low_indices=np.flatnonzero(seeds[:,2]<=quantiles[0]);low=seeds[low_indices]
        groups=[('spatial_low',low_indices),('spatial_lower_envelope',np.flatnonzero(
            seeds[:,2]<=quantiles[1])),('spatial_full',np.arange(len(seeds)))]
        if self.reference is not None:
            expected=(seeds[:,:2]-sensor[:2])@self.slope+sensor[2]-self.reference
            groups.append(('height_band',np.flatnonzero(abs(seeds[:,2]-expected)<=settings['search_band_m'])))
        coordinates=tuple(np.c_[p[:,:2],np.ones(len(p))] for p in (support,seeds,low))+(np.r_[sensor[:2],1.],)
        candidates=[];evaluated={}
        for name,indices in groups:
            key=indices.tobytes()
            if key not in evaluated:
                evaluated[key]=floor_candidate(fit_ground_plane(seeds[indices],config),
                    support,seeds,low,sensor,config,settings,coordinates)
            candidate=evaluated[key]
            if candidate is not None:candidates.append(dict(candidate,name=name))
        if not candidates:
            self.pending.clear()
            if self.reference is None:self.heights.clear()
            return None,dict(info,reason='no_ground_with_bounded_extrapolation')
        chosen=min(candidates,key=lambda c:c['loss']);height=chosen['height'];plane=chosen['plane']
        info.update(measured_height_m=height,inlier_points=chosen['inlier_points'],
            support_cells=chosen['support_cells'],height_uncertainty_m=chosen['height_uncertainty_m'],
            low_support_error_m=chosen['loss'],fit_source=chosen['name'])
        # An uncertain low floor must withdraw support, not lose to a more
        # observable but higher wall-contaminated hypothesis.
        if (chosen['height_uncertainty_m']>settings['max_extrapolation_error_m'] or
                chosen['loss']>tolerance/3):
            self.pending.clear()
            if self.reference is None:self.heights.clear()
            return None,dict(info,reason='ground_fit_or_extrapolation_uncertain')
        changed=False
        if self.reference is not None and abs(height-self.reference)>settings['height_tolerance_m']:
            # A table above a known floor cannot become ground by waiting.
            # Large upward changes require connected terrain or explicit reset.
            if height<self.reference:
                self.pending.clear()
                return None,dict(info,reason='raised_surface_without_ground_connection')
            self.pending.append(height)
            # Withdraw use immediately, but commit destructive changes only after
            # confirmation. Missing/uncertain scans cannot release this suspension.
            self.history_suspended=True
            info.update(reason='confirming_lower_ground',history_suspended=True,
                        reacquire_observations=len(self.pending))
            if (len(self.pending)<settings['reacquire_observations'] or
                    np.ptp(self.pending)>2*settings['max_height_mad_m']):return None,info
            self.heights.clear();self.reference=float(np.median(self.pending));self.pending.clear();changed=True
        else:self.pending.clear()
        self.heights.append(height)
        median=float(np.median(self.heights));mad=float(np.median(abs(np.asarray(self.heights)-median)))
        if self.reference is None:
            if (len(self.heights)<settings['min_observations'] or mad>settings['max_height_mad_m'] or
                    np.ptp(self.heights)>2*settings['max_height_mad_m']):
                return None,dict(info,status='warming')
        self.reference=median;self.slope=plane[:2].copy()
        self.history_suspended=False
        info.pop('reason',None)
        info.update(status='tracked',reference_height_m=self.reference,plane_world=plane.tolist(),
                    reference_changed=changed,clear_ground_history=changed,history_suspended=False)
        return plane,info
