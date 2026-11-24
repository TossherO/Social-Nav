"""Scan confirmation and long-lived obstacle evidence on the local grid."""

import numpy as np

from .geometry import shift_array


EXPIRY_LAYER = 'obstacle_expiry_unknown'
OBSTACLE_TIMING_DEFAULTS = dict(obstacle_confirmation_window_s=.5,
    near_obstacle_radius_m=3.5, near_obstacle_evidence_age_s=.5,
    long_obstacle_max_gap_s=.35, long_obstacle_min_span_s=2., long_obstacle_hold_s=2.)


class ObstacleMemory:
    def __init__(self):
        self.origin = None
        self.history = None

    def align(self, origin, shape, resolution):
        if self.history is None:
            self.history = np.full((5,*shape), -np.inf)
            self.ordinary_stamp = np.full(shape,-np.inf)
            self.person_stamp = np.full(shape,-np.inf)
            self.short_only = np.zeros(shape,bool)
        elif not np.array_equal(self.origin,origin):
            dx,dy = np.rint((self.origin-origin)/resolution).astype(int)
            self.history = np.stack([shift_array(layer,dx,dy,-np.inf) for layer in self.history])
            self.ordinary_stamp = shift_array(self.ordinary_stamp,dx,dy,-np.inf)
            self.person_stamp = shift_array(self.person_stamp,dx,dy,-np.inf)
            self.short_only = shift_array(self.short_only,dx,dy,False)
        self.origin = origin

    def unconfirmed(self, y, x, stamp, window):
        # Current scan supplies the third vote. Many points may share one cell.
        return stamp-self.history[1,y,x] > window+1e-6

    def near_short_only(self, y, x, stamp, *, center, resolution, radius, evidence_age, max_gap,
                        ordinary_ttl, previous):
        continuous = stamp-self.history[0,y,x]<=max_gap+1e-6
        result = continuous & self.short_only[y,x]
        if center is None or not len(x):
            return result
        cell_xy = self.origin+(np.column_stack((x,y))+.5)*resolution
        near = np.sum((cell_xy-center)**2,axis=1)<=radius**2
        eligible = (~continuous & near & (stamp-self.ordinary_stamp[y,x]>ordinary_ttl+1e-6)
                    & (stamp-self.history[3,y,x]>1e-6))
        person = eligible & (stamp-self.person_stamp[y,x]<=evidence_age+1e-6)
        result |= person
        candidates = np.flatnonzero(eligible&~person)
        if len(candidates) and previous is not None:
            result[candidates] = recent_measured_free(previous,cell_xy[candidates],stamp,evidence_age)
        return result

    def remember_sources(self, y, x, stamp, *, near_short, unconfirmed, person_cells):
        self.short_only[y,x] = near_short
        ordinary = ~unconfirmed
        self.ordinary_stamp[y[ordinary],x[ordinary]] = stamp
        if person_cells is not None:
            yy,xx = person_cells
            self.person_stamp[yy,xx] = stamp

    def observe(self, y, x, stamp, *, max_gap, min_span, hold):
        latest, previous, start, locked_until, expired_at = self.history
        expired = np.isfinite(locked_until)&(stamp-locked_until>1e-6)
        expired_at[expired] = stamp
        locked_until[expired] = -np.inf
        last = latest[y,x]
        streak = np.where(stamp-last<=max_gap+1e-6,start[y,x],stamp)
        locked = (stamp-streak>=min_span-1e-6)|np.isfinite(locked_until[y,x])
        locked_until[y[locked],x[locked]] = stamp+hold
        previous[y,x],latest[y,x],start[y,x] = last,stamp,streak

    def projection(self):
        latest,_,_,locked_until,expired_at = self.history
        active = np.isfinite(locked_until)
        pending = np.isfinite(expired_at)&~active
        if not (active|pending).any():
            return None
        return active, np.where(active,latest,-np.inf), pending, expired_at.copy(), latest>=expired_at

    def clear(self, fresh_free):
        if fresh_free is not None:
            self.history[4,fresh_free] = -np.inf


def recent_measured_free(snapshot, xy, stamp, max_age):
    """Use the preceding local view's ray evidence without historical fills."""
    meta = snapshot.metadata
    result = np.zeros(len(xy),bool)
    if (not meta.get('rays_used') or stamp-meta['stamp_s']>max_age+1e-6
            or meta.get('ground_tracking',{}).get('history_suspended')):
        return result
    ij = np.floor((xy-meta['origin_xy'])/meta['resolution']).astype(int)
    rows = np.flatnonzero(((ij>=0)&(ij<np.asarray(meta['shape'][::-1]))).all(axis=1))
    if not len(rows):
        return result
    x,y = ij[rows].T
    device_snapshot = hasattr(snapshot,'consume')
    layers = snapshot.consume() if device_snapshot else snapshot.layers
    if device_snapshot:
        import torch
        indices = torch.from_numpy(np.stack((y,x))).to(layers['occupancy'].device)
        y,x = indices.unbind()
    seen = layers['last_observed'][y,x]
    clear = ((layers['occupancy'][y,x]==0) & layers['ground_support'][y,x]
             & (layers['coverage'][y,x]>=meta['free_vertical_coverage_required']-1e-6)
             & (seen>=stamp-max_age-1e-6) & (seen<=meta['stamp_s']))
    for name in ('near_field_free','ground_conflict'):
        if name in layers:
            clear &= ~layers[name][y,x]
    result[rows] = clear.cpu().numpy() if device_snapshot else clear
    return result


def apply_obstacle_memory(layers, projection, required_coverage, *, device=None):
    """Apply occupancy policy and return a mask of newly observed free cells."""
    if device is None:
        active,hit_stamp,pending,expired_at,new_hit = projection
        def fill(name, mask, value):
            np.copyto(layers[name],value,where=mask,casting='unsafe')
        def as_bool(value):
            return value.astype(bool,copy=False)
    else:
        import torch
        active,hit_stamp,pending,expired_at,new_hit = [
            torch.from_numpy(a).pin_memory().to(device,non_blocking=True) for a in projection]
        def fill(name, mask, value):
            layers[name].masked_fill_(mask,value)
        def as_bool(value):
            return value.bool()
    occupancy = layers['occupancy']
    occupied = occupancy==100
    fresh_free = (pending & (occupancy==0) & (layers['coverage']>=required_coverage-1e-6)
                  & as_bool(layers['ground_support']) & (layers['last_observed']>expired_at))
    if 'near_field_free' in layers:
        fresh_free &= ~as_bool(layers['near_field_free'])
    unknown = pending & ~fresh_free & ~(occupied & new_hit)
    changed = active|unknown
    fill('occupancy',active,100)
    fill('occupancy',unknown,-1)
    fill('observed',unknown,False)
    fill('observed',active,True)
    if device is None:
        np.copyto(layers['last_observed'],hit_stamp,where=changed)
    else:
        torch.where(changed,hit_stamp,layers['last_observed'],out=layers['last_observed'])
    fill('coverage',changed,0)
    for name in ('obstacle_min_z','obstacle_max_z'):
        fill(name,unknown|(active&~occupied),float('nan'))
    for name in ('near_field_free','near_field_obstacle','person_background_free'):
        if name in layers:
            fill(name,changed,False)
    for name in ('near_field_free_classified_at','near_field_obstacle_stamp','near_field_obstacle_expires_at'):
        if name in layers:
            fill(name,changed,float('nan'))
    layers[EXPIRY_LAYER] = unknown
    return fresh_free
