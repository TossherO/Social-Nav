"""Scan confirmation and long-lived obstacle evidence on the local grid."""

import numpy as np

from .geometry import shift_array


EXPIRY_LAYER = 'obstacle_expiry_unknown'
_OBSTACLE_TILE_CELLS = 32
OBSTACLE_TIMING_DEFAULTS = dict(obstacle_confirmation_window_s=.5,
    long_obstacle_max_gap_s=.35, long_obstacle_min_span_s=2., long_obstacle_hold_s=2.)


class ObstacleMemory:
    def __init__(self):
        self.origin = None
        self.history = None
        self.outside = {}

    def align(self, origin, shape, resolution):
        if self.history is None:
            self.history = np.full((5,*shape), -np.inf)
        elif not np.array_equal(self.origin,origin):
            dx,dy = np.rint((self.origin-origin)/resolution).astype(int)
            # Only long-lived obstacles need to survive eviction of the rolling grid.
            old_x,old_y = np.rint(self.origin/resolution).astype(int)
            height,width = shape
            tile_cells = _OBSTACLE_TILE_CELLS
            for y,x in zip(*np.nonzero(np.isfinite(self.history[3])|np.isfinite(self.history[4]))):
                if not (0<=x+dx<width and 0<=y+dy<height):
                    world_x,world_y = old_x+x,old_y+y
                    tile = self.outside.setdefault((world_x//tile_cells,world_y//tile_cells),{})
                    tile[world_x,world_y] = self.history[:,y,x].copy()
            self.history = np.stack([shift_array(layer,dx,dy,-np.inf) for layer in self.history])
            new_x,new_y = old_x-dx,old_y-dy
            for ty in range(new_y//tile_cells,(new_y+height-1)//tile_cells+1):
                for tx in range(new_x//tile_cells,(new_x+width-1)//tile_cells+1):
                    tile = self.outside.get((tx,ty))
                    if tile is None:
                        continue
                    returning = [(x,y) for x,y in tile if 0<=x-new_x<width and 0<=y-new_y<height]
                    for x,y in returning:
                        self.history[:,y-new_y,x-new_x] = tile.pop((x,y))
                    if not tile:
                        del self.outside[tx,ty]
        self.origin = origin

    def unconfirmed(self, y, x, stamp, window):
        # Current scan supplies the third vote. Many points may share one cell.
        return stamp-self.history[1,y,x] > window+1e-6

    def clear_observed(self, tree, floor, support, origin, config, stamp, *, confirmation_window_s, metrics=None):
        latest,_,_,locked_until,cutoff = self.history
        # Only recent, currently unhit ordinary columns can be cleared early.
        # Established long-lived obstacles keep their separate expiry policy.
        age = stamp-latest
        candidates = (age>0)&(age<=max(config['obstacle_ttl_s'],confirmation_window_s))
        candidates &= ~np.isfinite(locked_until)&~np.isfinite(cutoff)
        cleared = tree.free_ray_columns(candidates,latest,floor,support,origin,config,
                                       metrics=metrics,clear_hits=True)
        self.history[:3,cleared] = -np.inf
        return cleared

    def observe(self, y, x, stamp, *, max_gap, min_span, hold):
        latest, previous, start, locked_until, cutoff = self.history
        expired = np.isfinite(locked_until)&(stamp>locked_until)
        cutoff[expired] = locked_until[expired]
        locked_until[expired] = -np.inf
        last = latest[y,x]
        streak = np.where(stamp-last<=max_gap+1e-6,start[y,x],stamp)
        locked = (stamp-streak>=min_span-1e-6)|np.isfinite(locked_until[y,x])
        locked_until[y[locked],x[locked]] = stamp+hold
        cutoff[y[locked],x[locked]] = -np.inf
        previous[y,x],latest[y,x],start[y,x] = last,stamp,streak

    def projection(self):
        latest,_,_,locked_until,cutoff = self.history
        active = np.isfinite(locked_until)
        pending = np.isfinite(cutoff)
        if not (active|pending).any():
            return None
        return active, np.where(active,latest,-np.inf), pending, cutoff, latest>cutoff

    def clear(self, fresh_free):
        if fresh_free is not None:
            self.history[4,fresh_free] = -np.inf


def apply_obstacle_memory(layers, projection, free_ray_columns, *, device=None):
    """Apply occupancy policy and return a mask of newly observed free cells."""
    if device is None:
        active,hit_stamp,pending,_,new_hit = projection
        def fill(name, mask, value):
            np.copyto(layers[name],value,where=mask,casting='unsafe')
        def as_bool(value):
            return value.astype(bool,copy=False)
    else:
        import torch
        active,hit_stamp,pending,new_hit,free_ray_columns = [
            torch.from_numpy(a).pin_memory().to(device,non_blocking=True)
            for a in (*projection[:3],projection[4],free_ray_columns)]
        def fill(name, mask, value):
            layers[name].masked_fill_(mask,value)
        def as_bool(value):
            return value.bool()
    occupancy = layers['occupancy']
    occupied = occupancy==100
    fresh_free = pending & free_ray_columns & (occupancy==0) & as_bool(layers['ground_support'])
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
