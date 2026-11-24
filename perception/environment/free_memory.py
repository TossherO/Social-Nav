"""Remember directly observed free columns on the rolling local grid lattice."""
import numpy as np

from .layers import GridSnapshot
from .obstacle_memory import EXPIRY_LAYER


LAYER = 'confirmed_free_stamp'
TILE_CELLS = 128


def identity(metadata, asset):
    association = metadata['association']
    return [metadata['session_id'], metadata['generation'], metadata['frame_id'],
            association['map_id'], association['map_frame'], association.get('association_session'),
            association.get('association_generation'), association.get('map_revision'),
            asset.manifest['snapshot_id'], asset.manifest['layers_sha256']]


class ConfirmedFreeMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = self.origin = self.resolution = self.last = None
        self.tiles = {}

    def update(self, snapshot, asset):
        from .person_objects import circle_view, observed_objects

        m, layers = snapshot.metadata, snapshot.layers
        proof = m.get('navigation_fusion')
        if not m['association'].get('valid'):
            self.reset()
            return snapshot
        r = m['resolution']
        origin = np.asarray(m['origin_xy'])
        stamp = m['stamp_s']
        key = identity(m, asset)
        if key != self.key or r != self.resolution or (self.last is not None and stamp < self.last):
            self.reset()
            self.key, self.origin, self.resolution = key, origin, r
        self.last = stamp
        delta = (origin-self.origin)/r
        if not np.allclose(delta,np.rint(delta),atol=1e-6,rtol=0.):
            self.reset()
            return snapshot
        x0,y0 = np.rint(delta).astype(int)
        h,w = layers['occupancy'].shape
        yy,xx = np.indices((h,w))
        points = np.stack((xx+.5,yy+.5),axis=-1)*r+origin
        transform = np.asarray(m['association']['map_from_local'])
        prior = asset.snapshot.metadata
        # Limit the ledger to the selected static map's raster, including its
        # unknown and occupied cells. Keep the remembered grid in odom so the
        # local raster can scroll without resampling earlier observations.
        sensor_pose = np.asarray(m['world_from_lidar'])
        xy = (points-sensor_pose[:2,3])@transform[:2,:2].T+(transform@sensor_pose)[:2,3]
        low = np.asarray(prior['origin_xy'])
        high = low+np.asarray(prior['shape'][::-1])*prior['resolution']
        in_map = ((xy>=low)&(xy<high)).all(axis=-1)
        # Raw hits still invalidate earlier free observations if support is unavailable.
        observed = layers['last_observed'] if proof else layers.get('last_observed', stamp)
        nav = layers['navigation_occupancy'] if proof else layers['occupancy']
        hits_now = nav==100
        hit_source = layers.get('navigation_source_stamp',observed) if proof else observed
        hit_time = np.where(hits_now,np.where(np.isfinite(hit_source)&(hit_source<=stamp),hit_source,stamp),-np.inf)
        people = observed_objects(snapshot)
        if people:
            person_hits = circle_view(GridSnapshot(m,dict(occupancy=np.zeros_like(nav))),people).layers['occupancy']==100
            hit_time[person_hits] = m['person_objects']['stamp_s']
        conflicts = layers.get('ground_conflict')
        if conflicts is not None:
            hit_time[conflicts] = stamp
        if EXPIRY_LAYER in layers:
            hit_time[layers[EXPIRY_LAYER]] = stamp
        occupied = np.isfinite(hit_time)
        free = np.zeros((h,w),bool)
        if proof:
            free = ((layers['occupancy']==0)&(nav==0)&~occupied&
                    layers['ground_support']&(layers['coverage']>=1.-1e-6)&
                    np.isfinite(observed)&(observed<=stamp)&in_map)
            if 'near_field_free' in layers:
                free &= ~layers['near_field_free']
        remembered = np.full((h,w),-np.inf) if proof else None
        for ty in range(y0//TILE_CELLS,(y0+h-1)//TILE_CELLS+1):
            for tx in range(x0//TILE_CELLS,(x0+w-1)//TILE_CELLS+1):
                left,top = tx*TILE_CELLS,ty*TILE_CELLS
                x,y = max(x0,left),max(y0,top)
                end_x,end_y = min(x0+w,left+TILE_CELLS),min(y0+h,top+TILE_CELLS)
                view = np.s_[y-y0:end_y-y0,x-x0:end_x-x0]
                part = np.s_[y-top:end_y-top,x-left:end_x-left]
                tile = self.tiles.get((tx,ty))
                if tile is None:
                    if not (free[view]|(occupied[view]&in_map[view])).any():
                        continue
                    tile = self.tiles[tx,ty] = np.full((2,TILE_CELLS,TILE_CELLS),-np.inf)
                dates,hits = tile
                confirmed,blocked = dates[part],hits[part]
                occupied_here = occupied[view]
                blocked[occupied_here] = np.maximum(blocked[occupied_here],hit_time[view][occupied_here])
                confirmed[occupied_here] = -np.inf
                if proof:
                    accept = free[view]&(observed[view]>blocked)
                    confirmed[accept] = np.maximum(confirmed[accept],observed[view][accept])
                    remembered[view] = confirmed
        if not proof:
            return snapshot
        remembered[~in_map|occupied] = -np.inf
        remembered.flags.writeable = False
        return GridSnapshot(dict(m,confirmed_free_memory=dict(identity=key,as_of=stamp)),
                            dict(layers,**{LAYER:remembered}))


def remembered_free(scene, *, asset=None, now=None):
    """Return causal remembered fields for planning and decision consumers."""
    snapshot = scene.geometry
    record = snapshot.metadata.get('confirmed_free_memory')
    if record is None:
        return None
    stamp = scene.record['as_of']
    association = scene.record['prior']
    if not association.get('valid'):
        return None
    key = record['identity']
    measured = association.get('measurement_stamp_s')
    if (key[:8]!=[scene.record['session_id'],scene.record['generation'],scene.record['frame_id'],
                  association['map_id'],association['map_frame'],association.get('association_session'),
                  association.get('association_generation'),association.get('map_revision')] or
            record['as_of']!=stamp or measured is None or
            not 0<=(stamp if now is None else now)-measured<=snapshot.metadata['navigation_fusion']['max_association_age_s']):
        return None
    if asset is not None and key[8:]!=[asset.manifest['snapshot_id'],asset.manifest['layers_sha256']]:
        return None
    values = snapshot.layers[LAYER]
    if (values.dtype!=np.float64 or values.shape!=snapshot.layers['occupancy'].shape or
            np.isnan(values).any() or np.any(values>stamp)):
        raise ValueError('Invalid confirmed-free evidence')
    return values
