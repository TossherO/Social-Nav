"""Read-only ground heights for display; never supplies navigation evidence."""
from collections import OrderedDict

import numpy as np

from navigation.planning.inputs import navigation_grid as static_navigation_grid


FREE_COLOR = 0x9EA9B4
OCCUPIED_COLOR = 0xCF5D53
CONDITIONAL_COLOR = 0xCB974A
UNKNOWN_COLOR = 0x707885
ESTIMATED_COLOR = 0x72A0B7
LOCAL_FREE_COLOR = 0x72C8C3
LOCAL_OCCUPIED_COLOR = 0xF47B63


class DisplayGrids:
    """Download selected display layers once per snapshot in the display worker."""
    def __init__(self,metrics=None,*,asset=None):self.cache=OrderedDict();self.metrics=metrics;self.asset=asset

    def get(self,source,*,navigation=True):
        if source is None:return None
        if not hasattr(source,'materialize'):return navigation_grid(source,self.asset) if navigation else source
        key=getattr(source,'owner',source.metadata['snapshot_id'])
        if key not in self.cache:
            fields=[k for k in ('occupancy','ground_z','ground_support','navigation_occupancy','navigation_source') if k in source._layers]
            raw=source.materialize(consumer='optional_ros_display',fields=fields,metrics=self.metrics)
            self.cache[key]=(raw,navigation_grid(raw,self.asset))
            if len(self.cache)>2:self.cache.popitem(last=False)
        return self.cache[key][1 if navigation else 0]


def navigation_grid(raw,asset=None):
    """Display the producer's navigation cells; supplementary heights stay display-only."""
    from perception.environment.layers import GridSnapshot
    from perception.environment.navigation import association_reason,prior_floor
    from perception.environment.support import METHOD
    from perception.environment.person_objects import observed_objects,circle_view
    m,l=raw.metadata,raw.layers;proof=m.get('navigation_fusion')
    if proof is None:
        if 'navigation_occupancy' in l:raise ValueError('Display navigation cells lack provenance')
        objects=observed_objects(raw)
        if not objects:return raw
        cells=circle_view(raw,objects).layers['occupancy']
        return GridSnapshot(dict(m,display_only=True,display_occupancy='navigation'),dict(l,occupancy=cells))
    if proof.get('method')!=METHOD:raise ValueError('Unsupported display navigation derivation')
    cells=l['navigation_occupancy'];layers=dict(l,occupancy=cells)
    ground=l.get('ground_z',np.full(cells.shape,np.nan))
    supported=l.get('ground_support',np.zeros(cells.shape,bool))&np.isfinite(ground)
    sources=supported.astype(np.uint8)
    missing=~supported&np.isin(l['navigation_source'],[3,10])
    if (missing.any() and asset is not None and
            proof['static_map']=={k:asset.manifest[k] for k in ('snapshot_id','layers_sha256')} and
            association_reason(raw,asset,proof['max_association_age_s']) is None):
        floor=prior_floor(raw,asset.snapshot,np.asarray(m['association']['map_from_local']),
            proof['floor_reference']['map_plane'],selected=missing)
        take=missing&np.isfinite(floor)
        if take.any():
            ground=ground.copy();ground[take]=floor[take]
            supported=supported|take;sources[take]=4
    if ground is not l.get('ground_z'):ground.flags.writeable=False
    supported.flags.writeable=False;sources.flags.writeable=False
    layers.update(ground_z=ground,ground_support=supported,display_ground_source=sources)
    layers['occupancy']=circle_view(GridSnapshot(m,layers),observed_objects(raw)).layers['occupancy']
    return GridSnapshot(dict(m,display_only=True,display_occupancy='navigation'),layers)


def sample_ground(grid,xy,offset=0.):
    """Sample measured or explicitly fitted display heights in the grid frame."""
    xy=np.asarray(xy,float).reshape(-1,2);z=np.full(len(xy),np.nan);source=np.zeros(len(xy),np.uint8)
    meta,layers=grid.metadata,grid.layers
    if 'ground_z' not in layers or 'ground_support' not in layers:return z,source
    ij=np.floor((xy-meta['origin_xy'])/meta['resolution']).astype(int);h,w=layers['ground_z'].shape
    inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    indices=np.flatnonzero(inside);x,y=ij[indices].T
    good=layers['ground_support'][y,x]&np.isfinite(layers['ground_z'][y,x])
    take=indices[good];z[take]=layers['ground_z'][y[good],x[good]]+offset
    source[take]=layers['display_ground_source'][y[good],x[good]] if 'display_ground_source' in layers else 1
    fitted=layers.get('ground_estimated')
    if fitted is not None:
        estimate=(~good)&fitted[y,x]&np.isfinite(layers['ground_z'][y,x])
        take=indices[estimate];z[take]=layers['ground_z'][y[estimate],x[estimate]]+offset
        source[take]=5
    return z,source


def annotation_height(grid,xy,fallback):
    """A label anchor, never evidence for a terrain cell or a navigable path."""
    z=sample_ground(grid,[xy])[0][0]
    return float(z) if np.isfinite(z) else float(fallback)


class DisplaySurface:
    def __init__(self,grid,row,*,offset=.03,mode='ground',plane_radius=1.5,max_age=.5):
        self.grid=None;self.offset=offset;self.mode=mode;self.plane_radius=plane_radius
        self.resolution=.1
        if grid is not None:
            meta=grid.metadata
            if (all(meta.get(k)==row[k] for k in ('session_id','generation','frame_id')) and
                    -1e-6<=row['stamp_s']-meta['stamp_s']<=max_age):
                self.grid=grid;self.resolution=meta['resolution']

    def sample(self,xy):
        """Return Z and source (0 unknown, 1 local, 2 near fit, 3 plane, 4 static)."""
        xy=np.asarray(xy,float).reshape(-1,2);z=np.full(len(xy),np.nan);source=np.zeros(len(xy),np.uint8)
        if self.mode=='plane':return np.full(len(xy),self.offset),np.full(len(xy),3,np.uint8)
        if self.grid is None:return z,source
        meta=self.grid.metadata
        z,source=sample_ground(self.grid,xy,self.offset)
        # The sensor blind spot can lack a cell. Only extend the current measured
        # floor fit close to the sensor, explicitly marked as a fitted height.
        ground=meta.get('ground_tracking',{});plane=np.asarray(ground.get('plane_world',[]),float)
        if ground.get('status')=='tracked' and plane.shape==(3,) and np.isfinite(plane).all() and self.plane_radius>0:
            center=np.asarray(meta['world_from_lidar'])[:2,3]
            near=(source==0)&(np.linalg.norm(xy-center,axis=1)<=self.plane_radius)
            z[near]=xy[near]@plane[:2]+plane[2]+self.offset;source[near]=2
        return z,source


def sample_poses(poses,spacing):
    """Densify sparse spatial references before looking up terrain cells."""
    poses=np.asarray(poses,float).reshape(-1,3)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1))]
    if len(poses)<2 or arc[-1]<1e-9:return poses.copy()
    at=np.linspace(0.,arc[-1],min(2000,int(np.ceil(arc[-1]/spacing))+1))
    return np.c_[np.interp(at,arc,poses[:,0]),np.interp(at,arc,poses[:,1]),np.interp(at,arc,np.unwrap(poses[:,2]))]


def complete_display_heights(z, fallback):
    """Fill display-only path gaps; no height here is navigation evidence."""
    z=np.asarray(z,float).copy()
    missing=~np.isfinite(z)
    known=np.flatnonzero(~missing)
    if known.size:
        z[missing]=np.interp(np.flatnonzero(missing),known,z[known])
    else:
        z[:]=float(fallback)
    return z,missing


def line_segments(points):
    """Never draw a bridge across an unknown height."""
    points=np.asarray(points,float).reshape(-1,3)
    if len(points)<2:return np.empty((0,3))
    valid=np.isfinite(points[:-1]).all(1)&np.isfinite(points[1:]).all(1)
    return np.stack([points[:-1][valid],points[1:][valid]],axis=1).reshape(-1,3)


class SearchMapSurface:
    """Draw a reference search's cells at display heights from the static asset."""

    def __init__(self, asset, offset=.005):
        from perception.environment.geometry import nearest_support

        grid = static_navigation_grid(asset)
        meta, layers = grid.metadata, grid.layers
        self.shape = layers['occupancy'].shape
        self.frame_id = meta['frame_id']
        self.resolution = meta['resolution']
        self.origin_xy = tuple(meta['origin_xy'])
        rows, columns = self.shape
        self.x = (self.origin_xy[0] + (np.arange(columns) + .5) * self.resolution).astype(np.float32)
        self.y = (self.origin_xy[1] + (np.arange(rows) + .5) * self.resolution).astype(np.float32)

        # A horizontal plane at the mapped median (or Z=0 without any ground)
        # keeps known search cells visible outside measured ground coverage.
        ground_z = layers.get('ground_z')
        height = np.zeros(self.shape, dtype=np.float32)
        if ground_z is not None:
            known = np.zeros(self.shape, dtype=bool)
            for layer in ('ground_support', 'ground_estimated'):
                if layer in layers:
                    known |= layers[layer]
            known &= np.isfinite(ground_z)
            if known.any():
                measured = ground_z[known]
                height.fill(np.median(measured))
                height[known] = measured
                if not known.all():
                    distance, nearest = nearest_support(known)
                    nearby = ~known & (distance * self.resolution <= 1.)
                    height[nearby] = ground_z[nearest[0][nearby], nearest[1][nearby]]
        self.offset = offset
        self.height = height.ravel()

    def path_points(self,xy,offset):
        """Lift a path already in the search frame onto the displayed terrain."""
        x,y=np.floor((xy-self.origin_xy)/self.resolution).astype(int).T
        valid=(x>=0)&(x<self.shape[1])&(y>=0)&(y<self.shape[0])
        z=np.full(len(x),np.nan)
        z[valid]=self.height[y[valid]*self.shape[1]+x[valid]]+offset
        return np.column_stack((xy,z))

    def cloud(self, snapshot):
        from navigation.planning.runtime import decode_display_map

        if (tuple(snapshot['shape']) != self.shape or snapshot['frame_id'] != self.frame_id or
                snapshot['resolution'] != self.resolution or tuple(snapshot['origin_xy']) != self.origin_xy):
            raise ValueError('Search display grid geometry differs from the static map')

        cells, conditional = decode_display_map(snapshot)
        cells = cells.ravel()
        indices = np.flatnonzero(cells >= 0)
        rows, columns = np.divmod(indices, self.shape[1])
        rgb = np.full(indices.size, FREE_COLOR, dtype=np.uint32)
        if conditional is not None:
            rgb[conditional.ravel()[indices]] = CONDITIONAL_COLOR
        rgb[cells[indices] == 100] = OCCUPIED_COLOR
        return coloured_cloud(self.x[columns], self.y[rows], self.height[indices] + self.offset, rgb,
                              self.frame_id, snapshot['stamp_s'])


def coloured_cloud(x, y, z, rgb, frame, stamp):
    from .scans import pointcloud_message
    from sensor_msgs.msg import PointField

    message = pointcloud_message([], frame, stamp)
    data = np.empty(len(x), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    data['x'], data['y'], data['z'], data['rgb'] = x, y, z, rgb
    message.fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
    message.width = len(x)
    message.point_step = data.dtype.itemsize
    message.row_step = message.width * message.point_step
    message.data = data.tobytes()
    return message


def _ground_layers(grid):
    layers = grid.layers
    shape = layers['occupancy'].shape
    ground = layers['ground_z'] if 'ground_z' in layers else np.full(shape, np.nan, np.float32)
    support = layers['ground_support'] if 'ground_support' in layers else np.zeros(shape, bool)
    fitted = layers['ground_estimated'] if 'ground_estimated' in layers else np.zeros(shape, bool)
    return ground, support, fitted


def _ground_plane(grid, ground, good):
    tracking = grid.metadata.get('ground_tracking', {})
    plane = np.asarray(tracking.get('plane_world', []))
    if tracking.get('status') == 'tracked' and plane.shape == (3,) and np.isfinite(plane).all():
        return plane
    return np.array([0., 0., np.median(ground[good]) if good.any() else 0.])


def _ground_points(grid, *, fill_known, local_overlay=False):
    cells, meta = grid.layers['occupancy'], grid.metadata
    ground, support, fitted = _ground_layers(grid)
    good = (support | fitted) & np.isfinite(ground)
    if local_overlay:
        visible = cells >= 0
    elif fill_known:
        visible = good | (cells >= 0)
    else:
        visible = good
    y, x = np.nonzero(visible)
    resolution = meta['resolution']
    origin_x, origin_y = meta['origin_xy']
    xy = np.column_stack((origin_x + (x + .5) * resolution, origin_y + (y + .5) * resolution))
    height = ground[y, x]
    values = cells[y, x]
    free, occupied = (LOCAL_FREE_COLOR, LOCAL_OCCUPIED_COLOR) if local_overlay else (FREE_COLOR, OCCUPIED_COLOR)
    rgb = np.where(values == 0, free, np.where(values == 100, occupied, UNKNOWN_COLOR))
    if not local_overlay:
        rgb[fitted[y, x] & ~support[y, x] & (values != 100)] = ESTIMATED_COLOR
    missing = ~good[y, x]
    if missing.any():
        plane = _ground_plane(grid, ground, good)
        height[missing] = xy[missing] @ plane[:2] + plane[2]
    return np.column_stack((xy, height)), rgb


def ground_cloud(grid, offset=.005, *, fill_known=False):
    """Draw terrain heights, optionally showing every known navigation occupancy cell."""
    points, rgb = _ground_points(grid, fill_known=fill_known)
    x, y, z = points.T
    return coloured_cloud(x, y, z + offset, rgb, grid.metadata['frame_id'], grid.metadata['stamp_s'])


class NavigationGridSurface:
    """Render the current window using its own transform and display heights."""

    def __init__(self, asset):
        self.static = None if asset is None else static_navigation_grid(asset)
        if self.static is not None:
            self.ground, self.support, self.fitted = _ground_layers(self.static)
            self.plane = _ground_plane(self.static, self.ground,
                (self.support | self.fitted) & np.isfinite(self.ground))

    def _project(self, points, transform, offset):
        if transform is not None:
            points = points @ transform[:3, :3].T + transform[:3, 3]
            meta = self.static.metadata
            ij = np.floor((points[:, :2] - meta['origin_xy']) / meta['resolution']).astype(int)
            rows, columns = self.ground.shape
            within = ((ij >= 0) & (ij < (columns, rows))).all(axis=1)
            indices = np.flatnonzero(within)
            x, y = ij[indices].T
            static_z = self.ground[y, x]
            supported = (self.support[y, x] | self.fitted[y, x]) & np.isfinite(static_z)
            visible = supported | (self.static.layers['occupancy'][y, x] >= 0)
            fallback = points[indices, :2] @ self.plane[:2] + self.plane[2]
            static_z = np.where(supported, static_z, fallback)
            # Keep the overlay above static cells when local display heights
            # fall below their background. XY and planner inputs are unchanged.
            points[indices[visible], 2] = np.maximum(points[indices[visible], 2], static_z[visible])
        points[:, 2] += offset
        return points

    def layer(self, grid, transform):
        meta = grid.metadata
        frame = meta['frame_id'] if transform is None else self.static.metadata['frame_id']
        points, rgb = _ground_points(grid, fill_known=True, local_overlay=True)
        x, y, z = self._project(points, transform, .015).T
        cloud = coloured_cloud(x, y, z, rgb, frame, meta['stamp_s'])
        height, width = grid.layers['occupancy'].shape
        xy = np.asarray(meta['origin_xy']) + meta['resolution'] * np.array(
            [(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
        ground, support, fitted = _ground_layers(grid)
        plane = _ground_plane(grid, ground, (support | fitted) & np.isfinite(ground))
        boundary = np.column_stack((xy, xy @ plane[:2] + plane[2]))
        return cloud, self._project(boundary, transform, .065)
