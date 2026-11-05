"""Read-only ground heights for display; never supplies navigation evidence."""
from collections import OrderedDict

import numpy as np


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
    m,l=raw.metadata,raw.layers;proof=m.get('navigation_fusion')
    if proof is None:
        if 'navigation_occupancy' in l:raise ValueError('Display navigation cells lack provenance')
        return raw
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
    return GridSnapshot(dict(m,display_only=True,display_occupancy='navigation'),layers)


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
        meta,layers=self.grid.metadata,self.grid.layers
        if 'ground_z' in layers and 'ground_support' in layers:
            ij=np.floor((xy-meta['origin_xy'])/meta['resolution']).astype(int);h,w=layers['ground_z'].shape
            inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            indices=np.flatnonzero(inside);x,y=ij[indices].T
            good=layers['ground_support'][y,x]&np.isfinite(layers['ground_z'][y,x])
            take=indices[good];z[take]=layers['ground_z'][y[good],x[good]]+self.offset
            source[take]=layers['display_ground_source'][y[good],x[good]] if 'display_ground_source' in layers else 1
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


def line_segments(points):
    """Never draw a bridge across an unknown height."""
    points=np.asarray(points,float).reshape(-1,3)
    if len(points)<2:return np.empty((0,3))
    valid=np.isfinite(points[:-1]).all(1)&np.isfinite(points[1:]).all(1)
    return np.stack([points[:-1][valid],points[1:][valid]],axis=1).reshape(-1,3)


def ground_cloud(grid,offset=.005):
    """A sparse coloured occupancy surface, without filling unsupported cells."""
    from .scans import pointcloud_message
    from sensor_msgs.msg import PointField
    layers,meta=grid.layers,grid.metadata
    good=layers.get('ground_support',np.zeros_like(layers['occupancy'],bool))&np.isfinite(layers.get('ground_z',np.nan))
    y,x=np.nonzero(good);r=meta['resolution'];origin=meta['origin_xy']
    xyz=np.c_[origin[0]+(x+.5)*r,origin[1]+(y+.5)*r,layers['ground_z'][y,x]+offset] if len(y) else np.empty((0,3))
    message=pointcloud_message(xyz,meta['frame_id'],meta['stamp_s'])
    data=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
    for i,k in enumerate(('x','y','z')):data[k]=xyz[:,i]
    values=layers['occupancy'][y,x]
    data['rgb']=np.where(values==0,0xD7DDE3,np.where(values==100,0xF05848,0xC89F50))
    message.fields.append(PointField(name='rgb',offset=12,datatype=PointField.UINT32,count=1))
    message.point_step=16;message.row_step=len(xyz)*16;message.data=data.tobytes()
    return message
