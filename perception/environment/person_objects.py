"""Frame-local person ownership and conservative circle fallback views.

Owned returns still terminate rays. Only their obstacle evidence is delegated;
background geometry and source measurement times remain unchanged.
"""
import numpy as np
import cv2


def point_owners(points, boxes, config, sensor, plane=None):
    """Bounded XY clustering inside weak boxes; no per-cell ground prerequisite."""
    settings=config['dynamic'];resolution=config['resolution']
    owner=np.full(len(points),-1,np.int32);best=np.full(len(points),np.inf)
    ground=None if plane is None else points[:,:2]@plane[:2]+plane[2]
    for number,box in enumerate(boxes):
        xy=points[:,:2]-box[:2];distance=np.linalg.norm(xy,axis=1)
        c,s=np.cos(box[6]),np.sin(box[6]);local=xy@np.array([[c,-s],[s,c]])
        half=np.clip(box[3:5]/2+settings['padding_m'],resolution,settings['max_radius_m'])
        keep=((abs(local)<=half).all(1)&(distance<=settings['object_radius_limit_m'])&
              (abs(points[:,2]-box[2])<=box[5]/2+settings['padding_m']))
        if ground is None:
            keep&=points[:,2]>box[2]-box[5]/2+config['min_obstacle_height_m']
        else:
            keep&=(points[:,2]-ground>=config['min_obstacle_height_m'])&(points[:,2]-ground<=config['max_height_m'])
        indices=np.flatnonzero(keep)
        if len(indices)<settings['min_cluster_points'] or distance[indices].min()>settings['seed_distance_m']:continue
        # Sparse vertical rings must not require 3D point-to-point continuity.
        step=np.clip(resolution+.012*np.linalg.norm(box[:2]-sensor[:2]),resolution,1.8*resolution)
        cells=np.floor((local[indices]+half)/step).astype(int)
        shape=np.ceil(2*half/step).astype(int)+1
        raster=np.zeros(tuple(shape[::-1]),np.uint8);raster[cells[:,1],cells[:,0]]=1
        _,labels=cv2.connectedComponents(raster,connectivity=8)
        seed=np.argmin(distance[indices]);label=labels[cells[seed,1],cells[seed,0]]
        indices=indices[labels[cells[:,1],cells[:,0]]==label]
        if len(indices)<settings['min_cluster_points']:continue
        selected=indices[distance[indices]<best[indices]]
        owner[selected]=number;best[selected]=distance[selected]
    return owner


def ownership_record(points, owners, boxes, tracks, people, stamp, identity, frame):
    objects=[]
    for index,(box,track) in enumerate(zip(boxes,tracks)):
        selected=owners==index
        if not selected.any():continue
        radius=max(.30,float(np.linalg.norm(points[selected,:2]-box[:2],axis=1).max())+.02)
        objects.append(dict(identity=[people['session_id'],people['generation'],str(track)],
            xy=box[:2].tolist(),radius_m=radius,points=int(selected.sum())))
    return dict(method='tracked_person_circles',stamp_s=float(stamp),
        local_identity=[*identity,frame],people_source=people['input_fingerprint'],objects=objects)


def observed_objects(snapshot):
    """Validate bounded producer geometry even when the tracking packet is absent."""
    m=snapshot.metadata;record=m.get('person_objects')
    if record is None:return []
    if (record.get('method')!='tracked_person_circles' or record.get('stamp_s')!=m['stamp_s'] or
            record.get('local_identity')!=[m['session_id'],m['generation'],m['frame_id']] or
            not isinstance(record.get('people_source'),str)):
        raise ValueError('Person object source mismatch')
    objects=record.get('objects');identities=set()
    if not isinstance(objects,list) or len(objects)>128:raise ValueError('Invalid person object budget')
    for obj in objects:
        key=tuple(obj['identity']);xy=obj['xy'];radius=obj['radius_m']
        if (len(key)!=3 or key in identities or np.shape(xy)!=(2,) or
                not np.isfinite(xy).all() or not np.isfinite(radius) or not 0<radius<=1.):
            raise ValueError('Invalid person object geometry')
        identities.add(key)
    return objects


def circle_view(grid, objects):
    """Whole-cell intersection with currently observed circles; no free-space writes."""
    if not objects:return grid
    from .layers import GridSnapshot
    m=grid.metadata;cells=grid.layers['occupancy'].copy();height,width=cells.shape
    resolution=m['resolution'];origin=np.asarray(m['origin_xy'])
    for obj in objects:
        centre=np.asarray(obj['xy']);radius=obj['radius_m']
        low=np.maximum(0,np.floor((centre-radius-origin)/resolution).astype(int))
        high=np.minimum([width,height],np.ceil((centre+radius-origin)/resolution).astype(int))
        if np.any(high<=low):continue
        y,x=np.indices(tuple((high-low)[::-1]))
        xy=np.stack([x+low[0]+.5,y+low[1]+.5],axis=-1)*resolution+origin
        gap=np.maximum(abs(xy-centre)-resolution/2,0.)
        patch=cells[low[1]:high[1],low[0]:high[0]];patch[(gap**2).sum(-1)<=radius**2]=100
    cells.flags.writeable=False
    return GridSnapshot(dict(m,navigation_view='background_with_current_person_circles'),dict(occupancy=cells))
