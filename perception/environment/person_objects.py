"""Frame-local person ownership and conservative circle fallback views.

Owned returns still terminate rays. Only their obstacle evidence is delegated;
background geometry and source measurement times remain unchanged.
"""
import numpy as np
import cv2


def point_owners(points, boxes, config, sensor, plane=None, *, edge_allowed=None):
    """Bounded XY clustering inside weak boxes; no per-cell ground prerequisite."""
    settings=config['dynamic'];resolution=config['resolution']
    owner=np.full(len(points),-1,np.int32)
    if not len(boxes) or not len(points):
        return owner
    best=np.full(len(points),np.inf)
    ground=None if plane is None else points[:,:2]@plane[:2]+plane[2]
    radius=settings['object_radius_limit_m']
    # Leave a conservative margin before applying the original Euclidean gate.
    extent=2*radius
    for number,box in enumerate(boxes):
        candidates=np.flatnonzero((abs(points[:,0]-box[0])<=extent)&(abs(points[:,1]-box[1])<=extent))
        if len(candidates)<settings['min_cluster_points']:
            continue
        if len(candidates)==1:
            # Preserve the original matrix multiplication at box/cell edges.
            candidates=np.arange(len(points))
        xy=points[candidates,:2]-box[:2];distance=np.linalg.norm(xy,axis=1)
        c,s=np.cos(box[6]),np.sin(box[6]);local=xy@np.array([[c,-s],[s,c]])
        half=np.clip(box[3:5]/2+settings['padding_m'],resolution,settings['max_radius_m'])
        heights=points[candidates,2]
        keep=((abs(local)<=half).all(1)&(distance<=radius)&
              (abs(heights-box[2])<=box[5]/2+settings['padding_m']))
        if ground is None:
            keep&=heights>box[2]-box[5]/2+config['min_obstacle_height_m']
        else:
            above_ground=heights-ground[candidates]
            keep&=(above_ground>=config['min_obstacle_height_m'])&(above_ground<=config['max_height_m'])
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
        selected=indices[distance[indices]<best[candidates[indices]]]
        owned=candidates[selected]
        owner[owned]=number;best[owned]=distance[selected]
    growth = settings.get('edge_growth_m', 0.)
    if growth and edge_allowed is not None and ground is not None and (owner >= 0).any():
        _grow_edges(points, boxes, owner, ground, config, growth, edge_allowed)
    return owner


def _grow_edges(points, boxes, owner, ground, config, growth, edge_allowed):
    from scipy.spatial import cKDTree

    settings = config['dynamic']
    # Only the original core supplies seeds, even when a point is near two people.
    core = owner.copy()
    height = points[:, 2]-ground
    pending = np.flatnonzero((core < 0) & (height >= config['min_obstacle_height_m'])
                             & (height <= config['max_height_m']))
    if not len(pending):
        return
    best = np.full(len(points), np.inf)
    radial_limit = min(.98, settings['object_radius_limit_m'] + .75*growth)
    for number in np.unique(core[core >= 0]):
        box = boxes[number]
        offset = points[pending, :2] - box[:2]
        near = (np.abs(offset) <= radial_limit).all(axis=1)
        xy = offset[near]
        candidates = pending[near]
        c, s = np.cos(box[6]), np.sin(box[6])
        local = xy @ np.array([[c, -s], [s, c]])
        half = np.clip(box[3:5]/2 + settings['padding_m'], config['resolution'], settings['max_radius_m'])
        eligible = ((xy**2).sum(axis=1) <= radial_limit**2) & (np.abs(local) <= half+.8*growth).all(axis=1)
        eligible &= np.abs(points[candidates, 2]-box[2]) <= box[5]/2 + settings['padding_m']
        candidates = candidates[eligible]
        if not len(candidates):
            continue
        distances, _ = cKDTree(points[core == number]).query(points[candidates], distance_upper_bound=growth)
        improve = distances < best[candidates]
        nearby = candidates[improve]
        if not len(nearby):
            continue
        permitted = edge_allowed(points[nearby], ground[nearby])
        owner[nearby[permitted]] = number
        best[nearby[permitted]] = distances[improve][permitted]


def static_edge_permission(prior, transform, stamp, points, ground):
    """Check time-valid prior free space and ground, away from occupied cells."""
    from .navigation import STATIC_GROUND_TOLERANCE_M

    allowed = np.zeros(len(points), dtype=bool)
    layers, meta = prior.layers, prior.metadata
    names = ('last_observed', 'ground_support', 'ground_z')
    if not len(points) or any(name not in layers for name in names):
        return allowed
    position = points @ transform[:2, :3].T + transform[:2, 3]
    ij = np.floor((position-meta['origin_xy'])/meta['resolution']).astype(int)
    occupancy = layers['occupancy']
    rows = np.flatnonzero(((ij >= 1) & (ij < np.asarray(occupancy.shape[::-1])-1)).all(axis=1))
    x, y = ij[rows].T
    seen = layers['last_observed'][y, x]
    mapped_ground = points[rows, :2] @ transform[2, :2] + ground[rows]*transform[2, 2] + transform[2, 3]
    accepted = ((occupancy[y, x] == 0) & layers['ground_support'][y, x] & np.isfinite(seen)
                & (seen <= stamp) & (np.abs(mapped_ground-layers['ground_z'][y, x]) <= STATIC_GROUND_TOLERANCE_M))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            accepted &= occupancy[y+dy, x+dx] != 100
    allowed[rows] = accepted
    return allowed


def extract_current_owners(points, boxes, predicted, config, sensor, plane, edge_allowed):
    """Use recent predictions only to find a supported core in the current scan."""
    if not predicted.any():
        return point_owners(points, boxes, config, sensor, plane, edge_allowed=edge_allowed)
    observed = np.flatnonzero(~predicted)
    owners = point_owners(points, boxes[observed], config, sensor, plane, edge_allowed=edge_allowed)
    assigned = owners >= 0
    owners[assigned] = observed[owners[assigned]]
    if edge_allowed is None or plane is None:
        return owners

    predicted_ids = np.flatnonzero(predicted)
    radius = config['dynamic']['object_radius_limit_m']
    nearby = np.zeros(len(points), dtype=bool)
    for box in boxes[predicted_ids]:
        nearby |= (np.abs(points[:, :2]-box[:2]) <= radius).all(axis=1)
    candidates = np.flatnonzero(nearby & ~assigned)
    ground = points[candidates, :2] @ np.asarray(plane[:2]) + plane[2]
    candidates = candidates[edge_allowed(points[candidates], ground)]
    core = point_owners(points[candidates], boxes[predicted_ids], config, sensor, plane)
    matched = core >= 0
    owners[candidates[matched]] = predicted_ids[core[matched]]
    return owners


def ownership_record(points, owners, boxes, tracks, people, stamp, identity, frame, *, current_only=None):
    objects=[]
    for index,(box,track) in enumerate(zip(boxes,tracks)):
        selected=owners==index
        if not selected.any():continue
        radius=max(.30,float(np.linalg.norm(points[selected,:2]-box[:2],axis=1).max())+.02)
        obj = dict(identity=[people['session_id'],people['generation'],str(track)],
                   xy=box[:2].tolist(),radius_m=radius,points=int(selected.sum()))
        if current_only is not None and current_only[index]:
            obj['current_only'] = True
        objects.append(obj)
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
                not np.isfinite(xy).all() or not np.isfinite(radius) or not 0<radius<=1.
                or type(obj.get('current_only', False)) is not bool):
            raise ValueError('Invalid person object geometry')
        identities.add(key)
    return objects


def circle_indices(grid, centre, radius):
    """Flat indices of raster cells touched by a circle in the grid frame."""
    height,width = grid.layers['occupancy'].shape
    resolution = grid.metadata['resolution']
    origin = np.asarray(grid.metadata['origin_xy'])
    centre = np.asarray(centre)
    low = np.maximum(0,np.floor((centre-radius-origin)/resolution).astype(int))
    high = np.minimum([width,height],np.ceil((centre+radius-origin)/resolution).astype(int))
    if np.any(high<=low):
        return np.empty(0, dtype=np.intp)
    y,x = np.mgrid[low[1]:high[1],low[0]:high[0]]
    xy = np.stack((x+.5,y+.5),axis=-1)*resolution+origin
    gap = np.maximum(abs(xy-centre)-resolution/2,0.)
    return (y*width+x)[(gap**2).sum(axis=-1)<=radius**2]


def circle_view(grid, objects):
    """Whole-cell intersection with currently observed circles; no free-space writes."""
    if not objects:return grid
    from .layers import GridSnapshot
    m=grid.metadata;cells=grid.layers['occupancy'].copy()
    for obj in objects:
        cells.flat[circle_indices(grid,obj['xy'],obj['radius_m'])]=100
    cells.flags.writeable=False
    return GridSnapshot(dict(m,navigation_view='background_with_current_person_circles'),dict(occupancy=cells))
