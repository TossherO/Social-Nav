"""Current static-support geometry and independent navigation consumption checks."""
import copy
import numpy as np

from .context import fingerprint
from .layers import GridSnapshot
from perception.detection_tracking.inference.geometry import validate_transform


NAVIGATION_DTYPES = dict(navigation_occupancy=np.int8, navigation_source=np.uint8,
                         navigation_source_stamp=np.float64)
CAPSULE_FIELDS = ('occupancy', 'person_background_free', 'ground_conflict', *NAVIGATION_DTYPES)
STATIC_GROUND_TOLERANCE_M = .15


def transform_plane(plane, transform):
    """Transform z=a*x+b*y+c as a covector, preserving roll and pitch."""
    normal = np.linalg.solve(transform.T, np.r_[-plane[:2], 1., -plane[2]])
    if abs(normal[2]) < 1e-6: raise ValueError('Ground plane is not a height field')
    return -normal[[0, 1, 3]]/normal[2]


def static_floor_reference(snapshot, prior, transform, radius, sensor_height_band):
    """Seed height lookup from the selected static map, without observing it again.

    The seed only locates static cells. Their individual elevations and complete
    collision-column overlaps are still checked by prior_floor/static_support.
    """
    m,pm=snapshot.metadata,prior.metadata
    sensor=transform@np.r_[np.asarray(m['world_from_lidar'])[:3,3],1.]
    resolution=pm['resolution'];origin=np.asarray(pm['origin_xy'])
    lo=np.maximum(0,np.floor((sensor[:2]-radius-.5-origin)/resolution).astype(int))
    hi=np.minimum(np.asarray(pm['shape'][::-1]),np.ceil((sensor[:2]+radius+.5-origin)/resolution).astype(int))
    if np.any(hi<=lo):return None
    region=np.s_[lo[1]:hi[1],lo[0]:hi[0]];layers=prior.layers
    if any(k not in layers for k in ('ground_z','ground_support','last_observed')):return None
    height=layers['ground_z'][region];times=layers['last_observed'][region]
    valid=(layers['occupancy'][region]==0)&layers['ground_support'][region]&np.isfinite(height)&np.isfinite(times)&(times<=m['stamp_s'])
    clearance=sensor[2]-height
    valid&=(clearance>=sensor_height_band[0])&(clearance<=sensor_height_band[1])
    if not valid.any():return None
    a=m['association']
    return dict(source='selected_static_map',map_plane=[0.,0.,float(np.median(height[valid]))],
        stamp_s=m['stamp_s'],source_stamp_s=float(times[valid].min()),
        association_identity=[a['map_id'],a['map_frame'],a.get('association_session'),a.get('association_generation'),a.get('map_revision')])


def association_reason(snapshot, asset, max_age, *, now=None):
    m = snapshot.metadata; a = m.get('association', {}); stamp = m['stamp_s']
    if asset is None: return 'static_map_missing'
    if m.get('geometry_status') not in ('OBSERVED', 'GROUND_UNAVAILABLE'):
        return 'local_geometry_invalid'
    if not a.get('valid') or a.get('status') != 'PRIOR_ASSOCIATED':
        return 'association_unavailable'
    if (a.get('local_session_id'), a.get('local_generation'), a.get('local_frame')) != (
            m['session_id'], m['generation'], m['frame_id']):
        return 'association_lifecycle_mismatch'
    measured = a.get('measurement_stamp_s')
    if (measured is None or not np.isfinite(measured) or a.get('stamp_s') != stamp or
            not 0 <= (stamp if now is None else now)-measured <= max_age):
        return 'association_future_or_stale'
    pm = asset.snapshot.metadata
    if a.get('map_id') != pm.get('map_id') or a.get('map_frame') != pm['frame_id']:
        return 'association_map_mismatch'
    try:
        transform = validate_transform(a['map_from_local'])
    except (KeyError, ValueError): return 'association_transform_invalid'
    if np.arccos(np.clip(transform[2, 2], -1., 1.)) > .15:
        return 'map_local_tilt_exceeds_planar_limit'
    return None


def static_support(snapshot, prior, transform, plane, height_band, selected, ground_tolerance, *, floor=None):
    """Require observed static FREE for every positive-area cell overlap.

    Ground is used only for coordinate projection. The XY projection of an
    inclined collision column is a zonotope; SAT includes all three generators,
    rather than sampling cell centres or silently discarding roll and pitch.
    """
    m, pm = snapshot.metadata, prior.metadata
    y, x = np.nonzero(selected); n = len(x)
    free = np.ones(n, bool); seen = np.full(n, np.inf)
    if not n:return free,seen
    res = float(m['resolution']); target = float(pm['resolution'])
    xy = np.c_[x+.5, y+.5]*res + m['origin_xy']
    intercept = plane[2] if floor is None else np.asarray(floor)[y,x]-xy@plane[:2]
    low, high = height_band
    z = xy@plane[:2]+intercept+(low+high)/2
    centres = np.c_[xy, z]@transform[:2, :3].T + transform[:2, 3]
    affine = transform[:2, :2]+np.outer(transform[:2, 2], plane[:2])
    inverse = np.linalg.inv(affine)
    ground_centre = np.c_[xy, xy@plane[:2]+intercept]@transform[:3, :3].T+transform[:3, 3]
    generators = np.c_[affine*res/2, transform[:2, 2]*(high-low)/2]
    extent = np.abs(generators).sum(axis=1)
    lo = np.floor((centres-extent-pm['origin_xy'])/target+1e-9).astype(int)
    hi = np.floor((centres+extent-pm['origin_xy'])/target-1e-9).astype(int)
    span = (hi-lo).max(axis=0)+1
    if np.prod(span) > 81: raise ValueError('Navigation prior projection exceeds overlap budget')
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    for v in generators.T:
        if np.linalg.norm(v) > 1e-10:
            axes.append(np.array([-v[1], v[0]])/np.linalg.norm(v))
    axes = np.asarray(axes).T
    radii = np.abs(generators.T@axes).sum(axis=0)+target/2*np.abs(axes).sum(axis=0)
    cells = prior.layers['occupancy']; times = prior.layers.get('last_observed')
    ground = prior.layers.get('ground_z'); supported = prior.layers.get('ground_support')
    for dy in range(span[1]):
        for dx in range(span[0]):
            ij = lo+[dx, dy]
            other = (ij+.5)*target+pm['origin_xy']
            overlap = (np.abs((other-centres)@axes) < radii-1e-10).all(axis=1)
            inside = ((ij >= 0)&(ij < np.asarray(cells.shape[::-1]))).all(axis=1)
            free[overlap & ~inside] = False
            rows = np.flatnonzero(overlap & inside)
            xx, yy = ij[rows].T
            accepted=cells[yy,xx]==0
            free[rows] &= accepted
            if ground is None or supported is None:
                free[rows] = False
            else:
                # Compare the two floor references at each supporting static
                # cell, including roll/pitch. Equal XY is insufficient on floors.
                local_xy = xy[rows]+(other[rows]-ground_centre[rows, :2])@inverse.T
                local_z = local_xy@plane[:2]+(intercept if np.ndim(intercept)==0 else intercept[rows])
                mapped_z = np.c_[local_xy, local_z]@transform[2, :3]+transform[2, 3]
                free[rows] &= (supported[yy, xx]&np.isfinite(ground[yy, xx])&
                               (np.abs(ground[yy, xx]-mapped_z) <= ground_tolerance))
            if times is None:
                free[rows] = False
            else:
                observed = times[yy, xx]
                free[rows] &= np.isfinite(observed)&(observed <= m['stamp_s'])
                seen[rows] = np.minimum(seen[rows], observed)
    free &= np.isfinite(seen)
    return free,seen


def prior_floor(snapshot, prior, transform, map_plane, *, selected=None):
    """Map elevation defines cached floor; current plane noise cannot erase it."""
    m,pm=snapshot.metadata,prior.metadata
    y,x=np.nonzero(np.ones(m['shape'],bool) if selected is None else selected)
    xy=np.c_[x+.5,y+.5]*m['resolution']+m['origin_xy']
    plane=transform_plane(np.asarray(map_plane),np.linalg.inv(transform))
    points=np.c_[xy,xy@plane[:2]+plane[2]]@transform[:3,:3].T+transform[:3,3]
    ij=np.floor((points[:,:2]-pm['origin_xy'])/pm['resolution']).astype(int)
    inside=((ij>=0)&(ij<np.asarray(pm['shape'][::-1]))).all(axis=1)
    result=np.full(len(xy),np.nan);rows=np.flatnonzero(inside);xx,yy=ij[rows].T
    layers=prior.layers
    accepted=layers['occupancy'][yy,xx]==0
    valid=accepted&layers['ground_support'][yy,xx]&np.isfinite(layers['ground_z'][yy,xx])
    rows,xx,yy=rows[valid],xx[valid],yy[valid]
    result[rows]=(layers['ground_z'][yy,xx]-xy[rows]@transform[2,:2]-transform[2,3])/transform[2,2]
    output=np.full(m['shape'],np.nan);output[y,x]=result
    return output


def navigation_geometry(scene, *, now=None, asset=None, check_asset=False, _layer_binding=None, include_people=True):
    """Default consumers retain current contact, even without a people packet."""
    from .person_objects import observed_objects,circle_view
    objects=observed_objects(scene.geometry)
    grid=_background_geometry(scene,now=now,asset=asset,check_asset=check_asset,_layer_binding=_layer_binding)
    return circle_view(grid,objects) if include_people else grid


def _background_geometry(scene, *, now=None, asset=None, check_asset=False, _layer_binding=None):
    """Reuse spatial checks only while actual source and asset bytes still match.

    Each reconstructed scene owns its small cache, including in independent
    safety. Source age is checked on every use. SceneInputs.verify may pass the
    binding it has just computed; other callers hash the actual arrays here.
    """
    raw = scene.geometry
    proof = raw.metadata.get('navigation_fusion')
    if proof is None:
        return _navigation_geometry(scene, now=now, asset=asset, check_asset=check_asset)
    from .context import layer_binding
    binding = layer_binding(raw) if _layer_binding is None else _layer_binding
    asset_binding = None
    if check_asset and asset is not None:
        asset_binding = (fingerprint(asset.manifest), layer_binding(asset.snapshot))
    key = (binding, scene.record['as_of'], fingerprint(scene.record.get('prior')),
           check_asset, asset_binding)
    cache = getattr(scene, '_navigation_checks', None)
    if cache is None or key not in cache:
        grid = _navigation_geometry(scene, now=now, asset=asset, check_asset=check_asset)
        if cache is not None:
            if len(cache) >= 2:
                cache.clear()
            source = raw.layers['navigation_source']
            cache[key] = bool(((source == 3) | (source == 10)).any())
        return grid
    if cache[key]:
        stamp = scene.record['as_of']
        measured = scene.record['prior'].get('measurement_stamp_s')
        if measured is None or not np.isfinite(measured) or not 0 <= (stamp if now is None else now)-measured <= proof['max_association_age_s']:
            raise ValueError('navigation_prior_future_or_stale')
    return GridSnapshot(dict(raw.metadata, navigation_view='historical_support_not_new_measurement',
        snapshot_id=raw.metadata['snapshot_id']+':navigation'), dict(occupancy=raw.layers['navigation_occupancy']))


def safety_capsule(snapshot):
    """Keep the current consumer fields in a separate metadata envelope."""
    return GridSnapshot(copy.deepcopy(snapshot.metadata),
        {k:v for k,v in snapshot.layers.items() if k in CAPSULE_FIELDS})


def _navigation_geometry(scene, *, now=None, asset=None, check_asset=False):
    """Validate the current derivation before selecting its collision grid."""
    raw = scene.geometry
    proof = raw.metadata.get('navigation_fusion')
    if proof is None:
        if any(k in raw.layers for k in NAVIGATION_DTYPES):
            raise ValueError('Navigation layers lack provenance')
        return raw
    from .support import METHOD, verify
    if proof.get('method') != METHOD:
        raise ValueError('Unsupported navigation derivation; rebuild from original observations')
    return verify(scene, now=now, asset=asset, check_asset=check_asset)
