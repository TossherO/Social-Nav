"""Read-only geometry queries bound to an immutable snapshot and its frame."""
import hashlib
import json

import numpy as np


def encoded(value):
    """Keep the original serialization of scan journals and source identities."""
    return json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _byte_storage(value):
    root = value
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    return root if isinstance(root, bytes) and not value.flags.writeable else None


def layer_binding(snapshot):
    metadata = json.dumps(snapshot.metadata,sort_keys=True,separators=(',',':'),
                          ensure_ascii=False,allow_nan=False).encode()
    layers = sorted(snapshot.layers.items())
    cached = getattr(snapshot, '_binding_cache', None)
    if (cached is not None and metadata == cached[0] and len(layers) == len(cached[1]) and
            all(name == old_name and value is old and str(value.dtype) == dtype and
                tuple(value.shape) == shape and _byte_storage(value) is owner
                for (name, value), (old_name, old, dtype, shape, owner) in zip(layers, cached[1]))):
        return cached[2]
    digest = hashlib.sha256(metadata)
    immutable = True
    for name,value in layers:
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        # Hash the same C-order bytes without a second full-array allocation.
        digest.update(np.ascontiguousarray(value))
        immutable &= _byte_storage(value) is not None
    result = digest.hexdigest()
    # Only physical immutable bytes can skip later array reads. Shared leases
    # and ordinary read-only views retain the full content check on every use.
    if immutable and hasattr(snapshot, '_binding_cache'):
        object.__setattr__(snapshot, '_binding_cache', (metadata,
            [(name, value, str(value.dtype), tuple(value.shape), _byte_storage(value)) for name,value in layers], result))
    return result


def json_safe(value):
    """Keep absent floating layer values explicit in JSON query responses."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def query_geometry(snapshot, points, *, frame):
    meta, layers = snapshot.metadata,snapshot.layers
    if frame != meta['frame_id']:
        raise ValueError('Transform the query explicitly into the snapshot frame')
    points = np.asarray(points,dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx2 query coordinates in metres')
    cells = np.floor((points-np.asarray(meta['origin_xy']))/meta['resolution']).astype(int)
    height,width = meta['shape']
    result = []
    for x,y in cells:
        row = dict(snapshot_id=meta['snapshot_id'],stamp_s=meta['stamp_s'],frame_id=frame)
        row.update({key:meta[key] for key in ('entity_type','map_type','view_type','source_map_type') if key in meta})
        if not 0 <= x < width or not 0 <= y < height:
            row.update(state='outside_coverage',ground_z=None,clearance_m=None,unknown_distance_m=None)
        else:
            row['state'] = {-1:'unknown',0:'free',100:'occupied'}[int(layers['occupancy'][y,x])]
            row['conditional_obstacle'] = bool(layers['conditional_obstacle'][y,x]) if 'conditional_obstacle' in layers else False
            if 'revised_free_stamp' in layers:
                value=float(layers['revised_free_stamp'][y,x])
                row['revised_free_stamp_s']=value if np.isfinite(value) else None
            for name,key in [('ground_z','ground_z'),('clearance_m','clearance'),('unknown_distance_m','unknown_distance')]:
                value = float(layers[key][y,x]) if key in layers else np.nan
                row[name] = value if np.isfinite(value) else None
            row['coverage'] = float(layers['coverage'][y,x])
            seen = float(layers['last_observed'][y,x])
            row['last_observed_s'] = seen if np.isfinite(seen) else None
            if 'near_field_free' in layers:
                row['historical_free']=bool(layers['near_field_free'][y,x])
                row['free_source']=('historical_background' if row['historical_free'] else 'ray_evidence') if row['state']=='free' else None
                for layer,field in [('near_field_free_classified_at','historical_free_classified_at_s'),
                                    ('near_field_obstacle_stamp','recent_obstacle_stamp_s'),
                                    ('near_field_obstacle_expires_at','recent_obstacle_expires_at_s')]:
                    if layer in layers:
                        value=float(layers[layer][y,x]);row[field]=value if np.isfinite(value) else None
            if 'assumed_free' in layers:
                row['assumed_free']=bool(layers['assumed_free'][y,x])
                if row['assumed_free']:row['free_source']='assumed_no_endpoint'
        result.append(row)
    return result


def crop_map(snapshot, bounds, *, frame, layers=('occupancy',), max_cells=None):
    if frame != snapshot.metadata['frame_id']:
        raise ValueError('Crop bounds must be in the snapshot frame')
    bounds = np.asarray(bounds,dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
        raise ValueError('Expected [xmin,ymin,xmax,ymax] bounds')
    meta = snapshot.metadata
    low = np.maximum(0,np.floor((bounds[:2]-meta['origin_xy'])/meta['resolution']).astype(int))
    high = np.minimum(np.array(meta['shape'])[::-1],np.ceil((bounds[2:]-meta['origin_xy'])/meta['resolution']).astype(int))
    if np.any(high<=low):
        raise ValueError('Crop is outside observed window')
    if max_cells is not None and int(np.prod(high-low)) > max_cells:
        raise ValueError('Map crop exceeds cell budget')
    return dict(snapshot_id=meta['snapshot_id'],frame_id=frame,stamp_s=meta['stamp_s'],resolution=meta['resolution'],
                origin_xy=(np.array(meta['origin_xy'])+low*meta['resolution']).tolist(),
                layers={name:snapshot.layers[name][low[1]:high[1],low[0]:high[0]].copy() for name in layers},
                **{key:meta[key] for key in ('entity_type','map_type','view_type','source_map_type') if key in meta})


def query_scene(bundle, request, settings):
    if (not isinstance(request,dict) or set(request)-{'operation','observation_id','input_fingerprint','bounds','layers','points','frame'}
            or request.get('observation_id') != bundle.record['observation_id']):
        raise ValueError('A cached scene observation_id and a read-only operation are required')
    if request.get('input_fingerprint',bundle.record['input_fingerprint']) != bundle.record['input_fingerprint']:
        raise ValueError('Scene input fingerprint mismatch')
    operation = request.get('operation')
    if operation == 'get_scene':
        import copy
        return copy.deepcopy(bundle.record)
    if operation == 'get_rgb':
        import cv2
        import base64
        if bundle.rgb is None:
            raise ValueError('This observation has no valid current RGB')
        success,data = cv2.imencode('.png',bundle.rgb[:,:,::-1])
        if not success:
            raise ValueError('RGB encoding failed')
        answer = dict(encoding='png',pixels_sha256=bundle.record['rgb']['sha256'],
                      data_base64=base64.b64encode(data).decode('ascii'))
    elif operation == 'get_prior':
        import copy
        answer = copy.deepcopy(bundle.record.get('prior_geometry',dict(status='unavailable',reason='not_enabled')))
    elif operation in ('query_geometry','query_prior_geometry'):
        target = bundle.geometry if operation == 'query_geometry' else bundle.prior_geometry
        if target is None:
            raise ValueError('This scene has no associated prior view')
        if len(request['points']) > 256:
            raise ValueError('Geometry query exceeds point budget')
        answer = query_geometry(target,request['points'],frame=request['frame'])
        if operation == 'query_prior_geometry':
            answer = dict(time_scope='offline_final_prior',current_free_space=False,cells=answer)
    elif operation in ('crop_map','crop_prior_map'):
        target = bundle.geometry if operation == 'crop_map' else bundle.prior_geometry
        if target is None:
            raise ValueError('This scene has no associated prior view')
        names = request.get('layers',['occupancy'])
        if not isinstance(names,list) or not 1 <= len(names) <= 8 or any(name not in target.layers for name in names):
            raise ValueError('Unsupported scene geometry layers')
        answer = crop_map(target,request['bounds'],frame=request['frame'],layers=names,
                          max_cells=settings['tool_crop_cells'])
        if operation == 'crop_prior_map':
            answer.update(time_scope='offline_final_prior',current_free_space=False)
    else:
        raise ValueError('Unsupported scene operation')
    return json_safe(dict(observation_id=bundle.record['observation_id'],
        input_fingerprint=bundle.record['input_fingerprint'],as_of=bundle.record['as_of'],data=answer))

