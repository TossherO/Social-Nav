"""Read-only geometry queries bound to an immutable snapshot and its frame."""
import hashlib
import json

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def layer_binding(snapshot):
    digest = hashlib.sha256(json.dumps(snapshot.metadata,sort_keys=True,separators=(',',':'),
                                      ensure_ascii=False,allow_nan=False).encode())
    for name,value in sorted(snapshot.layers.items()):
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


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
        if not 0 <= x < width or not 0 <= y < height:
            row.update(state='outside_coverage',ground_z=None,clearance_m=None,unknown_distance_m=None)
        else:
            row['state'] = {-1:'unknown',0:'free',100:'occupied'}[int(layers['occupancy'][y,x])]
            for name,key in [('ground_z','ground_z'),('clearance_m','clearance'),('unknown_distance_m','unknown_distance')]:
                value = float(layers[key][y,x])
                row[name] = value if np.isfinite(value) else None
            row['coverage'] = float(layers['coverage'][y,x])
            seen = float(layers['last_observed'][y,x])
            row['last_observed_s'] = seen if np.isfinite(seen) else None
        result.append(row)
    return result


def crop_map(snapshot, bounds, *, frame, layers=('occupancy',)):
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
    return dict(snapshot_id=meta['snapshot_id'],frame_id=frame,stamp_s=meta['stamp_s'],resolution=meta['resolution'],
                origin_xy=(np.array(meta['origin_xy'])+low*meta['resolution']).tolist(),
                layers={name:snapshot.layers[name][low[1]:high[1],low[0]:high[0]].copy() for name in layers})


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
        # Check the crop size before copying its arrays.
        import numpy as np
        meta = target.metadata
        bounds = np.asarray(request['bounds'],float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:]<=bounds[:2]):
            raise ValueError('Expected finite ordered XY crop bounds')
        low = np.maximum(0,np.floor((bounds[:2]-meta['origin_xy'])/meta['resolution']))
        high = np.minimum(np.array(meta['shape'])[::-1],np.ceil((bounds[2:]-meta['origin_xy'])/meta['resolution']))
        if np.prod(np.maximum(0,high-low)) > settings['tool_crop_cells']:
            raise ValueError('Scene crop exceeds cell budget')
        answer = crop_map(target,bounds,frame=request['frame'],layers=names)
        if operation == 'crop_prior_map':
            answer.update(time_scope='offline_final_prior',current_free_space=False)
    else:
        raise ValueError('Unsupported scene operation')
    return json_safe(dict(observation_id=bundle.record['observation_id'],
        input_fingerprint=bundle.record['input_fingerprint'],as_of=bundle.record['as_of'],data=answer))

