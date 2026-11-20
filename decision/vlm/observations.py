"""Task images from sealed runtime observations; no labels, model or ROS imports."""
import copy
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .annotation.rules import digest
from .scene_inputs import build_window, model_eligible, pointcloud_background, render_observation, renderer_identity
from .trajectory_inputs import build_trajectories
from perception.detection_tracking.inference.rgb_association import people_in_lidar
from perception.detection_tracking.inference.rgb_recording import recorded_input
from navigation.planning.people import stable_identity


def frame(entry, arrays, *, include_predicted=False, raw_rgb=False):
    row=entry['people'];rgb=arrays.get('rgb');association=entry.get('association')
    image_input=recorded_input(row,association) if association else None
    if raw_rgb:
        meta=entry.get('image_metadata') or {}
        image_input=dict(rgb_stamp_ns=meta.get('rgb_stamp_ns'),camera_frame=meta.get('rgb_frame'),
                         raw_rgb_sha256=None if rgb is None else hashlib.sha256(rgb.tobytes()).hexdigest())
    if rgb is None or image_input is None or image_input['raw_rgb_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest():
        raise ValueError('Decision RGB must bind the recorded 2D association')
    ns=entry['stamp_ns'];rgb_ns=image_input['rgb_stamp_ns']
    if type(rgb_ns) is not int or not 0<=ns-rgb_ns<=150_000_000:
        raise ValueError('Decision RGB must be causal and within 150 ms')
    people=[]
    for p in people_in_lidar(row):
        record=None if raw_rgb else association['result']['tracks'].get(p['key'])
        if not raw_rgb and record is None:raise ValueError('Association omits a source identity')
        people.append(dict(p,eligible=model_eligible(p,include_predicted=include_predicted),
            rgb_box=record.get('rgb_box') if record and record['status']=='matched' else None))
    result=dict(id=entry['id'],source_id=row['sequence_id'],stamp_ns=ns,people=people,
        image_size=[rgb.shape[1],rgb.shape[0]],has_rgb=True,clock_domain=row['clock_domain'],
        timestamp_source=row['timestamp_source'],rgb_stamp_ns=rgb_ns,
        source_binding=entry['buffer'].content_sha256,perception_fingerprint=row['input_fingerprint'])
    result['fingerprint']=digest(result)
    return result


def render_window(entries, history, read_arrays, *, extent, view_binding, image_preparation=None,
                  include_predicted=False, raw_rgb=False):
    """Render a leased window in memory; returned PNG bytes own their storage."""
    rows=[];assets={}
    for entry in entries:
        arrays=read_arrays(entry['buffer'])
        rows.append(frame(entry,arrays,include_predicted=include_predicted,raw_rgb=raw_rgb));assets[entry['id']]=arrays
    window=build_window(rows,rows[-1],digest([r['source_binding'] for r in rows]),extent=extent)
    if not window['ready'] or any(i is None for i in window['frames']):raise ValueError('Incomplete runtime decision window')
    trajectories=build_trajectories(history,window['frames'],window['mapping'])
    current=entries[-1]['people'];tracks={t['track_id']+'/g'+str(t['lifecycle_generation']):t for t in current['tracks']}
    identities={str(pid):list(stable_identity(current,tracks[key]))
                for key,pid in window['mapping'].items()}
    images=[];payloads=[]
    by_id={r['id']:r for r in rows}
    for index,ident in enumerate(window['frames']):
        arrays=assets[ident];rgb=io.BytesIO();Image.fromarray(arrays['rgb']).save(rgb,format='PNG')
        image=render_observation(by_id[ident],rgb.getvalue(),window['mapping'],window['fields']['dt'][index],extent,
            trajectories[ident],bev_background=pointcloud_background(arrays['raw'],extent),
            distinguish_predictions=include_predicted)
        if raw_rgb:
            from .scene_inputs import BEV_RECT
            enc=io.BytesIO();image.crop(BEV_RECT).save(enc,format='PNG')
            output=((f'frame_{index}_rgb.png',rgb.getvalue()),(f'frame_{index}_bev.png',enc.getvalue()))
        else:
            encoded=io.BytesIO();image.save(encoded,format='PNG')
            output=((f'frame_{index}.png',encoded.getvalue()),)
        for name,payload in output:
            payloads.append(payload);images.append(dict(path=name,sha256=hashlib.sha256(payload).hexdigest()))
    record=dict(kind='runtime_task_window',fields=window['fields'],mapping=window['mapping'],pid_to_track=identities,
        observation_id=current['observation_id'],stamp_s=current['stamp_s'],clock_domain=current['clock_domain'],
        localization_binding=current['localization_binding'],scene=copy.deepcopy(entries[-1].get('scene')),
        sources=[dict(observation_id=e['people']['observation_id'],stamp_s=e['people']['stamp_s'],
            **({'stamp_ns':e['stamp_ns']} if 'stamp_ns' in e['people'] else {}),
            sequence_id=e['people']['sequence_id'],clock_domain=e['people']['clock_domain'],
            localization_binding=e['people']['localization_binding'],world_from_lidar=e['people']['world_from_lidar'],
            perception_fingerprint=e['people']['input_fingerprint'],array_sha256=e['buffer'].content_sha256,
            rgb_input=(dict(rgb_stamp_ns=e['image_metadata']['rgb_stamp_ns'],camera_frame=e['image_metadata'].get('rgb_frame'))
                       if raw_rgb else e['association']['image_input']),
            association_sha256=None if raw_rgb else digest(e['association']),
            **(dict(association_record=copy.deepcopy(e['association'])) if image_preparation is not None else {})) for e in entries],
        images=images,trajectories=trajectories,renderer=renderer_identity(),display_view=view_binding,
        scope='causal_runtime_input_no_human_labels_no_vlm_request')
    if include_predicted:record['predicted_tracks_included']=True
    if raw_rgb:record['image_mode']='raw_rgb_with_separate_bev'
    if image_preparation is not None:record['image_preparation']=image_preparation
    record['input_fingerprint']=digest(record)
    return record,tuple(payloads)


def freeze_window(entries, history, read_arrays, output, *, extent, view_binding,image_preparation=None):
    """Render the same leased window into a new offline input directory."""
    from .vlm_api_common import save_json
    record,images=render_window(entries,history,read_arrays,extent=extent,view_binding=view_binding,
                                image_preparation=image_preparation)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for spec,payload in zip(record['images'],images):
        (out/spec['path']).write_bytes(payload)
    save_json(out/'input.json',record)
    return record
