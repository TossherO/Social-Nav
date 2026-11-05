"""Causal track/ego histories in each observation's own LiDAR frame."""
import hashlib
import json

import numpy as np


HISTORY_S = 3.
MAX_GAP_S = .35
MAX_POSE_AGE_S = .05


def local_scope(row):
    binding = row.get('localization_binding') or {}
    return [row.get(k) for k in ('sequence_id','session_id','generation','clock_domain','frame_id','detection_frame_id')] + [
        binding.get(k) for k in ('session_id','generation','frame_id','source')]


def pose(row):
    if (row.get('health')!='valid' or row.get('pose_health')!='valid' or
            row.get('pose_source') not in ('localization','dataset_pose','stationary_assumption')):
        return None
    time = row.get('pose_stamp_s')
    if time is None or not np.isfinite(time) or not 0 <= row['stamp_s']-time <= MAX_POSE_AGE_S+1e-7:
        return None
    matrix = np.asarray(row.get('world_from_lidar'),dtype=float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3],[0,0,0,1]) or
            not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-4) or
            not np.isclose(np.linalg.det(matrix[:3,:3]),1,atol=1e-4)):
        return None
    return matrix


def build_trajectories(entries, references, mapping):
    """Entries are indexed result metadata; no image or scan decoding is needed."""
    result = {}
    by_id = {entry['id']:entry for entry in entries}
    for ident in references:
        if ident is None:
            continue
        ref = by_id.get(ident)
        row = ref.get('result') if ref else None
        reference = pose(row) if row else None
        if reference is None:
            result[ident] = dict(status='unavailable',reason='reliable_timed_pose_missing',ego=[],people={},history_s=HISTORY_S)
            continue
        inverse = np.linalg.inv(reference)
        scope = local_scope(row)
        selected = [e for e in entries if 0 <= ref['stamp_ns']-e['stamp_ns'] <= HISTORY_S*1e9]
        ego, people = [], {key:[] for key in mapping}
        previous, active = None, {}
        bindings = []
        for entry in selected:
            raw = entry.get('result')
            transform = pose(raw) if raw else None
            bindings.append([entry['id'],raw])
            if transform is None or local_scope(raw)!=scope:
                previous, active = None, {}
                continue
            stamp = raw['stamp_s']
            dt = round((entry['stamp_ns']-ref['stamp_ns'])/1e9,6)
            if previous is None or stamp-previous>MAX_GAP_S:
                ego.append([]);active={}
            center = inverse[:3,:3]@transform[:3,3]+inverse[:3,3]
            ego[-1].append([dt,*np.round(center[:2],4).tolist()])
            tracks={t['track_id']+'/g'+str(t['lifecycle_generation']):t for t in raw.get('tracks',[])}
            for key in mapping:
                track=tracks.get(key)
                if not track or not track.get('confirmed') or track['state'] not in ('observed','predicted'):
                    active.pop(key,None)
                    continue
                xyz=np.asarray(track['box'][:3],dtype=float)
                if not np.isfinite(xyz).all():
                    active.pop(key,None);continue
                xy=(inverse[:3,:3]@xyz+inverse[:3,3])[:2]
                observed=track['state']=='observed'
                point=[dt,*np.round(xy,4).tolist(),observed]
                old=active.get(key)
                edge_observed=observed and (old is None or old['observed'])
                if old is None or old['segment']['observed']!=edge_observed:
                    segment=dict(observed=edge_observed,points=[] if old is None else [old['point']])
                    people[key].append(segment)
                else:
                    segment=old['segment']
                segment['points'].append(point)
                active[key]=dict(observed=observed,point=point,segment=segment)
            previous=stamp
        binding=hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        times=[p[0] for segment in ego for p in segment]
        result[ident]=dict(status='available',history_s=HISTORY_S,covered_s=max(times)-min(times) if times else 0,
                           ego=ego,people=people,source_binding=binding,
                           coordinate_semantics='past_world_centers_in_this_observation_lidar_frame')
    return result
