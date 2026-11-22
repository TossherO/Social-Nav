"""Evidence shared by the local follow worker and the control-thread consumer."""
import math

from interfaces.runtime import rgb_age_s
from perception.environment.context import fingerprint


RESULT_MAX_AGE_S = .5
VISUAL_HOLD_S = .75
SEARCH_LEASE_S = .75


def current_candidates(scene, entry, pixels):
    from decision.vlm.t2.target import candidates
    from perception.detection_tracking.inference.rgb_recording import recorded_input

    association = entry['association']
    image = recorded_input(entry['people'], association)
    stamp_ns = image['rgb_stamp_ns']
    if (type(stamp_ns) is not int or not 0 <= entry['stamp_ns']-stamp_ns <= 150_000_000 or
            image['image_size'] != [pixels.shape[1], pixels.shape[0]]):
        raise ValueError('follow_reid_rgb_missing_or_stale')
    value = candidates(scene, pixels, stamp_ns/1e9, association['detections'], association['result'],
                       detector=dict(identity=association['detector_identity']), rgb_stamp_ns=stamp_ns)
    if value['rgb_pixel_sha256'] != image['raw_rgb_sha256']:
        raise ValueError('follow_reid_association_rgb_mismatch')
    return value


def current_track_positions(source, model):
    if model.status != 'valid':
        return {}
    tracks = {key: track for key, track in zip(model.ids, model.tracks)}
    positions = {}
    for candidate in source['candidates']:
        support = candidate['track']
        if support is None:
            continue
        track = tracks.get(tuple(support['identity']))
        if (track is not None and support['people_fingerprint'] == model.source and
                support['support_stamp_s'] == model.origin == track['last_observed']):
            positions[candidate['id']] = track['xy'].tolist()
    return positions


def bound(value):
    return value.get('binding') == fingerprint({key: field for key, field in value.items() if key != 'binding'})


def search_action(search, candidates, selected):
    if (not bound(search) or search['clock_domain'] != candidates['clock_domain'] or
            search['source_stamp_s'] != candidates['stamp_s'] or search['scene_binding'] != candidates['scene_binding'] or
            (selected is not None and
             (selected not in search['candidates'] or selected.get('kind') not in ('wait', 'approach')))):
        raise ValueError('follow_reid_search_evidence_invalid')
    if selected is not None and selected['kind'] != 'wait':
        check = selected['check']
        if (check['valid'] is not True or check['clock_domain'] != candidates['clock_domain'] or
                check['source_stamp_s'] != candidates['stamp_s']):
            raise ValueError('follow_reid_search_geometry_source_mismatch')
    return selected


def fresh_source(source, scene, history, now):
    record = scene.record
    if (not bound(source) or source['clock_domain'] != record['clock_domain'] or
            source['local_identity'] != [record[k] for k in ('session_id', 'generation', 'frame_id')] or
            not 0 <= now-source['stamp_s'] <= RESULT_MAX_AGE_S or
            not 0 <= now-record['as_of'] <= RESULT_MAX_AGE_S or source['stamp_s'] > record['as_of'] or
            type(source['rgb_stamp_ns']) is not int or source['rgb_stamp_s'] != source['rgb_stamp_ns']/1e9):
        raise ValueError('follow_reid_source_invalid_or_expired')
    row = next((row for row in reversed(history.rows) if row['scene_binding'] == source['scene_binding']), None)
    if (row is None or row['stamp_s'] != source['stamp_s'] or
            not 0 <= rgb_age_s(row['stamp_s'], source['rgb_stamp_s'], row['stamp_ns'], source['rgb_stamp_ns']) <= .15):
        raise ValueError('follow_reid_source_history_unavailable')


def checked_departure(loss, last, history):
    """Verify the turn's outgoing direction against its observed 3D positions."""
    proof=loss.get('departure_evidence')
    yaw=loss.get('departure_yaw_rad')
    if loss['reason']!='turned_toward_image_side' or not isinstance(proof,list) or len(proof)!=4:
        raise ValueError('follow_reid_departure_evidence_invalid')
    rows={row['scene_binding']:row for row in history.rows}
    key=tuple(last['identity'])
    if proof[-1]['scene_binding']!=last['scene_binding'] or proof[-1]['rgb_stamp_ns']!=last['rgb_stamp_ns']:
        raise ValueError('follow_reid_departure_anchor_mismatch')
    previous=None
    for item in proof:
        row=rows[item['scene_binding']]
        track=row['tracks'].get(key)
        stamp=item['rgb_stamp_ns']
        if (track is None or track[0]!=last['lifecycle_generation'] or track[1]!='observed' or
                track[2]!=item['support_stamp_s'] or row['local_identity']!=last['local_identity'] or
                row['clock_domain']!=last['clock_domain'] or
                math.dist(row['observations'][key][0],item['position_xy_m'])>1e-6 or
                type(stamp) is not int or previous is not None and stamp<=previous or
                not 0<=rgb_age_s(row['stamp_s'],stamp/1e9,row['stamp_ns'],stamp)<=.15):
            raise ValueError('follow_reid_departure_source_mismatch')
        previous=stamp
    corner=proof[1]['position_xy_m']
    position=proof[-1]['position_xy_m']
    outgoing=math.atan2(position[1]-corner[1],position[0]-corner[0])
    if type(yaw) not in (float,int) or not math.isfinite(yaw) or abs(yaw-outgoing)>1e-6:
        raise ValueError('follow_reid_departure_direction_mismatch')
