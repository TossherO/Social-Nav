"""Causal one-to-one image/3D association; never creates or updates 3D tracks."""
import copy
import math
from types import SimpleNamespace

import numpy as np

from perception.sensors.image_geometry import project_box, project_point, sensor_box


def identity(track):
    return track['track_id']+'/g'+str(track['lifecycle_generation'])


def people_in_lidar(row):
    """Keep physical track identities while transforming valid world boxes to the scan."""
    result = []
    pose = row.get('world_from_lidar')
    if row.get('health') == row.get('pose_health') == 'valid' and pose is not None:
        matrix = np.asarray(pose,dtype=float)
        if (matrix.shape != (4,4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3],[0,0,0,1]) or
                not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-4)):
            raise ValueError('无效定位变换')
        inverse = np.linalg.inv(matrix)
        for track in row.get('tracks',[]):
            result.append(dict(key=identity(track),kind='track',box=sensor_box(track['box'],inverse),
                               observed=track['state']=='observed',confirmed=bool(track['confirmed'])))
    if not result:
        result = [dict(key=row['observation_id']+'/d'+str(i),kind='detection',box=box,
                       observed=True,confirmed=False) for i,box in enumerate(row.get('detections',[]))]
    return result


def iou(a, b):
    low, high = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    overlap = np.maximum(0., high-low).prod()
    area = np.prod(np.asarray(a[2:])-a[:2])+np.prod(np.asarray(b[2:])-b[:2])-overlap
    return float(overlap/area) if area > 0 else 0.


def causal_geometry(people, current, history, image_stamp_s, max_pose_age_s):
    """Use only an earlier reliable pose/track; otherwise retain explicit approximation."""
    result = copy.deepcopy(people)
    scope = ('sequence_id','session_id','generation','clock_domain','frame_id','pose_source')
    candidates = [row for row in history if row['stamp_s'] <= image_stamp_s and
                  row.get('health') == row.get('pose_health') == 'valid' and
                  row.get('pose_source') in ('localization','dataset_pose','stationary_assumption') and
                  row.get('pose_stamp_s') is not None and
                  0 <= image_stamp_s-row['pose_stamp_s'] <= max_pose_age_s and
                  all(row.get(key) == current.get(key) for key in scope) and
                  row.get('localization_binding') == current.get('localization_binding')]
    if not candidates:
        return result
    past = max(candidates, key=lambda row: row['stamp_s'])
    matrix = np.asarray(past['world_from_lidar'],dtype=float)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all():
        return result
    inverse = np.linalg.inv(matrix)
    tracks = {identity(track):track for track in past.get('tracks',[])}
    for person in result:
        track = tracks.get(person['key'])
        if not track or not track.get('confirmed') or track.get('velocity_xyz_mps') is None:
            continue
        velocity = np.asarray(track['velocity_xyz_mps'],dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            continue
        box = np.asarray(track['box'],dtype=float).copy()
        box[:3] += velocity*(image_stamp_s-past['stamp_s'])
        person['box'] = sensor_box(box,inverse)
        person['compensation'] = 'causal_track_prediction'
        person['geometry_stamp_s'] = past['stamp_s']
        person['pose_age_s'] = image_stamp_s-past['pose_stamp_s']
    return result


class ImageAssociator:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        for name in ('max_time_delta_s','max_uncompensated_delta_s','center_gate','cost_max',
                     'ambiguity_margin','overlap_weight','history_weight','history_gap_s'):
            if not isinstance(config[name], (int,float)) or not math.isfinite(config[name]) or config[name] < 0:
                raise ValueError('Invalid image association parameter: '+name)
        self.previous, self.scope, self.stamp = {}, None, None

    def step(self, people, detections, transform, camera, delta_s, stamp_s, scope, health='valid'):
        from scipy.optimize import linear_sum_assignment
        if not math.isfinite(stamp_s) or delta_s is not None and not math.isfinite(delta_s):
            raise ValueError('Image association timestamps must be finite')
        c = self.config
        if (scope != self.scope or self.stamp is None or stamp_s <= self.stamp or
                stamp_s-self.stamp > c['history_gap_s']):
            self.previous = {}
        self.scope, self.stamp = scope, stamp_s
        records = {p['key']:dict(status='unmatched',rgb_box=None,compensation=p.get('compensation','uncompensated'))
                   for p in people}
        result = dict(status='complete',tracks=records,delta_s=delta_s,matches=0)
        reason = ('pose_unavailable' if health != 'valid' else 'no_rgb' if delta_s is None else 'no_camera' if camera is None
                  else 'time_skew' if not math.isfinite(delta_s) or not 0 <= delta_s <= c['max_time_delta_s'] else None)
        if reason:
            result['status'] = reason
            for record in records.values():record['status'] = reason
            self.previous = {}
            return result
        info = SimpleNamespace(**camera)
        if (info.distortion_model not in ('plumb_bob','rational_polynomial','') or
                np.asarray(info.K).size != 9 or not np.isfinite(info.K).all() or
                not np.isfinite(info.D).all() or info.K[0] <= 0 or info.K[4] <= 0):
            raise ValueError('Invalid camera model for image association')
        boxes = []
        for detection in detections:
            box = np.asarray(detection['box'],dtype=float)
            if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]) or
                    detection.get('class_id') != 0 or not 0 <= detection['score'] <= 1):
                raise ValueError('Invalid image detection box')
            boxes.append(box)
        used = set()
        new_history = {}
        # Observed tracks get first choice; predictions cannot steal their image evidence.
        for observed in (True, False):
            candidates, proposals = [], []
            for person in people:
                record = records[person['key']]
                if person.get('kind') != 'track' or not person.get('confirmed'):
                    record['status'] = 'unconfirmed'
                    continue
                if bool(person.get('observed')) != observed:continue
                if record['compensation'] == 'uncompensated' and delta_s > c['max_uncompensated_delta_s']:
                    record['status'] = 'time_skew'
                    continue
                center = project_point(person['box'][:3],transform,info)
                if center is None:
                    record['status'] = 'no_projection'
                    continue
                coarse = project_box(person['box'],transform,info,[info.width,info.height])
                record['proposal_center'] = center
                record['proposal_box'] = coarse
                candidates.append(person)
                proposals.append((center,coarse))
            available = sorted(set(range(len(boxes)))-used)
            if not candidates or not available:continue
            costs = np.full((len(candidates),len(available)),np.inf)
            for i,(person,(center,coarse)) in enumerate(zip(candidates,proposals)):
                for j,index in enumerate(available):
                    box = boxes[index]
                    width,height = box[2:]-box[:2]
                    dx,dy = (np.asarray(center)-(box[:2]+box[2:])/2)/[width/2,height/2]
                    distance = math.hypot(dx,.35*dy)
                    if abs(dy) > 1.8 or distance > c['center_gate']:continue
                    cost = distance + c['overlap_weight']*(1-iou(coarse,box) if coarse else 1.)
                    old = self.previous.get(person['key'])
                    if old is not None:
                        shift = np.linalg.norm(((box[:2]+box[2:])/2-np.asarray(old))/[width,height])
                        cost += c['history_weight']*min(1.,float(shift))
                    if cost <= c['cost_max']:costs[i,j] = cost
            # Reject ambiguous rows AND competing columns before assignment.
            rejected_rows, rejected_cols = set(),set()
            for axis,rejected in ((1,rejected_rows),(0,rejected_cols)):
                values = costs if axis == 1 else costs.T
                for index,line in enumerate(values):
                    valid = np.sort(line[np.isfinite(line)])
                    if len(valid)>1 and valid[1]-valid[0] < c['ambiguity_margin']:rejected.add(index)
            for i in rejected_rows:
                records[candidates[i]['key']]['status'] = 'ambiguous'
            for j in rejected_cols:
                for i in np.flatnonzero(np.isfinite(costs[:,j])):
                    records[candidates[i]['key']]['status'] = 'ambiguous'
            clean = costs.copy()
            for i in rejected_rows:clean[i,:] = np.inf
            for j in rejected_cols:clean[:,j] = np.inf
            augmented = np.c_[np.where(np.isfinite(clean),clean,1e6),
                              np.full((len(candidates),len(candidates)),c['cost_max']+1e-6)]
            rows,cols = linear_sum_assignment(augmented)
            for i,j in zip(rows,cols):
                if j >= len(available) or not np.isfinite(clean[i,j]):continue
                index = available[j]
                key = candidates[i]['key']
                alternatives = np.r_[np.delete(costs[i],j),np.delete(costs[:,j],i)]
                finite = alternatives[np.isfinite(alternatives)]
                margin = float(finite.min()-clean[i,j]) if len(finite) else None
                # A global assignment can hand a track its second choice after another
                # track wins the best box. A finite assignment alone is not evidence.
                if margin is not None and margin < c['ambiguity_margin']:
                    records[key]['status'] = 'ambiguous'
                    continue
                records[key].update(status='matched',rgb_box=boxes[index].tolist(),detection_index=index,
                                    detection_score=float(detections[index]['score']),cost=float(clean[i,j]),
                                    margin=margin)
                used.add(index)
                new_history[key] = ((boxes[index][:2]+boxes[index][2:])/2).tolist()
                result['matches'] += 1
        self.previous = new_history
        result['unmatched_detection_count'] = len(detections)-len(used)
        return result
