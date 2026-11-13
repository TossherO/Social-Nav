"""First-follow candidates and immutable visual references; never commands motion."""
import copy
import hashlib
import io
import math

import numpy as np
from PIL import Image, ImageDraw

from perception.environment.context import fingerprint


def candidates(scene, pixels, rgb_stamp_s, detections, association, *, detector, rgb_stamp_ns=None):
    """Bind every 2D candidate to one scene and, only when justified, a 3D track."""
    from navigation.planning.people import stable_identity
    from perception.detection_tracking.inference.rgb_association import identity
    scene.verify()
    source = scene.record
    rgb = np.asarray(pixels)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or
            not math.isfinite(rgb_stamp_s) or not 0 <= source['as_of']-rgb_stamp_s <= .15):
        raise ValueError('Follow candidates require a causal RGB observation')
    if rgb_stamp_ns is not None and (type(rgb_stamp_ns) is not int or abs(rgb_stamp_ns/1e9-rgb_stamp_s) > 1e-6):
        raise ValueError('RGB nanoseconds do not match the source timestamp')
    bound_rgb = source.get('rgb') or {}
    pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    if bound_rgb.get('status') == 'valid' and (bound_rgb.get('sha256') != pixel_hash or
            bound_rgb.get('stamp_s') != rgb_stamp_s):
        raise ValueError('RGB differs from the scene-bound image')
    if not isinstance(detections, list) or len(detections) > 64:
        raise ValueError('Follow candidate count exceeds the bounded input')
    height, width = rgb.shape[:2]
    people = source.get('people') or {}
    row = people.get('data') if people.get('status') == 'valid' else None
    if row is not None and not 0 <= source['as_of']-row['stamp_s'] <= .3:
        row = None
    identity_error = False
    try:
        tracks = {} if row is None else {identity(t):t for t in row.get('tracks', [])}
        if row is not None and len(tracks) != len(row.get('tracks', [])):
            identity_error = True
    except (KeyError, TypeError):
        tracks, identity_error = {}, True
    if identity_error:
        row, tracks = None, {}
    matched = {}
    if association and association.get('status') == 'complete' and row is not None and tracks:
        if set(association.get('tracks', {})) != set(tracks):
            raise ValueError('Association and scene tracks differ')
        used = set()
        for key, value in association['tracks'].items():
            if value.get('status') != 'matched':
                continue
            index = value.get('detection_index')
            if type(index) is not int or not 0 <= index < len(detections) or index in used:
                raise ValueError('Image detection association is invalid or duplicated')
            used.add(index)
            if not np.allclose(value.get('rgb_box'), detections[index].get('box'), rtol=0, atol=1e-4):
                raise ValueError('Associated box differs from the original detection')
            track = tracks[key]
            last = track.get('last_observed_stamp_s')
            support_source = 'last_observed_stamp_s'
            if last is None and track.get('state') == 'observed':
                # Legacy records explicitly mark an observation in this people
                # frame but omit a separate per-track support timestamp.
                last = row['stamp_s']
                support_source = 'observed_state_at_people_stamp'
            if (track.get('confirmed') and track.get('state') == 'observed' and
                    type(last) in (int, float) and math.isfinite(last) and
                    0 <= source['as_of']-last <= .3):
                matched[index] = dict(identity=list(stable_identity(row, track)),
                    lifecycle_generation=track['lifecycle_generation'], track_key=key,
                    people_stamp_s=row['stamp_s'], people_fingerprint=row['input_fingerprint'],
                    support_stamp_s=last, support_time_source=support_source)
    output = []
    for index, item in enumerate(detections):
        box = np.asarray(item.get('box'), dtype=float)
        score = item.get('score')
        if (item.get('class_id') != 0 or box.shape != (4,) or not np.isfinite(box).all() or
                type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1 or
                not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError('Invalid image person detection')
        output.append(dict(id=index, box=box.tolist(), score=float(score), track=matched.get(index)))
    value = dict(observation_id=source['observation_id'], stamp_s=source['as_of'],
        clock_domain=source['clock_domain'], local_identity=[source[k] for k in ('session_id','generation','frame_id')],
        scene_binding=source['input_fingerprint'], rgb_stamp_s=rgb_stamp_s, rgb_stamp_ns=rgb_stamp_ns,
        rgb_pixel_sha256=pixel_hash, image_size=[width,height],
        detector=copy.deepcopy(detector), association_status=('track_identity_unavailable' if identity_error else
            (association or {}).get('status', 'unavailable')),
        association_binding=None if association is None else fingerprint(association),
        candidates=output)
    value['binding'] = fingerprint(value)
    return value


def verify_candidates(value, observation):
    if fingerprint({k:v for k,v in value.items() if k != 'binding'}) != value.get('binding'):
        raise ValueError('Follow candidates changed')
    r = observation.record
    if (value['observation_id'] != r['observation_id'] or value['scene_binding'] != r['scene_binding'] or
            value['clock_domain'] != r['clock_domain'] or value['local_identity'] != r['local_identity'] or
            value['stamp_s'] != r['stamp_s'] or value['rgb_stamp_s'] != r['rgb']['stamp_s'] or
            value['rgb_stamp_ns'] != r['rgb'].get('stamp_ns') or
            value['rgb_pixel_sha256'] != r['rgb']['pixel_sha256']):
        raise ValueError('Follow candidates do not belong to this RGB and scene')
    return value


def annotated(observation):
    observation.verify()
    value = verify_candidates(observation.record['follow_candidates'], observation)
    with Image.open(io.BytesIO(observation.image)) as raw:
        image = raw.convert('RGB')
    draw = ImageDraw.Draw(image)
    from .scene_inputs import font
    label_font = font(max(16, round(image.width*.025)))
    for item in value['candidates']:
        draw.rectangle(item['box'], outline='cyan', width=3)
        x, y = item['box'][:2]
        draw.text((x+2, max(0,y-22)), str(item['id']), font=label_font, fill='yellow', stroke_width=2, stroke_fill='black')
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def reference(observation, candidate_id, description, *, task_id, revision, goal_id):
    observation.verify()
    value = verify_candidates(observation.record['follow_candidates'], observation)
    if type(candidate_id) is not int or not isinstance(description, str) or not description.strip():
        raise ValueError('Reference requires a candidate and an appearance description')
    selected = next((x for x in value['candidates'] if x['id'] == candidate_id), None)
    if selected is None:
        raise ValueError('Candidate is not in this observation')
    with Image.open(io.BytesIO(observation.image)) as raw:
        image = raw.convert('RGB')
        box = selected['box']
        crop = image.crop((math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])))
        out = io.BytesIO()
        crop.save(out, format='PNG')
    image, crop = bytes(observation.image), out.getvalue()
    record = dict(task_id=task_id, task_revision=revision, goal_id=goal_id, description=description.strip(),
        observation_id=value['observation_id'], scene_binding=value['scene_binding'], candidate_binding=value['binding'],
        clock_domain=value['clock_domain'], local_identity=value['local_identity'], stamp_s=value['stamp_s'],
        rgb_stamp_s=value['rgb_stamp_s'], rgb_stamp_ns=value['rgb_stamp_ns'], candidate_id=candidate_id, box=copy.deepcopy(selected['box']),
        detector=copy.deepcopy(value['detector']), association_status=value['association_status'],
        association_binding=value['association_binding'], candidate_score=selected['score'],
        track=copy.deepcopy(selected['track']), image_sha256=hashlib.sha256(image).hexdigest(),
        crop_sha256=hashlib.sha256(crop).hexdigest(), historical_only=True, executable=False)
    record['reference_id'] = fingerprint(record)
    return record, image, crop
