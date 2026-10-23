"""One causal observation envelope for geometry, RGB, people and visual memory."""
from dataclasses import dataclass
import copy
import hashlib

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from .context import fingerprint, layer_binding


def latest_past(rows, stamp):
    return max((row for row in rows if row['stamp_s'] <= stamp),
               key=lambda row:row['stamp_s'],default=None)


def local_identity(meta):
    return meta['session_id'],meta['generation'],meta['frame_id']


def unavailable(reason, status='missing'):
    return dict(status=status,reason=reason)


def live_scene_status(bundle, pose, now, max_age):
    """Historical scene health never overrides a newer failed/reset local estimate."""
    if bundle is None:
        return dict(ready=False,reason='waiting_for_scene',age_s=None)
    row = bundle.record
    age = now-row['as_of']
    reason = None
    if pose is None or local_identity(pose) != local_identity(row):
        reason = 'localization_lifecycle_changed_or_unavailable'
    elif pose['health'] != 0:
        reason = 'current_localization_invalid'
    elif not np.isfinite(age) or not 0 <= age <= max_age:
        reason = 'scene_future_or_stale'
    elif row['health'] != 'valid':
        reason = 'scene_modalities_incomplete'
    return dict(ready=reason is None,reason=reason,age_s=float(age) if np.isfinite(age) else None,
                observation_id=row['observation_id'],as_of=row['as_of'])


@dataclass(frozen=True)
class SceneInputs:
    record: dict
    geometry: object
    rgb: object
    prior_geometry: object = None

    def verify(self):
        record = self.record
        if fingerprint({k:v for k,v in record.items() if k != 'input_fingerprint'}) != record['input_fingerprint']:
            raise ValueError('Scene metadata changed after observation binding')
        if layer_binding(self.geometry) != record['geometry']['layer_binding']:
            raise ValueError('Scene geometry changed after observation binding')
        prior = record.get('prior_geometry',{})
        if prior.get('status') == 'valid':
            if self.prior_geometry is None or layer_binding(self.prior_geometry) != prior['layer_binding']:
                raise ValueError('Scene prior geometry changed after observation binding')
        elif self.prior_geometry is not None:
            raise ValueError('An unavailable prior cannot carry aligned layers')
        if record['rgb']['status'] == 'valid' and (self.rgb is None or self.rgb.dtype != np.uint8 or
                list(self.rgb.shape) != record['rgb']['shape'] or
                hashlib.sha256(self.rgb.tobytes()).hexdigest() != record['rgb']['sha256']):
            raise ValueError('Scene RGB no longer matches the bound observation')
        return self

    @classmethod
    def load(cls, directory):
        import json
        from pathlib import Path
        import cv2
        from .layers import GridSnapshot
        source = Path(directory)
        image = cv2.imread(str(source/'rgb.png')) if (source/'rgb.png').is_file() else None
        record = json.loads((source/'scene.json').read_text())
        prior = GridSnapshot.load(source/'prior.npz') if record.get('prior_geometry',{}).get('status') == 'valid' else None
        return cls(record,GridSnapshot.load(source/'layers.npz'),
                   None if image is None else image[:,:,::-1],prior).verify()

    def save(self, directory):
        import json
        from pathlib import Path
        import cv2
        self.verify()
        target = Path(directory)
        target.mkdir(parents=True,exist_ok=True)
        self.geometry.save(target/'layers.npz')
        if self.prior_geometry is not None:
            self.prior_geometry.save(target/'prior.npz')
        with (target/'scene.json').open('x') as stream:
            stream.write(json.dumps(self.record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        if self.rgb is not None:
            success,data = cv2.imencode('.png',self.rgb[:,:,::-1])
            if not success:
                raise ValueError('Scene RGB encoding failed')
            with (target/'rgb.png').open('xb') as stream:
                stream.write(data.tobytes())


def assemble_scene(snapshot, *, clock_domain, rgb=None, people=None, observations=(),
                   memory_status=None, settings):
    """Cutoff is the grid's acquisition time, not callback time. Never extrapolate people."""
    meta = snapshot.metadata
    stamp = float(meta['stamp_s'])
    if not np.isfinite(stamp) or not clock_domain or not meta['session_id'] or not meta['frame_id']:
        raise ValueError('A scene requires a finite cutoff, clock domain and local lifecycle')
    identity = local_identity(meta)
    localized = meta['geometry_status'] in ('OBSERVED','GROUND_UNAVAILABLE') and meta.get('world_from_lidar') is not None
    pose = validate_transform(meta['world_from_lidar']) if localized else None
    geometry = dict(status='valid' if localized else 'invalid',snapshot_id=meta['snapshot_id'],stamp_s=stamp,
        frame_id=meta['frame_id'],session_id=meta['session_id'],generation=meta['generation'],age_s=0.,
        layer_binding=layer_binding(snapshot),geometry_status=meta['geometry_status'])
    rgb_status, pixels = unavailable('no_rgb_at_or_before_cutoff'),None
    if rgb is not None:
        age = stamp-float(rgb['stamp_s'])
        if not np.isfinite(age) or age < 0 or age > settings['max_rgb_age_s']:
            rgb_status = unavailable('future_or_stale_rgb')
        else:
            values = np.asarray(rgb['pixels'])
            if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3 or not values.size:
                raise ValueError('Scene RGB must be a decoded uint8 RGB image')
            if not isinstance(rgb.get('reference'),dict) or not rgb['reference']:
                raise ValueError('Scene RGB needs an explicit observation source')
            pixels = values.copy();pixels.flags.writeable = False
            rgb_status = dict(status='valid',stamp_s=rgb['stamp_s'],age_s=age,frame_id=rgb['frame_id'],
                shape=list(values.shape),sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                reference=copy.deepcopy(rgb['reference']))
    people_status = unavailable('no_people_at_or_before_cutoff')
    if people is not None:
        age = stamp-float(people['stamp_s'])
        binding = people.get('localization_binding') or {}
        reason = None
        if not np.isfinite(age) or not 0 <= age <= settings['max_people_age_s']:
            reason = 'future_or_stale_people'
        elif people.get('clock_domain') != clock_domain:
            reason = 'people_clock_mismatch'
        elif not localized or (binding.get('session_id'),binding.get('generation'),binding.get('frame_id')) != identity:
            reason = 'people_localization_lifecycle_mismatch'
        elif people['health'] != 'valid' or people['frame_id'] != meta['frame_id']:
            reason = 'people_localization_unavailable'
        elif (people.get('pose_stamp_s') is None or not 0 <= people['stamp_s']-people['pose_stamp_s'] <= settings['max_people_pose_age_s']):
            reason = 'people_pose_time_invalid'
        if reason:
            people_status = unavailable(reason,'degraded')
        else:
            validate_transform(people['world_from_lidar'])
            # Keep the detector/tracker lifecycle separate from the local SLAM lifecycle.
            fields = ('observation_id','session_id','generation','sequence_id','frame_id','stamp_s',
                      'input_fingerprint','localization_binding','pose_stamp_s','world_from_lidar',
                      'detections','detection_frame_id','tracks','covariance_order','velocity_semantics')
            people_status = dict(status='valid',age_s=age,data=copy.deepcopy({k:people[k] for k in fields}))
    history = []
    if localized:
        for row in sorted(observations,key=lambda item:item['stamp_s'],reverse=True):
            if local_identity(row) != identity:
                continue
            age = stamp-row['stamp_s']
            if not np.isfinite(age) or not 0 <= age <= settings['history_age_s'] or row['pose_stamp_s'] > row['stamp_s']:
                continue
            validate_transform(row['world_from_lidar'])
            history.append(copy.deepcopy(row))
            if len(history) == settings['history_limit']:
                break
    association = copy.deepcopy(meta.get('association') or {})
    if association.get('valid'):
        measured = association.get('measurement_stamp_s')
        if (not localized or (association.get('local_session_id'),association.get('local_generation')) != identity[:2]
                or measured is None or not 0 <= stamp-measured <= settings['max_association_age_s']
                or association.get('stamp_s') != stamp):
            association = dict(valid=False,status='LOCAL_ONLY',reason='association_not_bound_to_scene')
        else:
            validate_transform(association['map_from_local'])
    if not association.get('valid'):
        association['map_from_local'] = None
    complete = localized and rgb_status['status'] == 'valid' and people_status['status'] == 'valid'
    record = dict(protocol='perception_scene',observation_id='scene:%s:%s:%s'%(*identity[:2],meta['snapshot_id']),
        as_of=stamp,clock_domain=clock_domain,frame_id=identity[2],session_id=identity[0],generation=identity[1],
        health='valid' if complete else 'degraded' if localized else 'invalid',
        localization=dict(status='valid' if localized else 'invalid',stamp_s=stamp,
                          source=meta.get('localization_source','unspecified'),
                          sensor_frame=meta.get('sensor_frame','unspecified'),
                          world_from_lidar=pose.tolist() if pose is not None else None),
        geometry=geometry,rgb=rgb_status,people=people_status,prior=association,
        memory=dict(status=memory_status or ('valid' if history else 'empty'),observations=history),
        units=dict(position='metres',angle='radians',time='seconds'),
        scope='causal_scene_with_optional_prior_and_historical_evidence',navigation_reachability_verified=False)
    record['input_fingerprint'] = fingerprint(record)
    return SceneInputs(record,snapshot,pixels)


def snapshot_from_message(message):
    """Reconstruct the exact published layers; metadata and wire header must agree."""
    import json
    from .layers import GridSnapshot
    meta = json.loads(message.metadata_json)
    if (meta['snapshot_id'] != message.snapshot_id or local_identity(meta) !=
            (message.session_id,message.generation,message.header.frame_id) or
            abs(meta['stamp_s']-message.header.stamp.to_sec()) > 5e-7 or
            meta['shape'] != [message.info.height,message.info.width] or
            abs(meta['resolution']-message.info.resolution) > 1e-7 or
            not np.allclose(meta['origin_xy'],[message.info.origin.position.x,message.info.origin.position.y])):
        raise ValueError('Grid wire metadata and observation binding disagree')
    names = ('occupancy','ground_z','ground_support','observed','last_observed','coverage',
             'obstacle_min_z','obstacle_max_z','clearance','unknown_distance','frontier')
    dtypes = dict(occupancy=np.int8,ground_support=bool,observed=bool,frontier=bool,last_observed=np.float64)
    layers = {}
    for name in names:
        values = getattr(message,name)
        # rospy deserializes uint8[] as bytes, while other array fields are tuples.
        if isinstance(values,(bytes,bytearray,memoryview)):
            values = np.frombuffer(values,dtype=np.uint8)
        array = np.asarray(values,dtype=dtypes.get(name,np.float32)).reshape(meta['shape'])
        array.flags.writeable = False
        layers[name] = array
    return GridSnapshot(meta,layers)
