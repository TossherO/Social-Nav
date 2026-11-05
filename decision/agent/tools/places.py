"""Named destinations; semantic approval never implies geometric reachability."""
import copy
import hashlib
import json
import math
from pathlib import Path

from .. import ROOT


class Catalog:
    def __init__(self, value):
        self.value = copy.deepcopy(value)
        self.fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.places = {p['place_id']: p for p in value['places']}
        if len(self.places) != len(value['places']):
            raise ValueError('Duplicate place identity')
        for p in self.places.values():
            pose = p.get('pose')
            if pose is not None and (len(pose) not in (2, 3) or not all(math.isfinite(v) for v in pose)):
                raise ValueError('Invalid destination pose')
            if p.get('reference') != 'base_link':
                raise ValueError('Destination must reference base_link')

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.get('asset'):
            asset = ROOT/value['asset']
            if asset.exists():
                bound = json.loads(asset.read_text())
                if {p['place_id'] for p in bound['places']} != {p['place_id'] for p in value['places']}:
                    raise ValueError('Semantic asset does not match registered place identities')
                value = bound
        return cls(value)

    def lookup(self, name=''):
        return [copy.deepcopy(p) for p in self.places.values()
                if not name or name == p['place_id'] or name in [p['name'], *p.get('aliases', [])]]

    def bind_start(self, navigation, task_id):
        """Task-local copy; never write a measured origin into shared assets."""
        value = copy.deepcopy(self.value)
        origin = next((p for p in value['places'] if p['place_id'] == 'mission_start'), None)
        if origin is None:
            return Catalog(value)
        pose = navigation.get('origin_pose')
        if (pose is not None and len(pose) == 3 and all(math.isfinite(v) for v in pose) and
                navigation.get('origin_frame_id') == value['frame_id'] and
                0 <= navigation['stamp_s']-navigation.get('origin_stamp_s', navigation['stamp_s']) <= .5 and
                navigation.get('localization_status', 'valid') == 'valid' and not navigation.get('input_stale')):
            if value['scope'] == 'simulation':
                # The synthetic return fixture has an explicit approach yaw;
                # measured start XY and development heading are distinct.
                yaw = origin['pose'][2] if origin.get('pose') else pose[2]
                origin.update(pose=[*pose[:2], yaw])
            else:
                origin.update(pose=list(pose), status='task_start_bound', allow_shadow=True)
            origin['task_origin'] = dict(task_id=task_id, source=navigation['source'], stamp_s=navigation.get('origin_stamp_s', navigation['stamp_s']),
                clock_domain=navigation['clock_domain'], local_identity=[navigation.get(k) for k in ('session_id', 'generation', 'frame_id')],
                map_binding=copy.deepcopy(navigation.get('map_binding')), measured_pose=list(pose))
        return Catalog(value)

    def destination(self, place_id, scope):
        p = self.places[place_id]
        synthetic = self.value['scope'] == scope == 'simulation' and p.get('status') == 'simulation'
        shadow = scope == 'shadow' and self.value['scope'] == 'development_prior' and p.get('allow_shadow') is True
        if not synthetic and not shadow:
            raise ValueError('Destination is neither synthetic nor an explicit nonexecuting shadow candidate')
        if p.get('pose') is None or p.get('room_interior', False):
            raise ValueError('Unbound or excluded destination')
        return copy.deepcopy(p)

    def prompt(self):
        return dict(scope=self.value['scope'], floor_id=self.value['floor_id'],
                    places=[{k: p[k] for k in ('place_id', 'name', 'aliases', 'status')}
                            for p in self.places.values()], access=self.value['access'])
