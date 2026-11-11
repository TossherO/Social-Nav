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
        # Legacy object annotations become ordinary named points in memory;
        # the source catalog is never edited and identities stay unchanged.
        for landmark in self.value.pop('landmarks', []):
            item=copy.deepcopy(landmark)
            item['place_id']=item.pop('landmark_id')
            item['pose']=item.pop('position')
            item.setdefault('reference','base_link')
            item.setdefault('status','manual_annotation')
            self.value['places'].append(item)
        value=self.value
        self.fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.places = {p['place_id']: p for p in value['places']}
        self.condition = value.get('map_condition', 'complete')
        if self.condition not in ('complete', 'simple', 'none'):
            raise ValueError('Unknown map condition')
        if self.condition == 'complete':
            if not isinstance(value.get('frame_id'),str) or not value['frame_id']:
                raise ValueError('Complete map requires a coordinate frame')
            units = value.get('units',{})
            if units.get('position','metres')!='metres' or units.get('angle','radians')!='radians':
                raise ValueError('Complete-map annotations use metres and radians')
        self.regions = {p['region_id']: p for p in value.get('regions', [])}
        if len(self.regions)!=len(value.get('regions',[])):
            raise ValueError('Duplicate map annotation identity')
        ids = [*self.places, *self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError('Map annotation identities must be unique')
        if any(p.get('frame_id',value['frame_id'])!=value['frame_id'] for p in
               [*self.places.values(),*self.regions.values()]):
            raise ValueError('Annotations must use the catalog coordinate frame')
        for region in self.regions.values():
            polygon = region['polygon']
            if len(polygon) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in polygon):
                raise ValueError('Region requires a finite XY polygon')
            area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))
            if abs(area) <= 1e-10:
                raise ValueError('Region has zero signed area')
        if len(self.places) != len(value['places']):
            raise ValueError('Duplicate place identity')
        for p in self.places.values():
            pose = p.get('pose')
            if pose is not None and (len(pose) not in (2, 3) or not all(math.isfinite(v) for v in pose)):
                raise ValueError('Invalid destination pose')
            if p.get('reference') != 'base_link':
                raise ValueError('Destination must reference base_link')

    @classmethod
    def load(cls, path, *, condition=None):
        value = json.loads(Path(path).read_text())
        selected = condition or value.get('map_condition', 'complete')
        if selected == 'complete' and value.get('asset'):
            asset = ROOT/value['asset']
            if asset.exists():
                bound = json.loads(asset.read_text())
                if {p['place_id'] for p in bound['places']} != {p['place_id'] for p in value['places']}:
                    raise ValueError('Semantic asset does not match registered place identities')
                for key in ('regions','landmarks'):
                    if key in value:
                        bound[key] = copy.deepcopy(value[key])
                value = bound
        value['map_condition'] = selected
        if selected == 'none':
            value = dict(scope=value['scope'], floor_id=None, frame_id=None, map_condition='none',
                         places=[], regions=[], access={})
        return cls(value)

    def require_complete(self):
        if self.condition != 'complete':
            raise ValueError('map_condition_not_implemented:'+self.condition)

    def annotations(self):
        if self.condition != 'complete':
            return dict(points=[], regions=[])
        fields = ('place_id', 'name', 'aliases', 'pose', 'reference', 'status', 'description')
        return dict(points=[dict({k:p[k] for k in fields if k in p}, display_id='P'+str(i+1)) for i,p in enumerate(self.places.values())],
                    regions=[dict(copy.deepcopy(p),display_id='R'+str(i+1)) for i,p in enumerate(self.regions.values())])

    def lookup(self, name=''):
        if self.condition != 'complete':
            return []
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
        live = scope in ('live','manual') and self.value['scope'] == 'development_prior' and p.get('status') in ('manual_annotation','task_start_bound')
        if not synthetic and not shadow and not live:
            raise ValueError('Destination is neither synthetic nor an explicit nonexecuting shadow candidate')
        if p.get('pose') is None or p.get('room_interior', False):
            raise ValueError('Unbound or excluded destination')
        return copy.deepcopy(p)

    def prompt(self):
        metric = self.condition == 'complete'
        return dict(scope=self.value['scope'], map_condition=self.condition, floor_id=self.value['floor_id'],
                    frame_id=self.value['frame_id'] if metric else None,
                    units=dict(position='metres' if metric else 'unspecified', angle='radians'),
                    execution_supported=metric, annotations=self.annotations(),
                    access=self.value['access'])
