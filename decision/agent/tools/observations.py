"""Causal bounded Agent observations, independent of local T1/T2/T3 windows."""
from dataclasses import dataclass
import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image
from perception.environment.context import fingerprint


@dataclass(frozen=True)
class Observation:
    record: dict
    image: bytes | None

    def verify(self):
        r = self.record
        if fingerprint({k: v for k, v in r.items() if k != 'input_fingerprint'}) != r['input_fingerprint']:
            raise ValueError('Agent observation metadata changed')
        rgb = r['rgb']
        if rgb['status'] == 'valid':
            if self.image is None or hashlib.sha256(self.image).hexdigest() != rgb['encoded_sha256']:
                raise ValueError('Agent RGB asset changed')
            if not 0 <= r['stamp_s']-rgb['stamp_s'] <= .15:
                raise ValueError('Future or stale RGB')
        elif self.image is not None:
            raise ValueError('Unavailable RGB cannot carry an image')
        return self

    @classmethod
    def from_scene(cls, scene, *, rgb=None, source=None):
        from navigation.planning.footprint import VehicleGeometry
        from .. import ROOT
        scene.verify()
        s = scene.record
        geometry = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
        pose = s['localization'].get('world_from_lidar')
        local = None if pose is None else geometry.pose(pose).tolist()
        prior = s['prior']
        mapped = None
        if prior.get('valid') and pose is not None:
            mapped = geometry.pose(np.asarray(prior['map_from_local'])@np.asarray(pose)).tolist()
        image = None
        bound_rgb = dict(status='missing', reason='no_causal_rgb')
        if rgb is None and s['rgb']['status'] == 'valid':
            rgb = dict(pixels=scene.rgb, stamp_s=s['rgb']['stamp_s'], frame_id=s['rgb']['frame_id'], source=s['rgb'].get('reference'))
        if rgb is not None:
            age = s['as_of']-rgb['stamp_s']
            pixels = np.asarray(rgb['pixels'])
            if not 0 <= age <= .15 or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError('RGB must be causal uint8 pixels from this observation')
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='PNG')
            image = stream.getvalue()
            bound_rgb = dict(status='valid', stamp_s=rgb['stamp_s'], age_s=age, frame_id=rgb['frame_id'],
                             pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), shape=list(pixels.shape),
                             encoded_sha256=hashlib.sha256(image).hexdigest(), source=rgb.get('source'))
        cells = scene.geometry.layers['occupancy']
        r = dict(observation_id=s['observation_id'], stamp_s=s['as_of'], clock_domain=s['clock_domain'],
                 local_identity=[s[k] for k in ('session_id', 'generation', 'frame_id')], scene_binding=s['input_fingerprint'],
                 localization_status=s['localization']['status'], base_pose=local, map_base_pose=mapped,
                 pose_reference='base_link', geometry_reference='historical_assumed_not_current_calibration',
                 map_binding={k: copy.deepcopy(prior.get(k)) for k in ('valid', 'map_id', 'map_frame', 'map_revision', 'association_session', 'association_generation')},
                 geometry=dict(status=s['geometry']['status'], stamp_s=s['geometry']['stamp_s'],
                               free_cells=int(np.count_nonzero(cells == 0)), occupied_cells=int(np.count_nonzero(cells == 100)),
                               unknown_cells=int(np.count_nonzero(cells == -1))),
                 people=dict(status=s['people']['status'], count=(len(s['people']['data']['tracks']) if s['people']['status'] == 'valid' else None)),
                 rgb=bound_rgb, source=copy.deepcopy(source), scope='observed_input_not_execution_feedback')
        r['input_fingerprint'] = fingerprint(r)
        return cls(r, image).verify()

    def prompt(self):
        self.verify()
        # Source filenames, development expectations and offline labels are not
        # model inputs. Observation identity, source time and geometry remain.
        value = {k: copy.deepcopy(self.record[k]) for k in (
            'observation_id', 'stamp_s', 'clock_domain', 'local_identity', 'localization_status', 'base_pose',
            'map_base_pose', 'pose_reference', 'geometry_reference', 'map_binding', 'geometry', 'people', 'scope')}
        # Early frozen records may have count=0 beside status=missing. Preserve
        # their source hashes, but never interpret missing tracks as no people.
        if value['people']['status'] != 'valid':
            value['people']['count'] = None
        value['rgb'] = {k:copy.deepcopy(self.record['rgb'][k]) for k in
            ('status','stamp_s','age_s','frame_id','shape','reason') if k in self.record['rgb']}
        return value


class ObservationSet:
    def __init__(self, path, *, create=False):
        path = Path(path)
        if create:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path)
            self.db.execute('CREATE TABLE observations (seq INTEGER PRIMARY KEY, record TEXT NOT NULL, image BLOB)')
            self.db.commit()
        else:
            self.db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)

    def add(self, observation):
        observation.verify()
        with self.db:
            self.db.execute('INSERT INTO observations(record,image) VALUES (?,?)',
                            (json.dumps(observation.record, ensure_ascii=False, allow_nan=False), observation.image))

    def read(self, seq):
        row = self.db.execute('SELECT record,image FROM observations WHERE seq=?', (seq,)).fetchone()
        if row is None:
            raise KeyError(seq)
        return Observation(json.loads(row[0]), row[1]).verify()

    def sequences(self):
        return [row[0] for row in self.db.execute('SELECT seq FROM observations ORDER BY seq')]

    def close(self):
        self.db.close()
