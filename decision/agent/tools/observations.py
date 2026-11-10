"""Causal bounded Agent observations, independent of local T1/T2/T3 windows."""
from dataclasses import dataclass
import copy
import base64
import binascii
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import zlib
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
    def from_scene(cls, scene, *, rgb=None, source=None, geometry=None, sensors=None, include_local_grid=False):
        from navigation.planning.footprint import VehicleGeometry
        from .. import ROOT
        scene.verify()
        s = scene.record
        if geometry is None:
            geometry = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
            geometry_reference = 'historical_assumed_not_current_calibration'
        else:
            geometry_reference = dict(kind='runtime_vehicle_geometry',assumed=geometry.assumed,source=copy.deepcopy(geometry.source))
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
                 pose_reference='base_link', geometry_reference=geometry_reference,
                 map_binding={k: copy.deepcopy(prior.get(k)) for k in ('valid', 'map_id', 'map_frame', 'map_revision', 'association_session', 'association_generation')},
                 geometry=dict(status=s['geometry']['status'], stamp_s=s['geometry']['stamp_s'],
                               free_cells=int(np.count_nonzero(cells == 0)), occupied_cells=int(np.count_nonzero(cells == 100)),
                               unknown_cells=int(np.count_nonzero(cells == -1))),
                 people=dict(status=s['people']['status'], count=(len(s['people']['data']['tracks']) if s['people']['status'] == 'valid' else None)),
                 rgb=bound_rgb, source=copy.deepcopy(source), scope='observed_input_not_execution_feedback')
        if sensors is not None:
            r['sensors'] = copy.deepcopy(sensors)
        if include_local_grid and s['geometry']['status'] == 'valid' and prior.get('valid'):
            r['local_grid'] = local_grid_record(scene)
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
        if 'sensors' in self.record:
            value['sensors'] = copy.deepcopy(self.record['sensors'])
        return value

    def local_grid(self):
        """Decode only the bound occupancy projection, never an arbitrary scene."""
        self.verify()
        item = self.record.get('local_grid')
        if item is None:
            return None
        from perception.environment.layers import GridSnapshot
        shape = item['metadata']['shape']
        if len(shape) != 2 or any(type(n) is not int or not 0 < n <= 600 for n in shape):
            raise ValueError('Invalid local Agent grid size')
        metadata=item['metadata']
        if (not isinstance(metadata.get('frame_id'),str) or not metadata['frame_id'] or
                not isinstance(metadata.get('snapshot_id'),str) or not metadata['snapshot_id'] or
                np.shape(metadata.get('origin_xy')) != (2,) or
                not np.isfinite(metadata['origin_xy']).all() or
                not isinstance(metadata.get('resolution'),(int,float)) or
                not np.isfinite(metadata['resolution']) or metadata['resolution'] <= 0):
            raise ValueError('Invalid local Agent grid geometry')
        try:
            data = base64.b64decode(item['occupancy_zlib_base64'],validate=True)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(data,shape[0]*shape[1]+1)
        except (binascii.Error,zlib.error) as error:
            raise ValueError('Invalid local Agent grid encoding') from error
        if len(raw) != shape[0]*shape[1] or not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid local Agent grid payload')
        cells = np.frombuffer(raw,dtype=np.int8).reshape(shape)
        if not np.isin(cells,[-1,0,100]).all():
            raise ValueError('Invalid local occupancy values')
        return GridSnapshot(copy.deepcopy(item['metadata']),dict(occupancy=cells))


def local_grid_record(scene):
    """Capture a bounded, immutable occupancy-only projection of a sealed scene."""
    grid = scene.geometry
    cells = np.asarray(grid.layers['occupancy'])
    if cells.dtype != np.int8 or cells.ndim != 2 or max(cells.shape) > 600:
        raise ValueError('Invalid Agent local occupancy source')
    metadata = {k:copy.deepcopy(grid.metadata[k]) for k in ('frame_id','shape','resolution','origin_xy','snapshot_id')}
    if list(cells.shape) != metadata['shape']:
        raise ValueError('Agent local occupancy shape mismatch')
    encoded=base64.b64encode(zlib.compress(cells.tobytes(),1)).decode('ascii')
    if len(encoded) > 500000:
        raise ValueError('Agent local occupancy exceeds IPC budget')
    return dict(metadata=metadata,occupancy_zlib_base64=encoded,
                map_from_local=copy.deepcopy(scene.record['prior'].get('map_from_local')),
                source_scene_binding=scene.record['input_fingerprint'])


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
