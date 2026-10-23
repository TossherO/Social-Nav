"""Portable final environment packages and read-only, time-bounded queries."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import yaml

from perception.mapping_localization.results import file_fingerprint
from perception.sensors.image import rgb_array
from .context import crop_map, query_geometry
from .layers import GridSnapshot
from .memory import read_index, query_observations, query_places, add_place


ROOT = Path(__file__).resolve().parents[2]


def asset_reference(path):
    path = Path(path).expanduser().absolute()
    try:
        return dict(base='project_root', path=str(path.relative_to(ROOT)))
    except ValueError:
        return dict(base='absolute', path=str(path))


def resolve_asset(reference):
    if reference['base'] == 'project_root':
        return ROOT/reference['path']
    if reference['base'] == 'absolute':
        return Path(reference['path'])
    raise ValueError('Unsupported asset reference base')


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_navigation(snapshot, directory):
    """Standard map_server orientation, unknown=205; no footprint inflation."""
    directory = Path(directory)
    paths = [directory/'map.pgm', directory/'map.yaml']
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite a navigation export')
    cells = snapshot.layers['occupancy']
    if not np.isin(cells, [-1, 0, 100]).all():
        raise ValueError('Unknown occupancy code')
    pixels = np.where(cells == -1, 205, np.where(cells == 100, 0, 254)).astype(np.uint8)
    height, width = pixels.shape
    with paths[0].open('xb') as stream:
        stream.write(('P5\n%d %d\n255\n' % (width, height)).encode('ascii'))
        stream.write(np.flipud(pixels).tobytes())
    data = dict(image='map.pgm', resolution=snapshot.metadata['resolution'],
                origin=[*snapshot.metadata['origin_xy'], 0.0], negate=0,
                occupied_thresh=.65, free_thresh=.196)
    with paths[1].open('x') as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def observation_image(db, binding, row, *, bag_override=None):
    if row.get('image_storage') == 'embedded_png':
        import cv2
        entry = db.execute('SELECT sha256,data FROM image_assets WHERE observation_id=?', (row['id'],)).fetchone()
        if entry is None or hashlib.sha256(entry[1]).hexdigest() != entry[0]:
            raise ValueError('Stored live RGB asset is absent or changed')
        image = cv2.imdecode(np.frombuffer(entry[1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid stored PNG')
        rgb = image[:, :, ::-1].copy()
        if hashlib.sha256(rgb.tobytes()).hexdigest() != row['decoded_rgb_sha256']:
            raise ValueError('Stored RGB no longer matches its observation')
        return rgb
    bag = Path(bag_override) if bag_override else resolve_asset(binding['bag'])
    if file_fingerprint(bag) != binding['bag_fingerprint']:
        raise ValueError('RGB source no longer matches its original bag binding')
    import rosbag
    from genpy import Time
    ns = row['receipt_ns']
    if ns is None:
        radius = int(binding['source_search_s']*1e9)
        lower, upper = row['image_stamp_ns']-radius, row['image_stamp_ns']+radius
    else:
        lower = upper = ns
    with rosbag.Bag(str(bag), 'r') as recording:
        for _, image, receipt in recording.read_messages(topics=[row['rgb_topic']],
                start_time=Time(lower//10**9, lower%10**9), end_time=Time(upper//10**9, upper%10**9)):
            if (ns is not None and receipt.to_nsec() != ns) or image.header.stamp.to_nsec() != row['image_stamp_ns']:
                continue
            if (image.header.frame_id != row['optical_frame'] or
                    hashlib.sha256(image.data).hexdigest() != row['image_sha256']):
                raise ValueError('The selected RGB observation content changed')
            return rgb_array(image)
    raise ValueError('The indexed RGB observation is absent from the bounded source window')


class MemoryReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.db = read_index(self.directory/'observations.sqlite')
        self.binding = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0])

    def close(self):
        self.db.close()

    def summary(self):
        latest = self.db.execute('SELECT record_json FROM observations ORDER BY rowid DESC LIMIT 1').fetchone()
        row = json.loads(latest[0]) if latest else None
        return dict(memory_id=self.binding['memory_id'], time_scope=self.binding['time_scope'],
            observations=self.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
            latest={key:row[key] for key in ('id','stamp_s','frame_id','session_id','generation')} if row else None)

    def query(self, request):
        from .live_memory import query_live_memory
        return query_live_memory(self.db, self.binding, request)

    def read_image(self, observation, *, bag_override=None):
        row = self.db.execute('SELECT record_json FROM observations WHERE id=?', (observation,)).fetchone()
        if row is None:
            raise KeyError('Unknown visual observation')
        return observation_image(self.db, self.binding, json.loads(row[0]), bag_override=bag_override)


class EnvironmentMap:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = yaml.safe_load((self.directory/'manifest.yaml').read_text())
        layer = self.directory/'layers.npz'
        if content_hash(layer) != self.manifest['layers_sha256']:
            raise ValueError('Environment layers no longer match their binding')
        self.snapshot = GridSnapshot.load(layer)
        if self.snapshot.metadata['snapshot_id'] != self.manifest['snapshot_id']:
            raise ValueError('Environment snapshot identity mismatch')
        self.db = read_index(self.directory/'observations.sqlite')
        binding = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0])
        if binding != self.manifest['memory_binding']:
            self.db.close()
            raise ValueError('Observation index is bound to another environment')

    def close(self):
        self.db.close()

    def get_snapshot(self, ident):
        if ident != self.manifest['snapshot_id']:
            raise KeyError('This package does not contain the requested snapshot')
        return self.snapshot

    def crop_map(self, bounds, *, frame, layers=('occupancy',)):
        return crop_map(self.snapshot, bounds, frame=frame, layers=layers)

    def query_geometry(self, points, *, frame):
        return query_geometry(self.snapshot, points, frame=frame)

    def query_places(self, *, as_of, name=None, bounds=None, frame=None):
        self.check_frame(frame)
        return query_places(self.db, as_of=as_of, name=name, bounds=bounds)

    def query_observations(self, *, as_of, place=None, frame=None, **kwargs):
        self.check_frame(frame)
        if place is not None:
            if any(kwargs.get(key) is not None for key in ('position', 'radius', 'heading')):
                raise ValueError('Query a place reference or spatial filters separately')
            places = self.query_places(name=place, as_of=as_of, frame=frame)
            if len(places) != 1:
                raise ValueError('Place must resolve to exactly one known record')
            key = places[0]['observation_id']
            entry = self.db.execute('SELECT record_json FROM observations WHERE id=? AND stamp_s<=?', (key, float(as_of))).fetchone()
            return [json.loads(entry[0])] if entry else []
        return query_observations(self.db, as_of=as_of, **kwargs)

    def check_frame(self, frame):
        if frame != self.manifest['frame_id']:
            raise ValueError('Query coordinates must explicitly use the environment frame')

    def add_place(self, ident, name, observation, *, frame, **kwargs):
        self.check_frame(frame)
        # Writes happen only through the explicit manual edit command.
        with sqlite3.connect(str(self.directory/'observations.sqlite')) as db:
            db.execute('PRAGMA foreign_keys=ON')
            return add_place(db, ident, name, observation, **kwargs)

    def read_image(self, observation, *, bag_override=None):
        entry = self.db.execute('SELECT record_json FROM observations WHERE id=?', (observation,)).fetchone()
        if entry is None:
            raise KeyError('Unknown visual observation')
        row = json.loads(entry[0])
        binding = self.manifest['memory_binding']
        return observation_image(self.db, binding, row, bag_override=bag_override)

    def summary(self):
        return dict(snapshot_id=self.manifest['snapshot_id'], frame_id=self.manifest['frame_id'],
                    geometry_status=self.snapshot.metadata['geometry_status'],
                    time_scope=self.manifest['time_scope'], source=self.manifest['source'],
                    observation_time_range_s=list(self.db.execute('SELECT MIN(stamp_s),MAX(stamp_s) FROM observations').fetchone()),
                    observations=self.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
                    places=self.db.execute('SELECT COUNT(*) FROM places').fetchone()[0],
                    observed_transitions=self.db.execute('SELECT COUNT(*) FROM transitions').fetchone()[0])
