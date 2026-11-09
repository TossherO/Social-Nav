"""Portable read-only geometric environment packages."""
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .context import crop_map, query_geometry
from .layers import GridSnapshot


ROOT = Path(__file__).resolve().parents[2]


def asset_reference(path):
    path = Path(path).expanduser().absolute()
    try:
        return dict(base='project_root', path=str(path.relative_to(ROOT)))
    except ValueError:
        return dict(base='absolute', path=str(path))


def content_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export_map_server_grid(snapshot, directory):
    """Export format only: map_server occupancy PGM/YAML, unknown=205, no costmap inflation."""
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


class StaticEnvironmentMap:
    """Read a persistent geometric map asset; it never represents a running view."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = yaml.safe_load((self.directory/'manifest.yaml').read_text())
        if (self.manifest.get('kind') != 'static_environment' or
                self.manifest.get('time_scope') != 'offline_final_prior'):
            raise ValueError('Expected a static environment map package, not a local snapshot or localization map')
        layer = self.directory/'layers.npz'
        if content_hash(layer) != self.manifest['layers_sha256']:
            raise ValueError('Environment layers no longer match their binding')
        self.snapshot = GridSnapshot.load(layer)
        if (self.snapshot.metadata['snapshot_id'] != self.manifest['snapshot_id'] or
                self.snapshot.metadata['frame_id'] != self.manifest['frame_id']):
            raise ValueError('Environment snapshot identity/frame mismatch')
        if self.snapshot.metadata.get('projection_strategy') in ('docker_assume_free', 'ground_relative_assume_free'):
            mask=self.snapshot.layers.get('assumed_free');times=self.snapshot.layers.get('last_observed')
            if (mask is None or mask.dtype!=np.bool_ or times is None or
                    not np.array_equal(mask,self.snapshot.layers['occupancy']==0) or
                    not np.isnan(times).all() or self.snapshot.metadata.get('free_space_supported') is not False):
                raise ValueError('Assumed-free projection must preserve its mask and absence of observation times')
        # Older packages may contain extra assets. Geometry loading never opens them.

    def close(self):
        pass

    def get_snapshot(self, ident):
        if ident != self.manifest['snapshot_id']:
            raise KeyError('This package does not contain the requested snapshot')
        return self.snapshot

    def crop_map(self, bounds, *, frame, layers=('occupancy',)):
        return crop_map(self.snapshot, bounds, frame=frame, layers=layers)

    def query_geometry(self, points, *, frame):
        return query_geometry(self.snapshot, points, frame=frame)

    def summary(self):
        return dict(entity_type='map',map_type='static_environment_map',
            snapshot_id=self.manifest['snapshot_id'], frame_id=self.manifest['frame_id'],
            geometry_status=self.snapshot.metadata['geometry_status'],
            ground_estimator=self.snapshot.metadata.get('ground_estimator'),
            free_recovery_policy=self.snapshot.metadata.get('dynamic_filter', {}).get('geometric_free_recovery_policy'),
            projection_strategy=self.snapshot.metadata.get('projection_strategy'),
            free_assumption=self.snapshot.metadata.get('free_assumption'),
            free_space_supported=self.snapshot.metadata.get('free_space_supported'),
            time_scope=self.manifest['time_scope'], source=self.manifest['source'])
