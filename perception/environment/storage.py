"""Portable read-only geometric environment packages."""
from copy import deepcopy
from functools import cached_property
import hashlib
from pathlib import Path

import numpy as np
import yaml

from .context import crop_map, fingerprint, fixed_map_extent, query_geometry
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


def _fixed_navigation_window(snapshot, annotations):
    occupancy = snapshot.layers['occupancy']
    resolution = snapshot.metadata['resolution']
    if (occupancy.ndim != 2 or not occupancy.size or not np.isfinite(resolution) or resolution <= 0 or
            np.shape(snapshot.metadata['origin_xy']) != (2,) or not np.isfinite(snapshot.metadata['origin_xy']).all()):
        raise ValueError('Invalid static map geometry for navigation')
    if not np.isin(occupancy, (-1, 0, 100)).all():
        raise ValueError('Invalid static map occupancy for navigation')
    bounds, asset, _, _ = fixed_map_extent(snapshot, annotations, upper_exclusive=True)
    if not np.isfinite(asset).all() or np.any(asset[2:] <= asset[:2]):
        raise ValueError('Invalid static map extent for navigation')
    height, width = occupancy.shape
    origin = np.tile(asset[:2], 2)
    bounds = np.clip(bounds, origin, np.tile(asset[2:], 2))
    cells = (bounds-origin)/resolution
    nearest = np.rint(cells)
    # Bounds computed in metres may round to either side of the same grid line.
    roundoff = 4*np.spacing(np.maximum(np.abs(bounds), np.abs(origin)))
    cells = np.where(np.abs(bounds-(origin+nearest*resolution)) <= roundoff, nearest, cells)
    x0, y0 = np.floor(cells[:2]).astype(int)
    x1, y1 = np.ceil(cells[2:]).astype(int)
    return int(x0), int(y0), min(int(x1), width), min(int(y1), height)


class StaticEnvironmentMap:
    """Read a persistent geometric map asset; it never represents a running view."""
    def __init__(self, directory, *, navigation_annotations=None, navigation_window=None):
        if navigation_annotations is not None and navigation_window is not None:
            raise ValueError('Select navigation annotations or a preselected window')
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
        manual = self.snapshot.layers.get('manual_edit')
        self.manual_free_cells = (int(np.count_nonzero(manual & (self.snapshot.layers['occupancy'] == 0)))
                                  if manual is not None else 0)
        # Older packages may contain extra assets. Geometry loading never opens them.
        self._navigation_window = None
        if navigation_annotations is not None:
            self._navigation_window = _fixed_navigation_window(self.snapshot, navigation_annotations)
        elif navigation_window is not None:
            height, width = self.snapshot.layers['occupancy'].shape
            if (not isinstance(navigation_window, (list, tuple)) or len(navigation_window) != 4 or
                    any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in navigation_window)):
                raise ValueError('Navigation window must contain four integer cell offsets')
            x0, y0, x1, y1 = map(int, navigation_window)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError('Navigation window lies outside its source raster or is empty')
            low_x, low_y, high_x, high_y = _fixed_navigation_window(self.snapshot, {})
            if x0 > low_x or y0 > low_y or x1 < high_x or y1 < high_y:
                raise ValueError('Navigation window excludes the padded known occupancy extent')
            self._navigation_window = x0, y0, x1, y1

    @property
    def navigation_window(self):
        """Half-open source cell offsets (x0, y0, x1, y1) selected for this asset."""
        if self._navigation_window is None:
            self._navigation_window = _fixed_navigation_window(self.snapshot, {})
        return self._navigation_window

    @cached_property
    def navigation_snapshot(self):
        """Return a source-bound grid view with identical per-cell layer values."""
        x0, y0, x1, y1 = self.navigation_window
        metadata = deepcopy(self.snapshot.metadata)
        binding = dict(source_snapshot_id=metadata['snapshot_id'],
                       source_layers_sha256=self.manifest['layers_sha256'],
                       source_window_xyxy=[x0, y0, x1, y1])
        origin = (np.asarray(metadata['origin_xy'])+np.array([x0, y0])*metadata['resolution']).tolist()
        metadata.pop('map_type', None)
        metadata.update(binding, entity_type='view', view_type='fixed_navigation_view',
                        source_map_type='static_environment_map', shape=[y1-y0, x1-x0], origin_xy=origin,
                        snapshot_id='fixed_navigation_view:'+fingerprint(binding))
        return GridSnapshot(metadata, {name: layer[y0:y1, x0:x1] for name, layer in self.snapshot.layers.items()})

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
            manual_free_cells=self.manual_free_cells,
            fusion=self.snapshot.metadata.get('fusion'),
            projection_strategy=self.snapshot.metadata.get('projection_strategy'),
            free_assumption=self.snapshot.metadata.get('free_assumption'),
            free_space_supported=self.snapshot.metadata.get('free_space_supported'),
            time_scope=self.manifest['time_scope'], source=self.manifest['source'])
