"""Explicit map-frame coverage regions, independent of free-space assertions."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from .results import map_fingerprint


def asset_digest(map_path):
    return hashlib.sha256(json.dumps(map_fingerprint(map_path), sort_keys=True).encode()).hexdigest()


def save_coverage(path, map_path, regions, *, source='manual_map_selection'):
    """Bind an explicitly selected XY region to the unchanged map asset."""
    frame = yaml.safe_load((Path(map_path) / 'manifest.yaml').read_text())['frame']
    record = dict(frame_id=frame, source=source, map_asset_fingerprint=asset_digest(map_path),
                  regions_xy=[np.asarray(p, float).tolist() for p in regions])
    coverage = PriorCoverage(record, asset_fingerprint=record['map_asset_fingerprint'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as stream:
            yaml.safe_dump(record, stream, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return coverage


class PriorCoverage:
    def __init__(self, record, *, asset_fingerprint):
        if (not record.get('frame_id') or not record.get('source') or
                record.get('map_asset_fingerprint') != asset_fingerprint):
            raise ValueError('Coverage must name its frame, source and the loaded map asset fingerprint')
        self.record = record
        self.regions = [np.asarray(region, float) for region in record['regions_xy']]
        if not self.regions or any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or
                not np.isfinite(p).all() or abs(np.cross(p,np.roll(p,-1,axis=0)).sum()) < 1e-8
                for p in self.regions):
            raise ValueError('Coverage needs finite nondegenerate XY polygons in metres')
        self.fingerprint = hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()

    @classmethod
    def load(cls, path, *, asset_fingerprint):
        return cls(yaml.safe_load(Path(path).read_text()), asset_fingerprint=asset_fingerprint)

    def contains(self, position, frame):
        if frame != self.record['frame_id']:
            raise ValueError('Coverage coordinates differ from the registration map frame')
        point = np.asarray(position, float)[:2]
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Coverage query needs a finite map-frame position')
        for poly in self.regions:
            inside = False
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                delta = b-a
                if abs(np.cross(delta,point-a)) < 1e-8 and np.dot(point-a,point-b) <= 1e-8:
                    return True
                if (a[1] > point[1]) != (b[1] > point[1]):
                    inside ^= point[0] < (b[0]-a[0])*(point[1]-a[1])/(b[1]-a[1])+a[0]
            if inside:
                return True
        return False
