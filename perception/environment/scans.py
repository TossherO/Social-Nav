"""Optional compressed deskewed observations; never a copy of feature keyframes."""
import io
import json
from pathlib import Path
import sqlite3

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from .memory import encoded


class ScanJournal:
    def __init__(self, path, settings, binding):
        self.path, self.settings = Path(path), settings
        if any(type(settings[name]) is not int or settings[name] <= 0 for name in ('max_records','max_bytes')):
            raise ValueError('Scan journal budgets must be positive integers')
        if self.path.exists():
            raise FileExistsError('Select a new scan journal')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Single worker writes; shutdown closes only after that worker has joined.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE scans(stamp_ns INTEGER PRIMARY KEY, record_json TEXT NOT NULL, payload BLOB NOT NULL);
        ''')
        self.db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', encoded(binding)))
        self.db.commit()
        self.count, self.bytes, self.last = 0, 0, None

    def append(self, points, transform, stamp_ns, metadata, origins=None, offset_ms=None):
        if self.last is not None and stamp_ns <= self.last:
            raise ValueError('Scan journal cannot reorder or overwrite observations')
        matrix = validate_transform(transform)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected deskewed sensor endpoints')
        keep = np.isfinite(points).all(axis=1)
        arrays = dict(points=points[keep], local_from_lidar=matrix)
        if origins is not None:
            origins = np.asarray(origins, dtype=np.float32)
            if origins.shape != points.shape or not np.isfinite(origins).all():
                raise ValueError('Journal ray origins must match every sensor endpoint')
            arrays['origins'] = origins[keep]
        if offset_ms is not None:
            offsets = np.asarray(offset_ms, dtype=np.float32)
            # Point zero need not be the earliest firing in an organized scan.
            if offsets.shape != (len(points),) or not np.isfinite(offsets).all():
                raise ValueError('Invalid original point-time offsets')
            arrays['offset_ms'] = offsets[keep]
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        data = stream.getvalue()
        if self.count >= self.settings['max_records'] or self.bytes+len(data) > self.settings['max_bytes']:
            raise ValueError('Scan journal budget exceeded; no silent truncation is allowed')
        record = dict(metadata, stamp_ns=int(stamp_ns))
        if record['rays_used'] and record['ray_origin_model'] == 'pose_interpolated_origins' and origins is None:
            raise ValueError('Missing moving ray origins')
        self.db.execute('INSERT INTO scans VALUES (?,?,?)', (int(stamp_ns), encoded(record), data))
        self.db.commit()
        self.count += 1; self.bytes += len(data); self.last = stamp_ns

    def summary(self):
        return dict(records=self.count, compressed_bytes=self.bytes, last_stamp_ns=self.last)

    def close(self):
        if self.db is not None:
            self.db.execute('INSERT INTO metadata VALUES (?,?)', ('sealed', encoded(self.summary())))
            self.db.commit(); self.db.close(); self.db = None


def read_scans(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    try:
        sealed = db.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone()
        if sealed is None or json.loads(sealed[0])['records'] != db.execute('SELECT COUNT(*) FROM scans').fetchone()[0]:
            raise ValueError('Scan journal was not sealed cleanly')
        for stamp, raw, payload in db.execute('SELECT stamp_ns,record_json,payload FROM scans ORDER BY stamp_ns'):
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            metadata = json.loads(raw)
            if metadata['stamp_ns'] != stamp:
                raise ValueError('Scan payload timestamp mismatch')
            yield metadata, arrays
    finally:
        db.close()


def sealed_scan_summary(path):
    if not Path(path).is_file():
        raise ValueError('Scan journal is missing')
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    try:
        entry = db.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone()
        if entry is None or json.loads(entry[0])['records'] != db.execute('SELECT COUNT(*) FROM scans').fetchone()[0]:
            raise ValueError('Scan journal was not sealed cleanly')
        return json.loads(entry[0])
    finally:
        db.close()
