"""Optional compressed deskewed observations; never a copy of feature keyframes."""
import io
import json
from pathlib import Path
import sqlite3
from collections import deque
import copy
import threading
import time

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from .context import encoded


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

    def fail(self, reason):
        """A partial archive remains inspectable, but can never pass as sealed."""
        if self.db is not None:
            self.db.execute("DELETE FROM metadata WHERE key='sealed'")
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('failed',encoded(dict(reason=str(reason)))))
            self.db.commit();self.db.close();self.db=None


class AsyncScanJournal:
    """Single bounded archive worker; submit never waits for disk/compression.

    Overflow/failure is permanent and explicit. Geometry can keep publishing,
    while callers requiring a complete archive must reject this recording.
    """
    def __init__(self, path, settings, binding, *, writer_factory=ScanJournal):
        self.path,self.settings,self.binding=Path(path),settings,copy.deepcopy(binding)
        self.capacity=settings.get('queue_size',8)
        self.byte_budget=settings.get('max_pending_bytes',64*1024*1024)
        if any(type(n) is not int or n<=0 for n in (self.capacity,self.byte_budget)):
            raise ValueError('Async scan budgets must be positive integers')
        self.condition=threading.Condition()
        self.queue=deque();self.pending_bytes=0;self.in_flight=False
        self.done=False;self.error=None;self.state='recording'
        self.submitted=0;self.last_submitted=None;self.write_times=[]
        self.written=dict(records=0,compressed_bytes=0,last_stamp_ns=None)
        self.writer_factory=writer_factory
        self.worker=threading.Thread(target=self._work,name='environment-scan-archive',daemon=True)
        self.worker.start()

    def submit(self, points, transform, stamp_ns, metadata, origins=None, offset_ms=None):
        arrays=[np.asarray(points),np.asarray(transform),None if origins is None else np.asarray(origins),
                None if offset_ms is None else np.asarray(offset_ms)]
        size=sum(a.nbytes for a in arrays if a is not None)
        with self.condition:
            if self.done or self.error is not None:return False
            if (len(self.queue)+int(self.in_flight)>=self.capacity or self.pending_bytes+size>self.byte_budget
                    or self.last_submitted is not None and stamp_ns<=self.last_submitted):
                self.error='Scan archive queue budget exceeded or timestamps are not increasing'
                self.state='failed';self.condition.notify_all();return False
            # Freeze only the accepted inputs. The producer may reuse arrays as
            # soon as this call returns; the worker never borrows mutable state.
            arrays=[None if a is None else a.copy() for a in arrays]
            self.queue.append((arrays,int(stamp_ns),copy.deepcopy(metadata),size))
            self.pending_bytes+=size;self.submitted+=1;self.last_submitted=int(stamp_ns)
            self.condition.notify();return True

    def _work(self):
        writer=None
        try:
            writer=self.writer_factory(self.path,self.settings,self.binding)
            while True:
                with self.condition:
                    while not self.queue and not self.done and self.error is None:self.condition.wait()
                    if self.error is not None:raise RuntimeError(self.error)
                    if not self.queue and self.done:break
                    arrays,stamp,metadata,size=self.queue.popleft();self.in_flight=True
                started=time.monotonic()
                writer.append(arrays[0],arrays[1],stamp,metadata,arrays[2],arrays[3])
                with self.condition:
                    self.written=writer.summary();self.in_flight=False;self.pending_bytes-=size
                    self.write_times.append((time.monotonic()-started)*1000)
                    self.write_times=self.write_times[-512:]
                    self.condition.notify_all()
            writer.close()
            with self.condition:self.state='sealed'
        except Exception as failure:
            with self.condition:
                self.error=str(failure);self.state='failed';self.queue.clear()
                self.in_flight=False;self.pending_bytes=0
            if writer is not None:writer.fail(self.error)

    def summary(self):
        with self.condition:
            values=self.write_times
            return dict(self.written,state=self.state,error=self.error,submitted=self.submitted,
                pending=len(self.queue)+int(self.in_flight),pending_bytes=self.pending_bytes,
                write_ms=dict(samples=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95))) if values else None)

    def close(self, timeout=20):
        with self.condition:self.done=True;self.condition.notify_all()
        self.worker.join(timeout)
        if self.worker.is_alive():
            with self.condition:
                self.error='Scan archive did not drain within the shutdown budget';self.state='failed'
                self.condition.notify_all()
            raise RuntimeError(self.error)
        if self.error is not None:raise RuntimeError('Scan archive failed: '+self.error)


def read_scans(path, *, stamps=None):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    try:
        sealed = db.execute("SELECT value FROM metadata WHERE key='sealed'").fetchone()
        if (db.execute("SELECT 1 FROM metadata WHERE key='failed'").fetchone() is not None or
                sealed is None or json.loads(sealed[0])['records'] != db.execute('SELECT COUNT(*) FROM scans').fetchone()[0]):
            raise ValueError('Scan journal was not sealed cleanly')
        if stamps is None:
            records = db.execute('SELECT stamp_ns,record_json,payload FROM scans ORDER BY stamp_ns')
        else:
            # Indexed primary-key lookups avoid loading/decompressing irrelevant scans.
            def selected():
                for stamp in sorted(stamps):
                    row = db.execute('SELECT stamp_ns,record_json,payload FROM scans WHERE stamp_ns=?', (int(stamp),)).fetchone()
                    if row is None:
                        raise ValueError('Indexed scan is missing from the sealed journal')
                    yield row
            records = selected()
        for stamp, raw, payload in records:
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
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
        if (db.execute("SELECT 1 FROM metadata WHERE key='failed'").fetchone() is not None or
                entry is None or json.loads(entry[0])['records'] != db.execute('SELECT COUNT(*) FROM scans').fetchone()[0]):
            raise ValueError('Scan journal was not sealed cleanly')
        return json.loads(entry[0])
    finally:
        db.close()
