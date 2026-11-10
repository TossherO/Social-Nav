"""Session persistence and shared API spending reservations."""
import json
from pathlib import Path
import sqlite3
import time
import uuid


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, wall_s REAL, kind TEXT, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS map_images (view_id TEXT, role TEXT, sha256 TEXT, png BLOB, PRIMARY KEY(view_id,role))')
        self.db.execute('CREATE TABLE IF NOT EXISTS keyframes (keyframe_id TEXT PRIMARY KEY, task_id TEXT, revision INTEGER, metadata TEXT, image BLOB)')
        self.db.commit()

    def save(self, state, kind, value):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO state VALUES (1,?)', (encode(state),))
            self.db.execute('INSERT INTO events(wall_s,kind,value) VALUES (?,?,?)', (time.time(), kind, encode(value)))

    def load(self):
        row = self.db.execute('SELECT value FROM state WHERE id=1').fetchone()
        return None if row is None else json.loads(row[0])

    def event(self, kind, value):
        with self.db:
            self.db.execute('INSERT INTO events(wall_s,kind,value) VALUES (?,?,?)', (time.time(), kind, encode(value)))

    def spatial_view(self, view):
        view.verify()
        record = view.record
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO map_images VALUES (?,?,?,?)',
                [(record['view_id'],role,digest,image) for role,digest,image in
                 zip(record['image_roles'],record['image_sha256'],view.images)])

    def reference_map(self, reference):
        reference.verify()
        r = reference.record
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO map_images VALUES (?,?,?,?)',
                (r['reference_map_id'],r['image_role'],r['image_sha256'],reference.image))
        self.event('reference_map_prepared',r)

    def close(self):
        self.db.close()


class Budget:
    """Reserve before network I/O; errors/missing usage retain the full reservation.

    Rates are conservative RMB/million-token estimates, not an account balance.
    A shared ledger prevents a new session resetting this development allowance.
    """
    def __init__(self, path, limit=0.8, input_rate=1.2, output_rate=4.0):
        if not 0 < limit <= 10 or min(input_rate, output_rate) <= 0:
            raise ValueError('Development allowance must be positive and at most ten RMB')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, reserved REAL, charged REAL, status TEXT, record TEXT)')
        self.db.commit()
        self.limit, self.input_rate, self.output_rate = limit, input_rate, output_rate

    def reserve(self, input_bound, output_bound):
        reserved = (input_bound*self.input_rate + output_bound*self.output_rate)/1e6
        try:
            self.db.execute('BEGIN IMMEDIATE')
            spent = self.db.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
            if spent+reserved > self.limit:
                raise RuntimeError('API development budget exhausted; no request sent')
            call_id = uuid.uuid4().hex
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (call_id, reserved, reserved, 'reserved', '{}'))
            self.db.commit()
            return call_id
        except BaseException:
            self.db.rollback()
            raise

    def finish(self, call_id, record):
        usage = record.get('usage') or {}
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        known = all(type(v) is int and v >= 0 for v in (inp, out))
        charged = (inp*self.input_rate+out*self.output_rate)/1e6 if known else None
        with self.db:
            if charged is None:
                self.db.execute('UPDATE calls SET status=?,record=? WHERE id=?', ('usage_unknown', encode(record), call_id))
            else:
                self.db.execute('UPDATE calls SET charged=?,status=?,record=? WHERE id=?',
                                (charged, 'usage_recorded', encode(record), call_id))

    def summary(self):
        total, charged = self.db.execute('SELECT COUNT(*),COALESCE(SUM(charged),0) FROM calls').fetchone()
        return dict(calls=total, accounted_rmb=charged, allowance_rmb=self.limit,
                    input_rmb_per_million=self.input_rate, output_rmb_per_million=self.output_rate,
                    basis='conservative_estimate_or_unresolved_reservation_not_provider_bill')

    def close(self):
        self.db.close()


def shared_budget(root=None):
    """Use the local allowance with the same persistent shared ledger."""
    if root is None:
        from .. import ROOT
        root = ROOT
    root = Path(root)
    settings = root/'configs/agent_budget.local.json'
    limit = .8
    if settings.exists():
        value = json.loads(settings.read_text())
        if not isinstance(value,dict) or set(value) != {'limit_rmb'} or type(value['limit_rmb']) not in (int,float):
            raise ValueError('Invalid local Agent budget settings')
        limit = value['limit_rmb']
    return Budget(root/'outputs/decision/agent/api_budget.sqlite',limit=limit)
