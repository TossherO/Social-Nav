"""Session persistence and shared API spending reservations."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def image_parts(request):
    for message in request.get('messages',[]):
        content=message.get('content')
        if isinstance(content,list):
            for part in content:
                if isinstance(part,dict) and part.get('type')=='image_url':
                    yield part['image_url']


def restore_request_images(db, event):
    """Read a complete archived request, including original data URLs, on demand."""
    if event.get('image_storage')!='session_sha256':
        return event
    restored=copy.deepcopy(event)
    for image in image_parts(restored['request']):
        digest=image.pop('session_image_sha256',None)
        if digest is None:continue
        row=db.execute('SELECT url FROM request_images WHERE digest=?',(digest,)).fetchone()
        if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=digest:
            raise ValueError('Archived request image missing or changed')
        image['url']=row[0]
    return restored


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, wall_s REAL, kind TEXT, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS map_images (view_id TEXT, role TEXT, sha256 TEXT, png BLOB, PRIMARY KEY(view_id,role))')
        self.db.execute('CREATE TABLE IF NOT EXISTS request_images (digest TEXT PRIMARY KEY,url TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS keyframes (keyframe_id TEXT PRIMARY KEY, task_id TEXT, revision INTEGER, metadata TEXT, image BLOB)')
        self.db.execute('CREATE TABLE IF NOT EXISTS follow_references (task_id TEXT, revision INTEGER, goal_id TEXT, metadata TEXT, image BLOB, crop BLOB, PRIMARY KEY(task_id,revision,goal_id))')
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
            if kind=='api_request' and value.get('request'):
                value=copy.deepcopy(value)
                for image in image_parts(value['request']):
                    url=image.get('url','')
                    if not url.startswith('data:image/'):continue
                    digest=hashlib.sha256(url.encode()).hexdigest()
                    self.db.execute('INSERT OR IGNORE INTO request_images VALUES (?,?)',(digest,url))
                    del image['url']
                    image['session_image_sha256']=digest
                    value['image_storage']='session_sha256'
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

    def follow_reference(self, record, image, crop):
        with self.db:
            self.db.execute('INSERT INTO follow_references VALUES (?,?,?,?,?,?)',
                (record['task_id'],record['task_revision'],record['goal_id'],encode(record),image,crop))

    def read_follow_reference(self, task_id, revision, goal_id):
        import hashlib
        from perception.environment.context import fingerprint
        row = self.db.execute('SELECT metadata,image,crop FROM follow_references WHERE task_id=? AND revision=? AND goal_id=?',
                              (task_id,revision,goal_id)).fetchone()
        if row is None:
            return None
        record, image, crop = json.loads(row[0]), bytes(row[1]), bytes(row[2])
        if (record['reference_id'] != fingerprint({k:v for k,v in record.items() if k != 'reference_id'}) or
                record['image_sha256'] != hashlib.sha256(image).hexdigest() or
                record['crop_sha256'] != hashlib.sha256(crop).hexdigest()):
            raise ValueError('Follow visual reference changed')
        return record,image,crop

    def close(self):
        self.db.close()


class Budget:
    """Record estimated and reported use; an optional explicit limit can be applied."""
    def __init__(self, path, limit=0.8, input_rate=1.2, output_rate=4.0):
        if limit is not None and limit <= 0 or min(input_rate, output_rate) <= 0:
            raise ValueError('Allowance and rates must be positive when set')
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
            if self.limit is not None and spent+reserved > self.limit:
                raise RuntimeError('API development budget exhausted; no request sent')
            call_id = uuid.uuid4().hex
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (call_id, reserved, reserved, 'reserved', '{}'))
            self.db.commit()
            return call_id
        except BaseException:
            self.db.rollback()
            raise

    def estimate_output(self, model, thinking_mode, maximum):
        """Estimate output from recent comparable reported usage, capped by the request limit."""
        values=[]
        rows=self.db.execute("SELECT json_extract(record,'$.requested_model'),json_extract(record,'$.thinking_mode'),"
            "json_extract(record,'$.reasoning'),json_extract(record,'$.usage.completion_tokens') "
            "FROM calls WHERE status='usage_recorded' AND json_valid(record) ORDER BY rowid DESC LIMIT 64")
        for recorded_model,mode,reasoning,value in rows:
            if mode is None:mode='low' if reasoning else 'off'
            if recorded_model==model and mode==thinking_mode and type(value) is int and value>=0:
                values.append(value)
                if len(values)==16:break
        estimated=math.ceil(sum(values)/len(values)) if values else (1024 if thinking_mode=='low' else 512)
        return min(maximum,max(1,estimated))

    def finish(self, call_id, record):
        # Raw replies are retained by the owning session. Only accounting data
        # belongs in the shared ledger used across sessions.
        fields=('requested_model','returned_model','thinking_mode','usage','first_token_s',
                'full_return_s','wall_s','cache','purpose','finish_reason')
        record={k:record[k] for k in fields if k in record}
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
                    basis='reported_usage_or_unresolved_estimate_not_provider_bill')

    def close(self):
        self.db.close()


def shared_budget(root=None):
    """Keep the persistent shared usage ledger without enforcing an allowance."""
    if root is None:
        from .. import ROOT
        root = ROOT
    root = Path(root)
    return Budget(root/'outputs/decision/agent/api_budget.sqlite',limit=None)
