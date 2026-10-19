"""One local SQLite: bounded assets, personal preferences, durable edits and conflicts."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .data import blank_review, dumps, safe_directory
from .protocol import digest


class Cache:
    def __init__(self, directory, max_bytes=256 * 1024 * 1024):
        self.directory, self.max_bytes = Path(directory).expanduser(), max_bytes
        self.ready, self.init_lock, self.lock_file = False, threading.Lock(), None

    @contextmanager
    def open(self):
        with self.init_lock:
            if not self.ready:
                safe_directory(self.directory).mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(str(self.directory / 'cache.sqlite'), timeout=20) as database:
                    database.executescript('''
                        PRAGMA journal_mode=WAL;
                        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                        CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY,source_id TEXT NOT NULL,
                            ordinal INTEGER NOT NULL,server TEXT,bundle BLOB,encoding TEXT,access REAL NOT NULL);
                        CREATE TABLE IF NOT EXISTS outbox (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                            mutation_id TEXT UNIQUE NOT NULL,source_id TEXT NOT NULL,frame_id TEXT,
                            kind TEXT NOT NULL,body TEXT NOT NULL,attempted INTEGER NOT NULL DEFAULT 0,
                            status TEXT NOT NULL DEFAULT 'pending',error TEXT NOT NULL DEFAULT '');
                        CREATE TABLE IF NOT EXISTS conflicts (mutation_id TEXT PRIMARY KEY,
                            identity TEXT NOT NULL,body TEXT NOT NULL,error TEXT NOT NULL,archived REAL NOT NULL);
                    ''')
                    database.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('client_id', dumps(uuid.uuid4().hex)))
                self.ready = True
        database = sqlite3.connect(str(self.directory / 'cache.sqlite'), timeout=20)
        database.row_factory = sqlite3.Row
        try:
            yield database
            database.commit()
        except Exception:
            database.rollback()
            raise
        finally:
            database.close()

    def lock(self):
        if self.lock_file is not None:
            return
        with self.open():
            pass
        stream = (self.directory / 'client.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b'0')
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise ValueError('该缓存已有客户端运行；关闭原窗口或使用不同缓存目录') from error
        self.lock_file = stream

    def close(self):
        if self.lock_file is not None:
            self.lock_file.close()
            self.lock_file = None

    @staticmethod
    def get(database, key, default=None):
        row = database.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row['value']) if row else default

    @staticmethod
    def put(database, key, value):
        database.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, dumps(value)))

    def bump_queue(self, database):
        self.put(database, 'queue_generation', self.get(database, 'queue_generation', 0) + 1)

    def advance_floor(self, database, ident, epoch, revision):
        floor = max(self.get(database, 'floor:' + ident, [0, 0]), [epoch, revision])
        self.put(database, 'floor:' + ident, floor)
        return floor

    def meta(self, key, default=None):
        with self.open() as database:
            return self.get(database, key, default)

    def set_meta(self, key, value):
        with self.open() as database:
            self.put(database, key, value)

    def bind(self, identity, snapshot):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            previous = self.get(database, 'identity')
            if previous is not None and previous != identity:
                raise ValueError('服务器／工程身份已改变；原草稿与缓存保留，请指定新的 --cache-dir')
            self.put(database, 'identity', identity)
            self.put_index(database, snapshot)

    def put_index(self, database, snapshot):
        self.put(database, 'index', snapshot)
        indexed = {row['id']: row for row in snapshot['rows']}
        for source in snapshot['sources']:
            # A response-lost operation may already be visible on the server.
            # Until its receipt is reconciled, keep our acknowledged source base
            # so newly queued group/reset expectations do not count it twice.
            pending = database.execute('SELECT 1 FROM outbox WHERE source_id=?', (source['id'],)).fetchone()
            old = self.get(database, 'source:' + source['id'])
            if not pending and (old is None or source['revision'] >= old['revision']):
                self.put(database, 'source:' + source['id'], source)
        for row in database.execute('SELECT id,source_id,server FROM frames'):
            indexed_row = indexed.get(row['id'])
            indexed_state = [indexed_row.get('epoch', 0), indexed_row.get('revision', 0)] if indexed_row else [0, 0]
            self.advance_floor(database, row['id'], *indexed_state)
            if row['server'] is None:
                continue
            frame = json.loads(row['server'])
            pending = database.execute('SELECT 1 FROM outbox WHERE frame_id=?', (row['id'],)).fetchone()
            if (not pending and (indexed_row is None or indexed_state >= [frame['epoch'], frame['revision']] and
                    digest(frame['review']) != indexed_row.get('review_hash', digest(json.loads(indexed_row['review']))))):
                database.execute('UPDATE frames SET server=NULL WHERE id=?', (row['id'],))
        for key, value in snapshot['metadata'].items():
            if self.get(database, 'view:' + key) is None:
                self.put(database, 'view:' + key, value)

    def update_index(self, snapshot):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            self.put_index(database, snapshot)

    def snapshot(self):
        with self.open() as database:
            database.execute('BEGIN')
            snapshot = self.get(database, 'index')
            if snapshot is None:
                raise ValueError('首次使用该工程需要联网')
            snapshot['metadata'] = {key: self.get(database, 'view:' + key) for key in snapshot['metadata']}
            snapshot['sources'] = [self.source_in(database, source['id']) for source in snapshot['sources']]
            reviews = {row['id']: json.loads(row['server'])['review'] for row in
                       database.execute('SELECT id,server FROM frames WHERE server IS NOT NULL')}
            for row in database.execute("SELECT frame_id,body FROM outbox WHERE kind='review' ORDER BY seq"):
                reviews[row['frame_id']] = json.loads(row['body'])['review']
            for row in snapshot['rows']:
                if row['id'] in reviews:
                    row['review'] = dumps(reviews[row['id']])
            return snapshot

    def source_in(self, database, ident):
        source = self.get(database, 'source:' + ident)
        for row in database.execute("SELECT body FROM outbox WHERE source_id=? AND kind='group' ORDER BY seq", (ident,)):
            source.update(json.loads(row['body'])['group'])
        return source

    def state_in(self, database, ident):
        row = database.execute('SELECT server FROM frames WHERE id=?', (ident,)).fetchone()
        if row is None or row['server'] is None:
            raise KeyError('该帧尚无有效缓存，需要联网读取')
        frame = json.loads(row['server'])
        tail = database.execute("SELECT body FROM outbox WHERE frame_id=? AND kind='review' ORDER BY seq DESC LIMIT 1", (ident,)).fetchone()
        if tail:
            body = json.loads(tail['body'])
            frame.update(review=body['review'], revision=body['revision'] + 1, epoch=body['epoch'])
        return frame

    def state(self, ident):
        with self.open() as database:
            database.execute('BEGIN')
            return self.state_in(database, ident)

    def save_frame(self, frame, bundle=None, encoding=None):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            floor = self.get(database, 'floor:' + frame['id'], [0, 0])
            if [frame['epoch'], frame['revision']] < floor:
                return False  # A late read cannot undo index/receipt invalidation.
            previous = database.execute('SELECT server FROM frames WHERE id=?', (frame['id'],)).fetchone()
            if previous is not None and previous['server'] is not None:
                old = json.loads(previous['server'])
                if (old['epoch'], old['revision']) > (frame['epoch'], frame['revision']):
                    frame = old
            self.advance_floor(database, frame['id'], frame['epoch'], frame['revision'])
            database.execute('INSERT INTO frames(id,source_id,ordinal,server,access) VALUES (?,?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET server=excluded.server,access=excluded.access',
                (frame['id'], frame['source_id'], frame['ordinal'], dumps(frame), time.time()))
            if bundle is not None:
                database.execute('UPDATE frames SET bundle=?,encoding=? WHERE id=?', (bundle, encoding, frame['id']))
                self.prune(database, frame['id'])

    def bundle(self, ident, encoding=None):
        with self.open() as database:
            row = database.execute('SELECT bundle,encoding FROM frames WHERE id=?', (ident,)).fetchone()
            if row is not None:
                database.execute('UPDATE frames SET access=? WHERE id=?', (time.time(), ident))
            return row['bundle'] if row and (encoding is None or encoding == row['encoding']) else None

    def drop_bundle(self, ident):
        with self.open() as database:
            database.execute('UPDATE frames SET bundle=NULL,encoding=NULL WHERE id=?', (ident,))

    def prune(self, database, current):
        total = database.execute('SELECT COALESCE(SUM(LENGTH(bundle)),0) FROM frames').fetchone()[0]
        for row in database.execute('SELECT id,LENGTH(bundle) AS size FROM frames WHERE bundle IS NOT NULL AND id!=? ORDER BY access', (current,)):
            if total <= self.max_bytes:
                break
            database.execute('UPDATE frames SET bundle=NULL WHERE id=?', (row['id'],))
            total -= row['size']

    def enqueue(self, kind, source_id, review=None, expected=None, frame_id=None, group=None, current=None):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            if database.execute("SELECT 1 FROM outbox WHERE source_id=? AND kind='reset'", (source_id,)).fetchone():
                raise ValueError('该子包的清空结果尚未确认，请先同步／重连')
            source = self.get(database, 'source:' + source_id)
            count = database.execute('SELECT COUNT(*) FROM outbox WHERE source_id=?', (source_id,)).fetchone()[0]
            body = dict(mutation_id=uuid.uuid4().hex, kind=kind, source_id=source_id, epoch=source['epoch'],
                        revision=source['revision'] + count, client_id=self.get(database, 'client_id'))
            body.update(self.get(database, 'identity'))
            if kind == 'review':
                frame = self.state_in(database, frame_id)
                if frame['review'] != expected:
                    raise ValueError('本机草稿已改变，请重新读取；没有覆盖已暂存结果')
                body.update(frame_id=frame_id, fingerprint=frame['fingerprint'], epoch=frame['epoch'],
                            revision=frame['revision'], expected_hash=digest(expected), review=copy.deepcopy(review))
            elif kind == 'group':
                body['group'] = group
            elif kind == 'reset':
                if database.execute('SELECT 1 FROM outbox LIMIT 1').fetchone():
                    raise ValueError('清空前必须完成所有待同步编辑')
                body['current'] = current
            else:
                raise ValueError('不支持的队列操作')
            database.execute('INSERT INTO outbox(mutation_id,source_id,frame_id,kind,body) VALUES (?,?,?,?,?)',
                             (body['mutation_id'], source_id, frame_id, kind, dumps(body)))
            self.bump_queue(database)
            return body

    def first(self):
        with self.open() as database:
            row = database.execute('SELECT * FROM outbox ORDER BY seq LIMIT 1').fetchone()
            return dict(row) if row else None

    def attempted(self, ident):
        with self.open() as database:
            database.execute('UPDATE outbox SET attempted=1 WHERE mutation_id=?', (ident,))

    def conflict(self, ident, error):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            database.execute("UPDATE outbox SET status='conflict',error=? WHERE mutation_id=?", (error, ident))
            self.bump_queue(database)

    def acknowledge(self, result):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            row = database.execute('SELECT * FROM outbox WHERE mutation_id=?', (result['mutation_id'],)).fetchone()
            if row is None:
                return
            body = json.loads(row['body'])
            source = result['source']
            if (result['kind'] != body['kind'] or result['source_id'] != body['source_id'] or
                    source['id'] != body['source_id'] or
                    source['epoch'] != body['epoch'] + int(body['kind'] == 'reset')):
                raise ValueError('保存回执与操作身份不一致')
            if body['kind'] == 'review':
                frame = result['frame']
                if (frame['id'] != body['frame_id'] or frame['fingerprint'] != body['fingerprint'] or
                        frame['review'] != body['review'] or frame['revision'] != body['revision'] + 1):
                    raise ValueError('帧保存回执与原始编辑不一致')
                previous = database.execute('SELECT server FROM frames WHERE id=?', (frame['id'],)).fetchone()
                old = json.loads(previous['server']) if previous and previous['server'] else None
                floor = self.advance_floor(database, frame['id'], frame['epoch'], frame['revision'])
                if [frame['epoch'], frame['revision']] >= floor and (old is None or (old['epoch'], old['revision']) <= (frame['epoch'], frame['revision'])):
                    database.execute('UPDATE frames SET server=? WHERE id=?', (dumps(frame), frame['id']))
            elif body['kind'] == 'reset':
                if source['revision'] != body['revision'] + 1:
                    raise ValueError('清空回执修订不一致')
                database.execute('UPDATE frames SET server=NULL WHERE source_id=?', (source['id'],))
                for item in database.execute('SELECT id FROM frames WHERE source_id=?', (source['id'],)):
                    self.advance_floor(database, item['id'], source['epoch'], 0)
                if result['frame'] is not None:
                    self.advance_floor(database, result['frame']['id'], result['frame']['epoch'], result['frame']['revision'])
                    database.execute('UPDATE frames SET server=? WHERE id=?', (dumps(result['frame']), result['frame']['id']))
                snapshot = self.get(database, 'index')
                for item in snapshot['rows']:
                    if item['source_id'] == source['id']:
                        item['review'] = dumps(blank_review())
                        item.update(epoch=source['epoch'], revision=0)
                self.put(database, 'index', snapshot)
            elif source['revision'] != body['revision'] + 1 or any(source[key] != value for key, value in body['group'].items()):
                raise ValueError('分组回执与原始操作不一致')
            old = self.get(database, 'source:' + source['id'])
            if old is None or old['revision'] <= source['revision']:
                self.put(database, 'source:' + source['id'], source)
            database.execute('DELETE FROM outbox WHERE mutation_id=?', (result['mutation_id'],))
            self.bump_queue(database)

    def counts(self):
        with self.open() as database:
            database.execute('BEGIN')
            return dict(pending=database.execute('SELECT COUNT(*) FROM outbox').fetchone()[0],
                        conflicts=database.execute("SELECT COUNT(*) FROM outbox WHERE status='conflict'").fetchone()[0],
                        reset_pending=bool(database.execute("SELECT 1 FROM outbox WHERE kind='reset' LIMIT 1").fetchone()),
                        archived=database.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0],
                        generation=self.get(database, 'queue_generation', 0))

    def archive_conflicts(self):
        with self.open() as database:
            database.execute('BEGIN IMMEDIATE')
            sources = [row[0] for row in database.execute("SELECT DISTINCT source_id FROM outbox WHERE status='conflict'")]
            for source in sources:
                rows = list(database.execute('SELECT * FROM outbox WHERE source_id=?', (source,)))
                for row in rows:
                    database.execute('INSERT OR REPLACE INTO conflicts VALUES (?,?,?,?,?)',
                        (row['mutation_id'], dumps(self.get(database, 'identity')), row['body'], row['error'], time.time()))
                database.execute('DELETE FROM outbox WHERE source_id=?', (source,))
                database.execute('UPDATE frames SET server=NULL WHERE source_id=?', (source,))
            self.bump_queue(database)
            return len(sources)

    def drafts(self):
        with self.open() as database:
            database.execute('BEGIN')
            return dict(identity=self.get(database, 'identity'),
                pending=[dict(body=json.loads(row['body']), error=row['error']) for row in database.execute('SELECT * FROM outbox ORDER BY seq')],
                conflicts=[dict(identity=json.loads(row['identity']), body=json.loads(row['body']), error=row['error'])
                           for row in database.execute('SELECT * FROM conflicts ORDER BY archived')])
