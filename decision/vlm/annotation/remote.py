"""Independent HTTP client, bounded asset cache and durable ordered edits."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .rules import dumps
from .store import Conflict


ASSET_CACHE_BYTES = 1024*1024*1024
ASSET_TOUCH_INTERVAL = 30


def cache_root():
    base = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache'))
    return base / 'SocialNav' / 'vlm_annotation'


class RemoteBackend:
    def __init__(self, server, project_id, readonly=False, cache_dir=None, transport=None):
        import httpx
        from urllib.parse import urlsplit
        parsed = urlsplit(server)
        if parsed.scheme != 'http' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('服务地址须为不含凭证的http://主机:端口')
        self.server, self.project_id = server.rstrip('/'), project_id
        self.prefix = self.server + '/projects/' + project_id
        key = hashlib.sha256((self.server + '\n' + project_id).encode()).hexdigest()[:20]
        self.directory = Path(cache_dir) if cache_dir else cache_root() / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'cache.sqlite'
        self.lock = (self.directory / '.client.lock').open('a+b')
        self.lock.seek(0)
        self.lock.write(b'0')
        self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise Conflict('此缓存已在另一个窗口使用，请关闭原窗口或选择其他缓存')
        self.http = httpx.Client(timeout=20, trust_env=False, transport=transport)
        self.readonly = readonly
        self.writable, self.notice = False, ''
        self.namespace = ''
        self.memory_frames = {}
        with self.db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS assets(key TEXT PRIMARY KEY,payload BLOB,access REAL);
              CREATE TABLE IF NOT EXISTS queue(seq INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE,request TEXT,state TEXT);
              CREATE TABLE IF NOT EXISTS archives(seq INTEGER PRIMARY KEY AUTOINCREMENT,document TEXT NOT NULL);''')
        self.owner = self.get('owner') or uuid.uuid4().hex
        self.put('owner', self.owner)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def get(self, key):
        with self.db() as db:
            row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, dumps(value)))

    def request(self, method, path, **kwargs):
        response = self.http.request(method, self.prefix + path, **kwargs)
        if response.status_code >= 400:
            try:
                message = response.json().get('detail', response.text)
            except ValueError:
                message = response.text[:500]
            if response.status_code == 404 and path in ('/context-frames', '/delete-event', '/views') and message == 'Not Found':
                raise ValueError('Linux标注服务缺少'+path[1:]+'接口，请更新并重启该服务')
            if response.status_code == 409:
                raise Conflict(str(message))
            raise ValueError(str(message))
        return response

    def snapshot(self):
        self.notice = ''
        result = self.request('GET', '/index').json()
        old = self.get('identity')
        if old is not None and old != result['identity']:
            raise Conflict('服务器工程身份已改变；请保留旧缓存草稿并使用新的缓存目录')
        self.put('identity', result['identity'])
        self.namespace = result['identity'] + '/' + result['input_implementation']
        self.put('display_implementation',result.get('display_implementation'))
        if result['writable'] and not self.readonly:
            try:
                self.request('POST', '/writer', json=dict(owner=self.owner, acquire=True))
                self.writable = True
                self.flush()
                result = self.request('GET', '/index').json()
            except Conflict as error:
                self.writable, self.notice = False, str(error)
        else:
            self.writable = False
        self.put('snapshot', result)
        return self.with_drafts(dict(result, writable=self.writable, notice=self.notice))

    def with_drafts(self, result):
        events = {e['document']['id']: e for e in result['events']}
        with self.db() as db:
            for raw,state in db.execute('SELECT request,state FROM queue ORDER BY seq'):
                request = json.loads(raw)
                if request.get('action') == 'delete':
                    if state == 'queued':
                        events.pop(request['event_id'], None)
                else:
                    events[request['event']['id']] = dict(document=request['event'], revision=request['expected_revision']+1)
        result['events'] = list(events.values())
        return result

    def cached(self, kind, key, loader, binary=False):
        full = self.namespace + ':' + kind + ':' + key
        with self.db() as db:
            row = db.execute('SELECT payload,access FROM assets WHERE key=?', (full,)).fetchone()
            if row:
                if time.time()-row[1]>=ASSET_TOUCH_INTERVAL:
                    db.execute('UPDATE assets SET access=? WHERE key=?', (time.time(), full))
                value = row[0]
                return value if binary else json.loads(value)
        value = loader()
        payload = value if binary else dumps(value).encode()
        if payload is not None:
            with self.db() as db:
                db.execute('INSERT OR REPLACE INTO assets VALUES(?,?,?)', (full, payload, time.time()))
                self.trim_assets(db)
        return value

    def trim_assets(self, db):
        size=db.execute('SELECT COALESCE(SUM(LENGTH(payload)),0) FROM assets').fetchone()[0]
        if size>ASSET_CACHE_BYTES:
            for key,length in db.execute('SELECT key,LENGTH(payload) FROM assets ORDER BY access').fetchall():
                db.execute('DELETE FROM assets WHERE key=?',(key,));size-=length
                if size<=ASSET_CACHE_BYTES*3//4:break

    def frames(self, source):
        rows = self.cached('frames', source, lambda: self.request('GET', '/sources/'+source+'/frames').json())
        self.memory_frames.update({row['id']: row for row in rows})
        return rows

    def frame(self, frame):
        if frame in self.memory_frames and not self.memory_frames[frame].get('lazy'):
            return self.memory_frames[frame]
        return self.cached('frame', frame, lambda: self.request('GET', '/frame', params={'frame_id': frame}).json())

    def window(self, frame):
        return self.cached('window', frame, lambda: self.request('GET', '/window', params={'frame_id': frame}).json())

    def context_frames(self, frame):
        return self.cached('context-frames', frame,
                           lambda: self.request('GET', '/context-frames', params={'frame_id': frame}).json())

    def views(self, frame_ids, review=False):
        from .backend import unpack_views
        kind = ('view-review' if review else 'view-context')+'/'+str(self.get('display_implementation'))
        found,missing,decoded = {},[],{}
        with self.db() as db:
            for ident in frame_ids:
                key = self.namespace+':'+kind+':'+ident
                row = db.execute('SELECT payload,access FROM assets WHERE key=?',(key,)).fetchone()
                if row:
                    payload=row[0]
                    if payload.startswith(b'@'):
                        bundle_key=json.loads(payload[1:])['bundle']
                        if bundle_key not in decoded:
                            bundle=db.execute('SELECT payload,access FROM assets WHERE key=?',(bundle_key,)).fetchone()
                            if bundle is None:
                                missing.append(ident)
                                continue
                            decoded[bundle_key]={view['frame']['id']:view for view in unpack_views(bundle[0])}
                            if time.time()-bundle[1]>=ASSET_TOUCH_INTERVAL:
                                db.execute('UPDATE assets SET access=? WHERE key=?',(time.time(),bundle_key))
                        found[ident]=decoded[bundle_key][ident]
                    else:
                        # Existing individual ZIP cache entries remain readable.
                        found[ident]=unpack_views(payload)[0]
                    if time.time()-row[1]>=ASSET_TOUCH_INTERVAL:
                        db.execute('UPDATE assets SET access=? WHERE key=?',(time.time(),key))
                else:
                    missing.append(ident)
        if missing:
            payload = self.request('POST','/views',json=dict(frame_ids=missing,review=bool(review))).content
            rows = unpack_views(payload)
            if [row['frame']['id'] for row in rows]!=missing or any(row.get('review')!=bool(review) for row in rows):
                raise ValueError('观察响应与请求不匹配')
            # Persist the server's compressed bundle once, without re-encoding and
            # fsyncing each individual view on the playback path.
            bundle_key=self.namespace+':view-bundle:'+hashlib.sha256(payload).hexdigest()
            now=time.time()
            manifest=b'@'+dumps(dict(bundle=bundle_key)).encode()
            with self.db() as db:
                db.execute('INSERT OR REPLACE INTO assets VALUES(?,?,?)',(bundle_key,payload,now))
                for row in rows:
                    ident=row['frame']['id'];found[ident]=row
                    db.execute('INSERT OR REPLACE INTO assets VALUES(?,?,?)',(self.namespace+':'+kind+':'+ident,manifest,now))
                self.trim_assets(db)
        return [found[ident] for ident in frame_ids]

    def asset(self, frame, kind):
        def load():
            response = self.request('GET', '/asset', params=dict(frame_id=frame, kind=kind))
            if len(response.content) > 24*1024*1024:
                raise ValueError('资产超过客户端上限')
            return None if response.status_code == 204 else response.content
        return self.cached('asset/'+kind, frame, load, True)

    def review_images(self, frame):
        return [self.cached('review/'+str(index), frame,
                            lambda i=index: self.request('GET', '/review-image', params=dict(frame_id=frame, index=i)).content,
                            True) for index in range(3)]

    def pending(self):
        with self.db() as db:
            return db.execute('SELECT COUNT(*) FROM queue').fetchone()[0]

    def flush(self):
        while True:
            with self.db() as db:
                row = db.execute('SELECT id,request,state FROM queue ORDER BY seq LIMIT 1').fetchone()
            if row is None:
                return
            operation, raw, state = row
            if state != 'queued':
                raise Conflict('待同步草稿存在冲突；先导出草稿并处理，不能自动覆盖')
            request = json.loads(raw)
            response = self.request('GET', '/receipt/' + operation).json()
            if response is None:
                try:
                    endpoint = '/delete-event' if request.get('action') == 'delete' else '/commit'
                    response = self.request('POST', endpoint, json=request).json()
                except (Conflict, ValueError):
                    with self.db() as db:
                        db.execute('UPDATE queue SET state=? WHERE id=?', ('conflict', operation))
                    raise
            self.put('last_ack', response)
            with self.db() as db:
                db.execute('DELETE FROM queue WHERE id=?', (operation,))

    def commit(self, event, revision):
        request = dict(project_id=self.get('identity'), writer_id=self.owner, operation_id=uuid.uuid4().hex,
                       expected_revision=revision, event=event)
        return self.submit(request, dict(revision=revision+1, document=event))

    def delete_event(self, ident, revision):
        request = dict(project_id=self.get('identity'), writer_id=self.owner, operation_id=uuid.uuid4().hex,
                       expected_revision=revision, action='delete', event_id=ident)
        return self.submit(request, dict(revision=revision+1, event_id=ident, deleted=True))

    def submit(self, request, staged):
        if not self.writable:
            raise Conflict('当前为只读')
        with self.db() as db:
            db.execute('INSERT INTO queue(id,request,state) VALUES(?,?,?)', (request['operation_id'], dumps(request), 'queued'))
        try:
            self.flush()
            return self.get('last_ack')
        except (Conflict, ValueError):
            self.writable = False
            raise
        except Exception as error:
            import httpx
            if not isinstance(error, httpx.TransportError):
                raise
            return dict(staged, operation_id=request['operation_id'], staged=True)

    def renew(self):
        if self.writable:
            self.request('POST', '/writer', json=dict(owner=self.owner))

    def drafts(self):
        with self.db() as db:
            return dict(pending=[json.loads(r[0]) for r in db.execute('SELECT request FROM queue ORDER BY seq')],
                        archives=[json.loads(r[0]) for r in db.execute('SELECT document FROM archives ORDER BY seq')])

    def archive_conflicts(self, current_event=None):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            requests = [json.loads(r[0]) for r in db.execute('SELECT request FROM queue ORDER BY seq')]
            db.execute('INSERT INTO archives(document) VALUES(?)',
                       (dumps(dict(created_at=time.time(), pending=requests, current_event=current_event)),))
            db.execute('DELETE FROM queue')
        return len(requests)

    def close(self):
        try:
            if self.writable:
                try:
                    self.request('POST', '/writer', json=dict(owner=self.owner, release=True))
                except Exception:
                    # Release is best effort; a disconnected session expires. Labels remain queued.
                    pass
        finally:
            self.http.close()
            self.lock.close()
