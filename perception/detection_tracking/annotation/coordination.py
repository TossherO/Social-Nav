"""SQLite transactions shared by HTTP writes and the local editor's write guards."""
import json
import time
import uuid

from .data import Store, dumps
from .protocol import digest, public_frame, public_source, validate_payload


class Conflict(ValueError):
    pass


class WriterBusy(ValueError):
    pass


def initialize(store):
    store.connection.executescript('''
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS annotation_writer (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), session TEXT NOT NULL,
            client TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS annotation_sources (
            source_id TEXT PRIMARY KEY REFERENCES sources(id), epoch INTEGER NOT NULL, revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS annotation_frames (
            frame_id TEXT PRIMARY KEY REFERENCES frames(id), revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS annotation_receipts (
            mutation_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, result TEXT NOT NULL, created REAL NOT NULL);
        COMMIT;
    ''')


def writer(store, action, client, session=None, ttl=90.):
    if not isinstance(client, str) or not 1 <= len(client) <= 128:
        raise ValueError('客户端身份错误')
    with store.connection:
        store.connection.execute('BEGIN IMMEDIATE')
        row = store.connection.execute('SELECT * FROM annotation_writer WHERE singleton=1').fetchone()
        active = row is not None and row['expires'] > time.time()
        ours = row is not None and row['client'] == client and row['session'] == session
        if action == 'release':
            if ours:
                store.connection.execute('DELETE FROM annotation_writer WHERE singleton=1')
            return dict(released=ours)
        if action == 'acquire':
            if active and not ours:
                raise WriterBusy('工程已有写入会话；请先在另一设备完成同步并关闭，或以只读模式打开')
            session = session if active and ours else uuid.uuid4().hex
        elif action != 'renew' or not active or not ours:
            raise WriterBusy('写入会话已过期或不属于本客户端；请重新连接')
        expires = time.time() + ttl
        store.connection.execute('INSERT OR REPLACE INTO annotation_writer VALUES (1,?,?,?)', (session, client, expires))
        return dict(session=session, expires=expires, ttl=ttl)


def receipt(store, ident):
    if not store.coordinated():
        return None
    row = store.connection.execute('SELECT result FROM annotation_receipts WHERE mutation_id=?', (ident,)).fetchone()
    return json.loads(row['result']) if row else None


def commit(store, request):
    """One transaction: authorization + expected state + label + revision + receipt.

    The same operation ID/body can be acknowledged again after lease expiry or a
    service restart. A reused ID with another body is always a conflict.
    """
    if not isinstance(request, dict):
        raise ValueError('写入请求不是对象')
    ident = request.get('mutation_id', '')
    if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
        raise ValueError('操作ID错误')
    hashed = digest({key: value for key, value in request.items() if key != 'session'})
    with store.connection:
        store.connection.execute('BEGIN IMMEDIATE')
        existing = store.connection.execute('SELECT * FROM annotation_receipts WHERE mutation_id=?', (ident,)).fetchone()
        if existing:
            if existing['request_hash'] != hashed:
                raise Conflict('同一个操作ID不能用于不同内容')
            return json.loads(existing['result'])
        lease = store.connection.execute('SELECT * FROM annotation_writer WHERE singleton=1').fetchone()
        if (lease is None or lease['expires'] <= time.time() or lease['client'] != request.get('client_id') or
                lease['session'] != request.get('session')):
            raise WriterBusy('没有有效的工程写入会话')
        source_id, kind = request['source_id'], request['kind']
        source = store.source(source_id)
        if source['epoch'] != request['epoch']:
            raise Conflict('该子包已被清空或重建；旧请求不会再次写入')
        result = dict(mutation_id=ident, kind=kind, source_id=source_id)
        if kind == 'review':
            frame = store.frame(request['frame_id'])
            if (frame['source_id'] != source_id or not frame['fingerprint'] or
                    frame['fingerprint'] != request['fingerprint']):
                raise Conflict('标注与来源／原始扫描绑定不一致')
            if frame['revision'] != request['revision'] or digest(frame['review']) != request['expected_hash']:
                raise Conflict('标注已被修改；保留本机草稿后处理冲突，不覆盖服务器结果')
            validate_payload(request['review'], store.bounds, frame['candidates'])
            store._save_review(frame['id'], request['review'], expected=frame['review'])
            result['frame'] = public_frame(store.frame(frame['id']))
        elif kind in ('group', 'reset'):
            if source['revision'] != request['revision']:
                raise Conflict('子包内容或分组已改变；请先同步后再操作')
            if kind == 'group':
                group = request['group']
                if (set(group) != {'root_group', 'location_group', 'split'} or
                        any(not isinstance(value, str) or len(value) > 256 for value in group.values())):
                    raise ValueError('分组字段错误')
                store._set_group(source_id, **group)
            else:
                result.update(store._reset_source_reviews(source_id))
                current = request.get('current')
                frame = store.frame(current) if current is not None else None
                if frame is not None and frame['source_id'] != source_id:
                    raise ValueError('当前帧不属于待清空子包')
                result['frame'] = public_frame(frame) if frame is not None else None
        else:
            raise ValueError('不支持的写入操作')
        result['source'] = public_source(store.source(source_id))
        store.connection.execute('INSERT INTO annotation_receipts VALUES (?,?,?,?)',
                                 (ident, hashed, dumps(result), time.time()))
        return result
