"""Background HTTP backend, durable local drafts and receipt-first reconnection."""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile
from urllib.parse import quote, urlsplit

import httpx

from .backend import VIEW_METADATA
from .cache import Cache
from .protocol import FORMAT, MAX_BUNDLE, unpack_assets, validate_payload


class RemoteError(ValueError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def default_cache(server, project):
    base = Path(os.environ['LOCALAPPDATA']) if os.name == 'nt' and os.environ.get('LOCALAPPDATA') else Path.home() / '.cache'
    namespace = hashlib.sha256((server.rstrip('/') + '\n' + project).encode()).hexdigest()[:24]
    return base / 'SocialNav' / 'annotation' / namespace


class RemoteBackend:
    is_remote = True

    def __init__(self, server, project, cache_dir=None, read_only=False, encoding='jpeg', transport=None):
        parsed = urlsplit(server)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or
                parsed.query or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('--server须为无凭证、查询参数或路径的HTTP(S)地址')
        if not project or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in project):
            raise ValueError('工程ID格式错误')
        self.server, self.project = server.rstrip('/'), project
        self.base = '/projects/' + quote(project, safe='')
        self.cache = Cache(cache_dir or default_cache(server, project))
        self.explicit_read_only, self.read_only, self.encoding = read_only, read_only, encoding
        self.http = httpx.Client(base_url=self.server,
                                 timeout=httpx.Timeout(30., connect=5., write=10., pool=5.),
                                 follow_redirects=False, trust_env=False, transport=transport)
        self.sync_lock, self.status_lock = threading.RLock(), threading.Lock()
        self.stop, self.wake = threading.Event(), threading.Event()
        self.thread, self.session, self.identity = None, None, None
        self.lease_checked, self.refresh_serial = 0., 0
        self.blocked = False
        self.closed = False
        self.locations, self.bounds, self.writable = {}, None, False
        self._status = dict(online=False, pending=0, conflicts=0, archived=0, generation=0, reset_pending=False, error='', read_only=read_only,
                            refresh_serial=0, ready=False)

    def request(self, method, path, body=None, binary=False, timeout=30.):
        budget = MAX_BUNDLE if binary else 16 * 1024 * 1024
        with self.http.stream(method, path, json=body, timeout=timeout) as response:
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > budget:
                    raise ValueError('服务响应超过预算')
                chunks.append(chunk)
            payload = b''.join(chunks)
            if response.status_code >= 300:
                try:
                    detail = json.loads(payload).get('detail', '服务请求失败')
                except (ValueError, AttributeError):
                    detail = '服务请求失败'
                raise RemoteError(response.status_code, str(detail))
            return payload if binary else json.loads(payload)

    def status(self):
        with self.status_lock:
            return copy.deepcopy(self._status)

    def publish(self, **values):
        counts = self.cache.counts()
        with self.status_lock:
            if counts['generation'] >= self._status['generation']:
                self._status.update(counts)
            self._status.update(read_only=self.read_only, refresh_serial=self.refresh_serial, **values)

    def ensure_lease(self, force=False):
        if self.explicit_read_only or not self.writable:
            self.read_only = True
            return
        if not force and self.session and time.monotonic() - self.lease_checked < 20:
            return
        session = self.session or self.cache.meta('session')
        body = dict(action='acquire', client_id=self.cache.meta('client_id'), session=session, **self.identity)
        try:
            value = self.request('POST', self.base + '/writer', body)
        except RemoteError as error:
            if error.status == 423:
                self.read_only = True
            raise
        self.session, self.read_only = value['session'], False
        self.cache.set_meta('session', self.session)
        self.lease_checked = time.monotonic()

    def prepare(self, progress, detections=None):
        if detections is not None:
            raise ValueError('远程模式不导入检测文件；请在服务器离线准备')
        self.cache.lock()
        progress('连接标注数据服务……')
        try:
            capabilities = self.request('GET', '/capabilities')
            if (capabilities['format'] != FORMAT or
                    not {'review-cas', 'mutation-receipts', 'writer-lease', 'npy-xyzi'}.issubset(capabilities['features'])):
                raise ValueError('服务能力不兼容，禁止写入')
            snapshot = self.request('GET', self.base + '/index')
            self.identity = dict(server_id=capabilities['server_id'], project_id=self.project, project_identity=snapshot['identity'])
            self.cache.bind(self.identity, snapshot)
            self.use_index(snapshot)
            self.read_only = self.explicit_read_only or not snapshot['writable']
            try:
                self.ensure_lease(True)
                self.publish(online=True, ready=True, error='')
            except RemoteError as error:
                if error.status != 423:
                    raise
                self.publish(online=True, ready=True, error=str(error))
        except httpx.TransportError:
            self.identity = self.cache.meta('identity')
            snapshot = self.cache.meta('index')
            if self.identity is None or snapshot is None:
                raise ValueError('首次连接失败；请确认服务已启动、Tailscale连通及地址正确')
            self.read_only = self.explicit_read_only or not snapshot['writable']
            self.publish(online=False, ready=True, error='服务暂不可达；仅可使用已缓存扫描，编辑在本机暂存')
        if self.thread is None:
            self.use_index(snapshot)
            self.thread = threading.Thread(target=self.sync_loop, name='annotation-sync', daemon=True)
            self.thread.start()
        self.wake.set()
        return self.cache.snapshot()

    def use_index(self, snapshot):
        self.locations = {row['id']: (row['source_id'], row['ordinal']) for row in snapshot['rows']}
        self.bounds, self.writable = snapshot['bounds'], snapshot['writable']

    def locate(self, ident):
        # Original IDs can contain slashes and non-padded ordinals. Use the index,
        # never reinterpret a Linux path or infer another frame from a timestamp.
        row = self.locations.get(ident)
        if row is None:
            raise KeyError('扫描不属于当前工程')
        return self.base + '/sources/' + quote(row[0], safe='') + '/frames/' + str(row[1])

    def read_frame(self, ident):
        frame, bundle = None, self.cache.bundle(ident, self.encoding)
        decoded = None
        if self.status()['online']:
            try:
                if bundle is None:
                    # A cold frame needs ONE round trip: the bundle already has
                    # current review/candidates and immutable observation metadata.
                    bundle = self.request('GET', self.locate(ident) + '/bundle?encoding=' + self.encoding, binary=True)
                    decoded = unpack_assets(bundle, ident, self.encoding)
                    frame = decoded[0]
                    self.cache.save_frame(frame, bundle, self.encoding)
                else:
                    frame = self.request('GET', self.locate(ident))
                    self.cache.save_frame(frame)
            except httpx.TransportError:
                self.publish(online=False, error='连接中断；使用本机缓存，修改尚未同步')
        if frame is None:
            frame = self.cache.state(ident)
        if bundle is None:
            raise ValueError('所选编码的扫描资产尚未缓存，需要联网读取；未使用旧帧替代')
        if decoded is None:
            try:
                decoded = unpack_assets(bundle, ident, self.encoding)
            except (ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                self.cache.drop_bundle(ident)
                raise ValueError('本机资产缓存损坏，草稿保留；联网后重新选择此帧可重新下载') from error
        bound, points, image, image_ns, notice = decoded
        if bound['fingerprint'] != frame['fingerprint']:
            raise ValueError('状态与资产的扫描绑定不一致')
        working = self.cache.state(ident)
        if working['fingerprint'] != bound['fingerprint']:
            raise ValueError('当前草稿与资产输入绑定不一致')
        if self.encoding == 'jpeg' and image is not None:
            notice = (notice + ' RGB为原分辨率JPEG预览；可用--image-encoding png获取无损图。').strip()
        return working, points, image, image_ns, notice

    def set_meta(self, key, value):
        if key not in VIEW_METADATA:
            raise ValueError('不是界面设置')
        self.cache.set_meta('view:' + key, value)

    def save_position(self, frame):
        with self.cache.open() as database:
            database.execute('BEGIN IMMEDIATE')
            positions = self.cache.get(database, 'view:last_frames') or {}
            positions[frame['source_id']] = frame['id']
            self.cache.put(database, 'view:last_frames', positions)
            self.cache.put(database, 'view:last_frame', frame['id'])

    def assert_editable(self):
        # Already accepted GUI edits must still reach the LOCAL durable journal
        # if the lease expires or a conflict arrives while they are being saved.
        # The window blocks new edits; only a valid server lease permits delivery.
        if self.explicit_read_only or not self.writable:
            raise ValueError('该工程／客户端为只读')

    def save_review(self, ident, review, expected):
        self.assert_editable()
        frame = self.cache.state(ident)
        validate_payload(review, self.bounds, frame['candidates'])
        self.cache.enqueue('review', frame['source_id'], review, expected, ident)
        self.publish()
        self.wake.set()
        return dict(frame, review=copy.deepcopy(review))

    def save_group(self, ident, root_group, location_group, split):
        self.assert_editable()
        if split not in ('unassigned', 'train', 'val', 'holdout'):
            raise ValueError('未知划分')
        group = dict(root_group=root_group.strip(), location_group=location_group.strip(), split=split)
        if any(len(value) > 256 for value in group.values()):
            raise ValueError('分组字段过长')
        self.cache.enqueue('group', ident, group=group)
        self.publish()
        self.wake.set()
        with self.cache.open() as database:
            return self.cache.source_in(database, ident)

    def deliver(self, item):
        body = json.loads(item['body'])
        result = None
        if item['attempted']:
            try:
                result = self.request('GET', self.base + '/mutations/' + body['mutation_id'])
            except RemoteError as error:
                if error.status != 404:
                    raise
        if result is None:
            self.ensure_lease()
            self.cache.attempted(body['mutation_id'])  # Durable BEFORE the possibly successful HTTP request.
            result = self.request('POST', self.base + '/mutations', dict(body, session=self.session))
        self.cache.acknowledge(result)
        if body['kind'] == 'reset':
            self.refresh_serial += 1
        return result

    def sync_once(self, budget=32):
        with self.sync_lock:
            if not self.status()['online']:
                capabilities = self.request('GET', '/capabilities')
                snapshot = self.request('GET', self.base + '/index')
                identity = dict(server_id=capabilities['server_id'], project_id=self.project, project_identity=snapshot['identity'])
                self.cache.bind(identity, snapshot)
                self.identity = identity
                self.use_index(snapshot)
                self.lease_checked = 0.
            self.publish(online=True, error='')
            if self.explicit_read_only or not self.writable:
                capabilities = self.request('GET', '/capabilities')
                if capabilities['server_id'] != self.identity['server_id'] or capabilities['format'] != FORMAT:
                    raise ValueError('服务身份／格式改变，停止同步')
                return
            if self.cache.first() is None:
                self.ensure_lease()
            for _ in range(budget):
                if self.stop.is_set():
                    break
                item = self.cache.first()
                if item is None:
                    break
                if item['status'] == 'conflict':
                    self.publish(error=item['error'])
                    return
                try:
                    self.deliver(item)
                except RemoteError as error:
                    if error.status in (400, 404, 409, 422):
                        self.cache.conflict(item['mutation_id'], str(error))
                    if error.status in (401, 403):
                        self.blocked = True
                    raise
                self.publish(online=True, error='')
            self.publish(online=True, error='')

    def sync_loop(self):
        while not self.stop.is_set():
            self.wake.wait(3.)
            self.wake.clear()
            if self.stop.is_set():
                break
            if self.blocked:
                continue
            try:
                self.sync_once()
                state = self.status()
                if state['pending'] and not state['conflicts'] and not self.read_only:
                    self.wake.set()
            except httpx.TransportError:
                self.publish(online=False, error='连接中断；本机草稿已保留，恢复后先查询原操作回执')
            except (RemoteError, ValueError, KeyError) as error:
                if isinstance(error, RemoteError) and error.status in (401, 403):
                    self.blocked = True
                self.publish(error=str(error))

    def synchronize(self):
        with self.sync_lock:
            self.blocked = False
            self.sync_once(budget=128)
            snapshot = self.request('GET', self.base + '/index')
            if snapshot['identity'] != self.identity['project_identity']:
                raise ValueError('工程身份已变化，停止同步')
            self.cache.update_index(snapshot)
            self.use_index(snapshot)
            return self.cache.snapshot()

    def resolve_conflicts(self):
        with self.sync_lock:
            snapshot = self.request('GET', self.base + '/index')
            if snapshot['identity'] != self.identity['project_identity']:
                raise ValueError('工程身份已变化，原草稿保留；请使用新的缓存目录')
            self.cache.archive_conflicts()
            self.cache.update_index(snapshot)
            self.use_index(snapshot)
            self.refresh_serial += 1
            self.publish(error='')
            self.wake.set()
            return self.cache.snapshot()

    def reset_source_reviews(self, ident, current=None):
        self.assert_editable()
        with self.sync_lock:
            self.synchronize()
            body = self.cache.enqueue('reset', ident, current=current)
            self.publish()
            try:
                return self.deliver(self.cache.first())
            except Exception:
                self.wake.set()  # Same ID/receipt recovery, never a second reset.
                raise
            finally:
                self.publish()

    def close(self):
        if self.closed:
            return
        self.stop.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(35.)
            if self.thread.is_alive():
                raise RuntimeError('同步任务仍未结束；本机队列保留')
        if self.session is not None and self.identity is not None:
            try:
                self.request('POST', self.base + '/writer', dict(action='release', client_id=self.cache.meta('client_id'),
                             session=self.session, **self.identity), timeout=2.)
            except (httpx.TransportError, RemoteError):
                pass  # The server lease expires; durable drafts remain.
        self.http.close()
        self.cache.close()
        self.closed = True
