"""Allowlisted annotation API on loopback/Tailscale. HTTP and ROS stay separate."""
from contextlib import asynccontextmanager, closing
import ipaddress
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware

from .backend import VIEW_METADATA
from .coordination import Conflict, WriterBusy, commit, initialize, receipt, writer
from .data import Store, dumps
from .protocol import FEATURES, FORMAT, MAX_BUNDLE, digest, public_frame, public_source


ROOT = Path(__file__).resolve().parents[3]


class RosAssets:
    def __init__(self, project, python):
        self.project, self.python = project, python
        self.process, self.lock = None, threading.Lock()

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(3)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def read_exact(self, size, deadline):
        result = bytearray()
        descriptor = self.process.stdout.fileno()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([descriptor], [], [], max(0, remaining))[0]:
                raise TimeoutError('ROS资产读取超时')
            chunk = os.read(descriptor, min(size - len(result), 1024 * 1024))
            if not chunk:
                raise RuntimeError('ROS资产子进程意外退出；检查ROS解释器和环境入口')
            result.extend(chunk)
        return bytes(result)

    def read(self, ident, encoding='jpeg'):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.stop()
                self.process = subprocess.Popen([str(self.python), '-B', '-m',
                    'perception.detection_tracking.annotation.asset_worker', '--project', str(self.project)],
                    cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                self.process.stdin.write((json.dumps(dict(id=ident, encoding=encoding)) + '\n').encode())
                self.process.stdin.flush()
                deadline = time.monotonic() + 25
                size, = struct.unpack('!I', self.read_exact(4, deadline))
                if not 0 < size <= MAX_BUNDLE + 1:
                    raise ValueError('ROS资产长度错误')
                result = self.read_exact(size, deadline)
                if result[:1] == b'E':
                    raise ValueError(json.loads(result[1:])['detail'])
                if result[:1] != b'Z':
                    raise ValueError('ROS资产响应格式错误')
                return result[1:]
            except Exception:
                self.stop()
                raise  # Never replay a failed request silently.

    def close(self):
        with self.lock:
            self.stop()


class Project:
    def __init__(self, ident, entry, ros_python, assets_factory):
        self.ident, self.path = ident, Path(entry['path']).expanduser().resolve()
        self.read_only, self.lock_fd = entry.get('read_only', True), None
        self.assets = assets_factory(self.path, ros_python)
        try:
            with closing(Store(self.path, read_only=self.read_only)) as store:
                if not store.sources() or any(not source['indexed'] for source in store.sources()):
                    raise ValueError('服务只接收已有索引的标注工程；先使用离线predict／本机open准备')
                if not self.read_only:
                    import fcntl
                    self.lock_fd = os.open(str(self.path / 'annotations.sqlite'), os.O_RDONLY)
                    fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    initialize(store)
                self.identity = digest(dict(bounds=store.bounds, sources=[dict(id=s['id'], fingerprint=s['fingerprint'],
                    topic=s['topic'], rgb_topic=s['rgb_topic']) for s in store.sources()]))
        except Exception:
            self.close()
            raise

    def close(self):
        self.assets.close()
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None

    def snapshot(self):
        with closing(Store(self.path, read_only=True)) as store:
            store.connection.execute('BEGIN')
            rows = store.rows()
            for row in rows:
                review = json.loads(row['review'])
                row['review_hash'] = digest(review)
                row['review'] = dumps(dict(status=review['status']))
            return dict(identity=self.identity, writable=not self.read_only, bounds=store.bounds,
                        sources=[public_source(s) for s in store.sources()], rows=rows,
                        metadata={key: store.meta(key) for key in VIEW_METADATA})

    def frame(self, source, ordinal):
        with closing(Store(self.path, read_only=True)) as store:
            row = store.connection.execute('SELECT id FROM frames WHERE source_id=? AND ordinal=?', (source, ordinal)).fetchone()
            if row is None:
                raise KeyError('扫描不存在')
            return public_frame(store.frame(row['id']))


def create_app(config, assets_factory=RosAssets):
    server_id = config.get('server_id')
    if not isinstance(server_id, str) or not 1 <= len(server_id) <= 128:
        raise ValueError('配置需要稳定的server_id')
    projects, runtime = {}, uuid.uuid4().hex
    try:
        for ident, entry in config['projects'].items():
            if not ident or len(ident) > 64 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in ident):
                raise ValueError('工程ID只支持字母、数字、下划线和短横线')
            if len(projects) >= 8:
                raise ValueError('单服务最多登记8个工程')
            projects[ident] = Project(ident, entry, config.get('ros_python', '/usr/bin/python3'), assets_factory)
    except Exception:
        for project in projects.values():
            project.close()
        raise

    @asynccontextmanager
    async def lifespan(app):
        yield
        for project in projects.values():
            project.close()

    app = FastAPI(title='People annotation', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.state.projects = projects

    @app.middleware('http')
    async def no_store(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    for error_type, status in ((Conflict, 409), (WriterBusy, 423), (KeyError, 404), (ValueError, 400),
                               (TimeoutError, 504), (RuntimeError, 503)):
        async def handler(request, error, code=status):
            return JSONResponse({'detail': str(error)}, status_code=code)
        app.add_exception_handler(error_type, handler)

    def project_for(ident):
        if ident not in projects:
            raise KeyError('工程未登记')
        return projects[ident]

    def bound(project, payload):
        if (payload.get('server_id') != server_id or payload.get('project_id') != project.ident or
                payload.get('project_identity') != project.identity):
            raise Conflict('服务／工程身份与原请求不同，禁止写入')

    async def body(request):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 128 * 1024:
                raise ValueError('写入请求超过128KiB')
            chunks.append(chunk)
        value = json.loads(b''.join(chunks))
        if not isinstance(value, dict):
            raise ValueError('请求必须是JSON对象')
        return value

    @app.get('/health')
    def health():
        return dict(status='ok')

    @app.get('/capabilities')
    def capabilities():
        return dict(format=FORMAT, features=FEATURES, server_id=server_id, runtime_id=runtime, max_bundle_bytes=MAX_BUNDLE)

    @app.get('/projects')
    def list_projects():
        return [dict(id=p.ident, writable=not p.read_only) for p in projects.values()]

    @app.get('/projects/{ident}/index')
    def index(ident: str):
        return project_for(ident).snapshot()

    @app.get('/projects/{ident}/sources/{source}/frames/{ordinal}')
    def frame(ident: str, source: str, ordinal: int):
        return project_for(ident).frame(source, ordinal)

    @app.get('/projects/{ident}/sources/{source}/frames/{ordinal}/bundle')
    def bundle(ident: str, source: str, ordinal: int, encoding: str = 'jpeg'):
        if encoding not in ('jpeg', 'png'):
            raise ValueError('仅支持jpeg预览或png无损图像')
        project = project_for(ident)
        state = project.frame(source, ordinal)
        payload = project.assets.read(state['id'], encoding)
        return Response(payload, media_type='application/zip', headers={'Content-Encoding': 'identity'})

    @app.post('/projects/{ident}/writer')
    async def session(ident: str, request: Request):
        payload = await body(request)
        project = project_for(ident)
        bound(project, payload)
        if project.read_only:
            raise WriterBusy('该工程登记为只读')
        # Blocking SQLite belongs in the worker pool, including lease renewals.
        from starlette.concurrency import run_in_threadpool
        def apply():
            with closing(Store(project.path)) as store:
                return writer(store, payload.get('action'), payload.get('client_id'), payload.get('session'))
        return await run_in_threadpool(apply)

    @app.get('/projects/{ident}/mutations/{mutation_id}')
    def get_receipt(ident: str, mutation_id: str):
        with closing(Store(project_for(ident).path, read_only=True)) as store:
            result = receipt(store, mutation_id)
            if result is None:
                raise KeyError('未提交')
            return result

    @app.post('/projects/{ident}/mutations')
    async def mutate(ident: str, request: Request):
        payload = await body(request)
        project = project_for(ident)
        bound(project, payload)
        if project.read_only:
            raise WriterBusy('该工程登记为只读')
        from starlette.concurrency import run_in_threadpool
        def apply():
            with closing(Store(project.path)) as store:
                try:
                    return commit(store, payload)
                except (KeyError, TypeError) as error:
                    raise ValueError('写入字段缺失或类型错误') from error
        return await run_in_threadpool(apply)

    return app


def serve(args):
    import uvicorn
    config = json.loads(args.config.read_text(encoding='utf-8'))
    host = args.host or '127.0.0.1'
    address = ipaddress.ip_address(host)
    if not (address.is_loopback or address in ipaddress.ip_network('100.64.0.0/10') or
            address in ipaddress.ip_network('fd7a:115c:a1e0::/48')):
        raise ValueError('请绑定回环或明确的Tailscale接口地址')
    app = create_app(config)
    uvicorn.run(app, host=host, port=args.port, access_log=False, log_level='warning',
                limit_concurrency=16, timeout_keep_alive=5, server_header=False)
