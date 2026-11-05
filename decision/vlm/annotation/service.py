"""Independent, allowlisted T1/T2 API on loopback or Tailscale."""
import ipaddress
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware

from .store import Conflict, Store


def create_app(project, project_id='wheelchair-vlm', writable=False):
    store = Store(project)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            store.close()
    app = FastAPI(title='Social-Nav T1/T2 annotation', docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1500)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({'detail': str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({'detail': str(error)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({'detail': str(error)}, status_code=404)

    @app.middleware('http')
    async def bounded(request, next_call):
        size = request.headers.get('content-length', '0')
        if not size.isdigit() or int(size) > 8*1024*1024:
            return JSONResponse({'detail': '请求过大'}, status_code=413)
        if request.method == 'POST':
            body = await request.body()
            if len(body) > 8*1024*1024:
                return JSONResponse({'detail': '请求过大'}, status_code=413)
        return await next_call(request)

    def check(ident):
        if ident != project_id:
            raise KeyError('工程不存在')

    @app.get('/health')
    def health():
        return dict(kind='vlm_decision_annotation', writable=writable)

    @app.get('/projects')
    def projects():
        return [dict(id=project_id, writable=writable)]

    @app.get('/projects/{ident}/index')
    def index(ident: str):
        check(ident)
        result = store.snapshot()
        for source in result['sources']:
            source.pop('bag', None)
            source.pop('records', None)
            source.pop('reader', None)
            source.pop('rgb_recording', None)
        return dict(result, writable=writable)

    @app.get('/projects/{ident}/sources/{source}/frames')
    def frames(ident: str, source: str):
        check(ident)
        return store.frames(source)

    @app.get('/projects/{ident}/frame')
    def frame(ident: str, frame_id: str):
        check(ident)
        return store.frame(frame_id)

    @app.get('/projects/{ident}/window')
    def window(ident: str, frame_id: str):
        check(ident)
        return store.window(frame_id)

    @app.get('/projects/{ident}/context-frames')
    def context_frames(ident: str, frame_id: str):
        check(ident)
        return store.context_frames(frame_id)

    @app.post('/projects/{ident}/views')
    def views(ident: str, body: dict):
        check(ident)
        ids, review = body.get('frame_ids'),body.get('review',False)
        if not isinstance(ids,list) or not 1<=len(ids)<=12 or not all(isinstance(fid,str) for fid in ids) or type(review) is not bool:
            raise ValueError('观察批次须为1–12个帧ID，review须为布尔值')
        from .backend import build_views,pack_views
        started=time.perf_counter()
        rows=build_views(store,ids,review)
        built=time.perf_counter()
        payload=pack_views(rows)
        packed=time.perf_counter()
        return Response(payload,media_type='application/zip',headers={'Content-Encoding':'identity',
            'Server-Timing':f'view;dur={(built-started)*1000:.1f},pack;dur={(packed-built)*1000:.1f}'})

    @app.get('/projects/{ident}/asset')
    def asset(ident: str, frame_id: str, kind: str):
        check(ident)
        data = store.asset(frame_id, kind)
        if data is None:
            return Response(status_code=204)
        return Response(data, media_type='application/octet-stream')

    @app.get('/projects/{ident}/review-image')
    def review_image(ident: str, frame_id: str, index: int):
        check(ident)
        if not 0 <= index <= 2:
            raise ValueError('观察图序号须为0～2')
        from .backend import render_review
        return Response(render_review(store, frame_id)[index], media_type='image/png')

    @app.post('/projects/{ident}/writer')
    def writer(ident: str, body: dict):
        check(ident)
        if not writable:
            raise Conflict('服务以只读方式启动；编辑需Linux启动时显式加--write')
        return store.writer(body['owner'], acquire=body.get('acquire', False), release=body.get('release', False))

    @app.get('/projects/{ident}/receipt/{operation}')
    def receipt(ident: str, operation: str):
        check(ident)
        return store.receipt(operation)

    @app.post('/projects/{ident}/commit')
    def commit(ident: str, body: dict):
        check(ident)
        if not writable:
            raise Conflict('只读工程')
        if body.get('action') is not None:
            raise ValueError('事件删除请使用delete-event接口')
        return store.commit(body)

    @app.post('/projects/{ident}/delete-event')
    def delete_event(ident: str, body: dict):
        check(ident)
        if not writable:
            raise Conflict('只读工程')
        if body.get('action') != 'delete':
            raise ValueError('删除操作类型错误')
        return store.commit(body)

    return app


def serve(args):
    host = ipaddress.ip_address(args.host)
    if not (host.is_loopback or host in ipaddress.ip_network('100.64.0.0/10')):
        raise ValueError('请绑定当前Tailscale IP或127.0.0.1')
    import uvicorn
    uvicorn.run(create_app(args.project, args.project_id, args.write), host=args.host, port=args.port,
                access_log=False)
