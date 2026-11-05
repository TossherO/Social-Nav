"""Blocking annotation operations. The Qt window calls these in workers only."""
from functools import lru_cache
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import signal
import struct
import subprocess
import sys
import uuid
import zipfile

from decision.vlm.scene_inputs import font,MODEL_SIZE,RGB_RECT,rgb_geometry,draw_rgb_people
from .store import Conflict, Store
from .rules import dumps, require
from .assets import LazyAssets,ROOT


class LocalBackend:
    def __init__(self, project, readonly=False):
        self.store = Store(project)
        self.owner = uuid.uuid4().hex
        self.writable = not readonly
        self.notice = ''
        self.worker=ViewWorker(self.store.path,sys.executable) if os.name=='posix' else None
        if self.worker is not None:
            self.worker.source_binding=[self.store.identity,self.store.asset_identity,
                                        self.store.rgb_cache.identity if self.store.rgb_cache else None,self.store.view_binding]
        if self.writable:
            try:
                self.store.writer(self.owner, acquire=True)
            except Conflict as error:
                self.writable, self.notice = False, str(error)

    def snapshot(self):
        return dict(self.store.snapshot(), writable=self.writable, notice=self.notice)

    @lru_cache(maxsize=8)
    def frames(self, source):
        return self.store.frames(source)

    def window(self, frame):
        return self.worker.request('window',frame) if self.worker else self.store.window(frame)

    def frame(self, frame):
        return self.worker.request('frame',frame) if self.worker else self.store.frame(frame)

    @lru_cache(maxsize=100)
    def asset(self, frame, kind):
        return self.worker.request('asset',frame,kind) if self.worker else self.store.asset(frame, kind)

    def commit(self, event, revision):
        if not self.writable:
            raise Conflict('当前为只读')
        request=dict(project_id=self.store.identity,writer_id=self.owner,operation_id=uuid.uuid4().hex,expected_revision=revision,event=event)
        return self.worker.request('commit',request) if self.worker else self.store.commit(request)

    def delete_event(self, ident, revision):
        if not self.writable:
            raise Conflict('当前为只读')
        request=dict(project_id=self.store.identity,writer_id=self.owner,operation_id=uuid.uuid4().hex,
                     expected_revision=revision,action='delete',event_id=ident)
        return self.worker.request('commit',request) if self.worker else self.store.commit(request)

    def review_images(self, frame_id):
        return self.views([frame_id],True)[0]['histories'] if self.worker else render_review(self.store, frame_id)

    def context_frames(self, frame_id):
        return self.worker.request('context_frames',frame_id) if self.worker else self.store.context_frames(frame_id)

    def views(self, frame_ids, review=False):
        return self.worker.request('views',frame_ids,bool(review)) if self.worker else build_views(self.store,frame_ids,review)

    def renew(self):
        if self.writable:
            self.store.writer(self.owner)

    def close(self):
        try:
            if self.writable:
                self.store.writer(self.owner, release=True)
        finally:
            try:
                if self.worker:self.worker.close()
            finally:
                self.store.close()


class ViewWorker(LazyAssets):
    """Keep CPU-heavy local reads/rendering out of the Qt interpreter, without TCP."""
    def _start(self):
        env=dict(os.environ)
        env['PYTHONPATH']=str(ROOT)
        command=[self.python,'-B','-m','decision.vlm.annotation.backend','--view-worker',self.project]
        self.process=subprocess.Popen(command,
            cwd=str(ROOT),env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,start_new_session=True)

    def request(self, method, *args):
        payload=self.exchange(dict(method=method,args=args,source_binding=self.source_binding))
        if payload[:1]==b'E':
            error=json.loads(payload[1:])
            kind={'Conflict':Conflict,'KeyError':KeyError}.get(error['kind'],ValueError)
            raise kind(error['detail'])
        if payload[:1]==b'V':return unpack_views(payload[1:])
        if payload[:1]==b'B':return payload[1:]
        require(payload[:1]==b'J','本地观察响应类型错误')
        return json.loads(payload[1:])

    def close(self):
        with self.lock:
            process=self.process
            if process is None:return
            try:
                if not process.stdin.closed:process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(3)
            process.stdout.close()
            self.process=None


@lru_cache(maxsize=40)
def render_review(store, frame_id):
    window = store.window(frame_id)
    histories = []
    for index in range(len(window['frames'])):
        image = store.model_image(window, index)
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        histories.append(stream.getvalue())
    return histories


def build_view(store, frame_id, review=False):
    frame = store.frame(frame_id) if review else store.display_frame(frame_id)
    window = store.window(frame_id) if review else store.display_window(frame_id)
    context = [] if review else [store.context_frame(store.display_frame(fid)) if fid else None for fid in window['frames']]
    if review:
        histories = render_review(store,frame_id)
    else:
        # Display-only JPEGs avoid three expensive PNG encodes on every browse step.
        # The model renderer, fingerprints and lossless review/export are unchanged.
        histories = []
        for index,ident in enumerate(window['frames']):
            row = context[index]['frame'] if context[index] else None
            histories.append(context_rgb(store,row,window['mapping'],window['fields']['dt'][index]))
    context = [dict(row, cloud=store.asset(row['frame']['id'], 'cloud3d',display=True)) if row else None for row in context]
    current_cloud = context[-1]['cloud'] if context and context[-1] else None
    return dict(frame=frame, window=window, image=store.asset(frame_id, 'rgb' if review else 'preview',display=not review),
                cloud=current_cloud, cloud3d=current_cloud, histories=histories, context_frames=context,review=bool(review))


def context_rgb(store, frame, mapping, elapsed):
    """Fast browse-only RGB panel; BEV is drawn from scans by Qt, never this image."""
    from PIL import Image,ImageDraw
    cache = getattr(store,'_display_images',None)
    if cache is None:
        cache = store._display_images = OrderedDict()
    ident = frame['id'] if frame else None
    base = cache.get(ident)
    if base is None:
        image = Image.new('RGB',MODEL_SIZE,'#f5f7fa')
        rgb = store.asset(ident,'preview',display=True) if frame and frame.get('has_rgb') else None
        if rgb:
            pixels = Image.open(io.BytesIO(rgb)).convert('RGB')
            _,ox,oy,w,h = rgb_geometry(pixels.size)
            image.paste(pixels.resize((w,h)),(ox,oy))
        else:
            ImageDraw.Draw(image).text((25,30),'MISSING OBSERVATION',fill='#a32323',font=font(24))
        base = image
        cache[ident] = base
        if len(cache)>32:
            cache.popitem(last=False)
    image = base.copy()
    draw = ImageDraw.Draw(image)
    if frame:
        draw_rgb_people(image,frame,mapping)
    draw.text((8,2),f'RGB   t={elapsed:+.2f}s' if elapsed is not None else 'MISSING',fill='#213448',font=font(24))
    stream = io.BytesIO()
    image.save(stream,format='JPEG',quality=90,subsampling=0)
    return stream.getvalue()


def load_view(backend, frame_id, review=False):
    return backend.views([frame_id],review)[0]


def build_views(store, frame_ids, review=False):
    if len(frame_ids)==1:
        return [build_view(store,frame_ids[0],review)]
    # Only CPU display work overlaps; the persistent ROS reader remains serialized.
    with ThreadPoolExecutor(max_workers=2) as workers:
        return list(workers.map(lambda ident:build_view(store,ident,review),frame_ids))


def pack_views(views, compress=True):
    """One bounded response, with shared scans/images stored once across the batch."""
    output = io.BytesIO()
    assets = {}
    def reference(value):
        if value is None:
            return None
        key = hashlib.sha256(value).hexdigest()
        assets.setdefault(key,value)
        return key
    rows = []
    for view in views:
        row = dict(view)
        for name in ('image','cloud','cloud3d'):
            row[name] = reference(row[name])
        row['histories'] = [reference(raw) for raw in row['histories']]
        row['context_frames'] = [dict(part,cloud=reference(part['cloud'])) if part else None for part in row['context_frames']]
        rows.append(row)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED,compresslevel=1) as bundle:
        bundle.writestr('views.json',dumps(rows))
        for key,value in assets.items():
            # JPEG/PNG are already compressed; scans benefit from cheap deflate.
            bundle.writestr(key,value,compress_type=zipfile.ZIP_STORED if not compress or value.startswith((b'\xff\xd8',b'\x89PNG')) else zipfile.ZIP_DEFLATED)
    return output.getvalue()


def unpack_views(payload):
    require(len(payload)<=32*1024*1024,'观察批次过大')
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        require(len(bundle.infolist())<=200 and sum(item.file_size for item in bundle.infolist())<=64*1024*1024,'观察批次解包大小超限')
        rows = json.loads(bundle.read('views.json'))
        require(isinstance(rows,list) and 1<=len(rows)<=12,'观察批次帧数错误')
        assets = {}
        def resolve(key):
            if key is None:
                return None
            if key not in assets:
                assets[key] = bundle.read(key)
                require(hashlib.sha256(assets[key]).hexdigest()==key,'观察资产摘要不匹配')
            return assets[key]
        for row in rows:
            for name in ('image','cloud','cloud3d'):
                row[name] = resolve(row[name])
            row['histories'] = [resolve(key) for key in row['histories']]
            row['context_frames'] = [dict(part,cloud=resolve(part['cloud'])) if part else None for part in row['context_frames']]
    return rows


def view_worker(project):
    store=Store(project)
    source_binding=[store.identity,store.asset_identity,store.rgb_cache.identity if store.rgb_cache else None,store.view_binding]
    try:
        for line in sys.stdin.buffer:
            try:
                request=json.loads(line)
                require(request.get('source_binding')==source_binding,'工程图像来源已改变，请重启标注程序')
                method,args=request['method'],request['args']
                if method=='views':
                    payload=b'V'+pack_views(build_views(store,*args),compress=False)
                elif method=='asset':
                    value=store.asset(*args)
                    payload=b'Jnull' if value is None else b'B'+value
                elif method in ('frame','window','context_frames','commit'):
                    payload=b'J'+dumps(getattr(store,method)(*args)).encode()
                else:raise ValueError('未知本地观察操作')
            except Exception as error:
                payload=b'E'+dumps(dict(kind=type(error).__name__,detail=str(error))).encode()
            sys.stdout.buffer.write(struct.pack('!I',len(payload))+payload)
            sys.stdout.buffer.flush()
    finally:
        store.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--view-worker',required=True)
    args=parser.parse_args()
    view_worker(args.view_worker)
