"""Blocking annotation operations. The Qt window calls these in workers only."""
from functools import lru_cache
import io
import uuid

from decision.vlm.scene_inputs import render_observation
from .store import Conflict, Store


class LocalBackend:
    def __init__(self, project, readonly=False):
        self.store = Store(project)
        self.owner = uuid.uuid4().hex
        self.writable = not readonly
        self.notice = ''
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
        return self.store.window(frame)

    def frame(self, frame):
        return self.store.frame(frame)

    @lru_cache(maxsize=100)
    def asset(self, frame, kind):
        return self.store.asset(frame, kind)

    def commit(self, event, revision):
        if not self.writable:
            raise Conflict('当前为只读')
        return self.store.commit(dict(project_id=self.store.identity, writer_id=self.owner,
                                      operation_id=uuid.uuid4().hex, expected_revision=revision, event=event))

    def review_images(self, frame_id):
        return render_review(self.store, frame_id)

    def renew(self):
        if self.writable:
            self.store.writer(self.owner)

    def close(self):
        try:
            if self.writable:
                self.store.writer(self.owner, release=True)
        finally:
            self.store.close()


@lru_cache(maxsize=40)
def render_review(store, frame_id):
    frame, window = store.frame(frame_id), store.window(frame_id)
    histories = []
    for ident in window['frames']:
        row = store.frame(ident) if ident else None
        rgb = store.asset(ident, 'rgb') if ident else None
        image = render_observation(row, rgb, window['mapping'], (row['stamp_ns']-frame['stamp_ns'])/1e9 if row else 0,
                                   window['fields']['e'])
        stream = io.BytesIO()
        image.save(stream, format='PNG')
        histories.append(stream.getvalue())
    return histories


def load_view(backend, frame_id, review=False):
    frame = backend.frame(frame_id)
    window = backend.window(frame_id)
    if review:
        histories = backend.review_images(frame_id)
    else:
        histories = [backend.asset(ident, 'preview') if ident else None for ident in window['frames']]
    return dict(frame=frame, window=window, image=backend.asset(frame_id, 'rgb' if review else 'preview'),
                cloud=backend.asset(frame_id, 'cloud'), histories=histories)
