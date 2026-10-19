"""GUI data boundary. Call these blocking methods from workers, never Qt slots.

LocalBackend opens SQLite connections in the calling thread and caches raw bag
assets only. The HTTP backend implements the same operations without
putting ROS, server paths or database handles in the window. HTTP transport,
durable client queues and write sessions live in remote.py/cache.py.
"""
from contextlib import closing
import copy
from pathlib import Path

from .data import FrameLoader, Store, import_detections, index_source


VIEW_METADATA = ('sample_intervals', 'point_scale', 'last_frame', 'last_frames')


class LocalBackend:
    def __init__(self, project):
        self.project = Path(project)
        self.loader = FrameLoader(self.project)

    def prepare(self, progress, detections=None):
        with closing(Store(self.project)) as store:
            sources = store.sources()
        for source in sources:
            progress('索引子包：' + Path(source['path']).name)
            index_source(self.project, source['id'], progress)
        if detections:
            import_detections(self.project, detections, progress)
        return self.snapshot()

    def snapshot(self):
        with closing(Store(self.project)) as store:
            # One read transaction, including source/row lists and preferences.
            store.connection.execute('BEGIN')
            return dict(bounds=store.bounds, sources=store.sources(), rows=store.rows(),
                        metadata={key: store.meta(key) for key in VIEW_METADATA})

    def read_frame(self, ident):
        return self.loader.read(ident)

    def set_meta(self, key, value):
        if key not in VIEW_METADATA:
            raise ValueError('不是界面设置：' + key)
        with closing(Store(self.project)) as store:
            store.set_meta(key, value)

    def save_position(self, frame):
        with closing(Store(self.project)) as store:
            store.save_position(frame)

    def save_review(self, ident, review, expected):
        with closing(Store(self.project)) as store:
            store.save_review(ident, review, expected=expected)
            frame = store.frame(ident)
            # A second writer may commit between our commit and this read. The
            # acknowledgement is for OUR submitted state; the next CAS must still
            # reject that intervening edit rather than treating it as our baseline.
            frame['review'] = copy.deepcopy(review)
            return frame

    def save_group(self, ident, root_group, location_group, split):
        with closing(Store(self.project)) as store:
            store.set_group(ident, root_group, location_group, split)
            return store.source(ident)

    def reset_source_reviews(self, ident, current=None):
        with closing(Store(self.project)) as store:
            result = store.reset_source_reviews(ident)
            result['frame'] = store.frame(current) if current is not None else None
            return result

    def close(self):
        self.loader.close()
