"""Sequential all-scan detection, independent of playback speed and GUI sampling."""
import time

import numpy as np

from .data import BagReader, Store, index_source, decode_cloud
from .worker import DetectorClient


def predict_project(project, python, gpu, progress=print):
    store, detector, reader = Store(project), None, None
    summary = dict(complete=False, sources=[], processed=0, reused=0, failed_frame=None)
    started = time.monotonic()
    try:
        for source in store.sources():
            index_source(project, source['id'], progress)
        detector = DetectorClient(python, gpu, store.meta('settings'), store.meta('effective'))
        if list(detector.metadata['point_cloud_range']) != list(store.bounds):
            raise ValueError('检测模型与标注范围不一致')
        store.set_meta('offline_detection', summary)
        for source in store.sources():
            reader = BagReader(source)
            rows = list(store.connection.execute('SELECT id,stamp_ns FROM frames WHERE source_id=? ORDER BY ordinal', (source['id'],)))
            previous, intervals = None, []
            source_summary = dict(source_id=source['id'], path=source['path'], input_frames=len(rows),
                                  processed=0, reused=0, first_stamp_ns=None, last_stamp_ns=None)
            summary['sources'].append(source_summary)
            count = 0
            for ordinal, (_, cloud, recorded) in enumerate(reader.bag.read_messages(topics=[source['topic']])):
                if ordinal >= len(rows):
                    raise ValueError('bag 扫描数量已变化')
                frame = store.frame(rows[ordinal]['id'])
                summary['failed_frame'] = frame['id']
                if (frame['stamp_ns'] != cloud.header.stamp.to_nsec() or frame['record_ns'] != recorded.to_nsec()
                        or frame['sensor_frame'] != cloud.header.frame_id):
                    raise ValueError('扫描与已建立索引不一致')
                points = decode_cloud(cloud)
                bound = store.bind_points(frame, points)
                frame['fingerprint'] = bound
                if frame['candidates'] is not None:
                    if frame['model'] != detector.metadata:
                        raise ValueError('已有候选使用不同模型；请使用新工程，不覆盖人工复核依据')
                    key = 'reused'
                else:
                    boxes = detector.predict(frame, points)
                    store.save_candidates(frame['id'], boxes, detector.metadata, bound)
                    key = 'processed'
                summary[key] += 1
                source_summary[key] += 1
                stamp = frame['stamp_ns']
                if previous is not None:
                    intervals.append(stamp - previous)
                previous = stamp
                if source_summary['first_stamp_ns'] is None:
                    source_summary['first_stamp_ns'] = stamp
                source_summary['last_stamp_ns'] = stamp
                count += 1
                if count % 100 == 0:
                    progress('%s：%d / %d 帧（不丢帧）' % (source['path'], count, len(rows)))
                    store.set_meta('offline_detection', summary)
            if count != len(rows):
                raise ValueError('未完成所有输入扫描，不能标记完成')
            differences = np.asarray(intervals, dtype=np.int64)
            span = source_summary['last_stamp_ns'] - source_summary['first_stamp_ns'] if count else 0
            source_summary['observation_hz'] = (count - 1) * 1e9 / span if span > 0 else None
            source_summary['median_interval_ms'] = float(np.median(differences)) / 1e6 if len(differences) else None
            source_summary['nonpositive_intervals'] = int(np.sum(differences <= 0))
            source_summary['missing_predictions'] = store.connection.execute(
                'SELECT count(*) FROM frames WHERE source_id=? AND candidates IS NULL', (source['id'],)).fetchone()[0]
            if source_summary['missing_predictions']:
                raise ValueError('仍有扫描没有检测结果')
            progress('%s：完成 %d 帧，观测频率 %.3f Hz' %
                     (source['path'], count, source_summary['observation_hz'] or 0))
            reader.close()
            reader = None
        summary.update(complete=True, failed_frame=None)
    except BaseException as error:
        summary['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        summary['wall_s'] = time.monotonic() - started
        store.set_meta('offline_detection', summary)
        if reader:
            reader.close()
        if detector:
            detector.close()
        store.close()
    return summary
