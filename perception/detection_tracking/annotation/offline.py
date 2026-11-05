"""Sequential all-scan detection, independent of playback speed and GUI sampling."""
import time

import numpy as np

from .data import BagReader, Store, index_source, decode_cloud
from .worker import DetectorClient


def predict_project(project, python, gpu, progress=print, pose_results=None, max_pose_age_s=.02):
    store, detector, reader = Store(project), None, None
    summary = dict(complete=False, sources=[], processed=0, reused=0, failed_frame=None)
    started = time.monotonic()
    try:
        for source in store.sources():
            index_source(project, source['id'], progress)
        max_sweeps = store.meta('effective')['DATA_CONFIG'].get('MAX_SWEEPS', 1)
        providers = {}
        if max_sweeps > 1:
            if pose_results is None:
                raise ValueError('多帧候选预计算需要--pose-results；不能以未补偿历史或原始里程计替代')
            from .poses import RecordedPoses
            providers = {s['id']: RecordedPoses(pose_results, s, max_pose_age_s) for s in store.sources()}
        detector = DetectorClient(python, gpu, store.meta('settings'), store.meta('effective'))
        metadata = dict(detector.metadata)
        if providers:
            metadata['pose_sources'] = {ident: provider.provenance for ident, provider in providers.items()}
            summary['history_frames'] = dict(ready=0, partial=0, unavailable=0)
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
                pose = providers[source['id']].resolve(frame) if providers else None
                if frame['candidates'] is not None:
                    recorded_model = dict(frame['model'])
                    previous_context = recorded_model.pop('input_context', None)
                    if recorded_model != metadata:
                        raise ValueError('已有候选使用不同模型；请使用新工程，不覆盖人工复核依据')
                    if providers and detector.observe(frame, points, pose) != previous_context:
                        raise ValueError('已有候选的历史点云或位姿绑定不一致')
                    key = 'reused'
                else:
                    boxes = detector.predict(frame, points, pose) if providers else detector.predict(frame, points)
                    model = dict(metadata)
                    if providers:
                        model['input_context'] = detector.last_input_context
                    store.save_candidates(frame['id'], boxes, model, bound)
                    key = 'processed'
                if providers:
                    summary['history_frames'][detector.last_input_context['status']] += 1
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
