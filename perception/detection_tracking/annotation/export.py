"""Export only explicitly completed center reviews; never merge original pseudo labels."""
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np

from .data import BagReader, Store, input_fingerprint, safe_directory, validate_review


def center_info(frame, bounds):
    review = frame['review']
    validate_review(review, bounds)
    if review['status'] != 'complete' or not frame['fingerprint']:
        raise ValueError('只允许导出已读取有效扫描的完整复核帧')
    centers = np.asarray([c['xy'] for c in review['centers']], dtype=np.float32).reshape(-1, 2)
    boxes = np.zeros((len(centers), 7), dtype=np.float32)
    boxes[:, :2] = centers
    boxes[:, 3:6] = [.6, .6, 1.7]  # Compatibility placeholders; geometry masks stay zero.
    return dict(token=frame['id'], scene_id=frame['source_id'], frame_index=frame['ordinal'],
                lidar_path='points/' + frame['id'] + '.bin', sweeps=[], has_pose=False,
                label_is_exhaustive=True, center_label_is_exhaustive=True, is_target_domain=True,
                roi_supervision_enabled=False, gt_boxes=boxes,
                gt_names=np.full(len(centers), 'Pedestrian', dtype='<U10'),
                gt_cls_weights=np.ones(len(centers), dtype=np.float32),
                gt_reg_weights=np.ones(len(centers), dtype=np.float32),
                gt_box_reg_mask=np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (len(centers), 1)),
                gt_supervision_types=np.ones(len(centers), dtype=np.int32),  # SUPERVISION_CENTER
                weak_centers=centers)


def check_splits(sources):
    used = {}
    for source in sources:
        split = source['split']
        if split not in ('train', 'val', 'holdout') or not source['root_group'] or not source['location_group']:
            raise ValueError('导出前须明确每个已完成来源的根采集组、地点组与划分')
        for kind in ('root_group', 'location_group'):
            key = kind, source[kind]
            if key in used and used[key] != split:
                raise ValueError('同一采集／重复地点不能跨划分：' + source[kind])
            used[key] = split


def export_project(project, output, progress=None):
    output = safe_directory(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('导出目录已存在，请使用新目录；既有导出不会被覆盖')
    store = Store(project)
    # Short consistent database snapshot; slow bag I/O does not lock the annotator.
    try:
        store.connection.execute('BEGIN')
        frames = [store.frame(row['id']) for row in store.rows()
                  if json.loads(row['review'])['status'] == 'complete']
        sources = {s['id']: s for s in store.sources() if any(f['source_id'] == s['id'] for f in frames)}
        bounds = list(store.bounds)
        settings, effective = store.meta('settings'), store.meta('effective')
    finally:
        store.close()
    if not frames:
        raise ValueError('没有完整复核帧，未生成训练数据')
    check_splits(list(sources.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    readers, infos, bindings = {}, {k: [] for k in ('train', 'val', 'holdout')}, []
    try:
        with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=str(output.parent)) as temporary:
            root = Path(temporary)
            for index, frame in enumerate(frames):
                source = sources[frame['source_id']]
                if source['id'] not in readers:
                    readers[source['id']] = BagReader(source)
                points, _, _ = readers[source['id']].read(frame, rgb=False)
                actual = input_fingerprint(points, frame['source_id'], frame['id'],
                                           frame['stamp_ns'] / 1e9, frame['sensor_frame'])
                if actual != frame['fingerprint']:
                    raise ValueError('导出扫描内容与人工标签绑定不一致：' + frame['id'])
                info = center_info(frame, bounds)
                destination = root / info['lidar_path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.asarray(points, dtype='<f4').tofile(destination)
                infos[source['split']].append(info)
                bindings.append(dict(id=frame['id'], stamp_ns=frame['stamp_ns'], record_ns=frame['record_ns'],
                                     sensor_frame=frame['sensor_frame'], input_fingerprint=actual,
                                     pointcloud=info['lidar_path'], centers=len(info['gt_boxes']), model=frame['model']))
                if progress:
                    progress('导出 %d / %d' % (index + 1, len(frames)))
            for split, values in infos.items():
                if values:
                    with (root / ('rosbag_centers_' + split + '.pkl')).open('wb') as stream:
                        pickle.dump(values, stream, protocol=4)
            manifest = dict(complete=True, annotation_protocol='human_reviewed_centers',
                            point_cloud_range=bounds, max_sweeps=1,
                            geometry_supervision=False, settings=settings, effective_model=effective,
                            sources=list(sources.values()), frames=bindings,
                            counts={k: len(v) for k, v in infos.items()})
            (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            os.rename(temporary, output)
        return manifest['counts']
    finally:
        for reader in readers.values():
            reader.close()
