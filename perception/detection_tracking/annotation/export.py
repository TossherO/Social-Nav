"""Export reviewed XY centers, current RGB and deduplicated raw LiDAR history."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .data import BagReader, Store, input_fingerprint, resolve_sequence, safe_directory
from .protocol import digest, validate_payload


DATASET_FORMAT = 'rosbag_people_centers'


def reviewed_annotation(frame, bounds):
    """Unverified proposals never become positive labels or negative background."""
    review = frame['review']
    validate_payload(review, bounds, frame['candidates'])
    if review['status'] not in ('draft', 'complete') or not frame['fingerprint']:
        raise ValueError('只允许导出已绑定扫描的待标／已标帧')
    return dict(centers=[dict(id=c['id'], xy=list(c['xy']), candidate=c['candidate'],
                             origin='manual' if c['candidate'] is None else 'confirmed_detection')
                         for c in review['centers']],
                center_label_is_exhaustive=review['status'] == 'complete',
                ignored_candidate_count=sum(d in ('pending', 'uncertain') for d in review['decisions'].values()),
                rejected_candidate_count=sum(d == 'rejected' for d in review['decisions'].values()),
                note=review['note'])


def center_info(frame, bounds):
    """In-memory detector adapter; JSON datasets keep real XY labels only."""
    annotation = reviewed_annotation(frame, bounds)
    centers = np.asarray([c['xy'] for c in annotation['centers']], dtype=np.float32).reshape(-1, 2)
    boxes = np.zeros((len(centers), 7), dtype=np.float32)
    boxes[:, :2] = centers
    boxes[:, 3:6] = [.6, .6, 1.7]  # Compatibility geometry, never geometric truth.
    return dict(token=frame['id'], scene_id=frame['source_id'], frame_index=frame['ordinal'],
                lidar_path='points/' + frame['id'] + '.bin', sweeps=[], has_pose=False,
                label_is_exhaustive=annotation['center_label_is_exhaustive'],
                center_label_is_exhaustive=annotation['center_label_is_exhaustive'], is_target_domain=True,
                roi_supervision_enabled=False, gt_boxes=boxes,
                gt_names=np.full(len(centers), 'Pedestrian', dtype='<U10'),
                gt_cls_weights=np.ones(len(centers), dtype=np.float32),
                gt_reg_weights=np.ones(len(centers), dtype=np.float32),
                gt_box_reg_mask=np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (len(centers), 1)),
                gt_supervision_types=np.ones(len(centers), dtype=np.int32), weak_centers=centers)


def check_splits(sources):
    """Training split guard; a portable unsplit dataset can still be exported."""
    used = {}
    for source in sources:
        split = source['split']
        if split not in ('train', 'val', 'holdout') or not source['root_group'] or not source['location_group']:
            raise ValueError('训练前须明确每个来源的根采集组、地点组与划分')
        for kind in ('root_group', 'location_group'):
            key = kind, source[kind]
            if key in used and used[key] != split:
                raise ValueError('同一采集／重复地点不能跨划分：' + source[kind])
            used[key] = split


def plan_export(project, *, complete_only=False, history_frames=0, history_stride=1,
                sequences=None, start=0., duration=None, pose_results=None, max_pose_age_s=.02):
    """Take a short read-only label snapshot, then select raw history before I/O."""
    if (isinstance(history_frames, bool) or not isinstance(history_frames, int) or history_frames < 0 or
            isinstance(history_stride, bool) or not isinstance(history_stride, int) or history_stride < 1):
        raise ValueError('历史帧数须为非负整数，历史间隔须为正整数')
    if not math.isfinite(start) or start < 0 or duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('起始偏移须非负有限，时长须为正有限值')
    if not math.isfinite(max_pose_age_s) or max_pose_age_s <= 0:
        raise ValueError('位姿配对时间窗须为正有限秒数')
    store = Store(project, read_only=True)
    try:
        store.connection.execute('BEGIN')
        sources = store.sources()
        selected_sources = {resolve_sequence(sources, name) for name in sequences} if sequences else {s['id'] for s in sources}
        rows = store.rows()
        assets, samples, used_sources = {}, [], []
        for source in sources:
            if source['id'] not in selected_sources:
                continue
            source_rows = [r for r in rows if r['source_id'] == source['id']]
            if not source_rows:
                continue
            origin = source_rows[0]['record_ns']
            used = False
            for index, row in enumerate(source_rows):
                status = json.loads(row['review'])['status']
                elapsed = (row['record_ns'] - origin) / 1e9
                if status not in (('complete',) if complete_only else ('draft', 'complete')):
                    continue
                if elapsed < start or duration is not None and elapsed >= start + duration:
                    continue
                frame = store.frame(row['id'])
                annotation = reviewed_annotation(frame, store.bounds)
                assets.setdefault(frame['id'], frame)
                history = []
                for offset in range(1, history_frames + 1):
                    previous = index - offset * history_stride
                    if previous < 0:
                        break
                    past = store.frame(source_rows[previous]['id'])
                    if (past['stamp_ns'] >= frame['stamp_ns'] or past['record_ns'] > frame['record_ns'] or
                            past['sensor_frame'] != frame['sensor_frame']):
                        raise ValueError('历史点云时间或坐标系不满足因果条件：' + past['id'])
                    assets.setdefault(past['id'], past)
                    history.append(dict(pointcloud_id=past['id'], time_lag_s=(frame['stamp_ns'] - past['stamp_ns']) / 1e9))
                samples.append(dict(id=frame['id'], source_id=source['id'], ordinal=frame['ordinal'],
                                    status=status, pointcloud_id=frame['id'], history=history,
                                    image=None, annotation=annotation, detector_model=frame['model']))
                used = True
            if used:
                used_sources.append(source)
        if not samples:
            raise ValueError('所选范围没有可导出的待标／已标帧')
        source_splits = {s['id']: s['split'] for s in used_sources}
        summary = dict(samples=len(samples), pointclouds=len(assets),
                       history_only_pointclouds=len(assets) - len(samples),
                       centers=sum(len(s['annotation']['centers']) for s in samples),
                       statuses=dict(Counter(s['status'] for s in samples)),
                       splits=dict(Counter(source_splits[s['source_id']] for s in samples)),
                       samples_with_short_history=sum(len(s['history']) < history_frames for s in samples))
        plan = dict(sources=used_sources, assets=assets, samples=samples, counts=summary,
                    point_cloud_range=list(store.bounds),
                    selection=dict(complete_only=bool(complete_only), start_s=start, duration_s=duration),
                    history=dict(frames=history_frames, stride=history_stride, includes_current=False,
                                 motion_compensated=False, missing_history='use_available'))
        if pose_results is not None:
            from .poses import attach_poses
            attach_poses(plan, pose_results, max_pose_age_s)
        return plan
    finally:
        store.close()


def export_project(project, output, progress=None, *, complete_only=False, history_frames=0,
                   history_stride=1, sequences=None, start=0., duration=None, allow_missing_images=False,
                   max_image_age_s=.1, pose_results=None, max_pose_age_s=.02):
    if not math.isfinite(max_image_age_s) or max_image_age_s <= 0:
        raise ValueError('图像配对时间窗须为正有限秒数')
    output = safe_directory(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('导出目录已存在，请使用新目录；既有导出不会被覆盖')
    if Path(project).resolve() == output.resolve() or Path(project).resolve() in output.resolve().parents:
        raise ValueError('导出目录不能位于原标注工程内')
    plan = plan_export(project, complete_only=complete_only, history_frames=history_frames,
                       history_stride=history_stride, sequences=sequences, start=start, duration=duration,
                       pose_results=pose_results, max_pose_age_s=max_pose_age_s)
    samples = {s['id']: s for s in plan['samples']}
    output.parent.mkdir(parents=True, exist_ok=True)
    pointclouds, images = [], 0
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=str(output.parent)) as temporary:
        root = Path(temporary)
        for source in plan['sources']:
            reader = BagReader(source)
            try:
                frames = sorted((f for f in plan['assets'].values() if f['source_id'] == source['id']), key=lambda f: f['ordinal'])
                for frame in frames:
                    sample = samples.get(frame['id'])
                    points, image, image_ns = reader.read(frame, rgb=sample is not None, max_rgb_age_s=max_image_age_s)
                    actual = input_fingerprint(points, frame['source_id'], frame['id'], frame['stamp_ns'] / 1e9, frame['sensor_frame'])
                    if frame['fingerprint'] is not None and actual != frame['fingerprint']:
                        raise ValueError('导出扫描内容与标签绑定不一致：' + frame['id'])
                    pointpath = 'points/' + frame['source_id'] + '/%08d.bin' % frame['ordinal']
                    destination = root / pointpath
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.asarray(points, dtype='<f4').tofile(destination)
                    pointclouds.append(dict(id=frame['id'], source_id=source['id'], ordinal=frame['ordinal'],
                                            stamp_ns=frame['stamp_ns'], record_ns=frame['record_ns'],
                                            sensor_frame=frame['sensor_frame'], path=pointpath,
                                            input_fingerprint=actual, num_points=len(points)))
                    if 'pose' in frame:
                        pointclouds[-1]['pose'] = frame['pose']
                    if sample is not None:
                        if image is None or image_ns is None:
                            if not allow_missing_images:
                                raise ValueError('标注帧缺少%.1fms内的因果RGB；可调整时间窗或显式允许缺图：%s' % (max_image_age_s * 1000, frame['id']))
                        else:
                            imagepath = 'images/' + frame['source_id'] + '/%08d.png' % frame['ordinal']
                            destination = root / imagepath
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(image).save(destination, format='PNG', compress_level=1)
                            sample['image'] = dict(path=imagepath, stamp_ns=int(image_ns),
                                                   age_s=(frame['stamp_ns'] - image_ns) / 1e9, shape=list(image.shape))
                            images += 1
                    if progress:
                        progress('导出点云 %d / %d，图像 %d' % (len(pointclouds), plan['counts']['pointclouds'], images))
            finally:
                reader.close()
        sources = [dict(id=s['id'], bag_name=Path(s['path']).name, fingerprint=json.loads(s['fingerprint']),
                        topic=s['topic'], rgb_topic=s['rgb_topic'], root_group=s['root_group'],
                        location_group=s['location_group'], split=s['split'], epoch=s['epoch'], revision=s['revision'])
                   for s in plan['sources']]
        for source in sources:
            if source['id'] in plan.get('pose_sources', {}):
                source['pose_source'] = plan['pose_sources'][source['id']]
        label_digest = digest([dict(id=s['id'], status=s['status'], annotation=s['annotation']) for s in plan['samples']])
        document = dict(format=DATASET_FORMAT, point_features=['x', 'y', 'z', 'intensity'], point_dtype='float32_le',
                        point_cloud_range=plan['point_cloud_range'], history=plan['history'], selection=plan['selection'],
                        image_pairing=dict(mode='latest_past', max_age_s=max_image_age_s),
                        unverified_policy='ignore', geometry_supervision=False, label_snapshot_sha256=label_digest,
                        sources=sources, pointclouds=pointclouds, samples=plan['samples'],
                        counts=dict(plan['counts'], images=images, missing_images=len(samples) - images))
        (root / 'annotations.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        os.rename(temporary, output)
    return document['counts']
