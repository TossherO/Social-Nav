#!/usr/bin/env python3
"""Prepare image detections and associations from an existing read-only bag project."""
import argparse
from bisect import bisect_right
from collections import Counter
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from perception.detection_tracking.inference.config import load_config, project_path
from perception.detection_tracking.inference.rgb_cache import CacheWriter, RGBCache, digest, dumps, file_digest
from perception.detection_tracking.inference.rgb_association import ImageAssociator, causal_geometry, people_in_lidar


def matching_spec(config):
    policy = {key:config[key] for key in ('max_time_delta_s','max_uncompensated_delta_s',
              'center_gate','cost_max','ambiguity_margin','overlap_weight','history_weight','history_gap_s')}
    for key,path in (('implementation','perception/detection_tracking/inference/rgb_association.py'),
                     ('projection_implementation','perception/sensors/image_geometry.py'),
                     ('reader_implementation','decision/vlm/annotation/ros_reader.py'),
                     ('preparation_implementation','perception/scripts/prepare_rgb_association.py')):
        policy[key] = file_digest(ROOT/path)
    return policy


def reassociate(args,config):
    """Recompute only association from a frozen compatible detection cache, on CPU."""
    from decision.vlm.annotation.store import Store
    import numpy as np
    project=Path(args.project).resolve()
    store=Store(project,use_project_defaults=False)
    writer=None
    try:
        previous=RGBCache(args.detections_from,store.identity,store.asset_identity)
        manifest=previous.manifest
        detector=manifest['detector']
        for key in ('backend','imgsz','precision','confidence','nms_iou'):
            if detector[key] != config[key]:raise ValueError('旧二维检测与当前检测配置不符：'+key)
        artifact=Path(config['engine'] if config['backend']=='tensorrt' else config['weights'])
        if file_digest(artifact) != detector['sha256']:raise ValueError('旧二维检测与当前模型资产不符')
        allowed={source['id'] for source in manifest['source']['sources']}
        sources=[source for source in store.sources() if source['id'] in allowed and
                 (not args.source or any(token in (source['id'],source['name']) for token in args.source))]
        if not sources:raise ValueError('没有匹配的缓存来源')
        original_sources={s['id']:s for s in manifest['source']['sources']}
        for source in sources:
            if any(source.get(key) != original_sources[source['id']].get(key) for key in
                   ('bag_fingerprint','records_sha256','calibration_sha256')):
                raise ValueError('旧二维缓存与当前bag、感知结果或标定来源不符')
        source_binding=dict(project_id=store.identity,asset_identity=store.asset_identity,
                            sources=[original_sources[s['id']] for s in sources])
        matching=dict(matching_spec(config),detection_source_identity=previous.identity)
        writer=CacheWriter(args.output,source_binding,detector,matching)
        matcher=ImageAssociator(config)
        counts=Counter()
        started=time.perf_counter()
        with sqlite3.connect(previous.database.as_uri()+'?mode=ro',uri=True) as original:
            for source in sources:
                records=Path(source['records'])
                if file_digest(records) != source['records_sha256']:raise ValueError('原感知结果已改变')
                rows=sorted((json.loads(line) for line in records.open() if line.strip()),key=lambda row:row['stamp_s'])
                by_id={row['observation_id']:row for row in rows}
                stamps=[row['stamp_s'] for row in rows]
                with sqlite3.connect((project/'observations.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                    indexed=db.execute('SELECT id,elapsed,metadata FROM frames WHERE source_id=? ORDER BY stamp_ns',
                                       (source['id'],)).fetchall()
                for ident,elapsed,metadata in indexed:
                    if args.segments and not any(max(0.,a-1.2)<=elapsed<=b for a,b in args.segments):continue
                    if args.limit is not None and counts['observations']>=args.limit:break
                    saved=original.execute('SELECT binding FROM observations WHERE frame_id=?',(ident,)).fetchone()
                    if saved is None:raise ValueError('旧二维缓存缺少所需观察：'+ident)
                    frame=json.loads(saved[0])
                    previous.lookup(frame)  # Also refuses a source cache modified during this operation.
                    binding=frame['image_association_input']
                    index=json.loads(metadata)['result']
                    raw=by_id[index['observation_id']]
                    if raw['input_fingerprint'] != frame['perception_fingerprint']:
                        raise ValueError('原三维观察绑定不符')
                    people=people_in_lidar(raw)
                    image_ns=frame['rgb_stamp_ns']
                    detections=[]
                    if image_ns is not None:
                        image_binding=dict(source_id=source['id'],rgb_stamp_ns=image_ns,image_size=frame['image_size'],
                                           raw_rgb_sha256=binding['raw_rgb_sha256'])
                        key,cached=writer.image(image_binding)
                        if cached is None:
                            saved_image=original.execute('SELECT detections,timing_s FROM images WHERE key=?',(key,)).fetchone()
                            if saved_image is None:raise ValueError('旧二维缓存缺少原图检测')
                            detections,cost=json.loads(saved_image[0]),saved_image[1]
                            writer.put_image(key,image_binding,detections,cost)
                        else:detections,_=cached
                        end=bisect_right(stamps,image_ns/1e9)
                        people=causal_geometry(people,raw,rows[max(0,end-3):end],image_ns/1e9,.05)
                    scope=digest([source['id'],index.get('continuity'),binding['calibration_sha256']])
                    result=matcher.step(people,detections,np.asarray(binding['transform']),binding['camera'],
                        (frame['stamp_ns']-image_ns)/1e9 if image_ns is not None else None,
                        frame['stamp_ns']/1e9,scope,raw['health'])
                    if writer.observation(frame) is None:writer.put_observation(frame,result)
                    counts['observations']+=1
                    counts['matches']+=result['matches']
                if args.limit is not None and counts['observations']>=args.limit:break
        if not counts['observations']:raise ValueError('选择范围没有观察')
        summary=dict(counts,wall_s=time.perf_counter()-started,inferred_images=0,
                     reused_detection_source=previous.identity,detection_preparation=manifest['summary'])
        writer.finish(summary)
        return summary
    finally:
        if writer:writer.close()
        store.close()


def export_engine(config,device):
    """An explicit preparation action, never invoked by consumers or detector loading."""
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/social-nav-ultralytics')
    from perception.detection_tracking.inference.image_detector import bind_device
    runtime_device = bind_device(device)
    from ultralytics import YOLO
    import torch
    import tensorrt
    weights,output = Path(config['weights']),Path(config['engine'])
    if output.exists():raise ValueError('TensorRT引擎已存在，不能覆盖冻结模型：'+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='social-nav-yolo-export-') as directory:
        copy = Path(directory)/weights.name
        shutil.copyfile(weights,copy)
        model = YOLO(str(copy),task='detect')
        engine = Path(model.export(format='engine',imgsz=config['imgsz'],batch=1,half=True,dynamic=False,
                                  simplify=False,opset=17,nms=False,workspace=2,device=runtime_device,verbose=False))
        shutil.copyfile(engine,output)
    index = 0
    provenance = dict(weights_sha256=file_digest(weights),engine_sha256=file_digest(output),
        tensorrt=tensorrt.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(index),compute_capability=list(torch.cuda.get_device_capability(index)),
        gpu_uuid=str(torch.cuda.get_device_properties(index).uuid),
        imgsz=config['imgsz'],batch=1,precision='fp16',dynamic=False,nms=False,
        build_s=time.perf_counter()-started)
    output.with_suffix('.engine.json').write_text(dumps(provenance)+'\n',encoding='utf-8')
    return provenance


def prepare(args,config):
    import numpy as np
    from PIL import Image
    from decision.vlm.annotation.assets import LazyAssets
    from decision.vlm.annotation.store import Store
    from perception.detection_tracking.inference.image_detector import ImageDetector
    project = Path(args.project).resolve()
    if not project.is_dir():raise ValueError('观察工程不存在')
    store = Store(project,use_project_defaults=False)
    assets = LazyAssets(project,store.ros_python)
    writer = None
    try:
        sources = [source for source in store.sources() if not args.source or
                   any(token in (source['id'],source['name']) for token in args.source)]
        if not sources:raise ValueError('没有匹配的子bag来源')
        detector = ImageDetector(config,args.device)
        source_binding = dict(project_id=store.identity,asset_identity=store.asset_identity,
            sources=[{key:source.get(key) for key in ('id','bag_fingerprint','records_sha256','calibration_sha256')}
                     for source in sources])
        matching = matching_spec(config)
        writer = CacheWriter(args.output,source_binding,detector.metadata,matching)
        matcher = ImageAssociator(config)
        counts, timings = Counter(),[]
        started = time.perf_counter()
        for source in sources:
            if source.get('storage') != 'rosbag_lazy':
                raise ValueError('RGB准备需要带原bag与感知来源的按需读取工程')
            records = Path(source['records'])
            if source.get('records_sha256') and file_digest(records) != source['records_sha256']:
                raise ValueError('原感知结果已改变，不能复用旧观察索引')
            rows = [json.loads(line) for line in records.open() if line.strip()]
            rows.sort(key=lambda row:row['stamp_s'])
            by_id = {row['observation_id']:row for row in rows}
            stamps = [row['stamp_s'] for row in rows]
            with sqlite3.connect((project/'observations.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                frames = db.execute('SELECT id,stamp_ns,elapsed,metadata FROM frames WHERE source_id=? ORDER BY stamp_ns',
                                    (source['id'],)).fetchall()
            for ident,stamp_ns,elapsed,indexed in frames:
                if args.segments and not any(max(0.,a-1.2) <= elapsed <= b for a,b in args.segments):continue
                if args.limit is not None and counts['observations'] >= args.limit:break
                index = json.loads(indexed)
                raw = by_id.get(index['result']['observation_id'])
                if raw is None or raw['input_fingerprint'] != index['result']['input_fingerprint']:
                    raise ValueError('原感知结果缺少所索引的观察')
                material = assets.read(ident,display=True,association=True,rgb_only=True)
                frame = dict(material['metadata'],id=ident,source_id=source['id'])
                binding = frame['image_association_input']
                scope = digest([source['id'],frame.get('continuity'),binding['calibration_sha256']])
                saved = writer.observation(frame)
                if saved is not None:
                    matcher.previous = {key:((np.array(r['rgb_box'][:2])+r['rgb_box'][2:])/2).tolist()
                                        for key,r in saved['tracks'].items() if r['status']=='matched'}
                    matcher.scope,matcher.stamp = scope,frame['stamp_ns']/1e9
                    counts['reused_observations'] += 1
                    counts['observations'] += 1
                    continue
                detections = []
                if material['rgb'] is not None:
                    image_binding = dict(source_id=source['id'],rgb_stamp_ns=frame['rgb_stamp_ns'],
                        image_size=frame['image_size'],raw_rgb_sha256=binding['raw_rgb_sha256'])
                    key,cached = writer.image(image_binding)
                    if cached is None:
                        rgb = np.asarray(Image.open(io.BytesIO(material['rgb'])).convert('RGB'))
                        detections,cost = detector.detect(rgb)
                        timings.append(cost)
                        writer.put_image(key,image_binding,detections,cost)
                        counts['inferred_images'] += 1
                    else:
                        detections,_ = cached
                        counts['reused_images'] += 1
                delta = (frame['stamp_ns']-frame['rgb_stamp_ns'])/1e9 if frame['rgb_stamp_ns'] is not None else None
                people = frame['people']
                if frame['rgb_stamp_ns'] is not None:
                    image_s = frame['rgb_stamp_ns']/1e9
                    past_end = bisect_right(stamps,image_s)
                    people = causal_geometry(people,raw,rows[max(0,past_end-3):past_end],image_s,.05)
                result = matcher.step(people,detections,np.asarray(binding['transform']),binding['camera'],
                                      delta,frame['stamp_ns']/1e9,scope,frame['health'])
                writer.put_observation(frame,result)
                counts['observations'] += 1
                counts['matches'] += result['matches']
                counts['unmatched_detections'] += result.get('unmatched_detection_count',0)
                for record in result['tracks'].values():
                    counts['track_'+record['status']] += 1
                    counts['geometry_'+record['compensation']] += 1
                if counts['observations']%50 == 0:
                    print(f'已准备{counts["observations"]}观察／匹配{counts["matches"]}目标',flush=True)
            if args.limit is not None and counts['observations'] >= args.limit:break
        if not counts['observations']:raise ValueError('选择范围没有观察，不生成空缓存')
        summary = dict(counts,wall_s=time.perf_counter()-started,
            requested_segments=args.segments,source_names=[source['name'] for source in sources],
            detector_first_s=timings[0] if timings else None,
            detector_warm_p50_s=float(np.median(timings[1:])) if len(timings)>1 else None,
            detector_warm_p95_s=float(np.percentile(timings[1:],95)) if len(timings)>1 else None)
        writer.finish(summary)
        return summary
    finally:
        if writer:writer.close()
        assets.close()
        store.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',help='已有bag/感知结果的只读决策观察工程')
    parser.add_argument('--output',help='新的语义化RGB关联目录；相同来源与配置可断点继续')
    parser.add_argument('--detections-from',help='从同工程的完整缓存复用二维检测，仅CPU重算关联，不读图或加载模型')
    parser.add_argument('--source',nargs='+',help='子bag名称或来源ID；默认全部')
    parser.add_argument('--segments',nargs='+',help='各子bag内的START:END秒；自动补1.2秒历史（含配帧容差）')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--config')
    parser.add_argument('--device',default='0')
    parser.add_argument('--backend',choices=('pytorch','tensorrt'))
    parser.add_argument('--precision',choices=('fp32','fp16'))
    parser.add_argument('--export-engine',action='store_true',help='单独显式准备TensorRT FP16引擎')
    args = parser.parse_args()
    if not args.export_engine and (not args.project or not args.output):parser.error('需要--project和--output')
    try:
        args.segments = [tuple(map(float,item.split(':'))) for item in args.segments or []]
        if any(len(pair)!=2 or not 0 <= pair[0] < pair[1] or not all(__import__('math').isfinite(v) for v in pair)
               for pair in args.segments):raise ValueError()
    except ValueError:parser.error('segments格式为有限的非负START:END')
    if args.limit is not None and args.limit < 1:parser.error('limit须为正整数')
    return args


if __name__ == '__main__':
    args = arguments()
    config = load_config(args.config)['image_association']
    config['weights'],config['engine'] = str(project_path(config['weights'])),str(project_path(config['engine']))
    if args.backend:
        config['backend'] = args.backend
        config['precision'] = 'fp16' if args.backend=='tensorrt' else 'fp32'
    if args.precision:config['precision'] = args.precision
    print(dumps(export_engine(config,args.device) if args.export_engine else
                reassociate(args,config) if args.detections_from else prepare(args,config)),flush=True)
