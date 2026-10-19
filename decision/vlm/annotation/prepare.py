"""Register existing results and bag references; never predecode all observations."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

from .rules import digest, dumps, require
from .store import initialize_labels, initialize_observations


ROOT = Path(__file__).resolve().parents[3]


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def indexed_people(row):
    """Identity-only index. These placeholders are never used as model geometry."""
    localized = row.get('health') == 'valid' and row.get('pose_health') == 'valid' and row.get('world_from_lidar') is not None
    if localized and row.get('tracks'):
        return [dict(key=t['track_id']+'/g'+str(t['lifecycle_generation']), kind='track',
                     rgb_box=None, eligible=False) for t in row['tracks']]
    return [dict(key=row['observation_id']+'/d'+str(i), kind='detection', rgb_box=None, eligible=False)
            for i, _ in enumerate(row.get('detections', []))]


def compact_result(row):
    fields = ('observation_id', 'sequence_id', 'stamp_s', 'clock_domain', 'timestamp_source', 'detection_frame_id',
              'health', 'pose_health', 'world_from_lidar', 'detections', 'input_fingerprint', 'rgb_stamp_s')
    result = {key: row.get(key) for key in fields}
    result['tracks'] = [{key: t[key] for key in ('track_id', 'lifecycle_generation', 'box', 'state', 'confirmed')}
                        for t in row.get('tracks', [])]
    return result


def prepare(args):
    output = Path(args.output).expanduser().resolve()
    require(not output.exists(), '输出目录已存在；请为新观察工程指定新目录')
    results = [Path(p).expanduser().resolve() for p in args.results]
    bags = [Path(p).expanduser().resolve() for p in args.bag] if args.bag else []
    require(not bags or len(bags) == len(results), '--bag数量须与--results相同')
    calibration = Path(args.calibration).resolve()
    require(calibration.is_file(), '相机—雷达标定文件不存在')
    jobs = []
    for i, result in enumerate(results):
        records = result / 'people.jsonl' if result.is_dir() else result
        run = records.parent / 'run.json'
        require(records.is_file(), '缺少people.jsonl：' + str(records))
        metadata = json.loads(run.read_text()) if run.is_file() else {}
        bag = bags[i] if bags else Path(metadata.get('bag', '')).expanduser().resolve()
        require(bag.is_file(), '无法定位原bag；请按--results顺序显式提供--bag')
        require(output != bag and output not in bag.parents and output != records and output not in records.parents,
                '观察输出不得覆盖输入')
        jobs.append(dict(id=uuid.uuid4().hex, name=bag.stem, bag=str(bag), records=str(records),
                         expected_bag=metadata.get('playback', {}).get('bag_fingerprint'),
                         root_group=args.root_group or ('recording:' + bag.parent.name),
                         start=args.start, duration=args.duration))
    recipe = dict(jobs=jobs, pointcloud_topic=args.pointcloud_topic)
    print('读取各bag的首帧头信息与文件索引，不遍历图像或点云…', flush=True)
    with tempfile.TemporaryDirectory(prefix='vlm-annotation-prepare-') as temporary:
        request = Path(temporary) / 'request.json'
        request.write_text(dumps(recipe), encoding='utf-8')
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), '/opt/ros/noetic/lib/python3/dist-packages'])
        env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, ['/opt/ros/noetic/lib', env.get('LD_LIBRARY_PATH', '')]))
        env['PYTHONNOUSERSITE'] = '1'
        completed = subprocess.run([args.ros_python, '-B', '-m', 'decision.vlm.annotation.ros_reader', '--probe', str(request)],
                                   cwd=str(ROOT), env=env, text=True, capture_output=True)
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or '读取bag头信息失败')
        probes = json.loads(completed.stdout)
    import yaml
    calibration_data = yaml.safe_load(calibration.read_text())['lidar_to_camera_optical']
    reader = dict(pointcloud_topic=args.pointcloud_topic, rgb_topic=args.rgb_topic, camera_info_topic=args.camera_info_topic,
                  max_rgb_age_s=.1, max_points=6000, transform=calibration_data['matrix'],
                  lidar_frame=calibration_data['source_frame'], camera_frame=calibration_data['target_frame'])
    calibration_hash = file_digest(calibration)
    db = initialize_observations(output)
    counts, signatures = {}, []
    try:
        db.execute('INSERT INTO meta VALUES(?,?)', ('ros_python', dumps(args.ros_python)))
        for job in jobs:
            probe = probes[job['id']]
            require(probe['sensor_frame'].lstrip('/') == reader['lidar_frame'].lstrip('/'), '标定源坐标系与bag不匹配')
            meta = dict(bag=job['bag'], records=job['records'], root_group=job['root_group'], storage='rosbag_lazy',
                        bag_fingerprint=probe['bag_fingerprint'], origin_ns=probe['origin_ns'],
                        record_offset_ns=probe['record_offset_ns'], frame_id=probe['sensor_frame'],
                        reader=reader, calibration_sha256=calibration_hash)
            db.execute('INSERT INTO sources VALUES(?,?,?)', (job['id'], job['name'], dumps(meta)))
            count, previous, content = 0, None, hashlib.sha256()
            print('导入已有检测／跟踪／位姿：'+job['name'], flush=True)
            with Path(job['records']).open('rb') as stream:
                for line in stream:
                    content.update(line)
                    if not line.strip():
                        continue
                    row = compact_result(json.loads(line))
                    stamp = row['stamp_s']
                    require(isinstance(stamp, (float, int)) and math.isfinite(stamp), '结果观察时间无效')
                    ns = round(stamp*1e9)
                    require(previous is None or ns > previous, '感知结果时间重复或倒退')
                    previous = ns
                    elapsed = (ns-probe['origin_ns'])/1e9
                    if elapsed < args.start-.000001 or (args.duration is not None and elapsed > args.start+args.duration):
                        continue
                    metadata = dict(lazy=True, result=row, people=indexed_people(row), image_size=None,
                                    has_rgb=False, fingerprint=digest(row), sensor_frame=row['detection_frame_id'],
                                    clock_domain=row['clock_domain'], timestamp_source=row['timestamp_source'])
                    db.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?)',
                               (job['id']+'/'+str(count), job['id'], count, ns, elapsed, dumps(metadata), None, None, None))
                    count += 1
            require(count > 0, '指定范围没有已有感知结果：'+job['name'])
            meta['records_sha256'] = content.hexdigest()
            db.execute('UPDATE sources SET metadata=? WHERE id=?', (dumps(meta), job['id']))
            counts[job['name']] = count
            signatures.append((job['id'], meta['records_sha256'], probe['bag_fingerprint']))
            db.commit()
            print('  '+str(count)+'帧索引已就绪；图像／点云按需读取', flush=True)
        identity = digest(dict(sources=signatures, calibration=calibration_hash))
        db.execute('INSERT INTO meta VALUES(?,?)', ('asset_identity', dumps(identity)))
        db.execute('UPDATE meta SET value=? WHERE key=?', ('true', 'complete'))
        db.commit()
    finally:
        db.close()
    initialize_labels(output)
    print('轻量标注工程已就绪：' + str(output))
    print('来源／观察数：' + dumps(counts))
