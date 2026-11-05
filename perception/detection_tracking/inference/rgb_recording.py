"""Read recorded ROS associations without loading a detector or changing their identities."""
import json
from pathlib import Path

from .rgb_cache import digest, file_digest


def observation_record(row,*,pixels,rgb_stamp_ns,camera_info,calibration,detector,detections,result,detector_s,published=True):
    """Transport-independent archive, compatible with existing annotation import."""
    from types import SimpleNamespace
    import hashlib
    from perception.sensors.image_geometry import camera_model
    size=None if pixels is None else [pixels.shape[1],pixels.shape[0]]
    camera=camera_model(SimpleNamespace(**camera_info),size) if camera_info is not None and size is not None else None
    image_input=dict(rgb_stamp_ns=rgb_stamp_ns,image_size=size,
        raw_rgb_sha256=None if pixels is None else hashlib.sha256(pixels.tobytes()).hexdigest(),camera=camera,
        camera_info_stamp_ns=camera_info['stamp_ns'] if camera is not None else None,
        camera_frame=calibration['target_frame'],calibration_sha256=calibration['sha256'])
    return dict(**{k:row[k] for k in ('observation_id','sequence_id','session_id','generation','clock_domain','stamp_s')},
        perception_input_fingerprint=row['input_fingerprint'],image_input=image_input,detector_identity=digest(detector),
        detections=detections,detector_s=detector_s,result=result,reason='',published=bool(published))


IDENTITY = ('session_id','generation','sequence_id','clock_domain','observation_id')


def observation_key(row):
    return tuple(row.get(key) for key in IDENTITY)


def load_records(path,expected_sha256=None):
    path=Path(path)
    if expected_sha256 and file_digest(path)!=expected_sha256:
        raise ValueError('保存的RGB关联记录已改变')
    result={}
    with path.open() as stream:
        for line in stream:
            if not line.strip():continue
            record=json.loads(line)
            key=observation_key(record)
            if any(value is None for value in key) or key in result:
                raise ValueError('RGB关联记录身份缺失或重复')
            result[key]=record
    return result


def recorded_input(row,record):
    if record is None:return None
    if (observation_key(row)!=observation_key(record) or
            row['input_fingerprint']!=record['perception_input_fingerprint'] or
            abs(row['stamp_s']-record['stamp_s'])>1e-6):
        raise ValueError('RGB关联与原三维观察不符')
    return record['image_input']


def validate_binding(binding,frame):
    """Known original 3D identity is mandatory; unavailable 2D input stays explicit."""
    if (binding['id']!=frame['id'] or binding['source_id']!=frame['source_id'] or
            binding['perception_fingerprint']!=frame['perception_fingerprint'] or
            binding['observation_id']!=frame.get('perception_observation_id') or
            abs(binding['stamp_s']-frame['stamp_ns']/1e9)>1e-6):
        raise ValueError('保存的RGB关联与观察身份不符')
    actual=frame['image_association_input']
    if binding['calibration_sha256']!=actual['calibration_sha256']:
        raise ValueError('保存的RGB关联与标定不符')
    recorded=binding.get('image_input')
    if recorded and recorded.get('raw_rgb_sha256'):
        for name in ('raw_rgb_sha256','camera','camera_info_stamp_ns','camera_frame'):
            if actual.get(name)!=recorded.get(name):raise ValueError('保存的RGB关联与原图／内参不符')
        if frame['rgb_stamp_ns']!=recorded['rgb_stamp_ns'] or frame['image_size']!=recorded['image_size']:
            raise ValueError('保存的RGB关联图像配对不符')


def import_recordings(store,output):
    """Bind saved associations to an annotation index; missing work is never rerun."""
    from .rgb_association import people_in_lidar
    from .rgb_cache import CacheWriter
    import sqlite3
    sources=store.sources()
    if not all(source.get('rgb_recording') for source in sources):
        raise ValueError('部分来源没有已记录的二维结果；请拆分来源或显式选择重新推理')
    executions={s['id']:s['rgb_recording']['metadata'] for s in sources}
    detector=dict(backend='recorded_ros',executions=executions)
    source_binding=dict(project_id=store.identity,asset_identity=store.asset_identity,sources=[
        {key:s.get(key) for key in ('id','records_sha256','calibration_sha256','bag_fingerprint','rgb_recording')}
        for s in sources])
    writer=CacheWriter(output,source_binding,detector,dict(kind='recorded_ros',importer=file_digest(__file__)))
    counts=dict(observations=0,recorded=0,not_processed=0,inferred_images=0)
    try:
        for source in sources:
            saved=load_records(source['rgb_recording']['path'],source['rgb_recording']['sha256'])
            records=Path(source['records'])
            if file_digest(records)!=source['records_sha256']:raise ValueError('原三维结果已改变')
            with records.open() as stream:
                raw={row['observation_id']:row for row in (json.loads(line) for line in stream if line.strip())}
            with sqlite3.connect((store.path/'observations.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                frames=db.execute('SELECT id,metadata FROM frames WHERE source_id=? ORDER BY stamp_ns',(source['id'],)).fetchall()
            for ident,metadata in frames:
                index=json.loads(metadata)['result'];row=raw[index['observation_id']]
                record=saved.get(observation_key(row));image=recorded_input(row,record)
                people=people_in_lidar(row)
                keys={p['key'] for p in people}
                if record is not None:
                    if (record['detector_identity']!=digest(executions[source['id']]['detector']) or
                            image['calibration_sha256']!=source['calibration_sha256']):
                        raise ValueError('保存的二维结果与模型／标定来源不符')
                    result=record['result']
                    if set(result['tracks'])!=keys:raise ValueError('保存的RGB关联轨迹身份不符')
                    counts['recorded']+=1
                else:
                    result=dict(status='not_processed',matches=0,tracks={key:dict(status='not_processed',rgb_box=None,
                        compensation='unavailable') for key in keys})
                    counts['not_processed']+=1
                binding=dict(kind='recorded_ros',id=ident,source_id=source['id'],observation_id=row['observation_id'],
                    stamp_s=row['stamp_s'],perception_fingerprint=row['input_fingerprint'],
                    calibration_sha256=source['calibration_sha256'],image_input=image)
                previous=writer.db.execute('SELECT binding,result FROM observations WHERE frame_id=?',(ident,)).fetchone()
                if previous is None:writer.put_bound_observation(ident,binding,result)
                elif json.loads(previous[0])!=binding or json.loads(previous[1])!=result:
                    raise ValueError('已导入的关联来源改变，请使用新目录')
                counts['observations']+=1
        writer.finish(counts)
    finally:writer.close()
    return counts
