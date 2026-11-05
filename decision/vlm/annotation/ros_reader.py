"""Read bag headers at registration, then materialize requested frames only."""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import sys
import zipfile

import numpy as np
from PIL import Image

from perception.sensors.image_geometry import corners, sensor_box, project_box, camera_fov, camera_model
from perception.detection_tracking.inference.rgb_association import people_in_lidar
from decision.vlm.scene_inputs import model_eligible
from .rules import digest, dumps, require


# Keep a complete overlapping raw-image read window, not just its last chunk.
MESSAGE_CACHE_BYTES = 256*1024*1024


def edge_fingerprint(path):
    path = Path(path)
    size = path.stat().st_size
    result = hashlib.sha256(str(size).encode() + b'\0')
    with path.open('rb') as stream:
        result.update(stream.read(65536))
        if size > 65536:
            stream.seek(max(65536, size-65536))
            result.update(stream.read())
    return dict(kind='size_and_edge_sha256', size=size, sha256=result.hexdigest())


def encode_image(pixels, preview=False):
    image = Image.fromarray(pixels)
    stream = io.BytesIO()
    if preview:
        image.thumbnail((960, 720))
        image.save(stream, format='JPEG', quality=88)
    else:
        image.save(stream, format='PNG', compress_level=1)
    return stream.getvalue()




def people_for(row, transform, info, size):
    result = people_in_lidar(row)
    for person in result:
        person['rgb_box'] = project_box(person['box'],transform,info,size) if size else None
        person['eligible'] = model_eligible(person) if person['kind']=='track' else False
    return result


def probe(recipe):
    import rosbag
    result = {}
    for job in recipe['jobs']:
        actual = edge_fingerprint(job['bag'])
        if job.get('expected_bag'):
            require(actual == job['expected_bag'], '原bag与感知结果绑定不一致')
        with rosbag.Bag(job['bag'], 'r') as bag:
            first = next(bag.read_messages(topics=[recipe['pointcloud_topic']]), None)
            require(first is not None, 'bag缺少点云话题')
            ns = first[1].header.stamp.to_nsec()
            result[job['id']] = dict(origin_ns=ns, record_offset_ns=first[2].to_nsec()-ns,
                                     sensor_frame=first[1].header.frame_id, bag_fingerprint=actual)
    return result


class FrameReader:
    def __init__(self, project):
        import rosbag
        self.rosbag = rosbag
        self.db = sqlite3.connect((Path(project)/'observations.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.bags = OrderedDict()
        self.messages,self.message_bytes = OrderedDict(),0

    def cached_messages(self, source_id, bag, topics, low, high):
        """Reuse overlapping bag reads without changing record/header-time matching."""
        import rospy
        step = 250000000
        start,end = int(low*1e9),int(high*1e9)
        for part in range(start//step,end//step+1):
            key = (source_id,tuple(topics),part)
            if key in self.messages:
                self.messages.move_to_end(key)
                records,_ = self.messages[key]
            else:
                a,b = part*step,(part+1)*step
                records = [row for row in bag.read_messages(topics=topics,
                    start_time=rospy.Time(a//1000000000,a%1000000000),end_time=rospy.Time(b//1000000000,b%1000000000))
                    if a<=row[2].to_nsec()<b]
                size = sum(len(getattr(row[1],'data',b''))+4096 for row in records)
                self.messages[key] = records,size
                self.message_bytes += size
                while len(self.messages)>16 or self.message_bytes>MESSAGE_CACHE_BYTES:
                    _,(_,old) = self.messages.popitem(last=False)
                    self.message_bytes -= old
            for row in records:
                if start<=row[2].to_nsec()<=end:
                    yield row

    def source(self, ident):
        if ident in self.bags:
            entry = self.bags[ident]
            stat = Path(entry[0]['bag']).stat()
            require((stat.st_size, stat.st_mtime_ns) == entry[2], '原bag已改变，停止使用旧来源')
            self.bags.move_to_end(ident)
            return entry[0], entry[1], entry[3]
        row = self.db.execute('SELECT metadata FROM sources WHERE id=?', (ident,)).fetchone()
        require(row is not None, '来源不存在')
        source = json.loads(row[0])
        require(edge_fingerprint(source['bag']) == source['bag_fingerprint'], '原bag与索引绑定不一致')
        bag = self.rosbag.Bag(source['bag'], 'r')
        first_info = next(bag.read_messages(topics=[source['reader']['camera_info_topic']]), None)
        stat = Path(source['bag']).stat()
        self.bags[ident] = (source, bag, (stat.st_size, stat.st_mtime_ns), first_info[1] if first_info else None)
        if len(self.bags) > 4:
            _, old = self.bags.popitem(last=False)
            old[1].close()
        return source, bag, first_info[1] if first_info else None

    def read(self, ident, display=False, association=False, rgb_only=False):
        import rospy
        from perception.sensors.image import rgb_array
        from perception.sensors.pointcloud import input_fingerprint, pointcloud_fields
        frame = self.db.execute('SELECT * FROM frames WHERE id=?', (ident,)).fetchone()
        require(frame is not None, '观察不存在')
        source, bag, first_info = self.source(frame['source_id'])
        config = source['reader']
        row = json.loads(frame['metadata'])['result']
        expected = frame['stamp_ns']
        center = (expected+source['record_offset_ns'])/1e9
        require(not rgb_only or association, 'RGB准备模式需要关联来源信息')
        topics = [config['rgb_topic'], config['camera_info_topic'], config['pointcloud_topic']]
        clouds, images, infos = [], [], []
        for topic, message, _ in self.cached_messages(frame['source_id'],bag,topics,
                max(0,center-max(.5,config['max_rgb_age_s']+.1)),center+.5):
            ns = message.header.stamp.to_nsec()
            if topic == config['pointcloud_topic'] and abs(ns-expected) <= 1000:
                clouds.append(message)
            elif topic == config['rgb_topic']:
                images.append((ns, message))
            elif topic == config['camera_info_topic']:
                infos.append((ns, message))
        # Keep the exact scan header nanoseconds; JSON float seconds can lose sub-us bits.
        require(len(clouds) == 1, '未唯一匹配到原扫描；没有全包扫描或使用邻帧替代')
        cloud = clouds[0]
        ns = cloud.header.stamp.to_nsec()
        require(cloud.header.frame_id.lstrip('/') == config['lidar_frame'].lstrip('/') == row['detection_frame_id'].lstrip('/'), '雷达坐标系不匹配')
        points = np.empty((0,4),dtype=np.float32)
        if not rgb_only:
            fields = pointcloud_fields(cloud, ('x', 'y', 'z', 'intensity'))
            points = np.stack([fields[k] for k in ('x', 'y', 'z', 'intensity')], axis=1).astype(np.float32)
            bound = input_fingerprint(points, row['sequence_id'], row['observation_id'], row['stamp_s'], row['detection_frame_id'])
            require(bound == row['input_fingerprint'], '点云与已有检测输入指纹不一致')
        recorded=row.get('recorded_rgb_input')
        if config.get('rgb_pairing')=='recorded':
            desired=(recorded['rgb_stamp_ns'] if recorded and recorded.get('raw_rgb_sha256') else
                     round(row['rgb_stamp_s']*1e9) if row.get('rgb_stamp_s') is not None else None)
            matches=[(t,m) for t,m in images if desired is not None and abs(t-desired)<=1000 and t<=ns]
            require(desired is None or len(matches)==1,'原bag缺少感知实际配对的RGB，不能改用邻帧')
        else:
            matches = [(t, m) for t, m in images if 0 <= ns-t <= config['max_rgb_age_s']*1e9]
        image_ns, image = max(matches, key=lambda p: p[0]) if matches else (None, None)
        if image is not None:
            require(image.header.frame_id.lstrip('/') == config['camera_frame'].lstrip('/'), 'RGB光学坐标系与相机标定不匹配')
        if first_info is not None:
            infos.append((first_info.header.stamp.to_nsec(), first_info))
        info_limit = image_ns if association and image_ns is not None else ns
        infos = [(t, m) for t, m in infos if t <= info_limit and m.header.frame_id.lstrip('/') == config['camera_frame'].lstrip('/')]
        info_ns,info = max(infos, key=lambda p:p[0]) if infos else (None,None)
        if recorded and recorded.get('raw_rgb_sha256'):
            from types import SimpleNamespace
            info=SimpleNamespace(**recorded['camera']) if recorded['camera'] is not None else None
            info_ns=recorded['camera_info_stamp_ns']
        pixels = rgb_array(image) if image is not None else None
        size = [image.width, image.height] if image is not None else None
        people = people_for(row, np.asarray(config['transform']), info, size)
        visible = points[np.isfinite(points).all(axis=1)]
        visible = visible[(visible[:, 0] >= -5) & (visible[:, 0] <= 25) & (np.abs(visible[:, 1]) <= 15)]
        if len(visible) > config['max_points']:
            visible = visible[np.linspace(0, len(visible)-1, config['max_points'], dtype=int)]
        buffer = io.BytesIO()
        np.save(buffer, visible.astype('<f4'), allow_pickle=False)
        # This display-only scan never enters model rendering or input fingerprints.
        spatial = points[np.isfinite(points).all(axis=1)]
        spatial = spatial[(np.linalg.norm(spatial[:,:2],axis=1)<=35) & (spatial[:,2]>=-3) & (spatial[:,2]<=5)]
        if len(spatial)>30000:
            spatial = spatial[np.linspace(0,len(spatial)-1,30000,dtype=int)]
        spatial_buffer = io.BytesIO()
        np.save(spatial_buffer, spatial.astype('<f4'), allow_pickle=False)
        rgb = encode_image(pixels) if pixels is not None and (not display or rgb_only) else None
        preview=encode_image(pixels,True) if pixels is not None else None
        metadata = dict(stamp_ns=ns, elapsed=(ns-source['origin_ns'])/1e9, people=people, has_rgb=pixels is not None,
                        image_size=size, rgb_stamp_ns=image_ns, timestamp_source=row['timestamp_source'],
                        clock_domain=row['clock_domain'], sensor_frame=row['detection_frame_id'], health=row['health'],
                        perception_fingerprint=row['input_fingerprint'], rgb_sha256=hashlib.sha256(rgb).hexdigest() if rgb else None)
        # Do not invent continuity evidence absent from the original results.
        if 'continuity' in row:
            metadata['continuity'] = row['continuity']
            metadata['perception_context'] = {key: row.get(key) for key in
                ('session_id', 'generation', 'frame_id', 'pose_source', 'pose_stamp_s', 'localization_binding', 'detection_input')}
        if association:
            metadata['perception_observation_id']=row['observation_id']
            metadata['image_association_input'] = dict(camera=camera_model(info,size),
                camera_info_stamp_ns=info_ns,camera_frame=config['camera_frame'],transform=config['transform'],
                calibration_sha256=source.get('calibration_sha256'),
                raw_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest() if pixels is not None else None)
        metadata['fingerprint'] = digest(metadata) if not display else None
        model_scan = None
        if not display and not rgb_only:
            full_scan = io.BytesIO()
            np.save(full_scan, points.astype('<f4'), allow_pickle=False)
            model_scan = full_scan.getvalue()
            metadata['model_scan_binding'] = dict(kind='full_height_single_scan', stamp_ns=ns,
                sensor_frame=cloud.header.frame_id, perception_input_fingerprint=row['input_fingerprint'],
                sha256=hashlib.sha256(model_scan).hexdigest())
            # This derived asset must not change the RGB association's original frame binding.
        if display:
            metadata['input_verified'] = False
            metadata['camera_fov'] = camera_fov(np.asarray(config['transform']), info, size)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr('frame.json', dumps(metadata))
            bundle.writestr('cloud.npy', buffer.getvalue())
            bundle.writestr('cloud3d.npy', spatial_buffer.getvalue())
            if model_scan is not None:
                bundle.writestr('model_scan.npy', model_scan)
            if rgb is not None:
                bundle.writestr('rgb.png', rgb)
            if preview is not None:
                bundle.writestr('preview.jpg',preview)
        return output.getvalue()

    def close(self):
        for _, bag, _, _ in self.bags.values():
            bag.close()
        self.db.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--probe')
    modes.add_argument('--worker')
    args = parser.parse_args()
    if args.probe:
        print(dumps(probe(json.loads(Path(args.probe).read_text()))))
        return
    reader = FrameReader(args.worker)
    try:
        for line in sys.stdin.buffer:
            try:
                request = json.loads(line)
                payload = reader.read(request['frame_id'],display=request.get('display',False),
                                      association=request.get('association',False),rgb_only=request.get('rgb_only',False))
            except Exception as error:
                payload = b'E'+dumps({'detail': str(error)}).encode()
            sys.stdout.buffer.write(struct.pack('!I', len(payload))+payload)
            sys.stdout.buffer.flush()
    finally:
        reader.close()


if __name__ == '__main__':
    main()
