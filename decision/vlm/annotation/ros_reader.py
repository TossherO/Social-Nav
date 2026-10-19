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

from .rules import digest, dumps, require


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


def corners(box):
    local = np.array([[x, y, z] for z in (-.5, .5) for x, y in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]) * box[3:6]
    c, s = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def sensor_box(box, inverse):
    box = np.asarray(box, dtype=float)
    points = corners(box) @ inverse[:3, :3].T + inverse[:3, 3]
    center = box[:3] @ inverse[:3, :3].T + inverse[:3, 3]
    direction = inverse[:3, :3] @ np.array([np.cos(box[6]), np.sin(box[6]), 0])
    yaw = np.arctan2(direction[1], direction[0])
    c, s = np.cos(yaw), np.sin(yaw)
    axis = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = (points-center) @ axis
    return np.r_[center, np.ptp(local, axis=0), yaw, box[7]].tolist()


def project_box(box, transform, info, size):
    if info is None:
        return None
    import cv2
    optical = corners(np.asarray(box)) @ transform[:3, :3].T + transform[:3, 3]
    if np.any(optical[:, 2] <= .05):
        return None
    if info.distortion_model not in ('plumb_bob', 'rational_polynomial', ''):
        return None
    k = np.asarray(info.K).reshape(3, 3)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or (info.width, info.height) != tuple(size):
        return None
    pixels, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), k, np.asarray(info.D, dtype=float))
    pixels = pixels.reshape(-1, 2)
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    if not np.isfinite(pixels).all() or high[0] < 0 or high[1] < 0 or low[0] >= size[0] or low[1] >= size[1]:
        return None
    return [float(max(0, low[0])), float(max(0, low[1])), float(min(size[0], high[0])), float(min(size[1], high[1]))]


def people_for(row, transform, info, size):
    result = []
    pose = row.get('world_from_lidar')
    localized = row.get('health') == 'valid' and row.get('pose_health') == 'valid' and pose is not None
    if localized:
        matrix = np.asarray(pose, dtype=float)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all() and
                np.allclose(matrix[3], [0, 0, 0, 1]) and
                np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4), '无效定位变换')
        inverse = np.linalg.inv(matrix)
        for track in row.get('tracks', []):
            box = sensor_box(track['box'], inverse)
            rect = project_box(box, transform, info, size) if size else None
            result.append(dict(key=track['track_id'] + '/g' + str(track['lifecycle_generation']),
                               kind='track', box=box, rgb_box=rect, observed=track['state'] == 'observed',
                               confirmed=bool(track['confirmed']), eligible=bool(rect and track['confirmed'] and track['state'] == 'observed')))
    # Unassociated detections remain references; they never gain stable model IDs.
    if not result:
        for i, box in enumerate(row.get('detections', [])):
            result.append(dict(key=row['observation_id'] + '/d' + str(i), kind='detection', box=box,
                               rgb_box=project_box(box, transform, info, size) if size else None,
                               observed=True, confirmed=False, eligible=False))
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

    def read(self, ident):
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
        topics = [config['pointcloud_topic'], config['rgb_topic'], config['camera_info_topic']]
        clouds, images, infos = [], [], []
        for topic, message, _ in bag.read_messages(topics=topics,
                start_time=rospy.Time.from_sec(max(0, center-.5)), end_time=rospy.Time.from_sec(center+.5)):
            ns = message.header.stamp.to_nsec()
            if topic == config['pointcloud_topic'] and abs(ns-expected) <= 1000:
                clouds.append(message)
            elif topic == config['rgb_topic']:
                images.append((ns, message))
            elif topic == config['camera_info_topic']:
                infos.append((ns, message))
        require(len(clouds) == 1, '未唯一匹配到原扫描；没有全包扫描或使用邻帧替代')
        cloud = clouds[0]
        ns = cloud.header.stamp.to_nsec()
        require(cloud.header.frame_id.lstrip('/') == config['lidar_frame'].lstrip('/') == row['detection_frame_id'].lstrip('/'), '雷达坐标系不匹配')
        fields = pointcloud_fields(cloud, ('x', 'y', 'z', 'intensity'))
        points = np.stack([fields[k] for k in ('x', 'y', 'z', 'intensity')], axis=1).astype(np.float32)
        bound = input_fingerprint(points, row['sequence_id'], row['observation_id'], row['stamp_s'], row['detection_frame_id'])
        require(bound == row['input_fingerprint'], '点云与已有检测输入指纹不一致')
        matches = [(t, m) for t, m in images if 0 <= ns-t <= config['max_rgb_age_s']*1e9]
        image_ns, image = max(matches, key=lambda p: p[0]) if matches else (None, None)
        if first_info is not None:
            infos.append((first_info.header.stamp.to_nsec(), first_info))
        infos = [(t, m) for t, m in infos if t <= ns and m.header.frame_id.lstrip('/') == config['camera_frame'].lstrip('/')]
        info = max(infos, key=lambda p:p[0])[1] if infos else None
        pixels = rgb_array(image) if image is not None else None
        size = [image.width, image.height] if image is not None else None
        people = people_for(row, np.asarray(config['transform']), info, size)
        visible = points[np.isfinite(points).all(axis=1)]
        visible = visible[(visible[:, 0] >= -5) & (visible[:, 0] <= 25) & (np.abs(visible[:, 1]) <= 15)]
        if len(visible) > config['max_points']:
            visible = visible[np.linspace(0, len(visible)-1, config['max_points'], dtype=int)]
        buffer = io.BytesIO()
        np.save(buffer, visible.astype('<f4'), allow_pickle=False)
        rgb = encode_image(pixels) if pixels is not None else None
        metadata = dict(stamp_ns=ns, elapsed=(ns-source['origin_ns'])/1e9, people=people, has_rgb=rgb is not None,
                        image_size=size, rgb_stamp_ns=image_ns, timestamp_source=row['timestamp_source'],
                        clock_domain=row['clock_domain'], sensor_frame=row['detection_frame_id'], health=row['health'],
                        perception_fingerprint=row['input_fingerprint'], rgb_sha256=hashlib.sha256(rgb).hexdigest() if rgb else None)
        metadata['fingerprint'] = digest(metadata)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr('frame.json', dumps(metadata))
            bundle.writestr('cloud.npy', buffer.getvalue())
            if rgb is not None:
                bundle.writestr('rgb.png', rgb)
                bundle.writestr('preview.jpg', encode_image(pixels, True))
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
                payload = reader.read(request['frame_id'])
            except Exception as error:
                payload = b'E'+dumps({'detail': str(error)}).encode()
            sys.stdout.buffer.write(struct.pack('!I', len(payload))+payload)
            sys.stdout.buffer.flush()
    finally:
        reader.close()


if __name__ == '__main__':
    main()
