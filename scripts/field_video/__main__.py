"""Compose a field recording and its Agent timeline into an H.264 MP4."""

import argparse
from itertools import chain
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from perception.sensors.image import rgb_array
from perception.sensors.pointcloud import pointcloud_fields


RGB_TOPIC = '/camera/color/image_raw'
SCAN_TOPIC = '/rslidar_points'
EVENT_CONTEXT = 2.5
MAX_POSE_GAP = 1.
STATIONARY_DISTANCE = .08
STATIONARY_ANGLE = math.radians(5)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(intervals, removed):
    result = []
    for start, end in intervals:
        for left, right in removed:
            if right <= start:
                continue
            if left >= end:
                break
            if start < left:
                result.append((start, left))
            start = max(start, right)
        if start < end:
            result.append((start, end))
    return result


def stationary_intervals(poses, minimum, context):
    """Keep context at each end of a continuously observed stationary interval."""
    if not poses:
        return []
    anchor = previous = poses[0]
    cuts = []

    def finish_interval():
        if previous['t'] - anchor['t'] >= minimum and previous['t'] - anchor['t'] > 2 * context:
            cuts.append((anchor['t'] + context, previous['t'] - context))

    for pose in poses[1:]:
        yaw = pose['yaw'] - anchor['yaw']
        if (pose['floor'] != anchor['floor'] or pose['segment'] != anchor['segment']
                or not 0 < pose['t'] - previous['t'] <= MAX_POSE_GAP
                or math.hypot(pose['x'] - anchor['x'], pose['y'] - anchor['y']) > STATIONARY_DISTANCE
                or abs(math.atan2(math.sin(yaw), math.cos(yaw))) > STATIONARY_ANGLE):
            finish_interval()
            anchor = pose
        previous = pose
    finish_interval()
    return cuts


def video_intervals(data, first, last, args):
    events = [event['t'] for event in data['events']]
    evidence = [item['t'] for item in data['evidence']]
    recorded = [(first, max(last, first + 1 / args.fps))]
    outside = [(t, t + EVENT_CONTEXT) for t in events if t < first or t > last - EVENT_CONTEXT]
    footage = merge_intervals(recorded + outside)
    if args.mode == 'full':
        return footage
    protected = [(t - EVENT_CONTEXT, t + EVENT_CONTEXT) for t in events + evidence]
    protected += [(first, first + .5), (last - .5, last)]
    protected += [(first + start, first + end) for start, end in args.keep]
    cuts = stationary_intervals(data['poses'], args.wait_min, args.wait_context)
    cuts += [(first + start, first + end) for start, end in args.cut]
    cuts = subtract_intervals(merge_intervals(cuts), merge_intervals(protected))
    return merge_intervals(subtract_intervals(recorded, cuts) + outside)


def measurement_rows(bag, topic):
    previous = -math.inf
    for _, msg, _ in bag.read_messages(topics=[topic]):
        stamp = msg.header.stamp.to_sec()
        if stamp > 0 and stamp > previous:
            previous = stamp
            yield stamp, msg


def last_measurement_row(bag, topic):
    from genpy import Time

    end, start = bag.get_end_time(), bag.get_start_time()
    window = 2.
    while True:
        lower = max(start, end - window)
        kwargs = {} if lower == start else {'start_time': Time.from_sec(lower)}
        rows = ((msg.header.stamp.to_sec(), msg) for _, msg, _ in bag.read_messages(topics=[topic], **kwargs))
        last = max(rows, key=lambda row: row[0], default=None)
        if last is not None and last[0] > 0:
            return last
        if lower == start:
            raise ValueError(f'{topic} 没有有效测量时间')
        window *= 2


def decode_rgb(msg):
    return Image.fromarray(rgb_array(msg))


def decode_scan(msg):
    frame = msg.header.frame_id.lstrip('/')
    if not frame:
        return None, None
    fields = pointcloud_fields(msg, ['x', 'y', 'z'])
    points = np.column_stack((fields['x'], fields['y'], fields['z']))
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.any(points != 0, axis=1) & (np.abs(points[:, :2]) <= 30.).all(axis=1)]
    return points if len(points) else None, frame


class CausalSample:
    def __init__(self, rows, decode, max_age):
        self.rows = iter(rows)
        self.decode = decode
        self.max_age = max_age
        self.pending = next(self.rows, None)
        self.message = None
        self.value = None

    def at(self, stamp):
        latest = self.message
        while self.pending is not None and self.pending[0] <= stamp:
            latest = self.pending
            self.pending = next(self.rows, None)
        if latest is None or stamp - latest[0] > self.max_age:
            return None
        if latest is not self.message:
            self.message = latest
            self.value = self.decode(latest[1])
        return self.value


def recorded_frames(bag, has_scan, data, args, edits):
    from .canvas import Canvas

    rgb_rows = measurement_rows(bag, RGB_TOPIC)
    first_row = next(rgb_rows, None)
    if first_row is None:
        raise ValueError('bag 没有有效的前视彩色测量时间')
    first = first_row[0]
    last, last_message = last_measurement_row(bag, RGB_TOPIC)
    rgb = CausalSample(chain((first_row,), rgb_rows), decode_rgb, .4)
    scan = CausalSample(measurement_rows(bag, SCAN_TOPIC), decode_scan, .25) if has_scan else None
    canvas = Canvas(data)
    intervals = video_intervals(data, first, last, args)
    freeze_tail = any(event['t'] > last for event in data['events'])
    last_image = None
    previous = first
    for start, end in intervals:
        skipped = start - previous if first <= previous < start <= last else 0.
        if skipped > 0:
            edits.append((previous - first, start - first))
        for index in range(max(1, math.ceil((end - start) * args.fps))):
            stamp = start + index / args.fps
            frozen_at = None
            image = rgb.at(stamp) if first <= stamp <= last else None
            if image is not None:
                stamp = rgb.message[0]
            elif stamp > last and freeze_tail:
                if last_image is None:
                    last_image = decode_rgb(last_message)
                image = last_image
                frozen_at = last
            observation_time = stamp if frozen_at is None else frozen_at
            reading = scan.at(observation_time) if scan and first <= observation_time <= last else None
            scan_xyz, scan_frame = reading if reading is not None else (None, None)
            yield canvas.frame(stamp, image, elapsed=stamp - first,
                               skipped=skipped if skipped > .05 and index < 2 * args.fps else None,
                               scan_xyz=scan_xyz, scan_frame=scan_frame,
                               scan_stamp=scan.message[0] if reading is not None else None,
                               frozen_at=frozen_at, bookend=stamp >= last - 2)
        previous = end


def write_video(path, fps, frames):
    import cv2
    from .canvas import SIZE

    if os.path.lexists(path):
        raise FileExistsError(f'输出文件已存在：{path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.field-video-', dir=str(path.parent)) as staging:
        temporary = Path(staging) / 'video.mp4'
        writer = cv2.VideoWriter(str(temporary), cv2.CAP_FFMPEG,
                                 cv2.VideoWriter_fourcc(*'avc1'), fps, SIZE)
        count = 0
        try:
            if not writer.isOpened():
                raise RuntimeError('OpenCV 的 FFmpeg 后端无法初始化 H.264 编码器')
            for frame in frames:
                if frame.size != SIZE or frame.mode != 'RGB':
                    raise ValueError(f'视频合成帧必须为 {SIZE[0]}×{SIZE[1]} RGB')
                writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
                count += 1
        finally:
            writer.release()
        reader = cv2.VideoCapture(str(temporary), cv2.CAP_FFMPEG)
        try:
            ok, first_frame = reader.read()
            if not ok or first_frame.shape[:2] != SIZE[::-1] or int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                raise RuntimeError('H.264 编码结果无法通过帧数或画面尺寸检查')
        finally:
            reader.release()
        os.link(temporary, path)
    return count


def second_range(value):
    try:
        start, end = map(float, value.split(':'))
        if math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            return start, end
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('秒段须用 START:END，并从首张 RGB 原始时间计时，例如 12:20')


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线合成实机演示视频')
    parser.add_argument('demo_dir', type=Path, help='实机演示的解包目录')
    parser.add_argument('--output', type=Path, required=True, help='新建输出 .mp4 路径')
    parser.add_argument('--planning-python', help='离线复算参考路径的项目 Python，默认读取 SOCIAL_NAV_PYTHON')
    parser.add_argument('--mode', choices=('compact', 'full'), default='compact', help='剪辑等待或保留原始录像时长')
    parser.add_argument('--fps', type=int, default=12, help='输出帧率，默认 12')
    parser.add_argument('--wait-min', type=float, default=8., help='可压缩等待的最短静止秒数，默认 8')
    parser.add_argument('--wait-context', type=float, default=2., help='静止段两侧原速保留的秒数，默认 2')
    parser.add_argument('--cut', type=second_range, action='append', default=[], help='添加可剪辑的原录像秒段，可重复')
    parser.add_argument('--keep', type=second_range, action='append', default=[], help='保护原录像秒段，可重复')
    args = parser.parse_args(argv)
    if (args.output.suffix.lower() != '.mp4' or not 1 <= args.fps <= 60 or not math.isfinite(args.wait_min)
            or not math.isfinite(args.wait_context) or args.wait_min <= 0 or args.wait_context < 0):
        parser.error('输出须为 .mp4，帧率须为 1–60，等待参数须为有限秒数，且最短静止为正、两侧时间非负')
    if args.mode == 'full' and (args.cut or args.keep):
        parser.error('指定剪辑秒段时请使用 compact 模式')
    try:
        import rosbag
    except ImportError as exc:
        parser.error(f'请先加载 /opt/ros/noetic/setup.bash，再用 /usr/bin/python3 运行（{exc}）')

    from .timeline import load_demo

    try:
        if os.path.lexists(args.output):
            raise FileExistsError(f'输出文件已存在：{args.output}')
        data = load_demo(args.demo_dir, planning_python=args.planning_python)
        failures = data.get('reference_failures', [])
        if failures:
            reasons = '；'.join(dict.fromkeys(item['reason'] for item in failures))
            print(f'部分时段无法绘制参考线：{reasons}', file=sys.stderr)
        bag_path = data.get('bag')
        if not bag_path:
            bags = sorted((args.demo_dir / 'capture').glob('*.bag'))
            if len(bags) != 1:
                raise ValueError('演示目录的 capture/ 下须有一份已完成的 .bag')
            bag_path = bags[0]
        with rosbag.Bag(str(bag_path), 'r') as bag:
            types = bag.get_type_and_topic_info().topics
            if RGB_TOPIC not in types or types[RGB_TOPIC].msg_type != 'sensor_msgs/Image':
                raise ValueError(f'bag 缺少前视彩色图像话题 {RGB_TOPIC}')
            has_scan = SCAN_TOPIC in types and types[SCAN_TOPIC].msg_type == 'sensor_msgs/PointCloud2'
            edits = []
            count = write_video(args.output, args.fps, recorded_frames(bag, has_scan, data, args, edits))
    except (OSError, ValueError, RuntimeError, rosbag.ROSBagException) as exc:
        parser.exit(1, f'导出失败：{exc}\n')
    print(f'{args.output.resolve()}（{count} 帧，{count / args.fps:.1f} 秒）')
    if edits:
        ranges = '、'.join(f'{start:.2f}–{end:.2f} 秒' for start, end in edits)
        print(f'共跳过 {sum(end - start for start, end in edits):.2f} 秒。原录像秒段为 {ranges}。')
    else:
        print('原录像中无跳过秒段。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
