"""Local sensor bringup and raw bag recording; no navigation or chassis output."""
import argparse
from collections import deque
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
RECORDER = '/opt/ros/noetic/lib/rosbag/record'
MIN_FREE = 5*1024**3
TOPICS = {
    '/rslidar_points': 'sensor_msgs/PointCloud2',
    '/camera/imu': 'sensor_msgs/Imu',
    '/camera/color/image_raw': 'sensor_msgs/Image',
    '/camera/color/camera_info': 'sensor_msgs/CameraInfo',
    '/tf_static': 'tf2_msgs/TFMessage',
}
STREAMS = tuple(TOPICS)[:-1]
STOP = threading.Event()


def now():
    return datetime.datetime.now().astimezone().isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


@contextmanager
def exclusive(name):
    lock_dir = ROOT/'outputs/local_capture'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/('.'+name+'.lock')).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有同类采集脚本运行，请复用原终端或先正常停止它。')
        yield


def master():
    import rosgraph
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(2.)
    try:
        value = rosgraph.Master('/social_nav_local_capture')
        value.getPid()
        return value
    finally:
        socket.setdefaulttimeout(previous)


def wall_clock(value):
    if value.hasParam('/use_sim_time') and value.getParam('/use_sim_time'):
        raise RuntimeError('当前 ROS master 启用了 /use_sim_time。请先停止 bag 回放，再切换到真实采集时钟。')


def stop_process(process, *, grace=30.):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        print('进程未在期限内正常关闭，尝试终止；请检查是否留下 .bag.active。', flush=True)
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def sensors(args):
    command = [sys.executable, '-B', '-m', 'perception.sensors',
               '--serial-front', args.serial_front, '--fps', str(args.fps)]
    if args.dry_run:
        print(json.dumps(dict(command=command, auto_start_master_if_absent=True,
                              default_front_camera=args.serial_front, opens_chassis=False), ensure_ascii=False, indent=2))
        return 0
    with exclusive('sensors'):
        owned_master = driver = None
        try:
            try:
                value = master()
            except (OSError, RuntimeError):
                from urllib.parse import urlparse
                host = urlparse(os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')).hostname
                if host not in ('localhost', '127.0.0.1', '::1'):
                    raise RuntimeError('远程 ROS master 不可达，不会替你启动或替换远程 master。')
                owned_master = subprocess.Popen(['roscore'], start_new_session=True)
                deadline = time.monotonic()+15.
                while not STOP.is_set():
                    try:
                        value = master()
                        break
                    except (OSError, RuntimeError):
                        if owned_master.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('ROS master 启动失败。')
                        STOP.wait(.2)
                else:
                    return 130
            wall_clock(value)
            publishers = value.getSystemState()[0]
            active = [topic for topic, nodes in publishers if topic in TOPICS and nodes]
            if active:
                raise RuntimeError('已有传感器话题发布者，避免重复启动：'+', '.join(active))
            import rospkg
            packages = rospkg.RosPack()
            selected = Path(packages.get_path('wheelchair_sensors')).resolve()
            if selected != ROOT/'perception/sensors/ros/wheelchair_sensors':
                raise RuntimeError('未选中当前项目的 wheelchair_sensors 配置。')
            packages.get_path('realsense2_camera')
            packages.get_path('rslidar_sdk')
            print('启动雷达和前视相机，图像 {} FPS；Ctrl+C 停止。请先在录包终端结束录制。'.format(args.fps), flush=True)
            driver = subprocess.Popen(command, start_new_session=True)
            while driver.poll() is None and not STOP.wait(.2):
                pass
            return 0 if STOP.is_set() else driver.returncode
        finally:
            stop_process(driver)
            stop_process(owned_master, grace=10.)


def topic_preflight(timeout):
    import rospy
    rospy.init_node('social_nav_capture_check', anonymous=True, disable_signals=True)
    received = {}
    lock = threading.Lock()

    def callback(message, topic):
        # Host receipt freshness only, never a sensor measurement timestamp.
        with lock:
            received[topic] = time.monotonic()
    listeners = [rospy.Subscriber(topic, rospy.AnyMsg, callback, callback_args=topic,
                                 queue_size=1) for topic in TOPICS]
    deadline = time.monotonic()+timeout
    try:
        while not STOP.is_set():
            with lock:
                missing = [topic for topic in TOPICS if topic not in received or
                           topic in STREAMS and time.monotonic()-received[topic] > 2.]
            if not missing:
                return listeners
            if time.monotonic() >= deadline:
                raise RuntimeError('以下话题没有收到数据，请先启动传感器：'+', '.join(missing))
            STOP.wait(.1)
        raise RuntimeError('已取消录制，尚未创建 bag。')
    except BaseException:
        for listener in listeners:
            listener.unregister()
        raise


def available(path):
    values = os.statvfs(path)
    return values.f_bavail*values.f_frsize


def finish_bag(bag, output, report, clock_group):
    import rosbag
    if not bag.is_file() or Path(str(bag)+'.active').exists():
        raise RuntimeError('未得到正常关闭的 bag。保留现场文件，不自动 reindex 或截断。')
    with bag.open('rb') as stream:
        os.fsync(stream.fileno())
    with rosbag.Bag(str(bag), 'r') as data:
        types = data.get_type_and_topic_info().topics
        counts = {key: value.message_count for key, value in types.items()}
        report['bag_info'] = dict(duration_s=data.get_end_time()-data.get_start_time(),
                                  messages=data.get_message_count(), topic_counts=counts)
        missing = [key for key in TOPICS if counts.get(key, 0) == 0]
        if missing:
            raise RuntimeError('已保存 bag，但缺少必要话题：'+', '.join(missing))
    print('文件已关闭，正在计算完整 SHA256 并刷盘，请等待完成。', flush=True)
    digest = hashlib.sha256()
    size = 0
    with bag.open('rb') as stream:
        while True:
            block = stream.read(8*1024*1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    report.update(size_bytes=size, sha256=digest.hexdigest(), bag_closed_and_index_readable=True,
                  validation_scope='indexed_open_required_topic_counts_full_file_sha256_not_full_payload_or_sensor_accuracy')
    with (output/'SHA256SUMS').open('x') as stream:
        stream.write('{}  {}\n'.format(digest.hexdigest(), bag.name))
        stream.flush()
        os.fsync(stream.fileno())
    if clock_group:
        from perception.sensors.clock_session import record_bag_session
        record_bag_session(bag, clock_group)
    report['finished_at'] = now()
    save(output/'capture.json', report)
    for path in output.iterdir():
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record(args):
    # Resolve and announce local output before any writes; never replace a bag.
    requested = Path(args.output_root).expanduser().absolute()
    if requested.resolve() != requested:
        raise ValueError('输出根目录含软链接，请显式传入其真实路径：'+str(requested.resolve()))
    output = requested/(args.kind+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    bag = output/(args.kind+'.bag')
    command = [RECORDER, '--lz4', '--buffsize', '512', '--min-space', '1G', '-O', str(bag)]
    if args.duration is not None:
        command += ['--duration', str(args.duration)]
    command += list(TOPICS)
    if args.dry_run:
        print(json.dumps(dict(command=command, output=str(output), topics=TOPICS,
                              free_space_reserve_bytes=MIN_FREE, opens_chassis=False), ensure_ascii=False, indent=2))
        return 0
    if not Path(RECORDER).is_file():
        raise RuntimeError('找不到 ROS bag 录制程序：'+RECORDER)
    with exclusive('recorder'):
        value = master()
        wall_clock(value)
        published = dict(value.getPublishedTopics('/'))
        wrong = [topic for topic, kind in TOPICS.items() if topic in published and published[topic] != kind]
        if wrong:
            raise RuntimeError('话题类型与采集要求不符：'+', '.join(wrong))
        print('检查五个原始话题是否有数据……', flush=True)
        listeners = topic_preflight(args.topic_timeout)
        try:
            requested.mkdir(parents=True, exist_ok=True)
            if available(requested) < MIN_FREE:
                raise RuntimeError('可用空间不足 5 GiB，未开始录制。')
            output.mkdir(exist_ok=False)
            report = dict(kind=args.kind, bag=str(bag), started_at=now(), note=args.note,
                          topics=TOPICS, clock_group=args.clock_group, command=command,
                          complete=False, status='recording', autonomous_navigation=False)
            save(output/'capture.json', report)
            process = None
            reason = 'recorder_finished'
            try:
                with (output/'record.log').open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    print('正在录制：{}\n按 Ctrl+C 正常结束，随后会自动检查和刷盘。'.format(bag), flush=True)
                    next_status = time.monotonic()+10.
                    while process.poll() is None:
                        if STOP.wait(.25):
                            reason = 'operator_stop'
                            break
                        if available(output) < MIN_FREE:
                            reason = 'low_disk_space'
                            print('剩余空间低于 5 GiB，正在正常停止录制。', flush=True)
                            break
                        if time.monotonic() >= next_status:
                            active = Path(str(bag)+'.active')
                            size = active.stat().st_size if active.exists() else 0
                            print('录制中：{:.2f} GiB；剩余 {:.1f} GiB'.format(size/1024**3, available(output)/1024**3), flush=True)
                            next_status = time.monotonic()+10.
                    stop_process(process)
                report.update(recorder_exit_code=process.returncode, stop_reason=reason)
                warnings = []
                tail = deque(maxlen=12)
                with (output/'record.log').open() as log:
                    for line in log:
                        tail.append(line.rstrip())
                        lower = line.lower()
                        if any(token in lower for token in ('buffer exceeded', 'dropping oldest', 'not enough space')):
                            if len(warnings) < 20:
                                warnings.append(line.rstrip())
                report['recording_warnings'] = warnings
                report['complete'] = process.returncode == 0 and reason != 'low_disk_space' and not warnings
                report['status'] = 'saved' if report['complete'] else 'needs_review'
                finish_bag(bag, output, report, args.clock_group)
                if not report['complete']:
                    print('文件已保留，但录制存在异常或丢帧警告，请检查 record.log。\n'+ '\n'.join(tail), flush=True)
                    return 1
                print('采集完成：{}\nSHA256：{}\n记录：{}'.format(bag, report['sha256'], output/'capture.json'), flush=True)
                return 0
            except BaseException as error:
                report.update(complete=False, status='failed', error=type(error).__name__+': '+str(error))
                save(output/'capture.json', report)
                raise
            finally:
                stop_process(process)
        finally:
            for listener in listeners:
                listener.unregister()
            import rospy
            rospy.signal_shutdown('recording finished')


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('必须为有限正数')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sensor = commands.add_parser('sensors', help='启动本机雷达和前视相机，缺少本机 master 时自动启动')
    sensor.add_argument('--serial-front', default='112322077158')
    sensor.add_argument('--fps', type=int, choices=(15, 30), default=30)
    sensor.add_argument('--dry-run', action='store_true', help='只打印计划，不启动 master 或设备')
    capture = commands.add_parser('record', help='录制五个原始话题，Ctrl+C 后自动检查和生成摘要')
    capture.add_argument('kind', choices=('mapping', 'navigation'))
    capture.add_argument('--output-root', default='/home/tl/rosbag', help='新采集目录的父目录')
    capture.add_argument('--duration', type=positive, help='可选自动停止秒数；默认由 Ctrl+C 停止')
    capture.add_argument('--topic-timeout', type=positive, default=10., help='等待五个话题数据的秒数')
    capture.add_argument('--clock-group', help='可选：由操作者明确声明的连续驱动会话标识')
    capture.add_argument('--note', default='', help='地点、雷达起点位置与朝向等人工记录')
    capture.add_argument('--dry-run', action='store_true', help='只打印计划，不连接 ROS 或创建 bag')
    args = parser.parse_args(argv)
    if args.command == 'record' and args.clock_group is not None:
        from perception.sensors.clock_session import clock_group
        clock_group(args.clock_group)
    STOP.clear()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda *_: STOP.set())
    try:
        return sensors(args) if args.command == 'sensors' else record(args)
    except Exception as error:
        print('采集失败：'+str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
