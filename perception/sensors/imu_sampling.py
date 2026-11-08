"""Explicit local IIO sampling reload during sensor startup, before consumers run."""
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time


class StartupCancelled(RuntimeError):
    pass


def select_policy(sensors, front, *, lidar_only=False):
    policy = sensors.get('imu_sampling_reload')
    if lidar_only or policy is None:
        return None
    if not isinstance(policy, dict) or type(policy.get('enabled', True)) is not bool:
        raise ValueError('IMU sampling reload must be a mapping with a boolean enabled field')
    if policy.get('enabled') is False:
        return None
    if policy.get('sdk_serial') != front:
        return None
    names = ('usb_serial', 'firmware', 'topic', 'normal_value', 'temporary_value', 'source')
    if any(not isinstance(policy.get(k), str) or not policy[k].strip() for k in names):
        raise ValueError('IMU reload requires explicit device identity, settings and verification source')
    if not front.isdigit() or not policy['usb_serial'].isdigit():
        raise ValueError('IMU reload device serials must be explicit numeric strings')
    if not policy['topic'].startswith('/') or not policy['topic'].endswith('/imu'):
        raise ValueError('IMU reload requires an absolute unified IMU topic')
    rate = policy.get('expected_hz')
    if type(rate) not in (int, float) or not 1 <= rate <= 1000:
        raise ValueError('Invalid expected IMU frequency')
    try:
        a, b = (Decimal(policy[k]) for k in ('normal_value', 'temporary_value'))
        if not a.is_finite() or not b.is_finite() or not 0 < a <= 10000 or not 0 < b <= 10000 or a == b:
            raise ValueError('Invalid IMU reload sampling values')
    except InvalidOperation as error:
        raise ValueError('Invalid IMU reload sampling values') from error
    return dict(policy)


def find_gyro(usb_serial, root=Path('/sys/bus/iio/devices')):
    matches = []
    for device in root.glob('iio:device*'):
        if (device/'name').read_text().strip() != 'gyro_3d':
            continue
        for parent in device.resolve().parents:
            if (parent/'idVendor').is_file() and (parent/'serial').is_file():
                if ((parent/'serial').read_text().strip() == usb_serial and
                        (parent/'idVendor').read_text().strip().lower() == '8086' and
                        (parent/'idProduct').read_text().strip().lower() == '0b3a'):
                    matches.append(device/'in_anglvel_sampling_frequency')
                break
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one matching front-camera IIO gyro; found '+str(len(matches)))
    target = matches[0]
    if not target.is_file() or (target.parent/'buffer/enable').read_text().strip() != '1':
        raise RuntimeError('Matched IIO gyro is not streaming')
    return target


def reload_attribute(target, policy, pause):
    """Restore the original attribute even if cancelled during the temporary phase."""
    original = target.read_text().strip()
    if Decimal(original) != Decimal(policy['normal_value']):
        raise RuntimeError('IIO setting differs from the verified local normal value: '+original)
    print('IMU采样重载：临时切换设置，随后恢复原值。', flush=True)
    try:
        target.write_text(policy['temporary_value']+'\n')
        pause(3.)
    finally:
        target.write_text(original+'\n')
    pause(3.)
    if Decimal(target.read_text().strip()) != Decimal(original):
        raise RuntimeError('IIO sampling setting was not restored')
    return original


def sample_stats(samples):
    if len(samples) < 2:
        raise RuntimeError('Insufficient IMU samples')
    elapsed = (samples[-1][0]-samples[0][0])/1e9
    if elapsed <= 0:
        raise RuntimeError('Invalid IMU receipt interval')
    stamps = [row[1] for row in samples]
    return dict(count=len(samples), span_s=elapsed, rate_hz=(len(samples)-1)/elapsed,
                non_increasing=sum(b <= a for a, b in zip(stamps, stamps[1:])),
                publishers=sorted({row[2] for row in samples}))


def verify_samples(samples, expected_hz):
    stats = sample_stats(samples)
    if stats['span_s'] < 1.5 or not .9*expected_hz <= stats['rate_hz'] <= 1.1*expected_hz:
        raise RuntimeError('IMU frequency check failed: '+json.dumps(stats))
    if stats['non_increasing'] or stats['publishers'] == [''] or len(stats['publishers']) != 1:
        raise RuntimeError('IMU timestamp/publisher check failed: '+json.dumps(stats))
    return stats


@contextmanager
def device_lock(root, usb_serial):
    directory = root/'outputs/runtime/locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/('imu_'+usb_serial+'.lock')).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another sensor startup owns this camera IMU') from error
        yield


def check_no_publisher(policy):
    import rosgraph
    try:
        publishers = dict(rosgraph.Master('/social_nav_imu_startup').getSystemState()[0])
    except (OSError, rosgraph.MasterException):
        return  # roslaunch may create its own master; existing services are not stopped.
    if publishers.get(policy['topic']):
        raise RuntimeError('IMU publisher already exists; refusing a second startup/reload')


def prepare(driver, policy, stopped):
    import rospy
    import rosgraph
    from sensor_msgs.msg import Imu
    from realsense2_camera.srv import DeviceInfo

    def alive():
        if stopped.is_set():
            raise StartupCancelled('传感器启动已取消')
        if driver.poll() is not None:
            raise RuntimeError('Sensor driver exited during IMU preparation')

    master = rosgraph.Master('/social_nav_imu_startup')
    deadline = time.monotonic()+20.
    while True:
        alive()
        try:
            master.getPid()
            break
        except (OSError, rosgraph.MasterException):
            if time.monotonic() >= deadline:
                raise RuntimeError('ROS master did not become available')
            stopped.wait(.1)
    if master.hasParam('/use_sim_time') and master.getParam('/use_sim_time'):
        raise RuntimeError('IMU reload requires live sensor time')
    rospy.init_node('social_nav_imu_startup', anonymous=True, disable_signals=True)
    samples = []
    lock = threading.Lock()

    def callback(message):
        with lock:
            samples.append((time.monotonic_ns(), message.header.stamp.to_nsec(),
                            message._connection_header.get('callerid', '')))

    def pause(seconds):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            alive()
            stopped.wait(min(.05, max(0., until-time.monotonic())))

    subscriber = rospy.Subscriber(policy['topic'], Imu, callback, queue_size=8192, tcp_nodelay=True)
    try:
        deadline = time.monotonic()+20.
        while True:
            alive()
            with lock:
                count = len(samples)
            if count >= 20:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('No live IMU input for startup reload')
            stopped.wait(.05)
        namespace = policy['topic'].rsplit('/', 1)[0]+'/'
        services = [name for name, _ in master.getSystemState()[2]
                    if name.startswith(namespace) and name.endswith('/device_info')]
        if len(services) != 1:
            raise RuntimeError('Cannot uniquely identify the front-camera device-info service')
        identity = rospy.ServiceProxy(services[0], DeviceInfo)()
        if identity.serial_number != policy['sdk_serial'] or identity.firmware_version != policy['firmware']:
            raise RuntimeError('Camera identity/firmware differs from the verified local reload policy')
        publishers = dict(master.getSystemState()[0]).get(policy['topic'], [])
        if len(publishers) != 1:
            raise RuntimeError('Expected a single IMU publisher before reload')
        pause(1.)  # Let the normal sensor configuration finish before touching its IIO attribute.
        with lock:
            before = sample_stats(list(samples))
        target = find_gyro(policy['usb_serial'])
        original = reload_attribute(target, policy, pause)
        with lock:
            samples.clear()  # A new, explicitly post-reload verification window; source messages are untouched.
        pause(2.)
        with lock:
            current = list(samples)
        if not current or time.monotonic_ns()-current[-1][0] > 500000000:
            raise RuntimeError('IMU input stopped after sampling reload')
        after = verify_samples(current, policy['expected_hz'])
        if after['publishers'] != publishers:
            raise RuntimeError('IMU publisher changed during startup')
        return dict(status='ready', sdk_serial=identity.serial_number, usb_serial=policy['usb_serial'],
                    firmware=identity.firmware_version, original_setting=original, before=before, after=after,
                    ready_mono_ns=time.monotonic_ns(), source=policy['source'], motor_commands_sent=False)
    finally:
        subscriber.unregister()


def stop_driver(driver):
    if driver.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 15.), (signal.SIGTERM, 3.), (signal.SIGKILL, 3.)):
        try:
            os.killpg(driver.pid, sig)
        except ProcessLookupError:
            return
        try:
            driver.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass


def supervise(command, policy, root):
    stopped = threading.Event()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    driver = None
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(2.)
    try:
        with device_lock(root, policy['usb_serial']):
            try:
                check_no_publisher(policy)
                driver = subprocess.Popen(command, start_new_session=True)
                result = prepare(driver, policy, stopped)
                print('IMU_STARTUP '+json.dumps(result, ensure_ascii=False), flush=True)
                print('IMU采样重载完成，实测 {:.2f} Hz、时间递增；传感器已就绪，可启动录包或定位。'.format(result['after']['rate_hz']), flush=True)
                while driver.poll() is None and not stopped.wait(.1):
                    pass
                return 0 if stopped.is_set() else driver.returncode
            finally:
                if driver is not None:
                    stop_driver(driver)
    except StartupCancelled:
        return 130
    except Exception as error:
        print('IMU启动准备失败：'+str(error), flush=True)
        return 1
    finally:
        socket.setdefaulttimeout(previous_timeout)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
