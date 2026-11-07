"""Offline FAST-LIO + ROLL mapping from a bag, with bounded backend backlog."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
BACKEND_WINDOW = 4


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('--bag', required=True, help='Read-only input ROS bag')
    value.add_argument('--out', required=True, help='New run directory beneath outputs/ or /tmp/')
    value.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES selection; otherwise preserve current visibility')
    value.add_argument('--start', type=float, default=0., help='Start offset in recorded seconds')
    value.add_argument('--duration', type=float, help='Optional recorded duration; default is the full remainder')
    value.add_argument('--lidar-imu-calibration', help='Explicit calibration YAML; otherwise current project default')
    value.add_argument('--sensor-clock-group', help='Operator-declared sensor clock session')
    value.add_argument('--sensor-session', help='Bag-bound clock declaration; otherwise BAG.clock.json if present')
    value.add_argument('--lio-backend', choices=('cpu', 'cuda'))
    value.add_argument('--roll-backend', choices=('cpu', 'cuda'))
    value.add_argument('--deskew-backend', choices=('cpu', 'cuda'))
    value.add_argument('--loop-closure', choices=('on', 'off'), help='Optional override; default preserves module configuration')
    value.add_argument('--no-record-scans', action='store_true', help='Omit ray evidence; subsequent free-space export is unavailable')
    value.add_argument('--backend-timeout', type=float, default=120., help='Maximum seconds waiting on a full backend window')
    return value


def validate(args):
    if not math.isfinite(args.start) or args.start < 0:
        raise ValueError('--start must be finite and nonnegative')
    for key in ('duration', 'backend_timeout'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('--'+key.replace('_', '-')+' must be finite and positive')
    bag = Path(args.bag).expanduser().resolve()
    if not bag.is_file() or bag.stat().st_size == 0:
        raise ValueError('Input bag must be an existing nonempty file')
    requested = Path(args.out).expanduser().absolute()
    if os.path.lexists(requested):
        raise FileExistsError('Refusing an existing output: '+str(requested))
    output = requested.resolve()
    if not any(parent in output.parents for parent in (ROOT/'outputs', Path('/tmp').resolve())):
        raise ValueError('Mapping output must be a new directory beneath project outputs/ or /tmp/')
    if output in bag.parents:
        raise ValueError('Output must not contain the input bag')
    return bag, output


def bag_metadata(path, config):
    """Use chunk connection counts, without loading all per-message indexes."""
    import rosbag
    lidar, imu = config['common']['lid_topic'], config['common']['imu_topic']
    with rosbag.Bag(str(path), 'r', skip_index=True) as data:
        connections = list(data._get_connections(topics=[lidar, imu]))
        for topic, expected in ((lidar, 'sensor_msgs/PointCloud2'), (imu, 'sensor_msgs/Imu')):
            selected = [row for row in connections if row.topic == topic]
            if not selected or any(row.datatype != expected for row in selected):
                raise ValueError('Missing or incompatible mapping topic: '+topic)
        lidar_ids = [row.id for row in connections if row.topic == lidar]
        count = sum(chunk.connection_counts.get(identifier, 0) for chunk in data._chunks for identifier in lidar_ids)
        if count == 0:
            raise ValueError('Input bag contains no LiDAR scans')
        return dict(lidar_scans=count, start_s=data.get_start_time(), end_s=data.get_end_time())


def wait_for_backend(session, stop, timeout):
    began = time.monotonic()
    while session.jobs.qsize() >= BACKEND_WINDOW:
        if session.error:
            raise RuntimeError('Mapping backend failed: '+session.error)
        if stop.is_set():
            raise InterruptedError('Mapping cancelled before the selected input completed')
        if time.monotonic()-began > timeout:
            raise RuntimeError('Mapping backend made no room for new scans within its wait budget')
        stop.wait(.005)
    if session.error:
        raise RuntimeError('Mapping backend failed: '+session.error)
    return time.monotonic()-began


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2)+'\n')


def build(args, stop=None):
    import numpy as np
    import yaml
    from runtime.config import algorithm_config, DEFAULT_CALIBRATION
    from runtime.inputs import bag_observations, selected_messages
    from runtime.metrics import Metrics
    from perception.sensors.clock_session import check_time_calibration
    from .runtime import LocalizationSession
    from .context import pose_matrix
    from .results import file_fingerprint, map_fingerprint
    from perception.environment.layers import load_config
    from perception.environment.geometry import interpolate_origins
    from perception.environment.scans import ScanJournal

    stop = stop if stop is not None else threading.Event()
    bag, output = validate(args)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config, _ = algorithm_config(mode='mapping', bag=bag, calibration=args.lidar_imu_calibration,
                                lio_backend=args.lio_backend, roll_backend=args.roll_backend,
                                deskew_backend=args.deskew_backend)
    if args.loop_closure is not None:
        config['roll']['loopClosureEnableFlag'] = args.loop_closure == 'on'
    calibration = Path(args.lidar_imu_calibration or DEFAULT_CALIBRATION).expanduser().resolve()
    timing = check_time_calibration(config, bag=bag, group=args.sensor_clock_group,
                                    session_file=args.sensor_session, source_file=calibration)
    print('检查输入元数据；仅处理雷达和 IMU，不启动 ROS master。', flush=True)
    source = bag_metadata(bag, config)
    fingerprint = file_fingerprint(bag)
    environment = load_config()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output/'algorithm.yaml'
    config_path.write_text(yaml.safe_dump(config))
    run = dict(success=False, source='bag', clock_domain='recorded_bag_offline',
               backend='native_roll_fast_lio', mode='mapping', bag=str(bag), map=str(output/'map'),
               start=args.start, duration=args.duration, algorithm_parameters=config,
               imu_time_calibration=timing, scheduling='offline_bounded_backpressure_all_selected_scans',
               driver='perception.mapping_localization.build_map',
               playback=dict(records='localization.jsonl', bag_fingerprint=fingerprint, start_offset_s=args.start))
    write_json(output/'run.json', run)
    session = journal = None
    failure = None
    counts = dict(bag_lidar_scans=source['lidar_scans'], selected_lidar_scans=0, scans=0, valid=0, journal_scans=0)
    metrics = Metrics()
    positions = deque(maxlen=64)
    began = time.monotonic()
    waited = 0.
    first = last = None
    next_progress = began+5.
    full_bag = args.start == 0. and args.duration is None

    def messages():
        for record in selected_messages(bag, config, start=args.start, duration=args.duration):
            if record[0] == config['common']['lid_topic']:
                counts['selected_lidar_scans'] += 1
            yield record

    print('开始纯建图：{}\n输出：{}'.format(bag, output), flush=True)
    try:
        session = LocalizationSession(config_path, mode='mapping', allow_local_only=True, metrics=metrics)
        if not args.no_record_scans:
            journal = ScanJournal(output/'environment/scans.sqlite', environment['scan_recording'],
                dict(config=environment, sensor_frame='rslidar',
                     evidence='frontend_accepted_deskewed_points_not_feature_keyframes'))
        with (output/'localization.jsonl').open('x') as states, (output/'frontend.jsonl').open('x') as diagnostics:
            for batch in bag_observations(None, config, messages=messages()):
                if stop.is_set():
                    raise InterruptedError('Mapping cancelled before the selected input completed')
                waited += wait_for_backend(session, stop, args.backend_timeout)
                state, association, cloud, front, match = session.step(batch)
                counts['scans'] += 1
                if first is None:
                    first = batch['end']
                last = batch['end']
                state['clock_domain'] = 'recorded_bag_offline'
                states.write(json.dumps(state, allow_nan=False, separators=(',', ':'))+'\n')
                diagnostics.write(json.dumps(dict(stamp_s=state['stamp_s'], valid=front['valid'],
                    reason=front['reason'], residual_m=float(front['residual']) if np.isfinite(front['residual']) else None,
                    gyro_bias=front['gyro_bias'].tolist(), acc_bias=front['acc_bias'].tolist(),
                    processing_ms=front['processing_ms'], matching_ms=front['matching_ms'],
                    effective_points=front['effective_points']), allow_nan=False, separators=(',', ':'))+'\n')
                for key in ('processing_ms', 'matching_ms', 'map_update_ms'):
                    metrics.add('fast_lio_'+key, front[key]/1000.)
                if state['health'] == 0:
                    counts['valid'] += 1
                    positions.append((state['stamp_s'], state['position']))
                    if journal is not None and len(cloud) and front['reason'] in (0, 4):
                        origins = interpolate_origins(cloud[:, 4], state['stamp_s'], list(positions),
                                                      environment['max_origin_interpolation_gap_s'])
                        record = dict(session_id=state['session_id'], generation=state['generation'],
                            frame_id=state['frame_id'], sensor_frame=batch['sensor_frame'], rays_used=origins is not None,
                            ray_origin_model='pose_interpolated_origins' if origins is not None else 'endpoint_only_missing_origins',
                            pose_source='native_frontend_same_observation')
                        journal.append(cloud[:, :3], pose_matrix(state['position'], state['quaternion_xyzw']),
                                       state['stamp_ns'], record, origins=origins, offset_ms=cloud[:, 4])
                        counts['journal_scans'] += 1
                else:
                    positions.clear()
                if time.monotonic() >= next_progress:
                    states.flush()
                    diagnostics.flush()
                    print(json.dumps(dict(scans=counts['scans'], valid=counts['valid'],
                        source_elapsed_s=round(last-first, 1), wall_s=round(time.monotonic()-began, 1),
                        backend_pending=session.jobs.qsize())), flush=True)
                    next_progress = time.monotonic()+5.
        if stop.is_set():
            raise InterruptedError('Mapping cancelled before map export')
        if not counts['valid']:
            raise RuntimeError('No valid frontend scans; no map will be accepted')
        if counts['scans'] != counts['selected_lidar_scans']:
            raise RuntimeError('Not all selected scans were processed: {} of {}; inspect trailing IMU coverage'.format(
                counts['scans'], counts['selected_lidar_scans']))
        if full_bag and counts['scans'] != counts['bag_lidar_scans']:
            raise RuntimeError('Not all bag scans were processed: {} of {}; inspect trailing IMU coverage'.format(
                counts['scans'], counts['bag_lidar_scans']))
        print('输入处理结束，等待后台完成并保存最终地图。', flush=True)
        session.save(str(output/'map'))
        if stop.is_set():
            raise InterruptedError('Mapping cancelled during map export')
        manifest = yaml.safe_load((output/'map/manifest.yaml').read_text())
        if (not (output/'map/COMPLETE').is_file() or manifest.get('format') != 'roll_keyframes_v1' or
                type(manifest.get('keyframes')) is not int or manifest['keyframes'] <= 0):
            raise RuntimeError('Native map export did not produce a complete nonempty keyframe map')
    except BaseException as error:
        failure = error
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                if failure is None:
                    failure = error
        if journal is not None:
            try:
                if failure is None:
                    journal.close()
                else:
                    journal.fail(str(failure))
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is None and stop.is_set():
            failure = InterruptedError('Mapping cancelled during finalization')
        if failure is None:
            try:
                run['playback']['map_fingerprint'] = map_fingerprint(output/'map')
                if journal is not None:
                    run['environment_scans_fingerprint'] = file_fingerprint(output/'environment/scans.sqlite')
                if file_fingerprint(bag) != fingerprint:
                    raise ValueError('Input bag changed during mapping')
            except Exception as error:
                failure = error
        error_text = None if failure is None else type(failure).__name__+': '+str(failure)
        run.update(success=failure is None, error=error_text,
                   termination='selected_input_end' if failure is None else 'cancelled' if isinstance(failure, InterruptedError) else 'error')
        summary = dict(success=run['success'], error=error_text, counts=counts,
            full_bag_selected=full_bag, source_interval_s=None if first is None else [first, last],
            source_span_s=None if first is None else last-first, run_wall_s=time.monotonic()-began,
            scheduling=run['scheduling'], backend_high_water=BACKEND_WINDOW,
            backpressure_wait_s=waited, metrics=metrics.summary(),
            timing_scope='wall time from native construction through map save and scan sealing; excludes initial configuration and metadata checks; stages may overlap',
            scope='mapping_only_no_detector_no_navigation_no_motion', imu_time_calibration=timing,
            verification='map_export_completed_not_localization_accuracy_or_navigation_acceptance')
        write_json(output/'run.json', run)
        write_json(output/'summary.json', summary)
    if failure is not None:
        raise failure
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    stop = threading.Event()
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    for number in previous:
        signal.signal(number, lambda *_: stop.set())
    try:
        summary = build(args, stop)
        print('建图完成：{}\n有效扫描：{} / {}'.format(Path(args.out).expanduser().resolve()/'map',
                                                       summary['counts']['valid'], summary['counts']['scans']), flush=True)
        return 0
    except InterruptedError as error:
        print(str(error)+'；本次结果标记为未完成。', flush=True)
        return 130
    except Exception as error:
        print('建图失败：'+str(error), flush=True)
        return 1
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    raise SystemExit(main())
