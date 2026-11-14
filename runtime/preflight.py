"""Read-only deployment inspection. Never opens a chassis or starts inference."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def probe(python, code, payload=None, *, gpu=None):
    """Keep native imports and ABI failures outside this inspection process."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = gpu
    result = subprocess.run([str(python), '-B', '-c', code], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=30, cwd=ROOT, env=env)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or f'probe exited {result.returncode}')[-1500:].strip())
    return result.stdout.strip()


def serial_access(path):
    """Metadata/owner query only: no os.open, ioctl, read or write on the device."""
    path = resolve(path).resolve(strict=True)
    if not stat.S_ISCHR(path.stat().st_mode):
        raise ValueError('Chassis port must be a character device')
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError('Chassis device requires read/write permission')
    if not shutil.which('fuser'):
        raise RuntimeError('fuser is required by the exclusive serial owner')
    result = subprocess.run(['fuser', str(path)], capture_output=True, text=True, timeout=5)
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise RuntimeError('Port is in use or ownership cannot be established: '+
                           (result.stdout+' '+result.stderr).strip())
    return dict(realpath=str(path), opened=False, ownership='no_visible_owner_at_check_time')


def inspect(config_path, *, motion_path=None, motion_profile=None, vehicle_profile=None, chassis_port=None,
            execution=False, runtime_path=None, gpu=None):
    checks, assets = [], []

    def check(name, operation):
        try:
            value = operation()
            checks.append(dict(name=name, status='ok', detail=value))
            return value
        except Exception as error:
            checks.append(dict(name=name, status='error', detail=type(error).__name__+': '+str(error)))
            return None

    def asset(name, path, action='copy'):
        if not path:
            raise ValueError('Missing path for '+name)
        path = resolve(path)
        assets.append(dict(role=name, path=str(path), resolved_path=str(path.resolve()), deployment=action))
        if not path.exists():
            raise FileNotFoundError(str(path))
        if name in ('map', 'static_map'):
            if not path.is_dir():
                raise ValueError('Expected a map directory: '+str(path))
        elif not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Expected a nonempty resource file: '+str(path))
        return str(path)

    def yaml_file(path):
        import yaml
        value = yaml.safe_load(resolve(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('Expected a YAML mapping')
        return value

    config = check('interactive_config', lambda: yaml_file(config_path))
    if config is not None:
        for key in ('map', 'static_map', 'initial_pose_hint', 'motion_config'):
            check(key, lambda key=key: asset(key, config.get(key)))
        if config.get('coverage'):
            check('coverage', lambda: asset('coverage', config['coverage']))

        def map_pair():
            from .interactive import session_inputs
            _, static_map, _ = session_inputs(resolve(config_path))
            directory = resolve(config['map'])
            manifest = yaml_file(directory/'manifest.yaml')
            if manifest.get('format') != 'roll_keyframes_v1':
                raise ValueError('Interactive runtime requires a ROLL keyframe map')
            count = manifest.get('keyframes')
            if type(count) is not int or count <= 0:
                raise ValueError('Map must declare its nonempty keyframe count')
            for index in range(count):
                for kind in ('corner', 'surf'):
                    path = directory/'keyframes'/f'{kind}{index}.pcd'
                    if not path.is_file() or path.stat().st_size == 0:
                        raise ValueError('Missing keyframe cloud: '+str(path))
            return dict(frame_id=static_map.manifest['frame_id'], matching_map_coverage_and_height_hint=True)
        check('map_pair_and_coverage', map_pair)

    def load_motion():
        from navigation.planning.motion import load_motion_config
        from .interactive import motion_selection
        path,profile = motion_selection(config or {},path=motion_path,profile=motion_profile)
        asset('selected_motion_config', path)
        return load_motion_config(path,profile=profile)
    motion = check('motion_config_valid', load_motion)

    def vehicle():
        from navigation.control.vehicle import load_vehicle_profile,controller_settings
        asset('measured_vehicle_profile', vehicle_profile)
        if motion is None:
            raise ValueError('Motion configuration must pass first')
        geometry, selected, profile = load_vehicle_profile(resolve(vehicle_profile), motion, require_execution=execution)
        return dict(vehicle_id=profile['vehicle_id'], geometry_assumed=geometry.assumed,
                    controller=controller_settings(selected),
                    execution_max_longitudinal_speed_mps=selected.get('execution_max_longitudinal_speed_mps'),
                    execution_max_angular_rps=selected.get('execution_max_angular_rps'),
                    scope='declared_measurements_checked_against_selected_limits_not_physical_verification')
    if vehicle_profile or execution:
        check('measured_vehicle_profile', vehicle)
    if chassis_port or execution:
        def port():
            if not chassis_port:
                raise ValueError('Execution inspection requires --chassis-port')
            return serial_access(chassis_port)
        check('serial_access', port)

    def runtime_config():
        from .config import runtime_settings
        return runtime_settings(resolve(runtime_path or 'configs/runtime.yaml'))
    settings = check('runtime_config', runtime_config)
    libraries = {}
    if settings is not None:
        specs = {
            'environment': (settings['environment_library'], ['sn_occupancy_new_cuda', 'sn_occupancy_suspend_free_history',
                                                             'sn_occupancy_local_ground_conflicts',
                                                             'sn_navigation_support_with_ground_conflicts']),
            'dwa': (settings['dwa_library'], ['sn_dwa_predict']),
            'collision': (settings['collision_library'], ['sn_collision_batch_stream']),
            'lio': ('outputs/runtime/build/localization/libsocial_nav_lio.so', ['sn_lio_create', 'sn_lio_step']),
            'roll': ('outputs/runtime/build/localization/libsocial_nav_roll.so', ['sn_roll_create', 'sn_roll_step']),
            'reference': ('outputs/runtime/build/planning/libsocial_nav_pose_search.so', ['social_nav_pose_plan']),
        }
        for name, (path, symbols) in specs.items():
            found = check(name+'_library', lambda name=name, path=path: asset(name+'_library', path, 'build_on_target'))
            if found:
                libraries[found] = symbols
        for path, symbols in libraries.items():
            check('library_abi:'+Path(path).name, lambda path=path, symbols=symbols: probe(sys.executable,
                'import ctypes,json,sys; p,s=json.load(sys.stdin); lib=ctypes.CDLL(p); '
                '[getattr(lib,x) for x in s]; print("loadable; required symbols present; kernels not executed")', [path, symbols]))
        check('system_ros_python', lambda: probe(settings['input_python'],
            'import rosbag,rospy,sensor_msgs.msg; print("ROS native imports available; no master started")'))
    check('project_ros_messages', lambda: probe(sys.executable,
        'import rospy,sensor_msgs.msg; from social_nav_environment.msg import EnvironmentGrid; '
        'assert "ground_conflict" in EnvironmentGrid.__slots__, "Rebuild ROS environment messages for local ground conflicts"; '
        'print("runtime ROS imports and local ground conflict field available")'))

    def detector_config():
        from perception.detection_tracking.inference.config import load_config
        value = load_config()['detector']
        if not isinstance(value, dict):
            raise ValueError('Detector resource selection must be a mapping')
        return value
    detector = check('detector_resource_selection', detector_config)
    if detector is not None:
        for key in ('config', 'checkpoint'):
            check('detector_'+key, lambda key=key: asset('detector_'+key, detector.get(key)))
        check('detector_extensions', lambda: probe(sys.executable,
            'import json,sys,torch,spconv.pytorch; '
            'from perception.detection_tracking.inference.detector import SOURCE,_load_extensions; '
            'sys.path.insert(0,str(SOURCE)); '
            'print(json.dumps(_load_extensions(json.load(sys.stdin))))', detector.get('extension_root'), gpu=gpu))
        # Do not expose unrelated optional settings or deserialize checkpoint weights.
        checks[:] = [dict(row, detail={key:detector.get(key) for key in ('config','checkpoint','extension_root')})
                     if row['name']=='detector_resource_selection' and row['status']=='ok' else row for row in checks]
    check('cuda_device', lambda: probe(sys.executable,
        'import json,torch; '
        'assert torch.cuda.is_available(), "CUDA unavailable: check device visibility/sandbox before reinstalling"; '
        'print(json.dumps(dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0), '
        'torch=torch.__version__, cuda=torch.version.cuda)))', gpu=gpu))

    def rviz():
        if not shutil.which('rviz'):
            raise RuntimeError('rviz executable unavailable')
        from .rviz_tools import verify_rviz_tools
        # RViz uses system ROS libraries, not the model interpreter's libffi/Qt.
        python = settings['input_python'] if settings is not None else '/usr/bin/python3'
        library = verify_rviz_tools(python)
        asset('rviz_ground_pose_library', library, 'build_on_target')
        return 'plugin loadable; session panel ABI verified; GUI not opened'
    check('rviz_ground_pose_plugin', rviz)
    check('sensor_calibration', lambda: asset('sensor_calibration', 'perception/sensors/calibration/lidar_imu.yaml'))
    return dict(scope='read_only_deployment_inspection', software_checks_passed=all(c['status']=='ok' for c in checks),
                execution_requested_for_inspection=execution, hardware_opened=False, execution_enabled=False,
                physical_acceptance_verified=False, checks=checks, assets=assets,
                limits=['No model weights loaded, inference, sensor subscription or chassis I/O.',
                        'CUDA visibility/import checks do not certify native kernel architecture or runtime performance.',
                        'Target machine still needs a short bag/online observation test and physical protocol/braking acceptance.',
                        'Copy model and matching map/coverage/hint assets separately from Git; rebuild native/ROS/detector extensions on target.',
                        'Map matching uses existing asset fingerprints; a successful check is not a survey of the live scene.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='navigation/planning/configs/interactive.yaml')
    parser.add_argument('--motion-config')
    parser.add_argument('--motion-profile')
    parser.add_argument('--runtime-config')
    parser.add_argument('--vehicle-profile')
    parser.add_argument('--chassis-port')
    parser.add_argument('--gpu')
    parser.add_argument('--execution-check', action='store_true', help='Also require a confirmed vehicle profile and idle device; never execute')
    args = parser.parse_args(argv)
    report = inspect(args.config, motion_path=args.motion_config, motion_profile=args.motion_profile, runtime_path=args.runtime_config,
                     vehicle_profile=args.vehicle_profile, chassis_port=args.chassis_port,
                     execution=args.execution_check, gpu=args.gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['software_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
