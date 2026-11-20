"""Build localization parameters from module configuration and source provenance."""
import copy
import json
import math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION=ROOT/'perception/sensors/calibration/lidar_imu.yaml'


def ros_calibration(path):
    """Select the LiDAR/IMU parameters consumed by the ROS frontend."""
    calibration = yaml.safe_load(Path(path).read_text())
    parameters = {}
    for section, fields in (
        ('common', ('imu_time_offset', 'time_sync_en')),
        ('mapping', ('extrinsic_est_en', 'extrinsic_R', 'extrinsic_T')),
    ):
        values = calibration.get(section, {})
        selected = {name: values[name] for name in fields if name in values}
        if selected:
            parameters[section] = selected
    return parameters


def merge_config(base, override):
    """ROS-style YAML overlays add keys as well as override existing values."""
    result=copy.deepcopy(base)
    for key,value in override.items():
        result[key]=merge_config(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict) else copy.deepcopy(value)
    return result


def replay_start(run):
    playback = run.get('playback') or {}
    start = run.get('start')
    return float(playback.get('start_offset_s', 0.) or 0.) if start is None else float(start)


def _check_replay_seed(run, directory, bag, start):
    """Prefer the portable bag fingerprint. Older runs only recorded the path."""
    from .results import file_fingerprint, map_fingerprint
    playback = run.get('playback') or {}
    expected_map = playback.get('map_fingerprint')
    if expected_map is not None and map_fingerprint(directory) != expected_map:
        raise ValueError('Changed localization map requires --pose or --pose-file')
    if not math.isclose(start, replay_start(run), rel_tol=0., abs_tol=1e-9):
        raise ValueError('Changed replay start requires --pose or --pose-file')
    if bag is None:
        raise ValueError('A recorded bag is required to reuse a replay pose')
    expected = playback.get('bag_fingerprint')
    if expected is not None:
        same_bag = file_fingerprint(bag) == expected
    else:
        original = run.get('bag')
        same_bag = original and Path(original).expanduser().resolve() == Path(bag).expanduser().resolve()
    if not same_bag:
        raise ValueError('Changed replay bag requires --pose or --pose-file')


def algorithm_config(*,mode,source_run=None,map_directory=None,pose=None,pose_file=None,bag=None,start=None,calibration=None,coverage=None,lio_backend=None,roll_backend=None,deskew_backend=None,linearization_backend=None):
    """Bind source geometry while selecting current execution backends explicitly."""
    defaults={name:yaml.safe_load((ROOT/'perception/mapping_localization/configs'/(name+'.yaml')).read_text()) for name in ('fast_lio','roll')}
    if source_run is not None:
        run=json.loads((Path(source_run)/'run.json').read_text());config=copy.deepcopy(run['algorithm_parameters'])
    else:
        run={};config={}
        for name in ('fast_lio','roll',mode):
            parameters=defaults[name] if name in defaults else yaml.safe_load((ROOT/'perception/mapping_localization/configs'/(name+'.yaml')).read_text())
            config=merge_config(config,parameters)
        config=merge_config(config,yaml.safe_load(Path(calibration or DEFAULT_CALIBRATION).read_text()))
    if source_run is not None and calibration is not None:
        override=yaml.safe_load(Path(calibration).read_text())
        config=merge_config(config,override)
        # A new calibration must not inherit the old validation/scope declaration.
        config['calibration']=copy.deepcopy(override.get('calibration') or {})
        if 'imu_time_offset' not in override.get('common',{}):
            config.setdefault('common',{}).pop('imu_time_offset',None)
        config['common']['time_sync_en']=override.get('common',{}).get('time_sync_en',False)
    if mode=='odometry' and source_run is not None:config=merge_config(config,yaml.safe_load((ROOT/'perception/mapping_localization/configs/odometry.yaml').read_text()))
    config.setdefault('odometry',{})['publish_state']=mode=='odometry'
    roll=config.setdefault('roll',{});roll['localizationMode']=mode=='localization'
    if mode!='localization':
        roll.pop('localization_coverage',None)
    if mode=='localization':
        if map_directory is not None and source_run is not None and pose is None and pose_file is None:
            raise ValueError('A map override requires an explicitly bound pose or pose file')
        directory=Path(map_directory).resolve() if map_directory else Path(roll.get('saveKeyframeMapDirectory','')).parent
        if not (directory/'COMPLETE').is_file():raise ValueError('Localization needs a complete explicitly selected map')
        roll['saveKeyframeMapDirectory']=str(directory/'keyframes');roll['loopClosureEnableFlag']=False
        if start is None:
            start = replay_start(run)
        if bag is None:
            bag = run.get('bag')
        if pose_file:
            from perception.mapping_localization.initial_pose import load_seed
            pose=load_seed(pose_file,directory,bag,start)
        if pose is not None:roll['initialGuess']=list(pose)
        elif source_run is None or roll.get('initialGuess') is None:
            raise ValueError('Localization needs --pose or a bound --pose-file')
        else:
            _check_replay_seed(run, directory, bag, start)
        from perception.mapping_localization.coverage import PriorCoverage,asset_digest,full_map_coverage
        region=(PriorCoverage.load(coverage,asset_fingerprint=asset_digest(directory)) if coverage
                else full_map_coverage(directory))
        roll['localization_coverage']=region.record
    # Execution backends use current module policy, even for historical source
    # runs. Geometry, calibration and experimental math stay source-bound.
    for key,value in [('matching_backend',lio_backend),('deskew_backend',deskew_backend)]:
        value=defaults['fast_lio'][key] if value is None else value
        if value not in ('cpu','cuda'):raise ValueError(key+' must be cpu or cuda')
        config[key]=value
    roll_backend=defaults['roll']['roll']['matchingBackend'] if roll_backend is None else roll_backend
    if roll_backend not in ('cpu','cuda'):raise ValueError('roll_backend must be cpu or cuda')
    roll['matchingBackend']=roll_backend
    if roll_backend=='cuda':
        for key,value in defaults['roll']['roll'].items():
            if key.startswith('matchingCuda'):roll.setdefault(key,value)
    if linearization_backend is not None:
        if linearization_backend not in ('cpu','cuda'):raise ValueError('linearization_backend must be cpu or cuda')
        config['linearization_backend']=linearization_backend
    return config,run
