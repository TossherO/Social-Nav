"""Resolve existing algorithm parameters without a ROS parameter server."""
import copy
import json
import math
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION=ROOT/'perception/sensors/calibration/lidar_imu.yaml'


def runtime_settings(path):
    """Partial machine overrides inherit current transport/display defaults."""
    defaults=yaml.safe_load((ROOT/'configs/runtime.yaml').read_text())
    settings=_overlay(defaults,yaml.safe_load(Path(path).read_text()))
    display=settings['display'];route=settings['route_replay']
    for key in ('scan_hz','people_hz','map_hz','rgb_hz','navigation_hz'):
        value=display[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=60:
            raise ValueError('Display frequency must be in [0,60]: '+key)
    if type(display['max_scan_points']) is not int or not 1<=display['max_scan_points']<=200000:
        raise ValueError('Invalid display point budget')
    if type(display['path_z_m']) not in (int,float) or not math.isfinite(display['path_z_m']):
        raise ValueError('Display drawing height must be finite')
    if display['height_mode'] not in ('ground','plane'):raise ValueError('Invalid display height mode')
    radius=display['ground_plane_radius_m']
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<0:
        raise ValueError('Display floor-fit radius must be nonnegative and finite')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in route.values()):
        raise ValueError('Route replay bounds must be positive and finite')
    return settings


def _overlay(base, override):
    """ROS-style YAML overlays add keys as well as override existing values."""
    result=copy.deepcopy(base)
    for key,value in override.items():
        result[key]=_overlay(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict) else copy.deepcopy(value)
    return result


def algorithm_config(*,mode,source_run=None,map_directory=None,pose=None,pose_file=None,bag=None,start=0.,calibration=None,coverage=None,lio_backend=None,roll_backend=None,deskew_backend=None,linearization_backend=None):
    """Bind source geometry while selecting current execution backends explicitly."""
    defaults={name:yaml.safe_load((ROOT/'perception/mapping_localization/configs'/(name+'.yaml')).read_text()) for name in ('fast_lio','roll')}
    if source_run is not None:
        run=json.loads((Path(source_run)/'run.json').read_text());config=copy.deepcopy(run['algorithm_parameters'])
    else:
        run={};config={}
        for name in ('fast_lio','roll',mode):
            parameters=defaults[name] if name in defaults else yaml.safe_load((ROOT/'perception/mapping_localization/configs'/(name+'.yaml')).read_text())
            config=_overlay(config,parameters)
        config=_overlay(config,yaml.safe_load(Path(calibration or DEFAULT_CALIBRATION).read_text()))
    if source_run is not None and calibration is not None:
        override=yaml.safe_load(Path(calibration).read_text())
        config=_overlay(config,override)
        # A new calibration must not inherit the old validation/scope declaration.
        config['calibration']=copy.deepcopy(override.get('calibration') or {})
    if mode=='odometry' and source_run is not None:config=_overlay(config,yaml.safe_load((ROOT/'perception/mapping_localization/configs/odometry.yaml').read_text()))
    config.setdefault('odometry',{})['publish_state']=mode=='odometry'
    roll=config.setdefault('roll',{});roll['localizationMode']=mode=='localization'
    if mode=='localization':
        if map_directory is not None and source_run is not None and pose is None and pose_file is None:
            raise ValueError('A map override requires an explicitly bound pose or pose file')
        directory=Path(map_directory).resolve() if map_directory else Path(roll.get('saveKeyframeMapDirectory','')).parent
        if not (directory/'COMPLETE').is_file():raise ValueError('Localization needs a complete explicitly selected map')
        roll['saveKeyframeMapDirectory']=str(directory/'keyframes');roll['loopClosureEnableFlag']=False
        if pose_file:
            from perception.mapping_localization.initial_pose import load_seed
            pose=load_seed(pose_file,directory,bag,start)
        if pose is not None:roll['initialGuess']=list(pose)
        elif source_run is None:raise ValueError('Localization needs --pose or a bound --pose-file')
        if coverage:
            from perception.mapping_localization.coverage import PriorCoverage,asset_digest
            region=PriorCoverage.load(coverage,asset_fingerprint=asset_digest(directory))
            roll['localization_coverage']=dict(frame_id=region.record['frame_id'],regions_xy=region.record['regions_xy'])
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
