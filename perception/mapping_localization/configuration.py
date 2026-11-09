"""Build localization parameters from module configuration and source provenance."""
import copy
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION=ROOT/'perception/sensors/calibration/lidar_imu.yaml'


def merge_config(base, override):
    """ROS-style YAML overlays add keys as well as override existing values."""
    result=copy.deepcopy(base)
    for key,value in override.items():
        result[key]=merge_config(result[key],value) if isinstance(result.get(key),dict) and isinstance(value,dict) else copy.deepcopy(value)
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
