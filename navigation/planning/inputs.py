"""Configuration, immutable geometry inputs, and explicit request references."""
from pathlib import Path
import numpy as np
import yaml
from perception.environment.scene import SceneInputs
from perception.environment.storage import StaticEnvironmentMap
from perception.detection_tracking.inference.geometry import validate_transform
from .backend import ROOT


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or Path(__file__).with_name('configs')/'planning.yaml').read_text())
    from .motion import load_motion_config
    motion=load_motion_config()
    cfg['motion_constraints']={k:motion[k] for k in ('min_turning_radius_m','allow_reverse','allow_in_place_rotation')}
    cfg.setdefault('conditional_obstacle_cost',60)
    if type(cfg['conditional_obstacle_cost']) is not int or not 1<=cfg['conditional_obstacle_cost']<=180:
        raise ValueError('Conditional obstacle cost must be an integer in [1,180]')
    if cfg['algorithm'] not in ('dijkstra', 'astar', 'hybrid_astar'):
        raise ValueError('Unknown search algorithm')
    if cfg.get('search_geometry') not in ('circle','footprint'):
        raise ValueError('Unknown search geometry')
    if not np.isfinite(cfg['clearance_cost_gain']) or not 0<=cfg['clearance_cost_gain']<=1:
        raise ValueError('Clearance cost gain must be in [0,1]')
    pose=cfg['pose_search']
    pose.setdefault('cache_heuristic',True)
    if type(pose['cache_heuristic']) is not bool:raise ValueError('Invalid heuristic cache policy')
    if (type(pose['enabled']) is not bool or type(pose['max_states']) is not int or not 1<=pose['max_states']<=600000
            or not np.isfinite(pose['reverse_penalty']) or pose['reverse_penalty']<1
            or not np.isfinite(pose['turn_weight']) or pose['turn_weight']<=0):
        raise ValueError('Invalid footprint-state search configuration')
    if (type(pose['max_expansions']) is not int or not 1<=pose['max_expansions']<=pose['max_states'] or
            not np.isfinite(pose['max_search_s']) or not 0<pose['max_search_s']<=5 or
            not np.isfinite(pose['goal_approach_straight_m']) or not 0<=pose['goal_approach_straight_m']<=2 or
            not np.isfinite(pose['radius_margin_m']) or not 0<=pose['radius_margin_m']<=1):
        raise ValueError('Invalid curvature-search budget')
    if any(not np.isfinite(pose[k]) or not 0<pose[k]<=1 for k in ('state_resolution_m','local_state_resolution_m')):
        raise ValueError('Invalid search-state resolution')
    local=cfg['local_replanning']
    route=cfg.get('route_goals')
    if route is not None:
        if (type(route['max_candidates']) is not int or not 1<=route['max_candidates']<=8 or
                any(type(route[k]) not in (int,float) or not np.isfinite(route[k]) or route[k]<=0
                    for k in ('step_m','max_detour_ratio','max_detour_extra_m'))):
            raise ValueError('Invalid bounded route-goal selection')
    if (type(local['enabled']) is not bool or type(local['reuse_free_reference']) is not bool or not np.isfinite(local['lookahead_m'])
            or not 0<local['lookahead_m']<=5.):
        raise ValueError('Local replanning needs a finite lookahead in (0,5] metres')
    for key in ('safety_margin_m', 'clearance_cost_scale_m', 'max_geometry_age_s', 'max_association_age_s'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0: raise ValueError('Invalid '+key)
    for key in ('max_cells', 'max_path_samples'):
        if type(cfg[key]) is not int or cfg[key] <= 0: raise ValueError('Invalid '+key)
    if cfg['max_cells'] > 4000000: raise ValueError('Native grid budget is four million cells')
    people=cfg['people']
    if people.get('prediction_model')!='deterministic':
        raise ValueError('Unknown people prediction model')
    if type(people['required']) is not bool or type(people['max_tracks']) is not int or not 1<=people['max_tracks']<=128:
        raise ValueError('Invalid people prediction requirement or budget')
    if any(not np.isfinite(v) or v<=0 for k,v in people.items() if k not in ('required','max_tracks','prediction_model')):
        raise ValueError('People prediction parameters must be positive and finite')
    if people.get('hard_horizon_s',people['horizon_s'])>people['horizon_s']:
        raise ValueError('People hard horizon exceeds the prediction horizon')
    for key in ('native_library', 'geometry_reference'):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(value if value.is_absolute() else ROOT/value)
    return cfg


def validate_grid(grid):
    m = grid.metadata; cells = grid.layers['occupancy']
    if (cells.ndim != 2 or list(cells.shape) != m['shape'] or not cells.size or
            not np.isfinite(m['resolution']) or m['resolution'] <= 0 or
            np.shape(m['origin_xy']) != (2,) or not np.isfinite(m['origin_xy']).all() or
            not m.get('frame_id') or not np.isin(cells, [-1, 0, 100]).all()):
        raise ValueError('Invalid environment grid')


def cell_of(grid, xy):
    return np.floor((np.asarray(xy)-grid.metadata['origin_xy'])/grid.metadata['resolution']).astype(int)


def association_status(scene, asset, config):
    r = scene.record; a = r['prior']; stamp = float(r['as_of'])
    if not a.get('valid'): return 'association_unavailable'
    if asset is None: return 'static_map_missing'
    m = asset.snapshot.metadata
    if a.get('map_id') != m.get('map_id') or a.get('map_frame') != m['frame_id']:
        return 'association_map_mismatch'
    if (a.get('local_session_id'), a.get('local_generation'), a.get('local_frame')) != (
            r['session_id'], r['generation'], r['frame_id']):
        return 'association_local_lifecycle_mismatch'
    measured = a.get('measurement_stamp_s')
    if (measured is None or not np.isfinite(measured) or a.get('stamp_s') != stamp or
            not 0 <= stamp-measured <= config['max_association_age_s']):
        return 'association_future_or_stale'
    try: validate_transform(a['map_from_local'])
    except (KeyError, ValueError): return 'association_transform_invalid'
    return None


def scene_status(scene, now, clock_domain, config, *, check_navigation=True):
    r = scene.record
    if clock_domain != r['clock_domain']: return 'clock_domain_mismatch'
    if not np.isfinite(now) or not 0 <= now-r['as_of'] <= config['max_geometry_age_s']:
        return 'geometry_future_or_stale'
    loc = r['localization']; g = r['geometry']
    if loc['status'] != 'valid' or g['status'] != 'valid': return 'localization_or_geometry_invalid'
    if loc['stamp_s'] != r['as_of']: return 'pose_observation_mismatch'
    if (g['session_id'], g['generation'], g['frame_id']) != (r['session_id'], r['generation'], r['frame_id']):
        return 'geometry_lifecycle_mismatch'
    validate_transform(loc['world_from_lidar'])
    from perception.environment.navigation import navigation_geometry
    if check_navigation:
        try: navigation_geometry(scene, now=now)
        except ValueError as error: return str(error)
    return None


def request_status(request,scene,now,clock):
    """Optional task-policy lifecycle and expiry; manual goals remain explicit."""
    r=scene.record
    if request.get('completion_policy','goal') not in ('goal','continuous'):return 'invalid_completion_policy'
    if request.get('local_identity',[r[k] for k in ('session_id','generation','frame_id')])!=[r[k] for k in ('session_id','generation','frame_id')]:
        return 'request_localization_changed'
    if request.get('clock_domain',clock)!=clock:return 'request_clock_mismatch'
    if 'valid_until_s' in request:
        until=request['valid_until_s']
        if type(until) not in (int,float) or not np.isfinite(until) or now>until:return 'request_expired'
    return None


def load_input(*, scene=None, static_map=None):
    bundle = SceneInputs.load(scene) if scene else None
    asset = StaticEnvironmentMap(static_map) if static_map else None
    if bundle: validate_grid(bundle.geometry)
    if asset: validate_grid(asset.snapshot)
    if bundle is None and asset is None: raise ValueError('A static map or complete scene is required')
    return bundle, asset
