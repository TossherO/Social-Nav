"""Geometric reference routes, current local validation, and result acceptance."""
import copy
import hashlib
from collections import deque
import time
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.changes import PriorRevisions, reference_grid
from perception.environment.navigation import navigation_geometry
from .backend import GlobalPlanner
from .footprint import footprint_status, footprint_details, path_poses, swept_check, wrap
from .inputs import association_status, scene_status, cell_of, validate_grid,request_status
from .curvature import motion_valid,reference_tails


def se2(pose):
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0., 0., 1.]])


def transform_poses(poses, transform):
    result = np.asarray(poses).copy()
    result[:, :2] = result[:, :2] @ transform[:2, :2].T+transform[:2, 2]
    result[:, 2] += np.arctan2(transform[1, 0], transform[0, 0])
    return result


class Planner:
    def __init__(self, config, geometry, backend=None, *, require_actual_start=False):
        self.config, self.geometry = copy.deepcopy(config), geometry
        self.backend = backend
        self.require_actual_start = require_actual_start
        # One immutable input's private costmap is reused for algorithm comparisons.
        self._cost_binding, self._costs, self._clearance = None, None, None
        self._pose_binding,self._pose_grid=None,None
        self._pose_backend=None
        self._local_planner=None
        self._prior_revisions=PriorRevisions()
        self._previous_reference=None

    def reset(self):
        """An explicit replay seek starts a fresh computational history."""
        self._cost_binding,self._costs,self._clearance=None,None,None
        self._pose_binding,self._pose_grid=None,None
        self._pose_backend=None
        self._prior_revisions.reset()
        self._previous_reference=None
        if self._local_planner is not None:self._local_planner.reset()

    def observe(self, scene, asset, *, now, clock_domain, bounds=None,geometry_only=False):
        """Update prior corrections without a goal/search; invalid joins do not mutate history."""
        scene.verify()
        reason=scene_status(scene,now,clock_domain,self.config) or association_status(scene,asset,self.config)
        if reason:raise ValueError(reason)
        return self._prior_revisions.apply(asset.snapshot,scene.geometry,scene.record['prior'],
            scene_binding=scene.record['input_fingerprint'],bounds=bounds,overlay_current=False,geometry_only=geometry_only)

    def query(self, request, *, scene=None, asset=None, now=None, clock_domain=None):
        begin = time.perf_counter()
        request = copy.deepcopy(request)
        result = dict(status='invalid_input', reason=None, executable=False, current_path_valid=False,
            request=request, request_binding=fingerprint(request), path=[], frame_id=request.get('frame_id'),
            geometry_assumed=self.geometry.assumed, geometry_source=self.geometry.source,
            reference_point='base_link', generated_at_unix_s=time.time(), timing_ms={})
        binding = dict(scene=None if scene is None else scene.record['input_fingerprint'],
            static_map=None if asset is None else dict(snapshot_id=asset.manifest['snapshot_id'],
                 layers_sha256=asset.manifest['layers_sha256']))
        if scene is not None:
            binding.update(observation_id=scene.record['observation_id'], as_of=scene.record['as_of'],
                clock_domain=scene.record['clock_domain'], local_session=scene.record['session_id'],
                local_generation=scene.record['generation'], association=scene.record['prior'])
        result['input_binding'] = binding
        try:
            self._query(request, scene, asset, now, clock_domain, result)
        except (ValueError, KeyError, TypeError) as error:
            result.update(status='invalid_input', reason=str(error), path=[], current_path_valid=False)
        except (OSError, RuntimeError) as error:
            result.update(status='unavailable', reason=str(error), path=[], current_path_valid=False)
        if result.get('reference_continuity',{}).get('start_is_actual') is False:
            result.update(current_path_valid=False,global_reference_only=True)
            if result.get('current_local_path'):
                result['current_local_path'].update(valid=False,reason='requires_actual_start_motion_join')
        result['timing_ms']['total'] = (time.perf_counter()-begin)*1000
        result['result_binding'] = fingerprint(result)
        if scene is not None and result['status']=='path_found':
            from .continuity import context_binding
            continuity=result.get('reference_continuity',{})
            self._previous_reference=dict(context=context_binding(request,scene,asset),path=result['path'],
                source_result=result['result_binding'],origin_result=continuity.get('origin_result',result['result_binding']),
                origin_as_of=continuity.get('origin_as_of',result['input_binding']['as_of']),
                verified_as_of=result['input_binding']['as_of'])
        return result

    def _query(self, r, scene, asset, now, clock, out):
        prepare_begin=time.perf_counter()
        if scene is not None: scene.verify()
        for key in ('request_id', 'task_id', 'frame_id'):
            if not isinstance(r[key], str) or not r[key]: raise ValueError('Missing '+key)
        if type(r['goal_revision']) is not int or r['goal_revision'] < 0: raise ValueError('Invalid goal_revision')
        if r.get('target_reference') != 'base_link': raise ValueError('Goal reference must explicitly be base_link')
        mode = r['mode']
        if mode not in ('static', 'mapped', 'local'): raise ValueError('Unknown planning mode')
        goal = np.asarray(r['goal'], dtype=float)
        if goal.shape not in [(2,), (3,)] or not np.isfinite(goal).all(): raise ValueError('Invalid goal')
        algorithm = r.get('algorithm', self.config['algorithm'])
        if algorithm not in ('dijkstra', 'astar','hybrid_astar'): raise ValueError('Unknown algorithm')
        out['algorithm'] = algorithm
        if mode == 'static':
            if asset is None: raise ValueError('Static query needs an explicit static map')
            grid = asset.snapshot
            start = np.asarray(r['start'], dtype=float)
            if start.shape != (3,) or not np.isfinite(start).all(): raise ValueError('Static start must be [x,y,yaw]')
            out['start_source'] = 'explicit_static_diagnostic'
        else:
            if scene is None: raise ValueError('Local/mapped query needs a complete scene')
            from .people import PeopleModel
            local_grid,_,dynamic=PeopleModel(scene,self.config['people']).navigation_grid(scene)
            out['dynamic_geometry']=dynamic
            reason = scene_status(scene, now, clock, self.config) or request_status(r,scene,now,clock)
            if reason:
                out.update(status='unavailable', reason=reason); return
            local_pose = self.geometry.pose(scene.record['localization']['world_from_lidar'])
            contact_grid=navigation_geometry(scene,now=now,asset=asset,check_asset=True)
            out['navigation_geometry']=scene.geometry.metadata.get('navigation_fusion')
            out['local_start_footprint'] = footprint_status(contact_grid, self.geometry, local_pose,
                                                           self.config['safety_margin_m'])
            out['local_start_evidence']=footprint_details(contact_grid,self.geometry,local_pose,self.config['safety_margin_m'])
            if 'start' in r: raise ValueError('Scene start is derived from localization, not overridden')
            out['start_source'] = 'observation_localization_and_vehicle_reference'
            if mode == 'mapped':
                reason = association_status(scene, asset, self.config)
                if reason:
                    out.update(status='unavailable', reason=reason, goal_state='paused'); return
                grid = asset.snapshot
                transform = np.asarray(scene.record['prior']['map_from_local'])
                start = self.geometry.pose(transform @ np.asarray(scene.record['localization']['world_from_lidar']))
            else:
                grid = local_grid; start = local_pose
        validate_grid(grid)
        if mode=='mapped':
            grid=self.observe(scene,asset,now=now,clock_domain=clock,geometry_only=True)
            out['prior_revision']=grid.metadata['prior_revision']
            from .overlay import reference_overlay
            local_from_map=se2(local_pose)@np.linalg.inv(se2(start))
            grid,out['reference_overlay']=reference_overlay(grid,contact_grid,scene.geometry.layers['occupancy'],local_from_map,scene)
        measured_grid=grid
        if mode in ('static','mapped'):
            grid=reference_grid(grid,geometry_only=True)
        out['reference_policy']=grid.metadata.get('reference_policy','observed_free_only')
        if r['frame_id'] != grid.metadata['frame_id']: raise ValueError('Goal frame does not match planning frame')
        if mode in ('static', 'mapped') and r.get('map_snapshot_id') != asset.manifest['snapshot_id']:
            raise ValueError('Goal static map identity mismatch')
        out.update(start=start.tolist(), goal=goal.tolist(), frame_id=grid.metadata['frame_id'],
                   grid_snapshot_id=grid.metadata['snapshot_id'], conservative_radius_m=self.geometry.radius,
                   safety_margin_m=self.config['safety_margin_m'])
        out['timing_ms']['input_and_prior']=(time.perf_counter()-prepare_begin)*1000
        costs_begin = time.perf_counter()
        endpoints=[('start',start)]
        if len(goal)==3:endpoints.append(('goal',goal))
        for label, pose in endpoints:
            status = footprint_status(grid, self.geometry, pose, self.config['safety_margin_m'])
            if status != 'free':
                out.update(status='unavailable' if status=='unknown' else 'no_path',
                           reason=label+'_footprint_'+status,needs_observation=status=='unknown'); return
        goal_cell=cell_of(grid,goal[:2]);height,width=grid.layers['occupancy'].shape
        if (goal_cell<0).any() or goal_cell[0]>=width or goal_cell[1]>=height:
            out.update(status='no_path',reason='goal_outside');return
        value=grid.layers['occupancy'][goal_cell[1],goal_cell[0]]
        if value!=0:
            out.update(status='unavailable' if value==-1 else 'no_path',
                       reason='goal_unknown' if value==-1 else 'goal_occupied',needs_observation=bool(value==-1));return
        cost_binding = (grid.metadata['snapshot_id'], fingerprint(grid.metadata),
                       out['input_binding']['scene'] if mode == 'local' else fingerprint(out['input_binding']['static_map']))
        if mode=='mapped':
            # Observation time still binds every result/local check. Reuse only
            # the GLOBAL search graph when its actual geometry/cost inputs agree.
            geometry_hash=hashlib.sha256(grid.layers['occupancy'].tobytes())
            if 'conditional_obstacle' in grid.layers:geometry_hash.update(grid.layers['conditional_obstacle'].tobytes())
            cost_binding=(fingerprint(out['input_binding']['static_map']),geometry_hash.hexdigest(),
                grid.metadata['resolution'],tuple(grid.metadata['origin_xy']),tuple(grid.metadata['shape']))
        out['costmap_reused']=cost_binding==self._cost_binding
        if cost_binding != self._cost_binding:
            from .costmap import build_costmap
            radius=self.geometry.radius if self.config['search_geometry']=='circle' else self.geometry.inscribed_radius
            self._costs,self._clearance = build_costmap(grid, radius, self.config,return_clearance=True)
            self._cost_binding = cost_binding
        costs = self._costs
        out['proposal_radius_m']=self.geometry.radius if self.config['search_geometry']=='circle' else self.geometry.inscribed_radius
        out['timing_ms']['costmap_and_endpoints'] = (time.perf_counter()-costs_begin)*1000
        out['searchable_cells'] = int(np.count_nonzero(costs < 253))
        s, g = cell_of(grid, start[:2]), cell_of(grid, goal[:2])
        poses=None;attempts=[]
        previous=self._previous_reference
        if scene is not None and previous is not None and self.config['local_replanning']['reuse_free_reference']:
            from .continuity import context_binding
            if previous['context']==context_binding(r,scene,asset) and previous['verified_as_of']<=scene.record['as_of']:
                for candidate in reference_tails(previous['path'],start,self.config['motion_constraints']):
                    actual_start=bool(np.linalg.norm(candidate[0,:2]-start[:2])<1e-6 and abs(wrap(candidate[0,2]-start[2]))<1e-6)
                    if self.require_actual_start and not actual_start:continue
                    if swept_check(grid,self.geometry,candidate,self.config['safety_margin_m']+.002,
                            self.config['max_path_samples'],prefix=False)['status']=='free':
                        poses=candidate
                        out['reference_continuity']=dict(source_result=previous['source_result'],
                            origin_result=previous['origin_result'],origin_as_of=previous['origin_as_of'],
                            verified_as_of=out['input_binding']['as_of'],
                            current_scene_binding=out['input_binding']['scene'],
                            start_is_actual=actual_start,
                            method='retained_route_tail_and_full_current_grid_sweep')
                        break
        out['search_geometry']=self.config['search_geometry']
        blocked=next((name for name,cell in [('start',s),('goal',g)] if costs[cell[1],cell[0]]>=253),None)
        if poses is None and blocked is None and algorithm!='hybrid_astar':
            if self.backend is None: self.backend = GlobalPlanner(self.config['native_library'])
            cells, elapsed = self.backend.search(costs, s, g, algorithm)
            out['timing_ms']['native_search_and_traceback'] = elapsed
            if len(cells):
                xy=(cells+.5)*grid.metadata['resolution']+grid.metadata['origin_xy']
                xy=np.vstack((start[:2],xy,goal[:2]))
                poses=path_poses(xy,start[2],goal[2] if len(goal)==3 else None)
            else:attempts.append(dict(backend='noetic_'+algorithm,reason='search_or_traceback_failed'))
        elif poses is None and blocked is not None:attempts.append(dict(backend='search_endpoint',reason=blocked+'_search_circle_blocked'))
        check_begin = time.perf_counter()
        check=(swept_check(grid,self.geometry,poses,self.config['safety_margin_m'],self.config['max_path_samples'],prefix=False)
               if poses is not None else dict(status='not_generated'))
        footprint_ms=(time.perf_counter()-check_begin)*1000
        if poses is not None and not motion_valid(poses,self.config['motion_constraints']):
            check=dict(status='curvature_reference_required')
        out['search_backend']='revalidated_curvature_reference' if poses is not None and 'reference_continuity' in out else 'noetic_'+algorithm
        if check['status']!='free' and self.config['search_geometry']=='footprint' and self.config['pose_search']['enabled']:
            from .pose_grid import PoseGrid
            attempts.append(dict(backend=out['search_backend'],footprint_check=check))
            pose_binding=(cost_binding,mode)
            began=time.perf_counter()
            reused=pose_binding==self._pose_binding
            if not reused:
                if self._pose_backend is None:
                    from .backend import FootprintSearch
                    self._pose_backend=FootprintSearch(self.config['pose_search'].get('native_library',self.config['native_library']),
                        cache_heuristic=self.config['pose_search'].get('cache_heuristic',True))
                self._pose_grid=PoseGrid(grid,self.geometry,self.config,costs=costs,clearance=self._clearance,
                    local=mode=='local',backend=self._pose_backend)
                self._pose_binding=pose_binding
            poses,detail=self._pose_grid.search(start,goal,guide=poses)
            detail['geometry_cache_reused']=reused
            out['timing_ms']['footprint_state_search']=(time.perf_counter()-began)*1000
            attempts.append(dict(backend='bounded_curvature_hybrid_astar',**detail))
            out['search_backend']='bounded_curvature_hybrid_astar'
            check_begin=time.perf_counter()
            check=(swept_check(grid,self.geometry,poses,self.config['safety_margin_m'],self.config['max_path_samples'],prefix=False)
                   if poses is not None else dict(status='search_budget_exceeded' if '_budget' in str(detail.get('reason'))
                       else 'no_curvature_route'))
            footprint_ms+=(time.perf_counter()-check_begin)*1000
        out['search_attempts']=attempts
        out['footprint_check'] = check
        out['timing_ms']['footprint'] = footprint_ms
        if check['status'] != 'free':
            uncertain=check['status'] in ('unknown','uncertified_clearance','budget_exceeded','search_budget_exceeded')
            out.update(status='unavailable' if uncertain else 'no_path',
                       reason='returned_path_'+check['status']); return
        out.update(status='path_found', reason=None, path=poses.tolist(), current_path_valid=True,
                   path_length_m=float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()))
        if grid is not measured_grid:
            audit=swept_check(measured_grid,self.geometry,poses,self.config['safety_margin_m'],
                              self.config['max_path_samples'],prefix=False)
            out['conditional_reference']=audit['status']!='free'
            out['historical_occupancy_check']=audit
            out['footprint_check']['scope']='conditional_global_reference'
            if out['conditional_reference']:
                out.update(current_path_valid=False,global_reference_only=True,needs_observation=True)
        if mode == 'mapped':
            check_begin = time.perf_counter()
            # A planar navigation reference anchored at the same current base pose.
            # No cross-grid interpolation or 3D motion feasibility is implied.
            local_from_map = se2(local_pose) @ np.linalg.inv(se2(start))
            local_poses = transform_poses(poses, local_from_map)
            local_check = swept_check(local_grid, self.geometry, local_poses,
                self.config['safety_margin_m'], self.config['max_path_samples'])
            out['local_check'] = local_check
            out['local_check']['projection'] = 'SE2_at_current_base_pose'
            if local_check['status'] == 'free':
                prefix = local_poses.tolist()
            elif local_check.get('last_valid_pose') is not None:
                prefix = local_poses[:local_check['failed_segment']+1].tolist()+[local_check['last_valid_pose']]
            else:
                prefix = []
            out['current_local_path'] = dict(frame_id=scene.record['frame_id'], poses=prefix,
                complete=local_check['status'] == 'free', executable=False,
                valid=local_check['status']=='free' or local_check['valid_distance_m']>1e-6,
                source='static_reference_prefix')
            out['current_path_valid'] = local_check['status'] == 'free'
            out['needs_observation']=local_check['status'] in ('unknown','outside')
            out['timing_ms']['local_check'] = (time.perf_counter()-check_begin)*1000
            out['global_reference_only'] = not out['current_path_valid']
            local_settings=self.config['local_replanning']
            if (local_settings['enabled'] and out['local_start_footprint']=='free'
                    and local_check['status']!='free' and local_check['valid_distance_m']<local_settings['lookahead_m']):
                began=time.perf_counter()
                remaining=local_settings['lookahead_m'];travel=0.;target=None
                for a,b in zip(local_poses[:-1],local_poses[1:]):
                    length=float(np.linalg.norm(b[:2]-a[:2]))
                    if length<1e-10:continue
                    fraction=min(1.,remaining/length)
                    target=np.r_[a[:2]+fraction*(b[:2]-a[:2]),a[2]+fraction*wrap(b[2]-a[2])]
                    travel+=fraction*length;remaining-=fraction*length
                    if remaining<=1e-8:break
                if target is not None:
                    if self._local_planner is None:self._local_planner=Planner(self.config,self.geometry)
                    total_length=float(np.linalg.norm(np.diff(local_poses[:,:2],axis=0),axis=1).sum())
                    # A reference heading is a hint, not a requested terminal
                    # orientation. Preserve hard yaw only at the actual goal.
                    child_goal=(np.r_[target[:2],local_poses[-1,2]].tolist()
                                if len(r['goal'])==3 and travel>=total_length-1e-8 else target[:2].tolist())
                    child=dict(request_id=r['request_id']+':local-reference',
                        task_id=r['task_id']+':local-reference:'+scene.record['observation_id'],goal_revision=0,
                        mode='local',frame_id=scene.record['frame_id'],target_reference='base_link',goal=child_goal,
                        parent_request_binding=out['request_binding'],parent_scene_binding=out['input_binding']['scene'],
                        reference_path_fingerprint=fingerprint(poses.tolist()),reference_distance_m=travel,
                        reference_heading_rad=float(target[2]),
                        association_binding=fingerprint(scene.record['prior']))
                    local=self._local_planner.query(child,scene=scene,now=now,clock_domain=clock)
                    out['local_replan']=local
                    if local['status']=='path_found':
                        out['current_local_path'].update(poses=local['path'],valid=True,source='local_replan',
                            reference_goal=child_goal,reference_distance_m=travel)
                out['timing_ms']['local_replan']=(time.perf_counter()-began)*1000
        else:
            out['local_check'] = dict(status='not_applicable' if mode == 'static' else 'free')


class PlanningSession:
    """Serial owner of one query; call begin/invalidate/accept on that owner."""
    def __init__(self):
        self.epoch = 0; self.current = None; self.latest = None
        self.local_identity = None; self.blocked_goal = None; self.goal_revisions = {}; self.goals = {}
        self.observation_stamp = None; self.retired = deque(maxlen=32)

    def invalidate(self, reason):
        self.epoch += 1; self.current = None; self.latest = None
        if reason == 'seek':
            self.local_identity = None; self.observation_stamp = None; self.blocked_goal = None
            self.retired.clear(); self.goals.clear(); self.goal_revisions.clear()
        return dict(status='unavailable', reason=reason, executable=False)

    def begin(self, request, scene=None, asset=None):
        self.invalidate('new_query')
        key = (request['task_id'], request['goal_revision'])
        previous = self.goal_revisions.get(request['task_id'], -1)
        if key[1] < previous: raise ValueError('Goal revision is older than the latest accepted request')
        goal = fingerprint(dict({k:request.get(k) for k in ('goal', 'frame_id', 'target_reference', 'mode', 'map_snapshot_id','motion_intent')},
            goal_behavior=request.get('goal_behavior','pass_through')))
        if key in self.goals and self.goals[key] != goal: raise ValueError('Changed goal requires a new goal revision')
        if scene is not None:
            r = scene.record; identity = (r['session_id'], r['generation'], r['frame_id'])
            if identity in self.retired or (identity == self.local_identity and self.observation_stamp is not None
                                            and r['as_of'] < self.observation_stamp):
                raise ValueError('Out-of-order observation requires an explicit replay seek')
            if self.local_identity is not None and identity != self.local_identity and key[1] <= previous:
                self.blocked_goal = key
            if self.local_identity is not None and identity != self.local_identity: self.retired.append(self.local_identity)
            self.local_identity = identity
            self.observation_stamp = r['as_of']
        if key == self.blocked_goal: raise ValueError('Localization reset requires an explicit new goal revision')
        self.goal_revisions[key[0]] = key[1]
        self.goals[key] = goal
        map_binding = None if asset is None else dict(snapshot_id=asset.manifest['snapshot_id'], layers_sha256=asset.manifest['layers_sha256'])
        self.latest = (self.epoch, fingerprint(request), None if scene is None else scene.record['input_fingerprint'], fingerprint(map_binding))
        return self.latest

    def accept(self, ticket, result):
        if (ticket != self.latest or ticket[1] != result['request_binding'] or ticket[2] != result['input_binding']['scene']
                or ticket[3] != fingerprint(result['input_binding']['static_map'])
                or result.get('result_binding') != fingerprint({k:v for k,v in result.items() if k != 'result_binding'})):
            return False
        self.current = result if result['status'] == 'path_found' and result['current_path_valid'] else None
        return True
