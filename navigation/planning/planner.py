"""Geometric reference routes, current local validation, and result acceptance."""
import copy
from collections import deque
import time
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.changes import PriorRevisions, reference_grid
from perception.environment.navigation import navigation_geometry
from .backend import GlobalPlanner
from .footprint import footprint_status, footprint_details, goal_obstruction, path_poses, swept_check, wrap
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


def goal_cell_issue(grid,geometry,goal,margin):
    x,y=cell_of(grid,goal[:2]);height,width=grid.layers['occupancy'].shape
    if not (0<=x<width and 0<=y<height):return 'goal_outside'
    value=grid.layers['occupancy'][y,x]
    if value!=0:return 'goal_unknown' if value==-1 else 'goal_occupied'
    obstruction=goal_obstruction(grid,geometry,goal,margin) if len(goal)==2 else None
    return None if obstruction is None else 'goal_footprint_'+obstruction


def follow_goal_alternatives(grid,geometry,start,goal,margin):
    """Nearby endpoints on the vehicle side of the desired following point."""
    delta=goal-start[:2];distance=np.linalg.norm(delta)
    if distance<=.25:return []
    direction=delta/distance
    lateral=np.array([-direction[1],direction[0]])
    offsets=sorted(((back,side) for back in (0.,.2,.4,.6,.8) for side in (0.,.2,-.2,.4,-.4,.6,-.6,.8,-.8)
                    if 0<np.hypot(back,side)<=.8 and distance-back>=.25),
                   key=lambda offset:np.hypot(*offset))
    choices=[]
    for back,side in offsets:
        point=goal-back*direction+side*lateral
        if goal_cell_issue(grid,geometry,point,margin):continue
        dx,dy=point-start[:2]
        yaw=np.arctan2(dy,dx)
        if any(footprint_status(grid,geometry,[*point,yaw+offset],margin)=='free'
               for offset in (0.,-.35,.35)):
            # Evaluate spatially distinct alternatives rather than adjacent
            # cells with essentially the same approach.
            if all(np.linalg.norm(point-other)>=.24 for other in choices):
                choices.append(point)
                if len(choices)==3:break
    return choices


class Planner:
    def __init__(self, config, geometry, backend=None, *, require_actual_start=False):
        self.config, self.geometry = copy.deepcopy(config), geometry
        self.backend = backend
        self.require_actual_start = require_actual_start
        # One immutable input's private costmap is reused for algorithm comparisons.
        self._cost_binding, self._costs, self._clearance = None, None, None
        self._mapped_cost_input=None
        self._reference_work=None
        self._pose_binding,self._pose_grid=None,None
        self._pose_backend=None
        self._local_planner=None
        self._prior_revisions=PriorRevisions()
        self._previous_reference=None

    def reset(self):
        """An explicit replay seek starts a fresh computational history."""
        self._cost_binding,self._costs,self._clearance=None,None,None
        self._mapped_cost_input=None
        self._reference_work=None
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
        self._reference_work=None
        return self._prior_revisions.apply(asset.snapshot,scene.geometry,scene.record['prior'],
            scene_binding=scene.record['input_fingerprint'],bounds=bounds,overlay_current=False,geometry_only=geometry_only)

    def _mapped_reference(self, scene, asset, contact, local_from_map, blocked, follow_last_position, timing):
        from .overlay import reference_overlay,retain_obstacles
        source=asset.snapshot
        r=scene.record
        # The worker sends a tuple of built-in integers. Unusual inputs still
        # go through the original validation rather than comparing equal to a
        # previously validated tuple (for example, 1.0 compares equal to 1).
        cacheable=isinstance(blocked,(list,tuple)) and all(type(index) is int for index in blocked)
        key=(asset.manifest['snapshot_id'],asset.manifest['layers_sha256'],r['input_fingerprint'],
            source.metadata['frame_id'],source.metadata['resolution'],tuple(source.metadata['origin_xy']),
            tuple(source.metadata['shape']),tuple(blocked),follow_last_position) if cacheable else None
        cached=self._reference_work
        if key is not None and cached is not None and cached[0] is source and cached[1]==key:
            return cached[2:],True
        self._reference_work=None
        if self._prior_revisions.last is not None and scene.geometry.metadata['stamp_s']<self._prior_revisions.last:
            self._mapped_cost_input=None
        began=time.perf_counter()
        revised=self._prior_revisions.apply(source,scene.geometry,r['prior'],
            scene_binding=r['input_fingerprint'],overlay_current=False,geometry_only=True)
        timing['reference_prior_revision']=(time.perf_counter()-began)*1000
        began=time.perf_counter()
        retained=retain_obstacles(revised,blocked)
        timing['reference_retained_obstacles']=(time.perf_counter()-began)*1000
        began=time.perf_counter()
        measured,overlay=reference_overlay(retained,contact,scene.geometry.layers['occupancy'],local_from_map,scene)
        timing['reference_current_overlay']=(time.perf_counter()-began)*1000
        began=time.perf_counter()
        search=measured if follow_last_position else reference_grid(measured,geometry_only=True)
        timing['reference_conditional_grid']=(time.perf_counter()-began)*1000
        # These snapshots contain only derived/static arrays, never a view into
        # the shared observation slot that the worker releases after each job.
        if key is not None:
            self._reference_work=(source,key,measured,search,overlay)
        return (measured,search,overlay),False

    def query(self, request, *, scene=None, asset=None, now=None, clock_domain=None, reference_obstacles=(),
              on_mapped_grid=None):
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
            self._query(request, scene, asset, now, clock_domain, result, reference_obstacles, on_mapped_grid)
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

    def _query(self, r, scene, asset, now, clock, out, reference_obstacles, on_mapped_grid):
        prepare_begin=time.perf_counter()
        if scene is not None: scene.verify()
        timing=out['timing_ms']
        timing['scene_verification']=(time.perf_counter()-prepare_begin)*1000
        for key in ('request_id', 'task_id', 'frame_id'):
            if not isinstance(r[key], str) or not r[key]: raise ValueError('Missing '+key)
        if type(r['goal_revision']) is not int or r['goal_revision'] < 0: raise ValueError('Invalid goal_revision')
        if r.get('target_reference') != 'base_link': raise ValueError('Goal reference must explicitly be base_link')
        mode = r['mode']
        follow_last_position='follow_last_evidence_binding' in r
        mapped_local=False
        if mode not in ('static', 'mapped', 'local'): raise ValueError('Unknown planning mode')
        goal = np.asarray(r['goal'], dtype=float)
        if goal.shape not in [(2,), (3,)] or not np.isfinite(goal).all(): raise ValueError('Invalid goal')
        selection=r.get('route_goal_selection');search_budget=None
        if selection is not None:
            search_budget=selection['search_budget_s'];detour_limit=selection['max_path_length_m']
            if (not r.get('route_binding') or not r.get('route_segment_id') or
                    type(search_budget) not in (int,float) or not np.isfinite(search_budget) or
                    not 0<search_budget<=self.config['pose_search']['max_search_s'] or
                    detour_limit is not None and (type(detour_limit) not in (int,float) or
                        not np.isfinite(detour_limit) or detour_limit<=0)):
                raise ValueError('Invalid route candidate budget')
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
            mapped_local=(mode=='local' and r.get('completion_policy')=='continuous' and
                          asset is not None and not association_status(scene,asset,self.config))
            if mode == 'mapped' or mapped_local:
                reason = association_status(scene, asset, self.config)
                if reason:
                    out.update(status='unavailable', reason=reason, goal_state='paused'); return
                grid = asset.snapshot
                transform = np.asarray(scene.record['prior']['map_from_local'])
                start = self.geometry.pose(transform @ np.asarray(scene.record['localization']['world_from_lidar']))
                if mapped_local:
                    map_from_local=se2(start)@np.linalg.inv(se2(local_pose))
                    goal=transform_poses([[*goal[:2],goal[2] if len(goal)==3 else 0.]],map_from_local)[0,:len(goal)]
                    mode='mapped'
            else:
                grid = local_grid; start = local_pose
        validate_grid(grid)
        measured_grid=grid
        if mode=='mapped':
            reference_begin=time.perf_counter()
            local_from_map=se2(local_pose)@np.linalg.inv(se2(start))
            (measured_grid,grid,overlay),reused=self._mapped_reference(
                scene,asset,contact_grid,local_from_map,reference_obstacles,follow_last_position,timing)
            out['prior_revision']=copy.deepcopy(measured_grid.metadata['prior_revision'])
            out['reference_overlay']=copy.deepcopy(overlay)
            out['reference_overlay']['retained_occupied_cells']=len(reference_obstacles)
            out['reference_work_reused']=reused
            out['reference_search_frame']=grid.metadata['frame_id']
            timing['reference_work']=(time.perf_counter()-reference_begin)*1000
            if on_mapped_grid is not None:
                on_mapped_grid(grid,local_from_map)
            if follow_last_position and out['reference_overlay']['status']!='valid':
                out.update(status='unavailable',reason='follow_last_position_projection_unavailable');return
        elif mode=='static' and not follow_last_position:
            grid=reference_grid(grid,geometry_only=True)
        manual_free_cells = getattr(asset, 'manual_free_cells', None) if mode in ('static','mapped') else 0
        if manual_free_cells is None:
            manual = asset.snapshot.layers.get('manual_edit')
            manual_free_cells = (int(np.count_nonzero(manual & (asset.snapshot.layers['occupancy']==0)))
                                 if manual is not None else 0)
        out['manual_free_present'] = manual_free_cells > 0
        out['unobserved_free_assumed']=(grid.metadata.get('free_assumption')=='no_filtered_endpoint_in_cell' or
                                        out['manual_free_present'])
        out['reference_policy']=grid.metadata.get('reference_policy',
            'unobserved_cells_assumed_free' if out['unobserved_free_assumed'] else 'observed_free_only')
        if r['frame_id'] != (scene.record['frame_id'] if mapped_local else grid.metadata['frame_id']):
            raise ValueError('Goal frame does not match planning frame')
        if mode in ('static', 'mapped') and not mapped_local and r.get('map_snapshot_id') != asset.manifest['snapshot_id']:
            raise ValueError('Goal static map identity mismatch')
        out.update(start=start.tolist(), goal=goal.tolist(), frame_id=grid.metadata['frame_id'],
                   grid_snapshot_id=grid.metadata['snapshot_id'], conservative_radius_m=self.geometry.radius,
                   safety_margin_m=self.config['safety_margin_m'])
        if mapped_local:
            out.update(start=local_pose.tolist(),goal=r['goal'],frame_id=r['frame_id'])
        out['timing_ms']['input_and_prior']=(time.perf_counter()-prepare_begin)*1000
        costs_begin = time.perf_counter()
        endpoints=[('start',start)]
        if len(goal)==3:endpoints.append(('goal',goal))
        for label, pose in endpoints:
            status = footprint_status(grid, self.geometry, pose, self.config['safety_margin_m'])
            if status != 'free':
                timing['endpoints']=(time.perf_counter()-costs_begin)*1000
                out.update(status='unavailable' if status=='unknown' else 'no_path',
                           reason=label+'_footprint_'+status,needs_observation=status=='unknown'); return
        issue=goal_cell_issue(grid,self.geometry,goal,self.config['safety_margin_m'])
        alternatives=[]
        if issue:
            if issue.startswith('goal_footprint_'):
                out['goal_evidence']=dict(method='orientation_invariant_inner_disk',status=issue[len('goal_footprint_'):])
            if (algorithm=='hybrid_astar' and self.config['search_geometry']=='footprint' and len(goal)==2 and
                    r.get('completion_policy')=='continuous' and issue in ('goal_occupied','goal_footprint_occupied')):
                alternatives=follow_goal_alternatives(grid,self.geometry,start,goal,self.config['safety_margin_m'])
                out['follow_goal_adjustment']=dict(original_blockage=issue,candidate_count=len(alternatives))
            if not alternatives:
                timing['endpoints']=(time.perf_counter()-costs_begin)*1000
                unknown=issue.endswith('unknown')
                out.update(status='unavailable' if unknown else 'no_path',reason=issue)
                if issue!='goal_outside':out['needs_observation']=unknown
                return
            goal=alternatives[0]
        timing['endpoints']=(time.perf_counter()-costs_begin)*1000
        compare_begin=time.perf_counter()
        radius=self.geometry.radius if self.config['search_geometry']=='circle' else self.geometry.inscribed_radius
        cost_parameters=(radius,self.config['max_cells'],self.config['safety_margin_m'],
            self.config.get('clearance_cost_gain',1.),self.config['clearance_cost_scale_m'],
            self.config.get('conditional_obstacle_cost'))
        mapped_cost_input=None
        if mode=='mapped':
            scene_record=scene.record
            association=scene_record['prior']
            context=(asset.manifest['snapshot_id'],asset.manifest['layers_sha256'],
                scene_record['clock_domain'],scene_record['session_id'],scene_record['generation'],
                association.get('association_session'),association.get('association_generation'),
                association.get('map_revision'),grid.metadata['frame_id'],
                grid.metadata['resolution'],tuple(grid.metadata['origin_xy']),tuple(grid.metadata['shape']),
                cost_parameters)
            cells=grid.layers['occupancy']
            conditional=grid.layers.get('conditional_obstacle')
            old=self._mapped_cost_input
            # Compare both exact raster inputs. Observation fingerprints bind
            # the result even when its newly projected geometry costs the same.
            reused=False
            if old is not None:
                old_source,old_context,old_stamp,old_cells,old_conditional=old
                reused=(asset.snapshot is old_source and context==old_context and scene_record['as_of']>=old_stamp and
                    (cells is old_cells or (cells.dtype==old_cells.dtype and np.array_equal(cells,old_cells))) and
                    (conditional is old_conditional or (conditional is not None and old_conditional is not None
                        and conditional.dtype==old_conditional.dtype and np.array_equal(conditional,old_conditional))))
            cost_binding=self._cost_binding if reused else object()
            mapped_cost_input=(asset.snapshot,context,scene_record['as_of'],cells,conditional)
        else:
            cost_binding = (grid.metadata['snapshot_id'], fingerprint(grid.metadata),
                out['input_binding']['scene'] if mode == 'local' else fingerprint(out['input_binding']['static_map']),
                cost_parameters)
        timing['costmap_comparison']=(time.perf_counter()-compare_begin)*1000
        out['costmap_reused']=cost_binding==self._cost_binding
        if cost_binding != self._cost_binding:
            from .costmap import build_costmap
            build_begin=time.perf_counter()
            self._costs,self._clearance = build_costmap(grid, radius, self.config,return_clearance=True)
            timing['costmap_build']=(time.perf_counter()-build_begin)*1000
            self._cost_binding = cost_binding
        self._mapped_cost_input=mapped_cost_input
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
                previous_path=previous['path']
                if mapped_local:previous_path=transform_poses(previous_path,map_from_local)
                tails=reference_tails(previous_path,start,self.config['motion_constraints']) if (
                    np.linalg.norm(np.asarray(previous_path[-1])[:2]-goal[:2])<=1e-6) else ()
                for candidate in tails:
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
            limit=min(.18,self.config['pose_search']['max_search_s']) if alternatives else search_budget
            began_check=0.
            for endpoint in alternatives or [goal]:
                candidate,detail=self._pose_grid.search(start,endpoint,guide=poses,max_search_s=limit)
                detail['geometry_cache_reused']=reused
                attempts.append(dict(backend='bounded_curvature_hybrid_astar',**detail))
                goal=endpoint
                poses=candidate
                check_begin=time.perf_counter()
                check=(swept_check(grid,self.geometry,poses,self.config['safety_margin_m'],self.config['max_path_samples'],prefix=False)
                       if poses is not None else dict(status='search_budget_exceeded' if '_budget' in str(detail.get('reason'))
                           else 'no_curvature_route'))
                began_check+=time.perf_counter()-check_begin
                if not alternatives or check['status']=='free':break
                poses=None
            out['timing_ms']['footprint_state_search']=(time.perf_counter()-began)*1000
            out['search_backend']='bounded_curvature_hybrid_astar'
            footprint_ms+=began_check*1000
        out['search_attempts']=attempts
        out['footprint_check'] = check
        out['timing_ms']['footprint'] = footprint_ms
        if check['status'] != 'free':
            uncertain=check['status'] in ('unknown','uncertified_clearance','budget_exceeded','search_budget_exceeded')
            out.update(status='unavailable' if uncertain else 'no_path',
                       reason='returned_path_'+check['status']); return
        out.update(status='path_found', reason=None, path=poses.tolist(), current_path_valid=True,
                   path_length_m=float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()))
        if alternatives:
            endpoint=(transform_poses([[*goal,0.]],local_from_map)[0,:2] if mapped_local else goal)
            out['follow_goal_adjustment']['selected_goal']=endpoint.tolist()
        if mode=='static' and out['unobserved_free_assumed']:
            out.update(current_path_valid=False,global_reference_only=True,needs_observation=True)
        if selection is not None and selection['max_path_length_m'] is not None and out['path_length_m']>selection['max_path_length_m']:
            out.update(status='no_path',reason='route_candidate_detour',path=[],current_path_valid=False)
            return
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
            if mapped_local:out['path']=local_poses.tolist()
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
            if (local_settings['enabled'] and not follow_last_position and out['local_start_footprint']=='free'
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
