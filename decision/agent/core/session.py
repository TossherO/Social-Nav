"""One task record, tool dispatch, user events and model context."""
import copy
import uuid
from .memory import EvidenceMemory
from .tasks import TaskGoals
from .conversation import Conversation
from .keyframes import Keyframes
from .trajectory import Trajectory
from .context import robot_context
from ..tools.framework import TOOL_SPECS, validate_call
from ..tools.navigation import NavigationSkill, spatial_context
from ..tools.spatial import SpatialView, ReferenceMap, spatial_source_binding, layer_binding
from perception.environment.context import fingerprint

TERMINAL = {'succeeded', 'failed', 'cancelled'}


class AgentSession:
    def __init__(self, instruction, catalog, navigation, store, *, use_memory=True):
        self.store, self.navigation = store, navigation
        self.memory = EvidenceMemory() if use_memory else None
        self.observation = None
        self.spatial_view = None
        self.reference_map = None
        self.reference_catalog = catalog
        self.trajectory = Trajectory()
        self.requested_bounds = None
        self.requested_static_crop = False
        self.candidate_checks = []
        self.candidate_source = None
        self.candidate_space = None
        self.route_preview = None
        self.input_status = dict(rgb='not_provided',maps='not_provided')
        self._applying_context = None
        self._revalidated_view = None
        task_id = uuid.uuid4().hex
        self.catalog = catalog.bind_start(navigation.snapshot(), task_id)
        self.keyframes = Keyframes(store,task_id)
        if hasattr(navigation, 'catalog'):
            navigation.catalog = self.catalog
        self.task_goals = TaskGoals(self.catalog)
        if navigation.scope == 'manual':
            self.task_goals.settings['goal_position_tolerance_m'] = navigation.settings['goal_position_tolerance_m']
        self.conversation = Conversation(instruction,self.catalog.prompt(),store.event)
        self.state = dict(task_id=task_id, revision=1, instruction=instruction, status='ready', reason=None,
                          scope=navigation.scope, history=[], clarifications=[], completed_revisions=[], waiting_binding=None,
                          stop_target=None, pending_instruction=None, task_start=copy.deepcopy(self.catalog.places.get('mission_start')),
                          catalog_fingerprint=self.catalog.fingerprint, assistant_text=None)
        self.skill = NavigationSkill(navigation, self.catalog, task_id, self._save)
        self.registry = {name: getattr(self, name) for name in TOOL_SPECS}
        self._save('task_created')

    def _save(self, kind, value=None):
        self.state['navigation'] = copy.deepcopy(self.skill.record)
        self.state['evidence_memory'] = [] if self.memory is None else self.memory.snapshot()
        self.state['task_goals'] = self.task_goals.snapshot()
        self.state['spatial_view'] = None if self.spatial_view is None else copy.deepcopy(self.spatial_view.record)
        self.state['conversation'] = self.conversation.snapshot()
        self.state['candidate_checks'] = copy.deepcopy(self.candidate_checks)
        self.state['route_preview'] = copy.deepcopy(self.route_preview)
        self.store.save(self.state, kind, {} if value is None else value)

    def _history(self, kind, value):
        self.state['history'].append(dict(kind=kind, value=copy.deepcopy(value), revision=self.state['revision']))
        self.state['history'] = self.state['history'][-24:]

    def context(self):
        value = dict(task_id=self.state['task_id'], revision=self.state['revision'],
                     handle=copy.deepcopy(self.skill.record['handle']), **spatial_context(self.navigation.snapshot()))
        if self.observation is not None:
            r = self.observation.record
            value.update(observation_id=r['observation_id'], observation_binding=r['input_fingerprint'], source_stamp_s=r['stamp_s'])
            value['input_view_binding'] = fingerprint(self.observation.prompt())
        if self.spatial_view is not None:
            self.spatial_view.verify()
            value['spatial_view_id'] = self.spatial_view.record['view_id']
        value['reference_map_id'] = None if self.reference_map is None else self.reference_map.record['reference_map_id']
        value['task_requirements'] = fingerprint([g['requirement'] for g in self.task_goals.goals])
        value['candidate_checks'] = fingerprint(self.candidate_checks)
        value['route_preview'] = fingerprint(self.route_preview)
        value['conversation_revision'] = self.conversation.revision
        return value

    def valid_context(self, context):
        if context != self.context() or self.state['status'] != 'ready' or self.skill.record['context_lost']:
            return False
        current = self.navigation.snapshot()
        if not self.skill.record['connected'] or current.get('input_stale'):
            return False
        if self.spatial_view is not None and not self.spatial_view.current(current['stamp_s']):
            return False
        return (self.observation is None or
                0 <= current['stamp_s']-self.observation.record['stamp_s'] <= 3 and
                current.get('localization_status', 'valid') == 'valid')

    def valid_expired_context(self, context):
        """Permit bounded, non-actuating processing without refreshing evidence timestamps."""
        if (context != self.context() or self.state['status'] != 'ready' or
                self.skill.record['context_lost'] or not self.skill.record['connected']):
            return False
        current = self.navigation.snapshot()
        stamp = context.get('source_stamp_s')
        if stamp is None and self.spatial_view is not None:
            stamp = self.spatial_view.record.get('source_stamp_s')
        return (not current.get('input_stale') and current.get('localization_status', 'valid') == 'valid' and
                isinstance(stamp, (int, float)) and 0 <= current['stamp_s']-stamp <= 30)

    def revalidate_delayed_navigation(self, call, context):
        """Recheck a geometric target with a new view; retain the model's original call."""
        import math
        import numpy as np
        args = call['arguments']
        old = self.spatial_view
        try:
            goal = self.task_goals.current(args['goal_id'])['requirement']
        except (KeyError, ValueError):
            return dict(status='refused', reason='model_input_expired_request_fresh_observation')
        if (old is None or args['view_id'] != old.record['view_id'] or
                goal['kind'] not in ('point', 'region') or goal.get('facing') or goal.get('semantic_conditions') or
                goal.get('visual_preferences')):
            return dict(status='refused', reason='model_input_expired_request_fresh_observation')
        before = old.record.get('pose')
        self.prepare_current_images()
        current, view = self.navigation.snapshot(), self.spatial_view
        if (view is None or spatial_context(current) != {k: context.get(k) for k in ('clock_domain','local_identity','map_binding')} or
                current.get('input_stale') or current.get('localization_status', 'valid') != 'valid' or
                view.record.get('source_stamp_s') is None or not view.current(current['stamp_s']) or
                view.record.get('pose') is None or before is None):
            return dict(status='refused', reason='fresh_navigation_view_unavailable')
        after = view.record['pose']
        if (np.shape(before) != (3,) or np.shape(after) != (3,) or
                not np.isfinite(before).all() or not np.isfinite(after).all() or
                np.linalg.norm(np.asarray(before[:2])-after[:2]) > .3 or
                abs((before[2]-after[2]+math.pi)%(2*math.pi)-math.pi) > .4):
            return dict(status='refused', reason='robot_pose_changed_since_model_input')
        audit = dict(model_view_id=old.record['view_id'], model_source_stamp_s=old.record['source_stamp_s'],
                     checked_view_id=view.record['view_id'], checked_source_stamp_s=view.record['source_stamp_s'])
        self.store.event('delayed_navigation_revalidation', audit)
        self._revalidated_view = (old.record['view_id'], view.record['view_id'])
        try:
            result = self.call('navigate', _context=context, **args)
        finally:
            self._revalidated_view = None
        result['revalidation'] = audit
        return result

    def revalidate_delayed_candidate(self, call, context):
        """Recompute a candidate against a current source without rewriting its call."""
        old = self.spatial_view
        if old is None or call['arguments']['view_id'] != old.record['view_id']:
            return dict(status='refused', reason='model_input_expired_request_fresh_observation')
        self.prepare_current_images()
        current, view = self.navigation.snapshot(), self.spatial_view
        if (view is None or spatial_context(current) != {k: context.get(k) for k in ('clock_domain','local_identity','map_binding')} or
                current.get('input_stale') or current.get('localization_status', 'valid') != 'valid' or
                view.record.get('source_stamp_s') is None or not view.current(current['stamp_s'])):
            return dict(status='refused', reason='fresh_navigation_view_unavailable')
        audit = dict(model_view_id=old.record['view_id'], model_source_stamp_s=old.record['source_stamp_s'],
                     checked_view_id=view.record['view_id'], checked_source_stamp_s=view.record['source_stamp_s'])
        self.store.event('delayed_candidate_revalidation', audit)
        self._revalidated_view = (old.record['view_id'], view.record['view_id'])
        try:
            result = self.call('check_candidate', _context=context, **call['arguments'])
        finally:
            self._revalidated_view = None
        result['revalidation'] = audit
        return result

    def call(self, tool_name, *, _context=None, **arguments):
        call = validate_call(dict(tool=tool_name, arguments=arguments))
        if self.state['status'] != 'ready':
            raise ValueError('Task is not ready for another model action')
        self.store.event('tool_intent', call)
        self._applying_context = _context
        try:
            value = self.registry[tool_name](**arguments)
        except (OSError, EOFError, ValueError, KeyError) as error:
            value = dict(status='unavailable', error_type=type(error).__name__,
                         reason=str(error)[:240] if isinstance(error, ValueError) else type(error).__name__,
                         automatically_retried=False)
        finally:
            self._applying_context = None
        if tool_name == 'check_candidate' and value.get('status') == 'available':
            self._remember_candidate(value)
        if tool_name == 'navigate_route' and value.get('status') == 'refused' and value.get('diagnostics'):
            self.route_preview = dict(waypoints=copy.deepcopy(arguments['waypoints']),
                goal_id=arguments['goal_id'],view_id=arguments['view_id'],
                source_binding=self.spatial_view.record['input_source_binding'],
                diagnostics=copy.deepcopy(value['diagnostics']))
        self._history('tool', dict(call=call, result=value))
        self._save('tool_result', dict(call=call, value=value))
        return value

    def _remember_candidate(self, result):
        view = self.spatial_view.record
        source = view['input_source_binding']
        if self.candidate_source != source:
            raise ValueError('Candidate source changed before recording')
        camera = result['camera_view']
        entry = dict(pose=copy.deepcopy(result['pose']),view_id=result['view_id'],source_stamp_s=view['source_stamp_s'],
                     source_binding=source,historical_only=False,
                     static_footprint=result['static_footprint']['status'],local_footprint=result['local_footprint']['status'],
                     camera_status=camera['status'],camera_xy_m=copy.deepcopy(camera.get('camera_xy_m')),
                     points=copy.deepcopy(camera.get('points',[])),occlusion_checked=False,
                     actual_visibility_verified=False,checks=1)
        identity = (entry['pose'],sorted(p['point_id'] for p in entry['points']))
        for old in self.candidate_checks:
            if (old['pose'],sorted(p['point_id'] for p in old['points'])) == identity:
                entry['checks'] = old['checks']+1
                self.candidate_checks.remove(old)
                break
        self.candidate_checks = (self.candidate_checks+[entry])[-8:]

    def _clear_candidates(self):
        self.candidate_checks = []
        self.candidate_source = self.candidate_space = None
        self.route_preview = None

    def _sync_candidates(self, inputs):
        source = spatial_source_binding(inputs)
        space = fingerprint(dict(context=spatial_context(self.navigation.snapshot()),
                                 global_binding=layer_binding(inputs['global_grid'])))
        if self.candidate_space is not None and self.candidate_space != space:
            self.candidate_checks = []
            self.route_preview = None
        elif self.candidate_source is not None and self.candidate_source != source:
            for entry in self.candidate_checks:
                entry['historical_only'] = True
            self.route_preview = None
        self.candidate_space, self.candidate_source = space, source
        return source

    def _candidate_preview(self):
        if self.candidate_checks and not self.candidate_checks[-1]['historical_only']:
            return self.candidate_checks[-1]
        return None

    def lookup_places(self, name):
        fields = ('place_id', 'name', 'aliases', 'pose', 'reference', 'status')
        annotations = self.catalog.annotations()
        def matches(item, key):
            return not name or name in [item[key],item.get('name'),*item.get('aliases',[])]
        return dict(status='available',places=[{k:p[k] for k in fields if k in p} for p in self.catalog.lookup(name)],
                    regions=[r for r in annotations['regions'] if matches(r,'region_id')])

    def record_task(self, goals):
        if self.skill.record['handle'] is not None:
            return dict(status='refused', reason='active_navigation')
        return dict(status='recorded', goals=self.task_goals.record(goals),
                    original_instruction=self.state['instruction'], automatically_executed=False)

    def query_map(self, bounds=None):
        self.catalog.require_complete()
        inputs = self.navigation.spatial_inputs()
        result = self._build_map(inputs,bounds,static_crop=True)
        self.requested_bounds = copy.deepcopy(bounds)
        self.requested_static_crop = True
        return result

    def check_candidate(self, pose, view_id, point_ids=None):
        import math
        import numpy as np
        from navigation.planning.footprint import footprint_status, footprint_details
        from navigation.planning.inputs import load_config
        from navigation.planning.reference import planning_config
        self.catalog.require_complete()
        if self._revalidated_view is not None and view_id == self._revalidated_view[0]:
            view_id = self._revalidated_view[1]
        state = self.navigation.snapshot()
        if (self.spatial_view is None or self.spatial_view.record['view_id'] != view_id or
                not self.spatial_view.current(state['stamp_s']) or state.get('input_stale') or
                state.get('localization_status','valid') != 'valid'):
            return dict(status='refused',reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        inputs = self.navigation.spatial_inputs()
        if spatial_source_binding(inputs) != self.spatial_view.record['input_source_binding']:
            return dict(status='refused',reason='spatial_source_changed')
        pose = np.asarray(pose,dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Expected finite candidate pose')
        geometry = inputs.get('geometry')
        if geometry is None:
            return dict(status='unavailable',reason='vehicle_geometry_unavailable')
        margin = float(planning_config(load_config())['safety_margin_m'])
        grid = inputs['global_grid']
        if grid.metadata['frame_id'] != self.catalog.value['frame_id']:
            raise ValueError('Candidate and static map frame mismatch')
        occupancy = footprint_status(grid,geometry,pose,margin)
        body = dict(status=occupancy,details=footprint_details(grid,geometry,pose,margin),margin_m=margin,
                    source='static_prior_only',dynamic_obstacles_checked=False,route_checked=False,
                    geometry_assumed=geometry.assumed)
        local = transform = None
        scene = inputs.get('scene')
        if scene is not None and scene.record['geometry']['status'] == 'valid':
            from ..tools.spatial import se2
            local = scene.geometry
            if local.metadata['frame_id'] == self.catalog.value['frame_id']:
                transform = np.eye(3)
            elif inputs.get('pose') is not None and inputs.get('local_pose') is not None:
                transform = se2(inputs['pose'])@np.linalg.inv(se2(inputs['local_pose']))
        elif inputs.get('local_grid') is not None:
            matrix = np.asarray(inputs.get('local_grid_map_from_local'),dtype=float)
            if matrix.shape == (4,4) and np.isfinite(matrix).all():
                local = inputs['local_grid']
                transform = matrix[np.ix_([0,1,3],[0,1,3])]
        local_result = dict(status='unavailable',reason='no_bound_local_grid')
        if local is not None and transform is not None:
            inverse = np.linalg.inv(transform)
            xy = inverse@np.array([pose[0],pose[1],1.])
            local_yaw = pose[2]-math.atan2(transform[1,0],transform[0,0])
            local_pose = [float(xy[0]),float(xy[1]),float(local_yaw)]
            status = footprint_status(local,geometry,local_pose,margin)
            local_result = dict(status='not_covered' if status=='outside' else status,
                                source='bound_local_occupancy',source_stamp_s=inputs.get('stamp_s'),
                                details=footprint_details(local,geometry,local_pose,margin))
        sensors = inputs.get('sensors') or {}
        camera = sensors.get('camera') or {}
        view = camera.get('view') or {}
        fov = dict(status='unavailable',reason='matching_camera_geometry_unavailable',points=[])
        points = point_ids or []
        if len(points) != len(set(points)):
            raise ValueError('Duplicate point IDs')
        for point_id in points:
            if point_id not in self.catalog.places or self.catalog.places[point_id].get('pose') is None:
                raise ValueError('Unknown point ID: '+point_id)
        if sensors.get('status') == 'configured' and view.get('status') == 'approximate_pinhole':
            position = np.asarray(camera['position_m'][:2],float)
            edges = np.asarray(view['edge_yaw_rad'],float)
            if position.shape != (2,) or edges.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(edges).all():
                raise ValueError('Invalid camera geometry')
            c,s = math.cos(pose[2]),math.sin(pose[2])
            camera_xy = pose[:2]+np.array([[c,-s],[s,c]])@position
            wrap = lambda x: (x+math.pi)%(2*math.pi)-math.pi
            middle = edges[0]+wrap(edges[1]-edges[0])/2
            half = abs(wrap(edges[1]-edges[0]))/2
            results=[]
            for point_id in points:
                delta=np.asarray(self.catalog.places[point_id]['pose'][:2],float)-camera_xy
                distance=float(np.linalg.norm(delta))
                bearing=None if distance < 1e-9 else wrap(math.atan2(delta[1],delta[0])-pose[2])
                results.append(dict(point_id=point_id,distance_m=distance,bearing_from_base_rad=bearing,
                                    in_horizontal_fov=None if bearing is None else bool(abs(wrap(bearing-middle)) <= half)))
            fov=dict(status='approximate_pinhole',camera_xy_m=camera_xy.tolist(),edge_yaw_from_base_rad=edges.tolist(),
                     points=results,occlusion_checked=False,vertical_extent_checked=False,
                     actual_visibility_verified=False)
        return dict(status='available',view_id=view_id,pose=pose.tolist(),static_footprint=body,
                    local_footprint=local_result,camera_view=fov,
                    executable=False,execution_feedback=False)

    def _build_map(self, inputs, bounds, *, static_crop=False):
        self.prepare_reference(inputs['global_grid'])
        now = self.navigation.snapshot()
        self.trajectory.update(now,self.catalog.value['frame_id'],map_snapshot_id=self.catalog.value.get('map_snapshot_id'))
        trajectory = [] if inputs['stamp_s'] is None else self.trajectory.snapshot(
            frame_id=self.catalog.value['frame_id'],clock_domain=inputs['clock_domain'],now=inputs['stamp_s'])
        self._sync_candidates(inputs)
        preview = self._candidate_preview()
        self.spatial_view = SpatialView.build(self.catalog,inputs,bounds,trajectory=trajectory,
                                              static_crop=static_crop,candidate_preview=preview,
                                              route_preview=self.route_preview)
        self.store.spatial_view(self.spatial_view)
        self.store.event('trajectory_snapshot',dict(view_id=self.spatial_view.record['view_id'],segments=trajectory))
        return dict(status='available', view=self.spatial_view.record, target_candidates_generated=False)

    def prepare_reference(self, grid=None):
        if self.reference_map is not None or self.catalog.condition!='complete':
            return
        if grid is None:
            if not hasattr(self.navigation,'reference_grid'):
                return
            grid = self.navigation.reference_grid()
        self.reference_map = ReferenceMap.build(self.reference_catalog,grid)
        self.store.reference_map(self.reference_map)

    def prepare_current_images(self):
        if hasattr(self.navigation,'observe'):
            self.observation = None
            result = self.observe_scene()
            self.input_status['rgb'] = result.get('reason')
        if self.catalog.condition=='complete' and hasattr(self.navigation,'spatial_inputs'):
            current = self.navigation.snapshot()
            if (current.get('localization_status','valid')!='valid' or current.get('input_stale') or
                    self.skill.record['context_lost']):
                self.spatial_view = None
                self._clear_candidates()
                self.input_status['maps'] = 'localization_invalid_or_stale'
                return
            # Keep an explicitly requested crop while its source observation is
            # unchanged; otherwise regenerate both current maps from the source.
            try:
                inputs = self.navigation.spatial_inputs()
                source = self._sync_candidates(inputs)
                if self.spatial_view is not None:
                    self.spatial_view.verify()
                    preview = self._candidate_preview()
                    if (self.spatial_view.record.get('input_source_binding') == source and
                            self.spatial_view.record.get('candidate_preview') == preview and
                            self.spatial_view.record.get('route_preview') == self.route_preview):
                        return
                self._build_map(inputs,self.requested_bounds,static_crop=self.requested_static_crop)
                self.input_status['maps'] = None
            except (OSError,ValueError,KeyError) as error:
                self.spatial_view = None
                self._clear_candidates()
                self.input_status['maps'] = 'current_map_source_unavailable'
                self.store.event('current_maps_unavailable',dict(error_type=type(error).__name__))

    def observe_scene(self):
        observation = self.navigation.observe() if hasattr(self.navigation, 'observe') else None
        current = self.navigation.snapshot()
        if observation is None:
            return dict(status='unavailable', reason='no_current_rgb', observation_id=None)
        observation.verify()
        r = observation.record
        if (r['rgb']['status'] != 'valid' or r['clock_domain'] != current['clock_domain'] or
                r['local_identity'] != spatial_context(current)['local_identity'] or
                current.get('map_binding') is not None and r['map_binding'] != current['map_binding'] or
                current.get('input_stale') or current.get('localization_status', 'valid') != 'valid' or
                not 0 <= current['stamp_s']-r['stamp_s'] <= 3):
            return dict(status='unavailable', reason='observation_not_current_or_localization_invalid')
        self.observation = observation
        return dict(status='available', observation=observation.prompt())

    def save_keyframe(self, observation_id, reason):
        context = self._applying_context
        if (self.observation is None or context is None or not self.valid_context(context) or
                observation_id != self.observation.record['observation_id'] or
                context.get('observation_binding') != self.observation.record['input_fingerprint'] or
                context.get('input_view_binding') != fingerprint(self.observation.prompt())):
            return dict(status='refused',reason='save_requires_current_rgb_consumed_in_this_model_request')
        record = self.keyframes.save(self.observation,self.state['revision'],reason)
        return dict(status='saved',keyframe=record)

    def read_keyframe(self, keyframe_id):
        record,_ = self.keyframes.read(keyframe_id,self.state['revision'])
        current = self.navigation.snapshot()
        observation = record['observation']
        if observation['clock_domain'] != current['clock_domain'] or observation['stamp_s'] > current['stamp_s']:
            return dict(status='refused',reason='historical_keyframe_not_causal_in_current_clock')
        return dict(status='available',keyframe=record)

    def navigate(self, goal_id, view_id, orientation_source, pose=None, point_id=None, yaw=None):
        self.catalog.require_complete()
        if self._revalidated_view is not None and view_id == self._revalidated_view[0]:
            view_id = self._revalidated_view[1]
        if self.task_goals.current(goal_id)['requirement']['kind'] in ('circuit','sequence'):
            return dict(status='refused',reason='task_requires_a_route')
        now = self.navigation.snapshot()['stamp_s']
        if self.spatial_view is None or self.spatial_view.record['view_id'] != view_id or not self.spatial_view.current(now):
            return dict(status='refused', reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        if point_id is not None:
            point=self.catalog.destination(point_id,self.navigation.scope)
            pose=[*point['pose'][:2],yaw]
        # Check the submitted pose itself, even if the model checked a different
        # candidate earlier. A current local free observation can supersede a
        # static occupied cell; unknown evidence is left to the planner.
        if getattr(self.navigation,'geometry',None) is not None:
            candidate = self.check_candidate(pose,view_id)
            if candidate['status'] != 'available':
                return dict(status='refused',reason='candidate_geometry_unavailable',candidate=candidate)
            static = candidate['static_footprint']['status']
            local = candidate['local_footprint']['status']
            if local == 'occupied' or static == 'occupied' and local != 'free':
                return dict(status='refused',reason='candidate_footprint_occupied',candidate=candidate)
        selected = self.task_goals.select(goal_id,pose,orientation_source,view_id)
        if selected['status'] != 'selected':
            return selected
        if point_id is not None:
            selected['destination']['selected_point_id']=point_id
            self.task_goals.current(goal_id)['selected']['selected_point_id']=point_id
        result = self.skill.start(selected['destination'],self.state['revision'])
        if result['status'] in ('accepted', 'uncertain'):
            self.task_goals.current(goal_id)['selected']['handle'] = copy.deepcopy(result['handle'])
            self.spatial_view = self.observation = None
            self.requested_bounds = None
            self.requested_static_crop = False
            self._clear_candidates()
        return result

    def navigate_route(self, goal_id, view_id, orientation_source, waypoints, final_yaw, direction=None):
        import math
        import numpy as np
        self.catalog.require_complete()
        state=self.navigation.snapshot()
        if (self.spatial_view is None or self.spatial_view.record['view_id']!=view_id or
                self.spatial_view.record.get('pose') is None or self.spatial_view.record.get('source_stamp_s') is None or
                not self.spatial_view.current(state['stamp_s'])):
            return dict(status='refused',reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        inputs=self.navigation.spatial_inputs()
        if spatial_source_binding(inputs)!=self.spatial_view.record['input_source_binding']:
            return dict(status='refused',reason='spatial_source_changed')
        points=np.asarray(waypoints,dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or not math.isfinite(final_yaw):
            return dict(status='refused',reason='route_waypoints_must_be_finite')
        if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-6):
            return dict(status='refused',reason='route_has_duplicate_consecutive_points')
        current=self.task_goals.current(goal_id)
        requirement=current['requirement']
        if direction is not None and requirement['kind']!='circuit':
            return dict(status='refused',reason='direction_only_applies_to_circuit')
        if requirement['kind']=='circuit':
            from .route import circuit_plan_diagnostics,static_route_line_diagnostics
            if requirement.get('direction') and direction and requirement['direction']!=direction:
                return dict(status='refused',reason='route_direction_conflicts_with_user_requirement')
            direction=requirement.get('direction') or direction
            if direction is None:
                return dict(status='refused',reason='choose_a_circuit_direction')
            polygon=self.catalog.regions[requirement['target_id']]['polygon']
            diagnostics=circuit_plan_diagnostics(points,polygon,direction,self.task_goals.settings['goal_position_tolerance_m'])
            diagnostics['static_lines']=static_route_line_diagnostics(inputs['global_grid'],points)
            if diagnostics['reason']:
                return dict(status='refused',reason=diagnostics['reason'],diagnostics=diagnostics)
        if requirement['kind']=='sequence':
            index=0
            for point in points:
                if index<len(requirement['checkpoints']) and self.task_goals.checkpoint_met(requirement['checkpoints'][index],point):
                    index+=1
            if index!=len(requirement['checkpoints']) or not self.task_goals.checkpoint_met(requirement['checkpoints'][-1],points[-1]):
                return dict(status='refused',reason='route_does_not_include_ordered_user_checkpoints')
        selected=self.task_goals.select(goal_id,[*points[-1].tolist(),final_yaw],orientation_source,view_id)
        if selected['status']!='selected':return selected
        route=dict(waypoints=points.tolist(),final_yaw=float(final_yaw),
                   region_polygon=copy.deepcopy(polygon) if requirement['kind']=='circuit' else None,
                   direction=direction,direction_source=('user' if requirement.get('direction') else 'autonomous') if direction else None)
        selected['destination']['route']=route
        current['selected']['route']=copy.deepcopy(route)
        result=self.skill.start(selected['destination'],self.state['revision'])
        if requirement['kind']=='circuit' and result.get('status')=='refused':
            result['diagnostics']=diagnostics
        if result['status'] in ('accepted','uncertain'):
            current['selected']['handle']=copy.deepcopy(result['handle'])
            self.spatial_view=self.observation=None
            self.requested_bounds=None
            self.requested_static_crop=False
            self._clear_candidates()
        return result

    def check_goal(self, goal_id, observation_id=None, semantic_result=None, evidence=None,
                   facing_result=None, facing_evidence=None, preference_result=None, preference_evidence=None):
        goal = self.task_goals.current(goal_id)
        proofs = [p for p in self.skill.record['completed'] if p['goal_id'] == goal_id]
        selected = goal['selected'] or {}
        state = self.navigation.snapshot()
        live_arrival = (self.navigation.scope == 'live' and state.get('stop_observation') is True and
                        state.get('chassis_state') == 'armed' and state.get('chassis_status_fresh') is True and
                        state.get('origin_frame_id') == self.catalog.value['frame_id'] and
                        state.get('origin_stamp_s') is not None and
                        0 <= state['stamp_s']-state['origin_stamp_s'] <= .5)
        simulation_arrival = (self.navigation.scope == 'simulation' and state.get('execution_feedback') is True and
                              state.get('frame_id') == self.catalog.value['frame_id'])
        manual_arrival = (self.navigation.scope == 'manual' and state.get('manual_arrival_observation') is True and
                          state.get('origin_frame_id') == self.catalog.value['frame_id'] and
                          state.get('origin_stamp_s') is not None and
                          0 <= state['stamp_s']-state['origin_stamp_s'] <= .5 and
                          bool(proofs) and (state.get('manual_confirmation') or {}).get('handle') == proofs[-1]['handle'])
        held = self.navigation.scope == 'live' and self.skill.record['status'] == 'held'
        if ((self.skill.record['handle'] is not None and not held) or not proofs or
                proofs[-1]['handle'] != selected.get('handle') or
                (held and self.skill.record['handle'] != proofs[-1]['handle']) or
                proofs[-1]['handle']['plan_revision'] != self.state['revision'] or
                not (simulation_arrival or live_arrival or manual_arrival) or
                not state.get('stopped') or state.get('input_stale') or
                state.get('localization_status','valid') != 'valid'):
            return dict(status='unproven', reason='matching_task_navigation_and_stop_not_proven')
        proof = proofs[-1]
        measured_pose = state.get('origin_pose') if live_arrival or manual_arrival else state.get('pose')
        checks = self.task_goals.geometry(goal['requirement'], measured_pose)
        if goal['requirement']['kind']=='circuit':
            from .route import circuit_evidence
            start=proof.get('route_start_stamp_s')
            if start is None:
                return dict(status='unproven',reason='circuit_start_not_observed')
            traces=self.trajectory.snapshot(frame_id=self.catalog.value['frame_id'],clock_domain=state['clock_domain'],now=proof['stamp_s'])
            circuit=circuit_evidence(traces,self.catalog.regions[goal['requirement']['target_id']]['polygon'],
                selected['route']['direction'],start,proof['stamp_s'],self.task_goals.settings['goal_position_tolerance_m'])
            if circuit['status']!='met':
                return dict(status=circuit['status'],reason=circuit.get('reason','circuit_not_proven'),circuit=circuit)
            checks['circuit_met']=True
            checks['circuit_evidence']=circuit
        elif goal['requirement']['kind']=='sequence':
            from .route import sequence_evidence
            start=proof.get('route_start_stamp_s')
            if start is None:return dict(status='unproven',reason='sequence_start_not_observed')
            traces=self.trajectory.snapshot(frame_id=self.catalog.value['frame_id'],clock_domain=state['clock_domain'],now=proof['stamp_s'])
            sequence=sequence_evidence(traces,goal['requirement']['checkpoints'],self.task_goals.checkpoint_met,start,proof['stamp_s'])
            if sequence['status']!='met':
                return dict(status=sequence['status'],reason=sequence.get('reason','sequence_not_proven'),sequence=sequence)
            checks['sequence_met']=True
            checks['sequence_evidence']=sequence
        if goal['requirement']['kind']=='reference':
            import numpy as np
            actual=np.asarray(measured_pose,float);chosen=np.asarray(selected.get('pose'),float)
            checks['selected_position_met']=bool(actual.shape==(3,) and chosen.shape==(3,) and
                np.isfinite(actual).all() and np.isfinite(chosen).all() and
                np.linalg.norm(actual[:2]-chosen[:2])<=self.task_goals.settings['goal_position_tolerance_m'])
        if any(value is False for value in checks.values()):
            return dict(status='not_met', checks=checks)
        semantic = dict(status='not_required')
        facing = goal['requirement'].get('facing')
        if facing:
            from .tasks import angle_error
            actual,chosen=measured_pose,selected.get('pose')
            if (actual is None or chosen is None or
                    angle_error(actual[2],chosen[2]) > self.task_goals.settings['goal_yaw_tolerance_rad']):
                return dict(status='not_met',reason='actual_heading_differs_from_selected_heading',checks=checks)
            context = self._applying_context or {}
            view = self.spatial_view
            if (view is None or context.get('spatial_view_id') != view.record['view_id'] or
                    not self.valid_context(context) or view.record.get('pose') is None or
                    view.record.get('source_stamp_s') is None or view.record['source_stamp_s'] < proof['stamp_s'] or
                    facing_result is None or not facing_evidence or not facing_evidence.strip()):
                return dict(status='needs_semantic_evidence',checks=checks,
                            reason='read_a_fresh_post_arrival_map_then_judge_facing',target_id=facing)
            import numpy as np
            view_pose=np.asarray(view.record['pose'],dtype=float)
            if (view_pose.shape!=(3,) or not np.isfinite(view_pose).all() or
                    np.linalg.norm(view_pose[:2]-np.asarray(actual[:2])) > self.task_goals.settings['goal_position_tolerance_m'] or
                    angle_error(view_pose[2],actual[2]) > self.task_goals.settings['goal_yaw_tolerance_rad']):
                return dict(status='needs_semantic_evidence',checks=checks,
                            reason='map_view_pose_differs_from_current_pose',target_id=facing)
            semantic_facing=dict(status=facing_result,evidence=facing_evidence,target_id=facing,
                view_id=view.record['view_id'],stamp_s=view.record['source_stamp_s'],
                selected_yaw=chosen[2],actual_yaw=actual[2],
                source='same_agent_spatial_judgement_not_independent_ground_truth')
            if facing_result != 'satisfied':
                return dict(status='unknown' if facing_result == 'unknown' else 'not_met',
                            checks=checks,facing=semantic_facing)
            checks['facing_met']=True
            semantic.update(status='satisfied',facing=semantic_facing)
        conditions = goal['requirement'].get('semantic_conditions', [])
        if conditions:
            context = self._applying_context or {}
            observation = self.observation
            if (observation is None or observation_id != observation.record['observation_id'] or
                    context.get('observation_id') != observation_id or not self.valid_context(context) or
                    observation.record['rgb']['stamp_s'] < proof['stamp_s'] or
                    semantic_result is None or not evidence):
                return dict(status='needs_semantic_evidence', checks=checks, conditions=conditions,
                            reason='read_a_fresh_post_arrival_rgb_then_judge_all_conditions')
            semantic.update(status=semantic_result, evidence=evidence, observation_id=observation_id,
                            input_binding=observation.record['input_fingerprint'], stamp_s=observation.record['stamp_s'],
                            source='same_agent_visual_judgement_not_independent_ground_truth')
            if semantic_result != 'satisfied':
                return dict(status='unknown' if semantic_result == 'unknown' else 'not_met', checks=checks, semantic=semantic)
        preferences = goal['requirement'].get('visual_preferences', [])
        if preference_result is not None or preference_evidence is not None:
            if not preferences:
                return dict(status='refused',reason='visual_preference_not_registered')
            context=self._applying_context or {}
            observation=self.observation
            if (observation is None or observation_id != observation.record['observation_id'] or
                    context.get('observation_id') != observation_id or not self.valid_context(context) or
                    observation.record['rgb']['stamp_s'] < proof['stamp_s'] or
                    preference_result is None or not preference_evidence or not preference_evidence.strip()):
                return dict(status='needs_preference_evidence',checks=checks,preferences=preferences,
                            reason='read_a_fresh_post_arrival_rgb_then_judge_preference')
            semantic['visual_preferences']=dict(status=preference_result,evidence=preference_evidence,
                conditions=copy.deepcopy(preferences),observation_id=observation_id,
                input_binding=observation.record['input_fingerprint'],stamp_s=observation.record['stamp_s'],
                source='same_agent_visual_judgement_not_independent_ground_truth')
        elif preferences:
            semantic['visual_preferences']=dict(status='not_assessed',conditions=copy.deepcopy(preferences))
        self.task_goals.complete(goal_id,proof,checks,semantic)
        return dict(status='completed', goal_id=goal_id, checks=checks, semantic=semantic,
                    scope=self.navigation.scope, real_robot_success=False,
                    evidence_kind=('operator_confirmation_and_localization_estimate' if manual_arrival else
                                   'localization_estimate' if live_arrival else 'simulation'))

    def get_navigation_state(self):
        return self.navigation_summary()

    def cancel_navigation(self):
        result = self.skill.cancel()
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'waiting_navigation'
        return result

    def _observation_key(self):
        if hasattr(self.navigation, 'current'):
            r = self.navigation.current.record
        elif hasattr(self.navigation, 'value'):
            r = self.navigation.value.get('observation') or {}
        else:
            value = self.navigation.observe() if hasattr(self.navigation, 'observe') else None
            r = {} if value is None else value.record
        return r.get('input_fingerprint')

    def wait_for_event(self, event):
        if event == 'navigation':
            if self.skill.record['handle'] is None:
                return dict(status='available', navigation=self.skill.snapshot())
            self.state['status'] = 'waiting_navigation'
        elif event == 'observation':
            self.state.update(status='waiting_observation', waiting_binding=self._observation_key())
        else:
            f = self.navigation.snapshot()
            if self.skill.record['handle'] is not None or not (f.get('stopped') and f.get('execution_feedback') is True):
                return dict(status='refused', reason='parking_not_confirmed')
            self.state['status'] = 'waiting_user'
        return dict(status='waiting', event=event)

    def ask_user(self, question):
        if self.skill.record['handle'] is not None:
            return dict(status='refused', reason='cancel_and_confirm_navigation_first')
        self.state.update(status='needs_clarification', reason=question)
        return dict(status='waiting_user_answer', question=question)

    def finish_task(self, status, evidence):
        r = self.skill.record
        if status == 'failed':
            self.stop('failed', evidence)
            return dict(status=self.state['status'], reason=evidence)
        revision = self.state['revision']
        f = self.navigation.snapshot()
        live_arrival = (self.navigation.scope == 'live' and f.get('stop_observation') is True and
                        f.get('chassis_state') == 'armed' and f.get('chassis_status_fresh') is True)
        simulation_arrival = self.navigation.scope == 'simulation' and f.get('execution_feedback') is True
        manual_arrival = (self.navigation.scope == 'manual' and f.get('manual_arrival_observation') is True and
                          bool(r['completed']) and (f.get('manual_confirmation') or {}).get('handle') == r['completed'][-1]['handle'])
        held = self.navigation.scope == 'live' and r['status'] == 'held'
        if (not (simulation_arrival or live_arrival or manual_arrival) or (r['handle'] is not None and not held) or r['context_lost'] or
                not r['connected'] or not self.task_goals.finished() or
                not f.get('stopped') or f.get('input_stale') or
                f.get('localization_status', 'valid') != 'valid'):
            return dict(status='refused', reason='task_execution_not_proven',
                        uncompleted_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals if g['status']!='completed'])
        self.state.update(status='succeeded', reason=evidence)
        self.state['completed_revisions'].append(revision)
        return dict(status='succeeded', scope=self.navigation.scope, real_robot_success=False,
                    evidence_kind=('operator_confirmation_and_localization_estimate' if manual_arrival else
                                   'localization_estimate' if live_arrival else 'simulation'),
                    completed_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals])

    def stop(self, target, reason):
        self.state.update(stop_target=target, reason=reason)
        if self.skill.record.get('pending_completion') is not None:
            self.skill.record['pending_completion'] = None
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'stopping'
            if self.skill.record['status'] != 'stopping':
                self.skill.cancel(reason=reason)
        else:
            self.state['status'] = target
        self._save('task_stop_requested')

    def event(self, name, text=None):
        s = self.state
        previous_revision = s['revision']
        if s['status'] in TERMINAL:
            raise ValueError('Task already terminal')
        if name == 'arrived' and self.navigation.scope == 'manual':
            confirmation=self.navigation.confirm_arrival()
            self._history('user',dict(event='manual_arrival_confirmation',confirmation=confirmation))
            self.conversation.notice('manual_arrival_confirmation',confirmation=confirmation)
            self._save('manual_arrival_confirmation',confirmation)
        elif name in ('pause', 'cancel'):
            self.stop('paused' if name == 'pause' else 'cancelled', name)
        elif name == 'resume' and s['status'] == 'paused' and not self.skill.record['context_lost']:
            s.update(status='ready', reason=None)
            self._history('user', dict(event='resume'))
        elif name == 'continue' and s['status'] == 'waiting_user':
            s.update(status='ready', reason=None)
            self._history('user', dict(event='continue', source='user_event'))
        elif name == 'answer' and s['status'] == 'needs_clarification' and text and text.strip() and not self.skill.record['context_lost']:
            from .store import encode
            clarifications = [*s['clarifications'],dict(question=s['reason'],answer=text.strip())]
            if len(encode(clarifications).encode())>16384:
                raise ValueError('Clarification context exceeds 16384 bytes; replace with a concise complete instruction')
            s['clarifications'] = clarifications
            s.update(status='ready', reason=None)
            self._history('user', dict(event='answer', text=text.strip()))
        elif name == 'replace' and text and text.strip() and not self.skill.record['context_lost']:
            s['pending_instruction'] = text.strip()
            self.stop('replacing', 'user_replaced_instruction')
        else:
            raise ValueError('Event does not apply to current task')
        self._apply_replacement()
        if s['revision']==previous_revision:
            self.conversation.notice('user_event',event=name,text=text)
        self._save('user_event', dict(name=name, text=text))

    def _apply_replacement(self):
        s = self.state
        if s['status'] == 'replacing':
            s.update(instruction=s['pending_instruction'], pending_instruction=None, revision=s['revision']+1,
                     status='ready', reason=None, clarifications=[])
            self.observation = None
            self.spatial_view = None
            self.requested_bounds = None
            self.requested_static_crop = False
            self._clear_candidates()
            self.task_goals = TaskGoals(self.catalog)
            self.conversation = Conversation(s['instruction'],self.catalog.prompt(),self.store.event)
            s['assistant_text'] = None
            self._history('user', dict(event='replacement', instruction=s['instruction']))

    def tick(self):
        s = self.state
        changed = self.skill.tick()
        self.trajectory.update(self.navigation.snapshot(),self.catalog.value.get('frame_id'),
                               map_snapshot_id=self.catalog.value.get('map_snapshot_id'))
        r = self.skill.record
        if not r['connected'] and s['status'] == 'ready':
            s.update(status='waiting_navigation' if r['handle'] else 'needs_clarification', reason='navigation_connection_lost')
            self._save('task_connection_lost')
        lost_reason = r.get('reason') if r.get('reason') in ('chassis_disarmed_or_status_unavailable','operator_goal_preempted_agent') else 'spatial_context_changed'
        if r['context_lost'] and s['status'] not in TERMINAL and s.get('reason') != lost_reason:
            if self.memory is not None:
                self.memory.invalidate(**spatial_context(self.navigation.snapshot()))
            # Navigation has already requested cancellation. Never resend it.
            target = 'cancelled' if s.get('stop_target') == 'cancelled' else 'needs_clarification'
            s.update(status='stopping' if r['handle'] else target,
                     stop_target=target, reason=lost_reason)
            self._save('task_context_lost')
        if s['status'] == 'stopping' and r['handle'] is None:
            s['status'] = s['stop_target']
            self._apply_replacement()
            self._save('task_stop_confirmed')
        elif s['status'] == 'waiting_navigation' and (changed or r['handle'] is None):
            s['status'] = 'ready'
            self._history('navigation', self.skill.snapshot())
            self.conversation.notice('navigation_event',navigation=self.navigation_summary())
            self._save('navigation_event')
        elif s['status'] == 'waiting_observation' and self._observation_key() != s['waiting_binding']:
            s['status'] = 'ready'
            self.observation = None
            self._history('observation_event', dict(binding=self._observation_key()))
            self.conversation.notice('observation_event',binding=self._observation_key())
            self._save('observation_event')
        elif changed:
            self._history('navigation', self.skill.snapshot())
            self.conversation.notice('navigation_event',navigation=self.navigation_summary())
            self._save('navigation_event')
        return s

    def navigation_summary(self):
        record = self.skill.record
        feedback = self.navigation.snapshot()
        fields = ('stamp_s','clock_domain','source','session_id','generation','frame_id','pose','velocity',
                  'origin_pose','origin_frame_id','map_binding','localization_status','input_stale','stopped','execution_feedback',
                  'route_index','route_count','planning_status','planning_reason')
        return dict({key:copy.deepcopy(record[key]) for key in
                    ('status','goal_id','handle','selected_pose','reason','context_lost','connected')},
                    feedback={key:copy.deepcopy(feedback[key]) for key in fields if key in feedback},
                    last_completion=copy.deepcopy(record['completed'][-1:]))

    def prompt(self):
        self.prepare_reference()
        self.prepare_current_images()
        s, now = self.state, self.navigation.snapshot()
        goals = self.task_goals.snapshot()
        current = dict(task_id=s['task_id'],revision=s['revision'],status=s['status'],reason=s['reason'],
            clarifications=copy.deepcopy(s['clarifications']),
            task_goals=[dict(requirement=g['requirement'],status=g['status']) for g in goals],
            next_goal_id=next((g['requirement']['goal_id'] for g in goals if g['status']!='completed'),None),
            navigation=self.navigation_summary(),
            map=dict(condition=self.catalog.condition,frame_id=self.catalog.value['frame_id'],
                     catalog_fingerprint=self.catalog.fingerprint),
            recent_model_tool_choices=[] if self.memory is None else self.memory.recall(
                place_ids=[*self.catalog.places,''],now=now['stamp_s'],clock_domain=now['clock_domain']))
        if self.navigation.scope == 'manual':
            current['execution_mode'] = dict(mode='operator_driven',
                meaning='目标交给操作者使用原车手柄移动，程序不控制底盘；必须等待操作者确认和当前定位停稳检查。')
        current['keyframes'] = self.keyframes.index(s['revision'])
        current['recent_candidate_checks'] = copy.deepcopy(self.candidate_checks)
        current['requirements_registered'] = bool(goals)
        current['robot_context'] = robot_context(self.navigation)
        current['selected_goal'] = next((dict(goal_id=g['requirement']['goal_id'],
            allowed_orientation_sources=self.task_goals.orientation_sources(g['requirement']),selection=None if g['selected'] is None else
            dict({k:copy.deepcopy(g['selected'][k]) for k in ('pose','orientation_source','view_id')},
                 **({'route':{k:copy.deepcopy(g['selected']['route'][k]) for k in
                    ('waypoints','final_yaw','direction','direction_source')}} if 'route' in g['selected'] else {})))
            for g in goals if g['status']!='completed'),None)
        current['last_tool_feedback'] = None
        for entry in reversed(s['history']):
            if entry['kind']=='tool' and entry['revision']==s['revision']:
                value = entry['value'];result = value['result']
                current['last_tool_feedback'] = dict(tool=value['call']['tool'],
                    **{k:copy.deepcopy(result[k]) for k in ('status','reason','error_type','checks','conditions','diagnostics') if k in result})
                break
        origin = copy.deepcopy(s['task_start']) if self.catalog.condition=='complete' else None
        if origin is not None:
            origin['display_id'] = next(p['display_id'] for p in self.catalog.annotations()['points'] if p['place_id']=='mission_start')
        current['task_start'] = origin
        messages = self.conversation.messages()
        attachments = self.keyframes.attachments(messages,s['revision'])
        current['visible_keyframe_ids'] = list(attachments)
        return dict(messages=messages,current_state=current,reference_map=self.reference_map,keyframes=attachments,
                    input_status=copy.deepcopy(self.input_status))
