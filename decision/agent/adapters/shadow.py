"""Nonexecuting adapters for causal saved observations and runtime shadow IPC."""
import copy
import math
from interfaces.runtime import source_stamp_ns
from perception.environment.context import fingerprint
from .. import ROOT


def mapped_origin(catalog, record):
    """Only expose a map origin when the observation names this map asset."""
    binding = record.get('map_binding') or {}
    map_id = binding.get('map_id')
    asset = catalog.value.get('map_snapshot_id', '')
    source_id = catalog.value.get('localization_map_id')
    matches = binding.get('valid') and map_id and (source_id == map_id if source_id else
              asset == map_id or asset.startswith(map_id+':scan_fusion:'))
    return dict(map_binding=copy.deepcopy(binding), origin_pose=record.get('map_base_pose') if matches else None,
                origin_frame_id=binding.get('map_frame') if matches else None, origin_stamp_s=record.get('stamp_s'))


def apply_runtime_mapped_pose(state, control, catalog):
    """Use the current causal map pose; never revive an older RGB pose on loss."""
    if 'agent_mapped_pose' not in control:return state
    mapped=control['agent_mapped_pose'] or {}
    stamp=mapped.get('stamp_s')
    binding=mapped.get('map_binding') or {}
    map_id=catalog.value.get('localization_map_id')
    asset=catalog.value.get('map_snapshot_id','')
    bound_id=binding.get('map_id')
    same_map=(bound_id==map_id if map_id else bool(bound_id and
        (asset==bound_id or asset.startswith(bound_id+':scan_fusion:'))))
    valid=(binding.get('valid') is True and binding==state.get('map_binding') and
        same_map and binding.get('map_frame')==catalog.value.get('frame_id') and
        mapped.get('local_identity')==[state.get(k) for k in ('session_id','generation','frame_id')] and
        mapped.get('clock_domain')==state.get('clock_domain') and
        type(stamp) in (int,float) and math.isfinite(stamp) and 0<=state['stamp_s']-stamp<=.5 and
        isinstance(mapped.get('pose'),list) and len(mapped['pose'])==3 and
        all(type(v) in (int,float) and math.isfinite(v) for v in mapped['pose']))
    state['origin_pose']=copy.deepcopy(mapped['pose']) if valid else None
    state['origin_stamp_s']=stamp if valid else None
    state['origin_frame_id']=binding.get('map_frame') if valid else None
    state['association_measurement_stamp_s']=mapped.get('association_stamp_s') if valid else None
    return state


def bound_planning_observation(evidence, request, state):
    """Expose only outcomes for the active request and current spatial context."""
    if (not isinstance(evidence,dict) or not request or state.get('input_stale') or
            state.get('localization_status')!='valid' or state.get('environment_fault')):
        return None
    from navigation.planning.continuity import goal_binding
    stamp=evidence.get('source_stamp_s');limit=evidence.get('valid_until_s');now=state['stamp_s']
    if (any(type(t) not in (int,float) or not math.isfinite(t) for t in (stamp,limit,now)) or
            not stamp<=now<=limit or evidence.get('clock_domain')!=state['clock_domain'] or
            evidence.get('local_identity')!=[state.get(k) for k in ('session_id','generation','frame_id')] or
            evidence.get('map_binding')!=state.get('map_binding') or
            any(evidence.get(k)!=request[k] for k in ('request_id','task_id','goal_revision')) or
            evidence.get('goal_binding')!=goal_binding(request)):
        return None
    return copy.deepcopy(evidence)


def load_catalog_map(catalog, *, directory=None):
    from perception.environment.storage import StaticEnvironmentMap
    value = catalog.value
    asset = StaticEnvironmentMap(ROOT/value['map_asset'] if directory is None else directory,
                                 navigation_annotations=catalog.annotations())
    if (asset.manifest['snapshot_id'] != value.get('map_snapshot_id') or
            asset.manifest['frame_id'] != value['frame_id'] or
            value.get('localization_map_id',asset.snapshot.metadata.get('map_id')) != asset.snapshot.metadata.get('map_id')):
        raise ValueError('Semantic catalog and actual map asset do not match')
    return asset


def observation_rgb_key(record):
    """Identify the source RGB even when an observation has several views."""
    rgb = record.get('rgb') or {}
    stamp = rgb.get('stamp_s')
    if rgb.get('status')!='valid' or type(stamp) not in (int, float) or not math.isfinite(stamp):
        return None
    return (record['clock_domain'], tuple(record['local_identity']),
            source_stamp_ns(stamp, rgb.get('stamp_ns')))


def _frame_key(record):
    return (record.get('observation_id'), record.get('scene_binding'), observation_rgb_key(record))


class SavedShadowNavigation:
    scope = 'shadow'

    def __init__(self, catalog, observation):
        from navigation.planning.footprint import VehicleGeometry
        self.catalog = catalog
        self.geometry = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
        self.current = observation.verify()
        self.clock_domain = observation.record['clock_domain']
        self.handle = self.request = None
        self.status = 'idle'
        self.cancelled = False
        self.t = 0.
        self.first_stamp = observation.record['stamp_s']
        self._planner = self._asset = None

    def update(self, observation):
        observation.verify()
        if observation.record['clock_domain'] != self.clock_domain or observation.record['stamp_s'] < self.current.record['stamp_s']:
            raise ValueError('Shadow observations must advance in one source clock')
        self.current = observation
        self.t = observation.record['stamp_s']-self.first_stamp

    def snapshot(self):
        r = self.current.record
        return dict(scope=self.scope, source='recorded_localization_observer', clock_domain=self.clock_domain,
                    stamp_s=r['stamp_s'], session_id=r['local_identity'][0], generation=r['local_identity'][1], frame_id=r['local_identity'][2],
                    pose=r['base_pose'], velocity=None, stopped=None, handle=copy.deepcopy(self.handle), status=self.status,
                    collision=False, goal=None, execution_feedback=False, executable=False, proposal_cancelled=self.cancelled,
                    localization_status=r['localization_status'], **mapped_origin(self.catalog, r))

    def tick(self):
        return self.snapshot()

    def observe(self):
        return self.current

    def spatial_inputs(self):
        from perception.environment.scene import SceneInputs
        from perception.environment.storage import StaticEnvironmentMap
        from navigation.planning.footprint import VehicleGeometry
        r = self.current.record
        origin = mapped_origin(self.catalog, r)
        if (r['localization_status'] != 'valid' or origin['origin_pose'] is None or
                origin['origin_frame_id'] != self.catalog.value['frame_id']):
            raise ValueError('Current observation is not localized in the annotated map')
        if self._asset is None:
            self._asset = load_catalog_map(self.catalog)
        scene = SceneInputs.load(ROOT/self.current.record['source']['scene'])
        if scene.record['input_fingerprint'] != self.current.record['scene_binding']:
            raise ValueError('Saved scene changed after observation binding')
        return dict(global_grid=self._asset.snapshot,scene=scene,pose=r['map_base_pose'],local_pose=r['base_pose'],
                    stamp_s=r['stamp_s'],clock_domain=r['clock_domain'],
                    geometry=self.geometry,sensors=copy.deepcopy(r.get('sensors')))

    def reference_grid(self):
        from perception.environment.storage import StaticEnvironmentMap
        if self._asset is None:
            self._asset = load_catalog_map(self.catalog)
        return self._asset.snapshot

    def check_route(self, place):
        from perception.environment.scene import SceneInputs
        from perception.environment.storage import StaticEnvironmentMap
        from navigation.planning.inputs import load_config
        from navigation.planning.reference import planning_config
        from navigation.planning.planner import Planner
        from navigation.planning.footprint import VehicleGeometry
        from navigation.planning.motion import load_motion_config
        if self._planner is None:
            self._asset = load_catalog_map(self.catalog)
            cfg = planning_config(load_config())
            motion = load_motion_config()
            cfg['motion_constraints'] = {k: motion[k] for k in ('min_turning_radius_m', 'allow_reverse', 'allow_in_place_rotation')}
            geom = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
            self._planner = Planner(cfg, geom, require_actual_start=True)
        path = ROOT/self.current.record['source']['scene']
        scene = SceneInputs.load(path)
        if scene.record['input_fingerprint'] != self.current.record['scene_binding']:
            raise ValueError('Saved geometry changed after Agent observation binding')
        request = dict(request_id='agent-shadow-preflight', task_id='agent-shadow-preflight', goal_revision=1,
                       mode='mapped', frame_id=self.catalog.value['frame_id'], target_reference='base_link', goal=place['pose'], goal_behavior='stop',
                       map_snapshot_id=self._asset.manifest['snapshot_id'])
        result = self._planner.query(request, scene=scene, asset=self._asset, now=self.current.record['stamp_s'], clock_domain=self.clock_domain)
        return dict(status=result['status'], reason=result['reason'], reference_found=result['status'] == 'path_found',
                    scope=self.scope, executable=False, execution_feedback=False, source_observation=self.current.record['observation_id'])

    def start(self, place, handle):
        if self.status == 'proposed':
            raise RuntimeError('A shadow proposal is already active')
        self.handle = copy.deepcopy(handle)
        self.request = dict(place_id=place['place_id'], pose=place['pose'])
        self.status, self.cancelled = 'proposed', False
        return dict(accepted=True, handle=copy.deepcopy(handle), executable=False, execution_feedback=False)

    def cancel(self, handle):
        if handle != self.handle:
            raise ValueError('Cancellation belongs to another shadow proposal')
        self.status, self.cancelled = 'stopped', True


class RuntimeShadowNavigation:
    scope = 'shadow'

    def __init__(self, catalog, client):
        from navigation.planning.footprint import VehicleGeometry
        self.catalog, self.client = catalog, client
        self.geometry = VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json')
        self.value = client.snapshot(with_image=False)
        self._image_observation = None
        vehicle = self.value.get('vehicle_context')
        if vehicle is not None:
            self.geometry = VehicleGeometry(vehicle['footprint'],vehicle['base_from_lidar'],
                source=vehicle['source'],assumed=vehicle['assumed'])
            self.motion = copy.deepcopy(vehicle['motion'])
        if self.value['control'] is None:
            raise ValueError('Runtime has not published control state')
        self.clock_domain = self.value['control']['clock_domain']
        self.handle = self.request = None
        self.follow_active = False
        self.cancel_requested = False
        self.connected = True
        self.t = 0.
        self.first_stamp = self.value['control']['stamp_s']
        self._asset = None

    def spatial_inputs(self):
        from perception.environment.storage import StaticEnvironmentMap
        from navigation.planning.footprint import VehicleGeometry
        if self._asset is None:
            self._asset = load_catalog_map(self.catalog)
        record = self.value.get('observation') or {}
        state = self.snapshot()
        origin = mapped_origin(self.catalog, record)
        current = (not state['input_stale'] and state['localization_status']=='valid' and
                   origin['origin_pose'] is not None and origin['origin_frame_id']==self.catalog.value['frame_id'] and
                   record.get('local_identity')==[state.get(k) for k in ('session_id','generation','frame_id')] and
                   record.get('clock_domain')==self.clock_domain and
                   record.get('map_binding')==state.get('map_binding') and
                   0 <= state['stamp_s']-record.get('stamp_s',-float('inf')) <= 3.)
        grid = self.client.local_grid(self.value) if current else None
        return dict(global_grid=self._asset.snapshot,scene=None,local_grid=grid,
                    local_grid_observation_id=record.get('observation_id') if grid is not None else None,
                    local_grid_map_from_local=(record.get('local_grid') or {}).get('map_from_local') if grid is not None else None,
                    pose=record.get('map_base_pose') if current else None,local_pose=record.get('base_pose') if current else None,
                    stamp_s=record.get('stamp_s') if current else None,clock_domain=self.clock_domain,
                    geometry=getattr(self,'geometry',None) or VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json'),
                    sensors=copy.deepcopy(record.get('sensors')) if current else None)

    def reference_grid(self):
        return SavedShadowNavigation.reference_grid(self)

    def snapshot(self):
        return self._snapshot(self.value)

    def _snapshot(self, value):
        p = value['control']
        f = p['feedback']
        loc = p.get('localization') or {}
        identity = f.get('local_identity') or [loc.get(k) for k in ('session_id', 'generation', 'frame_id')]
        active = p.get('request')
        matches = self.handle is not None and active is not None and active.get('request_id') == self.handle['call_id']
        cancellation = p.get('cancelled_request') or {}
        follow = value.get('follow') or {}
        follow_match = self.follow_active and self.handle is not None and follow.get('handle') == self.handle
        cancelled = (self.cancel_requested and active is None and self.handle is not None and
                     (cancellation.get('task_id') == 'follow:'+self.handle['call_id'] if self.follow_active else
                      cancellation.get('task_id') == self.handle['task_id'] and
                      cancellation.get('goal_revision') == self.handle['goal_revision']))
        if (follow_match and follow.get('status')=='failed' and active is None and
                isinstance(follow.get('cancelled_request'),dict) and cancellation==follow['cancelled_request']):
            cancelled=True
        observation = value.get('observation') or {}
        spatial = mapped_origin(self.catalog, observation)
        if 'map_binding' in p:
            spatial['map_binding'] = copy.deepcopy(p['map_binding'])
        if (observation.get('local_identity') != identity or observation.get('clock_domain') != self.clock_domain or
                observation.get('map_binding') != spatial['map_binding'] or
                not 0 <= p['stamp_s']-observation.get('stamp_s', -float('inf')) <= .5):
            spatial['origin_pose'] = None
        return dict(scope=self.scope, source='runtime_shadow_observer', clock_domain=p['clock_domain'], stamp_s=p['stamp_s'],
                    session_id=identity[0], generation=identity[1], frame_id=identity[2], pose=f.get('pose'), velocity=f.get('velocity'),
                    stopped=None, handle=copy.deepcopy(self.handle) if matches or cancelled or follow_match else None,
                    status=('failed' if follow_match and follow.get('status')=='failed' else
                            'preempted' if follow_match and follow.get('status')=='preempted' else
                            'stopped' if cancelled else 'proposed' if matches else 'pending'), collision=False, goal=None,
                    follow_status=copy.deepcopy(follow) if follow_match else None,
                    execution_feedback=False, executable=False, proposal_cancelled=cancelled,
                    localization_status='valid' if loc.get('health') == 0 else 'invalid',
                    environment_fault=p.get('environment_fault'),environment_epoch=p.get('environment_epoch',0),
                    input_stale=not self.connected or value.get('received_age_s') is None or value['received_age_s'] > 1., **spatial)

    def _snapshot_options(self, *, with_image):
        options = dict(with_image=with_image)
        if self.follow_active and self.handle is not None:
            options['follow_call_id'] = self.handle['call_id']
        if getattr(self, 'want_follow_candidates', False):
            options['with_follow_candidates'] = True
        return options

    def tick(self):
        try:
            value = self.client.snapshot(**self._snapshot_options(with_image=False))
            if not isinstance(value.get('control'), dict):
                raise ValueError('Runtime has no control state')
        except (OSError, EOFError, ValueError):
            self.connected = False
            raise
        self.value, self.connected = value, True
        self.t = self.value['control']['stamp_s']-self.first_stamp
        return self.snapshot()

    @staticmethod
    def _selected_record(value):
        record = value.get('observation') or {}
        frame = value.get('follow_frame')
        if frame is None:
            return record
        pending = {key:frame.get(key) for key in
                   ('observation_id', 'stamp_s', 'scene_binding', 'clock_domain', 'local_identity', 'map_binding')}
        pending['rgb'] = dict(status='valid', stamp_s=frame['rgb_stamp_s'], stamp_ns=frame.get('rgb_stamp_ns'))
        if 'follow_candidates' in record and _frame_key(record)==_frame_key(pending):
            return record
        return pending

    def observation_record(self):
        return self._selected_record(self.value)

    def observation_key(self):
        record = self.observation_record()
        return fingerprint(_frame_key(record)) if record else None

    def _current_image_record(self, record, state):
        stamp = record.get('stamp_s')
        rgb_stamp = (record.get('rgb') or {}).get('stamp_s')
        return (not state['input_stale'] and state['localization_status']=='valid' and
                record.get('clock_domain')==state['clock_domain'] and
                record.get('local_identity')==[state[k] for k in ('session_id', 'generation', 'frame_id')] and
                (state.get('map_binding') is None or record.get('map_binding')==state['map_binding']) and
                type(stamp) in (int, float) and 0<=state['stamp_s']-stamp<=3 and
                observation_rgb_key(record) is not None and 0<=state['stamp_s']-rgb_stamp<=3)

    def observe(self):
        state = self.snapshot()
        record = self.observation_record()
        if state['input_stale'] or not record:
            return None
        if (record.get('rgb') or {}).get('status')!='valid':
            return self.client.observation(self.value)
        if not self._current_image_record(record, state):
            return None
        requested_rgb_ns = observation_rgb_key(record)[2]
        cached = self._image_observation
        preview = getattr(self, 'want_follow_candidates', False)
        owner = (self.handle, self.follow_active, preview)
        follow_frame = self.value.get('follow_frame') is not None or 'follow_candidates' in record
        following = (state.get('follow_status') or {}).get('status') in ('starting','running','searching')
        if (cached is not None and self._image_owner==owner and
                _frame_key(cached.record)==_frame_key(record) and
                ('follow_candidates' in cached.record)==follow_frame and
                self._current_image_record(cached.record, state)):
            self.value = dict(self.value, observation=cached.record)
            return cached
        try:
            value = self.client.snapshot(**self._snapshot_options(with_image=True))
            if not isinstance(value.get('control'), dict):
                return None
            actual = self._snapshot(value)
            context_fields = ('clock_domain', 'session_id', 'generation', 'frame_id', 'map_binding', 'environment_epoch')
            if (any(actual[key]!=state[key] for key in context_fields) or actual['stamp_s']<state['stamp_s'] or
                    following and (actual.get('follow_status') or {}).get('handle')!=self.handle):
                return None
            selected = self._selected_record(value)
            record = value.get('observation') or {}
            candidate_view = 'follow_candidates' in record
            if (not self._current_image_record(record, actual) or _frame_key(record)!=_frame_key(selected) or
                    (follow_frame or value.get('follow_frame') is not None) and not candidate_view or
                    candidate_view==follow_frame and observation_rgb_key(record)[2]<requested_rgb_ns):
                return None
            observation = self.client.observation(value)
        except (OSError, EOFError, ValueError):
            return None
        if observation is not None:
            self._image_observation = observation
            self._image_owner = copy.deepcopy(owner)
            self.value = dict(value, image_base64=None)
            self.t = actual['stamp_s']-self.first_stamp
        return observation

    def check_route(self, place):
        state = self.snapshot()
        if state.get('input_stale') or state.get('localization_status') != 'valid':
            return dict(reference_found=False, proposal_allowed=False, status='unavailable',
                        reason='runtime_localization_invalid_or_stale', executable=False, execution_feedback=False)
        binding = state.get('map_binding') or {}
        map_id = binding.get('map_id')
        asset = self.catalog.value.get('map_snapshot_id', '')
        source_id = self.catalog.value.get('localization_map_id')
        if not (binding.get('valid') and map_id and (source_id == map_id if source_id else
                asset == map_id or asset.startswith(map_id+':scan_fusion:'))):
            return dict(reference_found=False, proposal_allowed=False, status='unavailable',
                        reason='runtime_map_binding_unavailable', executable=False, execution_feedback=False)
        return dict(reference_found=False, proposal_allowed=True, status='runtime_geometry_check_pending',
                    reason=None, executable=False, execution_feedback=False)

    def start(self, place, handle):
        request = dict(request_id=handle['call_id'], task_id=handle['task_id'], goal_revision=handle['goal_revision'],
                       mode='mapped', frame_id=self.catalog.value['frame_id'], target_reference='base_link', goal=place['pose'], goal_behavior='stop',
                       map_snapshot_id=self.catalog.value['map_snapshot_id'])
        if 'map_keepout_circle' in place:
            request['map_keepout_circle'] = copy.deepcopy(place['map_keepout_circle'])
        if 'search_bounds' in place:
            request['search_bounds']=copy.deepcopy(place['search_bounds'])
            request['valid_until_s']=place['valid_until_s']
        self.handle, self.request, self.cancel_requested = copy.deepcopy(handle), request, False
        self.follow_active=False
        ack = self.client.exchange('goal', request)
        return dict(accepted=ack.get('accepted') is True, handle=copy.deepcopy(handle), executable=False, execution_feedback=False)

    def cancel(self, handle):
        if self.handle != handle:
            raise ValueError('Wrong runtime shadow call')
        # Persist intent before I/O. A lost ACK can be reconciled by a later
        # read-only query, without automatically resending the goal or cancel.
        self.cancel_requested = True
        ack = (self.client.exchange('follow_cancel',dict(call_id=handle['call_id'])) if self.follow_active else
               self.client.exchange('cancel', {k: handle[k] for k in ('task_id', 'goal_revision')}))
        if ack.get('accepted') is not True:
            self.cancel_requested = False
            raise ValueError('Runtime cancellation was not accepted')

    def start_follow(self, intent, handle):
        from ..tools.navigation import IntentRejected
        previous=(self.handle,self.request,self.cancel_requested,getattr(self,'follow_active',False))
        self.handle,self.request,self.cancel_requested,self.follow_active=copy.deepcopy(handle),None,False,True
        try:
            ack=self.client.exchange('follow_start',dict(handle=handle,intent=intent))
            if ack.get('accepted') is not True:raise IntentRejected(ack.get('error','follow_start_rejected'))
        except IntentRejected:
            self.handle,self.request,self.cancel_requested,self.follow_active=previous
            raise
        return dict(accepted=True,handle=copy.deepcopy(handle),execution_feedback=False)

    def finish_follow(self, handle):
        if self.handle!=handle or not self.follow_active:raise ValueError('Wrong follow call')
        self.cancel_requested=True
        ack=self.client.exchange('follow_finish',dict(call_id=handle['call_id']))
        if ack.get('accepted') is not True:raise ValueError('Follow finish was not accepted')


class RuntimeLiveNavigation(RuntimeShadowNavigation):
    """Live intents with localization-derived arrival, never hardware certification."""
    scope = 'live'

    def __init__(self, catalog, client):
        from navigation.control.tracking import load_config
        self.tracking = load_config()
        super().__init__(catalog, client)

    def start(self, place, handle):
        from ..tools.navigation import IntentRejected
        previous=(self.handle,self.request,self.cancel_requested,getattr(self,'follow_active',False))
        try:
            return super().start(place,handle)
        except IntentRejected:
            self.handle,self.request,self.cancel_requested,self.follow_active=previous
            raise

    def hold(self, handle):
        if self.handle != handle:
            raise ValueError('Wrong live navigation call')
        # A hold clears the goal without arming anything. The gateway verifies
        # the current owned arrival before forwarding it to independent safety.
        self.cancel_requested = True
        ack = self.client.exchange('hold', {k:handle[k] for k in ('task_id','goal_revision')})
        if ack.get('accepted') is not True:
            raise ValueError('Runtime hold was not accepted')

    def retarget(self,place,old_handle,new_handle):
        if self.handle!=old_handle or self.cancel_requested or self.follow_active:
            raise ValueError('Live target no longer active')
        request=copy.deepcopy(self.request)
        request.update(goal_revision=new_handle['goal_revision'],goal=copy.deepcopy(place['pose']))
        ack=self.client.exchange('retarget',dict(previous_revision=old_handle['goal_revision'],goal=request))
        if ack.get('accepted') is not True:raise ValueError('Retarget was not accepted')
        self.handle,self.request=copy.deepcopy(new_handle),request
        return dict(accepted=True,handle=copy.deepcopy(new_handle),execution_feedback=False)

    def snapshot(self):
        result = apply_runtime_mapped_pose(super().snapshot(),self.value['control'],self.catalog)
        control = self.value['control']
        follow=result.get('follow_status') or {}
        chassis = self.value.get('chassis') or {}
        owner = self.value.get('agent_owner') or {}
        active = control.get('request') or {}
        match = (self.handle is not None and active.get('request_id') == self.handle['call_id'] and
                 owner.get('request_id') == self.handle['call_id'] and
                 owner.get('arm_revision') == chassis.get('arm_revision'))
        enabled = chassis.get('state') == 'armed' and chassis.get('status_fresh') is True
        evidence = control.get('goal_observation') or {}
        arrival = (match and enabled and evidence.get('status') == 'criteria_held' and
                   evidence.get('task_id') == self.handle['task_id'] and
                   evidence.get('goal_revision') == self.handle['goal_revision'] and
                   evidence.get('source') == 'causal_algorithm_poses_not_chassis_feedback' and
                   result.get('origin_pose') is not None and not result['input_stale'] and
                   result['localization_status'] == 'valid')
        failed_follow=bool(self.follow_active and follow.get('status')=='failed')
        cancellation=control.get('cancelled_request') or {}
        cancelled = bool(result.get('proposal_cancelled') or failed_follow and not active and
                         cancellation==follow.get('cancelled_request'))
        idle = self.handle is None and not active and not owner
        feedback = (control.get('idle_feedback') if cancelled or idle else control.get('feedback')) or {}
        velocity = feedback.get('velocity')
        settings = self.tracking
        stopped = (isinstance(velocity, list) and len(velocity) == 2 and
                   all(isinstance(v, (int, float)) and math.isfinite(v) for v in velocity) and
                   abs(velocity[0]) <= settings['stopped_linear_mps'] and
                   abs(velocity[1]) <= settings['stopped_angular_rps'] and
                   feedback.get('status') == 'valid' and
                   feedback.get('source') == 'causal_algorithm_poses_not_chassis_feedback' and
                   feedback.get('local_identity')==[result.get(k) for k in ('session_id','generation','frame_id')] and
                   result.get('localization_status')=='valid' and
                   0 <= control['stamp_s']-feedback.get('stamp_s', -float('inf')) <= settings['feedback_max_age_s'])
        if failed_follow:
            status='failed' if cancelled and stopped else 'stopping'
        elif self.handle is not None and not self.cancel_requested and not match and owner.get('request_id') != self.handle['call_id']:
            status = 'preempted'
        elif (self.handle is not None and not self.cancel_requested and owner.get('request_id') == self.handle['call_id'] and
              owner.get('arm_revision') != chassis.get('arm_revision')):
            status = 'interrupted'
        elif self.handle is not None and not self.cancel_requested and not enabled:
            status = 'interrupted'
        else:
            status = 'arrived' if arrival else 'stopped' if cancelled and stopped else 'running' if match else 'pending'
        planning = None
        if match and enabled and not self.cancel_requested and not self.follow_active:
            reference=bound_planning_observation(control.get('reference_observation'),self.request,result)
            local=bound_planning_observation(control.get('control_observation'),self.request,result)
            # A complete negative reference result can explain why DWA could
            # not issue a proposal. A cached reference cannot clear a DWA stall.
            planning=reference if reference and reference['status']=='blocked' else local
        result.update(scope='live',source='runtime_live_observer',status=status,
                      handle=copy.deepcopy(self.handle) if match or cancelled or follow or status in ('preempted','interrupted') else None,
                      pose=copy.deepcopy(feedback.get('pose')),velocity=copy.deepcopy(feedback.get('velocity')),
                      stopped=bool(stopped),arrival_observation=bool(arrival),
                      stop_observation=bool(cancelled and stopped),stop_measurement_stamp_s=feedback.get('stamp_s'),physical_acceptance=False,
                      idle_stop_observation=bool(idle and stopped and enabled and not result['input_stale']),
                      stop_reason=follow.get('reason') if failed_follow else None,
                      chassis_state=chassis.get('state'),chassis_status_fresh=chassis.get('status_fresh') is True,
                      planning_status=control.get('planning_status'),planning_reason=control.get('planning_reason'),
                      planning_observation=planning,
                      goal=dict(task_id=evidence.get('task_id'),goal_revision=evidence.get('goal_revision')) if arrival else None)
        return result
