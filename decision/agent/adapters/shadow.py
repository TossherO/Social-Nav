"""Nonexecuting adapters for causal saved observations and runtime shadow IPC."""
import copy
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


def load_catalog_map(catalog):
    from perception.environment.storage import StaticEnvironmentMap
    asset = StaticEnvironmentMap(ROOT/catalog.value['map_asset'])
    value = catalog.value
    if (asset.manifest['snapshot_id'] != value.get('map_snapshot_id') or
            asset.manifest['frame_id'] != value['frame_id'] or
            value.get('localization_map_id',asset.snapshot.metadata.get('map_id')) != asset.snapshot.metadata.get('map_id')):
        raise ValueError('Semantic catalog and actual map asset do not match')
    return asset


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
        self.value = client.snapshot(with_image=True)
        vehicle = self.value.get('vehicle_context')
        if vehicle is not None:
            self.geometry = VehicleGeometry(vehicle['footprint'],vehicle['base_from_lidar'],
                source=vehicle['source'],assumed=vehicle['assumed'])
            self.motion = copy.deepcopy(vehicle['motion'])
        if self.value['control'] is None:
            raise ValueError('Runtime has not published control state')
        self.clock_domain = self.value['control']['clock_domain']
        self.handle = self.request = None
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
        observation = self.client.observation(self.value) if current else None
        grid = None if observation is None else observation.local_grid()
        return dict(global_grid=self._asset.snapshot,scene=None,local_grid=grid,
                    local_grid_observation_id=record.get('observation_id') if grid is not None else None,
                    local_grid_map_from_local=(record.get('local_grid') or {}).get('map_from_local') if grid is not None else None,
                    pose=record.get('map_base_pose') if current else None,
                    stamp_s=record.get('stamp_s') if current else None,clock_domain=self.clock_domain,
                    geometry=getattr(self,'geometry',None) or VehicleGeometry.historical(ROOT/'configs/wheelchair_geometry_reference.json'),
                    sensors=copy.deepcopy(record.get('sensors')) if current else None)

    def reference_grid(self):
        return SavedShadowNavigation.reference_grid(self)

    def snapshot(self):
        p = self.value['control']
        f = p['feedback']
        loc = p.get('localization') or {}
        identity = f.get('local_identity') or [loc.get(k) for k in ('session_id', 'generation', 'frame_id')]
        active = p.get('request')
        matches = self.handle is not None and active is not None and active.get('request_id') == self.handle['call_id']
        cancellation = p.get('cancelled_request') or {}
        cancelled = (self.cancel_requested and active is None and self.handle is not None and
                     cancellation.get('task_id') == self.handle['task_id'] and
                     cancellation.get('goal_revision') == self.handle['goal_revision'])
        observation = self.value.get('observation') or {}
        spatial = mapped_origin(self.catalog, observation)
        if 'map_binding' in p:
            spatial['map_binding'] = copy.deepcopy(p['map_binding'])
        if (observation.get('local_identity') != identity or observation.get('clock_domain') != self.clock_domain or
                observation.get('map_binding') != spatial['map_binding'] or
                not 0 <= p['stamp_s']-observation.get('stamp_s', -float('inf')) <= .5):
            spatial['origin_pose'] = None
        return dict(scope=self.scope, source='runtime_shadow_observer', clock_domain=p['clock_domain'], stamp_s=p['stamp_s'],
                    session_id=identity[0], generation=identity[1], frame_id=identity[2], pose=f.get('pose'), velocity=f.get('velocity'),
                    stopped=None, handle=copy.deepcopy(self.handle) if matches or cancelled else None,
                    status='stopped' if cancelled else 'proposed' if matches else 'pending', collision=False, goal=None,
                    execution_feedback=False, executable=False, proposal_cancelled=cancelled,
                    localization_status='valid' if loc.get('health') == 0 else 'invalid',
                    input_stale=not self.connected or self.value.get('received_age_s') is None or self.value['received_age_s'] > 1., **spatial)

    def tick(self):
        try:
            value = self.client.snapshot(with_image=True)
            if not isinstance(value.get('control'), dict):
                raise ValueError('Runtime has no control state')
        except (OSError, EOFError, ValueError):
            self.connected = False
            raise
        self.value, self.connected = value, True
        self.t = self.value['control']['stamp_s']-self.first_stamp
        return self.snapshot()

    def observe(self):
        if self.snapshot()['input_stale']:
            return None
        return self.client.observation(self.value)

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
        self.handle, self.request, self.cancel_requested = copy.deepcopy(handle), request, False
        ack = self.client.exchange('goal', request)
        return dict(accepted=ack.get('accepted') is True, handle=copy.deepcopy(handle), executable=False, execution_feedback=False)

    def cancel(self, handle):
        if self.handle != handle:
            raise ValueError('Wrong runtime shadow call')
        # Persist intent before I/O. A lost ACK can be reconciled by a later
        # read-only query, without automatically resending the goal or cancel.
        self.cancel_requested = True
        ack = self.client.exchange('cancel', {k: handle[k] for k in ('task_id', 'goal_revision')})
        if ack.get('accepted') is not True:
            self.cancel_requested = False
            raise ValueError('Runtime cancellation was not accepted')
