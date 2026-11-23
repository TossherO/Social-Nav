"""Model-facing references for observations, maps, and dialogue."""
import copy
import hashlib
import json
import re
from threading import Lock

from ..tools.framework import unique_fields, validate_call
from .store import encode


ACTION_REFERENCES = {
    'view_id': 'view', 'observation_id': 'obs', 'keyframe_id': 'key', 'reference_id': 'ref',
}
REFERENCE_SOURCES = {
    'view_id': 'visual_inputs.maps', 'observation_id': 'visual_inputs.rgb',
    'keyframe_id': 'keyframes', 'reference_id': 'follow_reference',
}
REFERENCE_FIELDS = dict(ACTION_REFERENCES, reference_map_id='map')
REFERENCE_PREFIXES = dict(view='v', obs='o', key='k', ref='r', map='m', track='p')
LIST_REFERENCES = {'visible_keyframe_ids': 'key', 'historical_keyframe_ids': 'key'}
AUDIT_FIELDS = {
    'task_id', 'call_id', 'session_id', 'local_identity', 'association_session',
    'map_id', 'snapshot_id', 'sha256', 'evidence_id', 'selection_id', 'binding',
}
AUDIT_SUFFIXES = ('_sha256', '_fingerprint', '_digest', '_session_id', '_task_id', '_call_id', '_map_id')
REJECTED_ARGUMENTS = {'invalid_tool_arguments', 'invalid_tool_reference'}
CURRENT_GOAL_TOOLS = frozenset({
    'pause_map_localization', 'navigate', 'navigate_route', 'select_follow_target',
    'start_local_task', 'set_navigation_monitor', 'record_monitor_review', 'complete_subtask',
})
CURRENT_VIEW_TOOLS = frozenset({'check_candidate', 'navigate', 'navigate_route'})
CURRENT_RGB_TOOLS = frozenset({'select_follow_target', 'save_keyframe', 'record_monitor_review'})
STAGE_TOOLS = {
    'point': ('navigate',),
    'region': ('navigate',),
    'reference': ('navigate',),
    'route': ('navigate_route',),
    'sequence': ('navigate_route',),
    'follow': ('select_follow_target', 'start_local_task'),
    'seating': ('start_local_task',),
    'observation': (),
    'elevator': ('pause_map_localization',),
}
STAGE_SPECIFIC_TOOLS = frozenset(
    tool for names in STAGE_TOOLS.values() for tool in names
) | {'record_elevator_observation', 'restart_map_localization'}
OPAQUE_TEXT = re.compile(
    r'(?:scene:)?[0-9a-f]{24,}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',
    re.IGNORECASE,
)


def reference_kind(key):
    if key in REFERENCE_FIELDS:
        return REFERENCE_FIELDS[key]
    for suffix, kind in ACTION_REFERENCES.items():
        if key.endswith('_'+suffix):
            return kind
    return None


def object_from_text(text):
    try:
        value = json.loads(text, object_pairs_hook=unique_fields)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class ModelReferenceNames:
    """Reserve short names for the lifetime of the Agent session."""

    def __init__(self):
        self.names = {kind: {} for kind in REFERENCE_PREFIXES}
        self.originals = {}
        self.lock = Lock()

    def name(self, kind, original):
        with self.lock:
            names = self.names[kind]
            if original not in names:
                token = REFERENCE_PREFIXES[kind]+str(len(names)+1)
                names[original] = token
                self.originals[token] = (kind, original)
            return names[original]

    def lookup(self, token):
        with self.lock:
            return self.originals.get(token)


class ModelProjection:
    """Stable display names and a separate allowlist of current tool inputs."""

    def __init__(self, names=None):
        self.names = names if names is not None else ModelReferenceNames()
        self.references = {}
        self.allowed = set()
        self.goal_id = None
        self.view = None
        self.rgb = None
        self.initial_pose = None
        self.initial_point_id = None
        self.coordinate_point = False
        self.orientation_source = None
        self.navigation_visual_goal = None
        self.default_visual_review = False
        self.visual_priority = None
        self.registered_round_table = False
        self.hidden_tools = {}

    def reference(self, kind, value):
        if not isinstance(value, str) or not value:
            return value
        token = self.names.name(kind, value)
        self.references[token] = (kind, value)
        return token

    def allow(self, state):
        inputs = state['visual_inputs']
        self.view = self.current_reference(inputs.get('maps'), 'view_id')
        self.rgb = self.current_reference(inputs.get('rgb'), 'observation_id')
        sources = [
            ('view', self.view), ('obs', self.rgb),
            ('ref', (state.get('follow_reference') or {}).get('reference_id')),
        ]
        sources.extend(('key', item['keyframe_id']) for item in state.get('keyframes', ()))
        for kind, original in sources:
            if original:
                self.allowed.add(self.reference(kind, original))
        self.goal_id = state.get('next_goal_id')
        selected = state.get('selected_goal') or {}
        if self.goal_id and selected.get('goal_id') == self.goal_id:
            self.configure_goal(state, selected)

    @staticmethod
    def current_reference(record, key):
        return record.get(key) if record and record.get('available') is not False else None

    def configure_goal(self, state, selected):
        completion = state.get('completion_evidence') or {}
        requirement = completion.get('requirement') or {}
        sources = selected.get('allowed_orientation_sources') or []
        if len(sources) == 1:
            self.orientation_source = sources[0]
        else:
            semantic_heading = selected.get('heading_required') or any(requirement.get(field) for field in (
                'facing', 'semantic_conditions', 'visual_preferences'))
            default = 'semantic' if semantic_heading else 'autonomous'
            if default in sources:
                self.orientation_source = default
        if self.view is not None and selected.get('selection') is None:
            initial = selected.get('initial_pose')
            self.initial_pose = tuple(initial) if initial else None
            self.initial_point_id = selected.get('initial_point_id')

        monitor = state.get('task_monitor') or {}
        conditional = (monitor.get('conditional_review') or {}).get('phase') == 'reviewing'
        adjustment = selected.get('visual_adjustment')
        if requirement.get('goal_id') == self.goal_id:
            self.configure_stage_tools(state, requirement, conditional, adjustment)
        if adjustment is None:
            return
        arrived = completion.get('arrival_stamp_s') is not None
        self.default_visual_review = arrived and not monitor.get('continuous_region_id')
        if self.default_visual_review and self.rgb and requirement.get('goal_id') == self.goal_id:
            priorities = [priority for priority, field in (
                ('required', 'semantic_conditions'), ('preferred', 'visual_preferences')
            ) if requirement.get(field)]
            if len(priorities) == 1:
                self.visual_priority = priorities[0]
        if not arrived and not conditional:
            self.hidden_tools['record_monitor_review'] = 'post_arrival_visual_review_requires_current_stop'
        elif arrived and not conditional and (self.rgb is None or 'current_rgb' in completion.get('missing', ())):
            self.hidden_tools['record_monitor_review'] = 'current_post_stop_rgb_required'
        review = adjustment.get('review') or {}
        if arrived or adjustment.get('requires_review'):
            if adjustment.get('remaining') == 0:
                self.hidden_tools['navigate'] = 'visual_adjustment_limit_reached'
            elif not arrived:
                self.hidden_tools['navigate'] = 'visual_adjustment_requires_current_stop'
            elif review.get('result') not in ('not_satisfied', 'partly_satisfied'):
                self.hidden_tools['navigate'] = 'post_arrival_visual_review_required'

    def configure_stage_tools(self, state, requirement, conditional, adjustment):
        stage = requirement.get('stage_type', 'navigation')
        stage = requirement.get('kind') if stage == 'navigation' else stage
        tools = STAGE_TOOLS.get(stage)
        if tools is None:
            return
        for tool in STAGE_SPECIFIC_TOOLS.difference(tools):
            self.hidden_tools[tool] = 'tool_unavailable_for_current_stage'
        self.navigation_visual_goal = bool(requirement.get('semantic_conditions') or
                                           requirement.get('visual_preferences'))
        self.coordinate_point = stage == 'point' and not requirement.get('target_id') and 'position' in requirement
        if not state.get('keyframes'):
            self.hidden_tools['read_keyframe'] = 'no_saved_keyframes'
        if adjustment is None and not conditional:
            self.hidden_tools['record_monitor_review'] = 'no_current_monitor_review'
        if stage == 'seating' and requirement.get('table_id'):
            self.registered_round_table = True
            self.hidden_tools['check_candidate'] = 'round_table_candidates_selected_by_t3'
            start = state.get('seating_start')
            if start is None:
                self.hidden_tools['start_local_task'] = 'seating_target_already_selected'
            else:
                self.hidden_tools['set_navigation_monitor'] = 'start_t3_before_navigation_monitor'
                self.hidden_tools['complete_subtask'] = 'start_t3_before_completion'
                if self.rgb is None or not start.get('stopped'):
                    self.hidden_tools['start_local_task'] = 'seating_requires_stopped_vehicle_and_current_rgb'

    def available_tools(self, names):
        return tuple(name for name in names if name not in self.hidden_tools)

    def tools(self, tools):
        """Advertise decisions the model can make given the sources in this request."""
        tools = copy.deepcopy(tools)
        for tool in tools:
            function = tool['function']
            name = function['name']
            schema = function['parameters']
            hidden = set()
            if self.goal_id and name in CURRENT_GOAL_TOOLS:
                hidden.add('goal_id')
            if self.view and name in CURRENT_VIEW_TOOLS:
                hidden.add('view_id')
            if self.rgb and name in CURRENT_RGB_TOOLS:
                hidden.add('observation_id')
            if self.orientation_source and name in ('navigate', 'navigate_route'):
                hidden.add('orientation_source')
            if name == 'navigate' and self.navigation_visual_goal is False:
                hidden.add('point_ids')
            if name == 'navigate' and self.coordinate_point:
                hidden.update(('point_id', 'yaw'))
            if self.visual_priority and name == 'record_monitor_review':
                hidden.add('visual_priority')
            if self.registered_round_table and name == 'start_local_task':
                hidden.update(('task_type', 'target_description', 'table_id', 'table_roi', 'reference_id'))
                function['description'] = '立即用已登记的圆桌启动 T3。局部 VLM 负责从桌周挑选停靠位姿。'
            for key in hidden:
                schema['properties'].pop(key, None)
            schema['required'] = [key for key in schema['required'] if key not in hidden]
            if name == 'navigate':
                if self.initial_pose is not None:
                    function['description'] += '首次可用 {} 采用当前阶段的 initial_pose。'
                if self.coordinate_point:
                    function['description'] += '自行提交位置时填写 pose=[x,y]，需要朝向时填写 pose=[x,y,yaw]。'
                else:
                    function['description'] += (
                        '只需要位置时用 pose=[x,y] 或 point_id。'
                        '需要朝向时用 pose=[x,y,yaw]，或用 point_id 加单独的 yaw。')
        return tools

    def validate_action(self, action):
        """Fill omitted request-scoped values before the normal tool validation."""
        name = action['tool']
        args = action['arguments']
        if not isinstance(args, dict):
            return validate_call(action, model_schema=True)
        args = copy.deepcopy(args)
        if self.goal_id and name in CURRENT_GOAL_TOOLS:
            args.setdefault('goal_id', self.goal_id)
        if self.registered_round_table and name == 'start_local_task':
            args.setdefault('task_type', 'seating')
        if self.view and name in CURRENT_VIEW_TOOLS:
            args.setdefault('view_id', self.reference('view', self.view))
        if self.visual_priority and name == 'record_monitor_review':
            args.setdefault('visual_priority', self.visual_priority)
        if self.rgb and name in CURRENT_RGB_TOOLS and (
                name != 'record_monitor_review' or 'visual_priority' in args or self.default_visual_review):
            args.setdefault('observation_id', self.reference('obs', self.rgb))
        if name == 'navigate':
            if self.initial_pose is not None and not {'pose', 'point_id', 'yaw'}.intersection(args):
                if self.initial_point_id:
                    args['point_id'] = self.initial_point_id
                    if len(self.initial_pose) == 3:
                        args['yaw'] = self.initial_pose[2]
                else:
                    args['pose'] = list(self.initial_pose)
            pose = args.get('pose')
            if self.orientation_source and ('yaw' in args or isinstance(pose, list) and len(pose) == 3):
                args.setdefault('orientation_source', self.orientation_source)
        if name == 'navigate_route' and self.orientation_source:
            args.setdefault('orientation_source', self.orientation_source)
        return validate_call(dict(tool=name, arguments=args), model_schema=True)

    def project(self, value, path=()):
        if isinstance(value, list):
            return [self.project(item, path) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            kind = reference_kind(key)
            if kind:
                result[key] = self.reference(kind, item)
            elif key in LIST_REFERENCES and isinstance(item, list):
                result[key] = [self.reference(LIST_REFERENCES[key], entry) for entry in item]
            elif (key in AUDIT_FIELDS or key.endswith(AUDIT_SUFFIXES) or
                    key.endswith('_binding') and not isinstance(item, dict)):
                continue
            elif isinstance(item, (tuple, list)) and (
                    key == 'target_identity' or key == 'identity' and path[-1:] == ('track',)):
                result[key] = self.reference('track', encode(item))
            elif key == 'last_completion' and isinstance(item, list) and all(isinstance(p, dict) for p in item):
                # The current stage's measurement still appears in completion_evidence.
                result[key] = [{field: proof[field] for field in ('goal_id', 'stamp_s', 'scope', 'source', 'stopped')
                                if field in proof} for proof in item]
            elif key == 'detail' and value.get('reason') in REJECTED_ARGUMENTS and isinstance(item, str):
                result[key] = OPAQUE_TEXT.sub('[internal ID]', item)
            else:
                result[key] = self.project(item, path+(key,))
        return result

    def remember_historical_alias(self, kind, token):
        if isinstance(token, str):
            source = self.names.lookup(token)
            if source is not None and source[0] == kind:
                self.references[token] = source

    def historical_reference(self, kind, value, originals):
        if not isinstance(value, str):
            return value
        if (kind, value) in originals:
            return originals[kind, value]
        if re.fullmatch(re.escape(kind)+r'-[0-9a-f]{16}', value):
            # Map a legacy alias only when its full source is known in this request.
            matches = [name for (source_kind, source), name in originals.items() if source_kind == kind and
                       kind+'-'+hashlib.sha256(source.encode('utf-8')).hexdigest()[:16] == value]
            return matches[0] if len(matches) == 1 else value
        if re.fullmatch(REFERENCE_PREFIXES[kind]+r'[1-9][0-9]*', value):
            # The assistant already copied this alias from an earlier request.
            self.remember_historical_alias(kind, value)
            return value
        return self.reference(kind, value)

    def messages(self, messages, *, protected_indices=()):
        material = copy.deepcopy(messages)
        rejected = set()
        for index, message in enumerate(material):
            role = message.get('role')
            if index in protected_indices or role == 'assistant':
                continue
            content = message.get('content')
            parts = content if isinstance(content, list) else [dict(type='text', text=content)]
            for part in parts:
                if part.get('type') != 'text':
                    continue
                parsed = object_from_text(part.get('text'))
                if parsed is None or role == 'user' and 'source' not in parsed:
                    continue
                if role == 'tool' and parsed.get('reason') in REJECTED_ARGUMENTS:
                    rejected.add(message['tool_call_id'])
                part['text'] = encode(self.project(parsed))
            if isinstance(content, str):
                message['content'] = parts[0]['text']

        originals = {record: token for token, record in self.references.items()}
        for index, message in enumerate(material):
            if index in protected_indices:
                continue
            for call in message.get('tool_calls') or ():
                function = call['function']
                arguments = function['arguments']
                parsed = object_from_text(arguments)
                if call['id'] in rejected:
                    if parsed is not None:
                        for key, kind in ACTION_REFERENCES.items():
                            self.remember_historical_alias(kind, parsed.get(key))
                    function['arguments'] = OPAQUE_TEXT.sub('[internal ID]', arguments)
                    continue
                if parsed is None:
                    continue
                projected = parsed.copy()
                for key, kind in ACTION_REFERENCES.items():
                    if key in parsed:
                        projected[key] = self.historical_reference(kind, parsed[key], originals)
                if projected != parsed:
                    function['arguments'] = encode(projected)
            # Current map captions carry the same identity as their JSON metadata.
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') != 'text' or not isinstance(part.get('text'), str):
                        continue
                    label, separator, raw_view = part['text'].partition(' | view_id=')
                    alias = originals.get(('view', raw_view))
                    if separator and alias:
                        part['text'] = label+separator+alias
        return material

    def resolve_action(self, action):
        result = copy.deepcopy(action)
        if (self.goal_id and result['tool'] in CURRENT_GOAL_TOOLS and
                result['arguments'].get('goal_id', self.goal_id) != self.goal_id):
            raise ValueError('goal_id: use the current stage in this request')
        for key, kind in ACTION_REFERENCES.items():
            if key not in result['arguments']:
                continue
            token = result['arguments'][key]
            if not isinstance(token, str) or token not in self.allowed or self.references[token][0] != kind:
                raise ValueError(f'{key}: use the reference from {REFERENCE_SOURCES[key]} in the current request')
            result['arguments'][key] = self.references[token][1]
        return result
