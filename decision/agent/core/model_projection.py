"""Model-facing references for observations, maps, and dialogue."""
import copy
import hashlib
import json
import re
from threading import Lock

from ..tools.framework import unique_fields
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

    def reference(self, kind, value):
        if not isinstance(value, str) or not value:
            return value
        token = self.names.name(kind, value)
        self.references[token] = (kind, value)
        return token

    def allow(self, state):
        inputs = state['visual_inputs']
        sources = [
            ('view', (inputs.get('maps') or {}).get('view_id')),
            ('obs', (inputs.get('rgb') or {}).get('observation_id')),
            ('ref', (state.get('follow_reference') or {}).get('reference_id')),
        ]
        sources.extend(('key', item['keyframe_id']) for item in state.get('keyframes', ()))
        for kind, original in sources:
            if original:
                self.allowed.add(self.reference(kind, original))

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
        for key, kind in ACTION_REFERENCES.items():
            if key not in result['arguments']:
                continue
            token = result['arguments'][key]
            if not isinstance(token, str) or token not in self.allowed or self.references[token][0] != kind:
                raise ValueError(f'{key}: use the reference from {REFERENCE_SOURCES[key]} in the current request')
            result['arguments'][key] = self.references[token][1]
        return result
