"""Compact model-facing identifiers without changing source records or validation."""
import copy
import hashlib
import json
import re

from .store import encode


REFERENCE_FIELDS = {
    'view_id': 'view', 'observation_id': 'obs', 'keyframe_id': 'key',
    'candidate_binding': 'candidates', 'reference_id': 'ref',
    'reference_map_id': 'map',
}
LIST_REFERENCES = {'visible_keyframe_ids': 'key', 'historical_keyframe_ids': 'key'}
OMIT_FIELDS = {
    'task_id', 'call_id', 'session_id', 'local_identity', 'association_session',
    'catalog_fingerprint', 'task_goal_binding', 'map_id', 'snapshot_id', 'sha256', 'evidence_id',
}
REFERENCE_TOKEN = re.compile(r'^(?:view|obs|key|candidates|ref|map)-[0-9a-f]{16}$')


class ModelProjection:
    """Per-request reversible references; no prefix matching or implicit fallback."""
    def __init__(self):
        self.references = {}

    def reference(self, kind, value, *, preserve_short=False):
        if not isinstance(value, str) or not value:
            return value
        if preserve_short and REFERENCE_TOKEN.fullmatch(value):
            return value
        token = kind+'-'+hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
        old = self.references.setdefault(token, (kind, value))
        if old != (kind, value):
            raise ValueError('Model reference collision')
        return token

    def project(self, value, path=(), *, preserve_short=False):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if (key in OMIT_FIELDS or key.endswith(('_sha256','_fingerprint')) or
                        key.endswith('_binding') and key != 'candidate_binding' and not isinstance(item,dict)):
                    continue
                if key == 'binding':
                    if path and path[-1] in ('follow_candidates', 'candidate_set'):
                        result[key] = self.reference('candidates', item, preserve_short=preserve_short)
                    continue
                if key in REFERENCE_FIELDS:
                    result[key] = self.reference(REFERENCE_FIELDS[key], item, preserve_short=preserve_short)
                elif key.endswith('_view_id'):
                    result[key] = self.reference('view', item, preserve_short=preserve_short)
                elif key.endswith('_observation_id'):
                    result[key] = self.reference('obs', item, preserve_short=preserve_short)
                elif key.endswith('_reference_id'):
                    result[key] = self.reference('ref', item, preserve_short=preserve_short)
                elif key in LIST_REFERENCES and isinstance(item, list):
                    result[key] = [self.reference(LIST_REFERENCES[key], entry, preserve_short=preserve_short) for entry in item]
                else:
                    result[key] = self.project(item, path+(key,),preserve_short=preserve_short)
            return result
        if isinstance(value, list):
            return [self.project(item, path,preserve_short=preserve_short) for item in value]
        return value

    def messages(self, messages, *, protected_indices=()):
        material = copy.deepcopy(messages)
        for index,message in enumerate(material):
            role=message.get('role')
            def text(value):
                if not isinstance(value,str):return value
                if index in protected_indices or role=='assistant':return value
                try: parsed=json.loads(value)
                except ValueError:return value
                if not isinstance(parsed,dict):return value
                # User instructions are not program-generated context.
                if role=='user' and 'source' not in parsed:return value
                return encode(self.project(parsed))
            content=message.get('content')
            if isinstance(content,str):message['content']=text(content)
            elif isinstance(content,list):
                for part in content:
                    if part.get('type')=='text':part['text']=text(part.get('text'))
            for call in message.get('tool_calls') or []:
                function=call.get('function') or {}
                arguments=function.get('arguments')
                if isinstance(arguments,str):
                    try: parsed=json.loads(arguments)
                    except ValueError:continue
                    if isinstance(parsed,dict):
                        projected=self.project(parsed,preserve_short=True)
                        if projected!=parsed:function['arguments']=encode(projected)
        # Image labels contain a view ID as plain text rather than structured JSON.
        originals={original:token for token,(_,original) in self.references.items()}
        for index,message in enumerate(material):
            if index in protected_indices:continue
            if isinstance(message.get('content'),list):
                for part in message['content']:
                    if part.get('type')=='text' and isinstance(part.get('text'),str):
                        for original,token in originals.items():
                            part['text']=part['text'].replace('view_id='+original,'view_id='+token)
        return material

    def resolve_action(self, action):
        result=copy.deepcopy(action)
        for key,kind in REFERENCE_FIELDS.items():
            if key not in result['arguments']:continue
            token=result['arguments'][key]
            if not isinstance(token,str) or token not in self.references or self.references[token][0]!=kind:
                raise ValueError('Model returned an unknown or wrong-type reference')
            result['arguments'][key]=self.references[token][1]
        return result
