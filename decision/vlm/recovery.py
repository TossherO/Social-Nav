"""Single causal T2 re-identification request to the configured DSPARK service."""
import base64
import hashlib
import json

from .follow import appearance_schema,validate_decision
from .follow_target import annotated,verify_candidates
from .project_config import load_config
from .vlm_api_common import request_dspark
from perception.environment.context import fingerprint


PROMPT = (
    'Compare IMAGE_0, an earlier cropped reference of the target person, with IMAGE_1, '
    'the current camera image whose person candidates have yellow numeric IDs. '
    'Use appearance and distinctive clothing or accessories, never position or track ID alone. '
    'Choose unique only when exactly one current candidate matches. Otherwise return ambiguous, '
    'none, or insufficient, with id -1. Return only the schema object; do not guess.'
)


def run(reference, crop, observation, *, client_factory=None):
    """Keep the raw response and original source times; never retry or change backend."""
    observation.verify();value=verify_candidates(observation.record['follow_candidates'],observation)
    if (reference.get('reference_id')!=fingerprint({k:v for k,v in reference.items() if k!='reference_id'}) or
            hashlib.sha256(crop).hexdigest()!=reference.get('crop_sha256') or
            reference.get('clock_domain')!=value['clock_domain'] or reference.get('rgb_stamp_s')>value['rgb_stamp_s']):
        raise ValueError('Recovery reference is invalid or noncausal')
    ids=[c['id'] for c in value['candidates']];schema=appearance_schema(ids)
    model=load_config()['model']['served_name']
    result=dict(status='unavailable',backend='remote_sglang_dspark',reference_id=reference['reference_id'],
        observation_id=value['observation_id'],candidate_binding=value['binding'],scene_binding=value['scene_binding'],
        clock_domain=value['clock_domain'],source_stamp_s=value['stamp_s'],rgb_stamp_s=value['rgb_stamp_s'],
        raw_content=None,response=None,automatically_retried=False)
    if not ids:
        result['reason']='no_current_candidates';return result
    def image(data):return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode('ascii')))
    body=dict(model=model,messages=[dict(role='system',content=PROMPT),dict(role='user',content=[
        dict(type='text',text='IMAGE_0: fixed historical target reference'),image(crop),
        dict(type='text',text='IMAGE_1: current candidate image'),image(annotated(observation)),
        dict(type='text',text=json.dumps(dict(target_description=reference['description'],candidate_ids=ids),ensure_ascii=False))])],
        chat_template_kwargs=dict(enable_thinking=False),temperature=0,max_tokens=128,
        response_format=dict(type='json_schema',json_schema=dict(name='follow_recovery',strict=True,schema=schema)),
        stream=True,stream_options=dict(include_usage=True))
    result['request_binding']=fingerprint(body)
    result.update(request_dspark(body,client_factory=client_factory))
    if result['status']=='complete':
        try:
            parsed=json.loads(result['raw_content']);validate_decision(parsed,schema)
            result['response']=parsed
        except Exception as error:
            result.update(status='unavailable',reason=type(error).__name__)
    return result
