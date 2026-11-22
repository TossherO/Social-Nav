"""Explicit hosted T2 transport; local validation and source binding remain mandatory."""
import copy
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from decision.agent.core.model import Model, input_estimate
from decision.agent.core.store import shared_budget
from perception.environment.context import fingerprint
from ..project_config import ROOT, load_config, project_path
from ..vlm_api_common import recorded_stream

LOCAL_BACKEND = ROOT/'configs/t2_backend.local.json'
API_CONFIG = ROOT/'configs/agent_api.local.json'
FOLLOW_BACKENDS = ('dspark', 'hosted', 'reid')


def _local_backend(path):
    path = Path(path)
    if not path.is_file():
        return {'backend': 'dspark'}
    value = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or 'backend' not in value or
            set(value) - {'backend', 'reid_weights'} or value['backend'] not in FOLLOW_BACKENDS or
            ('reid_weights' in value and (value['backend'] != 'reid' or
             not isinstance(value['reid_weights'], str) or not value['reid_weights'].strip()))):
        raise ValueError('Invalid local T2 backend configuration')
    return value


def follow_backend(explicit=None, *, path=LOCAL_BACKEND):
    backend = _local_backend(path)['backend'] if explicit is None else explicit
    if backend not in FOLLOW_BACKENDS:
        raise ValueError('Choose dspark, hosted or reid T2 backend')
    return backend


def follow_settings(explicit=None, *, weights=None, raw_rgb=False, path=LOCAL_BACKEND):
    backend = follow_backend(explicit, path=path)
    if backend != 'reid':
        if weights is not None:
            raise ValueError('--t2-reid-weights requires the reid T2 backend')
        return backend, None
    if raw_rgb:
        raise ValueError('The reid T2 backend requires RGB-to-track association and conflicts with --t2-raw-rgb')
    if weights is None:
        weights = _local_backend(path).get('reid_weights')
    if not isinstance(weights, str) or not weights.strip():
        raise ValueError('The reid T2 backend requires --t2-reid-weights or local reid_weights')
    model = ROOT / Path(weights).expanduser()
    if not model.is_file():
        raise ValueError('T2 ReID weights file is unavailable')
    model = model.resolve()
    if model.suffix not in ('.onnx', '.engine'):
        raise ValueError('T2 ReID weights must be an .onnx (OpenCV CPU) or .engine (TensorRT CUDA) file')
    if model.suffix == '.engine':
        from decision.osnet.reid_trt import metadata_path
        if not metadata_path(model).is_file():
            raise ValueError('T2 ReID TensorRT requires the matching .engine.json metadata file next to the engine')
    return backend, str(model)


def hosted_body(body):
    value=copy.deepcopy(body)
    value.pop('regex',None);value.pop('chat_template_kwargs',None)
    suffix = project_path(load_config()['tasks']['follow']['hosted_prompt_suffix'])
    value['messages'][0]['content'] += ' ' + suffix.read_text(encoding='utf-8').rstrip('\n')
    value.update(model=Model.model,enable_thinking=False,max_tokens=64,
                 response_format=dict(type='json_object'))
    return value


def request_hosted(body, *, client_factory=None, private_config=API_CONFIG, budget_factory=shared_budget, timeout_s=5):
    """Send once; retain raw output and account actual usage without exposing credentials."""
    request=hosted_body(body)
    result=dict(status='unavailable',raw_content=None,automatically_retried=False,backend='hosted',
                transport_request_binding=fingerprint(request))
    timing={};began=time.perf_counter();budget=None;call_id=None
    try:
        config=json.loads(Path(private_config).read_text(encoding='utf-8'))
        key=os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or config['api_key']
        base=config['base_url'].rstrip('/');parsed=urlsplit(base)
        if (parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment or
                not isinstance(key,str) or not key):raise ValueError('Invalid hosted API configuration')
        url=base if base.endswith('/chat/completions') else base+'/chat/completions'
        bound=input_estimate(request['messages'],[])
        budget=budget_factory();call_id=budget.reserve(bound,request['max_tokens'])
        factory=client_factory or httpx.Client
        with factory(timeout=timeout_s,trust_env=False,follow_redirects=False) as client:
            with client.stream('POST',url,headers={'Authorization':'Bearer '+key},json=request) as response:
                result['http_status']=response.status_code
                if response.status_code!=200:raise ValueError('Hosted API HTTP failure')
                reply=recorded_stream(response,began,timing,result)
        result.update(reply)
        if (reply['server_model']!=Model.model or reply['finish_reason']!='stop' or
                reply['reasoning_content_present'] or reply['usage'].get('reasoning_tokens',0)):
            raise ValueError('Invalid hosted completion metadata')
        result['status']='complete'
    except Exception as error:
        result['reason']='hosted_'+type(error).__name__
    finally:
        timing['full_return_s']=time.perf_counter()-began;result['timing']=timing
        if budget is not None:
            if call_id is not None:
                budget.finish(call_id,dict(requested_model=Model.model,returned_model=result.get('server_model'),
                    thinking_mode='off',usage=result.get('usage'),full_return_s=timing['full_return_s'],
                    purpose='t2_follow',finish_reason=result.get('finish_reason')))
            budget.close()
    return result
