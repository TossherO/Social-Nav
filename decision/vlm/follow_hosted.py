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
from .project_config import ROOT
from .vlm_api_common import recorded_stream

LOCAL_BACKEND = ROOT/'configs/t2_backend.local.json'
API_CONFIG = ROOT/'configs/agent_api.local.json'


def follow_backend(explicit=None, *, path=LOCAL_BACKEND):
    if explicit is not None:backend=explicit
    elif Path(path).is_file():
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value,dict) or set(value)!={'backend'}:raise ValueError('Invalid local T2 backend configuration')
        backend=value['backend']
    else:backend='dspark'
    if backend not in ('dspark','hosted'):raise ValueError('Choose dspark or hosted T2 backend')
    return backend


def hosted_body(body):
    value=copy.deepcopy(body)
    value.pop('regex',None);value.pop('chat_template_kwargs',None)
    value['messages'][0]['content'] += (
        ' The observation value must be the LETTER KEY of observation_options, such as "A", '
        'never the numeric id inside an option. Both fields are required. For example, '
        '{"target":null,"observation":"A"} or {"target":"P0","observation":null}.')
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
