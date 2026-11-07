"""Explicit hosted-API comparison of frozen local tasks; no default backend change.

Plans reference existing images and task inputs by SHA256. Follow histories are
advanced only by this run's accepted responses. No retries or response repair.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from . import vlm_api_common as common
from .annotation.rules import digest
from .follow import FollowHistory, follow_schema, validate_decision
from .project_config import ROOT
from .seating.inference import validate_selection
from decision.agent.core.model import Model
from decision.agent.core.store import Budget


def checked_asset(spec):
    path=Path(spec['path'])
    if not path.is_absolute():path=ROOT/path
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Frozen source changed: '+str(path))
    return path


def validation_schema(task,item,fields):
    return follow_schema(fields['v'],fields['last_target']) if task=='follow' else item['response_schema']


def transport_schema(task,schema,mode):
    if mode=='original':return schema
    if mode!='portable':raise ValueError('Unknown explicit schema policy')
    if task=='groups':
        # The provider rejects uniqueItems; the original schema and global PID
        # uniqueness remain mandatory local checks. Never repair a response.
        def remove_unique(value):
            if isinstance(value,dict):return {k:remove_unique(v) for k,v in value.items() if k!='uniqueItems'}
            if isinstance(value,list):return [remove_unique(v) for v in value]
            return value
        return remove_unique(schema)
    # Use a flat object for this explicit schema comparison. Cross-field
    # constraints are still checked against the original oneOf after receipt.
    branches=schema['oneOf'];properties={}
    for branch in branches:
        for name,spec in branch['properties'].items():
            enum=spec['enum'];target=properties.setdefault(name,dict(type='integer' if type(enum[0]) is int else 'string',enum=[]))
            target['enum'].extend(v for v in enum if v not in target['enum'])
    return dict(type='object',additionalProperties=False,required=branches[0]['required'],properties=properties)


def prepare_case(batch,case,history=None):
    item=common.read_json(checked_asset(case['input']))
    paths=[checked_asset(spec) for spec in case['images']]
    if len(paths)!=3:raise ValueError('This comparison requires three original images')
    if [s['sha256'] for s in case['images']]!=[s['sha256'] for s in item['images']]:
        raise ValueError('Images differ from the frozen task input')
    prompt=case.get('prompt',batch.get('prompt'))
    if hashlib.sha256(prompt['text'].encode()).hexdigest()!=prompt['sha256']:
        raise ValueError('Prompt changed')
    fields=copy.deepcopy(item['fields']);context=None;schema=item['response_schema']
    if batch['task']=='follow':
        if history is None:raise ValueError('Follow requires its own causal history')
        context=history.context(item,item['window'])
        fields['last_target']=context;schema=follow_schema(fields['v'],context)
    schema=transport_schema(batch['task'],schema,batch.get('schema_mode','original'))
    model=dict(user_prompt=json.dumps(fields,ensure_ascii=False,separators=(',',':')),response_schema=schema)
    body=common.make_body(prompt['text'],dict(paths=paths,model=model),max_tokens=batch['max_tokens'],model=Model.model)
    # Hosted API uses a top-level switch, while SGLang uses chat-template kwargs.
    body.pop('chat_template_kwargs');body['enable_thinking']=False
    return item,body,context


def validate_response(task,value,item,body):
    fields=json.loads(body['messages'][1]['content'][-1]['text'])
    if task=='groups':common.validate_groups(value,item['fields']['v'],item['response_schema'])
    elif task=='follow':validate_decision(value,validation_schema(task,item,fields))
    elif task=='seating':validate_selection(value,item['candidate_set'])
    else:raise ValueError('Unsupported task')


class HostedClient:
    """Same HTTPS credential/connection policy and shared allowance as the Agent."""
    def __init__(self,private_config,budget,*,transport=None):
        config=common.read_json(private_config)
        self.key=os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or config['api_key']
        base=config['base_url'].rstrip('/');parsed=urlsplit(base)
        if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('HTTPS endpoint required without embedded credentials')
        self.url=base if base.endswith('/chat/completions') else base+'/chat/completions'
        self.base=base;self.budget=budget;self.transport=transport

    def request(self,body,row):
        if body['model']!=Model.model or body.get('enable_thinking') is not False:
            raise ValueError('Explicit Agent model and non-thinking request required')
        text=[];images=0
        for message in body['messages']:
            content=message['content']
            if isinstance(content,str):text.append(content);continue
            for part in content:
                if part['type']=='text':text.append(part['text'])
                elif part['type']=='image_url':images+=1
                else:raise ValueError('Unexpected request content')
        bound=len((''.join(text)+json.dumps(body['response_format'],ensure_ascii=False)).encode())+4096+images*32768
        if images!=3 or bound>131072 or not 0<body['max_tokens']<=1024:
            raise ValueError('Bounded three-image development requests only')
        call_id=self.budget.reserve(bound,body['max_tokens'])
        row.update(call_id=call_id,input_token_reservation=bound,requested_model=Model.model,
                   request_sha256=digest(body),valid=False,automatic_retries=0)
        timing={};began=time.perf_counter()
        try:
            # Agent creates a dedicated connection per call; include TLS/upload
            # and streaming in the latency instead of timing server decode alone.
            with httpx.Client(timeout=httpx.Timeout(90,connect=10),trust_env=False,
                              follow_redirects=False,transport=self.transport) as client:
                with client.stream('POST',self.url,headers={'Authorization':'Bearer '+self.key},json=body) as response:
                    row['http_status']=response.status_code
                    if response.status_code!=200:
                        detail=response.read()[:8192].decode(errors='replace')
                        for private in (self.key,self.url,self.base):
                            if private:detail=detail.replace(private,'[private]')
                        row['http_error']=detail
                        raise ValueError('HTTP failure')
                    row.update(common.read_stream(response,began,timing))
            if (row['server_model']!=Model.model or row['finish_reason']!='stop' or
                    row['reasoning_content_present'] or row['usage'].get('reasoning_tokens',0)):
                raise ValueError('Unexpected completion metadata')
            return json.loads(row['raw_content'])
        except Exception as error:
            row['error_type']=type(error).__name__
            return None
        finally:
            timing['full_return_s']=time.perf_counter()-began;row['timing']=timing
            # Unknown usage retains its reservation, including HTTP failures.
            self.budget.finish(call_id,{k:v for k,v in row.items() if k not in ('fields','response_schema')})


def inspect_plan(plan):
    ids=set();count=0
    for batch in plan['batches']:
        if batch['task'] not in ('groups','follow','seating'):raise ValueError('Unknown task')
        history=FollowHistory(batch['hold_age_s']) if batch['task']=='follow' else None
        for case in batch['cases']:
            key=(batch['name'],case['id'])
            if key in ids:raise ValueError('Duplicate planned observation')
            ids.add(key);prepare_case(batch,case,history);count+=1
    if not 0<count<=40 or count!=plan['planned_requests']:raise ValueError('Development request budget mismatch')
    return count


def run(plan_path,output,private_config):
    plan=common.read_json(plan_path);count=inspect_plan(plan)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New output directory required; no implicit resume')
    sources=[checked_asset(c['input']).parent.resolve() for b in plan['batches'] for c in b['cases']]
    if any(output.is_relative_to(p) for p in sources):raise ValueError('Do not write into frozen inputs')
    output.mkdir(parents=True)
    common.save_json(output/'plan.json',plan)
    budget=Budget(ROOT/'outputs/decision/agent/api_budget.sqlite');before=budget.summary();rows=[]
    complete=False;failure=None
    try:
        client=HostedClient(private_config,budget)
        for batch in plan['batches']:
            history=FollowHistory(batch['hold_age_s']) if batch['task']=='follow' else None
            for case in batch['cases']:
                began=time.perf_counter();item,body,context=prepare_case(batch,case,history)
                row=dict(batch=batch['name'],task=batch['task'],case_id=case['id'],input=case['input'],images=case['images'],
                    elapsed_s=item.get('elapsed_s'),source_input_fingerprint=item['input_fingerprint'],history_context=context,
                    fields=json.loads(body['messages'][1]['content'][-1]['text']),response_schema=body['response_format']['json_schema']['schema'],
                    prompt_sha256=case.get('prompt',batch.get('prompt'))['sha256'],preparation_s=time.perf_counter()-began)
                row['schema_mode']=batch.get('schema_mode','original')
                row['validation_schema']=validation_schema(batch['task'],item,row['fields'])
                value=client.request(body,row)
                if value is not None:
                    try:
                        validate_response(batch['task'],value,item,body);row.update(response=value,valid=True)
                    except Exception as error:row['error_type']=type(error).__name__
                rows.append(row)
                with (output/'requests.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                print(json.dumps(dict(batch=batch['name'],case=case['id'],valid=row['valid'],response=row.get('response'),
                    timing=row.get('timing'),error=row.get('error_type')),ensure_ascii=False),flush=True)
                if not row['valid']:raise RuntimeError('Stopped on rejected response; no automatic retry')
                if history is not None:history.accept(value,item,item['window'])
        complete=True
    except Exception as error:
        failure=type(error).__name__
    finally:
        common.save_json(output/'summary.json',dict(complete=complete,failure=failure,planned_requests=count,requests=len(rows),
            model=Model.model,connection_policy='new_connection_per_request_matches_agent',image_policy='three_original_png_no_agent_thumbnail',
            automatic_retries=0,warmups=0,budget_before=before,budget_after=budget.summary(),
            scope='frozen_development_samples_no_chassis_no_default_backend_change'))
        budget.close()
    if not complete:raise RuntimeError('API evaluation stopped; inspect saved response and summary')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--api-config',type=Path,default=ROOT/'configs/agent_api.local.json')
    parser.add_argument('--run',action='store_true',help='Explicitly authorize the planned API calls; otherwise inspect only')
    args=parser.parse_args()
    if args.run:
        if args.output is None:parser.error('--run requires --output')
        run(args.plan,args.output,args.api_config)
    else:print(json.dumps(dict(requests=inspect_plan(common.read_json(args.plan)),model_calls=0,credentials_read=False)))


if __name__=='__main__':main()
