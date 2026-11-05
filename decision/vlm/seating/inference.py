"""Bound T3 schemas and one explicit remote inference attempt; no motion control."""
import hashlib
import json
import math
from pathlib import Path
import time

from jsonschema import Draft202012Validator

from .. import vlm_api_common as common
from ..annotation.rules import digest
from ..project_config import load_config, project_path


def binding(candidate_set):
    return digest(candidate_set)


def validate_candidate_set(data):
    required={'instruction','observation_id','stamp_s','frame_id','clock_domain','session_id','generation','revision','candidates'}
    if set(data) not in (required, required|{'context'}):raise ValueError('Unexpected candidate-set fields')
    if any(not isinstance(data[k],str) or not data[k].strip() for k in ['instruction','observation_id','frame_id','clock_domain','session_id']):
        raise ValueError('Explicit instruction and observation identity required')
    if type(data['stamp_s']) not in (int,float) or not math.isfinite(data['stamp_s']):raise ValueError('Invalid time')
    if any(type(data[k]) is not int or data[k]<0 for k in ['generation','revision']):raise ValueError('Invalid lifecycle')
    candidates=data['candidates']
    if not isinstance(candidates,list) or len(candidates)>8:raise ValueError('At most eight candidates')
    ids=[]
    for c in candidates:
        if set(c)!={'id','pose','motion','geometry'}:raise ValueError('Unexpected candidate fields')
        if type(c['id']) is not int or not 0<=c['id']<=7:raise ValueError('Invalid candidate ID')
        if (not isinstance(c['pose'],list) or len(c['pose'])!=3 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in c['pose'])):raise ValueError('Expected x/y/yaw')
        if c['motion'] not in {'forward','reverse','wait'} or c['geometry'] not in {'unknown','feasible','blocked'}:
            raise ValueError('Explicit motion and geometry status required')
        ids.append(c['id'])
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate ID')
    if 'context' in data:
        from .geometry import validate_context
        validate_context(data['context'],data)


def selection_schema(candidate_set):
    validate_candidate_set(candidate_set)
    ids=[c['id'] for c in candidate_set['candidates'] if c['geometry']!='blocked']
    branches=[]
    def branch(values,states):
        return dict(type='object',additionalProperties=False,required=['p','s'],
                    properties=dict(p=dict(type='integer',enum=values),s=dict(type='string',enum=states)))
    if ids:branches.append(branch(ids,['v']))
    branches.append(branch([-1],['n','u']))
    return dict(oneOf=branches)


def validate_selection(response,candidate_set):
    Draft202012Validator(selection_schema(candidate_set)).validate(response)
    if type(response['p']) is not int:raise ValueError('Candidate ID must be an integer')


def run(directory,base_url,*,prompt_path=None,result_name='remote'):
    import httpx
    from ..server import check_server
    directory=Path(directory);item=common.read_json(directory/'input.json')
    if digest({k:v for k,v in item.items() if k!='input_fingerprint'})!=item['input_fingerprint']:
        raise ValueError('Frozen input binding changed')
    validate_candidate_set(item['candidate_set'])
    if binding(item['candidate_set'])!=item['candidate_binding']:raise ValueError('Candidate set changed')
    images=item.get('images',[dict(path='image.png',sha256=item.get('image_sha256'))])
    paths=[]
    for spec in images:
        # Historical single-image inputs can reference a sibling image by symlink.
        # The fixed legacy filename and its frozen content hash remain authoritative.
        path=common.asset_path(directory,spec['path']) if 'images' in item else directory/'image.png'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Image changed')
        paths.append(path)
    if not paths:raise ValueError('At least one frozen image required')
    if item['response_schema']!=selection_schema(item['candidate_set']):raise ValueError('Schema changed')
    output=directory/result_name
    if not result_name or result_name in {'.','..'} or Path(result_name).name!=result_name:raise ValueError('Invalid result name')
    output.mkdir(exist_ok=False)
    project=load_config();task=project['tasks']['seating']
    prompt=Path(prompt_path or project_path(task['window_prompt'] if 'images' in item else task['prompt'])).read_text()
    if prompt_path is None and 'selection_context' in item:
        from .geometry import selection_context
        selected=item['selection_context']
        expected=selection_context(item['candidate_set']['context'],item['fields']['ego_pose'],
            item['candidate_set']['candidates'],selected['gate'])
        if selected!=expected:raise ValueError('Frozen prompt routing disagrees with current-pose input')
        if selected['mode']=='adjust_nearby_pose':
            prompt=prompt.rstrip()+'\n\n'+project_path(task['adjustment_prompt']).read_text()
    (output/'prompt.txt').write_text(prompt)
    model=project['model']['served_name']
    case=dict(paths=paths,model=dict(response_schema=item['response_schema'],
        user_prompt=json.dumps(item['fields'],ensure_ascii=False,separators=(',',':'))))
    body=common.make_body(prompt,case,max_tokens=task['max_tokens'],model=model)
    settings=dict(base_input_sha256=hashlib.sha256((directory/'input.json').read_bytes()).hexdigest(),
                  candidate_binding=item['candidate_binding'],request_sha256=digest(body),automatic_retries=0)
    config=dict(base_url=base_url.rstrip('/'),api_key='',model=model)
    with httpx.Client(timeout=90,trust_env=False,follow_redirects=False) as client:
        server=check_server(client,config,info=True)
        if server['status']!='ready' or server.get('server_args',{}).get('speculative_algorithm')!='DSPARK':
            raise ValueError('Expected remote DSPARK; no silent fallback')
        common.save_json(output/'settings.json',dict(settings,server=server))
        for warmup in [True,False]:
            row=dict(warmup=warmup,valid=False,request_sha256=digest(body));timing={};started=time.perf_counter()
            try:
                with client.stream('POST',config['base_url']+'/chat/completions',json=body) as response:
                    row['http_status']=response.status_code
                    if response.status_code!=200:raise ValueError('HTTP failure')
                    row.update(common.read_stream(response,started,timing))
                raw=json.loads(row['raw_content']);validate_selection(raw,item['candidate_set'])
                if row['finish_reason']!='stop' or row['server_model']!=model or row['reasoning_content_present'] or row['usage'].get('reasoning_tokens',0):
                    raise ValueError('Invalid completion metadata')
                row.update(response=raw,valid=True)
            except Exception as error:row['error_type']=type(error).__name__
            row['timing']=timing;common.save_json(output/('warmup.json' if warmup else 'response.json'),row)
            if not row['valid']:raise RuntimeError('Stopped on invalid response; raw output saved without retry')
    return row
