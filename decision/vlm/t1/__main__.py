"""Run local T1 group inference on a frozen request when the remote service is unavailable."""
import argparse
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from .. import vlm_api_common as common
from ..annotation.rules import digest
from ..local_inference import LocalEngine
from ..project_config import ROOT, load_config, project_path
from .schema import groups_schema, validate_groups


def payload_fields(body,task):
    if (body.get('temperature') != 0 or body.get('chat_template_kwargs',{}).get('enable_thinking') is not False
            or type(body.get('max_tokens')) is not int or not 1 <= body['max_tokens'] <= 4096
            or body.get('response_format',{}).get('type') != 'json_schema'):
        raise ValueError('Expected a bounded, non-thinking JSON Schema request')
    messages=body.get('messages',[])
    if len(messages)!=2 or [m.get('role') for m in messages]!=['system','user']:
        raise ValueError('Expected the current system plus user request')
    content=messages[1]['content']
    if len(content)!=7 or any(content[2*i].get('text')!='IMAGE_'+str(i) or
                            content[2*i+1].get('type')!='image_url' for i in range(3)):
        raise ValueError('Expected three real images in their original order')
    fields=json.loads(content[-1]['text'])
    expected={'v','dt','ts','e','o','d'}
    if task!='groups':raise ValueError('Local fallback currently supports groups only')
    if set(fields)!=expected:
        raise ValueError('Task fields mismatch; labels and answers must stay separate')
    groups_schema(fields['v'])
    Draft202012Validator.check_schema(body['response_format']['json_schema']['schema'])
    return fields


def validate_result(raw,body,task,fields):
    schema=body['response_format']['json_schema']['schema']
    if task!='groups':raise ValueError('Local fallback currently supports groups only')
    validate_groups(raw,fields['v'],schema)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True,help='Frozen body produced by vlm_api_common.make_body')
    parser.add_argument('--task',choices=['groups'],required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/vlm_local.local.json')
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--gpus')
    parser.add_argument('--fallback-reason',choices=['remote_unavailable','remote_gpu_busy'])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--dry-run',action='store_true',help='No private configuration, GPU or network access')
    return parser.parse_args(argv)


def run(args,engine_factory=LocalEngine):
    body=common.read_json(args.request)
    fields=payload_fields(body,args.task)
    if args.dry_run:
        print(json.dumps(dict(task=args.task,inference_requests=0,network_requests=0,gpu_loaded=False)))
        return 0
    if args.fallback_reason is None or args.out is None:
        raise ValueError('Select an explicit fallback reason and a new output directory')
    config=load_config();options=dict(config['local_inference'])
    if args.config.exists():
        private=common.read_json(args.config)
        if not isinstance(private,dict) or set(private)-{'model_path','gpus','max_memory_gib'}:
            raise ValueError('Unexpected local resource settings')
        options.update(private)
    if args.gpus is not None:
        if not re.fullmatch(r'\d+(?:,\d+)*',args.gpus):raise ValueError('Invalid GPU selection')
        options['gpus']=[int(g) for g in args.gpus.split(',')]
    gpus=options.get('gpus')
    if gpus is not None and (not isinstance(gpus,list) or not gpus or
            any(type(g) is not int or g<0 for g in gpus) or len(set(gpus))!=len(gpus)):
        raise ValueError('Invalid GPU selection')
    if type(options['max_memory_gib']) is not int or options['max_memory_gib']<=0:
        raise ValueError('Invalid memory budget')
    model=(args.model_path or project_path(options.get('model_path',config['model']['path']))).resolve()
    if not (model/'config.json').is_file():raise ValueError('Set an existing local model')
    output=args.out.resolve()
    if output.is_relative_to(model) or output.is_relative_to(args.request.parent.resolve()):
        raise ValueError('Output must not overwrite the request or model assets')
    if body.get('model')!=config['model']['served_name']:
        raise ValueError('The requested model differs from the configured frozen model')
    output.mkdir(parents=True,exist_ok=False)
    engine=engine_factory(model,options)
    settings=dict(backend='local_transformers',task=args.task,fallback_reason=args.fallback_reason,
                  logical_request_sha256=digest(body),automatic_retries=0,network_requests=0,
                  model_runtime=engine.metadata)
    common.save_json(output/'settings.json',settings)
    row=dict(valid=False,logical_request_sha256=digest(body))
    try:
        row.update(engine.generate(body))
        if row['finish_reason']!='stop' or not row['grammar_terminated']:
            raise ValueError('Incomplete constrained generation')
        raw=common.parse_response(row['raw_content']);validate_result(raw,body,args.task,fields)
        row.update(response=raw,valid=True)
    except Exception as error:
        row['error_type']=type(error).__name__
    common.save_json(output/'response.json',row)
    return 0 if row['valid'] else 1


if __name__=='__main__':
    raise SystemExit(run(arguments()))
