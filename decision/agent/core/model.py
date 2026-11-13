"""Native hosted tool calls; no retries, response repair or backend fallback."""
import copy
import io
import json
import os
from pathlib import Path
import time
import base64
import math
import hashlib
from urllib.parse import urlsplit
import httpx
import jsonschema
from .. import ROOT
from .store import Store, encode, shared_budget
from .media import image_part, png_part
from .context import project_dialogue, view_context, visual_inputs
from perception.environment.context import fingerprint
from ..tools.framework import Reply, native_tools


class ApiError(RuntimeError):
    pass


REGISTRATION_TOOLS=('record_task','lookup_places','query_map','observe_scene','wait_for_event','ask_user')


def input_bound(messages, tools):
    """Count text conservatively; never count Base64 as text tokens."""
    material = copy.deepcopy(messages)
    from PIL import Image
    images = visual_bound = 0
    for message in material:
        if isinstance(message.get('content'),list):
            for part in message['content']:
                if part.get('type')=='image_url':
                    url = part['image_url']['url']
                    if not url.startswith(('data:image/png;base64,','data:image/jpeg;base64,')):
                        raise ValueError('Only source-bound embedded PNG/JPEG images are accepted')
                    data = base64.b64decode(url.split(',',1)[1],validate=True)
                    with Image.open(io.BytesIO(data)) as picture:
                        width,height = picture.size
                        picture.verify()
                    if max(width,height)>2048:
                        raise ValueError('Image exceeds the 2048 pixel development limit')
                    # Reservation estimate, not a claim about server tokenization.
                    visual_bound += 1024+2*math.ceil(width/14)*math.ceil(height/14)
                    part['image_url'] = dict(url='[current image]')
                    images += 1
    if images > 6:
        raise ValueError('At most a reference map, three current images and two historical RGBs are supported')
    bound = len(encode(dict(messages=material,tools=tools)).encode())+4096+visual_bound
    if bound > 131072:
        raise ValueError('Request exceeds development input budget')
    return bound


def cache_messages(system, messages, mode):
    """Mark stable prefix checkpoints without caching/reusing runtime observations."""
    material = [dict(role='system',content=system),*copy.deepcopy(messages)]
    if mode=='implicit':
        return material
    targets = {0}
    for index,message in enumerate(material[1:-1],1):
        content = message.get('content')
        try:
            value = json.loads(content if isinstance(content,str) else content[0]['text'])
        except (ValueError,TypeError,KeyError,IndexError):
            continue
        if isinstance(value,dict) and value.get('source') in ('static_reference_map','initial_map_catalog'):
            targets.add(index)
            break
    # The final message is this request's current state and media. The preceding
    # complete history can grow, but its already observed contents stay unchanged.
    if len(material)>4:
        targets.add(len(material)-2)
    for index in sorted(targets):
        content = material[index].get('content')
        if isinstance(content,str):
            material[index]['content'] = [dict(type='text',text=content,cache_control=dict(type='ephemeral'))]
        elif isinstance(content,list):
            texts = [p for p in content if p.get('type')=='text']
            if texts:
                texts[-1]['cache_control'] = dict(type='ephemeral')
    return material


def cache_usage(usage, mode):
    usage = usage or {}
    details = usage.get('prompt_tokens_details') or {}
    prompt,cached = usage.get('prompt_tokens'),details.get('cached_tokens')
    ratio = cached/prompt if type(prompt) is int and prompt>0 and type(cached) is int and 0<=cached<=prompt else None
    return dict(mode=mode,prompt_tokens=prompt,cached_tokens=cached,
                cache_creation_input_tokens=details.get('cache_creation_input_tokens'),hit_fraction=ratio)


def retained_tool_reference(messages, key, record):
    """Reference metadata only when the exact full record is still in the dialogue."""
    for message in reversed(messages):
        if message.get('role')!='tool' or not isinstance(message.get('content'),str):
            continue
        try:
            result = json.loads(message['content'])
        except ValueError:
            continue
        if isinstance(result,dict) and result.get('status')=='available' and result.get(key)==record:
            return message['tool_call_id']
    return None


def observation_context(messages, source, key, record):
    reference = retained_tool_reference(messages,key,record)
    if reference is None:
        return dict(source=source,**{key:record})
    fields = ('view_id','source_stamp_s','frame_id','target_reference','units') if key=='view' else (
              'observation_id','stamp_s','clock_domain','pose_reference')
    return dict(source=source,**{key:{field:record[field] for field in fields if field in record}},
                details_in_tool_call_id=reference)


class Model:
    model = 'qwen3.8-flash'

    def __init__(self, private_config, budget, store, *, transport=None, cache_mode='implicit', thinking_mode='off', output_limit=None):
        if cache_mode not in ('implicit','explicit'):
            raise ValueError('Choose an explicit cache policy; no automatic fallback')
        if thinking_mode not in ('off','low'):
            raise ValueError('Choose off or low thinking; no automatic fallback')
        if output_limit is not None and (type(output_limit) is not int or not 1 <= output_limit <= 8192):
            raise ValueError('Output limit must be between 1 and 8192 tokens')
        self.cache_mode = cache_mode
        self.thinking_mode = thinking_mode
        self.output_limit = output_limit
        # Read credentials only for an explicitly requested API operation.
        self._private = json.loads(Path(private_config).read_text())
        self._key = os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or self._private['api_key']
        url = self._private['base_url'].rstrip('/')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API requires HTTPS and credentials outside the URL')
        self._url = url if url.endswith('/chat/completions') else url+'/chat/completions'
        self.budget, self.store, self.transport = budget, store, transport

    def request(self, system, messages, *, observation_id=None, max_tokens=2048, tool_names=None, tool_choice='auto'):
        if tool_choice not in ('auto','required'):
            raise ValueError('Unsupported tool choice')
        tools = native_tools(tool_names)
        offered={tool['function']['name'] for tool in tools}
        body = dict(model=self.model,messages=cache_messages(system,messages,self.cache_mode),tools=tools,
                    tool_choice=tool_choice,parallel_tool_calls=False,enable_thinking=self.thinking_mode=='low',temperature=0,
                    stream=True,stream_options=dict(include_usage=True))
        if self.thinking_mode == 'low':
            # Bound reasoning and final output together, including reservation.
            # Historical thinking is not used as dialogue context.
            output_bound = self.output_limit or 6144
            body.update(reasoning_effort='low',preserve_thinking=False,max_completion_tokens=output_bound)
        else:
            output_bound = self.output_limit or max_tokens
            body['max_tokens'] = output_bound
        bound = input_bound(body['messages'],tools)
        call_id = self.budget.reserve(bound,output_bound)
        self.store.event('api_request',dict(call_id=call_id,purpose='native_tool_decision',model=self.model,
            observation_id=observation_id,request=body,input_token_reservation=bound,cache_mode=self.cache_mode))
        began = time.perf_counter()
        first = usage = returned_model = finish_reason = status = None
        chunks,text_parts,reason_parts,calls = [],[],[],{}
        try:
            with httpx.Client(timeout=httpx.Timeout(90,connect=10),transport=self.transport,
                              follow_redirects=False,trust_env=False) as client:
                with client.stream('POST',self._url,headers={'Authorization':'Bearer '+self._key},json=body) as response:
                    status = response.status_code
                    if status != 200:
                        detail = response.read()[:16384].decode('utf-8',errors='replace')
                        for secret in (self._key,self._url,self._private['base_url']):
                            detail = detail.replace(secret,'[private]')
                        self.store.event('api_http_error',dict(call_id=call_id,http_status=status,detail=detail))
                        raise ApiError('API rejected native tool request; no retry performed')
                    for line in response.iter_lines():
                        if not line.startswith('data:'):
                            continue
                        payload = line[5:].strip()
                        if payload=='[DONE]':
                            break
                        chunk = json.loads(payload)
                        chunks.append(chunk)
                        if 'error' in chunk:
                            raise ApiError('API stream reported an error')
                        if chunk.get('usage') is not None:
                            usage = chunk['usage']
                        returned_model = chunk.get('model') or returned_model
                        for choice in chunk.get('choices',[]):
                            delta = choice.get('delta') or {}
                            if not isinstance(delta,dict):
                                raise ValueError('Unexpected stream delta')
                            fragments = delta.get('tool_calls') or []
                            if not isinstance(fragments,list):
                                raise ValueError('Tool deltas must be a list')
                            if first is None and (delta.get('content') or delta.get('reasoning_content') or fragments):
                                first = time.perf_counter()-began
                            text_parts.append(delta.get('content') or '')
                            reason_parts.append(delta.get('reasoning_content') or '')
                            for fragment in fragments:
                                if not isinstance(fragment,dict):
                                    raise ValueError('Unexpected tool delta')
                                index = fragment.get('index')
                                if type(index) is not int or index<0:
                                    raise ValueError('Tool delta requires an original stream index')
                                call = calls.setdefault(index,dict(id='',type='',function=dict(name='',arguments='')))
                                for field in ('id','type'):
                                    if fragment.get(field):
                                        if call[field] and call[field]!=fragment[field]:
                                            raise ValueError('Tool metadata changed within a stream')
                                        call[field] = fragment[field]
                                function = fragment.get('function') or {}
                                if not isinstance(function,dict):
                                    raise ValueError('Unexpected function delta')
                                for field,value in function.items():
                                    if field not in ('name','arguments') or not isinstance(value,str):
                                        raise ValueError('Unexpected function delta')
                                    call['function'][field] += value
                            finish_reason = choice.get('finish_reason') or finish_reason
            message = dict(role='assistant',content=''.join(text_parts) or None)
            if calls:
                message['tool_calls'] = [calls[index] for index in sorted(calls)]
            record = dict(call_id=call_id,purpose='native_tool_decision',requested_model=self.model,
                returned_model=returned_model,wall_s=time.time(),first_token_s=first,
                full_return_s=time.perf_counter()-began,usage=usage,finish_reason=finish_reason,
                raw_chunks=chunks,content=message['content'],tool_calls=message.get('tool_calls',[]),
                message=message,reasoning=''.join(reason_parts),cache=cache_usage(usage,self.cache_mode))
            self.budget.finish(call_id,record)
            self.store.event('api_response',record)
            expected = 'tool_calls' if calls else 'stop'
            if returned_model!=self.model or finish_reason!=expected or self.thinking_mode=='off' and record['reasoning']:
                raise ApiError('Unexpected model, finish reason or thinking output')
            reply=Reply.parse(message)
            if reply.action is not None and reply.action['tool'] not in offered:
                raise ValueError('Model called a tool not offered for this phase')
            return reply
        except (httpx.HTTPError,ValueError,KeyError,TypeError,jsonschema.ValidationError,ApiError) as error:
            self.store.event('api_failure',dict(call_id=call_id,error_type=type(error).__name__,http_status=status,
                full_return_s=time.perf_counter()-began,partial_chunks=chunks,automatically_retried=False))
            raise ApiError('Native tool request failed or response rejected; see session events, no retry performed') from None

    def decide(self, payload, *, observation=None, spatial_view=None):
        registration=payload.get('current_state',{}).get('requirements_registered') is False
        prompt='registration_prompt.txt' if registration else 'system_prompt.txt'
        system = (ROOT/'decision/agent/configs'/prompt).read_text().strip()
        if not registration and payload.get('current_state',{}).get('execution_mode',{}).get('mode') == 'operator_driven':
            system += ('\n\n当前是人工驾驶测试：navigate 只向操作者提供目标位姿，轮椅由操作者使用原车手柄驾驶。'
                       'navigate_route 可一次提交少量途经点，操作者连续经过，中间不要求停车或模型逐点决策。'
                       'Agent 不控制底盘、不负责行驶路径；提交目标后等待人工确认及当前定位停稳反馈。'
                       '绕行或顺序任务还须按实际定位轨迹检查；路线仅作参考，不证明通行安全。'
                       '不能把目标接收或人工确认单独说成到达，也不能声称自动导航成功。')
        reference = payload.get('reference_map')
        messages = project_dialogue(payload['messages'],reference)
        state = copy.deepcopy(payload['current_state'])
        state['visual_inputs'] = visual_inputs(state,observation,spatial_view,reference,
            payload.get('keyframes',{}),payload.get('input_status',{}))
        follow_image=payload.get('follow_reference_image')
        if follow_image is not None:
            follow_record,crop=follow_image
            feedback=state['navigation']['feedback']
            if (follow_record!=state.get('follow_reference') or
                    follow_record['reference_id']!=fingerprint({k:v for k,v in follow_record.items() if k!='reference_id'}) or
                    hashlib.sha256(crop).hexdigest()!=follow_record['crop_sha256'] or
                    follow_record['clock_domain']!=feedback['clock_domain'] or follow_record['rgb_stamp_s']>feedback['stamp_s']):
                raise ValueError('Follow reference image is invalid or noncausal')
            state['visual_inputs']['follow_reference']=dict(reference_id=follow_record['reference_id'],
                source_stamp_s=follow_record['rgb_stamp_s'],historical_only=True)
        content = [dict(type='text',text=encode(dict(source='runtime_current_state',
                    state=state)))]
        if follow_image is not None:
            content.extend([dict(type='text',text='历史目标参考裁剪，仅用于与当前候选比较，不代表当前可见。'),
                            image_part(crop,512,'PNG')])
        if observation is not None:
            observation.verify()
            rgb = observation.image
            if observation.record.get('follow_candidates') is not None:
                from decision.vlm.follow_target import annotated
                rgb = annotated(observation)
                content.append(dict(type='text',text='当前RGB中的黄色数字是本观察的跟随候选编号；原图和检测框另行绑定。'))
            content += [dict(type='text',text=encode(observation_context(messages,
                'current_rgb_observation','observation',observation.prompt()))),
                image_part(rgb,768,'JPEG')]
        if spatial_view is not None:
            spatial_view.verify()
            content.append(dict(type='text',text=encode(observation_context(messages,
                'current_spatial_view','view',view_context(spatial_view.record,reference)))))
            for role,data in zip(spatial_view.record['image_roles'],spatial_view.images):
                content += [dict(type='text',text=role+' | view_id='+spatial_view.record['view_id']),
                            image_part(data,1024,'PNG')]
        keyframes = payload.get('keyframes',{})
        remaining = set(keyframes)
        for message in reversed(messages):
            if message.get('role')!='user' or not isinstance(message.get('content'),str):
                continue
            try:
                value = json.loads(message['content'])
            except ValueError:
                continue
            if isinstance(value,dict) and value.get('source')=='historical_keyframe':
                identity = value['keyframe']['keyframe_id']
                if identity in remaining:
                    record,data = keyframes[identity]
                    if record != value['keyframe']:
                        raise ValueError('Historical attachment and saved keyframe disagree')
                    feedback = payload['current_state']['navigation']['feedback']
                    if (record['observation']['clock_domain']!=feedback['clock_domain'] or
                            record['source_stamp_s']>feedback['stamp_s']):
                        raise ValueError('Historical image is future or belongs to a different clock')
                    if (hashlib.sha256(data).hexdigest()!=record['image_sha256'] or
                            fingerprint({k:v for k,v in record.items() if k!='record_fingerprint'})!=record['record_fingerprint']):
                        raise ValueError('Historical keyframe changed before request')
                    message['content'] = [dict(type='text',text=message['content']),image_part(data,768,'JPEG')]
                    remaining.remove(identity)
        if remaining:
            raise ValueError('Historical image has no corresponding retained dialogue message')
        if reference is not None:
            reference.verify()
            messages.insert(0,dict(role='user',content=[
                dict(type='text',text=encode(dict(source='static_reference_map',reference=reference.record))),
                png_part(reference.image),dict(type='text',text='固定地图和全部固定标注；动态起点、实测轨迹与当前自车在本轮状态中。')]))
        messages.append(dict(role='user',content=content))
        return self.request(system,messages,observation_id=None if observation is None else observation.record['observation_id'],
                            tool_names=REGISTRATION_TOOLS if registration else None,
                            tool_choice='required' if registration else 'auto')


def api_work(config, session, payload, observation=None, spatial_view=None, *, cache_mode='implicit', thinking_mode='off'):
    # Background operations own their database connections and credentials.
    store, budget = Store(session), shared_budget()
    try:
        return Model(config,budget,store,cache_mode=cache_mode,thinking_mode=thinking_mode).decide(payload,observation=observation,spatial_view=spatial_view)
    finally:
        budget.close()
        store.close()
