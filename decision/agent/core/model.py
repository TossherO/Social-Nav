"""Native hosted tool calls; no retries, response repair or backend fallback."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import httpx
import jsonschema
from .. import ROOT
from .store import Budget, Store, encode
from ..tools.framework import Reply, native_tools


class ApiError(RuntimeError):
    pass


def image_part(data, size, format):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as source:
        picture = source.convert('RGB')
        picture.thumbnail((size,size))
        buffer = io.BytesIO()
        picture.save(buffer,format=format,**(dict(quality=80) if format=='JPEG' else {}))
    mime = 'jpeg' if format=='JPEG' else 'png'
    return dict(type='image_url',image_url=dict(url='data:image/'+mime+';base64,'+
                base64.b64encode(buffer.getvalue()).decode()))


def input_bound(messages, tools):
    """Count text conservatively; never count Base64 as text tokens."""
    material = copy.deepcopy(messages)
    images = 0
    for message in material:
        if isinstance(message.get('content'),list):
            for part in message['content']:
                if part.get('type')=='image_url':
                    part['image_url'] = dict(url='[current image]')
                    images += 1
    if images > 3:
        raise ValueError('At most global map, local map and current RGB are supported')
    bound = len(encode(dict(messages=material,tools=tools)).encode())+4096+images*32768
    if bound > 131072:
        raise ValueError('Request exceeds development input budget')
    return bound


def cache_messages(system, messages, mode):
    """Mark stable prefix checkpoints without caching/reusing runtime observations."""
    material = [dict(role='system',content=system),*copy.deepcopy(messages)]
    if mode=='implicit':
        return material
    targets = {0}
    if len(material)>2 and isinstance(material[2].get('content'),str):
        try:
            catalog = json.loads(material[2]['content'])
        except ValueError:
            catalog = None
        if isinstance(catalog,dict) and catalog.get('source')=='initial_map_catalog':
            targets.add(2)
    # The final message is this request's current state and media. The preceding
    # complete history can grow, but its already observed contents stay unchanged.
    if len(material)>4:
        targets.add(len(material)-2)
    for index in sorted(targets):
        content = material[index].get('content')
        if isinstance(content,str):
            material[index]['content'] = [dict(type='text',text=content,cache_control=dict(type='ephemeral'))]
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

    def __init__(self, private_config, budget, store, *, transport=None, cache_mode='implicit'):
        if cache_mode not in ('implicit','explicit'):
            raise ValueError('Choose an explicit cache policy; no automatic fallback')
        self.cache_mode = cache_mode
        # Read credentials only for an explicitly requested API operation.
        self._private = json.loads(Path(private_config).read_text())
        self._key = os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or self._private['api_key']
        url = self._private['base_url'].rstrip('/')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API requires HTTPS and credentials outside the URL')
        self._url = url if url.endswith('/chat/completions') else url+'/chat/completions'
        self.budget, self.store, self.transport = budget, store, transport

    def request(self, system, messages, *, observation_id=None, max_tokens=2048):
        tools = native_tools()
        body = dict(model=self.model,messages=cache_messages(system,messages,self.cache_mode),tools=tools,
                    tool_choice='auto',parallel_tool_calls=False,enable_thinking=False,temperature=0,
                    max_tokens=max_tokens,stream=True,stream_options=dict(include_usage=True))
        bound = input_bound(body['messages'],tools)
        call_id = self.budget.reserve(bound,max_tokens)
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
            if returned_model!=self.model or finish_reason!=expected or record['reasoning']:
                raise ApiError('Unexpected model, finish reason or thinking output')
            return Reply.parse(message)
        except (httpx.HTTPError,ValueError,KeyError,TypeError,jsonschema.ValidationError,ApiError) as error:
            self.store.event('api_failure',dict(call_id=call_id,error_type=type(error).__name__,http_status=status,
                full_return_s=time.perf_counter()-began,partial_chunks=chunks,automatically_retried=False))
            raise ApiError('Native tool request failed or response rejected; see session events, no retry performed') from None

    def decide(self, payload, *, observation=None, spatial_view=None):
        system = ('你是轮椅的单一长程导航Agent。根据原指令、用户补充、工具历史和当前状态，'
                  '通过原生工具调用每次选择一个工具；可以附带简短进度文字，不输出内部思维链。'
                  '保持用户要求的站点顺序、等待和返回条件；answer只是补充，只有replace才更换任务。'
                  '当前优先完整地图的室内点目标、区域目标及顺序串联；simple或none尚未实现，不暗用完整地图。'
                  '尚未登记任务要求时，先record_task登记整条原指令的有序要求；已经登记后不重复登记。这不是工具执行计划。'
                  '点目标使用标注点或用户明确位置；required_yaw只填写用户硬性要求，facing填写明确语义对应的标注对象。'
                  'semantic_conditions列出需要当前RGB判断的要求，不能省略原指令的条件或目标。'
                  '可以先lookup_places或query_map查询名称与空间；未知对象先查询或澄清，不猜身份。'
                  '工具不生成候选点或推荐朝向。query_map提供全局图、局部图、标注坐标、自车和足迹；可用bounds请求目标附近细节。'
                  'P/R/L是图中display_id；工具target_id和facing使用annotations里对应的place_id/region_id/landmark_id。'
                  '你直接选择完整连续pose=[x,y,yaw]，单位米和弧度，属于图中frame_id，参考base_link。'
                  'ROBOT箭头是车体前向；未提供相机外参时不假定相机光轴与它相同，视觉条件依据当前RGB。'
                  '没有用户朝向时也应根据语义、视野、布局和后续任务选择朝向，orientation_source如实标记user/semantic/autonomous。'
                  '区域位置由你选，点的位置遵守要求；提交navigate必须引用刚读取的view_id与当前goal_id，不从候选集合选ID。'
                  'navigate的accepted只代表接收；之后可wait_for_event(event=navigation)等结果。'
                  '到达后调用check_goal，完成这一子任务后才进行下一项。视觉条件要在到达后observe_scene，下一轮依据真实图像提交semantic_result和简短可见证据。'
                  '语义satisfied/not_satisfied/unknown不是几何或执行证明。几何不满足或语义不确定时保持未完成，依据新信息修改自己选的位姿。'
                  '影子状态、图像接近目标和ACK不能证明到达；没有实际反馈时不能finish_task(succeeded)。'
                  '要视觉确认时先observe_scene，下一轮会提供同源图像；证据不足等新观察或询问用户。'
                  '图像可以不含目标，未看见不等于不存在。历史证据只是过去的事实或模型动作，不证明当前可达。'
                  '不要反复查相同状态；等用户继续用wait_for_event(event=user_continue)，等新图用observation。'
                  '活动导航须先取消并等确认，才可改目标或询问用户。任务无法继续时说明失败，不能假报成功。'
                  '工具失败可依据实际原因选择其他动作，不重放失败API，不换模型。参数只填写所选工具要求的字段。'
                  '对话里的旧工具结果是历史，当前状态以runtime_current_state为准；图像仅来自本轮明确提供的观察。'
                  '纯文字回复会等待用户，不会自动判定任务成功。任务结束必须调用finish_task。')
        content = [dict(type='text',text=encode(dict(source='runtime_current_state',
                    state=payload['current_state'])))]
        if observation is not None:
            observation.verify()
            content += [dict(type='text',text=encode(observation_context(payload['messages'],
                'current_rgb_observation','observation',observation.prompt()))),
                image_part(observation.image,768,'JPEG')]
        if spatial_view is not None:
            spatial_view.verify()
            content.append(dict(type='text',text=encode(observation_context(payload['messages'],
                'current_spatial_view','view',spatial_view.record))))
            for role,data in zip(spatial_view.record['image_roles'],spatial_view.images):
                content += [dict(type='text',text=role+' | view_id='+spatial_view.record['view_id']),
                            image_part(data,1024,'PNG')]
        messages = [*copy.deepcopy(payload['messages']),dict(role='user',content=content)]
        return self.request(system,messages,observation_id=None if observation is None else observation.record['observation_id'])


def api_work(config, session, payload, observation=None, spatial_view=None, *, cache_mode='implicit'):
    # Background operations own their database connections and credentials.
    store, budget = Store(session), Budget(ROOT/'outputs/decision/agent/api_budget.sqlite')
    try:
        return Model(config,budget,store,cache_mode=cache_mode).decide(payload,observation=observation,spatial_view=spatial_view)
    finally:
        budget.close()
        store.close()
