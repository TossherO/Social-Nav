"""Native hosted tool calls; no retries, response repair or backend fallback."""
import copy
import io
import json
import os
from pathlib import Path
import socket
import time
import base64
import math
import hashlib
from contextlib import closing, contextmanager
from email.utils import parsedate_to_datetime
from threading import Lock, Timer
from urllib.parse import urlsplit
import httpx
import jsonschema
from .. import ROOT
from .store import Store, encode, shared_budget
from .media import image_part, png_part
from .async_model import request_cancelled, request_timing
from .context import project_dialogue, view_context, visual_inputs
from .model_projection import ModelProjection
from perception.environment.context import fingerprint
from ..tools.framework import Reply, EXECUTION_TOOLS, FOLLOW_RESTART_TOOLS, follow_supervision_active, native_tools


class ApiError(RuntimeError):
    pass


class ModelTemporarilyUnavailable(ApiError):
    """A transport or service failure that permits another model request."""

    def __init__(self, message='Agent model API temporarily unavailable', *, retry_after_s=None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


def http_client(transport=None):
    return httpx.Client(timeout=httpx.Timeout(120.,connect=10),transport=transport,
                        follow_redirects=False,trust_env=False)


SSE_TAIL_TIMEOUT_S = .05


def interrupt_http_tail(connection):
    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def stream_payloads(response, timing):
    done = False
    tail_timer = None
    try:
        for line in response.iter_lines():
            if request_cancelled():
                raise ApiError('Request cancelled by a task event')
            if done or not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload=='[DONE]':
                timing['transport_complete_s'] = time.perf_counter()
                # HTTP/1.x can finish locally once all Content-Length bytes
                # have arrived. Bound any remaining reads on a dedicated socket.
                http1 = response.http_version in ('HTTP/1.0','HTTP/1.1')
                fixed_body_received = (http1
                    and 'transfer-encoding' not in response.headers
                    and response.headers.get('content-length','').lstrip('0')==str(response.num_bytes_downloaded))
                if not response.is_closed and not fixed_body_received:
                    stream = response.extensions.get('network_stream') if http1 else None
                    connection = stream.get_extra_info('socket') if stream is not None else None
                    if connection is None:
                        return
                    tail_timer = Timer(SSE_TAIL_TIMEOUT_S,interrupt_http_tail,args=(connection,))
                    tail_timer.daemon = True
                    tail_timer.start()
                done = True
            else:
                yield payload
    except (httpx.TimeoutException,httpx.NetworkError,httpx.RemoteProtocolError):
        # A broken HTTP trailer costs the connection, while the completed
        # provider reply can still be validated and used.
        if not done:
            raise
    else:
        timing['http_body_complete_s'] = time.perf_counter()
        timing.setdefault('transport_complete_s', timing['http_body_complete_s'])
    finally:
        if tail_timer is not None:
            tail_timer.cancel()
            tail_timer.join()


class SessionTransport:
    """Own a lazy HTTP pool for the session's single model request at a time."""

    def __init__(self):
        self._lock = Lock()
        self._client = None
        self._in_use = False
        self._closed = False

    @contextmanager
    def client(self):
        with self._lock:
            if self._closed or request_cancelled():
                raise ApiError('Request cancelled before transport')
            if self._in_use:
                raise RuntimeError('Agent HTTP transport already in use')
            if self._client is None:
                self._client = http_client()
            client = self._client
            self._in_use = True
        try:
            yield client
        finally:
            with self._lock:
                client.cookies.clear()
                self._in_use = False
                if self._closed:
                    client.close()
                    self._client = None

    def close(self):
        with self._lock:
            self._closed = True
            if self._client is not None and not self._in_use:
                self._client.close()
                self._client = None


def retry_after_seconds(value):
    if value is None:
        return None
    value=value.strip()
    if value.isascii() and value.isdecimal():
        seconds=float(value)
        return seconds if math.isfinite(seconds) else None
    try:
        date=parsedate_to_datetime(value)
        if date.tzinfo is not None:
            return max(0.,date.timestamp()-time.time())
    except (ValueError,TypeError,OverflowError,OSError):
        pass
    return None


class InputEstimateExceeded(ValueError):
    def __init__(self, estimate):
        self.estimate = estimate
        super().__init__('Estimated request input exceeds local context limit')


REGISTRATION_TOOLS=('record_task','lookup_places','query_map','observe_scene','wait_for_event','ask_user')
ELEVATOR_TOOLS={
    'monitoring_ride':('record_elevator_observation','set_navigation_monitor','wait_for_event','ask_user'),
    'awaiting_destination_localization':('restart_map_localization','set_navigation_monitor','wait_for_event','ask_user'),
}


def input_estimate(messages, tools):
    """Approximate tokens from observed usage; Base64 is not counted as text."""
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
                    # Empirical estimate for the configured model and recorded
                    # image sizes; the service remains the source of actual usage.
                    visual_bound += math.ceil(width*height/1024)+32
                    part['image_url'] = dict(url='[current image]')
                    images += 1
    if images > 6:
        raise ValueError('At most a reference map, three current images and two historical RGBs are supported')
    estimate = math.ceil(len(encode(dict(messages=material,tools=tools)).encode())/3.6)+visual_bound
    if estimate > 131072:
        raise InputEstimateExceeded(estimate)
    return estimate


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
    """Find the native tool whose retained result contains this exact record."""
    for index in range(len(messages)-1,0,-1):
        message = messages[index]
        if message.get('role')!='tool' or not isinstance(message.get('content'),str):
            continue
        try:
            result = json.loads(message['content'])
        except ValueError:
            continue
        if isinstance(result,dict) and result.get('status')=='available' and result.get(key)==record:
            previous = messages[index-1]
            for call in previous.get('tool_calls') or ():
                if call['id']==message['tool_call_id']:
                    return call['function']['name']
    return None


def observation_context(messages, source, key, record):
    reference = retained_tool_reference(messages,key,record)
    if reference is None:
        return dict(source=source,**{key:record})
    fields = ('view_id','source_stamp_s','frame_id','target_reference','units') if key=='view' else (
              'observation_id','stamp_s','clock_domain','pose_reference')
    return dict(source=source,**{key:{field:record[field] for field in fields if field in record}},
                details_in_tool=reference)


class Model:
    model = 'qwen3.8-flash'

    def __init__(self, private_config, budget, store, *, transport=None, cache_mode='implicit', thinking_mode='off', output_limit=None, transient_sink=None,
                 session_transport=None):
        if cache_mode not in ('implicit','explicit'):
            raise ValueError('Choose an explicit cache policy; no automatic fallback')
        if thinking_mode not in ('off','low','medium'):
            raise ValueError('Choose off, low or medium thinking; no automatic fallback')
        if output_limit is not None and (type(output_limit) is not int or not 1 <= output_limit <= 8192):
            raise ValueError('Output limit must be between 1 and 8192 tokens')
        self.cache_mode = cache_mode
        self.thinking_mode = thinking_mode
        self.output_limit = output_limit
        self.transient_sink = transient_sink
        # Read credentials only for an explicitly requested API operation.
        self._private = json.loads(Path(private_config).read_text())
        self._key = os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or self._private['api_key']
        url = self._private['base_url'].rstrip('/')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API requires HTTPS and credentials outside the URL')
        self._url = url if url.endswith('/chat/completions') else url+'/chat/completions'
        self.budget, self.store, self.transport = budget, store, transport
        self.session_transport = session_transport

    def request(self, system, messages, *, observation_id=None, max_tokens=2048, tool_names=None, tool_choice='auto', projection=None, transient=False,
                supervised_follow=False, prepare_started_s=None):
        timing = request_timing(prepare_started_s if prepare_started_s is not None else time.perf_counter())
        if tool_choice not in ('auto','required'):
            raise ValueError('Unsupported tool choice')
        tools = native_tools(tool_names)
        if projection is not None:
            tools = projection.tools(tools)
        offered={tool['function']['name'] for tool in tools}
        body = dict(model=self.model,messages=cache_messages(system,messages,self.cache_mode),tools=tools,
                    tool_choice=tool_choice,parallel_tool_calls=False,enable_thinking=self.thinking_mode!='off',temperature=0,
                    stream=True,stream_options=dict(include_usage=True))
        if self.thinking_mode != 'off':
            # Bound reasoning and final output together, including reservation.
            # Historical thinking is not used as dialogue context.
            output_bound = self.output_limit or (24576 if self.thinking_mode=='medium' else 6144)
            body.update(reasoning_effort=self.thinking_mode,preserve_thinking=False,max_completion_tokens=output_bound)
        else:
            output_bound = self.output_limit or max_tokens
            body['max_tokens'] = output_bound
        try:
            bound = input_estimate(body['messages'],tools)
        except InputEstimateExceeded as error:
            self.store.event('api_input_rejected',dict(reason='estimated_context_limit',
                estimated_input_tokens=error.estimate,limit=131072,request_sent=False,timing=dict(timing)))
            raise
        estimated_output=self.budget.estimate_output(self.model,self.thinking_mode,output_bound)
        call_id = self.budget.reserve(bound,estimated_output)
        timing['call_id'] = call_id
        request_event=dict(call_id=call_id,purpose='elevator_monitor' if transient else 'native_tool_decision',model=self.model,
            observation_id=observation_id,input_token_reservation=bound,
            input_estimate_method='empirical_utf8_and_image_area',estimated_output_tokens=estimated_output,
            cache_mode=self.cache_mode,timing=dict(timing))
        if not transient:
            request_event['request'] = body
            if projection is not None and projection.references:
                request_event['model_references'] = projection.references
        self.store.event('api_request',request_event)
        # Keep the start used by existing first-token and full-return reports.
        began = time.perf_counter()
        settled=False
        first = usage = returned_model = finish_reason = status = None
        retry_after_s = None
        chunks,text_parts,reason_parts,calls = [],[],[],{}
        try:
            if request_cancelled():
                raise ApiError('Request cancelled before transport')
            if transient and self.transient_sink is not None:
                self.transient_sink(dict(kind='request',call_id=call_id,request=copy.deepcopy(body)))
            with (http_client(self.transport) if self.session_transport is None else self.session_transport.client()) as client:
                request = client.build_request('POST',self._url,headers={'Authorization':'Bearer '+self._key},json=body)
                timing['prepare_completed_s'] = time.perf_counter()
                if request_cancelled():
                    raise ApiError('Request cancelled before transport')
                timing['transport_started_s'] = time.perf_counter()
                with closing(client.send(request,stream=True)) as response:
                    status = response.status_code
                    if status != 200:
                        retry_after_s=retry_after_seconds(response.headers.get('Retry-After'))
                        error_body = response.read()
                        timing['http_body_complete_s'] = time.perf_counter()
                        detail = error_body[:16384].decode('utf-8',errors='replace')
                        for secret in (self._key,self._url,self._private['base_url']):
                            detail = detail.replace(secret,'[private]')
                        self.store.event('api_http_error',dict(call_id=call_id,http_status=status,
                                                               detail='[transient request]' if transient else detail))
                        raise ApiError('API rejected native tool request; no retry performed')
                    for payload in stream_payloads(response,timing):
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
                                timing['transport_first_delta_s'] = time.perf_counter()
                                first = timing['transport_first_delta_s']-began
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
            if request_cancelled():
                raise ApiError('Request cancelled by a task event')
            message = dict(role='assistant',content=''.join(text_parts) or None)
            if calls:
                message['tool_calls'] = [calls[index] for index in sorted(calls)]
            record = dict(call_id=call_id,purpose='native_tool_decision',requested_model=self.model,
                returned_model=returned_model,wall_s=time.time(),first_token_s=first,
                full_return_s=timing['transport_complete_s']-began,
                timing=dict(timing),usage=usage,finish_reason=finish_reason,
                raw_chunks=chunks,content=message['content'],tool_calls=message.get('tool_calls',[]),
                message=message,reasoning=''.join(reason_parts),thinking_mode=self.thinking_mode,
                cache=cache_usage(usage,self.cache_mode))
            saved=({k:record[k] for k in ('call_id','requested_model','returned_model','wall_s','first_token_s',
                                           'full_return_s','timing','usage','finish_reason','thinking_mode','cache')} if transient else record)
            if transient:saved['purpose']='elevator_monitor'
            self.budget.finish(call_id,saved)
            settled=True
            self.store.event('api_response',saved)
            expected = 'tool_calls' if calls else 'stop'
            if returned_model!=self.model or finish_reason!=expected or self.thinking_mode=='off' and record['reasoning']:
                raise ApiError('Unexpected model, finish reason or thinking output')
            reply=Reply.parse(message,report_argument_error=True,model_schema=True,
                              validate_action=None if projection is None else projection.validate_action)
            if transient and self.transient_sink is not None:
                self.transient_sink(dict(kind='response',call_id=call_id,message=copy.deepcopy(message)))
            if calls and message['tool_calls'][0]['function']['name'] not in offered:
                name=message['tool_calls'][0]['function']['name']
                if supervised_follow and name in FOLLOW_RESTART_TOOLS:
                    return Reply(reply.message,None,dict(status='refused',reason='local_follow_owns_recovery',tool=name))
                if projection is not None and name in projection.hidden_tools:
                    return Reply(reply.message,None,dict(status='refused',reason=projection.hidden_tools[name],tool=name))
                raise ValueError('Model called a tool not offered for this phase')
            if reply.action is not None and projection is not None:
                from ..tools.framework import validate_call
                try:
                    action=projection.resolve_action(reply.action)
                except ValueError as error:
                    return Reply(reply.message,None,dict(status='refused',reason='invalid_tool_reference',
                                tool=reply.action['tool'],detail=str(error)))
                validate_call(action,model_schema=True)
                reply=Reply(reply.message,action)
            return reply
        except (httpx.HTTPError,ValueError,KeyError,TypeError,jsonschema.ValidationError,ApiError) as error:
            timing['failed_s'] = time.perf_counter()
            complete = timing.get('transport_complete_s')
            full_return_s = None if complete is None else complete-began
            if not settled:
                self.budget.finish(call_id,dict(requested_model=self.model,thinking_mode=self.thinking_mode,
                    usage=usage,full_return_s=full_return_s,first_token_s=first))
            self.store.event('api_failure',dict(call_id=call_id,error_type=type(error).__name__,http_status=status,
                retry_after_s=retry_after_s,full_return_s=full_return_s,timing=dict(timing),
                partial_chunks=[] if transient else chunks,automatically_retried=False))
            transport_failed=(status in (None,200) and
                              isinstance(error,(httpx.TimeoutException,httpx.NetworkError,httpx.RemoteProtocolError)))
            if not request_cancelled() and (transport_failed or status in (408,429,500,502,503,504)):
                raise ModelTemporarilyUnavailable(retry_after_s=retry_after_s) from None
            raise ApiError('Native tool request failed or response rejected; see session events, no retry performed') from None

    def decide(self, payload, *, observation=None, spatial_view=None):
        prepare_started_s = time.perf_counter()
        registration=payload.get('current_state',{}).get('requirements_registered') is False
        prompt='registration_prompt.txt' if registration else 'system_prompt.txt'
        system = (ROOT/'decision/agent/configs'/prompt).read_text().strip()
        reference = payload.get('reference_map')
        messages = project_dialogue(payload['messages'],reference)
        state = copy.deepcopy(payload['current_state'])
        ride_inputs=payload.get('ride_inputs')
        phase=(state.get('ride') or {}).get('status') if ride_inputs else None
        if ride_inputs and phase not in ELEVATOR_TOOLS:
            raise ValueError('Unknown elevator phase for transient model request')
        state['visual_inputs'] = visual_inputs(state,observation,spatial_view,reference,
            payload.get('keyframes',{}),payload.get('input_status',{}))
        if ride_inputs:
            state['visual_inputs'].update(rgb=dict(available=True,source='current_elevator_rgb'),
                maps=dict(available=False,reason='map_localization_paused'),
                local_lidar_available=any(row[0]=='current_local_lidar' for row in payload['ride_inputs']))
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
        for role,record,data,encoding in payload.get('ride_inputs') or ():
            content.extend([dict(type='text',text=encode(dict(source=role,measurement=record))),
                            image_part(data,1024,encoding)])
        if follow_image is not None:
            content.extend([dict(type='text',text='历史目标参考裁剪，仅用于与当前候选比较，不代表当前可见。'),
                            image_part(crop,512,'PNG')])
        if observation is not None:
            observation.verify()
            rgb = observation.image
            if observation.record.get('follow_candidates') is not None:
                from decision.vlm.t2.target import annotated
                rgb = annotated(observation)
                if observation.record['follow_candidates'].get('mode')=='raw_rgb':
                    content.append(dict(type='text',text='当前 RGB 为原图，持续跟随期间未进行二维人物检测或关联。'))
                else:
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
                            image_part(data,1024,'PNG',preserve_png=True)]
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
        projection=ModelProjection(payload.get('model_reference_names'))
        projection.allow(state)
        messages=projection.messages(messages,protected_indices=(1 if reference is not None else 0,))
        execution_tools=EXECUTION_TOOLS
        supervised_follow=not (ride_inputs or registration) and follow_supervision_active(state.get('navigation',{}))
        if supervised_follow:
            execution_tools=tuple(name for name in execution_tools if name not in FOLLOW_RESTART_TOOLS)
        if not (ride_inputs or registration):
            execution_tools=projection.available_tools(execution_tools)
        return self.request(system,messages,observation_id=None if observation is None else observation.record['observation_id'],
                            tool_names=(ELEVATOR_TOOLS[phase] if ride_inputs else
                                        REGISTRATION_TOOLS if registration else execution_tools),
                            tool_choice='required' if self.thinking_mode=='off' else 'auto',projection=projection,
                            transient=bool(payload.get('ride_inputs')),supervised_follow=supervised_follow,
                            prepare_started_s=prepare_started_s)


def api_work(config, session, payload, observation=None, spatial_view=None, *, cache_mode='implicit', thinking_mode='off',transient_sink=None,
             session_transport=None):
    # Background operations own their database connections and credentials.
    store = Store(session)
    try:
        budget = shared_budget()
        try:
            return Model(config,budget,store,cache_mode=cache_mode,thinking_mode=thinking_mode,
                         transient_sink=transient_sink,session_transport=session_transport).decide(
                             payload,observation=observation,spatial_view=spatial_view)
        finally:
            budget.close()
    finally:
        store.close()
