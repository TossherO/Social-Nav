"""Independent hosted VLM client. No retries, response repair or backend fallback."""
import base64
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
from ..tools.framework import action_schema, validate_call

class ApiError(RuntimeError):
    pass


class Model:
    model = 'qwen3.8-flash'

    def __init__(self, private_config, budget, store, *, transport=None):
        # Read credentials only for an explicitly requested API operation.
        self._private = json.loads(Path(private_config).read_text())
        self._key = os.environ.get('SOCIAL_NAV_AGENT_API_KEY') or self._private['api_key']
        url = self._private['base_url'].rstrip('/')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API requires HTTPS and credentials outside the URL')
        self._url = url if url.endswith('/chat/completions') else url+'/chat/completions'
        self.budget, self.store, self.transport = budget, store, transport

    def request(self, purpose, system, user, schema, *, image=None, observation_id=None):
        content = [dict(type='text', text=encode(user))]
        image_bound = 0
        if image is not None:
            from PIL import Image
            with Image.open(io.BytesIO(image) if isinstance(image, bytes) else image) as source:
                picture = source.convert('RGB')
                picture.thumbnail((768, 768))
                buf = io.BytesIO()
                picture.save(buf, format='JPEG', quality=80)
            content.append(dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode())))
            image_bound = 32768
        if image is not None and not observation_id:
            raise ValueError('Visual request needs a source observation identity')
        # UTF-8 byte bound deliberately overestimates text tokens. Images are
        # resized and assigned a separate generous reservation before sending.
        input_bound = len((system+encode(user)+encode(schema)).encode())+4096+image_bound
        if input_bound > 65536:
            raise ValueError('Request exceeds development input budget')
        body = dict(model=self.model, messages=[dict(role='system', content=system), dict(role='user', content=content)],
                    enable_thinking=False, temperature=0, max_tokens=1024, stream=True,
                    stream_options=dict(include_usage=True),
                    response_format=dict(type='json_schema', json_schema=dict(name=purpose, strict=True, schema=schema)))
        call_id = self.budget.reserve(input_bound, body['max_tokens'])
        # No credentials, headers or private endpoint in the persisted request.
        self.store.event('api_request', dict(call_id=call_id, purpose=purpose, model=self.model,
            observation_id=observation_id, request=body, input_token_reservation=input_bound))
        began = time.perf_counter()
        first = None
        chunks, text_parts, reason_parts = [], [], []
        usage = None
        returned_model = None
        finish_reason = None
        status = None
        try:
            # Dedicated API connection: do not inherit unrelated shell proxies,
            # proxy credentials or certificate overrides from other workflows.
            with httpx.Client(timeout=httpx.Timeout(90, connect=10), transport=self.transport,
                              follow_redirects=False, trust_env=False) as client:
                with client.stream('POST', self._url, headers={'Authorization': 'Bearer '+self._key}, json=body) as response:
                    status = response.status_code
                    if status != 200:
                        # Keep a bounded, credential-redacted error for diagnosis.
                        detail = response.read()[:16384].decode('utf-8', errors='replace')
                        for secret in (self._key, self._url, self._private['base_url']):
                            detail = detail.replace(secret, '[private]')
                        self.store.event('api_http_error', dict(call_id=call_id, http_status=status, detail=detail))
                        raise ApiError('API rejected request (HTTP '+str(status)+'); no retry performed')
                    for line in response.iter_lines():
                        if not line.startswith('data:'):
                            continue
                        payload = line[5:].strip()
                        if payload == '[DONE]':
                            break
                        chunk = json.loads(payload)
                        chunks.append(chunk)
                        if 'error' in chunk:
                            raise ApiError('API stream reported an error; no retry performed')
                        if chunk.get('usage') is not None:
                            usage = chunk['usage']
                        returned_model = chunk.get('model') or returned_model
                        for choice in chunk.get('choices', []):
                            delta = choice.get('delta') or {}
                            if delta.get('content') or delta.get('reasoning_content'):
                                if first is None:
                                    first = time.perf_counter()-began
                            text_parts.append(delta.get('content') or '')
                            reason_parts.append(delta.get('reasoning_content') or '')
                            finish_reason = choice.get('finish_reason') or finish_reason
            record = dict(call_id=call_id, purpose=purpose, requested_model=self.model, returned_model=returned_model,
                wall_s=time.time(), first_token_s=first, full_return_s=time.perf_counter()-began, usage=usage,
                finish_reason=finish_reason, raw_chunks=chunks, content=''.join(text_parts), reasoning=''.join(reason_parts))
            self.budget.finish(call_id, record)
            self.store.event('api_response', record)
            if returned_model != self.model or finish_reason != 'stop' or record['reasoning']:
                raise ApiError('Unexpected model, truncated response or thinking output; raw response retained')
            value = json.loads(record['content'])
            jsonschema.validate(value, schema)
            return value
        except (httpx.HTTPError, ValueError, KeyError, jsonschema.ValidationError, ApiError) as error:
            # Never expose exception strings from HTTP clients: they may embed
            # the private URL or service body. Partial replies stay in this DB.
            self.store.event('api_failure', dict(call_id=call_id, error_type=type(error).__name__, http_status=status,
                full_return_s=time.perf_counter()-began, partial_chunks=chunks, automatically_retried=False))
            raise ApiError('API request failed or response rejected; see session events, no retry performed') from None


    def decide(self, payload, *, observation=None):
        system = ('你是轮椅的单一长程导航Agent。根据原指令、用户补充、工具历史和当前状态，'
                  '每次只选择一个工具，返回JSON对象tool和arguments，不输出额外字段或内部思维链。'
                  '保持用户要求的站点顺序、等待和返回条件；answer只是补充，只有replace才更换任务。'
                  '可以先lookup_places查询名称，未知地点向用户澄清，不猜坐标或身份。'
                  'navigate的accepted只代表接收；之后可wait_for_event(event=navigation)等结果。'
                  '影子状态、图像接近目标和ACK不能证明到达；没有实际反馈时不能finish_task(succeeded)。'
                  '要视觉确认时先observe_scene，下一轮会提供同源图像；证据不足等新观察或询问用户。'
                  '图像可以不含目标，未看见不等于不存在。历史证据只是过去的事实或模型动作，不证明当前可达。'
                  '不要反复查相同状态；等用户继续用wait_for_event(event=user_continue)，等新图用observation。'
                  '活动导航须先取消并等确认，才可改目标或询问用户。任务无法继续时说明失败，不能假报成功。'
                  '工具失败可依据实际原因选择其他动作，不重放失败API，不换模型。arguments只填写所选工具要求的字段。')
        value = self.request('tool_decision', system, payload, action_schema(),
                             image=None if observation is None else observation.image,
                             observation_id=None if observation is None else observation.record['observation_id'])
        try:
            return validate_call(value)
        except jsonschema.ValidationError:
            raise ApiError('Tool arguments rejected; raw response retained, no repair or retry') from None


def api_work(config, session, payload, observation=None):
    # Background operations own their database connections and credentials.
    store, budget = Store(session), Budget(ROOT/'outputs/decision/agent/api_budget.sqlite')
    try:
        return Model(config, budget, store).decide(payload, observation=observation)
    finally:
        budget.close()
        store.close()
