"""Single-flight VLM request and response handling for T1 and T3."""
import base64
import copy
import json

from perception.environment.context import fingerprint
from .navigation import DecisionRequest
from .project_config import load_config, project_path
from .vlm_api_common import request_dspark


def _image(data):
    if not isinstance(data, bytes):
        raise ValueError('Immutable encoded image bytes required')
    return dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode('ascii')))


class OnlineDecision:
    """Prepare, send and accept one decision against its bound observation."""
    def __init__(self, policy, *, kind, task, deadline_s=None):
        if policy.kind != kind:
            raise ValueError('Expected '+kind+' policy')
        self.policy = policy
        self.task = task
        self.gate = DecisionRequest(policy, deadline_s=deadline_s)
        self.active = None
        self.attempted = False
        self.response = None

    def _prepare(self, scene, envelope, fields, images, *, rgb_observation_binding=None):
        settings = load_config()
        task = settings['tasks'][self.task]
        prompt = project_path(task['prompt']).read_text()
        content = []
        for index, image in enumerate(images):
            content.extend([dict(type='text', text='IMAGE_'+str(index)), _image(image)])
        content.append(dict(type='text', text=json.dumps(fields, ensure_ascii=False, separators=(',', ':'))))
        body = dict(model=settings['model']['served_name'],
            messages=[dict(role='system', content=prompt), dict(role='user', content=content)],
            chat_template_kwargs=dict(enable_thinking=False), temperature=0,
            max_tokens=task.get('max_tokens',settings['inference']['max_tokens']),
            response_format=dict(type='json_schema',json_schema=dict(name='social_nav_REL', strict=True,
                                                                     schema=envelope['response_schema'])),
            stream=True, stream_options=dict(include_usage=True))
        result = dict(kind=self.policy.kind, envelope=copy.deepcopy(envelope),
                      scene_binding=scene.record['input_fingerprint'], body=body,
                      rgb_observation_binding=rgb_observation_binding, request_binding=fingerprint(body))
        self.active = fingerprint(result)
        self.attempted = False
        self.response = None
        return result

    def _check(self, prepared):
        if (self.active is None or fingerprint(prepared)!=self.active or
                fingerprint(prepared['body'])!=prepared['request_binding']):
            raise ValueError('Online decision request changed or is no longer active')

    def infer(self, prepared, *, client_factory=None):
        """One explicit DSPARK call; the caller retains the returned raw result."""
        self._check(prepared)
        if self.attempted:raise ValueError('This request was already attempted; no automatic retry')
        self.attempted=True
        result=dict(kind=self.policy.kind,request_binding=prepared['request_binding'],
                    **request_dspark(prepared['body'],client_factory=client_factory))
        self.response=copy.deepcopy(result)
        return result

    def accept(self, prepared, response, *, scene, now):
        self._check(prepared)
        if (response!=self.response or response.get('status')!='complete' or response.get('request_binding')!=prepared['request_binding'] or
                not isinstance(response.get('raw_content'),str)):
            raise ValueError('A complete bound model response is required')
        try:return self.gate.accept(prepared['envelope'],response['raw_content'],scene=scene,now=now)
        finally:self.active=None

    def discard(self, prepared, response):
        """Release one failed attempt without retrying or refreshing its source."""
        self._check(prepared)
        if (not isinstance(response,dict) or response!=self.response or response.get('request_binding')!=prepared['request_binding'] or
                response.get('status')!='unavailable'):
            raise ValueError('A bound failed response is required')
        self.gate.last_response=dict(envelope=copy.deepcopy(prepared['envelope']),
            raw_response=response.get('raw_content'),accepted=False,reason=response.get('reason'))
        self.gate.pending=None
        self.active=None
        return copy.deepcopy(self.gate.last_response)

    def cancel(self):
        self.active=None
        return self.gate.cancel()
