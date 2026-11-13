"""In-memory T1/T3 requests with the existing source and navigation gates.

The caller owns scheduling, current-scene delivery and any actual motion.
This interface neither starts a service nor issues navigation commands.
"""
import base64
import copy
import hashlib
import io
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
    """One in-flight T1 or T3 decision; acceptance is bound to a current scene."""
    def __init__(self, policy, *, deadline_s=None):
        if policy.kind not in ('social', 'seating'):
            raise ValueError('Only social and seating decisions are supported')
        self.policy = policy
        self.gate = DecisionRequest(policy, deadline_s=deadline_s)
        self.active = None
        self.attempted = False
        self.response = None

    def _prepare(self, scene, envelope, fields, images, *, rgb_observation_binding=None):
        settings = load_config()
        task = settings['tasks']['groups' if self.policy.kind == 'social' else 'seating']
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

    def social(self, scene, window, images, *, now):
        if self.policy.kind != 'social':raise ValueError('Expected social policy')
        if not isinstance(images,(list,tuple)) or len(images)!=3 or len(window.get('images',[]))!=3:
            raise ValueError('T1 requires three bound causal images')
        for image,spec in zip(images,window['images']):
            if not isinstance(image,bytes) or hashlib.sha256(image).hexdigest()!=spec.get('sha256'):
                raise ValueError('T1 image differs from source window')
        envelope=self.gate.begin(scene,window=window,now=now)
        try:return self._prepare(scene,envelope,window['fields'],images)
        except BaseException:
            self.gate.pending=None
            raise

    def seating(self, scene, candidates, *, now, observation=None):
        if self.policy.kind != 'seating':raise ValueError('Expected seating policy')
        # Existing historical edge contexts are offline hypotheses, not a live
        # source of current table geometry or collision clearance.
        if 'context' in candidates:raise ValueError('Current seating input cannot reuse historical context')
        from .seating.inputs import render_scene
        image=io.BytesIO()
        render_scene(scene,candidates,image,observation=observation)
        fields=dict(instruction=candidates['instruction'],units=dict(position='m',yaw='rad'),
                    frame_id=candidates['frame_id'],ego_pose=scene.record['localization']['world_from_lidar'],
                    candidates=copy.deepcopy(candidates['candidates']))
        envelope=self.gate.begin(scene,candidates=candidates,now=now)
        rgb_binding=None if observation is None else observation.record['input_fingerprint']
        try:return self._prepare(scene,envelope,fields,[image.getvalue()],rgb_observation_binding=rgb_binding)
        except BaseException:
            self.gate.pending=None
            raise

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
