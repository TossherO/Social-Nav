"""Prepare a T3 request from current table geometry and source observations."""
import copy
import io

from ..online import OnlineDecision
from .inputs import render_scene


class SeatingDecision(OnlineDecision):
    def __init__(self, policy, *, deadline_s=None):
        super().__init__(policy, kind='seating', task='seating', deadline_s=deadline_s)

    def seating(self, scene, candidates, *, now, observation=None):
        if 'context' in candidates:
            raise ValueError('Current seating input cannot reuse historical context')
        image = io.BytesIO()
        render_scene(scene, candidates, image, observation=observation)
        fields = dict(instruction=candidates['instruction'], units=dict(position='m', yaw='rad'),
                      frame_id=candidates['frame_id'], ego_pose=scene.record['localization']['world_from_lidar'],
                      candidates=copy.deepcopy(candidates['candidates']))
        envelope = self.gate.begin(scene, candidates=candidates, now=now)
        rgb_binding = None if observation is None else observation.record['input_fingerprint']
        try:
            return self._prepare(scene, envelope, fields, [image.getvalue()], rgb_observation_binding=rgb_binding)
        except BaseException:
            self.gate.pending = None
            raise
