"""Bind causal T1 images and person IDs before a groups request."""
import hashlib

from ..online import OnlineDecision


class GroupsDecision(OnlineDecision):
    def __init__(self, policy, *, deadline_s=None):
        super().__init__(policy, kind='social', task='groups', deadline_s=deadline_s)

    def social(self, scene, window, images, *, now):
        if not isinstance(images, (list, tuple)) or len(images) != 3 or len(window.get('images', [])) != 3:
            raise ValueError('T1 requires three bound causal images')
        for image, spec in zip(images, window['images']):
            if not isinstance(image, bytes) or hashlib.sha256(image).hexdigest() != spec.get('sha256'):
                raise ValueError('T1 image differs from source window')
        envelope = self.gate.begin(scene, window=window, now=now)
        try:
            return self._prepare(scene, envelope, window['fields'], images)
        except BaseException:
            self.gate.pending = None
            raise
