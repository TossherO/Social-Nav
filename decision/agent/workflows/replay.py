"""Explicit decision fixtures for offline checks; never model inference."""
import copy
from ..tools.framework import Reply


class ActionReplay:
    """Explicit tool fixture, not a model and never execution feedback."""
    def __init__(self, actions):
        from ..tools.framework import validate_call
        if not isinstance(actions, list) or not 1 <= len(actions) <= 64:
            raise ValueError('Use one to sixty-four explicit tool actions')
        self.actions = [copy.deepcopy(validate_call(a)) for a in actions]
        self.index = 0

    def __call__(self, payload, observation, spatial_view=None):
        if self.index == len(self.actions):
            return Reply.fixture(dict(tool='ask_user', arguments=dict(question='离线工具轨迹已结束，等待明确的下一步操作。')))
        action = copy.deepcopy(self.actions[self.index])
        self.index += 1
        return Reply.fixture(action)
