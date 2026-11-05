"""Bounded test inputs and pacing; no model decisions or execution feedback."""
import math


class SimulationDriver:
    def __init__(self, navigation, *, max_s, fast=False, scripted_user_events=False):
        if not math.isfinite(max_s) or max_s <= 0:
            raise ValueError('Positive finite simulation budget required')
        self.navigation, self.fast = navigation, fast
        self.until = navigation.t+max_s
        self.scripted_user_events = scripted_user_events

    def poll(self, actor):
        if self.navigation.t >= self.until:
            return 'simulation_time_limit'
        if self.scripted_user_events and actor.state['status'] == 'waiting_user':
            actor.session.store.event('user_event_source', dict(source='explicit_simulation_fixture'))
            actor.event('continue')
        return None

    def interval(self, actor):
        # Bound source-clock progress during API wall time; fixtures can run fast.
        return (.001 if self.fast else .1) if actor.model.busy else .001


class SavedObservationDriver:
    def __init__(self, navigation, dataset, sequences):
        self.navigation, self.dataset = navigation, dataset
        self.remaining = iter(sequences[1:])

    def poll(self, actor):
        # Freeze the source during a model request. A new observation only enters
        # after the current decision asks to wait; clarification waits for a user.
        if actor.model.busy or actor.state['status'] not in ('waiting_observation', 'waiting_navigation'):
            return None
        sequence = next(self.remaining, None)
        if sequence is None:
            return 'input_exhausted'
        self.navigation.update(self.dataset.read(sequence))
        return None

    def interval(self, actor):
        return .01
