"""Navigation tool lifecycle over a supplied backend; no long-task reasoning."""
import copy
import math
import uuid


def spatial_context(value):
    return dict(clock_domain=value.get('clock_domain'),
                local_identity=[value.get(k) for k in ('session_id', 'generation', 'frame_id')],
                map_binding=copy.deepcopy(value.get('map_binding')))


class NavigationSkill:
    def __init__(self, navigation, catalog, task_id, emit):
        if navigation.scope not in ('simulation', 'shadow'):
            raise ValueError('Only simulation and nonexecuting shadow adapters are implemented')
        self.navigation, self.catalog, self.task_id, self.emit = navigation, catalog, task_id, emit
        self.initial = navigation.snapshot()
        self.context = spatial_context(self.initial)
        self.record = dict(status='idle', handle=None, place_id=None, reason=None, context_lost=False,
                           connected=True, completed=[], attempts=[], goal_revision=0)
        self.last_stamp = None

    def _emit(self, event):
        self.emit(event, copy.deepcopy(self.record))

    def start(self, place_id, revision):
        r = self.record
        state = self.navigation.snapshot()
        if r['context_lost'] or not r['connected'] or state.get('input_stale') or state.get('localization_status', 'valid') != 'valid':
            return dict(status='refused', reason='navigation_context_unavailable')
        if r['handle'] is not None:
            return dict(status='refused', reason='cancel_and_confirm_existing_call_first')
        attempt = dict(place_id=place_id, revision=revision, status='refused')
        r['attempts'].append(attempt)
        if place_id not in self.catalog.places or self.catalog.places[place_id].get('pose') is None:
            return dict(status='refused', reason='destination_not_bound')
        place = self.catalog.destination(place_id, self.navigation.scope)
        route = self.navigation.check_route(place)
        if not route['reference_found'] and not (self.navigation.scope == 'shadow' and route.get('proposal_allowed')):
            return dict(status='refused', reason=route.get('reason') or route['status'], route=route)
        r['goal_revision'] += 1
        handle = dict(task_id=self.task_id, plan_revision=revision, goal_revision=r['goal_revision'],
                      call_id=uuid.uuid4().hex, catalog_fingerprint=self.catalog.fingerprint)
        r.update(status='submitting', handle=handle, place_id=place_id, reason=None)
        attempt.update(status='active', call_id=handle['call_id'])
        self.last_stamp = None
        self._emit('navigation_intent')
        try:
            ack = self.navigation.start(place, handle)
            if ack.get('accepted') is not True or ack.get('handle') != handle:
                raise ValueError('Navigation ACK rejected or belongs to another call')
        except (OSError, EOFError, ValueError) as error:
            r['reason'] = 'submission_uncertain_'+type(error).__name__
            self.cancel(reason=r['reason'])
            return dict(status='uncertain', handle=handle, reason=r['reason'])
        r['status'] = 'running' if self.navigation.scope == 'simulation' else 'shadow_proposed'
        self._emit('navigation_ack')
        return dict(status='accepted', handle=handle, scope=self.navigation.scope, execution_completed=False, route=route)

    def cancel(self, *, reason='cancel_requested'):
        r = self.record
        if r['handle'] is None:
            return dict(status='idle', execution_completed=False)
        r.update(status='stopping', reason=reason)
        self._emit('navigation_stop_intent')
        try:
            self.navigation.cancel(r['handle'])
        except (OSError, EOFError, ValueError) as error:
            self.emit('navigation_stop_unconfirmed', dict(error_type=type(error).__name__, automatically_retried=False))
        return dict(status='stopping', handle=copy.deepcopy(r['handle']), execution_completed=False)

    def tick(self):
        r = self.record
        try:
            feedback = self.navigation.tick()
        except (OSError, EOFError, ValueError) as error:
            if r['connected']:
                r['connected'] = False
                if r['handle'] is not None and r['status'] != 'stopping':
                    self.cancel(reason='connection_lost')
                self.emit('navigation_connection_lost', dict(error_type=type(error).__name__, automatically_retried=False))
            return False
        r['connected'] = True
        current = spatial_context(feedback)
        if current != self.context and not r['context_lost']:
            r['context_lost'] = True
            if r['handle'] is not None and r['status'] != 'stopping':
                self.cancel(reason='spatial_context_changed')
            self._emit('navigation_context_lost')
        handle = r['handle']
        if handle is None:
            return False
        if (feedback.get('input_stale') or feedback.get('localization_status', 'valid') != 'valid') and r['status'] != 'stopping':
            self.cancel(reason='navigation_state_unavailable')
            return False
        if feedback.get('handle') != handle:
            return False
        if (feedback.get('scope') != self.navigation.scope or feedback.get('source') != self.initial.get('source') or
                current != spatial_context(self.navigation.snapshot())):
            if r['status'] != 'stopping':
                self.cancel(reason='feedback_identity_invalid')
            return False
        stamp, now = feedback.get('stamp_s'), self.navigation.snapshot().get('stamp_s')
        same_cancel = self.navigation.scope == 'shadow' and r['status'] == 'stopping' and feedback.get('proposal_cancelled')
        if (type(stamp) not in (int, float) or type(now) not in (int, float) or
                not math.isfinite(stamp) or not math.isfinite(now) or not 0 <= now-stamp <= .5 or
                (not r['context_lost'] and self.last_stamp is not None and
                 (stamp < self.last_stamp or stamp == self.last_stamp and not same_cancel))):
            return False
        self.last_stamp = stamp
        if feedback.get('collision') and r['status'] != 'stopping':
            self.cancel(reason='simulation_collision')
            return False
        if r['status'] == 'stopping':
            stopped = (feedback.get('stopped') and feedback.get('execution_feedback') is True if self.navigation.scope == 'simulation'
                       else feedback.get('proposal_cancelled') is True)
            if stopped and feedback.get('status') in ('stopped', 'failed'):
                r.update(status='cancelled', handle=None)
                self._mark_attempt(handle, 'cancelled')
                self._emit('navigation_cancel_confirmed')
                return True
            return False
        goal = feedback.get('goal') or {}
        if (feedback.get('status') == 'arrived' and self.navigation.scope == 'simulation' and
                feedback.get('execution_feedback') is True and feedback.get('stopped') and
                goal.get('task_completed') and goal.get('task_id') == self.task_id and
                goal.get('goal_revision') == handle['goal_revision']):
            proof = dict(place_id=r['place_id'], handle=copy.deepcopy(handle), stamp_s=stamp,
                         scope=self.navigation.scope, source=feedback['source'], context=current,
                         execution_feedback=True, stopped=True)
            r['completed'].append(proof)
            r.update(status='succeeded', handle=None)
            self._mark_attempt(handle, 'succeeded')
            self._emit('navigation_completed')
            return True
        if feedback.get('status') in ('stopped', 'failed'):
            self.cancel(reason=feedback.get('stop_reason') or 'navigation_stopped_without_arrival')
        return False

    def _mark_attempt(self, handle, status):
        for attempt in self.record['attempts']:
            if attempt.get('call_id') == handle['call_id']:
                attempt['status'] = status

    def snapshot(self):
        value = self.navigation.snapshot()
        return dict(copy.deepcopy(self.record), feedback=value, scope=self.navigation.scope,
                    execution_feedback=value.get('execution_feedback') is True)
