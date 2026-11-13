"""Navigation tool lifecycle over a supplied backend; no long-task reasoning."""
import copy
import math
import uuid


class IntentRejected(ValueError):
    """The runtime explicitly rejected an intent before accepting it."""


def spatial_context(value):
    return dict(clock_domain=value.get('clock_domain'),
                local_identity=[value.get(k) for k in ('session_id', 'generation', 'frame_id')],
                map_binding=copy.deepcopy(value.get('map_binding')))


class NavigationSkill:
    def __init__(self, navigation, catalog, task_id, emit):
        if navigation.scope not in ('simulation', 'shadow', 'live', 'manual'):
            raise ValueError('Unknown navigation adapter scope')
        self.navigation, self.catalog, self.task_id, self.emit = navigation, catalog, task_id, emit
        self.initial = navigation.snapshot()
        self.context = spatial_context(self.initial)
        self.record = dict(status='idle', handle=None, place_id=None, reason=None, context_lost=False,
                           connected=True, completed=[], attempts=[], goal_revision=0, goal_id=None, selected_pose=None,
                           pending_completion=None, stage_type='navigation',follow_ready=False,search_deadline_s=None)
        self.last_stamp = None
        self.blocked_since = None
        self.blocked_notified = False

    def _emit(self, event):
        self.emit(event, copy.deepcopy(self.record))

    def start(self, destination, revision):
        r = self.record
        self.catalog.require_complete()
        goal_id = destination['goal_id']
        place_id = destination['place_id']
        state = self.navigation.snapshot()
        if r['context_lost'] or not r['connected'] or state.get('input_stale') or state.get('localization_status', 'valid') != 'valid':
            return dict(status='refused', reason='navigation_context_unavailable')
        if r['handle'] is not None and not (self.navigation.scope == 'live' and r['status'] == 'held'):
            return dict(status='refused', reason='cancel_and_confirm_existing_call_first')
        if 'route' in destination and not callable(getattr(self.navigation,'start_route',None)):
            return dict(status='refused',reason='route_execution_not_supported_by_adapter')
        attempt = dict(goal_id=goal_id, place_id=place_id, revision=revision, status='refused')
        r['attempts'].append(attempt)
        if ((self.navigation.scope == 'simulation') != (self.catalog.value['scope'] == 'simulation') or
                self.navigation.scope in ('shadow','live','manual') and self.catalog.value['scope'] != 'development_prior' or
                destination['frame_id'] != self.catalog.value['frame_id']):
            return dict(status='refused', reason='destination_scope_or_frame_mismatch')
        requirement = destination['requirement']
        if requirement['kind'] == 'point' and requirement['target_id']:
            # Retain the named-point access policy without borrowing its pose.
            self.catalog.destination(requirement['target_id'], self.navigation.scope)
        if requirement['kind'] == 'sequence':
            for ident in requirement['checkpoints']:
                if ident in self.catalog.places:
                    self.catalog.destination(ident,self.navigation.scope)
        place = copy.deepcopy(destination)
        route = self.navigation.check_route(place)
        if not route['reference_found'] and not (self.navigation.scope in ('shadow','live','manual') and route.get('proposal_allowed')):
            return dict(status='refused', reason=route.get('reason') or route['status'], route=route)
        r['goal_revision'] += 1
        handle = dict(task_id=self.task_id, plan_revision=revision, goal_revision=r['goal_revision'],
                      call_id=uuid.uuid4().hex, catalog_fingerprint=self.catalog.fingerprint,
                      goal_id=goal_id, task_goal_binding=destination['task_goal_binding'], view_id=destination['view_id'])
        previous_hold = ({key:copy.deepcopy(r[key]) for key in ('status','handle','place_id','goal_id','selected_pose','reason','stage_type','follow_ready','search_deadline_s')}
                         if self.navigation.scope == 'live' and r['status'] == 'held' else None)
        r.update(status='submitting', handle=handle, place_id=place_id, goal_id=goal_id,
                 selected_pose=copy.deepcopy(destination['pose']),reason=None,stage_type='navigation',
                 search_deadline_s=destination.get('valid_until_s'))
        attempt.update(status='active', call_id=handle['call_id'])
        self.last_stamp = None
        self.blocked_since = None
        self.blocked_notified = False
        self._emit('navigation_intent')
        try:
            ack = (self.navigation.start_route(place, handle) if 'route' in place
                   else self.navigation.start(place, handle))
            if ack.get('accepted') is not True or ack.get('handle') != handle:
                raise ValueError('Navigation ACK rejected or belongs to another call')
        except IntentRejected as error:
            if previous_hold is None:
                r.update(status='refused', handle=None, reason=str(error))
            else:
                r.update(previous_hold)
            self._mark_attempt(handle, 'refused')
            self._emit('navigation_refused')
            return dict(status='refused', reason=str(error))
        except (OSError, EOFError, ValueError) as error:
            r['reason'] = 'submission_uncertain_'+type(error).__name__
            self.cancel(reason=r['reason'])
            return dict(status='uncertain', handle=handle, reason=r['reason'])
        r['status'] = 'shadow_proposed' if self.navigation.scope == 'shadow' else 'running'
        self._emit('navigation_ack')
        return dict(status='accepted', handle=handle, scope=self.navigation.scope, execution_completed=False, route=route)

    def start_follow(self, goal_id, reference, revision, requirement):
        r=self.record;state=self.navigation.snapshot()
        if self.navigation.scope=='manual' or not callable(getattr(self.navigation,'start_follow',None)):
            return dict(status='refused',reason='follow_not_supported_by_adapter')
        if (r['context_lost'] or not r['connected'] or state.get('input_stale') or
                state.get('localization_status','valid')!='valid'):
            return dict(status='refused',reason='navigation_context_unavailable')
        if r['handle'] is not None and not (self.navigation.scope=='live' and r['status']=='held'):
            return dict(status='refused',reason='cancel_and_confirm_existing_call_first')
        route=self.navigation.check_route({})
        if not route.get('proposal_allowed'):
            return dict(status='refused',reason=route.get('reason') or 'navigation_context_unavailable')
        from perception.environment.context import fingerprint
        r['goal_revision']+=1
        handle=dict(task_id=self.task_id,plan_revision=revision,goal_revision=r['goal_revision'],
            call_id=uuid.uuid4().hex,catalog_fingerprint=self.catalog.fingerprint,goal_id=goal_id,
            task_goal_binding=fingerprint(requirement),view_id=None)
        previous={key:copy.deepcopy(r[key]) for key in ('status','handle','place_id','goal_id','selected_pose','reason','stage_type','follow_ready','search_deadline_s')}
        r.update(status='submitting',handle=handle,place_id=None,goal_id=goal_id,selected_pose=None,reason=None,
                 stage_type='follow',follow_ready=False,search_deadline_s=None)
        r['attempts'].append(dict(goal_id=goal_id,place_id=None,revision=revision,status='active',call_id=handle['call_id']))
        self.last_stamp=None;self._emit('follow_intent')
        try:
            ack=self.navigation.start_follow(reference,handle)
            if ack.get('accepted') is not True or ack.get('handle')!=handle:raise ValueError('Follow ACK identity invalid')
        except IntentRejected as error:
            r.update(previous);self._mark_attempt(handle,'refused');self._emit('follow_refused')
            return dict(status='refused',reason=str(error))
        except (OSError,EOFError,ValueError) as error:
            r['reason']='submission_uncertain_'+type(error).__name__;self.cancel(reason=r['reason'])
            return dict(status='uncertain',handle=handle,reason=r['reason'])
        r['status']='shadow_proposed' if self.navigation.scope=='shadow' else 'running'
        self._emit('follow_ack')
        return dict(status='accepted',handle=copy.deepcopy(handle),scope=self.navigation.scope,execution_completed=False)

    def finish_follow(self, semantic):
        r=self.record;handle=r['handle'];state=self.navigation.snapshot()
        if (r['stage_type']!='follow' or handle is None or r['status'] not in ('running','blocked','shadow_proposed') or
                (state.get('follow_status') or {}).get('status')!='running' or state.get('input_stale') or
                state.get('status') not in ('running','proposed')):
            return dict(status='refused',reason='follow_not_running')
        proof=dict(goal_id=r['goal_id'],place_id=None,handle=copy.deepcopy(handle),stamp_s=state['stamp_s'],
            scope=self.navigation.scope,source=state['source'],context=spatial_context(state),stopped=True,
            execution_feedback=False,physical_acceptance=False,evidence_kind='follow_condition_and_stop_observation',
            semantic=copy.deepcopy(semantic))
        r.update(status='holding',reason='follow_finish_requested',pending_completion=proof)
        self._emit('follow_finish_intent')
        try:self.navigation.finish_follow(handle)
        except (OSError,EOFError,ValueError) as error:
            r['pending_completion']=None;self.cancel(reason='follow_finish_unconfirmed_'+type(error).__name__)
        return dict(status='stopping',handle=copy.deepcopy(handle),execution_completed=False)

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

    def lose_follow(self, reason):
        r=self.record
        if r['handle'] is None:return dict(status='idle',execution_completed=False)
        if not callable(getattr(self.navigation,'lose_follow',None)):
            return self.cancel(reason=reason)
        r.update(status='stopping',reason=reason,pending_completion=None)
        self._emit('follow_loss_intent')
        try:self.navigation.lose_follow(r['handle'],reason)
        except (OSError,EOFError,ValueError) as error:
            self.emit('navigation_stop_unconfirmed',dict(error_type=type(error).__name__,automatically_retried=False))
        return dict(status='stopping',handle=copy.deepcopy(r['handle']),execution_completed=False)

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
        follow=(feedback.get('follow_status') or {}) if r['stage_type']=='follow' else {}
        if (r['handle'] is not None and follow.get('handle')==r['handle'] and follow.get('status')=='failed' and
                r['status']!='stopping'):
            r.update(status='stopping',reason=follow.get('reason') or 'follow_failed',pending_completion=None)
            self._emit('follow_failed_stopping')
        if (self.navigation.scope == 'live' and
                (feedback.get('chassis_state') != 'armed' or feedback.get('chassis_status_fresh') is not True) and
                not r['context_lost']):
            r.update(context_lost=True, reason='chassis_disarmed_or_status_unavailable')
            if r['handle'] is not None and r['status'] != 'stopping':
                self.cancel(reason=r['reason'])
            self._emit('navigation_interrupted')
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
        if (r.get('search_deadline_s') is not None and now>r['search_deadline_s'] and
                r['status'] not in ('stopping','holding','held')):
            self.cancel(reason='follow_search_timeout')
            return False
        if (r['stage_type']=='follow' and not r['follow_ready'] and follow.get('status')=='running' and
                feedback.get('status') in ('running','proposed')):
            r['follow_ready']=True;self._emit('follow_started');return True
        if self.navigation.scope in ('live','shadow') and feedback.get('status') == 'preempted':
            r.update(status='interrupted', handle=None, context_lost=True, reason='operator_goal_preempted_agent')
            self._mark_attempt(handle, 'interrupted')
            self._emit('navigation_interrupted')
            return True
        if self.navigation.scope == 'live' and feedback.get('status') == 'interrupted' and r['status'] != 'stopping':
            self.cancel(reason='chassis_disarmed_or_status_unavailable')
            return False
        if feedback.get('collision') and r['status'] != 'stopping':
            self.cancel(reason='simulation_collision')
            return False
        if r['status'] == 'held':
            return False
        if r['status'] == 'holding':
            if ((feedback.get('stop_observation') is True or self.navigation.scope=='shadow' and feedback.get('proposal_cancelled') is True)
                    and feedback.get('status') == 'stopped'):
                proof = r.get('pending_completion')
                if proof is not None:
                    if r['stage_type']=='follow':proof['stop_stamp_s']=stamp
                    r['completed'].append(proof)
                    r.update(status='held', pending_completion=None, reason=None)
                    self._mark_attempt(handle, 'succeeded')
                    self._emit('navigation_completed')
                    return True
            return False
        if r['status'] == 'stopping':
            stopped = (feedback.get('stopped') and feedback.get('execution_feedback') is True if self.navigation.scope == 'simulation'
                       else feedback.get('stop_observation') is True if self.navigation.scope == 'live'
                       else feedback.get('proposal_cancelled') is True)
            if stopped and feedback.get('status') in ('stopped', 'failed'):
                failed=bool(follow.get('status')=='failed')
                r.update(status='failed' if failed else 'cancelled', handle=None, pending_completion=None)
                self._mark_attempt(handle, 'failed' if failed else 'cancelled')
                self._emit('navigation_cancel_confirmed')
                return True
            return False
        if self.navigation.scope == 'live':
            if feedback.get('planning_reason') == 'no_admissible_velocity':
                if self.blocked_since is None:self.blocked_since=stamp
                if not self.blocked_notified and stamp-self.blocked_since >= 10.:
                    self.blocked_notified=True
                    r.update(status='blocked',reason='no_admissible_velocity')
                    self._emit('navigation_blocked')
                    return True
            else:
                self.blocked_since=None
                if self.blocked_notified:
                    self.blocked_notified=False
                    r.update(status='running',reason=None)
                    self._emit('navigation_unblocked')
                    return True
        goal = feedback.get('goal') or {}
        arrived = ((self.navigation.scope == 'simulation' and feedback.get('execution_feedback') is True and goal.get('task_completed')) or
                   (self.navigation.scope == 'live' and feedback.get('arrival_observation') is True) or
                   (self.navigation.scope == 'manual' and feedback.get('manual_arrival_observation') is True))
        if (feedback.get('status') == 'arrived' and arrived and feedback.get('stopped') and
                goal.get('task_id') == self.task_id and
                goal.get('goal_revision') == handle['goal_revision']):
            proof = dict(goal_id=r['goal_id'], place_id=r['place_id'], handle=copy.deepcopy(handle), stamp_s=stamp,
                         scope=self.navigation.scope, source=feedback['source'], context=current,
                         pose=copy.deepcopy(feedback.get('origin_pose') if self.navigation.scope in ('live','manual') else feedback.get('pose')),
                         frame_id=feedback.get('origin_frame_id') if self.navigation.scope in ('live','manual') else feedback.get('frame_id'),
                         execution_feedback=self.navigation.scope == 'simulation', stopped=True,
                         physical_acceptance=False, evidence_kind=('operator_confirmation_and_localization_estimate' if self.navigation.scope == 'manual'
                             else 'localization_estimate' if self.navigation.scope == 'live' else 'simulation'))
            if self.navigation.scope == 'manual':
                proof['operator_confirmation']=copy.deepcopy(feedback.get('manual_confirmation'))
            if feedback.get('route_start_stamp_s') is not None:
                proof['route_start_stamp_s']=feedback['route_start_stamp_s']
            if self.navigation.scope == 'live':
                r['pending_completion'] = proof
                r.update(status='holding',reason='goal_arrival_hold')
                self._emit('navigation_hold_intent')
                try:
                    self.navigation.hold(handle)
                except (OSError, EOFError, ValueError) as error:
                    r['pending_completion'] = None
                    self.cancel(reason='arrival_hold_unconfirmed_'+type(error).__name__)
                return False
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
