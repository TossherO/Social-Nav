"""One model -> tool -> result loop, woken by tool events or user input."""
import copy
import math
import time
import jsonschema
from .async_model import AsyncModel
from .session import TERMINAL
from ..tools.framework import Reply
from perception.environment.context import fingerprint


class AgentLoop:
    def __init__(self, session, work, *, max_requests=32):
        self.session, self.work = session, work
        self.model = AsyncModel()
        self.requests, self.max_requests = 0, max_requests
        self.exit_reason = None
        self.route_refusal = None
        self.navigation_refusal = None

    @property
    def state(self):
        return self.session.state

    def event(self, name, text=None):
        self.session.event(name, text)
        self.model.invalidate()
        self.route_refusal = None
        self.navigation_refusal = None

    def repeated_route_refusal(self, value, output):
        if value['tool']!='navigate_route':
            return False
        if self.session.navigation.scope not in ('manual','simulation') or self.session.skill.record['handle'] is not None:
            self.route_refusal=None
            return False
        if output.get('status') not in ('refused','unavailable'):
            self.route_refusal=None
            return False
        args={k:copy.deepcopy(v) for k,v in value['arguments'].items() if k!='view_id'}
        reason=output.get('reason')
        key=fingerprint(dict(revision=self.session.state['revision'],arguments=args,reason=reason))
        previous=self.route_refusal
        count=previous[1]+1 if previous is not None and previous[0]==key else 1
        self.route_refusal=(key,count)
        if count<3:return False
        self.session.store.event('route_repeated_refusal',dict(goal_id=args.get('goal_id'),reason=reason,count=count,
            action='await_user_before_retry',automatically_retried=False))
        return True

    def repeated_navigation_refusal(self, value, output):
        """Stop repeating the same rejected target; a new view ID is not a new target."""
        if (value['tool']!='navigate' or self.session.skill.record['handle'] is not None or
                output.get('status') not in ('refused','unavailable')):
            self.navigation_refusal=None
            return False
        args={k:copy.deepcopy(v) for k,v in value['arguments'].items() if k!='view_id'}
        reason=output.get('reason')
        key=fingerprint(dict(revision=self.session.state['revision'],arguments=args,reason=reason))
        previous=self.navigation_refusal
        count=previous[1]+1 if previous is not None and previous[0]==key else 1
        self.navigation_refusal=(key,count)
        if count<3:return False
        self.session.store.event('navigation_repeated_refusal',dict(goal_id=args.get('goal_id'),reason=reason,count=count,
            action='await_user_before_retry',automatically_retried=False))
        return True

    def run(self, *, console=None, driver=None, emit=None, on_tick=None, timeout_s=None):
        """Run one task; optional drivers supply inputs and test-time pacing.

        driver.poll(actor) may supply new input or return an exit reason;
        driver.interval(actor) controls pacing. Neither defines model decisions.
        """
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('Positive finite wall-time budget required')
        deadline = None if timeout_s is None else time.monotonic()+timeout_s
        emit = emit or (lambda value: None)
        previous = None
        self.exit_reason = 'run_error'
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    self.exit_reason = 'wall_time_limit'
                    break
                if self.state['status'] in TERMINAL:
                    self.exit_reason = 'task_terminal'
                    break
                if console is not None:
                    console.poll(self, emit)
                self.tick()
                if on_tick is not None:
                    on_tick(self)
                state = self.state
                key = (state['status'], state['revision'], self.session.skill.record['status'],state.get('assistant_text'))
                if key != previous:
                    emit(dict(status=key[0], revision=key[1], navigation=key[2], reason=state['reason'],assistant_text=key[3]))
                    previous = key
                if state['status'] in TERMINAL:
                    self.exit_reason = 'task_terminal'
                    break
                reason = None if driver is None else driver.poll(self)
                if reason is not None:
                    self.exit_reason = reason
                    break
                if console is None and state['status'] in ('waiting_user', 'needs_clarification', 'paused'):
                    self.exit_reason = state['status']
                    break
                interval = .05 if driver is None else driver.interval(self)
                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.exit_reason = 'keyboard_interrupt'
        finally:
            self.model.invalidate()
            held_follow = (self.session.navigation.scope == 'live' and
                           self.session.skill.record['status'] == 'failed' and
                           getattr(self.session.navigation, 'handle', None) is not None)
            if self.session.skill.record['handle'] is not None or self.state['status'] == 'ready' or held_follow:
                self.stop(reason=self.exit_reason)
            self.session._save('agent_run_finished', dict(reason=self.exit_reason))
        return self.state

    def stop(self, *, reason='run_exit', timeout_s=2.):
        """Invalidate model work and reconcile cancellation, retaining uncertainty."""
        session = self.session
        self.model.invalidate()
        if session.state['status'] not in TERMINAL:
            if session.skill.record['status'] == 'stopping':
                # Keep an outstanding cancel and the user's original intent;
                # a timeout or lost ACK must not resend it during shutdown.
                if session.state['status'] != 'stopping':
                    session.state.update(status='stopping', stop_target='cancelled', reason=reason)
                    session._save('task_stop_requested')
            else:
                session.stop('cancelled', reason)
        deadline = time.monotonic()+timeout_s
        while session.skill.record['handle'] is not None and time.monotonic() < deadline:
            session.tick()
            if session.navigation.scope in ('shadow','live'):
                time.sleep(.02)
        session._save('session_checkpoint', dict(stop_confirmed=session.skill.record['handle'] is None))

    def tick(self):
        s = self.session.tick()
        result = self.model.poll()
        if result is not None:
            valid = not result['stale'] and self.session.valid_context(result['context'])
            if not valid:
                self.session.store.event('model_result_discarded', dict(context=result['context'], reason='cancelled_or_superseded'))
                if s['status'] == 'ready' and self.session.observation is not None:
                    self.session.observation = None
                    current = self.session.context()
                    if (result['context'].get('handle') != current.get('handle') or
                            result['context'].get('conversation_revision') != current.get('conversation_revision')):
                        # A new event already supplies a reason to reconsider;
                        # prompt preparation will request current observations.
                        self.session._save('discard_observation_after_context_change')
                    else:
                        self.session.wait_for_event('observation')
                        self.session._save('await_fresh_observation')
                if self.session.spatial_view is not None and not self.session.spatial_view.current(self.session.navigation.snapshot()['stamp_s']):
                    self.session.spatial_view = None
                    self.session._save('spatial_view_expired')
            elif not result['ok']:
                self.session.stop('needs_clarification', 'model_'+result['error_type'])
                self.session._save('model_failed', dict(automatically_retried=False))
            else:
                reply = result['value']
                try:
                    if not isinstance(reply,Reply):
                        raise ValueError('Model worker must return a checked native reply')
                    self.session.conversation.validate(reply)
                    s['assistant_text'] = reply.message.get('content')
                    if reply.action is None:
                        self.session.conversation.commit(reply)
                        unregistered=not self.session.task_goals.goals
                        reason='model_text_without_task_registration' if unregistered else s['assistant_text']
                        self.session.stop('needs_clarification',reason)
                        self.session._save('assistant_message',dict(message=reply.message,task_success=False,
                            unregistered_task=unregistered,automatically_retried=False))
                        return s
                    value = reply.action
                    # Every proposed target is checked with current geometry,
                    # independently of how long the model took to respond.
                    if value['tool'] == 'navigate':
                        output = self.session.revalidate_navigation(value, result['context'])
                        if output.get('status') == 'refused' and not output.get('revalidation'):
                            self.session._history('tool', dict(call=value, result=output))
                    elif value['tool'] == 'check_candidate':
                        output = self.session.revalidate_candidate(value, result['context'])
                        if output.get('status') == 'refused' and not output.get('revalidation'):
                            self.session._history('tool', dict(call=value, result=output))
                    elif value['tool'] == 'navigate_route':
                        output = self.session.revalidate_route(value, result['context'])
                        if output.get('status') == 'refused' and not output.get('revalidation'):
                            self.session._history('tool', dict(call=value, result=output))
                    else:
                        output = self.session.call(value['tool'], _context=result['context'], **value['arguments'])
                    keyframe = output.get('keyframe') if value['tool'] in ('save_keyframe','read_keyframe') and output.get('status') in ('saved','available') else None
                    self.session.conversation.commit(reply,output,keyframe=keyframe)
                    if self.session.memory is not None and 'observation_id' in result['context']:
                        self.session.memory.remember(dict(kind='historical_model_tool_choice', call=value,
                            result_status=output.get('status')), result['context'],
                            place_id=value['arguments'].get('place_id') or self.session.skill.record['place_id'] or '',
                            clock_domain=result['context']['clock_domain'], map_binding=result['context']['map_binding'])
                    source_stamp=result['context'].get('source_stamp_s')
                    current_stamp=self.session.navigation.snapshot().get('stamp_s')
                    age=(current_stamp-source_stamp if type(current_stamp) in (int,float) and
                         type(source_stamp) in (int,float) and 0<=current_stamp-source_stamp else None)
                    self.session._save('model_action_applied', dict(call=value,result=output,context=result['context'],
                                                                  source_age_s=age))
                    if self.repeated_route_refusal(value,output):
                        self.session.stop('needs_clarification','repeated_route_refusal')
                        return s
                    if self.repeated_navigation_refusal(value,output):
                        self.session.stop('needs_clarification','repeated_navigation_refusal')
                        return s
                    if ((value['tool'] in ('navigate', 'navigate_route','start_local_task') and output.get('status')=='accepted' or
                         value['tool']=='finish_local_task' and output.get('status')=='stopping') and
                            self.session.skill.record['handle'] is not None):
                        # Do not ask the model to restate that it is waiting:
                        # a text-only reply has ambiguous intent and otherwise
                        # triggers clarification and cancels the active goal.
                        self.session.wait_for_event('navigation')
                        self.session._save('await_navigation_after_acceptance')
                except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
                    self.session.stop('needs_clarification', 'model_protocol_rejected')
                    self.session._save('model_protocol_rejected', dict(error_type=type(error).__name__))
        if self.model.busy or s['status'] != 'ready' or self.session.skill.record['context_lost'] or not self.session.skill.record['connected']:
            return s
        stage=next((g for g in self.session.task_goals.goals if g['status']!='completed'),None)
        if (self.session.navigation.scope=='manual' and len(self.session.task_goals.floor_catalogs)>1 and
                stage is not None and stage['requirement'].get('stage_type')=='elevator'):
            # The manual workflow starts its independent camera monitor in on_tick.
            return s
        if self.requests >= self.max_requests:
            self.session.stop('needs_clarification', 'model_request_limit')
            return s
        now = self.session.navigation.snapshot()['stamp_s']
        if self.session.observation is not None and not 0 <= now-self.session.observation.record['stamp_s'] <= 3:
            self.session.observation = None
            self.session._save('observation_expired_before_request')
        if self.session.spatial_view is not None and not self.session.spatial_view.current(now):
            self.session.spatial_view = None
            self.session._save('spatial_view_expired_before_request')
        payload = self.session.prompt()
        context = self.session.context()
        observation = copy.deepcopy(self.session.observation)
        spatial_view = copy.deepcopy(self.session.spatial_view)
        self.model.submit('tool', context, lambda: self.work(payload, observation, spatial_view))
        self.requests += 1
        return s

    def close(self):
        value = self.model.close()
        self.session.store.event('model_worker_closed', value)
        return value
