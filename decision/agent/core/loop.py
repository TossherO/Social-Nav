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
        self.refusals = {}
        self.before_model = None
        self.before_request = None

    @property
    def state(self):
        return self.session.state

    def event(self, name, text=None):
        self.session.event(name, text)
        self.model.invalidate()
        self.refusals.clear()

    def repeated_refusal(self, value, output):
        tool=value['tool']
        if tool not in ('navigate','navigate_route'):
            return None
        name='route' if tool=='navigate_route' else 'navigation'
        if (self.session.skill.record['handle'] is not None or output.get('status') not in ('refused','unavailable')
                or tool=='navigate_route' and self.session.navigation.scope not in ('manual','simulation')):
            self.refusals.pop(tool,None)
            return None
        args={k:v for k,v in value['arguments'].items() if k!='view_id'}
        reason=output.get('reason')
        key=fingerprint(dict(revision=self.session.state['revision'],arguments=args,reason=reason))
        previous,count=self.refusals.get(tool,(None,0))
        count=count+1 if previous==key else 1
        self.refusals[tool]=(key,count)
        if count<3:return None
        self.session.store.event(name+'_repeated_refusal',dict(goal_id=args.get('goal_id'),reason=reason,
            count=count,action='await_user_before_retry',automatically_retried=False))
        return 'repeated_'+name+'_refusal'

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
            # Live run cleanup releases the final confirmed hold and disarms once.
            final_hold=(self.state['status']=='succeeded' and self.session.navigation.scope=='live' and
                        self.session.skill.record['status']=='held')
            if (self.session.skill.record['handle'] is not None and not final_hold or self.state['status'] == 'ready' or held_follow
                    or self.session.seating_pending is not None or
                    self.state.get('elevator_active') and self.state['status'] not in TERMINAL):
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
        if self.before_model is not None:self.before_model(self)
        result = self.model.poll()
        if result is not None:
            valid = not result['stale'] and self.session.valid_context(result['context'])
            if not valid:
                self.session.store.event('model_result_discarded', dict(context=result['context'], reason='cancelled_or_superseded'))
                if (result['context'].get('conditional_review_cycle') and s['status']=='ready' and
                        result['context'].get('conditional_review_epoch')!=self.session.monitor.region_epoch and
                        self.session.skill.record['handle'] is not None):
                    self.session.observation=None
                    s.update(status='waiting_monitor',waiting_binding=None)
                    self.session._save('conditional_review_superseded')
                    return s
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
                    transient = bool(s.get('elevator_active'))
                    s['assistant_text'] = None if transient else reply.message.get('content')
                    if reply.argument_error is not None:
                        if transient:
                            self.session.elevator.feedback=reply.argument_error
                            self.session.conversation.consume_transient(reply)
                            self.session.wait_for_event('monitor')
                            return s
                        self.session.conversation.commit(reply,reply.argument_error)
                        self.session._save('tool_arguments_rejected',reply.argument_error)
                        return s
                    if reply.action is None:
                        if transient:
                            self.session.elevator.feedback=dict(status='refused',reason='native_tool_required')
                            self.session.wait_for_event('monitor')
                            return s
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
                    if not transient and value['tool'] in ('navigate','navigate_route','check_candidate'):
                        output = self.session.revalidate_spatial(value,result['context'])
                        if output.get('status') == 'refused' and not output.get('revalidation'):
                            self.session._history('tool',dict(call=value,result=output))
                    else:
                        output = self.session.call(value['tool'], _context=result['context'], **value['arguments'])
                    if transient:
                        self.session.conversation.consume_transient(reply)
                        self.session.elevator.feedback=(None if value['tool']=='record_elevator_observation' and
                            output.get('status')=='awaiting_destination_localization' else output)
                        if s['status']=='ready':
                            self.session.wait_for_event('monitor')
                        return s
                    keyframe = output.get('keyframe') if value['tool'] in ('save_keyframe','read_keyframe') and output.get('status') in ('saved','available') else None
                    handoff = value['tool'] == 'record_task' and output.get('status') == 'recorded'
                    self.session.conversation.commit(reply,output,keyframe=keyframe,audit_only=handoff)
                    if not handoff and self.session.memory is not None and 'observation_id' in result['context']:
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
                    refusal=self.repeated_refusal(value,output)
                    if refusal:
                        self.session.stop('needs_clarification',refusal)
                        return s
                    if ((value['tool'] in ('navigate', 'navigate_route','start_local_task') and output.get('status')=='accepted' or
                         value['tool'] in ('complete_subtask','finish_local_task') and output.get('status')=='stopping') and
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
        if self.requests >= self.max_requests:
            self.session.stop('needs_clarification', 'model_request_limit')
            return s
        if self.before_request is not None and not self.before_request(self):
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
