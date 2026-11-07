"""One model -> tool -> result loop, woken by tool events or user input."""
import copy
import math
import time
import jsonschema
from .async_model import AsyncModel
from .session import TERMINAL
from ..tools.framework import Reply


class AgentLoop:
    def __init__(self, session, work, *, max_requests=32):
        self.session, self.work = session, work
        self.model = AsyncModel()
        self.requests, self.max_requests = 0, max_requests
        self.exit_reason = None

    @property
    def state(self):
        return self.session.state

    def event(self, name, text=None):
        self.session.event(name, text)
        self.model.invalidate()

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
            if self.session.skill.record['handle'] is not None or self.state['status'] == 'ready':
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
            if session.navigation.scope == 'shadow':
                time.sleep(.02)
        session._save('session_checkpoint', dict(stop_confirmed=session.skill.record['handle'] is None))

    def tick(self):
        s = self.session.tick()
        result = self.model.poll()
        if result is not None:
            if result['stale'] or not self.session.valid_context(result['context']):
                self.session.store.event('model_result_discarded', dict(context=result['context'], reason='cancelled_expired_or_superseded'))
                if s['status'] == 'ready' and self.session.observation is not None:
                    self.session.observation = None
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
                        self.session.stop('needs_clarification',s['assistant_text'])
                        self.session._save('assistant_message',dict(message=reply.message,task_success=False))
                        return s
                    value = reply.action
                    output = self.session.call(value['tool'], _context=result['context'], **value['arguments'])
                    keyframe = output.get('keyframe') if value['tool'] in ('save_keyframe','read_keyframe') and output.get('status') in ('saved','available') else None
                    self.session.conversation.commit(reply,output,keyframe=keyframe)
                    if self.session.memory is not None and 'observation_id' in result['context']:
                        self.session.memory.remember(dict(kind='historical_model_tool_choice', call=value,
                            result_status=output.get('status')), result['context'],
                            place_id=value['arguments'].get('place_id') or self.session.skill.record['place_id'] or '',
                            clock_domain=result['context']['clock_domain'], map_binding=result['context']['map_binding'])
                    self.session._save('model_action_applied', dict(call=value, result=output, context=result['context']))
                except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
                    self.session.stop('needs_clarification', 'model_protocol_rejected')
                    self.session._save('model_protocol_rejected', dict(error_type=type(error).__name__))
        if self.model.busy or s['status'] != 'ready' or self.session.skill.record['context_lost'] or not self.session.skill.record['connected']:
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
