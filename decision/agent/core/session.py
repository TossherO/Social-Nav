"""One task record, tool dispatch, user events and model context."""
import copy
import uuid
from .memory import EvidenceMemory
from .tasks import TaskGoals
from .conversation import Conversation
from ..tools.framework import TOOL_SPECS, validate_call
from ..tools.navigation import NavigationSkill, spatial_context
from ..tools.spatial import SpatialView
from perception.environment.context import fingerprint

TERMINAL = {'succeeded', 'failed', 'cancelled'}


class AgentSession:
    def __init__(self, instruction, catalog, navigation, store, *, use_memory=True):
        self.store, self.navigation = store, navigation
        self.memory = EvidenceMemory() if use_memory else None
        self.observation = None
        self.spatial_view = None
        self._applying_context = None
        task_id = uuid.uuid4().hex
        self.catalog = catalog.bind_start(navigation.snapshot(), task_id)
        if hasattr(navigation, 'catalog'):
            navigation.catalog = self.catalog
        self.task_goals = TaskGoals(self.catalog)
        self.conversation = Conversation(instruction,self.catalog.prompt(),store.event)
        self.state = dict(task_id=task_id, revision=1, instruction=instruction, status='ready', reason=None,
                          scope=navigation.scope, history=[], clarifications=[], completed_revisions=[], waiting_binding=None,
                          stop_target=None, pending_instruction=None, task_start=copy.deepcopy(self.catalog.places.get('mission_start')),
                          catalog_fingerprint=self.catalog.fingerprint, assistant_text=None)
        self.skill = NavigationSkill(navigation, self.catalog, task_id, self._save)
        self.registry = {name: getattr(self, name) for name in TOOL_SPECS}
        self._save('task_created')

    def _save(self, kind, value=None):
        self.state['navigation'] = copy.deepcopy(self.skill.record)
        self.state['evidence_memory'] = [] if self.memory is None else self.memory.snapshot()
        self.state['task_goals'] = self.task_goals.snapshot()
        self.state['spatial_view'] = None if self.spatial_view is None else copy.deepcopy(self.spatial_view.record)
        self.state['conversation'] = self.conversation.snapshot()
        self.store.save(self.state, kind, {} if value is None else value)

    def _history(self, kind, value):
        self.state['history'].append(dict(kind=kind, value=copy.deepcopy(value), revision=self.state['revision']))
        self.state['history'] = self.state['history'][-24:]

    def context(self):
        value = dict(task_id=self.state['task_id'], revision=self.state['revision'],
                     handle=copy.deepcopy(self.skill.record['handle']), **spatial_context(self.navigation.snapshot()))
        if self.observation is not None:
            r = self.observation.record
            value.update(observation_id=r['observation_id'], observation_binding=r['input_fingerprint'], source_stamp_s=r['stamp_s'])
            value['input_view_binding'] = fingerprint(self.observation.prompt())
        if self.spatial_view is not None:
            self.spatial_view.verify()
            value['spatial_view_id'] = self.spatial_view.record['view_id']
        value['task_requirements'] = fingerprint([g['requirement'] for g in self.task_goals.goals])
        value['conversation_revision'] = self.conversation.revision
        return value

    def valid_context(self, context):
        if context != self.context() or self.state['status'] != 'ready' or self.skill.record['context_lost']:
            return False
        current = self.navigation.snapshot()
        if not self.skill.record['connected'] or current.get('input_stale'):
            return False
        if self.spatial_view is not None and not self.spatial_view.current(current['stamp_s']):
            return False
        return (self.observation is None or
                0 <= current['stamp_s']-self.observation.record['stamp_s'] <= 3 and
                current.get('localization_status', 'valid') == 'valid')

    def call(self, tool_name, *, _context=None, **arguments):
        call = validate_call(dict(tool=tool_name, arguments=arguments))
        if self.state['status'] != 'ready':
            raise ValueError('Task is not ready for another model action')
        self.store.event('tool_intent', call)
        self._applying_context = _context
        try:
            value = self.registry[tool_name](**arguments)
        except (OSError, EOFError, ValueError, KeyError) as error:
            value = dict(status='unavailable', error_type=type(error).__name__,
                         reason=str(error)[:240] if isinstance(error, ValueError) else type(error).__name__,
                         automatically_retried=False)
        finally:
            self._applying_context = None
        self._history('tool', dict(call=call, result=value))
        self._save('tool_result', dict(call=call, value=value))
        return value

    def lookup_places(self, name):
        fields = ('place_id', 'name', 'aliases', 'pose', 'reference', 'status')
        return dict(status='available', places=[{k: p[k] for k in fields if k in p} for p in self.catalog.lookup(name)])

    def record_task(self, goals):
        if self.skill.record['handle'] is not None:
            return dict(status='refused', reason='active_navigation')
        return dict(status='recorded', goals=self.task_goals.record(goals),
                    original_instruction=self.state['instruction'], automatically_executed=False)

    def query_map(self, bounds=None):
        self.catalog.require_complete()
        inputs = self.navigation.spatial_inputs()
        self.spatial_view = SpatialView.build(self.catalog,inputs,bounds)
        self.store.spatial_view(self.spatial_view)
        return dict(status='available', view=self.spatial_view.record, target_candidates_generated=False)

    def observe_scene(self):
        observation = self.navigation.observe() if hasattr(self.navigation, 'observe') else None
        current = self.navigation.snapshot()
        if observation is None:
            return dict(status='unavailable', reason='no_current_rgb', observation_id=None)
        observation.verify()
        r = observation.record
        if (r['rgb']['status'] != 'valid' or r['clock_domain'] != current['clock_domain'] or
                r['local_identity'] != spatial_context(current)['local_identity'] or
                current.get('map_binding') is not None and r['map_binding'] != current['map_binding'] or
                current.get('input_stale') or current.get('localization_status', 'valid') != 'valid' or
                not 0 <= current['stamp_s']-r['stamp_s'] <= 3):
            return dict(status='unavailable', reason='observation_not_current_or_localization_invalid')
        self.observation = observation
        return dict(status='available', observation=observation.prompt())

    def navigate(self, goal_id, pose, view_id, orientation_source):
        self.catalog.require_complete()
        now = self.navigation.snapshot()['stamp_s']
        if self.spatial_view is None or self.spatial_view.record['view_id'] != view_id or not self.spatial_view.current(now):
            return dict(status='refused', reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        selected = self.task_goals.select(goal_id,pose,orientation_source,view_id)
        if selected['status'] != 'selected':
            return selected
        result = self.skill.start(selected['destination'],self.state['revision'])
        if result['status'] in ('accepted', 'uncertain'):
            self.task_goals.current(goal_id)['selected']['handle'] = copy.deepcopy(result['handle'])
            self.spatial_view = self.observation = None
        return result

    def check_goal(self, goal_id, observation_id=None, semantic_result=None, evidence=None):
        goal = self.task_goals.current(goal_id)
        proofs = [p for p in self.skill.record['completed'] if p['goal_id'] == goal_id]
        selected = goal['selected'] or {}
        state = self.navigation.snapshot()
        if (self.skill.record['handle'] is not None or not proofs or
                proofs[-1]['handle'] != selected.get('handle') or
                proofs[-1]['handle']['plan_revision'] != self.state['revision'] or
                self.navigation.scope != 'simulation' or state.get('execution_feedback') is not True or
                not state.get('stopped') or state.get('input_stale') or
                state.get('localization_status','valid') != 'valid' or
                state.get('frame_id') != self.catalog.value['frame_id']):
            return dict(status='unproven', reason='matching_task_navigation_and_stop_not_proven')
        proof = proofs[-1]
        checks = self.task_goals.geometry(goal['requirement'], state.get('pose'))
        if not all(checks.values()):
            return dict(status='not_met', checks=checks)
        semantic = dict(status='not_required')
        conditions = goal['requirement'].get('semantic_conditions', [])
        if conditions:
            context = self._applying_context or {}
            observation = self.observation
            if (observation is None or observation_id != observation.record['observation_id'] or
                    context.get('observation_id') != observation_id or not self.valid_context(context) or
                    observation.record['stamp_s'] < proof['stamp_s'] or
                    semantic_result is None or not evidence):
                return dict(status='needs_semantic_evidence', checks=checks, conditions=conditions,
                            reason='read_a_fresh_post_arrival_rgb_then_judge_all_conditions')
            semantic = dict(status=semantic_result, evidence=evidence, observation_id=observation_id,
                            input_binding=observation.record['input_fingerprint'], stamp_s=observation.record['stamp_s'],
                            source='same_agent_visual_judgement_not_independent_ground_truth')
            if semantic_result != 'satisfied':
                return dict(status='unknown' if semantic_result == 'unknown' else 'not_met', checks=checks, semantic=semantic)
        self.task_goals.complete(goal_id,proof,checks,semantic)
        return dict(status='completed', goal_id=goal_id, checks=checks, semantic=semantic,
                    scope='simulation', real_robot_success=False)

    def get_navigation_state(self):
        return self.skill.snapshot()

    def cancel_navigation(self):
        result = self.skill.cancel()
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'waiting_navigation'
        return result

    def _observation_key(self):
        if hasattr(self.navigation, 'current'):
            r = self.navigation.current.record
        elif hasattr(self.navigation, 'value'):
            r = self.navigation.value.get('observation') or {}
        else:
            value = self.navigation.observe() if hasattr(self.navigation, 'observe') else None
            r = {} if value is None else value.record
        return r.get('input_fingerprint')

    def wait_for_event(self, event):
        if event == 'navigation':
            if self.skill.record['handle'] is None:
                return dict(status='available', navigation=self.skill.snapshot())
            self.state['status'] = 'waiting_navigation'
        elif event == 'observation':
            self.state.update(status='waiting_observation', waiting_binding=self._observation_key())
        else:
            f = self.navigation.snapshot()
            if self.skill.record['handle'] is not None or not (f.get('stopped') and f.get('execution_feedback') is True):
                return dict(status='refused', reason='parking_not_confirmed')
            self.state['status'] = 'waiting_user'
        return dict(status='waiting', event=event)

    def ask_user(self, question):
        if self.skill.record['handle'] is not None:
            return dict(status='refused', reason='cancel_and_confirm_navigation_first')
        self.state.update(status='needs_clarification', reason=question)
        return dict(status='waiting_user_answer', question=question)

    def finish_task(self, status, evidence):
        r = self.skill.record
        if status == 'failed':
            self.stop('failed', evidence)
            return dict(status=self.state['status'], reason=evidence)
        revision = self.state['revision']
        f = self.navigation.snapshot()
        if (self.navigation.scope != 'simulation' or r['handle'] is not None or r['context_lost'] or
                not r['connected'] or not self.task_goals.finished() or
                not f.get('stopped') or f.get('execution_feedback') is not True or f.get('input_stale') or
                f.get('localization_status', 'valid') != 'valid'):
            return dict(status='refused', reason='task_execution_not_proven',
                        uncompleted_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals if g['status']!='completed'])
        self.state.update(status='succeeded', reason=evidence)
        self.state['completed_revisions'].append(revision)
        return dict(status='succeeded', scope='simulation', real_robot_success=False,
                    completed_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals])

    def stop(self, target, reason):
        self.state.update(stop_target=target, reason=reason)
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'stopping'
            self.skill.cancel(reason=reason)
        else:
            self.state['status'] = target
        self._save('task_stop_requested')

    def event(self, name, text=None):
        s = self.state
        previous_revision = s['revision']
        if s['status'] in TERMINAL:
            raise ValueError('Task already terminal')
        if name in ('pause', 'cancel'):
            self.stop('paused' if name == 'pause' else 'cancelled', name)
        elif name == 'resume' and s['status'] == 'paused' and not self.skill.record['context_lost']:
            s.update(status='ready', reason=None)
            self._history('user', dict(event='resume'))
        elif name == 'continue' and s['status'] == 'waiting_user':
            s.update(status='ready', reason=None)
            self._history('user', dict(event='continue', source='user_event'))
        elif name == 'answer' and s['status'] == 'needs_clarification' and text and text.strip() and not self.skill.record['context_lost']:
            s['clarifications'].append(dict(question=s['reason'], answer=text.strip()))
            s['clarifications'] = s['clarifications'][-8:]
            s.update(status='ready', reason=None)
            self._history('user', dict(event='answer', text=text.strip()))
        elif name == 'replace' and text and text.strip() and not self.skill.record['context_lost']:
            s['pending_instruction'] = text.strip()
            self.stop('replacing', 'user_replaced_instruction')
        else:
            raise ValueError('Event does not apply to current task')
        self._apply_replacement()
        if s['revision']==previous_revision:
            self.conversation.notice('user_event',event=name,text=text)
        self._save('user_event', dict(name=name, text=text))

    def _apply_replacement(self):
        s = self.state
        if s['status'] == 'replacing':
            s.update(instruction=s['pending_instruction'], pending_instruction=None, revision=s['revision']+1,
                     status='ready', reason=None, clarifications=[])
            self.observation = None
            self.spatial_view = None
            self.task_goals = TaskGoals(self.catalog)
            self.conversation = Conversation(s['instruction'],self.catalog.prompt(),self.store.event)
            s['assistant_text'] = None
            self._history('user', dict(event='replacement', instruction=s['instruction']))

    def tick(self):
        s = self.state
        changed = self.skill.tick()
        r = self.skill.record
        if not r['connected'] and s['status'] == 'ready':
            s.update(status='waiting_navigation' if r['handle'] else 'needs_clarification', reason='navigation_connection_lost')
            self._save('task_connection_lost')
        if r['context_lost'] and s['status'] not in TERMINAL and s.get('reason') != 'spatial_context_changed':
            if self.memory is not None:
                self.memory.invalidate(**spatial_context(self.navigation.snapshot()))
            # Navigation has already requested cancellation. Never resend it.
            target = 'cancelled' if s.get('stop_target') == 'cancelled' else 'needs_clarification'
            s.update(status='stopping' if r['handle'] else target,
                     stop_target=target, reason='spatial_context_changed')
            self._save('task_context_lost')
        if s['status'] == 'stopping' and r['handle'] is None:
            s['status'] = s['stop_target']
            self._apply_replacement()
            self._save('task_stop_confirmed')
        elif s['status'] == 'waiting_navigation' and (changed or r['handle'] is None):
            s['status'] = 'ready'
            self._history('navigation', self.skill.snapshot())
            self.conversation.notice('navigation_event',navigation=self.navigation_summary())
            self._save('navigation_event')
        elif s['status'] == 'waiting_observation' and self._observation_key() != s['waiting_binding']:
            s['status'] = 'ready'
            self.observation = None
            self._history('observation_event', dict(binding=self._observation_key()))
            self.conversation.notice('observation_event',binding=self._observation_key())
            self._save('observation_event')
        elif changed:
            self._history('navigation', self.skill.snapshot())
            self.conversation.notice('navigation_event',navigation=self.navigation_summary())
            self._save('navigation_event')
        return s

    def navigation_summary(self):
        record = self.skill.record
        feedback = self.navigation.snapshot()
        fields = ('stamp_s','clock_domain','source','session_id','generation','frame_id','pose','velocity',
                  'origin_pose','origin_frame_id','map_binding','localization_status','input_stale','stopped','execution_feedback')
        return dict({key:copy.deepcopy(record[key]) for key in
                    ('status','goal_id','handle','selected_pose','reason','context_lost','connected')},
                    feedback={key:copy.deepcopy(feedback[key]) for key in fields if key in feedback},
                    last_completion=copy.deepcopy(record['completed'][-1:]))

    def prompt(self):
        s, now = self.state, self.navigation.snapshot()
        goals = self.task_goals.snapshot()
        current = dict(task_id=s['task_id'],revision=s['revision'],status=s['status'],reason=s['reason'],
            clarifications=copy.deepcopy(s['clarifications']),
            task_goals=[dict(requirement=g['requirement'],status=g['status']) for g in goals],
            next_goal_id=next((g['requirement']['goal_id'] for g in goals if g['status']!='completed'),None),
            navigation=self.navigation_summary(),
            map=dict(condition=self.catalog.condition,frame_id=self.catalog.value['frame_id'],
                     catalog_fingerprint=self.catalog.fingerprint),
            evidence_memory=[] if self.memory is None else self.memory.recall(
                place_ids=[*self.catalog.places,''],now=now['stamp_s'],clock_domain=now['clock_domain']))
        return dict(messages=self.conversation.messages(),current_state=current)
