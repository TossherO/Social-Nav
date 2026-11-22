"""One task record, tool dispatch, user events and model context."""
import copy
import math
from time import perf_counter
import uuid
from .memory import EvidenceMemory
from .tasks import TaskGoals, goal_requires_heading, next_unfinished_goal, user_quoted
from .conversation import Conversation
from .keyframes import Keyframes
from .trajectory import Trajectory
from .monitor import TaskMonitor
from .completion import stage_evidence
from .context import robot_context
from .model_projection import ModelReferenceNames
from .store import encode
from ..adapters.shadow import observation_rgb_key
from ..tools.framework import FOLLOW_RESTART_TOOLS, TOOL_SPECS, follow_supervision_active, validate_call
from ..tools.navigation import NavigationSkill, spatial_context
from ..tools.spatial import SpatialView, ReferenceMap, spatial_source_binding, layer_binding
from perception.environment.context import fingerprint

TERMINAL = {'succeeded', 'partial', 'failed', 'cancelled'}


class AgentSession:
    def __init__(self, instruction, catalog, navigation, store, *, use_memory=True, floor_catalogs=None):
        self.store, self.navigation = store, navigation
        self.model_reference_names = ModelReferenceNames()
        self.elevator = None
        self.memory = EvidenceMemory() if use_memory else None
        self.observation = None
        self.follow_reference = None
        self.follow_loss = None
        self.spatial_view = None
        self.reference_map = None
        self.reference_catalog = catalog
        self.trajectory = Trajectory()
        self.requested_bounds = None
        self.requested_static_crop = False
        self.candidate_checks = []
        self.candidate_source = None
        self.candidate_space = None
        self.route_preview = None
        self.input_status = dict(rgb='not_provided',maps='not_provided')
        self._applying_context = None
        self.presented_completion = None
        self.seating_pending = None
        self.seating_active = None
        self._revalidated_view = None
        self.environment_epoch=0
        task_id = uuid.uuid4().hex
        self.catalog = catalog.bind_start(navigation.snapshot(), task_id)
        self.keyframes = Keyframes(store,task_id)
        if hasattr(navigation, 'catalog'):
            navigation.catalog = self.catalog
        if floor_catalogs is not None:
            if navigation.scope!='manual':raise ValueError('Cross-floor task catalogs require manual mode')
            floor_catalogs=dict(floor_catalogs)
            floor_catalogs[self.catalog.value['floor_id']]=self.catalog
        self.task_goals = TaskGoals(self.catalog,floor_catalogs=floor_catalogs)
        self.monitor = TaskMonitor(self.catalog)
        self.conversation = self._new_conversation(instruction)
        self.state = dict(task_id=task_id, revision=1, instruction=instruction, status='ready', reason=None,
                          scope=navigation.scope, history=[], clarifications=[], completed_revisions=[], waiting_binding=None,
                          stop_target=None, pending_instruction=None, task_start=copy.deepcopy(self.catalog.places.get('mission_start')),
                          initial_floor=self.catalog.value.get('floor_id'),catalog_fingerprint=self.catalog.fingerprint, assistant_text=None)
        self.skill = NavigationSkill(navigation, self.catalog, task_id, self._save)
        self.registry = {name: getattr(self, name) for name in TOOL_SPECS}
        self._save('task_created')

    def _save(self, kind, value=None):
        self.state['navigation'] = copy.deepcopy(self.skill.record)
        self.state['evidence_memory'] = [] if self.memory is None else self.memory.snapshot()
        self.state['task_goals'] = self.task_goals.snapshot()
        self.state['task_monitor'] = self.monitor.snapshot()
        self.state['follow_reference'] = copy.deepcopy(self.follow_reference)
        self.state['follow_loss'] = copy.deepcopy(self.follow_loss)
        self.state['spatial_view'] = None if self.spatial_view is None else copy.deepcopy(self.spatial_view.record)
        self.state['conversation'] = self.conversation.snapshot()
        self.state['candidate_checks'] = copy.deepcopy(self.candidate_checks)
        self.state['route_preview'] = copy.deepcopy(self.route_preview)
        self.store.save(self.state, kind, {} if value is None else value)

    def rebind_manual_floor(self, catalog, navigation, transition):
        """Move an existing manual task to an independently localized floor.

        The transition owner must invalidate in-flight model work before this
        operation. Nothing here launches a runtime or moves the wheelchair.
        """
        from .elevator_transition import ElevatorTransition
        if not isinstance(transition,ElevatorTransition):
            raise ValueError('A recorded elevator transition is required')
        proof=transition.snapshot()
        if (self.navigation.scope!='manual' or navigation.scope!='manual' or
                self.skill.record['handle'] is not None or navigation.handle is not None or
                proof['status']!='completed' or proof['observations'] or proof['task_id']!=self.state['task_id'] or
                proof['from_floor']!=self.catalog.value.get('floor_id') or
                proof['to_floor']!=catalog.value.get('floor_id') or
                proof['destination_map_id']!=catalog.value.get('localization_map_id')):
            raise ValueError('Floor transition does not match the current task and target catalog')
        stage=self.task_goals.current(proof['goal_id'])
        requirement=stage['requirement']
        if (requirement.get('stage_type')!='elevator' or requirement.get('to_floor')!=proof['to_floor'] or
                requirement.get('floor_id')!=proof['from_floor']):
            raise ValueError('Current task stage is not the matching elevator transition')
        state=navigation.tick()
        binding=state.get('map_binding') or {}
        mapped=(navigation.value.get('control') or {}).get('agent_mapped_pose') or {}
        association=proof['association']
        if (state.get('input_stale') or state.get('localization_status')!='valid' or
                state.get('origin_pose') is None or state.get('origin_frame_id')!=catalog.value['frame_id'] or
                binding.get('valid') is not True or binding.get('map_id')!=proof['destination_map_id'] or
                association.get('valid') is not True or association.get('map_id')!=proof['destination_map_id'] or
                association.get('map_frame')!=catalog.value['frame_id'] or
                association.get('association_session')!=binding.get('association_session') or
                association.get('association_generation')!=binding.get('association_generation') or
                mapped.get('source')!='current_localization_and_scene_association' or
                type(mapped.get('association_stamp_s')) not in (int,float) or
                not math.isfinite(mapped['association_stamp_s']) or
                mapped['association_stamp_s']<association.get('measurement_stamp_s',float('inf')) or
                mapped.get('local_identity')!=[association.get(k) for k in ('local_session_id','local_generation','local_frame')] or
                mapped.get('map_binding')!=binding):
            raise ValueError('Destination runtime does not match the transition localization evidence')
        # Loading the grid also verifies the static-map and semantic-catalog identity.
        navigation.reference_grid()
        previous=dict(floor_id=self.catalog.value['floor_id'],catalog_fingerprint=self.catalog.fingerprint,
                      navigation=copy.deepcopy(self.skill.record))
        stage['destination_localization'] = proof
        self.navigation,self.catalog,self.reference_catalog=navigation,catalog,catalog
        self.environment_epoch=state.get('environment_epoch',0)
        navigation.catalog=catalog
        self.task_goals.catalog=catalog
        self.task_goals.floor_catalogs[catalog.value['floor_id']]=catalog
        self.monitor=TaskMonitor(catalog)
        self.skill=NavigationSkill(navigation,catalog,self.state['task_id'],self._save)
        self.observation=self.spatial_view=self.reference_map=None
        self.requested_bounds=None;self.requested_static_crop=False
        self.route_preview=None;self.candidate_checks=[];self.candidate_source=self.candidate_space=None
        self.trajectory=Trajectory()
        if self.memory is not None:self.memory.invalidate(**spatial_context(state))
        self.input_status=dict(rgb='not_provided',maps='not_provided')
        self.state['catalog_fingerprint']=catalog.fingerprint
        self.conversation.introduction[1]=dict(role='user',content=encode(
            dict(source='initial_map_catalog',catalog=catalog.prompt())))
        self.conversation.notice('floor_changed',from_floor=proof['from_floor'],to_floor=proof['to_floor'],
                                 evidence=proof,previous_navigation=previous['navigation'])
        self.monitor.watch(proof['goal_id'], requirement['description'])
        self._save('manual_floor_rebound',dict(previous=previous,transition=proof))
        return dict(status='rebound',floor_id=proof['to_floor'],task_id=self.state['task_id'])

    def _new_conversation(self,instruction):
        dialogue=Conversation(instruction,self.catalog.prompt(),self.store.event)
        floors=self.task_goals.floor_catalogs
        if len(floors)>1:
            dialogue.introduction.append(dict(role='user',content=encode(dict(
                source='cross_floor_catalogs',catalogs={floor:catalog.prompt() for floor,catalog in floors.items()}))))
        return dialogue

    def _history(self, kind, value):
        self.state['history'].append(dict(kind=kind, value=copy.deepcopy(value), revision=self.state['revision']))
        self.state['history'] = self.state['history'][-24:]

    def record_elevator_observation(self, judgment, evidence, retain_observation):
        if not self.state.get('elevator_active') or not self._applying_context:
            return dict(status='refused',reason='elevator_monitor_context_required')
        return self.elevator.observe(judgment,evidence,retain_observation)

    def pause_map_localization(self, goal_id):
        if self.task_goals.current(goal_id).get('destination_localization'):
            return dict(status='refused',reason='destination_localization_already_available')
        if self.elevator is None:
            return dict(status='refused',reason='cross_floor_equipment_unavailable')
        result=self.elevator.start(self,goal_id)
        self.observation=self.spatial_view=None
        self._clear_candidates()
        self.state.update(status='waiting_monitor',elevator_active=True)
        return result

    def restart_map_localization(self,pose,evidence):
        if not self.state.get('elevator_active') or not self._applying_context:
            return dict(status='refused',reason='destination_scan_required')
        return self.elevator.restart(self,pose,evidence)

    def context(self):
        stage = next_unfinished_goal(self.task_goals.goals)
        navigation=self.navigation.snapshot()
        value = dict(task_id=self.state['task_id'], revision=self.state['revision'],
                     current_stage=None if stage is None else stage['binding'],
                     handle=copy.deepcopy(self.skill.record['handle']),
                     follow_reference_id=None if self.follow_reference is None else self.follow_reference['reference_id'],
                     **spatial_context(navigation))
        if navigation.get('environment_epoch'):value['environment_epoch']=navigation['environment_epoch']
        if self.skill.record['association_epoch']:value['association_epoch']=self.skill.record['association_epoch']
        if self.observation is not None:
            r = self.observation.record
            value.update(observation_id=r['observation_id'], observation_binding=r['input_fingerprint'], source_stamp_s=r['stamp_s'])
            value['input_view_binding'] = fingerprint(self.observation.prompt())
        if self.spatial_view is not None:
            self.spatial_view.verify()
            value['spatial_view_id'] = self.spatial_view.record['view_id']
        value['reference_map_id'] = None if self.reference_map is None else self.reference_map.record['reference_map_id']
        value['task_requirements'] = fingerprint([g['requirement'] for g in self.task_goals.goals])
        value['candidate_checks'] = fingerprint(self.candidate_checks)
        value['route_preview'] = fingerprint(self.route_preview)
        value['conversation_revision'] = self.conversation.revision
        value['completion_offer'] = self.presented_completion
        if self.skill.record['stage_type']=='follow':
            follow=self.navigation.snapshot().get('follow_status') or {}
            value['follow_target_identity']=copy.deepcopy(follow.get('target_identity'))
            value['follow_target_generation']=follow.get('target_generation')
        if (self.monitor.settings or {}).get('continuous_region_id') is not None:
            value['conditional_review_epoch']=self.monitor.region_epoch
            if self.monitor.review_phase=='reviewing':value['conditional_review_cycle']=True
        if self.state.get('elevator_active'):
            frame=self.elevator.frame or {}
            value['ride_input']=(self.elevator.transition.status,(frame.get('record') or {}).get('stamp_s'),
                                 (frame.get('scan_record') or {}).get('stamp_s'))
        return value

    def valid_context(self, context):
        if (context != self.context() or self.state['status'] != 'ready' or self.skill.record['context_lost']
                or self.skill.record['association_waiting']):
            return False
        if self.state.get('elevator_active'):
            return self.elevator.frame is not None
        current = self.navigation.snapshot()
        if not self.skill.record['connected'] or current.get('input_stale') or current.get('environment_fault'):
            return False
        stamp=context.get('source_stamp_s')
        if stamp is not None and (type(stamp) not in (int,float) or not math.isfinite(stamp) or
                                  stamp > current['stamp_s']):
            return False
        return current.get('localization_status', 'valid') == 'valid'

    def _current_rgb_context(self, context, observation, state):
        if observation is None or not self.valid_context(context):
            return False
        rgb = observation.record['rgb']
        stamp = rgb.get('stamp_s')
        now = state.get('stamp_s')
        return (rgb.get('status') == 'valid' and observation.record['clock_domain'] == state.get('clock_domain') and
                type(stamp) in (int, float) and type(now) in (int, float) and 0 <= now-stamp)

    def revalidate_spatial(self, call, context, *, _timing=None):
        """Apply geometry tools to a fresh view, retaining the model's original view ID."""
        import numpy as np
        tool, args = call['tool'], call['arguments']
        goal = None
        if tool != 'check_candidate':
            try:
                goal = self.task_goals.current(args['goal_id'])['requirement']
            except ValueError:
                return dict(status='refused',reason='goal_is_not_current')
            if goal.get('stage_type','navigation') != 'navigation':
                return dict(status='refused',reason='current_stage_does_not_support_navigation',
                            stage_type=goal['stage_type'])
            expected = 'navigate_route' if goal['kind'] in ('route','sequence') else 'navigate'
            if tool != expected:
                return dict(status='refused',reason='wrong_navigation_tool',expected_tool=expected,
                            goal_kind=goal['kind'])
            if tool == 'navigate_route' and self.navigation.scope not in ('manual','simulation'):
                return dict(status='refused',reason='route_execution_not_supported_by_adapter')
        old = self.spatial_view
        if old is None or args['view_id'] != old.record['view_id']:
            return dict(status='refused',reason='model_view_not_current')
        if not self.valid_context(context):
            return dict(status='refused',reason='model_input_context_unavailable')
        self.prepare_current_images()
        current, view = self.navigation.snapshot(), self.spatial_view
        if (view is None or spatial_context(current) != {k:context.get(k) for k in ('clock_domain','local_identity','map_binding')}
                or current.get('input_stale') or current.get('localization_status','valid')!='valid'
                or view.record.get('source_stamp_s') is None or not view.current(current['stamp_s'])):
            return dict(status='refused',reason='fresh_navigation_view_unavailable')
        if goal is not None:
            before, after = old.record.get('pose'), view.record.get('pose')
            if before is None or after is None:
                return dict(status='refused',reason='fresh_navigation_view_unavailable')
            if (np.shape(before)!=(3,) or np.shape(after)!=(3,) or not np.isfinite(before).all()
                    or not np.isfinite(after).all() or np.linalg.norm(np.asarray(before[:2])-after[:2])>.3
                    or abs((before[2]-after[2]+math.pi)%(2*math.pi)-math.pi)>.4):
                return dict(status='refused',reason='robot_pose_changed_since_model_input')
        audit=dict(model_view_id=old.record['view_id'],model_source_stamp_s=old.record['source_stamp_s'],
                   checked_view_id=view.record['view_id'],checked_source_stamp_s=view.record['source_stamp_s'])
        if tool=='navigate':
            audit.update(visual_preference_not_verified=bool(goal.get('visual_preferences')),
                         visual_conditions_not_verified=bool(goal.get('semantic_conditions') or goal.get('facing')))
        event={'navigate':'navigation','navigate_route':'route','check_candidate':'candidate'}[tool]
        self.store.event(event+'_revalidation',audit)
        self._revalidated_view=(audit['model_view_id'],audit['checked_view_id'])
        try:
            result=self.call(tool,_context=context,_timing=_timing,**args)
        finally:
            self._revalidated_view=None
        result['revalidation']=audit
        return result

    revalidate_navigation = revalidate_spatial
    revalidate_route = revalidate_spatial
    revalidate_candidate = revalidate_spatial

    def _invoke_tool(self, tool_name, arguments, timing):
        if timing is not None:
            timing['tool_invoked_s'] = perf_counter()
        try:
            return self.registry[tool_name](**arguments)
        finally:
            if timing is not None:
                timing['tool_returned_s'] = perf_counter()

    def call(self, tool_name, *, _context=None, _timing=None, **arguments):
        call = validate_call(dict(tool=tool_name, arguments=arguments))
        if self.state['status'] != 'ready':
            raise ValueError('Task is not ready for another model action')
        if self.skill.record['association_waiting']:
            return dict(status='unavailable',reason='waiting_for_map_association')
        if self.state.get('elevator_active'):
            if tool_name not in ('record_elevator_observation','restart_map_localization','set_navigation_monitor',
                                 'ask_user','wait_for_event'):
                return dict(status='refused',reason='map_localization_paused',
                            expected=self.elevator.transition.status)
            if not _context or not self.valid_context(_context):
                return dict(status='refused',reason='ride_observation_superseded')
            self._applying_context=_context
            try:
                return self._invoke_tool(tool_name,arguments,_timing)
            except (OSError,EOFError,ValueError,KeyError) as error:
                return dict(status='refused',reason=str(error)[:240])
            finally:self._applying_context=None
        self.store.event('tool_intent', call)
        self._applying_context = _context
        try:
            if (_context is not None and tool_name in FOLLOW_RESTART_TOOLS and
                    follow_supervision_active(self.skill.record)):
                value=dict(status='refused',reason='local_follow_owns_recovery',
                           follow_phase=self.skill.follow_phase, guidance='继续监控同一次跟随。用 wait_for_event 等待更新。')
            else:
                value = self._invoke_tool(tool_name,arguments,_timing)
        except (OSError, EOFError, ValueError, KeyError) as error:
            value = dict(status='unavailable', error_type=type(error).__name__,
                         reason=str(error)[:240] if isinstance(error, ValueError) else type(error).__name__,
                         automatically_retried=False)
        finally:
            self._applying_context = None
        if tool_name == 'check_candidate' and value.get('status') == 'available':
            self._remember_candidate(value)
        if tool_name == 'navigate_route' and value.get('status') == 'refused' and value.get('diagnostics'):
            preview_view=arguments['view_id']
            if self._revalidated_view is not None and preview_view==self._revalidated_view[0]:
                preview_view=self._revalidated_view[1]
            self.route_preview = dict(waypoints=copy.deepcopy(arguments['waypoints']),
                goal_id=arguments['goal_id'],view_id=preview_view,
                source_binding=self.spatial_view.record['input_source_binding'],
                diagnostics=copy.deepcopy(value['diagnostics']))
        if tool_name != 'record_task' or value.get('status') != 'recorded':
            self._history('tool', dict(call=call, result=value))
        self._save('tool_result', dict(call=call, value=value))
        return value

    def _remember_candidate(self, result):
        view = self.spatial_view.record
        source = view['input_source_binding']
        if self.candidate_source != source:
            raise ValueError('Candidate source changed before recording')
        camera = result['camera_view']
        entry = dict(pose=copy.deepcopy(result['pose']),view_id=result['view_id'],source_stamp_s=view['source_stamp_s'],
                     source_binding=source,historical_only=False,
                     static_footprint=result['static_footprint']['status'],local_footprint=result['local_footprint']['status'],
                     camera_status=camera['status'],camera_xy_m=copy.deepcopy(camera.get('camera_xy_m')),
                     points=copy.deepcopy(camera.get('points',[])),occlusion_checked=False,
                     actual_visibility_verified=False,checks=1)
        identity = (entry['pose'],sorted(p['point_id'] for p in entry['points']))
        for old in self.candidate_checks:
            if (old['pose'],sorted(p['point_id'] for p in old['points'])) == identity:
                entry['checks'] = old['checks']+1
                self.candidate_checks.remove(old)
                break
        self.candidate_checks = (self.candidate_checks+[entry])[-8:]

    def _clear_candidates(self):
        self.candidate_checks = []
        self.candidate_source = self.candidate_space = None
        self.route_preview = None

    def complete_subtask(self, goal_id, evidence):
        goal = self.task_goals.current(goal_id)
        context = self._applying_context
        if not evidence.strip() or not context or not self.valid_context(context):
            return dict(status='refused',reason='current_decision_context_required')
        facts, proof = stage_evidence(self,goal)
        if not context.get('completion_offer') or facts['missing']:
            return dict(status='unproven',missing=facts['missing'],reason='completion_evidence_unavailable')
        if context['completion_offer'] != self._offer_identity(goal,proof):
            return dict(status='unproven',reason='completion_evidence_changed')
        if goal['requirement'].get('stage_type')=='follow':
            outcome = self.finish_local_task(goal_id,context.get('observation_id'),evidence)
            if outcome['status']=='stopping' and self.skill.record.get('pending_completion'):
                self.skill.record['pending_completion']['agent_decision'] = evidence.strip()
                if facts.get('completion_region') is not None:
                    self.skill.record['pending_completion']['semantic']['completion_region']=copy.deepcopy(facts['completion_region'])
                self.monitor.note_review(goal_id,'satisfied',evidence.strip(),source_stamp_s=facts['rgb']['source_stamp_s'])
            return outcome
        semantic = dict(source='agent_decision',evidence=evidence.strip(),rgb=facts.get('rgb'),map_view=facts.get('map_view'))
        self.monitor.note_review(goal_id,'satisfied',evidence.strip(),source_stamp_s=facts['measurement_stamp_s'])
        checks=dict(facts.get('checks',{}))
        checks.update({k:copy.deepcopy(facts[k]) for k in ('measured_pose','measurement_stamp_s','distance_to_selected_m',
                      'yaw_error_rad','table_selection','destination_association','completion_region') if k in facts})
        self._complete_stage(goal_id,proof,checks,semantic)
        return self._close_after_decision(goal_id,evidence.strip())

    def _offer_identity(self, goal, proof):
        return fingerprint(dict(binding=goal['binding'],selected=(goal.get('selected') or {}).get('handle'),
                                proof_handle=proof.get('handle'),arrival_stamp=proof.get('stamp_s') if proof.get('handle') else None,
                                association=proof.get('association')))

    def _close_after_decision(self, goal_id, evidence):
        self.presented_completion = None
        if self.state.get('pending_continue'):
            return dict(status='completed',goal_id=goal_id,task_status='waiting_user')
        if self.task_goals.resolved():
            result = self.finish_task('succeeded',evidence)
            if result['status'] not in ('succeeded','partial'):
                self.state.update(status='waiting_navigation',reason='awaiting_final_stop')
                self._save('awaiting_final_stop')
            return dict(status='completed',goal_id=goal_id,task_status=self.state['status'])
        return dict(status='completed',goal_id=goal_id,task_status='ready')

    def _complete_stage(self, goal_id, proof, checks, semantic):
        """Retain completion evidence, then discard only stage-scoped working state."""
        goal = self.task_goals.current(goal_id)
        if goal['requirement'].get('stage_type')=='seating':
            from .seating import cancel_selection
            cancel_selection(self)
        semantic=copy.deepcopy(semantic)
        if self.monitor.last_review is not None:
            semantic['agent_task_review']=copy.deepcopy(self.monitor.last_review)
        if (self.navigation.scope=='manual' and goal['requirement'].get('stage_type','navigation') in ('navigation','seating') and
                proof.get('handle') is not None):
            self.navigation.release_target(proof['handle'])
        self.task_goals.complete(goal_id, proof, checks, semantic)
        self.monitor = TaskMonitor(self.catalog)
        if goal['requirement'].get('stage_type') == 'follow':
            self.follow_reference = None
            self.follow_loss = None
        self.observation = self.spatial_view = None
        self.requested_bounds = None
        self.requested_static_crop = False
        self._clear_candidates()
        if goal['requirement'].get('wait_after'):
            self._wait_for_continue(goal['requirement']['wait_after'],after_goal=goal_id)
        else:
            self._enter_current_stage()

    def _enter_current_stage(self):
        stage = next_unfinished_goal(self.task_goals.goals)
        if stage is not None:
            state = self.navigation.snapshot()
            stage['entered'] = dict(stamp_s=state['stamp_s'], clock_domain=state['clock_domain'])
            requirement=stage['requirement']
            focus = requirement['description']
            options = self.task_goals.monitor_for(requirement)
            triggers = {key:value for key,value in options.items() if key != 'focus' and value != []}
            if triggers:
                self.monitor.configure(requirement['goal_id'],options.get('focus',focus),**triggers)
            else:
                self.monitor.watch(requirement['goal_id'],options.get('focus',focus))

    def _sync_candidates(self, inputs):
        source = spatial_source_binding(inputs)
        space = fingerprint(dict(context=spatial_context(self.navigation.snapshot()),
                                 global_binding=layer_binding(inputs['global_grid'])))
        if self.candidate_space is not None and self.candidate_space != space:
            self.candidate_checks = []
            self.route_preview = None
        elif self.candidate_source is not None and self.candidate_source != source:
            for entry in self.candidate_checks:
                entry['historical_only'] = True
            self.route_preview = None
        self.candidate_space, self.candidate_source = space, source
        return source

    def _candidate_preview(self):
        if self.candidate_checks and not self.candidate_checks[-1]['historical_only']:
            return self.candidate_checks[-1]
        return None

    def lookup_places(self, name, floor_id=None):
        fields = ('place_id', 'name', 'aliases', 'pose', 'reference', 'status')
        catalog=self.catalog if floor_id is None else self.task_goals.floor_catalogs.get(floor_id)
        if catalog is None:return dict(status='unavailable',reason='unknown_floor')
        annotations = catalog.annotations()
        def matches(item, key):
            return not name or name in [item[key],item.get('name'),*item.get('aliases',[])]
        return dict(status='available',floor_id=catalog.value.get('floor_id'),places=[{k:p[k] for k in fields if k in p} for p in catalog.lookup(name)],
                    regions=[r for r in annotations['regions'] if matches(r,'region_id')])

    def record_task(self, goals):
        if self.skill.record['handle'] is not None:
            return dict(status='refused', reason='active_navigation')
        self.task_goals.record(goals,user_messages=self._user_messages())
        self.conversation.begin_execution(self.task_goals.execution_plan(),
                                          self.state['clarifications'])
        self.state['history'].clear()
        if self.memory is not None:
            self.memory = EvidenceMemory()
        self.observation = self.spatial_view = self.requested_bounds = None
        self.requested_static_crop = False
        self._clear_candidates()
        self._enter_current_stage()
        return dict(status='recorded', goals=self.task_goals.snapshot(),
                    original_instruction=self.state['instruction'], automatically_executed=False)

    def query_map(self, bounds=None):
        self.catalog.require_complete()
        inputs = self.navigation.spatial_inputs()
        result = self._build_map(inputs,bounds,static_crop=True)
        self.requested_bounds = copy.deepcopy(bounds)
        self.requested_static_crop = True
        return result

    def check_candidate(self, pose, view_id, point_ids=None):
        import math
        import numpy as np
        from navigation.planning.footprint import footprint_status, footprint_details, goal_obstruction
        from navigation.planning.inputs import load_config
        from navigation.planning.reference import planning_config
        self.catalog.require_complete()
        if self._revalidated_view is not None and view_id == self._revalidated_view[0]:
            view_id = self._revalidated_view[1]
        state = self.navigation.snapshot()
        if (self.spatial_view is None or self.spatial_view.record['view_id'] != view_id or
                not self.spatial_view.current(state['stamp_s']) or state.get('input_stale') or
                state.get('localization_status','valid') != 'valid'):
            return dict(status='refused',reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        inputs = self.navigation.spatial_inputs()
        if spatial_source_binding(inputs) != self.spatial_view.record['input_source_binding']:
            return dict(status='refused',reason='spatial_source_changed')
        pose = np.asarray(pose,dtype=float)
        if pose.shape not in ((2,),(3,)) or not np.isfinite(pose).all():
            raise ValueError('Expected finite candidate pose')
        geometry = inputs.get('geometry')
        if geometry is None:
            return dict(status='unavailable',reason='vehicle_geometry_unavailable')
        margin = float(planning_config(load_config())['safety_margin_m'])
        grid = inputs['global_grid']
        if grid.metadata['frame_id'] != self.catalog.value['frame_id']:
            raise ValueError('Candidate and static map frame mismatch')
        def occupancy_details(snapshot, goal):
            if len(goal)==3:
                return dict(status=footprint_status(snapshot,geometry,goal,margin),
                            details=footprint_details(snapshot,geometry,goal,margin))
            return dict(status=goal_obstruction(snapshot,geometry,goal,margin) or 'undetermined',
                        details=None)
        body = dict(**occupancy_details(grid,pose),margin_m=margin,
                    source='static_prior_only',dynamic_obstacles_checked=False,route_checked=False,
                    geometry_assumed=geometry.assumed)
        local = transform = None
        scene = inputs.get('scene')
        if scene is not None and scene.record['geometry']['status'] == 'valid':
            from ..tools.spatial import se2
            local = scene.geometry
            if local.metadata['frame_id'] == self.catalog.value['frame_id']:
                transform = np.eye(3)
            elif inputs.get('pose') is not None and inputs.get('local_pose') is not None:
                transform = se2(inputs['pose'])@np.linalg.inv(se2(inputs['local_pose']))
        elif inputs.get('local_grid') is not None:
            matrix = np.asarray(inputs.get('local_grid_map_from_local'),dtype=float)
            if matrix.shape == (4,4) and np.isfinite(matrix).all():
                local = inputs['local_grid']
                transform = matrix[np.ix_([0,1,3],[0,1,3])]
        local_result = dict(status='unavailable',reason='no_bound_local_grid')
        if local is not None and transform is not None:
            inverse = np.linalg.inv(transform)
            xy = inverse@np.array([pose[0],pose[1],1.])
            local_pose = xy[:2].tolist()
            if len(pose)==3:
                local_pose.append(float(pose[2]-math.atan2(transform[1,0],transform[0,0])))
            local_result = dict(**occupancy_details(local,local_pose),
                                source='bound_local_occupancy',source_stamp_s=inputs.get('stamp_s'))
            if local_result['status']=='outside':local_result['status']='not_covered'
        sensors = inputs.get('sensors') or {}
        camera = sensors.get('camera') or {}
        view = camera.get('view') or {}
        fov = dict(status='unavailable',reason='matching_camera_geometry_unavailable',points=[])
        points = point_ids or []
        if len(points) != len(set(points)):
            raise ValueError('Duplicate point IDs')
        for point_id in points:
            if point_id not in self.catalog.places or self.catalog.places[point_id].get('pose') is None:
                raise ValueError('Unknown point ID: '+point_id)
        if len(pose)==3 and sensors.get('status') == 'configured' and view.get('status') == 'approximate_pinhole':
            position = np.asarray(camera['position_m'][:2],float)
            edges = np.asarray(view['edge_yaw_rad'],float)
            if position.shape != (2,) or edges.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(edges).all():
                raise ValueError('Invalid camera geometry')
            c,s = math.cos(pose[2]),math.sin(pose[2])
            camera_xy = pose[:2]+np.array([[c,-s],[s,c]])@position
            wrap = lambda x: (x+math.pi)%(2*math.pi)-math.pi
            middle = edges[0]+wrap(edges[1]-edges[0])/2
            half = abs(wrap(edges[1]-edges[0]))/2
            results=[]
            for point_id in points:
                delta=np.asarray(self.catalog.places[point_id]['pose'][:2],float)-camera_xy
                distance=float(np.linalg.norm(delta))
                bearing=None if distance < 1e-9 else wrap(math.atan2(delta[1],delta[0])-pose[2])
                results.append(dict(point_id=point_id,distance_m=distance,bearing_from_base_rad=bearing,
                                    in_horizontal_fov=None if bearing is None else bool(abs(wrap(bearing-middle)) <= half)))
            fov=dict(status='approximate_pinhole',camera_xy_m=camera_xy.tolist(),edge_yaw_from_base_rad=edges.tolist(),
                     points=results,occlusion_checked=False,vertical_extent_checked=False,
                     actual_visibility_verified=False)
        return dict(status='available',view_id=view_id,pose=pose.tolist(),static_footprint=body,
                    local_footprint=local_result,camera_view=fov,
                    executable=False,execution_feedback=False)

    def _build_map(self, inputs, bounds, *, static_crop=False):
        self.prepare_reference(inputs['global_grid'])
        now = self.navigation.snapshot()
        self.trajectory.update(now,self.catalog.value['frame_id'],map_snapshot_id=self.catalog.value.get('map_snapshot_id'),
                               localization_map_id=self.catalog.value.get('localization_map_id'))
        trajectory = [] if inputs['stamp_s'] is None else self.trajectory.snapshot(
            frame_id=self.catalog.value['frame_id'],clock_domain=inputs['clock_domain'],now=inputs['stamp_s'])
        self._sync_candidates(inputs)
        preview = self._candidate_preview()
        self.spatial_view = SpatialView.build(self.catalog,inputs,bounds,trajectory=trajectory,
                                              static_crop=static_crop,candidate_preview=preview,
                                              route_preview=self.route_preview,trajectory_truncated=self.trajectory.truncated,
                                              trajectory_sample_interval_s=self.trajectory.sample_interval)
        self.store.spatial_view(self.spatial_view)
        self.store.event('trajectory_snapshot',dict(view_id=self.spatial_view.record['view_id'],segments=trajectory))
        return dict(status='available', view=self.spatial_view.record, target_candidates_generated=False)

    def prepare_reference(self, grid=None):
        if self.reference_map is not None or self.catalog.condition!='complete':
            return
        if grid is None:
            if not hasattr(self.navigation,'reference_grid'):
                return
            grid = self.navigation.reference_grid()
        self.reference_map = ReferenceMap.build(self.reference_catalog,grid)
        self.store.reference_map(self.reference_map)

    def prepare_current_images(self):
        if hasattr(self.navigation,'observe'):
            self.observation = None
            result = self.observe_scene()
            self.input_status['rgb'] = result.get('reason')
        if self.catalog.condition=='complete' and hasattr(self.navigation,'spatial_inputs'):
            current = self.navigation.snapshot()
            if (current.get('localization_status','valid')!='valid' or current.get('input_stale') or
                    self.skill.record['context_lost']):
                self.spatial_view = None
                self._clear_candidates()
                self.input_status['maps'] = 'localization_invalid_or_stale'
                return
            # Keep an explicitly requested crop while its source observation is
            # unchanged; otherwise regenerate both current maps from the source.
            try:
                inputs = self.navigation.spatial_inputs()
                source = self._sync_candidates(inputs)
                if self.spatial_view is not None:
                    self.spatial_view.verify()
                    preview = self._candidate_preview()
                    if (self.spatial_view.record.get('input_source_binding') == source and
                            self.spatial_view.record.get('candidate_preview') == preview and
                            self.spatial_view.record.get('route_preview') == self.route_preview):
                        return
                self._build_map(inputs,self.requested_bounds,static_crop=self.requested_static_crop)
                self.input_status['maps'] = None
            except (OSError,ValueError,KeyError) as error:
                self.spatial_view = None
                self._clear_candidates()
                self.input_status['maps'] = 'current_map_source_unavailable'
                self.store.event('current_maps_unavailable',dict(error_type=type(error).__name__))

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

    def select_follow_target(self,goal_id,observation_id,candidate_id,target_description):
        current=self.task_goals.current(goal_id)
        if current['requirement'].get('stage_type')!='follow':
            return dict(status='refused',reason='current_stage_is_not_follow')
        if self.skill.record.get('handle') is not None and not self._confirmed_live_hold(self.navigation.snapshot()):
            return dict(status='refused',reason='cancel_and_confirm_existing_call_first')
        context=self._applying_context or {};observation=self.observation
        if (observation is None or not self.valid_context(context) or
                context.get('observation_id')!=observation_id or
                observation.record.get('observation_id')!=observation_id or
                context.get('observation_binding')!=observation.record.get('input_fingerprint') or
                context.get('input_view_binding')!=fingerprint(observation.prompt()) or
                'follow_candidates' not in observation.record):
            return dict(status='refused',reason='current_model_observation_with_follow_candidates_required')
        from decision.vlm.t2.target import reference
        record,image,crop=reference(observation,candidate_id,target_description,
            task_id=self.state['task_id'],revision=self.state['revision'],goal_id=goal_id)
        self.store.follow_reference(record,image,crop)
        self.follow_reference=record
        self._save('follow_target_confirmed',dict(reference=record,executable=False))
        return dict(status='selected',reference_id=record['reference_id'],candidate_id=candidate_id,
                    has_3d_track=record['track'] is not None,executable=False,execution_completed=False)

    def start_local_task(self, goal_id, task_type, target_description=None, table_roi=None, reference_id=None):
        current=self.task_goals.current(goal_id)
        if current['requirement'].get('stage_type')!=task_type:
            return dict(status='refused',reason='local_task_type_does_not_match_stage')
        if task_type=='seating':
            if not isinstance(target_description,str) or not target_description.strip():
                return dict(status='refused',reason='seating_description_required')
            from .seating import start_selection
            return start_selection(self,current,target_description.strip(),table_roi)
        if table_roi is not None:
            return dict(status='refused',reason='table_roi_only_applies_to_seating')
        ref=self.follow_reference
        if (not isinstance(ref,dict) or not isinstance(reference_id,str) or not reference_id or
                reference_id!=ref.get('reference_id') or ref.get('goal_id')!=goal_id or
                ref.get('task_id')!=self.state['task_id'] or ref.get('task_revision')!=self.state['revision'] or
                not isinstance(ref.get('description'),str) or not ref['description'].strip() or
                ref.get('reference_id')!=fingerprint({k:v for k,v in ref.items() if k!='reference_id'})):
            return dict(status='refused',reason='agent_confirmed_follow_reference_required')
        saved=self.store.read_follow_reference(self.state['task_id'],self.state['revision'],goal_id,reference_id)
        if saved is None or saved[0]!=ref:
            return dict(status='refused',reason='agent_confirmed_follow_reference_unavailable')
        intent=dict(target_description=ref['description'],reference=copy.deepcopy(ref))
        result=self.skill.start_follow(goal_id,intent,self.state['revision'],current['requirement'])
        if result['status'] in ('accepted','uncertain'):
            self.follow_loss=None
            current['selected']=dict(handle=copy.deepcopy(result['handle']),follow_reference_id=ref['reference_id'],
                                     pose=None,orientation_source=None,view_id=None)
            self.observation=self.spatial_view=None;self._clear_candidates()
        return result

    def _remember_follow_loss(self, goal_id):
        navigation=self.navigation.snapshot();follow=navigation.get('follow_status') or {}
        if (follow.get('status')=='failed' and (follow.get('handle') or {}).get('goal_id')==goal_id and
                (follow.get('handle') or {}).get('task_id')==self.state['task_id']):
            last=follow.get('last_evidence')
            if last is not None and last.get('binding')!=fingerprint({k:v for k,v in last.items() if k!='binding'}):
                raise ValueError('Last follow evidence changed')
            if self.follow_loss is None or self.follow_loss['follow']['handle']!=follow['handle']:
                loss=dict(goal_id=goal_id,task_id=self.state['task_id'],task_revision=self.state['revision'],
                          observed_stamp_s=navigation['stamp_s'],context=spatial_context(navigation),follow=copy.deepcopy(follow))
                loss['binding']=fingerprint(loss);self.follow_loss=loss
                self.store.event('follow_loss',loss)
        if self.follow_loss is None or self.follow_loss['goal_id']!=goal_id:
            return None
        if (self.follow_loss['binding']!=fingerprint({k:v for k,v in self.follow_loss.items() if k!='binding'}) or
                self.follow_loss['context']!=spatial_context(navigation)):
            raise ValueError('Follow loss context changed')
        return self.follow_loss

    def _observed_follow_target(self, identity, generation):
        if not identity or type(generation) is not int or self.observation is None:
            return None
        record=self.observation.record
        candidates=record.get('follow_candidates') or {}
        matched=[item for item in candidates.get('candidates',[]) if item.get('track') and
                 item['track']['identity']==identity and item['track']['lifecycle_generation']==generation]
        if len(matched)!=1:
            return None
        return dict(target_identity=identity,target_generation=generation,candidate_id=matched[0]['id'],candidate_binding=candidates['binding'],
                    observation_id=record['observation_id'],rgb_stamp_s=record['rgb']['stamp_s'])

    def finish_local_task(self, goal_id, observation_id=None, evidence=None):
        current=self.task_goals.current(goal_id)
        stage_type=current['requirement'].get('stage_type')
        if stage_type=='seating':
            return dict(status='unavailable',reason='stage_skill_not_available',stage_type=stage_type)
        if stage_type!='follow' or self.skill.record['goal_id']!=goal_id:
            return dict(status='refused',reason='current_stage_is_not_active_local_task')
        context=self._applying_context or {};observation=self.observation
        state=self.navigation.snapshot()
        if (observation is None or observation_id!=observation.record['observation_id'] or
                context.get('observation_id')!=observation_id or
                not self._current_rgb_context(context,observation,state) or
                not isinstance(evidence,str) or not evidence.strip()):
            return dict(status='refused',reason='current_rgb_and_stage_evidence_required')
        target=self._observed_follow_target(context.get('follow_target_identity'),context.get('follow_target_generation'))
        if target is None:
            return dict(status='refused',reason='follow_target_not_identified_in_observed_rgb')
        semantic=dict(**target,source_stamp_s=observation.record['rgb']['stamp_s'],
            observation_binding=observation.record['input_fingerprint'],evidence=evidence.strip(),
            source='agent_visual_judgement_not_independent_ground_truth',
            observation_age_s=state['stamp_s']-observation.record['rgb']['stamp_s'],
            current_scene_verified=False)
        return self.skill.finish_follow(semantic)

    def save_keyframe(self, observation_id, reason):
        context = self._applying_context
        if (self.observation is None or context is None or not self.valid_context(context) or
                not 0 <= self.navigation.snapshot()['stamp_s']-self.observation.record['rgb']['stamp_s'] <= 3 or
                observation_id != self.observation.record['observation_id'] or
                context.get('observation_binding') != self.observation.record['input_fingerprint'] or
                context.get('input_view_binding') != fingerprint(self.observation.prompt())):
            return dict(status='refused',reason='save_requires_current_rgb_consumed_in_this_model_request')
        record = self.keyframes.save(self.observation,self.state['revision'],reason)
        return dict(status='saved',keyframe=record)

    def read_keyframe(self, keyframe_id):
        record,_ = self.keyframes.read(keyframe_id,self.state['revision'])
        current = self.navigation.snapshot()
        observation = record['observation']
        if observation['clock_domain'] != current['clock_domain'] or observation['stamp_s'] > current['stamp_s']:
            return dict(status='refused',reason='historical_keyframe_not_causal_in_current_clock')
        return dict(status='available',keyframe=record)

    def navigate(self, goal_id, view_id, orientation_source=None, pose=None, point_id=None, yaw=None):
        self.catalog.require_complete()
        requirement = self.task_goals.current(goal_id)['requirement']
        stage_type = requirement.get('stage_type', 'navigation')
        if stage_type != 'navigation':
            return dict(status='refused',reason='current_stage_does_not_support_navigation',stage_type=stage_type,
                        explanation='当前阶段不是导航；如需先移动，应重新登记包含导航的任务。')
        if self.skill.record['handle'] is not None and not (self.navigation.scope == 'live' and self.skill.record['status'] == 'held'):
            return dict(status='refused', reason='cancel_and_confirm_existing_call_first')
        if self._revalidated_view is not None and view_id == self._revalidated_view[0]:
            view_id = self._revalidated_view[1]
        if requirement['kind'] in ('route','sequence'):
            return dict(status='refused',reason='task_requires_a_route')
        now = self.navigation.snapshot()['stamp_s']
        if self.spatial_view is None or self.spatial_view.record['view_id'] != view_id or not self.spatial_view.current(now):
            return dict(status='refused', reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        named_point = requirement['target_id'] if requirement['kind']=='point' else ''
        if named_point:
            xy = self.catalog.places[named_point]['pose'][:2]
            if ((point_id is not None and point_id != named_point) or
                    (pose is not None and math.dist(pose[:2],xy) > 1e-6)):
                return dict(status='refused',reason='use_registered_annotation_point',
                            point_id=named_point,position=xy)
        if point_id is not None:
            point=self.catalog.destination(point_id,self.navigation.scope)
            pose=list(point['pose'][:2])
            if yaw is not None:pose.append(yaw)
        if len(pose)==2 and goal_requires_heading(requirement):
            return dict(status='refused',reason='heading_required_for_current_goal',
                        allowed_orientation_sources=self.task_goals.orientation_sources(requirement))
        # Check the submitted pose itself, even if the model checked a different
        # candidate earlier. A current local free observation can supersede a
        # static occupied cell; unknown evidence is left to the planner.
        if getattr(self.navigation,'geometry',None) is not None:
            candidate = self.check_candidate(pose,view_id)
            if candidate['status'] != 'available':
                return dict(status='refused',reason='candidate_geometry_unavailable',candidate=candidate)
            static = candidate['static_footprint']['status']
            local = candidate['local_footprint']['status']
            if local == 'occupied' or static == 'occupied' and local != 'free':
                return dict(status='refused',reason='candidate_footprint_occupied',candidate=candidate)
        selected = self.task_goals.select(goal_id,pose,orientation_source,view_id)
        if selected['status'] != 'selected':
            return selected
        if point_id is not None:
            selected['destination']['selected_point_id']=point_id
            self.task_goals.current(goal_id)['selected']['selected_point_id']=point_id
        result = self.skill.start(selected['destination'],self.state['revision'])
        if result['status'] in ('accepted', 'uncertain'):
            self.task_goals.current(goal_id)['selected']['handle'] = copy.deepcopy(result['handle'])
            self.spatial_view = self.observation = None
            self.requested_bounds = None
            self.requested_static_crop = False
            self._clear_candidates()
        return result

    def navigate_route(self, goal_id, view_id, orientation_source, waypoints, final_yaw=None, waypoint_reason=None):
        stage_type = self.task_goals.current(goal_id)['requirement'].get('stage_type', 'navigation')
        if stage_type != 'navigation':
            return dict(status='refused', reason='stage_skill_not_available', stage_type=stage_type)
        if self.skill.record['handle'] is not None and not (self.navigation.scope == 'live' and self.skill.record['status'] == 'held'):
            return dict(status='refused', reason='cancel_and_confirm_existing_call_first')
        if not callable(getattr(self.navigation,'start_route',None)):
            return dict(status='refused',reason='route_execution_not_supported_by_adapter')
        import math
        import numpy as np
        self.catalog.require_complete()
        state=self.navigation.snapshot()
        if self._revalidated_view is not None and view_id == self._revalidated_view[0]:
            view_id=self._revalidated_view[1]
        if (self.spatial_view is None or self.spatial_view.record['view_id']!=view_id or
                self.spatial_view.record.get('pose') is None or self.spatial_view.record.get('source_stamp_s') is None or
                not self.spatial_view.current(state['stamp_s'])):
            return dict(status='refused',reason='query_a_current_spatial_view_first')
        self.spatial_view.verify()
        inputs=self.navigation.spatial_inputs()
        if spatial_source_binding(inputs)!=self.spatial_view.record['input_source_binding']:
            return dict(status='refused',reason='spatial_source_changed')
        points=np.asarray(waypoints,dtype=float)
        if (points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all() or
                final_yaw is not None and not math.isfinite(final_yaw)):
            return dict(status='refused',reason='route_waypoints_must_be_finite')
        if np.any(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-6):
            return dict(status='refused',reason='route_has_duplicate_consecutive_points')
        current=self.task_goals.current(goal_id)
        requirement=current['requirement']
        if requirement['kind'] not in ('route','sequence'):
            return dict(status='refused',reason='task_does_not_require_a_route')
        if final_yaw is None and any(requirement.get(key) for key in ('facing','semantic_conditions','visual_preferences')):
            return dict(status='refused',reason='route_final_yaw_required_for_semantic_goal')
        if final_yaw is None and 'required_yaw' in requirement:
            final_yaw=float(requirement['required_yaw'])
        if requirement['kind']=='sequence':
            index=0
            for point in points:
                if index<len(requirement['checkpoints']) and self.task_goals.checkpoint_met(requirement['checkpoints'][index],point):
                    index+=1
            if index!=len(requirement['checkpoints']) or not self.task_goals.checkpoint_met(requirement['checkpoints'][-1],points[-1]):
                return dict(status='refused',reason='route_does_not_include_ordered_user_checkpoints')
        from .route import waypoint_occupancy_diagnostics
        from navigation.planning.inputs import load_config
        from navigation.planning.reference import planning_config
        waypoint_check=waypoint_occupancy_diagnostics(inputs,points,float(planning_config(load_config())['safety_margin_m']))
        distinct=len({tuple(point) for point in points.tolist()})
        review_threshold=max(3,len(requirement.get('checkpoints',[]))+2)
        guidance = None
        if distinct > review_threshold:
            guidance = ('按顺序保留用户指定的全部检查点，核对额外途经点。' if requirement['kind']=='sequence'
                        else '先考虑用三个以内的已知空闲位置表达路线。')
            guidance += '修正占据检查拒绝的位置。确需更多点时在 waypoint_reason 说明具体依据。'
        blocked=next((item['reason'] for item in waypoint_check['waypoints'] if item['reason']),None)
        if blocked:
            result=dict(status='refused',reason=blocked,diagnostics=waypoint_check)
            if guidance is not None:result.update(distinct_waypoints=distinct,guidance=guidance)
            return result
        if distinct>review_threshold and not (isinstance(waypoint_reason,str) and waypoint_reason.strip()):
            return dict(status='refused',reason='route_waypoints_review_required',distinct_waypoints=distinct,
                explanation=guidance,
                diagnostics=waypoint_check)
        selected_yaw=float(self.spatial_view.record['pose'][2] if final_yaw is None else final_yaw)
        selected=self.task_goals.select(goal_id,[*points[-1].tolist(),selected_yaw],orientation_source,view_id)
        if selected['status']!='selected':return selected
        route=dict(waypoints=points.tolist(),final_yaw=None if final_yaw is None else float(final_yaw),
                   yaw_required=final_yaw is not None,waypoint_reason=None if waypoint_reason is None else waypoint_reason.strip())
        selected['destination']['route']=route
        current['selected']['route']=copy.deepcopy(route)
        result=self.skill.start(selected['destination'],self.state['revision'])
        result['waypoint_checks']=waypoint_check
        if result['status'] in ('accepted','uncertain'):
            current['selected']['handle']=copy.deepcopy(result['handle'])
            self.spatial_view=self.observation=None
            self.requested_bounds=None
            self.requested_static_crop=False
            self._clear_candidates()
        return result

    def observation_stop(self, state):
        return (self._confirmed_live_hold(state) or self._confirmed_manual_idle(state) or
                   self.navigation.scope == 'live' and self.skill.record['handle'] is None and
                   state.get('idle_stop_observation') is True and state.get('stopped') is True and
                   state.get('chassis_state') == 'armed' and state.get('chassis_status_fresh') is True and
                   not state.get('input_stale') and state.get('localization_status') == 'valid' or
                   self.navigation.scope == 'simulation' and self.skill.record['handle'] is None and
                   state.get('execution_feedback') is True and state.get('stopped') is True and
                   not state.get('input_stale') and state.get('localization_status', 'valid') == 'valid')

    def check_goal(self, **arguments):
        from ..workflows.scripted_review import check_goal
        return check_goal(self,**arguments)

    def get_navigation_state(self):
        return self.navigation_summary()

    def set_navigation_monitor(self, goal_id, focus, interval_s=None, region_ids=None, progress_interval_s=None,
                               distance_m=None, rotation_rad=None, stopped_s=None, timeout_s=None,
                               continuous_region_id=None,review_mode=None):
        requirement=self.task_goals.current(goal_id)['requirement']
        if requirement.get('stage_type')=='elevator':
            if any(v for v in (region_ids,progress_interval_s,distance_m,rotation_rad,stopped_s,timeout_s,continuous_region_id,review_mode)):
                return dict(status='refused',reason='only_independent_frame_or_periodic_review_available')
            if interval_s is None:
                self.monitor.watch(goal_id,focus)
                return dict(status='configured',monitor=self.monitor.snapshot(),execution_completed=False)
        active = self.skill.record
        if active['handle'] is not None and active['goal_id'] != goal_id:
            return dict(status='refused', reason='another_goal_owns_navigation')
        self.monitor.configure(goal_id, focus, interval_s, region_ids, progress_interval_s,
                               distance_m,rotation_rad,stopped_s,timeout_s,continuous_region_id,review_mode)
        return dict(status='configured', monitor=self.monitor.snapshot(), execution_completed=False)

    def record_monitor_review(self,goal_id,result,evidence):
        self.task_goals.current(goal_id)
        context=self._applying_context or {}
        if ((self.monitor.settings or {}).get('goal_id')!=goal_id or
                self.monitor.conditional_region_state!='inside' or self.monitor.review_phase!='reviewing' or
                not isinstance(evidence,str) or not evidence.strip() or
                not self.valid_context(context) or self.observation is None or
                context.get('observation_binding')!=self.observation.record.get('input_fingerprint')):
            return dict(status='refused',reason='conditional_review_or_observation_unavailable')
        evidence=evidence.strip()
        self.monitor.note_review(goal_id,result,evidence,source_stamp_s=context.get('source_stamp_s'))
        self.monitor.await_next_observation(observation_rgb_key(self.observation.record))
        self.state.update(status='waiting_monitor',waiting_binding=None)
        self._save('monitor_review',dict(goal_id=goal_id,result=result,evidence=evidence,
                                         source_stamp_s=context.get('source_stamp_s')))
        return dict(status='recorded',result=result,execution_completed=False)

    def cancel_navigation(self):
        from .seating import cancel_selection
        cancel_selection(self)
        result = self.skill.cancel()
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'waiting_navigation'
        return result

    def _observation_key(self):
        if hasattr(self.navigation, 'observation_key'):
            return self.navigation.observation_key()
        if hasattr(self.navigation, 'current'):
            r = self.navigation.current.record
        elif hasattr(self.navigation, 'value'):
            r = self.navigation.value.get('observation') or {}
        else:
            value = self.navigation.observe() if hasattr(self.navigation, 'observe') else None
            r = {} if value is None else value.record
        return r.get('input_fingerprint')

    def _monitor_observation_key(self):
        """Use only a current RGB observation for the next conditional review."""
        if hasattr(self.navigation,'observation_record'):
            r=self.navigation.observation_record()
        elif hasattr(self.navigation,'current'):
            r=self.navigation.current.record
        elif hasattr(self.navigation,'observe'):
            observation=self.navigation.observe()
            r={} if observation is None else observation.record
        else:return None
        state=self.navigation.snapshot();stamp=r.get('stamp_s');now=state.get('stamp_s')
        rgb_stamp=(r.get('rgb') or {}).get('stamp_s')
        if ((r.get('rgb') or {}).get('status')!='valid' or
                r.get('clock_domain')!=state.get('clock_domain') or
                r.get('local_identity')!=spatial_context(state)['local_identity'] or
                state.get('map_binding') is not None and r.get('map_binding')!=state.get('map_binding') or
                state.get('input_stale') or state.get('localization_status','valid')!='valid' or
                type(stamp) not in (int,float) or type(now) not in (int,float) or not 0<=now-stamp<=3 or
                type(rgb_stamp) not in (int,float) or not 0<=now-rgb_stamp<=3):
            return None
        return observation_rgb_key(r)

    def _confirmed_live_hold(self, feedback):
        return (self.navigation.scope == 'live' and self.skill.record['status'] == 'held' and
                self.skill.record['handle'] is not None and feedback.get('handle') == self.skill.record['handle'] and
                feedback.get('stop_observation') is True and feedback.get('stopped') is True and
                feedback.get('chassis_state') == 'armed' and feedback.get('chassis_status_fresh') is True and
                not feedback.get('input_stale') and feedback.get('localization_status') == 'valid')

    def _confirmed_manual_idle(self, feedback):
        return (self.navigation.scope == 'manual' and self.skill.record['handle'] is None and
                feedback.get('stopped') is True and not feedback.get('input_stale') and
                feedback.get('localization_status') == 'valid' and feedback.get('origin_pose') is not None and
                feedback.get('origin_frame_id') == self.catalog.value['frame_id'])

    def _user_messages(self):
        return [self.state['instruction'], *(item['answer'] for item in self.state['clarifications'])]

    def _wait_for_continue(self,quote,*,after_goal=None):
        self.state.update(status='waiting_user',reason=quote,pending_continue=dict(instruction_quote=quote,after_goal=after_goal))

    def wait_for_event(self, event, instruction_quote=None):
        if instruction_quote is not None and event!='user_continue':
            return dict(status='refused',reason='instruction_quote_only_for_user_continue')
        if self.state.get('elevator_active'):
            if event not in ('monitor','observation'):
                return dict(status='refused',reason='independent_sensor_observation_expected')
            self.elevator.frame=None
            self.state['status']='waiting_monitor'
            return dict(status='waiting',event=event)
        if event in ('navigation','observation','monitor') and self.monitor.await_next_observation(
                observation_rgb_key(self.observation.record)
                if self.observation is not None and self._applying_context is not None else None):
            self.state.update(status='waiting_monitor',waiting_binding=None)
            return dict(status='waiting',event='monitor')
        if event == 'navigation':
            if self.skill.record['handle'] is None:
                return dict(status='available', navigation=self.skill.snapshot())
            self.state['status'] = 'waiting_navigation'
        elif event == 'observation':
            self.state.update(status='waiting_observation', waiting_binding=self._observation_key())
        elif event == 'monitor':
            if self.monitor.settings is None or all(self.monitor.settings.get(k) in (None,[]) for k in
                    ('interval_s','region_ids','progress_interval_s','distance_m','rotation_rad','stopped_s','timeout_s','continuous_region_id')):
                return dict(status='refused',reason='configure_monitor_trigger_first')
            self.state['status']='waiting_monitor'
        else:
            stage=next_unfinished_goal(self.task_goals.goals)
            planned_wait=stage['requirement'].get('wait_after') if stage else None
            if planned_wait and instruction_quote in (None,planned_wait):
                return dict(status='refused',reason='registered_wait_starts_after_complete_subtask')
            if not user_quoted(instruction_quote,self._user_messages()):
                return dict(status='refused',reason='user_continue_requires_explicit_user_instruction')
            if not self.observation_stop(self.navigation.snapshot()):
                return dict(status='refused', reason='parking_not_confirmed')
            self._wait_for_continue(instruction_quote)
        return dict(status='waiting', event=event)

    def ask_user(self, question):
        if self.skill.record['handle'] is not None and not self._confirmed_live_hold(self.navigation.snapshot()):
            return dict(status='refused', reason='cancel_and_confirm_navigation_first')
        self.state.update(status='needs_clarification', reason=question)
        return dict(status='waiting_user_answer', question=question)

    def fail_task(self, reason):
        return self.finish_task('failed',reason)

    def finish_task(self, status, evidence):
        r = self.skill.record
        if status == 'failed':
            self.stop('failed', evidence)
            return dict(status=self.state['status'], reason=evidence)
        revision = self.state['revision']
        f = self.navigation.snapshot()
        live_arrival = (self.navigation.scope == 'live' and
                        (f.get('stop_observation') is True or self.skill.record['handle'] is None and
                         f.get('idle_stop_observation') is True) and
                        f.get('chassis_state') == 'armed' and f.get('chassis_status_fresh') is True)
        simulation_arrival = self.navigation.scope == 'simulation' and f.get('execution_feedback') is True
        manual_arrival = (self.navigation.scope == 'manual' and f.get('manual_arrival_observation') is True and
                          bool(r['completed']) and (f.get('manual_arrival_evidence') or f.get('manual_confirmation') or {}).get('handle') == r['completed'][-1]['handle'])
        last_goal=self.task_goals.goals[-1] if self.task_goals.goals else None
        manual_observation = (self._confirmed_manual_idle(f) and last_goal is not None and
                              last_goal['status']=='completed' and
                              last_goal['requirement'].get('stage_type')=='observation' and
                              (last_goal.get('completion') or {}).get('proof',{}).get('scope')=='manual')
        elevator_proof={} if last_goal is None else (last_goal.get('completion') or {}).get('proof',{})
        manual_elevator=(self._confirmed_manual_idle(f) and last_goal is not None and last_goal['status']=='completed' and
                         last_goal['requirement'].get('stage_type')=='elevator' and elevator_proof.get('status')=='completed' and
                         elevator_proof.get('destination_map_id')==(f.get('map_binding') or {}).get('map_id'))
        manual_follow=(self._confirmed_manual_idle(f) and last_goal is not None and
                       last_goal['requirement'].get('stage_type')=='follow' and f.get('stop_observation') and
                       elevator_proof.get('handle')==f.get('handle') and elevator_proof.get('agent_decision'))
        held = self.navigation.scope == 'live' and r['status'] == 'held'
        if (not (simulation_arrival or live_arrival or manual_arrival or manual_observation or manual_elevator or manual_follow) or (r['handle'] is not None and not held) or r['context_lost'] or
                not r['connected'] or not self.task_goals.resolved() or
                not f.get('stopped') or f.get('input_stale') or
                f.get('localization_status', 'valid') != 'valid'):
            return dict(status='refused', reason='task_execution_not_proven',
                        uncompleted_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals if g['status']!='completed'])
        result_status='succeeded' if self.task_goals.finished() else 'partial'
        self.state.update(status=result_status, reason=evidence)
        if result_status=='succeeded':self.state['completed_revisions'].append(revision)
        manual_source=(f.get('manual_arrival_evidence') or f.get('manual_confirmation') or {}).get('source') if manual_arrival else None
        return dict(status=result_status, scope=self.navigation.scope, real_robot_success=False,
                    evidence_kind=('follow_condition_and_manual_stop' if manual_follow else
                                   'agent_elevator_judgment_and_destination_localization' if manual_elevator else
                                   'agent_visual_judgement_and_localization_stop_estimate' if manual_observation else
                                   ('automatic_localization_and_stop_estimate' if manual_source=='automatic_localization_and_stop_estimate'
                                    else 'operator_confirmation_and_localization_estimate') if manual_arrival else
                                   'localization_estimate' if live_arrival else 'simulation'),
                    completed_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals if g['status']=='completed'],
                    skipped_goals=[g['requirement']['goal_id'] for g in self.task_goals.goals if g['status']=='skipped'])

    def stop(self, target, reason):
        from .seating import cancel_selection
        cancel_selection(self)
        self.seating_pending=None
        self.state.update(stop_target=target, reason=reason)
        if self.skill.record.get('pending_completion') is not None:
            self.skill.record['pending_completion'] = None
        if (self.skill.record['handle'] is None and self.skill.record['status']=='failed' and
                self.navigation.scope=='live' and getattr(self.navigation,'handle',None) is not None and
                (self.navigation.snapshot().get('follow_status') or {}).get('recovery_available') is True):
            self.skill.record['handle']=copy.deepcopy(self.navigation.handle)
            self.skill.record['status']='stopping';self.state['status']='stopping'
            try:self.navigation.cancel(self.navigation.handle)
            except (OSError,EOFError,ValueError) as error:
                self.store.event('navigation_stop_unconfirmed',dict(error_type=type(error).__name__,automatically_retried=False))
            self._save('task_stop_requested')
            return
        if self.skill.record['handle'] is not None:
            self.state['status'] = 'stopping'
            if self.skill.record['status'] != 'stopping':
                self.skill.cancel(reason=reason)
        else:
            self.state['status'] = target
        self._save('task_stop_requested')

    def event(self, name, text=None):
        s = self.state
        previous_revision = s['revision']
        if s['status'] in TERMINAL:
            raise ValueError('Task already terminal')
        if s.get('elevator_active') and name=='replace':
            raise ValueError('Finish or cancel the elevator stage before replacing the task')
        if s.get('elevator_active') and name=='arrived':
            raise ValueError('Map arrival is unavailable while localization is paused')
        if (s.get('elevator_active') and name=='review' and s['status']=='waiting_monitor' and
                self.elevator.standard_attempted):
            self.elevator.reopen_manual_selection(self)
            return
        if (name == 'review' and s['status'] in ('ready','waiting_navigation','waiting_observation','waiting_monitor') and
                not self.skill.record['context_lost'] and self.skill.record['connected']):
            goal=next_unfinished_goal(self.task_goals.goals)
            if goal is None:raise ValueError('No current subtask to review')
            event=dict(goal_id=goal['requirement']['goal_id'],focus=goal['requirement'].get('description'),
                       handle=copy.deepcopy(self.skill.record.get('handle')),
                       reasons=[dict(kind='operator_review_requested',stamp_s=self.navigation.snapshot().get('stamp_s'))])
            self.monitor.note_trigger(event)
            s.update(status='ready',reason=None,waiting_binding=None)
            self._history('task_monitor',event)
            self.conversation.notice('task_monitor',**event)
            self.store.event('task_monitor_event',event)
        elif name == 'arrived' and self.navigation.scope == 'manual':
            goal=next_unfinished_goal(self.task_goals.goals)
            confirmation=self.navigation.confirm_arrival()
            self._history('user',dict(event='manual_arrival_confirmation',confirmation=confirmation))
            self.conversation.notice('manual_arrival_confirmation',confirmation=confirmation)
            if s['status']=='needs_clarification':s.update(status='ready',reason=None)
            self._save('manual_arrival_confirmation',confirmation)
        elif name in ('pause', 'cancel'):
            self.stop('paused' if name == 'pause' else 'cancelled', name)
        elif name == 'skip' and self.navigation.scope == 'manual':
            if s.get('pending_continue'):
                raise ValueError('Waiting for the requested continue signal before the next subtask')
            goal=next_unfinished_goal(self.task_goals.goals)
            if goal is None:raise ValueError('No current subtask to skip')
            if s.get('elevator_active') or goal['requirement'].get('stage_type')=='elevator':
                raise ValueError('Cannot skip an elevator stage; floor transition must be reconciled')
            if s['status']=='stopping' or self.skill.record['context_lost']:
                raise ValueError('Cannot skip while navigation is stopping or its context is lost')
            s['pending_skip_goal']=goal['requirement']['goal_id']
            self.stop('skipping','operator_skipped_subtask')
            self._apply_skip()
        elif name == 'resume' and s['status'] == 'paused' and not self.skill.record['context_lost']:
            s.update(status='ready', reason=None)
            self._history('user', dict(event='resume'))
        elif name == 'continue' and s['status'] == 'waiting_user':
            s.update(status='ready', reason=None)
            goal_id=s.pop('pending_continue',{}).get('after_goal')
            if goal_id is not None:
                if self.task_goals.resolved():
                    completed=next(g['completion'] for g in self.task_goals.goals if g['requirement']['goal_id']==goal_id)
                    self._close_after_decision(goal_id,completed['semantic'].get('evidence') or '完成依据已记录，已收到继续口令')
                else:
                    self._enter_current_stage()
            self._history('user', dict(event='continue', source='user_event'))
        elif name == 'answer' and s['status'] == 'needs_clarification' and text and text.strip() and not self.skill.record['context_lost']:
            from .store import encode
            clarifications = [*s['clarifications'],dict(question=s['reason'],answer=text.strip())]
            if len(encode(clarifications).encode())>16384:
                raise ValueError('Clarification context exceeds 16384 bytes; replace with a concise complete instruction')
            s['clarifications'] = clarifications
            s.update(status='ready', reason=None)
            self._history('user', dict(event='answer', text=text.strip()))
        elif name == 'replace' and text and text.strip() and not self.skill.record['context_lost']:
            s['pending_instruction'] = text.strip()
            self.stop('replacing', 'user_replaced_instruction')
        else:
            raise ValueError('Event does not apply to current task')
        self._apply_replacement()
        if s.get('pending_continue') and s['status']=='ready':
            s.update(status='waiting_user',reason=s['pending_continue']['instruction_quote'])
        if s.get('elevator_active') and name in ('pause','resume','answer','review','continue'):
            self.elevator.frame=None
            self.monitor.reset_runtime()
            if s['status']=='ready':
                s['status']='waiting_monitor'
        if s['revision']==previous_revision:
            self.conversation.notice('user_event',event=name,text=text)
        self._save('user_event', dict(name=name, text=text))

    def _apply_replacement(self):
        s = self.state
        if s['status'] == 'replacing':
            s.pop('pending_continue',None)
            s.update(instruction=s['pending_instruction'], pending_instruction=None, revision=s['revision']+1,
                     status='ready', reason=None, clarifications=[])
            self.observation = None
            self.follow_reference = None
            self.follow_loss = None
            self.spatial_view = None
            self.requested_bounds = None
            self.requested_static_crop = False
            self._clear_candidates()
            self.task_goals = TaskGoals(self.catalog,floor_catalogs=self.task_goals.floor_catalogs)
            self.monitor = TaskMonitor(self.catalog)
            self.conversation = self._new_conversation(s['instruction'])
            s['assistant_text'] = None
            self._history('user', dict(event='replacement', instruction=s['instruction']))

    def _apply_skip(self):
        s=self.state
        if s['status']!='skipping':return
        goal_id=s.pop('pending_skip_goal')
        goal=self.task_goals.current(goal_id)
        selected=(goal.get('selected') or {}).get('handle')
        if selected is not None and getattr(self.navigation,'handle',None)==selected:
            self.navigation.release_target(selected)
        self.task_goals.skip(goal_id)
        self.monitor=TaskMonitor(self.catalog)
        self.observation=self.spatial_view=None
        self.requested_bounds=None;self.requested_static_crop=False
        self.presented_completion=None;self._clear_candidates()
        self._history('user',dict(event='subtask_skipped',goal_id=goal_id))
        self.conversation.notice('subtask_skipped',goal_id=goal_id,operator='user',completed=False)
        s.update(status='partial' if self.task_goals.resolved() else 'ready',
                 reason='operator_skipped_subtask' if self.task_goals.resolved() else None,waiting_binding=None,stop_target=None)
        if s['status']=='ready':self._enter_current_stage()
        self._save('subtask_skipped',dict(goal_id=goal_id,source='operator',completed=False))

    def _drain_manual_arrival_events(self):
        drain=getattr(self.navigation,'drain_arrival_events',None)
        if callable(drain):
            for event in drain():self.store.event(event['event'],event)

    def tick(self):
        s = self.state
        if s.get('elevator_active'):
            if s['status'] in ('waiting_monitor','ready'):
                sample=self.elevator.poll(self)
                if sample is not None:
                    self.monitor.update_clock(sample['record']['stamp_s'],first_observation=True)
                    if self._wake_monitor():
                        self.elevator.frame=sample
            return s
        stage=next_unfinished_goal(self.task_goals.goals)
        # Starting the independent observation is equipment setup for an
        # already registered elevator stage, not a model judgment about entry.
        # Do it before checking the source map again: movement into the lift can
        # invalidate that map while an asynchronous model request is in flight.
        if (stage is not None and stage['requirement'].get('stage_type')=='elevator' and
                not stage.get('destination_localization') and self.elevator is not None and
                self.navigation.scope=='manual' and self.skill.record.get('handle') is None and
                not self.skill.record['association_waiting'] and
                s['status'] in ('ready','waiting_navigation','waiting_observation','waiting_monitor')):
            ready=getattr(self,'elevator_start_ready',None)
            if ready is not None and not ready():return s
            try:
                result=self.pause_map_localization(stage['requirement']['goal_id'])
                if result.get('status')!='accepted':
                    raise ValueError(result.get('reason','elevator_observation_start_refused'))
                self._save('elevator_observation_started',dict(source='registered_elevator_stage',result=result))
            except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                s.update(status='needs_clarification',reason='elevator_observation_start_failed')
                self._save('elevator_observation_start_failed',dict(error_type=type(error).__name__,
                    reason=str(error)[:240],automatically_retried=False))
            return s
        self.navigation.want_follow_candidates=bool(stage and stage['requirement'].get('stage_type')=='follow' and
                                                    self.skill.record['handle'] is None)
        association_epoch=self.skill.record['association_epoch']
        was_association_waiting=self.skill.record['association_waiting']
        changed = self.skill.tick()
        feedback=self.navigation.snapshot()
        association_waiting=self.skill.record['association_waiting']
        if (association_epoch!=self.skill.record['association_epoch'] or
                was_association_waiting and not association_waiting):
            self.observation=self.spatial_view=self.presented_completion=None
            self._clear_candidates()
            self.monitor.reset_runtime()
            if not association_waiting and not self.skill.record['context_lost']:
                if self.memory is not None:self.memory.invalidate(**spatial_context(feedback))
                self.conversation.notice('navigation_association_recovered',phase_preserved=True,new_evidence_required=True)
                self._save('task_association_recovered')
        if association_waiting and s['status'] not in ('stopping',*TERMINAL):
            return s
        epoch=feedback.get('environment_epoch',0)
        if epoch!=self.environment_epoch:
            self.environment_epoch=epoch
            self.observation=self.spatial_view=self.presented_completion=None
            self._clear_candidates()
            self.monitor.reset_runtime()
            self._save('navigation_environment_reset',dict(epoch=epoch,reason='environment_voxel_budget_exceeded'))
        if (feedback.get('environment_fault') and self.skill.record['connected'] and
                not self.skill.record['context_lost'] and s['status'] not in TERMINAL):
            return s
        if self.seating_active is not None:
            from .seating import poll_updates
            try:poll_updates(self)
            except (ValueError,KeyError,TypeError,OSError,EOFError) as error:
                self.stop('needs_clarification','seating_update_unavailable')
                self._save('seating_update_unavailable',dict(error_type=type(error).__name__))
        if self.seating_pending is not None:
            if s['status']=='waiting_seating' and self.skill.record['connected'] and not self.skill.record['context_lost']:
                from .seating import poll_selection
                try:
                    poll_selection(self)
                except (ValueError,KeyError,TypeError,OSError,EOFError) as error:
                    self.seating_pending=None
                    self.stop('needs_clarification','seating_selection_unavailable')
                    self._save('seating_selection_unavailable',dict(error_type=type(error).__name__))
                return s
            self.seating_pending=None
            if s['status']=='waiting_seating':
                s.update(status='needs_clarification',reason='seating_navigation_context_lost')
        self._drain_manual_arrival_events()
        self.trajectory.update(self.navigation.snapshot(),self.catalog.value.get('frame_id'),
                               map_snapshot_id=self.catalog.value.get('map_snapshot_id'),
                               localization_map_id=self.catalog.value.get('localization_map_id'))
        r = self.skill.record
        if r['stage_type']=='follow' and r['status']=='failed' and r['goal_id'] is not None:
            self._remember_follow_loss(r['goal_id'])
        if s['status'] in ('ready','waiting_navigation','waiting_monitor') and not r['context_lost'] and r['connected']:
            stage = next_unfinished_goal(self.task_goals.goals)
            proof = r['completed'][-1] if r['completed'] else {}
            if (stage and proof.get('agent_decision') and stage['requirement']['goal_id']==proof['goal_id'] and
                    (stage.get('selected') or {}).get('handle')==proof['handle'] and
                    proof['handle']['plan_revision']==s['revision'] and self._follow_stop_confirmed(proof)):
                s.update(status='ready',reason=None)
                self._complete_stage(proof['goal_id'],proof,dict(follow_stopped=True),proof['semantic'])
                self._close_after_decision(proof['goal_id'],proof['agent_decision'])
                self._save('follow_stop_confirmed')
                return s
            if s.get('reason')=='awaiting_final_stop' and self.task_goals.resolved():
                semantic=self.task_goals.goals[-1]['completion']['semantic']
                self.finish_task('succeeded',semantic.get('evidence') or '末段完成依据已记录，已确认停稳')
                if s['status'] in ('succeeded','partial'):
                    self._save('final_stop_confirmed')
                return s
        if not r['connected'] and s['status'] == 'ready':
            s.update(status='waiting_navigation' if r['handle'] else 'needs_clarification', reason='navigation_connection_lost')
            self._save('task_connection_lost')
        lost_reason = r.get('reason') if r.get('reason') in ('chassis_disarmed_or_status_unavailable','operator_goal_preempted_agent') else 'spatial_context_changed'
        if r['context_lost'] and s['status'] not in TERMINAL and s.get('reason') != lost_reason:
            if self.memory is not None:
                self.memory.invalidate(**spatial_context(self.navigation.snapshot()))
            # Navigation has already requested cancellation. Never resend it.
            target = 'cancelled' if s.get('stop_target') == 'cancelled' else 'needs_clarification'
            s.update(status='stopping' if r['handle'] else target,
                     stop_target=target, reason=lost_reason)
            self._save('task_context_lost')
        if s['status'] == 'stopping' and r['handle'] is None:
            s['status'] = s['stop_target']
            self._apply_replacement()
            self._apply_skip()
            self._save('task_stop_confirmed')
        elif s['status'] in ('waiting_navigation','waiting_monitor') and (changed or
                s['status']=='waiting_navigation' and r['handle'] is None):
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
        stage=next_unfinished_goal(self.task_goals.goals)
        if (changed and stage is not None and stage['requirement']['goal_id']==r.get('goal_id') and
                not r.get('context_lost')):
            trigger=dict(goal_id=r['goal_id'],focus=(self.monitor.settings or {}).get('focus'),
                         handle=copy.deepcopy(r.get('handle')),
                         reasons=[dict(kind='navigation_state_changed',status=r['status'],reason=r.get('reason'),
                                       stamp_s=self.navigation.snapshot().get('stamp_s'))])
            if r['stage_type']=='follow' and r['handle'] is not None:
                trigger['reasons'][0]['follow_phase']=self.skill.follow_phase
            self.monitor.note_trigger(trigger)
            self.conversation.notice('task_monitor',**trigger)
            self.store.event('task_monitor_event',trigger)
        monitor_feedback = self.navigation.snapshot()
        if (s['status']=='waiting_monitor' and r['handle'] is None and
                (self.monitor.settings or {}).get('goal_id') is not None):
            self.monitor.update_idle(self.monitor.settings['goal_id'],monitor_feedback)
        elif (spatial_context(monitor_feedback) != self.skill.context or
                monitor_feedback.get('source') != self.skill.initial.get('source')):
            self.monitor.reset_runtime()
        else:
            self.monitor.update(r, monitor_feedback)
        if self.monitor.review_phase=='waiting_observation':
            self.monitor.observe(self._monitor_observation_key(),monitor_feedback.get('stamp_s'))
        self._wake_monitor()
        return s

    def _wake_monitor(self):
        if self.state['status'] not in ('waiting_navigation','waiting_observation','waiting_monitor'):
            return False
        event=self.monitor.take()
        if event is None:return False
        self.state.update(status='ready',waiting_binding=None)
        if not self.state.get('elevator_active'):
            self._history('task_monitor',event)
            self.conversation.notice('task_monitor',**event)
            self._save('task_monitor_event',event)
        return True

    def _follow_stop_confirmed(self,proof):
        state=self.navigation.snapshot()
        return (self._confirmed_live_hold(state) or self._confirmed_manual_idle(state) and
                state.get('stop_observation') and state.get('handle')==proof['handle'])

    def navigation_summary(self):
        record = self.skill.record
        feedback = self.navigation.snapshot()
        fields = ('stamp_s','clock_domain','source','session_id','generation','frame_id','pose','velocity',
                  'origin_pose','origin_frame_id','map_binding','localization_status','input_stale','stopped','execution_feedback',
                  'route_index','route_count','manual_arrival_observation','manual_arrival_invalidation',
                  'planning_status','planning_reason','follow_status','environment_fault')
        return dict({key:copy.deepcopy(record[key]) for key in
                    ('status','stage_type','goal_id','handle','selected_pose','reason','context_lost','connected','association_waiting')},
                    follow_phase=self.skill.follow_phase if record['stage_type']=='follow' and record['handle'] is not None else None,
                    feedback={key:copy.deepcopy(feedback[key]) for key in fields if key in feedback},
                    last_completion=copy.deepcopy(record['completed'][-1:]))

    def prompt(self):
        ride=self.elevator if self.state.get('elevator_active') else None
        if ride is None:
            self.prepare_reference()
            self.prepare_current_images()
        s, now = self.state, self.navigation.snapshot()
        goals = self.task_goals.goals
        stage = copy.deepcopy(next_unfinished_goal(goals))
        current = dict(task_id=s['task_id'],revision=s['revision'],status=s['status'],reason=s['reason'],scope=self.navigation.scope,
            clarifications=copy.deepcopy(s['clarifications']),
            task_goals=[dict(goal_id=g['requirement']['goal_id'],status=g['status']) for g in goals],
            next_goal_id=stage['requirement']['goal_id'] if stage else None,
            stage_entered=stage.get('entered') if stage else None,
            navigation=self.navigation_summary(), task_monitor=self.monitor.snapshot(), follow_reference=copy.deepcopy(self.follow_reference),
            follow_loss=copy.deepcopy(self.follow_loss),
            map=dict(condition=self.catalog.condition,frame_id=self.catalog.value['frame_id'],
                     catalog_fingerprint=self.catalog.fingerprint),
            recent_model_tool_choices=[] if self.memory is None else self.memory.recall(
                place_ids=[*self.catalog.places,''],now=now['stamp_s'],clock_domain=now['clock_domain']))
        if self.navigation.scope == 'manual':
            current['execution_mode'] = dict(mode='operator_driven',
                meaning='目标或少量有序途经点交给操作者使用原车手柄移动，程序不控制底盘；中间经过点无须停车或模型逐点决策，终点根据当前定位与持续停稳估计自动检查，人工确认是备用方式；结合途经记录、实际轨迹和当前观察判断任务。')
        elif self.navigation.scope == 'shadow':
            current['execution_mode'] = dict(mode='nonexecuting_shadow',
                meaning='只观察记录或回放并生成建议，不控制底盘；记录中的运动不能证明建议已执行或目标已到达。')
        if len(self.task_goals.floor_catalogs)>1:
            current['cross_floor']=dict(current_floor=self.catalog.value['floor_id'],
                                        available_floors=list(self.task_goals.floor_catalogs))
        current['keyframes'] = self.keyframes.index(s['revision'])
        self.presented_completion = None
        if stage is not None and ride is None:
            facts,proof = stage_evidence(self,stage)
            current['completion_evidence'] = facts
            if not facts['missing']:
                self.presented_completion = self._offer_identity(stage,proof)
        current['recent_candidate_checks'] = copy.deepcopy(self.candidate_checks)
        current['requirements_registered'] = bool(goals)
        current['robot_context'] = robot_context(self.navigation)
        current['selected_goal'] = None
        if stage:
            selected = stage['selected']
            selection = None
            if selected is not None:
                selection = {k:selected[k] for k in ('pose','orientation_source','view_id')}
                if 'route' in selected:
                    selection['route'] = {k:selected['route'].get(k) for k in ('waypoints','final_yaw','waypoint_reason')}
            current['selected_goal'] = dict(goal_id=current['next_goal_id'],selection=selection,
                allowed_orientation_sources=self.task_goals.orientation_sources(stage['requirement']))
            if stage['requirement']['kind'] in ('point','region','reference'):
                current['selected_goal']['heading_required']=goal_requires_heading(stage['requirement'])
        current['last_tool_feedback'] = None
        for entry in reversed(s['history']):
            if entry['kind']=='tool' and entry['revision']==s['revision']:
                value = entry['value'];result = value['result']
                current['last_tool_feedback'] = dict(tool=value['call']['tool'],
                    **{k:copy.deepcopy(result[k]) for k in ('status','reason','error_type','checks','conditions','diagnostics',
                        'details','guidance','arrival_invalidation','distinct_waypoints','explanation') if k in result})
                break
        origin = copy.deepcopy(s['task_start']) if self.catalog.condition=='complete' else None
        if origin is not None and s.get('initial_floor')!=self.catalog.value.get('floor_id'):
            origin=None
        if origin is not None:
            origin['display_id'] = next(p['display_id'] for p in self.catalog.annotations()['points'] if p['place_id']=='mission_start')
        current['task_start'] = origin
        messages = self.conversation.messages()
        attachments = self.keyframes.attachments(messages,s['revision'])
        follow_image=None
        if (stage is not None and stage['requirement'].get('stage_type')=='follow' and
                self.follow_reference is not None):
            saved=self.store.read_follow_reference(s['task_id'],s['revision'],self.follow_reference['goal_id'],self.follow_reference['reference_id'])
            if saved is not None:
                follow_image=(saved[0],saved[2])
                attachments=dict(list(attachments.items())[-1:])
        current['visible_keyframe_ids'] = list(attachments)
        if ride is not None:
            feedback=dict(stamp_s=ride.frame['record']['stamp_s'],clock_domain='ros_time',
                          localization_status='paused',input_stale=True)
            current.update(navigation=dict(feedback=feedback),task_start=None,recent_model_tool_choices=[],
                           ride=dict(status=ride.transition.status,last_attempt=ride.last_attempt,
                                     previous_feedback=getattr(ride,'feedback',None)))
            standard=getattr(ride,'standard',None)
            if standard is not None:
                current['ride']['standard_start']=dict(floor_id=ride.transition.to_floor,reference='rslidar',
                    pose_rpy_xyz=copy.deepcopy(standard['pose_rpy_xyz']),
                    reference_lidar_pose_xyz_yaw=copy.deepcopy(ride.standard_reference[0]['lidar_in_map']),
                    meaning='用户指定的标准起点初值；参考图是历史观察，不能证明当前位置或定位成功。')
            attachments={}
        return dict(messages=messages,current_state=current,reference_map=None if ride else self.reference_map,
                    model_reference_names=self.model_reference_names,
                    ride_inputs=ride.attachments() if ride else None,keyframes=attachments,
                    follow_reference_image=follow_image,input_status=copy.deepcopy(self.input_status))
