"""Navigation tool lifecycle over a supplied backend; no long-task reasoning."""
import copy
import math
import uuid
from interfaces.runtime import source_stamp_ns


class IntentRejected(ValueError):
    """The runtime explicitly rejected an intent before accepting it."""


def spatial_context(value):
    return dict(clock_domain=value.get('clock_domain'),
                local_identity=[value.get(k) for k in ('session_id', 'generation', 'frame_id')],
                map_binding=copy.deepcopy(value.get('map_binding')))


def _blockage_signature(observation):
    return tuple(observation.get(key) for key in ('source','kind','reason','blocked_position'))


def _planning_goal(observation):
    return tuple(observation.get(key) for key in ('goal_binding','request_id','task_id','goal_revision','segment_id'))


class NavigationSkill:
    def __init__(self, navigation, catalog, task_id, emit):
        if navigation.scope not in ('simulation', 'shadow', 'live', 'manual'):
            raise ValueError('Unknown navigation adapter scope')
        self.navigation, self.catalog, self.task_id, self.emit = navigation, catalog, task_id, emit
        self.initial = navigation.snapshot()
        self.context = spatial_context(self.initial)
        self.record = dict(status='idle', handle=None, place_id=None, reason=None, context_lost=False,
                           connected=True, completed=[], attempts=[], goal_revision=0, goal_id=None, selected_pose=None,
                           pending_completion=None, stage_type='navigation',follow_ready=False,search_deadline_s=None,
                           association_waiting=False,association_epoch=0,blockage=None,
                           recovery_cancel_stamp_s=None,recovery_stop=None)
        self.association_wait_stamp=None
        self.last_stamp = None
        self.blockage_run = None
        self.blockage_latest_ns = {}
        self.blockage_episode = 0
        self.environment_epoch = self.initial.get('environment_epoch',0)
        self.follow_phase = None
        self.follow_blocked_since = None

    def _emit(self, event):
        self.emit(event, copy.deepcopy(self.record))

    def _clear_blockage(self):
        self.blockage_run=None
        self.blockage_latest_ns.clear()
        self.follow_blocked_since=None
        self.record['blockage']=None

    @staticmethod
    def current_planning_observation(feedback):
        observation=feedback.get('planning_observation')
        if observation is not None and observation['source_stamp_s']<=feedback['stamp_s']<=observation['valid_until_s']:
            return observation
        return None

    def blockage_matches_feedback(self, blockage, feedback):
        stamp=feedback.get('stamp_s')
        if (type(stamp) not in (int,float) or not math.isfinite(stamp) or
                self.last_stamp is not None and stamp<self.last_stamp or
                feedback.get('handle')!=blockage.get('handle') or feedback.get('route_index')!=blockage.get('route_index') or
                feedback.get('status')!='running' or feedback.get('input_stale') or feedback.get('environment_fault') or
                feedback.get('scope')!=self.navigation.scope or feedback.get('source')!=self.initial.get('source') or
                feedback.get('localization_status','valid')!='valid' or spatial_context(feedback)!=self.context or
                feedback.get('environment_epoch',0)!=self.environment_epoch):
            return False
        observation=self.current_planning_observation(feedback)
        if observation is None:
            return True
        if _planning_goal(observation)!=_planning_goal(blockage):
            return False
        return (observation['status']=='unknown' or
                observation['status']=='blocked' and _blockage_signature(observation)==_blockage_signature(blockage))

    def _observe_blockage(self, feedback):
        r=self.record
        observation=self.current_planning_observation(feedback)
        route_index=feedback.get('route_index')
        active=r['blockage']
        if active is not None and route_index!=active['route_index']:
            self._clear_blockage()
            r.update(status='running',reason=None)
            self._emit('navigation_unblocked')
            return True
        if observation is None:
            # Keep measurements across a matching background search, without
            # treating the wait itself as another scan or extending its expiry.
            pending = feedback.get('pending_reference_request_id')
            run = self.blockage_run
            if (not pending or run is None or run['reference_request_id'] != pending or
                    feedback['stamp_s'] > run['valid_until_s']):
                self.blockage_run = None
            if active is not None:
                active['current']=False
            return False
        stamp=observation['source_stamp_s']
        stamp_ns=source_stamp_ns(stamp,observation.get('source_stamp_ns'))
        source=observation['source']
        last_ns=self.blockage_latest_ns.get(source)
        new_goal=active is not None and _planning_goal(active)!=_planning_goal(observation)
        new_blockage=(active is not None and observation['status']=='blocked' and
                      (new_goal or _blockage_signature(active)!=_blockage_signature(observation)))
        if last_ns is not None and (stamp_ns<last_ns or
                                    stamp_ns==last_ns and observation['status']=='blocked' and not new_blockage):
            # Same-scan refreshes cannot restore evidence or add to the hold time.
            return False
        self.blockage_latest_ns[source]=stamp_ns
        if observation['status']=='unknown':
            self.blockage_run=None
            if new_goal:
                r.update(status='running',reason=None,blockage=None)
                self._emit('navigation_blockage_changed')
                return True
            elif active is not None:
                active['current']=False
            return False
        if observation['status']=='clear':
            self.blockage_run=None
            if active is not None and stamp_ns>=source_stamp_ns(active['source_stamp_s'],active.get('source_stamp_ns')):
                r.update(status='running',reason=None,blockage=None)
                self._emit('navigation_unblocked')
                return True
            return False
        if observation['status']!='blocked' or observation.get('kind') not in (
                'destination_occupied','route_blocked','local_control_blocked'):
            return False
        signature=_blockage_signature(observation)
        if new_blockage:
            r.update(status='running',reason=None,blockage=None)
            active=None
        signature=(*signature,_planning_goal(observation),route_index)
        run=self.blockage_run
        continuous=run is not None and run['signature']==signature and stamp<=run['valid_until_s']
        if continuous:
            run['count']+=1
        else:
            self.blockage_run=run=dict(signature=signature,first_stamp_s=stamp,count=1,
                                       reference_request_id=observation.get('request_id') if source=='reference' else None)
        run['valid_until_s']=observation['valid_until_s']
        confirmed=active is not None and active['current'] and continuous
        hold_s=10. if source=='dwa' else 3.
        if not confirmed and (run['count']<2 or stamp-run['first_stamp_s']<hold_s):
            if active is not None:
                active['current']=False
            if new_blockage:
                self._emit('navigation_blockage_changed')
            return new_blockage
        if active is None:
            self.blockage_episode+=1
        evidence=dict(copy.deepcopy(observation),handle=copy.deepcopy(r['handle']),
                      route_index=route_index,episode=self.blockage_episode,first_source_stamp_s=run['first_stamp_s'],
                      observation_count=run['count'],current=True)
        r.update(status='blocked',reason=observation['reason'],blockage=evidence)
        if active is not None:
            return False
        self._emit('navigation_blocked')
        return True

    def _observe_follow_or_search_velocity(self, feedback, stamp):
        r=self.record
        if feedback.get('planning_reason')=='no_admissible_velocity':
            if self.follow_blocked_since is None:
                self.follow_blocked_since=stamp
            elif r['status']=='running' and stamp-self.follow_blocked_since>=10.:
                r.update(status='blocked',reason='no_admissible_velocity')
                self._emit('navigation_blocked')
                return True
        else:
            self.follow_blocked_since=None
            if r['status']=='blocked':
                r.update(status='running',reason=None)
                self._emit('navigation_unblocked')
                return True
        return False

    def _confirm_recovery_stop(self, feedback, stamp):
        r=self.record
        requested=r.get('recovery_cancel_stamp_s')
        if requested is None:
            return True
        measured=feedback.get('stop_measurement_stamp_s')
        if (r['context_lost'] or feedback.get('input_stale') or feedback.get('environment_fault') or
                feedback.get('localization_status')!='valid' or feedback.get('handle')!=r['handle'] or
                spatial_context(feedback)!=self.context or feedback.get('stopped') is not True or
                type(measured) not in (int,float) or not math.isfinite(measured) or
                not requested<measured<=stamp or stamp-measured>.5):
            return False
        r['recovery_stop']=dict(handle=copy.deepcopy(r['handle']),source=feedback['source'],
            stop_measurement_stamp_s=measured,stamp_s=stamp,context=spatial_context(feedback))
        return True

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
            for checkpoint in requirement['checkpoints']:
                for ident in ([checkpoint] if isinstance(checkpoint,str) else checkpoint['any_of']):
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
        self._clear_blockage()
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
        r.update(status='shadow_proposed' if self.navigation.scope=='shadow' else 'running',
                 recovery_cancel_stamp_s=None,recovery_stop=None)
        self._emit('navigation_ack')
        return dict(status='accepted', handle=handle, scope=self.navigation.scope, execution_completed=False, route=route)

    def start_follow(self, goal_id, intent, revision, requirement):
        r=self.record;state=self.navigation.snapshot()
        if not callable(getattr(self.navigation,'start_follow',None)):
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
        self._clear_blockage()
        self.last_stamp=None;self.follow_phase=None;self._emit('follow_intent')
        try:
            ack=self.navigation.start_follow(intent,handle)
            if ack.get('accepted') is not True or ack.get('handle')!=handle:raise ValueError('Follow ACK identity invalid')
        except IntentRejected as error:
            r.update(previous);self._mark_attempt(handle,'refused');self._emit('follow_refused')
            return dict(status='refused',reason=str(error))
        except (OSError,EOFError,ValueError) as error:
            r['reason']='submission_uncertain_'+type(error).__name__;self.cancel(reason=r['reason'])
            return dict(status='uncertain',handle=handle,reason=r['reason'])
        r.update(status='shadow_proposed' if self.navigation.scope=='shadow' else 'running',
                 recovery_cancel_stamp_s=None,recovery_stop=None)
        self._emit('follow_ack')
        return dict(status='accepted',handle=copy.deepcopy(handle),scope=self.navigation.scope,execution_completed=False)

    def retarget(self,destination):
        """Update the active seating goal without treating it as a completed stage."""
        r=self.record;old=r['handle'];state=self.navigation.snapshot()
        if (old is None or r['status'] not in ('running','blocked') or r['context_lost'] or
                not r['connected'] or state.get('input_stale') or state.get('localization_status')!='valid' or
                destination['goal_id']!=r['goal_id'] or not callable(getattr(self.navigation,'retarget',None))):
            return dict(status='refused',reason='seating_target_no_longer_active')
        new=copy.deepcopy(old);new['goal_revision']=r['goal_revision']+1
        new['view_id']=destination['view_id']
        try:
            ack=self.navigation.retarget(destination,old,new)
            if ack.get('accepted') is not True or ack.get('handle')!=new:raise ValueError('Retarget ACK invalid')
        except IntentRejected as error:
            return dict(status='refused',reason=str(error))
        except (OSError,EOFError,ValueError) as error:
            if self.navigation.scope=='live':
                try:self.navigation.client.exchange('cancel_owned',dict(task_id=old['task_id'],call_id=old['call_id']))
                except (OSError,EOFError,ValueError):pass
            else:
                try:self.navigation.cancel(old)
                except (OSError,EOFError,ValueError):pass
            r.update(status='stopping',reason='retarget_uncertain_'+type(error).__name__)
            self._emit('navigation_stop_intent')
            return dict(status='uncertain',reason=r['reason'])
        r.update(goal_revision=new['goal_revision'],handle=new,selected_pose=copy.deepcopy(destination['pose']),
                 status='running',reason=None,pending_completion=None)
        self._clear_blockage()
        self.last_stamp=None
        self._emit('navigation_retargeted')
        return dict(status='accepted',handle=copy.deepcopy(new),execution_completed=False)

    def finish_follow(self, semantic):
        r=self.record;handle=r['handle'];state=self.navigation.snapshot()
        if (r['stage_type']!='follow' or handle is None or r['status'] not in ('running','blocked','shadow_proposed') or
                (state.get('follow_status') or {}).get('status')!='running' or state.get('input_stale') or
                state.get('status') not in ('running','proposed')):
            return dict(status='refused',reason='follow_not_running')
        proof=dict(goal_id=r['goal_id'],place_id=None,handle=copy.deepcopy(handle),stamp_s=state['stamp_s'],
            scope=self.navigation.scope,source=state['source'],context=spatial_context(state),stopped=False,
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
        self._clear_blockage()
        r.update(status='stopping', reason=reason)
        self._emit('navigation_stop_intent')
        try:
            self.navigation.cancel(r['handle'])
        except (OSError, EOFError, ValueError) as error:
            self.emit('navigation_stop_unconfirmed', dict(error_type=type(error).__name__, automatically_retried=False))
        return dict(status='stopping', handle=copy.deepcopy(r['handle']), execution_completed=False)

    def _wait_for_manual_association(self, feedback, current):
        """Wait only for a map association in the same healthy local session."""
        r=self.record
        if self.navigation.scope!='manual' or not callable(getattr(self.navigation,'suspend_for_association',None)):
            return None
        previous=self.context.get('map_binding') or {}
        incoming=current['map_binding'] or {}
        same_local=(feedback.get('scope')=='manual' and feedback.get('localization_status')=='valid' and not feedback.get('input_stale') and
                    feedback.get('source')==self.initial.get('source') and
                    all(current[k]==self.context[k] for k in ('local_identity','clock_domain')))
        stable_fields=('map_id','map_frame','map_revision','association_session')
        anchored=(previous.get('valid') is True and all(previous.get(k) is not None for k in stable_fields) and
                  all(previous.get(k) for k in ('map_id','map_frame','association_session')) and
                  type(previous.get('association_generation')) is int)
        same_map=(incoming.get('valid') is True and all(incoming.get(k)==previous[k] for k in stable_fields)
                  and type(incoming.get('association_generation')) is int
                  and incoming['association_generation']>=previous['association_generation']) if anchored else False
        missing_map=(incoming.get('valid') is not True and
                     incoming.get('map_id') in (None,'',previous.get('map_id')))
        now=feedback.get('stamp_s')
        if (r['context_lost'] or r['stage_type']=='follow' or feedback.get('status')=='preempted'
                or not same_local or not anchored or not (missing_map or same_map)
                or type(now) not in (float,int) or not math.isfinite(now)):
            if r['association_waiting']:
                r['association_waiting']=False;self.navigation.waiting_for_association=False
            return None
        if not r['association_waiting']:
            if current==self.context or r['status']=='stopping' or feedback.get('environment_fault'):
                return None
            handle=self.navigation.suspend_for_association()
            self._clear_blockage()
            if r['status']=='blocked':
                r.update(status='running',reason=None)
            self.association_wait_stamp=now
            r['association_waiting']=True;r['association_epoch']+=1
            if handle is not None and r['status']=='succeeded' and r['handle'] is None:
                # The adapter still owns an arrived target awaiting Agent review.
                r.update(status='running',handle=handle)
                self._mark_attempt(handle,'active')
            self.last_stamp=None
            self._emit('navigation_association_waiting')
            return False
        if (r['status']=='stopping' and feedback.get('proposal_cancelled') is True and
                feedback.get('status')=='stopped' and self._confirm_recovery_stop(feedback,now)):
            self._mark_attempt(r['handle'],'cancelled')
            r.update(status='cancelled',handle=None,pending_completion=None)
            self._emit('navigation_cancel_confirmed')
            return True
        stamps=(feedback.get('origin_stamp_s'),feedback.get('association_measurement_stamp_s'),
                feedback.get('stop_measurement_stamp_s'))
        if (not same_map or incoming['association_generation']<=previous['association_generation']
                or feedback.get('environment_fault') or feedback.get('stopped') is not True or
                feedback.get('origin_pose') is None or feedback.get('origin_frame_id')!=incoming.get('map_frame') or
                any(type(t) not in (int,float) or not math.isfinite(t) or not self.association_wait_stamp<t<=now for t in stamps)
                or now-stamps[0]>.5 or now-stamps[1]>.5 or now-stamps[2]>.5):
            return False
        self.context=current
        self.navigation.waiting_for_association=False
        r['association_waiting']=False
        self._emit('navigation_association_recovered')
        return True

    def tick(self):
        r = self.record
        try:
            feedback = self.navigation.tick()
        except (OSError, EOFError, ValueError) as error:
            if r['association_waiting']:
                r['association_waiting']=False;self.navigation.waiting_for_association=False
            if r['connected']:
                r['connected'] = False
                if r['handle'] is not None and r['status'] != 'stopping':
                    self.cancel(reason='connection_lost')
                self.emit('navigation_connection_lost', dict(error_type=type(error).__name__, automatically_retried=False))
            return False
        r['connected'] = True
        epoch=feedback.get('environment_epoch',0)
        if epoch!=self.environment_epoch:
            self.environment_epoch=epoch
            self._clear_blockage()
            if r['status']=='blocked':
                r.update(status='running',reason=None)
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
        association=self._wait_for_manual_association(feedback,current)
        if association is not None:return association
        if (feedback.get('environment_fault') and not r['context_lost'] and not feedback.get('input_stale')
                and feedback.get('localization_status')=='valid'
                and current['local_identity']==self.context['local_identity']
                and current['clock_domain']==self.context['clock_domain']):
            # Compare the map identity when real geometry is available again.
            return False
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
        if r['stage_type']=='follow' and follow.get('handle')==handle:
            previous,self.follow_phase=self.follow_phase,follow.get('status')
            if r['follow_ready'] and self.follow_phase=='searching' and previous!='searching':
                self._emit('follow_searching');return True
            if r['follow_ready'] and self.follow_phase=='running' and previous=='searching':
                self._emit('follow_reacquired');return True
        if (r['stage_type']=='follow' and not r['follow_ready'] and follow.get('status')=='running' and
                feedback.get('status') in ('running','proposed')):
            r['follow_ready']=True;self._emit('follow_started');return True
        if self.navigation.scope in ('live','shadow','manual') and feedback.get('status') == 'preempted':
            self._clear_blockage()
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
                    measured=feedback.get('stop_measurement_stamp_s',stamp)
                    if r['stage_type']=='follow':
                        if type(measured) not in (int,float) or not proof['stamp_s']<measured<=stamp:
                            return False
                        proof.update(stopped=True,stop_stamp_s=measured)
                    elif self.navigation.scope=='live':
                        if type(measured) not in (int,float) or not proof['stamp_s']<=measured<=stamp:
                            return False
                        proof['stop_stamp_s']=measured
                    r['completed'].append(proof)
                    if self.navigation.scope=='manual':
                        r.update(status='succeeded',handle=None,pending_completion=None,reason=None)
                    else:
                        r.update(status='held', pending_completion=None, reason=None)
                    self._mark_attempt(handle, 'succeeded')
                    self._emit('navigation_completed')
                    return True
            return False
        if r['status'] == 'stopping':
            stopped = (feedback.get('stopped') and feedback.get('execution_feedback') is True if self.navigation.scope == 'simulation'
                       else feedback.get('stop_observation') is True if self.navigation.scope == 'live'
                       else feedback.get('proposal_cancelled') is True)
            if stopped and feedback.get('status') in ('stopped', 'failed') and self._confirm_recovery_stop(feedback,stamp):
                failed=bool(follow.get('status')=='failed')
                r.update(status='failed' if failed else 'cancelled', handle=None, pending_completion=None)
                self._mark_attempt(handle, 'failed' if failed else 'cancelled')
                self._emit('navigation_cancel_confirmed')
                return True
            return False
        goal = feedback.get('goal') or {}
        arrived = ((self.navigation.scope == 'simulation' and feedback.get('execution_feedback') is True and goal.get('task_completed')) or
                   (self.navigation.scope == 'live' and feedback.get('arrival_observation') is True) or
                   (self.navigation.scope == 'manual' and feedback.get('manual_arrival_observation') is True))
        if (feedback.get('status') == 'arrived' and arrived and feedback.get('stopped') and
                goal.get('task_id') == self.task_id and
                goal.get('goal_revision') == handle['goal_revision']):
            self._clear_blockage()
            proof = dict(goal_id=r['goal_id'], place_id=r['place_id'], handle=copy.deepcopy(handle), stamp_s=stamp,
                         scope=self.navigation.scope, source=feedback['source'], context=current,
                         pose=copy.deepcopy(feedback.get('origin_pose') if self.navigation.scope in ('live','manual') else feedback.get('pose')),
                         frame_id=feedback.get('origin_frame_id') if self.navigation.scope in ('live','manual') else feedback.get('frame_id'),
                         execution_feedback=self.navigation.scope == 'simulation', stopped=True,
                         physical_acceptance=False, evidence_kind=(('automatic_localization_and_stop_estimate' if
                             (feedback.get('manual_arrival_evidence') or {}).get('source')=='automatic_localization_and_stop_estimate'
                             else 'operator_confirmation_and_localization_estimate') if self.navigation.scope == 'manual'
                             else 'localization_estimate' if self.navigation.scope == 'live' else 'simulation'))
            if self.navigation.scope == 'manual':
                proof['operator_confirmation']=copy.deepcopy(feedback.get('manual_confirmation'))
                proof['manual_arrival_evidence']=copy.deepcopy(feedback.get('manual_arrival_evidence'))
            if feedback.get('route_start_stamp_s') is not None:
                proof['route_start_stamp_s']=feedback['route_start_stamp_s']
            if feedback.get('route_count') is not None:
                proof['route_visits']=copy.deepcopy(feedback.get('route_visits'))
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
        elif (self.navigation.scope=='live' and
                (r['stage_type']=='follow' or r.get('search_deadline_s') is not None)):
            return self._observe_follow_or_search_velocity(feedback,stamp)
        elif (self.navigation.scope in ('manual','live','simulation') and r['stage_type']=='navigation' and
                r.get('search_deadline_s') is None and
                r['status'] in ('running','blocked') and feedback.get('status')=='running'):
            return self._observe_blockage(feedback)
        return False

    def _mark_attempt(self, handle, status):
        for attempt in self.record['attempts']:
            if attempt.get('call_id') == handle['call_id']:
                attempt['status'] = status

    def snapshot(self):
        value = self.navigation.snapshot()
        return dict(copy.deepcopy(self.record), feedback=value, scope=self.navigation.scope,
                    execution_feedback=value.get('execution_feedback') is True)
