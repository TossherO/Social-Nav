"""Previous multi-field completion protocol for explicitly scripted action replays."""
import copy
import math
import time

from ..core.tasks import angle_error
from ..core.completion import stage_evidence


def _check_observation_stage(session, goal_id, goal, state, observation_id, semantic_result, evidence):
    import numpy as np
    stopped = session.observation_stop(state)
    if not stopped:
        return dict(status='unproven', reason='observation_requires_confirmed_stop')
    observation = session.observation
    context = session._applying_context or {}
    entered = goal.get('entered') or {}
    if (observation is None or observation_id != observation.record['observation_id'] or
            context.get('observation_id') != observation_id or
            not session._current_rgb_context(context, observation, state) or
            entered.get('clock_domain') != observation.record['clock_domain'] or
            type(entered.get('stamp_s')) not in (int, float) or
            observation.record['rgb']['stamp_s'] < entered['stamp_s'] or
            semantic_result is None or not evidence or not evidence.strip()):
        return dict(status='needs_semantic_evidence', reason='read_current_rgb_after_stage_entry')
    observed = np.asarray(observation.record.get('base_pose'), dtype=float)
    measured = np.asarray(state.get('pose'), dtype=float)
    if (observed.shape != (3,) or measured.shape != (3,) or not np.isfinite(observed).all() or
            not np.isfinite(measured).all() or
            np.linalg.norm(observed[:2]-measured[:2]) > session.task_goals.settings['goal_position_tolerance_m'] or
            abs(math.atan2(math.sin(observed[2]-measured[2]), math.cos(observed[2]-measured[2]))) >
            session.task_goals.settings['goal_yaw_tolerance_rad']):
        return dict(status='needs_semantic_evidence', reason='observation_pose_changed')
    if semantic_result != 'satisfied':
        return dict(status='unknown' if semantic_result == 'unknown' else 'not_met')
    semantic = dict(observation_id=observation_id, source_stamp_s=observation.record['rgb']['stamp_s'],
                    observation_binding=observation.record['input_fingerprint'], evidence=evidence.strip(),
                    source='agent_visual_judgement_not_independent_ground_truth',
                    observation_age_s=state['stamp_s']-observation.record['rgb']['stamp_s'],
                    current_scene_verified=False)
    proof = dict(scope=session.navigation.scope, clock_domain=state['clock_domain'], stamp_s=state['stamp_s'],
                 handle=copy.deepcopy(session.skill.record['handle']), stopped=True, physical_acceptance=False)
    session._complete_stage(goal_id, proof, dict(observation_after_stage_entry=True), semantic)
    return dict(status='completed', goal_id=goal_id, scope=session.navigation.scope, real_robot_success=False,
                evidence_kind='agent_visual_judgement_and_stop_feedback')

def _manual_goal_evidence_issues(session, state, proofs, selected):
    """Explain the same manual arrival prerequisites used by check_goal."""
    issues = []
    proof = proofs[-1] if proofs else None
    confirmation = state.get('manual_arrival_evidence') or state.get('manual_confirmation') or {}
    if session.skill.record['handle'] is not None: issues.append('navigation_handle_still_active')
    if proof is None: issues.append('completion_record_missing')
    elif proof.get('handle') != selected.get('handle'): issues.append('goal_handle_mismatch')
    elif proof['handle'].get('plan_revision') != session.state['revision']: issues.append('task_revision_mismatch')
    if proof is None or confirmation.get('handle') != proof.get('handle'):
        issues.append('manual_arrival_evidence_missing_or_mismatched')
    if state.get('origin_frame_id') != session.catalog.value['frame_id']: issues.append('map_frame_unavailable_or_mismatched')
    if state.get('origin_pose') is None: issues.append('mapped_pose_unavailable')
    stamp = state.get('origin_stamp_s')
    if stamp is None or not 0 <= state['stamp_s']-stamp <= .5: issues.append('mapped_pose_out_of_date')
    if state.get('input_stale'): issues.append('runtime_input_stale')
    if state.get('localization_status','valid') != 'valid': issues.append('localization_invalid')
    if not state.get('stopped'): issues.append('stopped_estimate_unavailable')
    if not issues and state.get('manual_arrival_observation') is not True:
        issues.append('manual_arrival_observation_unavailable')
    return issues

def _wait_for_manual_goal_evidence(session, goal_id, state, proofs, selected):
    """Briefly refresh transiently unavailable evidence without another model call."""
    issues = _manual_goal_evidence_issues(session,state, proofs, selected)
    transient = {'mapped_pose_unavailable', 'mapped_pose_out_of_date', 'runtime_input_stale',
                 'localization_invalid', 'stopped_estimate_unavailable'}
    if (proofs and getattr(session.navigation,'automatic_arrival',False) and
            getattr(session.navigation,'handle',None)==proofs[-1].get('handle') and
            getattr(session.navigation,'status',None)=='running'):
        transient.add('manual_arrival_evidence_missing_or_mismatched')
    if not issues or not set(issues) <= transient:
        return state, issues
    started = time.monotonic()
    deadline = started + 1.0
    attempts = 0
    session.store.event('manual_goal_evidence_wait_started', dict(goal_id=goal_id, issues=issues, timeout_s=1.0))
    while issues and set(issues) <= transient and time.monotonic() < deadline:
        time.sleep(min(.05, max(0., deadline-time.monotonic())))
        try:
            state = session.navigation.tick()
        except (OSError, EOFError, ValueError) as error:
            issues = ['runtime_refresh_failed']
            session.store.event('manual_goal_evidence_refresh_failed', dict(goal_id=goal_id, error_type=type(error).__name__))
            break
        attempts += 1
        issues = _manual_goal_evidence_issues(session,state, proofs, selected)
    session.store.event('manual_goal_evidence_wait_finished', dict(goal_id=goal_id, issues=issues,
        attempts=attempts, elapsed_s=time.monotonic()-started, ready=not issues))
    return state, issues

def _check_follow_stage(session,goal_id,state,selected,proofs,observation_id,semantic_result,evidence):
    if (session.navigation.scope!='live' or session.skill.record['status']!='held' or not proofs or
            proofs[-1]['handle']!=selected.get('handle') or session.skill.record['handle']!=proofs[-1]['handle'] or
            proofs[-1]['handle']['plan_revision']!=session.state['revision'] or
            state.get('stop_observation') is not True or state.get('input_stale') or
            state.get('localization_status')!='valid' or state.get('chassis_state')!='armed' or
            state.get('chassis_status_fresh') is not True):
        return dict(status='unproven',reason='follow_finish_and_stop_not_proven')
    proof=proofs[-1]
    context=session._applying_context or {};observation=session.observation
    if (observation is None or observation_id!=observation.record['observation_id'] or
            context.get('observation_id')!=observation_id or
            not session._current_rgb_context(context,observation,state) or
            observation.record['rgb']['stamp_s']<proof['stop_stamp_s'] or
            semantic_result is None or not evidence or not evidence.strip()):
        return dict(status='needs_semantic_evidence',reason='read_post_stop_rgb_and_confirm_follow_condition')
    import numpy as np
    observed=np.asarray(observation.record.get('base_pose'),dtype=float)
    measured=np.asarray(state.get('pose'),dtype=float)
    if (observed.shape!=(3,) or measured.shape!=(3,) or not np.isfinite(observed).all() or
            not np.isfinite(measured).all() or
            np.linalg.norm(observed[:2]-measured[:2])>session.task_goals.settings['goal_position_tolerance_m'] or
            angle_error(observed[2],measured[2])>
            session.task_goals.settings['goal_yaw_tolerance_rad']):
        return dict(status='needs_semantic_evidence',reason='post_stop_observation_pose_changed')
    if semantic_result!='satisfied':return dict(status='unknown' if semantic_result=='unknown' else 'not_met')
    semantic=dict(before_stop=proof['semantic'],after_stop=dict(observation_id=observation_id,
        source_stamp_s=observation.record['rgb']['stamp_s'],evidence=evidence.strip(),
        source='agent_visual_judgement_not_independent_ground_truth',
        observation_age_s=state['stamp_s']-observation.record['rgb']['stamp_s'],current_scene_verified=False))
    session._complete_stage(goal_id,proof,dict(follow_stopped=True),semantic)
    return dict(status='completed',goal_id=goal_id,scope='live',real_robot_success=False,
                evidence_kind='agent_visual_judgement_and_localization_stop')

def check_goal(session, goal_id, observation_id=None, semantic_result=None, evidence=None,
               facing_result=None, facing_evidence=None, preference_result=None, preference_evidence=None,
               review_result=None, review_evidence=None):
    goal = session.task_goals.current(goal_id)
    stage_type = goal['requirement'].get('stage_type', 'navigation')
    context=session._applying_context
    if context is None and review_result is not None:
        return dict(status='unproven',reason='task_review_context_unavailable')
    if context is not None:
        if stage_type=='navigation' and (review_result is None or not review_evidence or not review_evidence.strip()):
            return dict(status='needs_task_review',reason='agent_review_result_and_evidence_required')
        if review_result is not None:
            if not review_evidence or not review_evidence.strip():
                return dict(status='needs_task_review',reason='agent_review_result_and_evidence_required')
            if not session.valid_context(context):
                return dict(status='unproven',reason='task_review_context_changed')
            selected = goal.get('selected') or {}
            review = goal.get('required_visual_review')
            conditions = goal['requirement'].get('semantic_conditions',[])
            if conditions and review_result=='satisfied' and (
                    (review and review['handle']==selected.get('handle')) or goal.get('visual_adjustments')):
                facts,_ = stage_evidence(session,goal)
                if not facts.get('rgb') or 'new_rgb_after_unsatisfied_visual_review' in facts['missing']:
                    return dict(status='needs_semantic_evidence', checks=facts.get('checks'), conditions=conditions,
                                reason='read_a_fresh_post_arrival_rgb_then_judge_all_conditions')
            session.monitor.note_review(goal_id,review_result,review_evidence.strip(),source_stamp_s=context.get('source_stamp_s'))
            if review_result != 'satisfied':
                return dict(status='unknown' if review_result=='unknown' else 'not_met',
                            reason='agent_task_review',review=dict(result=review_result,evidence=review_evidence.strip()))
        elif stage_type in ('observation','follow') and semantic_result is not None and evidence:
            # Existing visual-stage verdict is already the model's task review.
            session.monitor.note_review(goal_id,semantic_result,evidence.strip(),source_stamp_s=context.get('source_stamp_s'))
    if stage_type == 'seating':
        return dict(status='unavailable', reason='stage_skill_not_available', stage_type=stage_type)
    proofs = [p for p in session.skill.record['completed'] if p['goal_id'] == goal_id]
    selected = goal['selected'] or {}
    state = session.navigation.snapshot()
    if stage_type == 'observation':
        return _check_observation_stage(session,goal_id, goal, state, observation_id, semantic_result, evidence)
    if stage_type=='follow':
        return _check_follow_stage(session,goal_id,state,selected,proofs,observation_id,semantic_result,evidence)
    if stage_type=='elevator':
        return dict(status='unproven',reason='destination_map_association_required')
    manual_issues = None
    if session.navigation.scope == 'manual':
        state, manual_issues = _wait_for_manual_goal_evidence(session,goal_id, state, proofs, selected)
        session._drain_manual_arrival_events()
    from ..core.completion import navigation_facts
    result = navigation_facts(session,goal,state,selected,proofs,require_final_yaw=True)
    if result['status'] != 'available':
        if manual_issues is not None:
            result['details'] = manual_issues
            invalidation=state.get('manual_arrival_invalidation')
            if invalidation is not None:result['arrival_invalidation']=copy.deepcopy(invalidation)
            if ('manual_arrival_evidence_missing_or_mismatched' in manual_issues and
                    getattr(session.navigation,'automatic_arrival',False)):
                result['guidance']='自动到达证据正在等待有效定位与停稳；不要把文字回复当作新的到达证据。'
            session.store.event('manual_goal_evidence_unproven', dict(goal_id=goal_id, issues=manual_issues,
                arrival_invalidation=copy.deepcopy(invalidation)))
        return result
    proof, checks, measured_pose = result['proof'], result['checks'], result['measured_pose']
    live_arrival = session.navigation.scope == 'live'
    manual_arrival = session.navigation.scope == 'manual'
    semantic = dict(status='not_required')
    requirement = goal['requirement']
    facing = requirement.get('facing')
    qualitative_reference = (requirement['kind']=='reference' and requirement['target_id'] and
                             'max_distance_m' not in requirement and not requirement.get('semantic_conditions') and
                             not requirement.get('visual_preferences'))
    actual,chosen=measured_pose,selected.get('pose')
    if facing:
        if (actual is None or chosen is None or
                angle_error(actual[2],chosen[2]) > session.task_goals.settings['goal_yaw_tolerance_rad']):
            return dict(status='not_met',reason='actual_heading_differs_from_selected_heading',checks=checks)
    if facing or qualitative_reference:
        import numpy as np
        context = session._applying_context or {}
        view = session.spatial_view
        target = facing or requirement['target_id']
        reason = ('read_a_fresh_post_arrival_map_then_judge_facing' if facing
                  else 'review_proximity_on_the_post_arrival_map')
        if (view is None or context.get('spatial_view_id') != view.record['view_id'] or
                not session.valid_context(context) or view.record.get('pose') is None or
                view.record.get('source_stamp_s') is None or view.record['source_stamp_s'] < proof['stamp_s'] or
                facing and (facing_result is None or not facing_evidence or not facing_evidence.strip()) or
                qualitative_reference and review_result!='satisfied'):
            return dict(status='needs_semantic_evidence',checks=checks,reason=reason,target_id=target)
        view_pose=np.asarray(view.record['pose'],dtype=float)
        if (view_pose.shape!=(3,) or not np.isfinite(view_pose).all() or
                np.linalg.norm(view_pose[:2]-np.asarray(actual[:2])) > session.task_goals.settings['goal_position_tolerance_m'] or
                angle_error(view_pose[2],actual[2]) > session.task_goals.settings['goal_yaw_tolerance_rad']):
            return dict(status='needs_semantic_evidence',checks=checks,
                        reason='map_view_pose_differs_from_current_pose',target_id=target)
        if qualitative_reference:
            semantic.update(status='satisfied',reference=dict(target_id=requirement['target_id'],
                evidence=review_evidence,view_id=view.record['view_id'],stamp_s=view.record['source_stamp_s'],
                source='agent_spatial_judgement'))
    if facing:
        semantic_facing=dict(status=facing_result,evidence=facing_evidence,target_id=facing,
            view_id=view.record['view_id'],stamp_s=view.record['source_stamp_s'],
            selected_yaw=chosen[2],actual_yaw=actual[2],
            source='same_agent_spatial_judgement_not_independent_ground_truth',
            observation_age_s=state['stamp_s']-view.record['source_stamp_s'],current_scene_verified=False)
        if facing_result != 'satisfied':
            return dict(status='unknown' if facing_result == 'unknown' else 'not_met',
                        checks=checks,facing=semantic_facing)
        checks['facing_met']=True
        semantic.update(status='satisfied',facing=semantic_facing)
    conditions = goal['requirement'].get('semantic_conditions', [])
    if conditions:
        context = session._applying_context or {}
        observation = session.observation
        if (observation is None or observation_id != observation.record['observation_id'] or
                context.get('observation_id') != observation_id or not session.valid_context(context) or
                observation.record['rgb']['stamp_s'] < proof['stamp_s'] or
                semantic_result is None or not evidence):
            return dict(status='needs_semantic_evidence', checks=checks, conditions=conditions,
                        reason='read_a_fresh_post_arrival_rgb_then_judge_all_conditions')
        semantic.update(status=semantic_result, evidence=evidence, observation_id=observation_id,
                        input_binding=observation.record['input_fingerprint'], stamp_s=observation.record['stamp_s'],
                        source='same_agent_visual_judgement_not_independent_ground_truth',
                        observation_age_s=state['stamp_s']-observation.record['rgb']['stamp_s'],current_scene_verified=False)
        if semantic_result != 'satisfied':
            return dict(status='unknown' if semantic_result == 'unknown' else 'not_met', checks=checks, semantic=semantic)
    preferences = goal['requirement'].get('visual_preferences', [])
    if preference_result is not None or preference_evidence is not None:
        if not preferences:
            return dict(status='refused',reason='visual_preference_not_registered')
        context=session._applying_context or {}
        observation=session.observation
        if (observation is None or observation_id != observation.record['observation_id'] or
                context.get('observation_id') != observation_id or not session.valid_context(context) or
                observation.record['rgb']['stamp_s'] < proof['stamp_s'] or
                preference_result is None or not preference_evidence or not preference_evidence.strip()):
            return dict(status='needs_preference_evidence',checks=checks,preferences=preferences,
                        reason='read_a_fresh_post_arrival_rgb_then_judge_preference')
        semantic['visual_preferences']=dict(status=preference_result,evidence=preference_evidence,
            conditions=copy.deepcopy(preferences),observation_id=observation_id,
            input_binding=observation.record['input_fingerprint'],stamp_s=observation.record['stamp_s'],
            source='same_agent_visual_judgement_not_independent_ground_truth',
            observation_age_s=state['stamp_s']-observation.record['rgb']['stamp_s'],current_scene_verified=False)
    elif preferences:
        semantic['visual_preferences']=dict(status='not_assessed',conditions=copy.deepcopy(preferences))
    session._complete_stage(goal_id,proof,checks,semantic)
    manual_source=(state.get('manual_arrival_evidence') or state.get('manual_confirmation') or {}).get('source') if manual_arrival else None
    return dict(status='completed', goal_id=goal_id, checks=checks, semantic=semantic,
                scope=session.navigation.scope, real_robot_success=False,
                evidence_kind=(('automatic_localization_and_stop_estimate' if manual_source=='automatic_localization_and_stop_estimate'
                                else 'operator_confirmation_and_localization_estimate') if manual_arrival else
                               'localization_estimate' if live_arrival else 'simulation'))
