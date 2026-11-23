"""Apply current T3 proposals to their original seating stage."""
import copy
import json
import math
import time
import uuid

from ..tools.navigation import spatial_context
from decision.vlm.t3 import table_keepout_circle


def round_table_for_catalog(catalog, table_id):
    from decision.vlm.t3.geometry import validate_round_table
    from .. import ROOT

    config = ROOT/'decision/vlm/t3/configs/round_tables.json'
    place = catalog.places.get(table_id)
    if not place or place.get('pose') is None:
        raise ValueError('seating_table_center_not_annotated')
    radius = json.loads(config.read_text()).get(catalog.value['floor_id'], {}).get(table_id)
    if radius is None:
        raise ValueError('seating_table_radius_not_configured')
    table = dict(id=table_id, center_xy_m=place['pose'][:2], radius_m=radius,
                 map_id=catalog.value.get('localization_map_id'))
    try:
        validate_round_table(table)
    except ValueError as error:
        raise ValueError('seating_round_table_geometry_invalid') from error
    return table


def observation_history(goal, after_s):
    history = dict(after_s=after_s, visited_xy_m=copy.deepcopy(goal.get('seating_views', [])))
    if rejected := goal.get('seating_rejected_views'):
        history['rejected_views'] = copy.deepcopy(rejected)
    return history


def remember_rejected_view(goal, pose, reason):
    rejected = goal.setdefault('seating_rejected_views', [])
    rejected[:] = [item for item in rejected if math.dist(pose[:2], item['pose'][:2]) >= .35]
    rejected.append(dict(pose=list(pose), reason=reason))
    del rejected[:-8]


def start_selection(session,goal,instruction,roi,*,table_id=None):
    nav=session.navigation
    context=session._applying_context
    if nav.scope not in ('manual','live') or not hasattr(nav,'client'):
        return dict(status='unavailable',reason='stage_skill_not_available',stage_type='seating')
    if ((roi is None) == (table_id is None)):
        return dict(status='refused',reason='seating_requires_exactly_one_of_table_id_or_table_roi')
    if (not context or not session._current_rgb_context(context,session.observation,nav.snapshot()) or
            not session.observation_stop(nav.snapshot())):
        return dict(status='refused',reason='seating_requires_stopped_vehicle_and_current_rgb')
    if session.skill.record['handle'] and not session._confirmed_live_hold(nav.snapshot()):
        return dict(status='refused',reason='cancel_and_confirm_existing_call_first')
    selection=dict(selection_id=uuid.uuid4().hex,observation_binding=context['observation_binding'],
                   instruction=instruction)
    if table_id is None:
        selection['roi']=roi
    else:
        try:
            selection['round_table'] = round_table_for_catalog(session.catalog, table_id)
        except ValueError as error:
            return dict(status='refused',reason=str(error),table_id=table_id)
        if nav.scope == 'manual':
            selection['manual_observation'] = observation_history(goal, 0.)
    ack=nav.client.exchange('agent_seating_select',selection)
    if not ack.get('accepted') or ack.get('selection_id')!=selection['selection_id']:
        return dict(status='unavailable',reason=ack.get('error') or 'seating_selection_rejected')
    session.seating_pending=dict(**selection,goal_id=goal['requirement']['goal_id'],
        revision=session.state['revision'],context=spatial_context(nav.snapshot()),started=time.monotonic())
    session.state['status']='waiting_seating'
    return dict(status='selecting',selection_id=selection['selection_id'])


def footprint_approved(session, pending, result, checked):
    if checked.get('status')!='available':
        return False
    local = checked['local_footprint']['status']
    static = checked['static_footprint']['status']
    if (result.get('status')=='observation' and result.get('mapped_reference_verified') is True and
            'manual_observation' in pending and session.navigation.scope=='manual'):
        return static=='free' and local in ('free','unknown')
    return (local=='free' and (static=='free' or
             (session.navigation.scope=='manual' and 'manual_observation' in pending and
              result.get('conditional_prior_cleared') is True and static in ('occupied','unknown'))))


def poll_selection(session):
    pending=session.seating_pending
    result=session.navigation.value.get('seating') or {}
    if result.get('selection_id')!=pending['selection_id'] or result.get('status')=='selecting':
        if time.monotonic()-pending['started']<15:
            return
        result=dict(status='unavailable',reason='seating_selection_timeout')
    session.seating_pending=None
    session.state.update(status='ready',reason=None)
    goal=session.task_goals.current(pending['goal_id'])
    feedback=session.navigation.snapshot()
    reason=result.get('reason')
    footprint_check = None
    if result.get('status') in ('selected','observation'):
        observing = result['status']=='observation'
        binding=feedback.get('map_binding') or {}
        if ((observing and (session.navigation.scope!='manual' or 'manual_observation' not in pending)) or
                pending['revision']!=session.state['revision'] or spatial_context(feedback)!=pending['context'] or
                result.get('observation_binding')!=pending['observation_binding'] or
                result.get('local_identity')!=pending['context']['local_identity'] or
                result.get('clock_domain')!=feedback['clock_domain'] or not result.get('map_binding') or
                any(binding.get(k)!=v for k,v in result['map_binding'].items()) or
                result['map_binding'].get('map_frame')!=session.catalog.value['frame_id'] or
                not 0<=feedback['stamp_s']-result['geometry_stamp_s']<=1.5 or
                not session.observation_stop(feedback)):
            reason='seating_proposal_expired_or_context_changed'
        else:
            session.prepare_current_images()
            view=session.spatial_view.record['view_id'] if session.spatial_view else None
            checked=session.check_candidate(result['pose'],view)
            if not footprint_approved(session,pending,result,checked):
                reason='seating_footprint_not_currently_free'
                footprint_check = dict(status=checked.get('status'), reason=checked.get('reason'),
                    static=checked.get('static_footprint', {}).get('status'),
                    local=checked.get('local_footprint', {}).get('status'))
            else:
                destination=dict(goal_id=pending['goal_id'],task_goal_binding=goal['binding'],place_id=pending['goal_id'],
                    pose=result['pose'],orientation_source='semantic',view_id=view,reference='base_link',
                    frame_id=session.catalog.value['frame_id'],requirement=goal['requirement'])
                if 'round_table' in pending:
                    destination['map_keepout_circle'] = table_keepout_circle(pending['round_table'])
                    if not observing and 'manual_observation' in pending:
                        destination['use_local_approach'] = True
                outcome=session.skill.start(destination,session.state['revision'])
                if outcome['status'] in ('accepted','uncertain'):
                    if not observing:
                        goal['selected']=dict(copy.deepcopy(destination),handle=outcome['handle'],seating=copy.deepcopy(result))
                    session.observation=session.spatial_view=None
                    session.state['status']='waiting_navigation'
                    if outcome['status']=='accepted':
                        session.seating_active=dict(**pending,sequence=result.get('sequence',1),handle=copy.deepcopy(outcome['handle']))
                        if observing:
                            session.seating_active['observation_pose'] = result['pose']
                            goal.setdefault('seating_views',[]).append(result['pose'][:2])
                            session.conversation.notice('seating_observation_goal',pose=result['pose'],
                                                        focus=result.get('observation_focus'))
                else:reason=outcome.get('reason') or 'seating_navigation_refused'
    if 'manual_observation' in pending:
        for failure in result.get('view_checks', []):
            if failure['reason'] and failure['reason'] != 'seating_view_repeats_rejected_pose':
                remember_rejected_view(goal, failure['pose'], failure['reason'])
        if result.get('status') == 'observation' and footprint_check is not None:
            # A missing/stale observation is a transient check failure, not a geometric rejection.
            if footprint_check['status'] == 'available':
                cause = f"seating_view_footprint_static_{footprint_check['static']}_local_{footprint_check['local']}"
                remember_rejected_view(goal, result['pose'], cause)
    if session.seating_active is None:
        cancel_selection(session,pending)
    session.conversation.notice('seating_selection',goal_id=pending['goal_id'],selection_id=pending['selection_id'],
                                status=session.state['status'],reason=reason,model_reason=result.get('model_reason'))
    session._save('seating_selection',dict(request=pending,result=result,reason=reason,footprint_check=footprint_check))


def cancel_selection(session,selection=None):
    value=selection or session.seating_active or session.seating_pending
    session.seating_active=None
    if value:
        try:session.navigation.client.exchange('agent_seating_cancel',dict(selection_id=value['selection_id']))
        except (OSError,EOFError,ValueError):pass


def poll_observation_arrival(session, active):
    """Resume seating after reaching a view or stopping after a blocked approach."""
    skill = session.skill.record
    feedback = session.navigation.snapshot()
    if (active['revision']!=session.state['revision'] or skill.get('context_lost') or not skill.get('connected') or
            spatial_context(feedback)!=active['context']):
        session.stop('needs_clarification','seating_observation_context_changed')
        return
    if skill['handle']==active['handle']:
        if 'blocked_cancel_stamp_s' in active:
            return
        observation = session.skill.current_planning_observation(feedback)
        blockage = skill.get('blockage') or {}
        confirmed = (skill['status']=='blocked' and blockage.get('current') is True and
                     observation is not None and observation['status']=='blocked' and
                     session.skill.blockage_matches_feedback(blockage,feedback))
        reason = session.state.get('reason')
        if observation is not None and observation['status']=='blocked':
            if observation['reason']=='start_footprint_occupied':
                reason = ('seating_observation_start_occupied' if confirmed
                          else 'seating_observation_reference_waiting')
            elif confirmed:
                active['blocked_cancel_stamp_s'] = feedback['stamp_s']
                skill['recovery_cancel_stamp_s'] = feedback['stamp_s']
                session.skill.cancel(reason='seating_observation_path_blocked')
                reason = 'seating_observation_reselect_after_stop'
        elif observation is not None and observation['status']=='clear' and skill['status']=='running':
            reason = None
        if reason != session.state.get('reason'):
            session.state['reason'] = reason
            session._save('seating_observation_reference',dict(goal_id=active['goal_id'],reason=reason,
                          planning_observation=observation))
        return
    proof = skill['completed'][-1] if skill['completed'] else {}
    reselecting = 'blocked_cancel_stamp_s' in active
    if reselecting:
        stopped = skill.get('recovery_stop') or {}
        if skill['status']!='cancelled' or stopped.get('handle')!=active['handle']:
            session.stop('needs_clarification','seating_observation_blocked_stop_unavailable')
            return
        if not session.observation_stop(feedback):
            return
    elif (proof.get('handle')!=active['handle'] or not session.observation_stop(feedback) or
            not feedback.get('manual_arrival_observation')):
        session.stop('needs_clarification','seating_observation_arrival_unavailable')
        return
    if 'wait_started' not in active:
        active['wait_started'] = time.monotonic()
        # Require a new RGB and scan after the arrival/stop measurement.
        active['after_s'] = feedback['stamp_s']
    selection = {key: copy.deepcopy(active[key]) for key in ('instruction','round_table','observation_binding')}
    selection['selection_id'] = uuid.uuid4().hex
    goal = session.task_goals.current(active['goal_id'])
    selection['manual_observation'] = observation_history(goal, active['after_s'])
    ack = session.navigation.client.exchange('agent_seating_select',selection)
    if ack.get('error')=='seating_waiting_for_new_view' and time.monotonic()-active['wait_started']<8.:
        return
    if not ack.get('accepted') or ack.get('selection_id')!=selection['selection_id']:
        session.stop('needs_clarification',ack.get('error') or 'seating_observation_restart_refused')
        return
    session.seating_active = None
    session.seating_pending = dict(**selection,goal_id=active['goal_id'],revision=active['revision'],
                                   context=active['context'],started=time.monotonic())
    session.state.update(status='waiting_seating',reason=None)
    event = 'seating_observation_reselected' if reselecting else 'seating_observation_arrived'
    session._save(event,dict(goal_id=active['goal_id'],pose=active['observation_pose'],after_s=active['after_s']))


def poll_updates(session):
    active=session.seating_active
    if active is None:return
    if 'observation_pose' in active:
        poll_observation_arrival(session,active)
        return
    skill=session.skill.record
    status=skill['status']
    finished_manual=(status=='succeeded' and skill['handle'] is None and session.navigation.scope=='manual' and
                     bool(skill['completed']) and skill['completed'][-1].get('handle')==active['handle'])
    if (session.state['status'] not in ('ready','waiting_navigation','waiting_monitor') or
            (skill['handle']!=active['handle'] and not finished_manual) or
            status not in ('running','blocked','holding','held','succeeded')):
        cancel_selection(session);return
    # Do not replace the goal while a previously issued hold is being confirmed.
    if status=='holding':return
    result=session.navigation.value.get('seating') or {}
    if result.get('selection_id')!=active['selection_id'] or result.get('sequence',0)<=active['sequence']:return
    active['sequence']=result['sequence']
    feedback=session.navigation.snapshot();binding=feedback.get('map_binding') or {}
    reason=result.get('reason')
    if result.get('status')=='selected':
        if (active['revision']!=session.state['revision'] or spatial_context(feedback)!=active['context'] or
                result.get('observation_binding')!=active['observation_binding'] or
                result.get('local_identity')!=active['context']['local_identity'] or
                result.get('clock_domain')!=feedback.get('clock_domain') or not result.get('map_binding') or
                any(binding.get(k)!=v for k,v in result['map_binding'].items()) or
                not 0<=feedback['stamp_s']-result['geometry_stamp_s']<=1.5):
            reason='seating_proposal_expired_or_context_changed'
        else:
            goal=session.task_goals.current(active['goal_id'])
            session.prepare_current_images()
            view=session.spatial_view.record['view_id'] if session.spatial_view else None
            checked=session.check_candidate(result['pose'],view)
            if not footprint_approved(session,active,result,checked):
                reason='seating_footprint_not_currently_free'
            else:
                destination=dict(goal_id=active['goal_id'],task_goal_binding=goal['binding'],place_id=active['goal_id'],
                    pose=result['pose'],orientation_source='semantic',view_id=view,reference='base_link',
                    frame_id=session.catalog.value['frame_id'],requirement=goal['requirement'])
                if 'round_table' in active:
                    destination['map_keepout_circle'] = table_keepout_circle(active['round_table'])
                    if 'manual_observation' in active:
                        destination['use_local_approach'] = True
                old=goal['selected']['pose'];new=result['pose']
                angle=abs(math.atan2(math.sin(new[2]-old[2]),math.cos(new[2]-old[2])))
                if math.hypot(new[0]-old[0],new[1]-old[1])>.15 or angle>.15:
                    if status in ('held','succeeded'):
                        outcome=session.skill.start(destination,session.state['revision'])
                    else:
                        outcome=session.skill.retarget(destination)
                    if outcome['status']!='accepted':reason=outcome.get('reason') or 'seating_retarget_refused'
                    else:
                        active['handle']=copy.deepcopy(outcome['handle'])
                        goal['selected']=dict(copy.deepcopy(destination),handle=outcome['handle'],seating=copy.deepcopy(result))
                        session.state['status']='waiting_navigation'
                else:
                    goal['selected']['seating']=copy.deepcopy(result)
                session.observation=session.spatial_view=None
    if reason:
        cancel_selection(session)
        session.stop('needs_clarification',reason)
    session._save('seating_update',dict(selection_id=active['selection_id'],result=result,reason=reason))
