"""Apply current T3 proposals to their original seating stage."""
import copy
import math
import time
import uuid

from ..tools.navigation import spatial_context


def start_selection(session,goal,instruction,roi):
    nav=session.navigation
    context=session._applying_context
    if nav.scope not in ('manual','live') or not hasattr(nav,'client'):
        return dict(status='unavailable',reason='stage_skill_not_available',stage_type='seating')
    if (not roi or not context or not session._current_rgb_context(context,session.observation,nav.snapshot()) or
            not session.observation_stop(nav.snapshot())):
        return dict(status='refused',reason='seating_requires_stopped_vehicle_and_current_table_roi')
    if session.skill.record['handle'] and not session._confirmed_live_hold(nav.snapshot()):
        return dict(status='refused',reason='cancel_and_confirm_existing_call_first')
    selection=dict(selection_id=uuid.uuid4().hex,observation_binding=context['observation_binding'],
                   instruction=instruction,roi=roi)
    ack=nav.client.exchange('agent_seating_select',selection)
    if not ack.get('accepted') or ack.get('selection_id')!=selection['selection_id']:
        return dict(status='unavailable',reason=ack.get('error') or 'seating_selection_rejected')
    session.seating_pending=dict(**selection,goal_id=goal['requirement']['goal_id'],
        revision=session.state['revision'],context=spatial_context(nav.snapshot()),started=time.monotonic())
    session.state['status']='waiting_seating'
    return dict(status='selecting',selection_id=selection['selection_id'])


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
    if result.get('status')=='selected':
        binding=feedback.get('map_binding') or {}
        if (pending['revision']!=session.state['revision'] or spatial_context(feedback)!=pending['context'] or
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
            if (checked.get('status')!='available' or checked['static_footprint']['status']!='free' or
                    checked['local_footprint']['status']!='free'):
                reason='seating_footprint_not_currently_free'
            else:
                destination=dict(goal_id=pending['goal_id'],task_goal_binding=goal['binding'],place_id=pending['goal_id'],
                    pose=result['pose'],orientation_source='semantic',view_id=view,reference='base_link',
                    frame_id=session.catalog.value['frame_id'],requirement=goal['requirement'])
                outcome=session.skill.start(destination,session.state['revision'])
                if outcome['status'] in ('accepted','uncertain'):
                    goal['selected']=dict(copy.deepcopy(destination),handle=outcome['handle'],seating=copy.deepcopy(result))
                    session.observation=session.spatial_view=None
                    session.state['status']='waiting_navigation'
                    if outcome['status']=='accepted':
                        session.seating_active=dict(**pending,sequence=result.get('sequence',1),handle=copy.deepcopy(outcome['handle']))
                else:reason=outcome.get('reason') or 'seating_navigation_refused'
    if session.seating_active is None:
        cancel_selection(session,pending)
    session.conversation.notice('seating_selection',goal_id=pending['goal_id'],selection_id=pending['selection_id'],
                                status=session.state['status'],reason=reason)
    session._save('seating_selection',dict(request=pending,result=result,reason=reason))


def cancel_selection(session,selection=None):
    value=selection or session.seating_active or session.seating_pending
    session.seating_active=None
    if value:
        try:session.navigation.client.exchange('agent_seating_cancel',dict(selection_id=value['selection_id']))
        except (OSError,EOFError,ValueError):pass


def poll_updates(session):
    active=session.seating_active
    if active is None:return
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
            if (checked.get('status')!='available' or checked['static_footprint']['status']!='free' or
                    checked['local_footprint']['status']!='free'):
                reason='seating_footprint_not_currently_free'
            else:
                destination=dict(goal_id=active['goal_id'],task_goal_binding=goal['binding'],place_id=active['goal_id'],
                    pose=result['pose'],orientation_source='semantic',view_id=view,reference='base_link',
                    frame_id=session.catalog.value['frame_id'],requirement=goal['requirement'])
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
