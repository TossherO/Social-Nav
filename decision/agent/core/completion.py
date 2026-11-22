"""Measured execution evidence shared by the model input and completion commit."""
import copy
import math

from .tasks import angle_error, inside
from ..tools.navigation import spatial_context


def navigation_facts(session, goal, state, selected, proofs, *, require_final_yaw=False):
    live_arrival = (session.navigation.scope == 'live' and state.get('stop_observation') is True and
                    state.get('chassis_state') == 'armed' and state.get('chassis_status_fresh') is True and
                    state.get('origin_frame_id') == session.catalog.value['frame_id'] and
                    state.get('origin_stamp_s') is not None and
                    0 <= state['stamp_s']-state['origin_stamp_s'] <= .5)
    simulation_arrival = (session.navigation.scope == 'simulation' and state.get('execution_feedback') is True and
                          state.get('frame_id') == session.catalog.value['frame_id'])
    manual_arrival = (session.navigation.scope == 'manual' and state.get('manual_arrival_observation') is True and
                      state.get('origin_frame_id') == session.catalog.value['frame_id'] and
                      state.get('origin_stamp_s') is not None and
                      0 <= state['stamp_s']-state['origin_stamp_s'] <= .5 and
                      bool(proofs) and (state.get('manual_arrival_evidence') or state.get('manual_confirmation') or {}).get('handle') == proofs[-1]['handle'])
    held = session.navigation.scope == 'live' and session.skill.record['status'] == 'held'
    if ((session.skill.record['handle'] is not None and not held) or not proofs or
            proofs[-1]['handle'] != selected.get('handle') or
            (held and session.skill.record['handle'] != proofs[-1]['handle']) or
            proofs[-1]['handle']['plan_revision'] != session.state['revision'] or
            proofs[-1].get('context')!=spatial_context(state) or proofs[-1].get('source')!=state.get('source') or
            not (simulation_arrival or live_arrival or manual_arrival) or
            not state.get('stopped') or state.get('input_stale') or
            state.get('localization_status','valid') != 'valid'):
        return dict(status='unproven', reason='matching_task_navigation_and_stop_not_proven')
    proof = copy.deepcopy(proofs[-1])
    if manual_arrival:
        proof['current_manual_arrival_evidence']=copy.deepcopy(state.get('manual_arrival_evidence') or state.get('manual_confirmation'))
    measured_pose = state.get('origin_pose') if live_arrival or manual_arrival else state.get('pose')
    if measured_pose is None:
        return dict(status='unproven',reason='current_base_pose_missing')
    checks = session.task_goals.geometry(goal['requirement'], measured_pose, selected)
    if goal['requirement']['kind'] in ('route','sequence'):
        from .route import validate_route_progress, visited_checkpoints
        progress=validate_route_progress(selected.get('route'),proof)
        if progress['status']!='met':
            return dict(status='unproven',reason=progress['reason'],route_progress=progress)
        checks['route_progress']=progress
        if goal['requirement']['kind']=='sequence':
            checkpoints=goal['requirement']['checkpoints']
            count=visited_checkpoints(goal.get('route_attempts',[]),checkpoints,session.task_goals.checkpoint_met,
                                     current_visits=progress['visits'],handle=proof['handle'],context=proof['context'])
            checks['sequence_checkpoint_progress']=dict(confirmed_count=count,required_count=len(checkpoints))
            if count!=len(checkpoints):
                return dict(status='unproven',reason='ordered_user_checkpoint_evidence_incomplete',checks=checks)
    if goal['requirement']['kind']=='reference' or selected.get('nearby_stop'):
        import numpy as np
        actual=np.asarray(measured_pose,float);chosen=np.asarray(selected.get('pose'),float)
        checks['selected_position_met']=bool(actual.shape==(3,) and chosen.shape in ((2,),(3,)) and
            np.isfinite(actual).all() and np.isfinite(chosen).all() and
            np.linalg.norm(actual[:2]-chosen[:2])<=session.task_goals.settings['goal_position_tolerance_m'])
    if any(value is False for value in checks.values()):
        return dict(status='not_met', checks=checks)
    # A valid first stop can supply visual review evidence before the final yaw is reached.
    final_yaw = goal['requirement'].get('required_yaw')
    if final_yaw is not None and 'required_yaw_met' not in checks:
        checks['required_yaw_met'] = session.task_goals.yaw_met(measured_pose,final_yaw)
        if require_final_yaw and checks['required_yaw_met'] is False:
            return dict(status='not_met', checks=checks)

    return dict(status='available', proof=proof, checks=checks, measured_pose=measured_pose)


def same_pose(first, second, settings):
    if not all(isinstance(pose,(tuple,list)) and len(pose)==3 and
               all(type(v) in (int,float) and math.isfinite(v) for v in pose) for pose in (first,second)):
        return False
    return (math.dist(first[:2],second[:2]) <= settings['goal_position_tolerance_m'] and
            angle_error(first[2],second[2]) <= settings['goal_yaw_tolerance_rad'])


def stage_evidence(session, goal):
    req = goal['requirement']
    stage = req.get('stage_type','navigation')
    state = session.navigation.snapshot()
    settings = session.task_goals.settings
    selected = goal.get('selected') or {}
    proofs = [p for p in session.skill.record['completed'] if p['goal_id']==req['goal_id']]
    pose = state.get('pose') if session.navigation.scope=='simulation' else state.get('origin_pose')
    facts = dict(requirement=copy.deepcopy(req), scope=session.navigation.scope,
                 measured_pose=pose, frame_id=session.catalog.value['frame_id'], stopped=state.get('stopped'),
                 measurement_stamp_s=state.get('origin_stamp_s') if session.navigation.scope!='simulation' else state['stamp_s'],
                 source=state.get('source'), clock_domain=state['clock_domain'], missing=[])
    proof = {}
    missing = facts['missing']
    entered = goal.get('entered') or {}
    earliest = entered.get('stamp_s')
    if state.get('input_stale') or state.get('localization_status','valid')!='valid':
        missing.append('fresh_localization')
    if session.navigation.scope=='shadow':
        missing.append('execution_feedback_unavailable_in_shadow')
    completion_catalog = session.catalog
    rgb = (session.observation.record if session.observation else {})
    view = (session.spatial_view.record if session.spatial_view else {})
    if stage in ('navigation','seating'):
        arrival = navigation_facts(session,goal,state,selected,proofs)
        proof = arrival.pop('proof',{})
        facts.update(arrival)
        if selected.get('pose') and pose:
            target=selected['pose']
            facts['distance_to_selected_m']=math.dist(pose[:2],target[:2])
            if len(target)==3:
                facts['yaw_error_rad']=angle_error(pose[2],target[2])
            if req['kind'] in ('route','sequence') and facts['distance_to_selected_m']>settings['goal_position_tolerance_m']:
                missing.append('route_endpoint')
        if arrival['status']!='available':
            missing.append(arrival.get('reason','navigation_geometry'))
        elif facts['checks'].get('required_yaw_met') is False:
            missing.append('required_yaw')
        if stage=='seating':
            selection=selected.get('seating')
            if not selection or not same_pose(pose,selected.get('pose'),settings):
                missing.append('seating_selected_pose')
            else:
                import numpy as np
                edge=selection['mapped_edge']
                start,end=np.asarray(edge['endpoints_xy_m'])
                tangent=end-start
                length=float(np.linalg.norm(tangent))
                corners=session.navigation.geometry.polygon(pose)-start
                along=corners@(tangent/length)
                normal=[math.cos(edge['interior_yaw_rad']),math.sin(edge['interior_yaw_rad'])]
                gap=float(-max(corners@normal))
                within=bool(min(along)>=0 and max(along)<=length)
                facts['table_selection']={k:selection[k] for k in ('candidate_id','candidate_binding','mapped_edge','source_stamp_s')}
                facts['table_selection'].update(measured_front_gap_m=gap,inside_observed_span=within)
                if gap<0 or not within:
                    missing.append('seating_measured_edge_clearance')
        if proof:
            earliest = proof['stamp_s']
            facts['arrival_stamp_s'] = earliest
            earliest = proof.get('stop_stamp_s',earliest)
            facts['handle'] = proof['handle']
        if req.get('facing') or (req['kind']=='reference' and req.get('target_id') and 'max_distance_m' not in req
                                 and not req.get('semantic_conditions') and not req.get('visual_preferences')):
            if (not view or pose is None or earliest is None or view['source_stamp_s']<earliest or
                    not same_pose(view.get('pose'),pose,settings)):
                missing.append('current_post_arrival_map')
            if req.get('facing') and facts.get('yaw_error_rad',math.inf)>settings['goal_yaw_tolerance_rad']:
                missing.append('selected_heading')
    elif stage=='follow':
        follow = state.get('follow_status') or {}
        target = session._observed_follow_target(follow.get('target_identity'),follow.get('target_generation'))
        facts['observed_follow_target'] = target
        if not target or follow.get('status')!='running' or follow.get('handle')!=selected.get('handle'):
            missing.append('current_follow_target')
    elif stage=='elevator':
        proof = goal.get('destination_localization') or {}
        association = proof.get('association') or {}
        binding = state.get('map_binding') or {}
        facts['destination_association'] = association
        if (not proof or session.catalog.value.get('floor_id')!=req['to_floor'] or not binding.get('valid') or
                any(association.get(k)!=binding.get(k) for k in ('map_id','association_session','association_generation')) or
                [association.get(k) for k in ('local_session_id','local_generation','local_frame')]!=
                [state.get(k) for k in ('session_id','generation','frame_id')]):
            missing.append('destination_map_association')
            completion_catalog = None
        if not session._confirmed_manual_idle(state):
            missing.append('destination_stop')
    elif stage=='observation':
        if not session.observation_stop(state):
            missing.append('confirmed_stop')
        proof = dict(scope=session.navigation.scope,stamp_s=state['stamp_s'],clock_domain=state['clock_domain'],
                     stopped=state.get('stopped'),physical_acceptance=False)
    else:
        missing.append('stage_skill_not_available')
    region_id = req.get('robot_inside_region_id')
    if region_id:
        region = completion_catalog.regions.get(region_id) if completion_catalog is not None else None
        stamp=facts['measurement_stamp_s']
        frame=state.get('frame_id') if session.navigation.scope=='simulation' else state.get('origin_frame_id')
        valid=(region is not None and frame==facts['frame_id'] and
               not state.get('input_stale') and state.get('localization_status','valid')=='valid' and
               isinstance(pose,(tuple,list)) and len(pose)==3 and
               all(type(v) in (int,float) and math.isfinite(v) for v in pose) and
               type(stamp) in (int,float) and type(state.get('stamp_s')) in (int,float) and
               0<=state['stamp_s']-stamp<=.5)
        if session.navigation.scope!='simulation':
            binding=state.get('map_binding') or {}
            valid=valid and binding.get('valid') is True and binding.get('map_frame')==facts['frame_id']
        region_state='unknown' if not valid else 'inside' if inside(pose[:2],region['polygon']) else 'outside'
        facts['completion_region']=dict(floor_id=req.get('to_floor',session.catalog.value.get('floor_id')),
                                        region_id=region_id,status=region_state,stamp_s=stamp if valid else None)
        if region_state!='inside':missing.append('completion_region_'+region_state)
    adjustments = goal.get('visual_adjustments',[])
    adjustment = adjustments[-1] if adjustments else None
    needs_rgb = stage in ('observation','follow','seating') or bool(req.get('semantic_conditions') or adjustment)
    rgb_stamp = (rgb.get('rgb') or {}).get('stamp_s')
    rgb_source_matches = (rgb.get('local_identity')==spatial_context(state)['local_identity'] and
                          (state.get('map_binding') is None or rgb.get('map_binding')==state['map_binding']))
    if (rgb and rgb['rgb'].get('status')=='valid' and rgb['clock_domain']==state['clock_domain'] and
            (stage!='navigation' or rgb_source_matches) and
            earliest is not None and type(rgb_stamp) in (int,float) and earliest<=rgb_stamp<=state['stamp_s'] and
            (adjustment is None or rgb_stamp>adjustment['review']['source_stamp_s']) and
            (stage=='follow' or same_pose(rgb.get('base_pose'),state.get('pose'),settings))):
        facts['rgb'] = dict(observation_id=rgb['observation_id'],source_stamp_s=rgb_stamp,
                            age_s=state['stamp_s']-rgb_stamp,input_fingerprint=rgb['input_fingerprint'])
    elif needs_rgb:
        missing.append('current_rgb')
    review = goal.get('required_visual_review')
    if (stage=='navigation' and review and
            review['handle']==selected.get('handle') and
            (not facts.get('rgb') or facts['rgb']['source_stamp_s']<=review['source_stamp_s'])):
        missing.append('new_rgb_after_unsatisfied_visual_review')
    if view:
        facts['map_view'] = dict(view_id=view['view_id'],source_stamp_s=view['source_stamp_s'])
    return facts, proof
