"""Operator-driven Agent testing against a read-only live observation runtime."""
import json
import copy
import math
import queue
import subprocess
from pathlib import Path
import time
from ..adapters.manual import ManualNavigation
from ..adapters.runtime import ManualRuntimeClient
from ..core.session import AgentSession
from ..core.tasks import next_unfinished_goal
from ..core.store import Store, shared_budget
from ..tools.places import Catalog
from .common import emit, make_loop
from .interaction import Console
from .manual_reference import ManualReference,reference_request


class RouteDisplay:
    """ROS runs in the system interpreter; the Agent sends display data only."""
    def __init__(self, frame):
        self.frame=frame
        self.process=None
        self.failed=False

    def send(self, points, route_index=None, final_yaw=None):
        if self.failed:return
        if self.process is None:
            try:
                self.process=subprocess.Popen(
                    ['bash','-c','source perception/scripts/ros1_env.sh && exec /usr/bin/python3 -B -m decision.agent.workflows.manual_rviz'],
                    stdin=subprocess.PIPE, text=True)
            except OSError:
                self.failed=True
                emit(dict(event='manual_rviz_unavailable',reason='marker_bridge_start_failed'))
                return
        try:
            self.process.stdin.write(json.dumps(dict(frame_id=self.frame,points=points,route_index=route_index,
                                                     final_yaw=final_yaw),allow_nan=False)+'\n')
            self.process.stdin.flush()
        except (BrokenPipeError,OSError,ValueError):
            self.failed=True
            emit(dict(event='manual_rviz_unavailable',reason='marker_bridge_disconnected'))

    def check(self):
        if self.process is not None and not self.failed and self.process.poll() is not None:
            self.failed=True
            emit(dict(event='manual_rviz_unavailable',reason='marker_bridge_exited'))

    def close(self):
        if self.process is None:return
        try:self.process.stdin.close()
        except (BrokenPipeError,OSError):pass
        try:self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()


def display_points(grid, points):
    """Use supported ground heights; mark map-median fallbacks as approximate."""
    import numpy as np
    from perception.visualization.surface import sample_ground
    heights,sources=sample_ground(grid,points,offset=.12)
    approx=np.flatnonzero(sources==5).tolist()
    if 'ground_z' in grid.layers and 'ground_support' in grid.layers:
        ground=grid.layers['ground_z'];support=grid.layers['ground_support']&np.isfinite(ground)
        if support.any():
            fallback=float(np.median(ground[support]))+.12
            for i,z in enumerate(heights):
                if not np.isfinite(z):heights[i]=fallback;approx.append(i)
    return ([[float(p[0]),float(p[1]),float(z) if np.isfinite(z) else None]
             for p,z in zip(points,heights)],approx)


def display_final_yaw(pose, route):
    """Show a selected single-goal heading or an explicitly selected route heading."""
    if route is not None:return route.get('final_yaw')
    return float(pose[2]) if len(pose)==3 else None


def floor_files(floor):
    import yaml
    from .. import ROOT
    filename={'floor1':'interactive_agent_floor1.yaml','floor2':'interactive_agent.yaml'}[floor]
    path=ROOT/'navigation/commissioning'/filename
    return path,ROOT/yaml.safe_load(path.read_text())['catalog']


def floor_catalogs_for_manual(catalog):
    """Load both configured and map-validated catalogs for this wheelchair."""
    from ..adapters.shadow import load_catalog_map
    configs={}
    catalogs={}
    for floor in ('floor1','floor2'):
        configs[floor],catalog_path=floor_files(floor)
        selected=Catalog.load(catalog_path)
        if floor==catalog.value.get('floor_id') and selected.fingerprint!=catalog.fingerprint:
            raise ValueError('Cross-floor mode requires the configured, matching floor catalog')
        asset=load_catalog_map(selected)
        asset.close()
        catalogs[floor]=selected
    if catalog.value.get('floor_id') not in catalogs:
        raise ValueError('Cross-floor mode requires a configured floor')
    return catalogs,configs


def run(args):
    if args.duration <= 0 or not args.interactive:
        raise ValueError('Manual driving requires interactive input and a positive duration')
    catalog=Catalog.load(args.catalog,condition=args.map_condition)
    from ..core.reuse import load_registered_task,verify_registered_task,catalog_assets,import_registered_task
    reuse_path=getattr(args,'reuse_session',None)
    if not reuse_path and getattr(args,'reuse_digest',None):raise ValueError('历史任务校验值需要对应的历史会话')
    plan=load_registered_task(reuse_path) if reuse_path else None
    if plan is not None:
        if getattr(args,'reuse_digest',None) and args.reuse_digest!=plan['digest']:
            raise ValueError('历史任务在预览后发生变化，请重新选择')
        if args.instruction and args.instruction!=plan['instruction']:
            raise ValueError('复用历史登记时任务指令必须与历史指令一致')
        args.instruction=plan['instruction']
        if args.actions:raise ValueError('复用历史登记不能与动作回放同时使用')
    elif not args.instruction or not args.instruction.strip():
        raise ValueError('请输入任务指令，或选择历史任务登记')
    cross_floor=bool(getattr(args,'cross_floor',False))
    catalogs=configs=None
    if cross_floor:
        if args.actions:raise ValueError('Cross-floor live monitoring requires a model; --actions is not supported')
        from .floor_handoff import observation_owner
        observation_owner(args.runtime_socket)
        catalogs,configs=floor_catalogs_for_manual(catalog)
    configured=catalogs or {catalog.value.get('floor_id'):catalog}
    verified_assets=verify_registered_task(plan,configured,catalog.value.get('floor_id')) if plan is not None else None
    navigation=ManualNavigation(catalog,ManualRuntimeClient(args.runtime_socket),automatic_arrival=True)
    state=navigation.snapshot()
    deadline=time.monotonic()+3.
    while (state.get('origin_pose') is None and not state.get('input_stale') and
           state.get('localization_status')=='valid' and time.monotonic()<deadline):
        time.sleep(.05)
        state=navigation.tick()
    if (state.get('input_stale') or state.get('localization_status')!='valid' or
            state.get('origin_pose') is None or state.get('origin_frame_id')!=catalog.value['frame_id']):
        raise ValueError('Current mapped localization and observation are required')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    store=Store(output/'session.sqlite')
    actor=None
    display=RouteDisplay(catalog.value['frame_id'])
    reference=ManualReference(args.runtime_socket)
    equipment=None
    recorder=None
    source_socket=args.runtime_socket
    try:
        session=AgentSession(args.instruction,catalog,navigation,store,use_memory=not args.no_memory,
                             floor_catalogs=catalogs)
        assets=verified_assets if verified_assets is not None else catalog_assets(configured)
        store.event('task_asset_provenance',dict(floors=assets))
        if plan is not None:
            import_registered_task(session,plan,assets,reuse_path)
        if cross_floor:
            from .floor_handoff import ElevatorRide
            equipment=ElevatorRide(configs,source_socket,output)
            session.elevator=equipment
        transient_queue=queue.Queue(maxsize=32)
        def transient_sink(value):
            try:transient_queue.put_nowait(value)
            except queue.Full:pass
        actor=make_loop(args,session,output/'session.sqlite',
                        transient_sink=transient_sink if getattr(args,'gui_commands',False) else None)
        if getattr(args,'record_bag',False):
            from .task_bag import TaskBagRecorder
            recorder=TaskBagRecorder(output,session,emit,output_root=getattr(args,'bag_output_root','/home/tl/rosbag'))
            actor.before_request=recorder.before_request
            session.elevator_start_ready=lambda:recorder.before_request(actor)
        last=None
        last_progress=None
        last_observation=None
        last_status_at=0.
        last_reference=None
        transient_phase=None
        trace_cursor=None
        trace_saved_at=0.
        def save_trace(force=False):
            nonlocal trace_cursor,trace_saved_at
            if navigation.route_start_stamp_s is None or (not force and time.monotonic()-trace_saved_at<2.):
                return
            state=navigation.snapshot()
            segments=session.trajectory.snapshot(frame_id=session.catalog.value['frame_id'],
                clock_domain=state['clock_domain'],now=state['stamp_s'])
            captured=[]
            for segment in segments:
                samples=[p for p in segment['samples'] if p['stamp_s']>=navigation.route_start_stamp_s and
                         (trace_cursor is None or p['stamp_s']>trace_cursor)]
                if samples:captured.append(dict(segment,samples=samples))
            if captured:
                session.store.event('manual_route_trace_samples',dict(task_id=session.state['task_id'],
                    revision=session.state['revision'],handle=navigation.handle,segments=captured,
                    source='measured_localization_not_planned_route'))
                trace_cursor=max(p['stamp_s'] for segment in captured for p in segment['samples'])
            trace_saved_at=time.monotonic()
        def report(_actor):
            nonlocal last,last_progress,last_observation,trace_cursor,trace_saved_at,last_status_at,last_reference
            nonlocal navigation,reference,source_socket
            nonlocal transient_phase
            if recorder is not None:recorder.update()
            ride=session.elevator if session.state.get('elevator_active') else None
            phase=ride.transition.status if ride and ride.transition else None
            if phase not in ('monitoring_ride','awaiting_destination_localization'):phase=None
            if transient_phase and transient_phase!=phase:
                emit(dict(event='elevator_transient_clear'))
            transient_phase=phase
            while True:
                try:value=transient_queue.get_nowait()
                except queue.Empty:break
                if phase and value.get('phase')==phase:
                    emit(dict(event='elevator_transient',data={k:v for k,v in value.items() if k!='phase'}))
            stage=next_unfinished_goal(session.task_goals.goals)
            requirement=None if stage is None else stage['requirement']
            if session.navigation is not navigation:
                navigation=session.navigation
                source_socket=session.elevator.socket_path
                reference.close()
                reference=ManualReference(source_socket)
                last=last_progress=last_reference=trace_cursor=None
                display.send([])
                display.frame=session.catalog.value['frame_id']
                emit(dict(event='destination_localization_available',floor_id=session.catalog.value['floor_id'],
                          runtime_socket=source_socket))
            if session.state.get('elevator_active'):
                reference.set(None,None)
                if time.monotonic()-last_status_at>=.5:
                    last_status_at=time.monotonic()
                    emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                        goal_id=requirement['goal_id'],goal_description=requirement['description'],
                        floor_id=requirement['floor_id'],call_id=None,status=session.state['status'],reason=session.state.get('reason'),
                        navigation_status='elevator',localization_status='unavailable',input_stale=True,
                        elevator_status=session.elevator.transition.status,model_busy=_actor.model.busy,
                        elevator_standard_start=session.elevator.standard is not None,
                        elevator_manual_selection=(session.elevator.last_attempt or {}).get('source')=='operator_rviz',
                        elevator_decision=session.elevator.feedback,
                        error=(session.elevator.wait_reason or (session.elevator.last_attempt or {}).get('result')
                               or session.state.get('reason')),
                        floor_handoff=session.elevator.handoff.snapshot() if session.elevator.handoff else None))
                return
            display.check()
            record=session.skill.record
            handle=record.get('handle')
            following=record['stage_type']=='follow'
            association_waiting=record['association_waiting']
            active=record['status'] in ('running','blocked') and handle is not None
            request=None
            if active and not following and not association_waiting:
                request=reference_request(handle,record['selected_pose'],navigation.route,
                                          navigation.route_index,session.catalog,
                                          require_yaw=navigation.request['require_yaw'])
                reference.set(handle['call_id'],request)
            else:reference.set(None,None)
            if active:
                preview=(navigation.value.get('control') or {}).get('manual_preview') or {}
                error=reference.status()
                index=0 if navigation.route is None else navigation.route_index
                expected=handle['call_id'] if following else handle['call_id']+':'+str(index)
                if (preview.get('owner')!=handle['call_id'] or preview.get('request_id')!=expected or
                        not following and (request is None or preview.get('goal_revision')!=request['goal_revision'])):
                    preview={}
                summary=dict(status='unavailable' if error else preview.get('status','waiting'),
                    reason=error or (preview.get('planning_observation') or {}).get('reason') or preview.get('reason'),
                    reference_stamp_s=preview.get('reference_stamp_s'))
                key=(handle['call_id'],summary['status'],summary['reason'])
                if key!=last_reference:
                    last_reference=key
                    event=dict(event='manual_reference_status',goal_id=record['goal_id'],**summary)
                    session.store.event('manual_reference_status',event);emit(event)
            else:last_reference=None
            if getattr(args,'gui_commands',False) and time.monotonic()-last_status_at>=.5:
                last_status_at=time.monotonic()
                feedback=navigation.snapshot()
                current=next_unfinished_goal(session.task_goals.goals)
                selected=None if current is None else (current.get('selected') or {}).get('handle')
                completed=record.get('completed') or []
                recovery_confirmation=(handle is None and selected is not None and navigation.handle==selected and
                    navigation.status=='running' and completed and completed[-1].get('handle')==selected and
                    completed[-1].get('goal_id')==current['requirement']['goal_id'])
                confirmation_handle=selected if recovery_confirmation else handle
                pose=feedback.get('origin_pose');target=record.get('selected_pose')
                stamp=feedback.get('origin_stamp_s')
                valid_pose=(not feedback.get('input_stale') and feedback.get('localization_status')=='valid' and
                    feedback.get('origin_frame_id')==session.catalog.value['frame_id'] and
                    isinstance(stamp,(int,float)) and 0<=feedback['stamp_s']-stamp<=.5 and pose is not None)
                distance=(math.hypot(pose[0]-target[0],pose[1]-target[1]) if valid_pose and target is not None else None)
                yaw_error=(abs(math.atan2(math.sin(pose[2]-target[2]),math.cos(pose[2]-target[2])))
                           if valid_pose and target is not None and (navigation.request or {}).get('require_yaw') else None)
                dwell=(max(0.,stamp-navigation.arrival_since) if stamp is not None and navigation.arrival_since is not None else None)
                emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                    goal_id=None if current is None else current['requirement']['goal_id'],
                    goal_description=None if current is None else current['requirement'].get('description'),
                    completion_region_id=None if current is None else current['requirement'].get('robot_inside_region_id'),
                    floor_id=session.catalog.value.get('floor_id'),goal_distance_m=distance,
                    goal_tolerance_m=navigation.settings['goal_position_tolerance_m'],
                    goal_yaw_error_rad=yaw_error,goal_yaw_tolerance_rad=navigation.settings['goal_yaw_tolerance_rad'],
                    arrival_dwell_s=dwell,arrival_dwell_required_s=navigation.settings['goal_dwell_s'],
                    monitor=session.monitor.snapshot(),
                    call_id=None if handle is None else handle['call_id'],status=session.state['status'],
                    confirmation_call_id=None if confirmation_handle is None else confirmation_handle['call_id'],
                    manual_confirmation_available=bool(not association_waiting and not following and
                        confirmation_handle is not None and navigation.status=='running'),
                    reason=session.state.get('reason'),navigation_status=record['status'],
                    blockage=copy.deepcopy(record['blockage']),
                    detour_preview_conflict=(feedback.get('planning_observation') or {}).get('reason')=='detour_preview_crosses_prior_blockage',
                    stage_type=record['stage_type'],follow_status=copy.deepcopy(feedback.get('follow_status')),
                    model_busy=_actor.model.busy,localization_status=feedback.get('localization_status'),
                    input_stale=feedback.get('input_stale'),origin_pose=feedback.get('origin_pose'),
                    environment_fault=feedback.get('environment_fault'),
                    association_waiting=association_waiting,
                    source_stamp_s=feedback.get('origin_stamp_s'),stopped=feedback.get('stopped'),
                    route_index=feedback.get('route_index'),route_count=feedback.get('route_count'),
                    reference_preview={k:v for k,v in ((navigation.value.get('control') or {}).get('manual_preview') or {}).items()
                                       if k!='reference_path'},
                    reference_preview_error=reference.status()))
            if last is not None and navigation.route is not None:
                save_trace(force=not active)
            if association_waiting:
                if last is not None:display.send([])
                last=last_progress=None
                return
            if following and handle is not None:
                current_preview=(navigation.value.get('control') or {}).get('manual_preview') or {}
                target=current_preview.get('goal') if current_preview.get('owner')==handle['call_id'] else None
                key=(handle['call_id'],tuple(target or ()))
                if last!=key:
                    last=key
                    if target:
                        geometry,_=display_points(navigation.spatial_inputs()['global_grid'],[target[:2]])
                        display.send(geometry,None,target[2] if len(target)>2 else None)
                    else:display.send([])
            elif active and handle['call_id']!=last:
                last=handle['call_id']
                trace_cursor=None;trace_saved_at=0.
                pose=record['selected_pose']
                route=navigation.route
                points=route['waypoints'] if route else [pose[:2]]
                preview=None
                missing_heights=None
                approximate_heights=None
                try:
                    from ..tools.spatial import display_extent,render
                    inputs=navigation.spatial_inputs()
                    current=inputs.get('pose')
                    bounds=display_extent(inputs['global_grid'],session.catalog.annotations(),
                        extras=points+([] if current is None else [current[:2]]))['bounds']
                    preview=output/('target_%04d.png' % handle['goal_revision'])
                    preview.write_bytes(render(session.catalog,inputs['global_grid'],bounds,current,
                        geometry=navigation.geometry,local_view=True,static_crop=True,
                        candidate_preview=dict(pose=pose),
                        route_preview=None if route is None else dict(waypoints=points,diagnostics={})))
                    display_geometry,approximate_heights=display_points(inputs['global_grid'],points)
                    missing_heights=[i for i,point in enumerate(display_geometry) if point[2] is None]
                    display.send(display_geometry,navigation.route_index,display_final_yaw(pose,route))
                except (ValueError,KeyError,TypeError,OSError):
                    preview=None
                emit(dict(event='manual_target',goal_id=record['goal_id'],place_id=record['place_id'],
                    target_map_xy_yaw=pose,yaw_degrees=(round(math.degrees(pose[2]),1) if len(pose)==3 else None),
                    route_waypoints=points if route else None,
                    rviz_missing_height_indices=missing_heights,
                    rviz_approximate_height_indices=approximate_heights,
                    target_image=None if preview is None else str(preview),
                    position_tolerance_m=navigation.settings['goal_position_tolerance_m'],
                    yaw_required=navigation.request['require_yaw'],
                    yaw_tolerance_degrees=(round(math.degrees(navigation.settings['goal_yaw_tolerance_rad']),1)
                                           if navigation.request['require_yaw'] else None),
                    instruction='使用原车手柄按建议路线驾驶；到达后依据实时定位和停稳估计自动检查。路线仅供参考。'))
            if active and navigation.route is not None:
                progress=(handle['call_id'],navigation.route_index)
                if progress!=last_progress:
                    last_progress=progress
                    event=dict(event='manual_route_progress',goal_id=record['goal_id'],
                               next_waypoint_index=navigation.route_index,
                               waypoint_count=len(navigation.route['waypoints']),index_base=0,
                               source='localization_estimate',completion_not_proven=True,
                               waypoint_visits=copy.deepcopy(navigation.route_visits))
                    session.store.event('manual_route_progress',event)
                    emit(event)
                    try:
                        inputs=navigation.spatial_inputs()
                        geometry,_=display_points(inputs['global_grid'],navigation.route['waypoints'])
                        display.send(geometry,navigation.route_index,
                                     display_final_yaw(record['selected_pose'],navigation.route))
                    except (ValueError,KeyError,TypeError,OSError):pass
            elif last is not None and not active:
                display.send([]);last=None;last_progress=None
            stage=next_unfinished_goal(session.task_goals.goals)
            key=None if stage is None else (session.state['revision'],stage['requirement']['goal_id'])
            if stage is not None and stage['requirement'].get('stage_type')=='observation' and key!=last_observation:
                last_observation=key
                emit(dict(event='manual_observation_stage',goal_id=key[1],
                          instruction='请停稳，Agent 将使用本阶段的当前画面进行观察。'))
        actor.run(console=Console(structured=getattr(args,'gui_commands',False)),timeout_s=args.duration,on_tick=report,emit=emit)
        if navigation.route is not None:
            save_trace(force=True)
        result=dict(scope='operator_driven_localization_estimate',task=session.state,exit_reason=actor.exit_reason,
                    decision_turns=actor.requests,autonomous_navigation_verified=False,physical_acceptance=False)
        if not args.actions:
            budget=shared_budget();result['api_budget']=budget.summary();budget.close()
        (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        emit(dict(scope=result['scope'],status=session.state['status'],exit_reason=actor.exit_reason,
                  autonomous_navigation_verified=False))
        return result
    finally:
        if recorder:recorder.close()
        if actor:actor.close()
        if equipment:equipment.close()
        reference.close()
        display.close()
        store.close()
