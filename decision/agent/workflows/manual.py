"""Operator-driven Agent testing against a read-only live observation runtime."""
import json
import copy
import math
import subprocess
from pathlib import Path
import time
from ..adapters.manual import ManualNavigation
from ..adapters.runtime import ManualRuntimeClient
from ..core.session import AgentSession
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
    return float(pose[2]) if route is None else route.get('final_yaw')


def floor_catalogs_for_manual(catalog):
    """Load both configured and map-validated catalogs for this wheelchair."""
    import yaml
    from .. import ROOT
    from ..adapters.shadow import load_catalog_map
    configs={'floor1':ROOT/'navigation/commissioning/interactive_agent_floor1.yaml',
             'floor2':ROOT/'navigation/commissioning/interactive_agent.yaml'}
    catalogs={}
    for floor,path in configs.items():
        settings=yaml.safe_load(path.read_text())
        catalog_path=ROOT/settings.get('catalog',f'outputs/decision/agent/lab_floors_20261114/{floor}_catalog.json')
        selected=catalog if floor==catalog.value.get('floor_id') else Catalog.load(catalog_path)
        if selected.fingerprint!=Catalog.load(catalog_path).fingerprint:
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
    cross_floor=bool(getattr(args,'cross_floor',False))
    catalogs=configs=None
    if cross_floor:
        if args.actions:raise ValueError('Cross-floor live monitoring requires a model; --actions is not supported')
        from .floor_handoff import observation_owner
        observation_owner(args.runtime_socket)
        catalogs,configs=floor_catalogs_for_manual(catalog)
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
    elevator_camera=elevator_monitor=elevator_transition=handoff=source_pose=floor_transform=None
    source_socket=args.runtime_socket
    elevator_report=None
    try:
        session=AgentSession(args.instruction,catalog,navigation,store,use_memory=not args.no_memory,
                             floor_catalogs=catalogs)
        actor=make_loop(args,session,output/'session.sqlite')
        last=None
        last_progress=None
        last_observation=None
        last_status_at=0.
        last_reference=None
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
            nonlocal navigation,reference,source_socket,elevator_camera,elevator_monitor,elevator_transition,handoff,source_pose,floor_transform,elevator_report
            stage=next((g for g in session.task_goals.goals if g['status']!='completed'),None)
            requirement=None if stage is None else stage['requirement']
            if (cross_floor and requirement is not None and requirement.get('stage_type')=='elevator' and
                    session.state['status']=='ready' and elevator_monitor is None):
                from ..adapters.elevator_camera import ElevatorCamera
                from ..core.elevator_links import capture_source_pose,fit_floor_transform
                from ..core.elevator_model import decide_elevator
                from ..core.elevator_monitor import ElevatorMonitor
                from ..core.elevator_transition import ElevatorTransition
                try:
                    reference.set(None,None)
                    display.send([])
                    current=navigation.tick()
                    source_pose=capture_source_pose(current,session.catalog)
                    target=session.task_goals.floor_catalogs[requirement['to_floor']]
                    links=json.loads((Path(__file__).resolve().parents[1]/'configs/elevator_links.json').read_text())['links']
                    floor_transform=fit_floor_transform(links,session.catalog,target)
                    elevator_transition=ElevatorTransition(task_id=session.state['task_id'],goal_id=requirement['goal_id'],
                        from_floor=requirement['floor_id'],to_floor=requirement['to_floor'],
                        destination_map_id=target.value['localization_map_id'])
                    elevator_camera=ElevatorCamera();elevator_camera.start()
                    elevator_monitor=ElevatorMonitor(elevator_camera,elevator_transition,
                        lambda context,history,frame:decide_elevator(args.api_config,output/'session.sqlite',context,history,frame,
                            cache_mode=args.cache_mode,thinking_mode=args.thinking_mode),store,history_policy='agent')
                    _actor.model.invalidate()
                    session.state.update(status='waiting_elevator',reason=None,elevator_active=True)
                    session._save('elevator_monitor_started',dict(goal_id=requirement['goal_id'],
                        from_floor=requirement['floor_id'],to_floor=requirement['to_floor']))
                    emit(dict(event='elevator_monitor_started',floor_id=requirement['floor_id'],to_floor=requirement['to_floor']))
                except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                    session.stop('needs_clarification','elevator_start_failed')
                    session._save('elevator_start_failed',dict(error_type=type(error).__name__))
                    return
            if (elevator_monitor is not None and session.state.get('elevator_active') and
                    session.state['status']=='ready'):
                session.state['status']='waiting_elevator'
            if (elevator_monitor is not None and session.state.get('elevator_active') and
                    session.state['status']=='paused'):
                if time.monotonic()-last_status_at>=.5:
                    last_status_at=time.monotonic()
                    emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                        goal_id=requirement['goal_id'],call_id=None,status='paused',floor_id=requirement['floor_id'],
                        goal_description=requirement['description'],navigation_status='elevator',localization_status='unavailable',
                        input_stale=True,model_busy=elevator_monitor.model.busy,elevator_status=elevator_transition.status))
                return
            if session.state['status']=='waiting_elevator' and elevator_monitor is not None:
                from ..core.elevator_links import capture_source_pose,destination_lidar_seed
                if elevator_transition.status=='awaiting_entry':
                    try:source_pose=capture_source_pose(navigation.tick(),session.catalog)
                    except (OSError,EOFError,ValueError,KeyError,TypeError):pass
                monitor_state=elevator_monitor.tick()
                if elevator_transition.status=='awaiting_destination_localization' and handoff is None and elevator_report is None:
                    from .floor_handoff import FloorHandoff
                    try:
                        seed=destination_lidar_seed(floor_transform,source_pose,navigation.geometry.base_from_lidar)
                        handoff=FloorHandoff(session=session,transition=elevator_transition,
                            target_catalog=session.task_goals.floor_catalogs[requirement['to_floor']],config_path=configs[requirement['to_floor']],
                            seed=seed,output=output,source_socket=source_socket)
                        store.event('floor_handoff_started',dict(from_floor=requirement['floor_id'],to_floor=requirement['to_floor']))
                    except (OSError,EOFError,ValueError,KeyError,TypeError) as error:
                        elevator_report='floor_handoff_start_failed: '+type(error).__name__
                handoff_state=None if handoff is None else handoff.tick()
                if handoff_state and handoff_state['status']=='complete':
                    navigation=session.navigation
                    reference.close();source_socket=str(handoff.run_dir/'run/commands.sock')
                    reference=ManualReference(source_socket)
                    handoff.close();handoff=None
                    elevator_monitor.close();elevator_monitor=None;elevator_camera=None;elevator_transition=None
                    source_pose=floor_transform=None;last=last_progress=last_reference=None
                    display.send([])
                    session.state.update(status='ready',reason=None,elevator_active=False)
                    session._save('elevator_stage_completed')
                    emit(dict(event='elevator_stage_completed',floor_id=session.catalog.value['floor_id'],
                              runtime_socket=source_socket))
                    elevator_report=None
                    return
                if handoff_state and handoff_state['status']=='failed':elevator_report=handoff_state['error']
                if monitor_state.get('error'):elevator_report=monitor_state['error']
                if time.monotonic()-last_status_at>=.5:
                    last_status_at=time.monotonic()
                    emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                        goal_id=requirement['goal_id'],call_id=None,status=session.state['status'],floor_id=requirement['floor_id'],
                        goal_description=requirement['description'],navigation_status='elevator',
                        localization_status='unavailable',input_stale=True,model_busy=monitor_state['model_busy'],
                        elevator_status=monitor_state['status'],elevator_decision=monitor_state['latest_decision'],
                        floor_handoff=handoff_state,error=elevator_report))
                return
            display.check()
            record=session.skill.record
            handle=record.get('handle')
            if record['status']=='running' and handle is not None:
                request=reference_request(handle,record['selected_pose'],navigation.route,
                                          navigation.route_index,session.catalog,
                                          require_yaw=navigation.request['require_yaw'])
                reference.set(handle['call_id'],request)
            else:reference.set(None,None)
            if record['status']=='running' and handle is not None:
                preview=(navigation.value.get('control') or {}).get('manual_preview') or {}
                error=reference.status()
                index=0 if navigation.route is None else navigation.route_index
                expected=handle['call_id']+':'+str(index)
                if preview.get('owner')!=handle['call_id'] or preview.get('request_id')!=expected:preview={}
                summary=dict(status='unavailable' if error else preview.get('status','waiting'),
                    reason=error or preview.get('reason'),reference_stamp_s=preview.get('reference_stamp_s'))
                key=(handle['call_id'],summary['status'],summary['reason'])
                if key!=last_reference:
                    last_reference=key
                    event=dict(event='manual_reference_status',goal_id=record['goal_id'],**summary)
                    session.store.event('manual_reference_status',event);emit(event)
            else:last_reference=None
            if getattr(args,'gui_commands',False) and time.monotonic()-last_status_at>=.5:
                last_status_at=time.monotonic()
                feedback=navigation.snapshot()
                current=next((g for g in session.task_goals.goals if g['status']!='completed'),None)
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
                    floor_id=session.catalog.value.get('floor_id'),goal_distance_m=distance,
                    goal_tolerance_m=navigation.settings['goal_position_tolerance_m'],
                    goal_yaw_error_rad=yaw_error,goal_yaw_tolerance_rad=navigation.settings['goal_yaw_tolerance_rad'],
                    arrival_dwell_s=dwell,arrival_dwell_required_s=navigation.settings['goal_dwell_s'],
                    monitor=session.monitor.snapshot(),
                    call_id=None if handle is None else handle['call_id'],status=session.state['status'],
                    confirmation_call_id=None if confirmation_handle is None else confirmation_handle['call_id'],
                    manual_confirmation_available=bool(confirmation_handle is not None and navigation.status=='running'),
                    reason=session.state.get('reason'),navigation_status=record['status'],
                    model_busy=_actor.model.busy,localization_status=feedback.get('localization_status'),
                    input_stale=feedback.get('input_stale'),origin_pose=feedback.get('origin_pose'),
                    source_stamp_s=feedback.get('origin_stamp_s'),stopped=feedback.get('stopped'),
                    route_index=feedback.get('route_index'),route_count=feedback.get('route_count'),
                    reference_preview={k:v for k,v in ((navigation.value.get('control') or {}).get('manual_preview') or {}).items()
                                       if k!='reference_path'},
                    reference_preview_error=reference.status()))
            if last is not None and navigation.route is not None:
                save_trace(force=handle is None or record['status']!='running')
            if record['status']=='running' and handle is not None and handle['call_id']!=last:
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
                    target_map_xy_yaw=pose,yaw_degrees=round(math.degrees(pose[2]),1),
                    route_waypoints=points if route else None,
                    rviz_missing_height_indices=missing_heights,
                    rviz_approximate_height_indices=approximate_heights,
                    target_image=None if preview is None else str(preview),
                    position_tolerance_m=navigation.settings['goal_position_tolerance_m'],
                    yaw_required=navigation.request['require_yaw'],
                    yaw_tolerance_degrees=(round(math.degrees(navigation.settings['goal_yaw_tolerance_rad']),1)
                                           if navigation.request['require_yaw'] else None),
                    instruction='使用原车手柄按建议路线驾驶；到达后依据实时定位和停稳估计自动检查。路线仅供参考。'))
            if record['status']=='running' and handle is not None and navigation.route is not None:
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
            elif last is not None and (handle is None or record['status']!='running'):
                display.send([]);last=None;last_progress=None
            stage=next((g for g in session.task_goals.goals if g['status']!='completed'),None)
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
        if actor:actor.close()
        if elevator_monitor:elevator_monitor.close()
        if elevator_camera:elevator_camera.close()
        if handoff:handoff.close()
        reference.close()
        display.close()
        store.close()
