"""Operator-driven Agent testing against a read-only live observation runtime."""
import json
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

    def send(self, points, route_index=None):
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
            self.process.stdin.write(json.dumps(dict(frame_id=self.frame,points=points,route_index=route_index),allow_nan=False)+'\n')
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
    heights,_=sample_ground(grid,points,offset=.12)
    approx=[]
    if 'ground_z' in grid.layers and 'ground_support' in grid.layers:
        ground=grid.layers['ground_z'];support=grid.layers['ground_support']&np.isfinite(ground)
        if support.any():
            fallback=float(np.median(ground[support]))+.12
            for i,z in enumerate(heights):
                if not np.isfinite(z):heights[i]=fallback;approx.append(i)
    return ([[float(p[0]),float(p[1]),float(z) if np.isfinite(z) else None]
             for p,z in zip(points,heights)],approx)


def run(args):
    if args.duration <= 0 or not args.interactive:
        raise ValueError('Manual driving requires interactive input and a positive duration')
    catalog=Catalog.load(args.catalog,condition=args.map_condition)
    navigation=ManualNavigation(catalog,ManualRuntimeClient(args.runtime_socket))
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
    try:
        session=AgentSession(args.instruction,catalog,navigation,store,use_memory=not args.no_memory)
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
                emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                    goal_id=None if current is None else current['requirement']['goal_id'],
                    call_id=None if handle is None else handle['call_id'],status=session.state['status'],
                    reason=session.state.get('reason'),navigation_status=record['status'],
                    model_busy=_actor.model.busy,localization_status=feedback.get('localization_status'),
                    input_stale=feedback.get('input_stale'),origin_pose=feedback.get('origin_pose'),
                    source_stamp_s=feedback.get('origin_stamp_s'),stopped=feedback.get('stopped'),
                    route_index=feedback.get('route_index'),route_count=feedback.get('route_count'),
                    reference_preview=(navigation.value.get('control') or {}).get('manual_preview'),
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
                        route_preview=None if route is None else dict(waypoints=points,diagnostics=dict(intersecting_segment_indices=[]))))
                    display_geometry,approximate_heights=display_points(inputs['global_grid'],points)
                    missing_heights=[i for i,point in enumerate(display_geometry) if point[2] is None]
                    display.send(display_geometry,navigation.route_index)
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
                    instruction='使用原车手柄按建议路线驾驶；终点停稳后在此终端输入 arrived。路线仅供参考。'))
            if record['status']=='running' and handle is not None and navigation.route is not None:
                progress=(handle['call_id'],navigation.route_index)
                if progress!=last_progress:
                    last_progress=progress
                    event=dict(event='manual_route_progress',goal_id=record['goal_id'],
                               next_waypoint_index=navigation.route_index,
                               waypoint_count=len(navigation.route['waypoints']),index_base=0,
                               source='localization_estimate',completion_not_proven=True)
                    session.store.event('manual_route_progress',event)
                    emit(event)
                    try:
                        inputs=navigation.spatial_inputs()
                        geometry,_=display_points(inputs['global_grid'],navigation.route['waypoints'])
                        display.send(geometry,navigation.route_index)
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
        reference.close()
        display.close()
        store.close()
