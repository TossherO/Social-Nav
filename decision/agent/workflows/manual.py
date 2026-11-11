"""Operator-driven Agent testing against a read-only live observation runtime."""
import json
import math
from pathlib import Path
import time
from ..adapters.manual import ManualNavigation
from ..adapters.runtime import ManualRuntimeClient
from ..core.session import AgentSession
from ..core.store import Store, shared_budget
from ..tools.places import Catalog
from .common import emit, make_loop
from .interaction import Console


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
    try:
        session=AgentSession(args.instruction,catalog,navigation,store,use_memory=not args.no_memory)
        actor=make_loop(args,session,output/'session.sqlite')
        last=None
        def report(_actor):
            nonlocal last
            record=session.skill.record
            handle=record.get('handle')
            if record['status']=='running' and handle is not None and handle['call_id']!=last:
                last=handle['call_id']
                pose=record['selected_pose']
                preview=None
                try:
                    from ..tools.spatial import display_extent,render
                    inputs=navigation.spatial_inputs()
                    current=inputs.get('pose')
                    bounds=display_extent(inputs['global_grid'],session.catalog.annotations(),
                        extras=[pose[:2]]+([] if current is None else [current[:2]]))['bounds']
                    preview=output/('target_%04d.png' % handle['goal_revision'])
                    preview.write_bytes(render(session.catalog,inputs['global_grid'],bounds,current,
                        geometry=navigation.geometry,local_view=True,static_crop=True,
                        candidate_preview=dict(pose=pose)))
                except (ValueError,KeyError,TypeError,OSError):
                    preview=None
                emit(dict(event='manual_target',goal_id=record['goal_id'],place_id=record['place_id'],
                    target_map_xy_yaw=pose,yaw_degrees=round(math.degrees(pose[2]),1),
                    target_image=None if preview is None else str(preview),
                    position_tolerance_m=navigation.settings['goal_position_tolerance_m'],
                    yaw_required=navigation.request['require_yaw'],
                    yaw_tolerance_degrees=(round(math.degrees(navigation.settings['goal_yaw_tolerance_rad']),1)
                                           if navigation.request['require_yaw'] else None),
                    instruction='使用原车手柄驶向目标并停稳；完成后在此终端输入 arrived。'))
        actor.run(console=Console(),timeout_s=args.duration,on_tick=report,emit=emit)
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
        store.close()
