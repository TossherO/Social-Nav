"""Stepwise saved observations or live runtime shadow; neither executes motion."""
import json
import time
from pathlib import Path
from ..tools.places import Catalog
from .interaction import Console
from .common import emit, make_loop
from .drivers import SavedObservationDriver
from ..adapters.shadow import SavedShadowNavigation, RuntimeShadowNavigation
from ..tools.observations import ObservationSet
from ..adapters.runtime import RuntimeClient
from ..core.store import Store, shared_budget
from ..core.session import AgentSession
from ..core.tasks import next_unfinished_goal


def require_bag_session(path):
    from runtime.transport import exchange_request
    status=exchange_request(path,'session_status',{},timeout=.5)
    if status.get('live') is not False or status.get('hardware_enabled') is not False:
        raise ValueError('Bag replay requires a nonexecuting bag session')


def replay_t2_result(item):
    """Summarize VLM choices and local identity decisions for the replay window."""
    result = {key: item.get(key) for key in
              ('event', 'source_stamp_s', 'backend', 'status', 'selection', 'raw_content', 'timing', 'reason')}
    selection = result['selection']
    if result['backend'] == 'reid' and isinstance(selection, dict):
        candidate = selection.get('candidate_id')
        choice = item.get('search_choice') or {}
        result.update(status=selection['status'], reason=selection['reason'], choice=choice or None,
                      selection=f'人物 {candidate}' if candidate is not None else choice.get('kind'))
    else:
        result['choice'] = (item.get('choices') or {}).get(selection)
    return result


def wait_for_bag_navigation(catalog, path, *, timeout_s=300., emit_status=emit):
    """Wait for a fresh, map-associated bag observation before creating a task."""
    from runtime.transport import exchange_request
    deadline=time.monotonic()+timeout_s
    navigation=None
    emit_status(dict(event='replay_preparing',instruction='等待 bag 定位与当前地图观察就绪。'))
    while time.monotonic()<deadline:
        if not Path(path).exists():
            time.sleep(.05);continue
        try:
            status=exchange_request(path,'session_status',{},timeout=.5)
        except (OSError,EOFError,TimeoutError):
            time.sleep(.05);continue
        if status.get('live') is not False or status.get('hardware_enabled') is not False:
            raise ValueError('Bag replay requires a nonexecuting bag session')
        if status.get('source_ended'):
            raise ValueError('Bag ended before localization became ready')
        try:
            if navigation is None:navigation=RuntimeShadowNavigation(catalog,RuntimeClient(path))
            else:navigation.tick()
            state=navigation.snapshot()
            if (not state.get('input_stale') and state.get('localization_status')=='valid' and
                    state.get('origin_pose') is not None and state.get('origin_frame_id')==catalog.value['frame_id'] and
                    (state.get('map_binding') or {}).get('valid') is True and
                    navigation.clock_domain=='runtime_bag'):
                return navigation
        except (OSError,EOFError,ValueError,KeyError,TypeError):
            navigation=None
        time.sleep(.05)
    raise ValueError('Timed out waiting for fresh bag localization and observation')


def hold_after_t2_failure(session, actor):
    """A transport or model failure needs explicit operator action, never an automatic retry."""
    record=session.skill.record
    reason=record.get('reason') or ''
    if (record.get('stage_type')!='follow' or record.get('status')!='failed' or
            not reason.startswith('follow_t2_') or session.state.get('reason')==reason and
            session.state.get('status')=='needs_clarification'):
        return False
    actor.model.invalidate()
    session.stop('needs_clarification',reason)
    session._save('replay_t2_failure_waiting',dict(reason=reason,automatically_retried=False))
    return True


def start_unique_bag_follow(session, goal_id):
    """Wait for one tracked person, or hand a multi-person view back to the Agent."""
    feedback=session.navigation.snapshot()
    if (feedback.get('input_stale') or feedback.get('localization_status')!='valid' or
            (feedback.get('map_binding') or {}).get('valid') is not True):
        return 'waiting'
    record=session.navigation.observation_record()
    group=record.get('follow_candidates')
    if group is None and session.navigation.value.get('follow_frame') is None:
        return 'waiting'
    if group is not None:
        candidates=group.get('candidates',[])
        if len(candidates)>1:
            return 'agent_selection'
        if not candidates or not candidates[0].get('track'):
            return 'waiting'
    if session.call('observe_scene').get('status')!='available':
        return 'waiting'
    record=session.observation.record
    candidates=(record.get('follow_candidates') or {}).get('candidates',[])
    if len(candidates)>1:
        return 'agent_selection'
    if not candidates or not candidates[0].get('track'):
        return 'waiting'
    observation_id=record['observation_id']
    selected=session.call('select_follow_target',goal_id=goal_id,observation_id=observation_id,
        candidate_id=candidates[0]['id'],target_description='The person in the fixed reference',
        _context=session.context())
    if selected.get('status')!='selected':
        return 'waiting'
    result=session.call('start_local_task',goal_id=goal_id,task_type='follow',reference_id=selected['reference_id'])
    if result.get('status') not in ('accepted','uncertain'):
        return 'waiting'
    session.wait_for_event('navigation')
    session._save('replay_local_follow_started',dict(goal_id=goal_id,source='unique_current_associated_candidate',
                                                   status=result['status']))
    return 'started'


def run(args):
    if args.duration <= 0 or args.interactive and args.observations:
        raise ValueError('Positive duration required; interactive input is for runtime shadow')
    catalog = Catalog.load(args.catalog,condition=args.map_condition)
    reuse_path=getattr(args,'reuse_session',None)
    auto_follow=bool(getattr(args,'auto_follow',False))
    local_follow=bool(getattr(args,'auto_follow_single_person',False))
    if local_follow and not auto_follow:
        raise ValueError('Local follow startup requires --auto-follow')
    if (reuse_path or auto_follow) and (not getattr(args,'require_bag',False) or args.observations):
        raise ValueError('Task reuse and automatic follow require a nonexecuting bag runtime')
    if auto_follow and not reuse_path:raise ValueError('Automatic follow requires a registered task session')
    if getattr(args,'reuse_digest',None) and not reuse_path:raise ValueError('Task digest requires its source session')
    plan=assets=None
    if reuse_path:
        from ..core.reuse import load_registered_task,verify_registered_task
        plan=load_registered_task(reuse_path)
        if getattr(args,'reuse_digest',None) and args.reuse_digest!=plan['digest']:
            raise ValueError('历史任务在预览后发生变化，请重新选择')
        if args.instruction and args.instruction!=plan['instruction']:
            raise ValueError('复用历史登记时任务指令必须与历史指令一致')
        args.instruction=plan['instruction']
        if args.actions:raise ValueError('任务复用不能与动作回放同时使用')
        assets=verify_registered_task(plan,{catalog.value.get('floor_id'):catalog},catalog.value.get('floor_id'))
        if auto_follow and (not plan['goals'] or plan['goals'][0].get('stage_type')!='follow'):
            raise ValueError('Automatic T2 start requires a follow task as the first registered stage')
    dataset = ObservationSet(args.observations) if args.observations else None
    if dataset is not None:
        sequences = args.sequences or dataset.sequences()
        if not sequences or len(sequences) > 12 or sequences != sorted(set(sequences)):
            dataset.close()
            raise ValueError('Use one to twelve increasing observation IDs')
        navigation = SavedShadowNavigation(catalog, dataset.read(sequences[0]))
    else:
        sequences = None
        if auto_follow:
            navigation=wait_for_bag_navigation(catalog,args.runtime_socket)
        else:
            if getattr(args,'require_bag',False):
                require_bag_session(args.runtime_socket)
            navigation = RuntimeShadowNavigation(catalog, RuntimeClient(args.runtime_socket))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output/'session.sqlite')
    actor = None
    try:
        session = AgentSession(args.instruction, catalog, navigation, store, use_memory=not args.no_memory)
        if plan is not None:
            from ..core.reuse import import_registered_task
            import_registered_task(session,plan,assets,reuse_path)
        if auto_follow:
            instruction=('任务已进入跟随阶段，单人且轨迹有效时直接跟随，多人时由 Agent 选人。' if local_follow else
                         '任务登记已载入；Agent 将观察并确认本次的跟随人物。')
            emit(dict(event='replay_auto_follow_started',instruction=instruction))
        actor = make_loop(args, session, output/'session.sqlite')
        if auto_follow:
            def before_model(actor):
                if hold_after_t2_failure(session,actor):
                    emit(dict(event='replay_t2_failed',instruction='T2 请求失败，已暂停自动推进；请检查失败原因后手动决定如何继续。'))
            actor.before_model=before_model
        if local_follow:
            initial_goal=plan['goals'][0]['goal_id']
            def start_before_model(actor):
                result=start_unique_bag_follow(session,initial_goal)
                if result=='agent_selection':
                    actor.before_request=None
                    emit(dict(event='replay_follow_requires_selection',instruction='画面中有多名人物，交由 Agent 按原指令确认目标。'))
                    return True
                if result=='started':
                    actor.before_request=None
                    emit(dict(event='replay_auto_follow_started',instruction='已用当前 bag 观察确认行人，开始影子跟随。'))
                return False
            actor.before_request=start_before_model
        console = Console(structured=getattr(args,'gui_commands',False)) if args.interactive else None
        seen, previous = [], None
        last_status=0.
        reference_path=Path(args.runtime_socket).parent/'reference.jsonl' if getattr(args,'require_bag',False) else None
        reference_offset=reference_path.stat().st_size if reference_path is not None and reference_path.exists() else 0
        last_t2=None
        def record(actor):
            nonlocal previous,last_status,reference_offset,last_t2
            key = (session._observation_key(), session.state['status'], actor.requests)
            if key != previous:
                seen.append(dict(observation_binding=key[0], task_status=key[1],
                                 decision_turns=actor.requests, navigation_status=session.skill.record['status']))
                previous = key
            if getattr(args,'gui_commands',False) and time.monotonic()-last_status>=.5:
                last_status=time.monotonic()
                if reference_path is not None and reference_path.exists():
                    with reference_path.open(encoding='utf-8') as stream:
                        stream.seek(reference_offset)
                        for _ in range(50):
                            line=stream.readline()
                            if not line or not line.endswith('\n'):break
                            reference_offset=stream.tell()
                            item=json.loads(line)
                            event=item.get('event')
                            if event not in ('follow_t2_result','follow_t2_rejected','follow_t2_input_unavailable'):continue
                            last_t2=replay_t2_result(item)
                            session.store.event('t2_replay',last_t2)
                feedback=navigation.snapshot()
                current=next_unfinished_goal(session.task_goals.goals)
                requirement={} if current is None else current['requirement']
                record=session.skill.record
                handle=record.get('handle') or {}
                follow=feedback.get('follow_status') or {}
                emit(dict(event='manual_status',task_id=session.state['task_id'],revision=session.state['revision'],
                    goal_id=requirement.get('goal_id'),goal_description=requirement.get('description'),
                    floor_id=session.catalog.value.get('floor_id'),call_id=handle.get('call_id'),
                    status=session.state['status'],reason=session.state.get('reason'),
                    navigation_status=record['status'],model_busy=actor.model.busy,
                    localization_status=feedback.get('localization_status'),input_stale=feedback.get('input_stale'),
                    origin_pose=feedback.get('origin_pose'),source_stamp_s=feedback.get('origin_stamp_s'),
                    stopped=None,monitor=session.monitor.snapshot(),follow_status=follow,t2_result=last_t2,mode='bag_shadow'))
        driver = SavedObservationDriver(navigation, dataset, sequences) if dataset is not None else None
        actor.run(console=console, driver=driver, timeout_s=args.duration, on_tick=record, emit=emit)
        session._save('shadow_finished', dict(real_robot_success=False, execution_feedback=False))
        result = dict(scope='causal_saved_observation_shadow' if dataset is not None else 'runtime_nonexecuting_shadow',
                      task=session.state, observations=seen, model_requests=0 if args.actions else actor.requests,
                      decision_turns=actor.requests, exit_reason=actor.exit_reason,
                      real_robot_success=False, navigation_execution_verified=False)
        if not args.actions:
            budget = shared_budget()
            result['api_budget'] = budget.summary()
            budget.close()
        (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(dict(scope=result['scope'], status=session.state['status'], model_requests=result['model_requests'],
                              real_robot_success=False), ensure_ascii=False), flush=True)
        return result
    finally:
        if actor:
            actor.close()
        store.close()
        if dataset:
            dataset.close()
