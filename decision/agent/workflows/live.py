"""Explicit live Agent: goal intents only; the operator separately arms the chassis."""
import json
from pathlib import Path
import time
from ..adapters.runtime import LiveRuntimeClient
from ..adapters.shadow import RuntimeLiveNavigation
from ..core.session import AgentSession
from ..core.store import Store, shared_budget
from ..tools.places import Catalog
from .common import emit, make_loop
from .interaction import Console


def cleanup_goal(navigation, skill, *, timeout_s=2.):
    result = dict(requested=False, stop_observed=False)
    # A held arrival already cleared the motion goal, but still owns the
    # arming revision. Explicitly cancel it on exit to disarm the chassis.
    held = (skill.record['status'] == 'held' or
            skill.record['status']=='failed' and getattr(navigation,'follow_active',False))
    if navigation.handle is not None and (not navigation.cancel_requested or held):
        try:
            navigation.cancel(navigation.handle)
            result['requested'] = True
            deadline = time.monotonic()+timeout_s
            while time.monotonic() < deadline:
                state = navigation.tick()
                if state.get('stop_observation'):
                    result['stop_observed'] = True
                    break
                time.sleep(.05)
        except (OSError, EOFError, ValueError) as error:
            result['error_type'] = type(error).__name__
    return result


def run(args):
    if args.duration <= 0:
        raise ValueError('Positive duration required')
    catalog = Catalog.load(args.catalog, condition=args.map_condition)
    navigation = RuntimeLiveNavigation(catalog, LiveRuntimeClient(args.runtime_socket))
    state = navigation.snapshot()
    deadline = time.monotonic()+3.
    while (state.get('origin_pose') is None and not state.get('input_stale') and
           state.get('localization_status') == 'valid' and time.monotonic() < deadline):
        time.sleep(.05)
        state = navigation.tick()
    if (state.get('input_stale') or state.get('localization_status') != 'valid' or
            state.get('origin_pose') is None or state.get('origin_frame_id') != catalog.value['frame_id']):
        raise ValueError('Current mapped localization and observation are required')
    if state.get('chassis_state') != 'armed' or not state.get('chassis_status_fresh'):
        raise ValueError('Operator must arm the chassis before starting Agent execution')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output/'session.sqlite')
    actor = None
    try:
        session = AgentSession(args.instruction, catalog, navigation, store, use_memory=not args.no_memory)
        actor = make_loop(args, session, output/'session.sqlite')
        actor.run(console=Console() if args.interactive else None, timeout_s=args.duration, emit=emit)
        cleanup = cleanup_goal(navigation, session.skill)
        store.event('live_goal_cleanup', cleanup)
        result = dict(scope='live_localization_based', task=session.state, exit_reason=actor.exit_reason,
                      decision_turns=actor.requests, physical_acceptance=False,
                      arrival_evidence='localization_estimate_not_independent_chassis_acceptance',cleanup=cleanup)
        if not args.actions:
            budget = shared_budget()
            result['api_budget'] = budget.summary()
            budget.close()
        (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        emit(dict(scope=result['scope'], status=session.state['status'], exit_reason=actor.exit_reason,
                  physical_acceptance=False))
        return result
    finally:
        if actor:
            actor.close()
        store.close()
