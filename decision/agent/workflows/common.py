"""Shared loop assembly and JSON output for CLI workflows."""
import json
from pathlib import Path
from ..core.loop import AgentLoop


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def make_loop(args, session, session_path, *, transient_sink=None):
    close_work = None
    if args.actions:
        from .replay import ActionReplay
        actions = json.loads(Path(args.actions).read_text())
        session.store.event('decision_source', dict(source='explicit_tool_action_fixture', model_calls=0))
        work = ActionReplay(actions)
    else:
        from ..core.model import SessionTransport, api_work
        transport = SessionTransport()
        close_work = transport.close
        def work(payload, observation, spatial_view):
            elevator=getattr(session,'elevator',None)
            transition=getattr(elevator,'transition',None)
            request_phase=getattr(transition,'status',None)
            def active_ride_sink(value):
                if (transient_sink is not None and transition is not None and
                        getattr(getattr(session,'elevator',None),'transition',None) is transition and
                        request_phase in ('monitoring_ride','awaiting_destination_localization') and
                        transition.status==request_phase):
                    transient_sink(dict(value,phase=request_phase))
            return api_work(
                args.api_config,session_path,payload,observation,spatial_view,cache_mode=args.cache_mode,
                thinking_mode=(('off' if payload['current_state']['requirements_registered'] else getattr(args,'planning_thinking_mode','low'))
                               if args.thinking_mode=='planning' else args.thinking_mode),
                transient_sink=active_ride_sink if transient_sink is not None else None,
                session_transport=transport)
    return AgentLoop(session, work, max_requests=args.max_requests, close_work=close_work)
