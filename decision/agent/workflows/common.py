"""Shared loop assembly and JSON output for CLI workflows."""
import json
from pathlib import Path
from ..core.loop import AgentLoop
from ..core.model_settings import DEFAULT_MODEL, GPT_MODEL, default_efforts, validate_effort


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def model_options(args):
    model_name = getattr(args, 'model', DEFAULT_MODEL)
    if model_name == GPT_MODEL and getattr(args, 'cache_mode', 'implicit') == 'explicit':
        raise ValueError('OpenAI Responses API requires implicit caching')
    phases = ('registration', 'execution', 'monitor')
    efforts = dict(zip(phases, default_efforts(model_name)))
    for phase in phases:
        option = 'planning' if phase == 'registration' else phase
        override = getattr(args, f'{option}_thinking_mode', None)
        if override is not None:
            efforts[phase] = validate_effort(model_name, override)
    mode = getattr(args, 'thinking_mode', 'planning')
    if mode != 'planning':
        efforts = dict.fromkeys(phases, validate_effort(model_name, mode))
    return model_name, efforts


def make_loop(args, session, session_path, *, transient_sink=None):
    close_work = None
    if args.actions:
        from .replay import ActionReplay
        actions = json.loads(Path(args.actions).read_text())
        session.store.event('decision_source', dict(source='explicit_tool_action_fixture', model_calls=0))
        work = ActionReplay(actions)
    else:
        from ..core.model import SessionTransport, api_work
        model_name, efforts = model_options(args)
        if getattr(session, 'store', None) is not None:
            session.store.event('decision_source', dict(source='hosted_model', model=model_name, thinking_efforts=efforts))
        transport = SessionTransport(model_name=model_name)
        close_work = transport.close
        def work(payload, observation, spatial_view):
            current_efforts = efforts
            phase = payload.get('decision_phase')
            if phase is None:
                # Legacy callers can still invoke work directly with a prompt.
                current = payload['current_state']
                if not current['requirements_registered']:
                    phase = 'registration'
                else:
                    phase = 'monitor' if payload.get('ride_inputs') else 'execution'
                _, current_efforts = model_options(args)
            elevator=getattr(session,'elevator',None)
            transition=getattr(elevator,'transition',None)
            ride = payload['current_state'].get('ride') or {}
            request_phase=ride.get('status',getattr(transition,'status',None))
            def active_ride_sink(value):
                if (transient_sink is not None and transition is not None and
                        getattr(getattr(session,'elevator',None),'transition',None) is transition and
                        request_phase in ('monitoring_ride','awaiting_destination_localization') and
                        transition.status==request_phase):
                    transient_sink(dict(value,phase=request_phase))
            return api_work(
                args.api_config,session_path,payload,observation,spatial_view,cache_mode=args.cache_mode,
                model_name=model_name,thinking_mode=current_efforts[phase],decision_phase=phase,
                transient_sink=active_ride_sink if transient_sink is not None else None,
                session_transport=transport)
    return AgentLoop(session, work, max_requests=args.max_requests, close_work=close_work)
