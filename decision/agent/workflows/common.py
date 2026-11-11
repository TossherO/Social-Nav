"""Shared loop assembly and JSON output for CLI workflows."""
import json
from pathlib import Path
from ..core.loop import AgentLoop


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def make_loop(args, session, session_path):
    if args.actions:
        from .replay import ActionReplay
        actions = json.loads(Path(args.actions).read_text())
        session.store.event('decision_source', dict(source='explicit_tool_action_fixture', model_calls=0))
        work = ActionReplay(actions)
    else:
        from ..core.model import api_work
        work = lambda payload, observation, spatial_view: api_work(
            args.api_config,session_path,payload,observation,spatial_view,cache_mode=args.cache_mode,
            thinking_mode=args.thinking_mode)
    return AgentLoop(session, work, max_requests=args.max_requests)
