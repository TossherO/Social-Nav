"""Stepwise saved observations or live runtime shadow; neither executes motion."""
import json
from pathlib import Path
from .. import ROOT
from ..tools.places import Catalog
from .interaction import Console
from .common import emit, make_loop
from .drivers import SavedObservationDriver
from ..adapters.shadow import SavedShadowNavigation, RuntimeShadowNavigation
from ..tools.observations import ObservationSet
from ..adapters.runtime import RuntimeClient
from ..core.store import Store, Budget
from ..core.session import AgentSession


def run(args):
    if args.duration <= 0 or args.interactive and args.observations:
        raise ValueError('Positive duration required; interactive input is for runtime shadow')
    catalog = Catalog.load(args.catalog)
    dataset = ObservationSet(args.observations) if args.observations else None
    if dataset is not None:
        sequences = args.sequences or dataset.sequences()
        if not sequences or len(sequences) > 12 or sequences != sorted(set(sequences)):
            dataset.close()
            raise ValueError('Use one to twelve increasing observation IDs')
        navigation = SavedShadowNavigation(catalog, dataset.read(sequences[0]))
    else:
        sequences = None
        navigation = RuntimeShadowNavigation(catalog, RuntimeClient(args.runtime_socket))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output/'session.sqlite')
    actor = None
    try:
        session = AgentSession(args.instruction, catalog, navigation, store, use_memory=not args.no_memory)
        actor = make_loop(args, session, output/'session.sqlite')
        console = Console() if args.interactive else None
        seen, previous = [], None
        def record(actor):
            nonlocal previous
            key = (session._observation_key(), session.state['status'], actor.requests)
            if key != previous:
                seen.append(dict(observation_binding=key[0], task_status=key[1],
                                 decision_turns=actor.requests, navigation_status=session.skill.record['status']))
                previous = key
        driver = SavedObservationDriver(navigation, dataset, sequences) if dataset is not None else None
        actor.run(console=console, driver=driver, timeout_s=args.duration, on_tick=record, emit=emit)
        session._save('shadow_finished', dict(real_robot_success=False, execution_feedback=False))
        result = dict(scope='causal_saved_observation_shadow' if dataset is not None else 'runtime_nonexecuting_shadow',
                      task=session.state, observations=seen, model_requests=0 if args.actions else actor.requests,
                      decision_turns=actor.requests, exit_reason=actor.exit_reason,
                      real_robot_success=False, navigation_execution_verified=False)
        if not args.actions:
            budget = Budget(ROOT/'outputs/decision/agent/api_budget.sqlite')
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
