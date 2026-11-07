"""Synthetic navigation workflow; no ROS or chassis output."""
import json
from pathlib import Path
from .. import ROOT
from ..core.session import AgentSession
from ..core.store import Budget, Store
from ..tools.places import Catalog
from .common import emit, make_loop
from .drivers import SimulationDriver
from .interaction import Console


def run_simulation(args):
    from ..adapters.simulation import SimulationNavigation
    catalog = Catalog.load(args.catalog,condition=args.map_condition)
    if catalog.value['scope'] != 'simulation':
        raise ValueError('Simulation requires a synthetic catalog; no API request sent')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    store = Store(out/'session.sqlite')
    actor = None
    try:
        session = AgentSession(args.instruction, catalog, SimulationNavigation(), store, use_memory=not args.no_memory)
        actor = make_loop(args, session, out/'session.sqlite')
        console = Console() if args.interactive else None
        driver = SimulationDriver(session.navigation, max_s=args.max_s, fast=bool(args.actions),
                                  scripted_user_events=args.scripted_user_events)
        actor.run(console=console, driver=driver,
                  emit=lambda value: emit(dict(value, simulation_s=session.navigation.t)))
        result = dict(scope='simulation', task=session.state, navigation=session.skill.snapshot(),
                      model_requests=0 if args.actions else actor.requests, decision_turns=actor.requests,
                      exit_reason=actor.exit_reason, real_robot_success=False)
        if not args.actions:
            budget = Budget(ROOT/'outputs/decision/agent/api_budget.sqlite')
            result['api_budget'] = budget.summary()
            budget.close()
        (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        emit(dict(status=session.state['status'], model_requests=result['model_requests'], real_robot_success=False))
        return result
    finally:
        if actor:
            actor.close()
        store.close()
