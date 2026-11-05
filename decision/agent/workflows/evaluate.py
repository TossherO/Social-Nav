"""Explicit tool-decision fixtures over the real project physics; no model API."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from .. import CONFIG_DIR
from ..tools.places import Catalog
from ..core.loop import AgentLoop
from .replay import ActionReplay
from .drivers import SimulationDriver
from ..adapters.simulation import SimulationNavigation
from ..core.store import Store
from ..core.session import AgentSession


def action(tool_name, **arguments):
    return dict(tool=tool_name, arguments=arguments)


def multistop_actions():
    calls = [action('lookup_places', name='')]
    for place in ('elevator', 'lobby_entrance', 'lobby_center', 'mission_start'):
        calls += [action('navigate', place_id=place), action('wait_for_event', event='navigation')]
        if place == 'lobby_entrance':
            calls.append(action('wait_for_event', event='user_continue'))
    return calls+[action('finish_task', status='succeeded', evidence='合成反馈已确认所有测试站点到达并停稳。')]


def scenario(name, output):
    store = Store(output/(name+'.sqlite'))
    navigation = SimulationNavigation()
    catalog = Catalog.load(CONFIG_DIR/'simulation.json')
    if name == 'blocked_destination':
        navigation.world.static[37:44, 47:54] = 100
    session = AgentSession('Explicit synthetic tool fixture: '+name, catalog, navigation, store)
    if name == 'multistop':
        work = ActionReplay(multistop_actions())
    elif name == 'blocked_destination':
        work = ActionReplay([action('navigate', place_id='elevator'),
                             action('ask_user', question='目的地几何检查未通过，请确认后续任务。')])
    elif name == 'missing_visual_evidence':
        work = ActionReplay([action('navigate', place_id='elevator'), action('wait_for_event', event='navigation'),
                             action('observe_scene'), action('ask_user', question='仿真没有相机，无法完成视觉确认。')])
    else:
        def work(payload, observation):
            return (action('wait_for_event', event='navigation') if session.skill.record['handle'] is not None else
                    action('navigate', place_id='lobby_center' if session.state['revision'] > 1 else 'corridor_wait'))
    actor = AgentLoop(session, work, max_requests=64)
    events = set()
    started, sampled = time.perf_counter(), -1e9
    def record(actor):
        nonlocal sampled
        if navigation.t-sampled >= 1.:
            store.event('navigation_sample', navigation.snapshot())
            sampled = navigation.t
        if name == 'multistop' and actor.state['status'] == 'waiting_user':
            events.add('continue')

    def scripted_input(actor, emit):
        state, t = actor.state, navigation.t
        if t >= 2 and 'pause' not in events:
            actor.event('pause'); events.add('pause')
        if t >= 4 and state['status'] == 'paused' and 'resume' not in events:
            actor.event('resume'); events.add('resume')
        if t >= 6 and 'replace' not in events:
            actor.event('replace', '改去大厅中央'); events.add('replace')
        if t >= 8 and state['revision'] == 2 and 'cancel' not in events:
            actor.event('cancel'); events.add('cancel')

    try:
        driver = SimulationDriver(navigation, max_s=200., fast=True, scripted_user_events=name == 'multistop')
        console = SimpleNamespace(poll=scripted_input) if name == 'lifecycle' else None
        if console is not None:
            store.event('user_event_source', dict(source='explicit_simulation_fixture'))
        state = actor.run(console=console, driver=driver, on_tick=record)
        expected = dict(multistop='succeeded', lifecycle='cancelled',
                        blocked_destination='needs_clarification', missing_visual_evidence='needs_clarification')[name]
        passed = state['status'] == expected and not navigation.collision and navigation.snapshot()['stopped']
        if name == 'multistop':
            passed &= [p['place_id'] for p in session.skill.record['completed']] == ['elevator','lobby_entrance','lobby_center','mission_start']
        if name == 'lifecycle':
            passed &= events == {'pause', 'resume', 'replace', 'cancel'} and state['revision'] == 2
        result = dict(case=name, passed=bool(passed), status=state['status'], expected=expected,
                      simulation_s=navigation.t, cpu_wall_s=time.perf_counter()-started, scripted_events=sorted(events),
                      decision_turns=actor.requests, model_requests=0, exit_reason=actor.exit_reason,
                      task=state, navigation=navigation.snapshot(),
                      scope='synthetic_feedback_tool_fixture', real_robot_success=False)
        session._save('scenario_summary', result)
        return result
    finally:
        actor.close()
        store.close()


def evaluate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ('multistop', 'lifecycle', 'blocked_destination', 'missing_visual_evidence'):
        row = scenario(name, output)
        results.append(row)
        print(json.dumps({k:row[k] for k in ('case','passed','status','simulation_s','cpu_wall_s')},ensure_ascii=False),flush=True)
    (output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    if not all(r['passed'] for r in results):
        raise RuntimeError('A feedback tool scenario failed; see summary')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    evaluate(args.output)


if __name__ == '__main__':
    main()
