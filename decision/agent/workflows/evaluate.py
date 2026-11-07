"""Scripted indoor goal decisions over native navigation feedback; no model API."""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from .. import CONFIG_DIR
from ..tools.places import Catalog
from ..core.loop import AgentLoop
from .drivers import SimulationDriver
from ..adapters.simulation import SimulationNavigation
from ..core.store import Store
from ..core.session import AgentSession
from ..tools.framework import Reply


def action(tool_name, **arguments):
    return dict(tool=tool_name, arguments=arguments)


def point(goal_id, target, **conditions):
    return dict(goal_id=goal_id,kind='point',target_id=target,description='Synthetic requirement: '+target,**conditions)


def scenario(name, output):
    store = Store(output/(name+'.sqlite'))
    navigation = SimulationNavigation()
    catalog = Catalog.load(CONFIG_DIR/'simulation.json')
    if name == 'blocked_destination':
        navigation.world.static[37:44,47:54] = 100
    session = AgentSession('Explicit indoor task fixture: '+name,catalog,navigation,store)
    if name == 'point_heading':
        goals = [point('p1','elevator',required_yaw=.3)]
        poses = [[5,4,.3]]
    elif name == 'point_region_chain':
        goals = [point('p1','elevator'),dict(goal_id='r1',kind='region',target_id='lobby_area',
                 description='Arrive inside the lounge facing the annotated window',facing='front_window')]
        poses = [[5,4,0],[7,4,0]]
    else:
        goals = [point('p1','elevator',**({'semantic_conditions':['The entrance appearance is visible.']}
                                        if name == 'missing_visual_evidence' else {}))]
        poses = [[5,4,0]]
    calls = [action('record_task',goals=goals)]
    for goal,pose in zip(goals,poses):
        calls += [action('query_map'),action('navigate',goal_id=goal['goal_id'],pose=pose,
                  view_id='bound_by_explicit_fixture_at_request',orientation_source='user' if 'required_yaw' in goal else 'semantic'),
                  action('wait_for_event',event='navigation')]
        if name == 'missing_visual_evidence':
            calls += [action('observe_scene'),action('check_goal',goal_id=goal['goal_id']),
                      action('ask_user',question='仿真无真实RGB，无法核验视觉条件。')]
        elif name == 'blocked_destination':
            calls = calls[:-1]+[action('ask_user',question='所选位姿几何检查未通过，请确认后续要求。')]
        else:
            calls += [action('check_goal',goal_id=goal['goal_id'])]
    if name not in ('blocked_destination','missing_visual_evidence'):
        calls.append(action('finish_task',status='succeeded',evidence='对应仿真执行和子任务检查已完成。'))
    pending = iter(calls)

    def work(payload, observation, spatial_view):
        if name == 'lifecycle':
            target = 'lobby_center' if session.state['revision'] > 1 else 'corridor_wait'
            if not session.task_goals.goals:
                return action('record_task',goals=[point('p1',target)])
            if session.skill.record['handle'] is not None:
                return action('wait_for_event',event='navigation')
            if spatial_view is None:
                return action('query_map')
            return action('navigate',goal_id='p1',pose=catalog.places[target]['pose'],
                          view_id=spatial_view.record['view_id'],orientation_source='autonomous')
        call = copy.deepcopy(next(pending))
        if call['tool'] == 'navigate':
            # This binding belongs only to an explicit scripted fixture. Actual
            # VLM responses must name the view and are never patched or repaired.
            call['arguments']['view_id'] = spatial_view.record['view_id']
        return call

    actor = AgentLoop(session,lambda *args: Reply.fixture(work(*args)),max_requests=64)
    events = set()
    started,sampled = time.perf_counter(),-1e9
    def record(actor):
        nonlocal sampled
        if navigation.t-sampled >= 1.:
            store.event('navigation_sample',navigation.snapshot())
            sampled = navigation.t

    def scripted_input(actor,emit):
        state,t = actor.state,navigation.t
        if t >= 2 and 'pause' not in events:
            actor.event('pause');events.add('pause')
        if t >= 4 and state['status'] == 'paused' and 'resume' not in events:
            actor.event('resume');events.add('resume')
        if t >= 6 and 'replace' not in events:
            actor.event('replace','改去大厅中央');events.add('replace')
        if t >= 8 and state['revision'] == 2 and 'cancel' not in events:
            actor.event('cancel');events.add('cancel')

    try:
        console = SimpleNamespace(poll=scripted_input) if name == 'lifecycle' else None
        if console is not None:
            store.event('user_event_source',dict(source='explicit_simulation_fixture'))
        state = actor.run(console=console,driver=SimulationDriver(navigation,max_s=160.,fast=True),on_tick=record)
        expected = 'cancelled' if name == 'lifecycle' else 'needs_clarification' if name in (
            'blocked_destination','missing_visual_evidence') else 'succeeded'
        passed = state['status'] == expected and not navigation.collision and navigation.snapshot()['stopped']
        if expected == 'succeeded':
            passed &= session.task_goals.finished()
        if name == 'lifecycle':
            passed &= events == {'pause','resume','replace','cancel'} and state['revision'] == 2
        result = dict(case=name,passed=bool(passed),status=state['status'],expected=expected,
            simulation_s=navigation.t,cpu_wall_s=time.perf_counter()-started,scripted_events=sorted(events),
            decision_turns=actor.requests,model_requests=0,exit_reason=actor.exit_reason,
            task=state,navigation=navigation.snapshot(),scope='synthetic_indoor_goal_fixture',real_robot_success=False)
        session._save('scenario_summary',result)
        return result
    finally:
        actor.close();store.close()


def evaluate(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    results = []
    for name in ('point_heading','point_region_chain','blocked_destination','missing_visual_evidence','lifecycle'):
        row = scenario(name,output)
        results.append(row)
        print(json.dumps({k:row[k] for k in ('case','passed','status','simulation_s','cpu_wall_s')},ensure_ascii=False),flush=True)
    (output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    if not all(r['passed'] for r in results):
        raise RuntimeError('An indoor goal feedback case failed; see summary')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    evaluate(Path(p.parse_args().output))


if __name__ == '__main__':
    main()
