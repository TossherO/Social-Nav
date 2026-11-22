"""Summarize an integrated run recorded with --trace-timing, without loading ROS or CUDA."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def rows(directory, name):
    with (directory / (name + '.jsonl')).open() as stream:
        for line in stream:
            yield json.loads(line)


def distribution(values):
    if not values:
        return {'n': 0}
    p50, p95 = np.percentile(values, [50, 95])
    return dict(n=len(values), p50=float(p50), p95=float(p95), maximum=float(max(values)))


def map_and_goal_timings(trace, input_ready=None):
    first_control = {}
    goals = {}
    first_reference = {}
    publications = []
    queued = {}
    waits = []
    input_to_publish = []
    has_goal_trace = False
    for row in trace:
        event = row['event']
        stamp = row.get('measurement_stamp_s')
        if event == 'map_submit':
            queued[stamp] = row['monotonic_s']
        elif event == 'map_start' and stamp in queued:
            waits.append(1000 * (row['monotonic_s'] - queued.pop(stamp)))
        elif event == 'map_safety_sent':
            publications.append(row)
            if input_ready is not None and stamp in input_ready:
                input_to_publish.append(1000 * (row['monotonic_s'] - input_ready[stamp]))
        elif event == 'control_cycle':
            has_goal_trace |= 'goal_key' in row
            if row.get('scene_binding') is not None:
                first_control.setdefault(row['scene_binding'], 1000 * (row['now'] - stamp))
            if row.get('goal_key') is not None:
                goal = (row['reference_epoch'], *row['goal_key'])
                goals.setdefault(goal, row['goal_checked_mono'])
            if row.get('reference_goal_key') is not None:
                key = (row['reference_epoch'], *row['reference_goal_key'])
                if key in goals:
                    first_reference.setdefault(key, 1000 * (row['reference_checked_mono'] - goals[key]))
    hz = None
    if len(publications) > 1:
        elapsed = publications[-1]['monotonic_s'] - publications[0]['monotonic_s']
        if elapsed > 0:
            hz = (len(publications) - 1) / elapsed
    return dict(published_maps=len(publications), published_maps_per_wall_s=hz,
                has_goal_use_trace=has_goal_trace,
                input_ready_to_map_sent_ms=distribution(input_to_publish),
                submit_to_map_start_ms=distribution(waits),
                map_source_to_first_control_ms=distribution(list(first_control.values())),
                observed_goal_revisions=len(goals),
                goal_to_first_available_reference_ms=distribution(list(first_reference.values())))


def summarize(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    input_ready = {row['measurement_stamp_s']: row['monotonic_s']
                   for row in rows(directory, 'timing_input') if row['event'] == 'input_ready'}
    report = map_and_goal_timings(rows(directory, 'timing'), input_ready)
    safety_uses = {}
    gate_times = []
    telemetry_dropped = 0
    for row in rows(directory, 'safety'):
        telemetry_dropped = max(telemetry_dropped, row.get('telemetry_dropped', 0))
        for use in row.get('grid_uses', ()):
            key = (use['measurement_stamp_s'], use['snapshot_id'])
            safety_uses.setdefault(key, 1000 * (use['source_clock_s'] - use['measurement_stamp_s']))
        if 'independent_gate' in row['timing_ms']:
            gate_times.append(row['timing_ms']['independent_gate'])
    report.update(map_source_to_first_safety_check_ms=distribution(list(safety_uses.values())),
                  active_safety_gate_ms=distribution(gate_times), safety_telemetry_dropped=telemetry_dropped)
    dwa = []
    for row in rows(directory, 'control'):
        proposal = row.get('proposal') or {}
        if proposal.get('control_mode') == 'dwa' and proposal.get('candidate_count', 0) > 0:
            dwa.append(row['compute_ms'])
    report['control_with_dwa_candidates_ms'] = distribution(dwa)
    times = defaultdict(list)
    events = Counter()
    exits = Counter()
    for row in rows(directory, 'reference'):
        events[row['event']] += 1
        if row['event'] not in ('discarded', 'reference_ready'):
            continue
        searches = [a for a in row.get('search_attempts', ()) if a.get('search_ms') is not None]
        group = 'native_search' if searches else 'no_timed_native_search'
        exits[(group, row.get('result_reason') or row.get('result_status') or 'no_result')] += 1
        for stage, value in row.get('planning_timing_ms', {}).items():
            times[group + '/' + stage].append(value)
        for attempt in searches:
            kind = 'expanded' if attempt.get('expanded', 0) else 'no_expansion'
            times['native_attempt/' + kind].append(attempt['search_ms'])
    report['reference_events'] = dict(events)
    report['reference_outcomes'] = {'/'.join(key): n for key, n in exits.items()}
    report['reference_stages_ms'] = {k: distribution(v) for k, v in sorted(times.items())}
    report['run_counts'] = summary['counts']
    report['map_scheduling'] = summary.get('map_scheduling')
    report['trace_loss'] = {name: json.loads((directory / (name + '.summary.json')).read_text())['dropped']
                            for name in ('timing', 'timing_input')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='run directory containing timing.jsonl and summary.json')
    print(json.dumps(summarize(parser.parse_args().run), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
