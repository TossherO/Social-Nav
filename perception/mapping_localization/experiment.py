"""Replay actual ordered frontend/registration events with and without a coverage mask."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .context import LocalizationContext
from .coverage import PriorCoverage
from .results import file_fingerprint


def compare_events(events, *, coverage, source, max_map_age, confirm_matches):
    contexts = [LocalizationContext(source,requires_map=True,allow_local_only=True,
        max_map_age=max_map_age,confirm_matches=confirm_matches,coverage=region) for region in (None,coverage)]
    records = []
    for event in events:
        if event['kind'] == 'match':
            for context in contexts:
                context.add_match(event['row'])
        elif event['kind'] == 'frontend':
            pairs = [context.update(event['row']) for context in contexts]
            if pairs == [None,None]:
                continue
            if any(pair is None for pair in pairs):
                raise ValueError('Coverage policy changed local observation acceptance')
            a,b = pairs[0][0],pairs[1][0]
            if any(a[k] != b[k] for k in ('session_id','generation','frame_id','health','position','quaternion_xyzw')):
                raise ValueError('Coverage policy changed the continuous local pose/lifecycle')
            records.append(dict(stamp_s=b['stamp_s'],position=b['position'],session_id=b['session_id'],
                generation=b['generation'],health=b['health'],baseline=pairs[0][1]['status'],
                masked=pairs[1][1]['status'],coverage=pairs[1][1]['coverage_status'],
                association_generation=pairs[1][1]['association_generation'],
                measurement_stamp_s=pairs[1][1]['measurement_stamp_s']))
        else:
            raise ValueError('Unknown recorded context event')
    if not records:
        raise ValueError('No recorded frontend events')
    transitions = []
    for row in records:
        if not transitions or transitions[-1]['status'] != row['masked']:
            transitions.append(dict(stamp_s=row['stamp_s'],offset_s=row['stamp_s']-records[0]['stamp_s'],
                status=row['masked'],association_generation=row['association_generation'],coverage=row['coverage']))
    return dict(scope='same_source_partial_coverage_interface_experiment',
        matcher_uses_full_prior=True,geometric_out_of_map_registration_verified=False,
        independent_route_accuracy_verified=False,local_pose_and_lifecycle_unchanged=True,
        states=dict(baseline=dict(Counter(r['baseline'] for r in records)),
                    masked=dict(Counter(r['masked'] for r in records))),
        transitions=transitions,records=records)


def plot_comparison(result):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11,5),constrained_layout=True)
    FigureCanvasAgg(figure)
    route,time = figure.subplots(1,2)
    rows = result['records'];positions = np.asarray([r['position'] for r in rows])
    colors = dict(PRIOR_ASSOCIATED='#157b54',LOCAL_ONLY='#dd7522',ASSOCIATING='#3771bb',LOCAL_INVALID='#9d3046')
    route.plot(positions[:,0],positions[:,1],color='#c6ccd0',linewidth=1)
    for status,color in colors.items():
        select = np.array([r['masked'] == status for r in rows])
        if select.any():
            route.scatter(positions[select,0],positions[select,1],s=12,c=color,label=status)
    route.axis('equal');route.legend(fontsize=8)
    route.set(xlabel='Continuous odom x (m)',ylabel='y (m)',title='One frontend / masked prior acceptance')
    stamp = np.array([r['stamp_s']-rows[0]['stamp_s'] for r in rows])
    time.step(stamp,[int(r['baseline']=='PRIOR_ASSOCIATED') for r in rows],where='post',label='Full prior',color='#7b8390')
    time.step(stamp,[int(r['masked']=='PRIOR_ASSOCIATED') for r in rows],where='post',label='Coverage mask',color='#157b54')
    time.set(xlabel='Recorded seconds',ylabel='Accepted prior association',yticks=[0,1],title='Real registrations / no frontend restart')
    time.legend()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Successful mixed-mode run with --keep-details')
    parser.add_argument('--out',required=True,help='New comparison directory')
    args = parser.parse_args()
    run,target = Path(args.run),Path(args.out)
    settings = json.loads((run/'run.json').read_text())
    if not settings.get('success') or not settings.get('association_coverage_record') or target.exists():
        raise ValueError('Use a successful coverage-masked run and a new comparison directory')
    coverage = PriorCoverage(settings['association_coverage_record'],
        asset_fingerprint=settings['association_coverage_record']['map_asset_fingerprint'])
    events_path = run/'context_events.jsonl'
    with events_path.open() as stream:
        events = [json.loads(line) for line in stream]
    config = settings.get('environment_configuration')
    if config is None:
        raise ValueError('Run did not freeze its environment configuration')
    result = compare_events(events,coverage=coverage,source=settings['localization_source'],
        max_map_age=config['association_max_age_s'],confirm_matches=config['association_confirm_matches'])
    actual = [json.loads(line) for line in (run/'localization.jsonl').read_text().splitlines()]
    if [(r['stamp_s'],r['status']) for r in actual] != [(r['stamp_s'],r['masked']) for r in result['records']]:
        raise ValueError('Ordered-event replay differs from the actual public localization states')
    result.update(source_events=file_fingerprint(events_path),coverage_fingerprint=coverage.fingerprint,
                  actual_public_states_match=True)
    target.mkdir(parents=True)
    (target/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot_comparison(result).savefig(target/'comparison.png',dpi=130)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'},ensure_ascii=False))


if __name__ == '__main__':
    main()
