"""Bounded T2 appearance and causal identity diagnostics; inputs and labels stay separate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time

from jsonschema import Draft202012Validator

from . import vlm_api_common as common
from .annotation.rules import digest


PROMPT = Path(__file__).with_name('configs') / 'prompts/wheelchair_t2_appearance.txt'
GEOMETRY_FIELDS = {'v', 'dt', 'ts', 'e', 'o', 'd'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def appearance_schema(pids):
    if (not isinstance(pids, list) or len(set(pids)) != len(pids)
            or any(type(pid) is not int or not 0 <= pid <= 99 for pid in pids)):
        raise ValueError('Invalid current PID set')
    branches = []
    if pids:
        branches.append(dict(type='object', additionalProperties=False, required=['id', 'match'],
                             properties=dict(id=dict(type='integer', enum=pids), match=dict(const='unique'))))
    branches.append(dict(type='object', additionalProperties=False, required=['id', 'match'],
                         properties=dict(id=dict(const=-1), match=dict(enum=['ambiguous', 'none', 'insufficient']))))
    return {'oneOf': branches}


def validate_decision(raw, schema):
    Draft202012Validator(schema).validate(raw)
    if type(raw['id']) is not int:
        raise ValueError('PID must be an integer, not a float or boolean')


def follow_schema(pids, history):
    appearance_schema(pids)
    branches = []
    def branch(ids, states):
        return dict(type='object', additionalProperties=False, required=['id', 'state'],
                    properties=dict(id=dict(type='integer', enum=ids), state=dict(enum=states)))
    if pids:
        branches.append(branch(pids, ['visible']))
    if history['hold_allowed']:
        if type(history.get('current_pid')) is not int or history['current_pid'] not in pids:
            raise ValueError('Holding must reference a current allowed PID')
        branches.append(branch([history['current_pid']], ['occluded']))
    branches.append(branch([-1], ['uncertain', 'lost' if history['confirmed_before'] else 'not_found']))
    return dict(oneOf=branches)


class FollowHistory:
    """An offline semantic replay state, not a robot controller or oracle tracker."""
    def __init__(self, hold_age_s=None):
        if hold_age_s is None:
            from .project_config import load_config
            hold_age_s = load_config()['tasks']['follow']['visual_memory_s']
        self.confirmed = False
        self.key = self.source = self.last_visual_ns = None
        self.hold_age_s = hold_age_s
        self.revoked = False

    def context(self, item, window):
        age = ((item['stamp_ns'] - self.last_visual_ns) / 1e9 if self.last_visual_ns is not None else None)
        if age is not None and age < 0:
            raise ValueError('Replay observations must move forward in time')
        pid = window['mapping'].get(self.key) if item['source'] == self.source else None
        trace = window.get('trajectories', {}).get(window['frames'][-1], {})
        def covers(segments, stitch=False):
            chains = []
            for points in segments:
                times = [p[0] for p in points if p[0] <= 0]
                if not times:
                    continue
                if stitch and chains and times[0] == chains[-1][-1]:
                    chains[-1].extend(times[1:])
                else:
                    chains.append(times)
            return any(age is not None and times[0] <= -age+.15 and times[-1] >= -.15
                       and all(0 <= b-a <= .350001 for a,b in zip(times,times[1:])) for times in chains)
        continuous = bool(pid is not None and age is not None and trace.get('status') == 'available'
                          and covers(trace.get('ego', [])) and covers(
                              [s['points'] for s in trace.get('people', {}).get(self.key, [])], stitch=True))
        return dict(confirmed_before=self.confirmed, current_pid=pid,
                    last_visual_age_s=round(age, 6) if age is not None else None,
                    continuous_track=continuous,
                    hold_allowed=bool(continuous and 0 <= age <= self.hold_age_s and not self.revoked))

    def accept(self, response, item, window):
        if response['state'] == 'visible':
            self.key = next(key for key, pid in window['mapping'].items() if pid == response['id'])
            self.source, self.last_visual_ns = item['source'], item['stamp_ns']
            self.confirmed, self.revoked = True, False
        elif response['state'] != 'occluded':
            self.revoked = True


def appearance_fields(window, description):
    if set(window['fields']) != GEOMETRY_FIELDS:
        raise ValueError('Unexpected observation fields')
    if not isinstance(description, str) or not description.strip():
        raise ValueError('A fixed initial visible description is required')
    return dict(copy.deepcopy(window['fields']), target_description=description.strip())


def causal_description(reference_frame, observation):
    if (reference_frame['source_id'] != observation['source_id']
            or reference_frame['stamp_ns'] > observation['stamp_ns']):
        raise ValueError('Description reference must precede the observation in the same recording')


def select_samples(rows, limit):
    """Balance physical targets and coverage before observing model outputs."""
    if limit < 1:
        raise ValueError('Positive sample budget required')
    groups = {}
    for row in rows:
        groups.setdefault((row['event_id'], row['target_id']), []).append(row)
    selected, chosen = [], set()
    for group in groups.values():
        group.sort(key=lambda row: row['stamp_ns'])
    # At most four samples per target in the first pass: two visible represented
    # examples, then a coverage/visibility challenge and one further time point.
    def visible(row):
        return row['label']['coverage'] == 'represented' and row['label']['visibility'] in ('clear', 'partial')
    for turn in range(4):
        for key in sorted(groups):
            group = groups[key]
            available = [r for r in group if (r['target_id'], r['frame_id']) not in chosen]
            if not available:
                continue
            preferred = ([r for r in available if visible(r)] if turn < 2
                         else [r for r in available if not visible(r)] if turn == 2 else available)
            pool = preferred or available
            previous = [r for r in selected if (r['event_id'], r['target_id']) == key]
            row = (max(pool, key=lambda r: min(abs(r['stamp_ns']-p['stamp_ns']) for p in previous))
                   if previous else pool[0])
            selected.append(row)
            chosen.add((row['target_id'], row['frame_id']))
            if len(selected) == limit:
                return selected
    for row in sorted(rows, key=lambda r: (r['stamp_ns'], r['target_id'])):
        if (row['target_id'], row['frame_id']) not in chosen:
            selected.append(row)
            if len(selected) == limit:
                break
    return selected


def prepare(project, output, limit=20, target_id=None, period_s=None):
    from .annotation.store import Store, database
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; do not overwrite frozen inputs or results')
    store = Store(project)
    try:
        with database(store.path / 'observations.sqlite', True) as db:
            observations = {r['id']: dict(r) for r in db.execute('SELECT id,source_id,stamp_ns FROM frames')}
        rows = []
        for record in store.events():
            event = record['document']
            if event.get('deleted'):
                continue
            for target in event.get('targets', {}).values():
                if target_id is not None and target['id'] != target_id:
                    continue
                for ident, label in target['samples'].items():
                    if label['status'] == 'skipped':
                        continue
                    frame = observations[ident]
                    rows.append(dict(event_id=event['id'], event_revision=record['revision'],
                        target_id=target['id'], target=target, label=label, frame_id=ident,
                        stamp_ns=frame['stamp_ns'], source_id=frame['source_id'], split=event['split']))
        if period_s is not None:
            if target_id is None or period_s <= 0:
                raise ValueError('Replay requires one explicit target and a positive observation period')
            selected = []
            for row in sorted(rows, key=lambda row: row['stamp_ns']):
                if not selected or (row['stamp_ns'] - selected[-1]['stamp_ns']) / 1e9 >= period_s-.02:
                    selected.append(row)
                    if len(selected) >= limit:
                        break
        else:
            selected = select_samples(rows, limit)
        if not selected:
            raise ValueError('No existing T2 samples')
        output.mkdir(parents=True)
        index, references = [], {}
        replay = period_s is not None
        prompt_source = PROMPT.with_name('wheelchair_t2_follow.txt') if replay else PROMPT
        settings = dict(task='wheelchair_follow_replay' if replay else 'wheelchair_follow_appearance',
            mode='replay' if replay else 'appearance', scope='development_draft_diagnostic_not_five_state_gold',
            project_id=store.identity, asset_identity=store.asset_identity, project=str(store.path),
            selection='balanced_targets_temporal_spread_and_coverage_before_inference',
            available_samples=len(rows), planned_requests=len(selected), warmups=1,
            automatic_retries=0, selected_before_inference=True, max_tokens=128,
            renderer_source=file_hash(Path(__file__).with_name('scene_inputs.py')),
            prompt_sha256=file_hash(prompt_source), backend='remote_sglang_dspark', source_events={})
        if replay:
            settings.update(period_s=period_s, hold_age_s=FollowHistory().hold_age_s,
                            selection='one_target_chronological_native_observations_no_future_state')
        for ordinal, row in enumerate(selected, 1):
            case_id = 'F%02d' % ordinal
            target, label = row['target'], row['label']
            frame = store.frame(row['frame_id'])
            reference_frame = store.frame(target['reference_frame'])
            causal_description(reference_frame, frame)
            window = store.window(row['frame_id'])
            if not window['ready']:
                raise ValueError('Three real RGB observations are required: ' + row['frame_id'])
            fields = appearance_fields(window, target['description'])
            binding = dict(visual_input_fingerprint=window['fingerprint'], description=target['description'],
                           description_reference=target['reference_frame'], reference_fingerprint=reference_frame['fingerprint'])
            fingerprint = digest(dict(task=settings['task'], **binding))
            directory = output / 'inputs' / case_id
            directory.mkdir(parents=True)
            images = []
            for i in range(3):
                path = directory / ('image_%d.png' % i)
                store.model_image(window, i).save(path)
                images.append(dict(path=str(path.relative_to(output)), sha256=file_hash(path)))
            item = dict(case_id=case_id, frame_id=row['frame_id'], stamp_ns=frame['stamp_ns'],
                source=frame['source_id'], elapsed_s=frame['elapsed'], fields=fields,
                response_schema=follow_schema(fields['v'], FollowHistory().context(
                    dict(stamp_ns=frame['stamp_ns'], source=frame['source_id']), window)) if replay else appearance_schema(fields['v']), images=images,
                input_fingerprint=fingerprint, observation_binding=binding)
            write_json(directory / 'input.json', dict(item, window=window))
            item['source_input'] = dict(path=str((directory / 'input.json').relative_to(output)),
                                        sha256=file_hash(directory / 'input.json'))
            index.append(item)
            references[case_id] = dict(event_id=row['event_id'], event_revision=row['event_revision'],
                target_id=row['target_id'], target_person_id=target.get('person_id'),
                frame_id=row['frame_id'], label=copy.deepcopy(label), label_sha256=digest(label),
                target_description=target['description'], description_reference=target['reference_frame'],
                target_pid=window['mapping'].get(label.get('true_key')), pid_to_track={str(v):k for k,v in window['mapping'].items()},
                evaluated_input_fingerprint=fingerprint, original_split=row['split'],
                reference_scope='native_draft_identity_and_coverage_only_not_description_uniqueness_gold')
            settings['source_events'][row['event_id']] = row['event_revision']
            print('Prepared', case_id, row['frame_id'], flush=True)
        write_json(output / 'inputs.json', index)
        write_json(output / 'reference.json', references)
        (output / 'prompt.txt').write_bytes(prompt_source.read_bytes())
        settings['inputs_sha256'] = file_hash(output / 'inputs.json')
        settings['reference_sha256'] = file_hash(output / 'reference.json')
        write_json(output / 'settings.json', settings)
    finally:
        store.close()


def load_inputs(output):
    output = Path(output).resolve()
    settings = common.read_json(output / 'settings.json')
    for name, expected in [('inputs.json', settings['inputs_sha256']), ('reference.json', settings['reference_sha256']),
                           ('prompt.txt', settings['prompt_sha256'])]:
        if file_hash(output / name) != expected:
            raise ValueError('Frozen artifact changed: ' + name)
    items = common.read_json(output / 'inputs.json')
    for item in items:
        for spec in [*item['images'], item['source_input']]:
            path = common.asset_path(output, spec['path'])
            if file_hash(path) != spec['sha256']:
                raise ValueError('Frozen input asset changed')
        if set(item['fields']) != GEOMETRY_FIELDS | {'target_description'}:
            raise ValueError('Unexpected task fields; labels must remain separate')
        Draft202012Validator.check_schema(item['response_schema'])
    return settings, items


def run(output, base_url, model='qwen3.8-27b-fp8', prompt_path=None, result_name=None, result_directory=None):
    import httpx
    from .server import check_server
    output = Path(output).resolve()
    settings, items = load_inputs(output)
    if result_name is not None and result_directory is not None:
        raise ValueError('Choose either a result name or an external result directory')
    if prompt_path is not None and result_name is None and result_directory is None:
        raise ValueError('A prompt comparison requires a new result destination')
    result = output
    if result_name is not None:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', result_name):
            raise ValueError('Invalid result name')
        result = output / result_name
        if result.exists():
            raise ValueError('Named result already exists; no implicit replay')
    elif result_directory is not None:
        result = Path(result_directory).resolve()
        if result.exists():
            raise ValueError('Result directory already exists; no implicit replay')
    if (result / 'requests.jsonl').exists() or (result / 'warmups.jsonl').exists():
        raise ValueError('Requests already exist; no implicit resume or replay')
    if result != output:
        content = Path(prompt_path or output/'prompt.txt').read_bytes()
        result.mkdir(parents=True)
        (result / 'prompt.txt').write_bytes(content)
    prompt_source = result / 'prompt.txt'
    prompt_hash = file_hash(prompt_source)
    base_url = base_url.rstrip('/')
    config = dict(base_url=base_url, origin=base_url.removesuffix('/v1'), api_key='', model=model)
    with httpx.Client(timeout=90, trust_env=False, follow_redirects=False,
                      limits=httpx.Limits(max_connections=1)) as client:
        server = check_server(client, config, info=True)
        if server['status'] != 'ready' or server.get('server_args', {}).get('speculative_algorithm') != 'DSPARK':
            raise ValueError('Expected ready remote DSPARK service; no silent backend switch')
        server['runner_source_sha256'] = file_hash(__file__)
        server.update(input_index_sha256=settings['inputs_sha256'], prompt_sha256=prompt_hash,
                      planned_requests=len(items), mode=settings.get('mode', 'appearance'),
                      frozen_inputs=str(output))
        write_json(result / 'server.json', server)
        prompt = prompt_source.read_text()
        replay = settings.get('mode') == 'replay'
        history = FollowHistory(settings.get('hold_age_s', 2.5)) if replay else None
        for item, warmup in [(items[0], True)] + [(item, False) for item in items]:
            fields = copy.deepcopy(item['fields'])
            window = common.read_json(common.asset_path(output, item['source_input']['path']))['window'] if replay else None
            context = history.context(item, window) if replay else None
            schema = follow_schema(fields['v'], context) if replay else item['response_schema']
            if replay:
                fields['last_target'] = context
            paths = [common.asset_path(output, spec['path']) for spec in item['images']]
            case = dict(paths=paths, model=dict(fields=fields, response_schema=schema,
                        user_prompt=json.dumps(fields, ensure_ascii=False, separators=(',', ':'))))
            body = common.make_body(prompt, case, max_tokens=settings['max_tokens'], model=model)
            row = dict(case_id=item['case_id'], warmup=warmup, input_fingerprint=item['input_fingerprint'],
                       request_sha256=digest(body), prompt_sha256=prompt_hash, valid=False)
            if replay:
                row.update(base_input_fingerprint=item['input_fingerprint'], history_context=context,
                           input_fingerprint=digest(dict(task=settings['task'], base=item['input_fingerprint'],
                                                         history=context, schema=schema)))
            timing, started, transport_failed = {}, time.perf_counter(), False
            try:
                with client.stream('POST', base_url + '/chat/completions', json=body) as response:
                    row['http_status'] = response.status_code
                    if response.status_code != 200:
                        transport_failed = True
                        raise ValueError('Non-success HTTP response')
                    row.update(common.read_stream(response, started, timing))
                raw = json.loads(row['raw_content'])
                validate_decision(raw, schema)
                if (row['server_model'] != model or row['finish_reason'] != 'stop'
                        or row['reasoning_content_present'] or row['usage'].get('reasoning_tokens', 0)):
                    raise ValueError('Invalid completion metadata')
                row.update(response=raw, valid=True)
            except Exception as error:
                row['error_type'] = type(error).__name__
                if 'raw_content' not in row:
                    transport_failed = True
            row['timing'] = timing
            with (result / ('warmups.jsonl' if warmup else 'requests.jsonl')).open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print('Warmup' if warmup else item['case_id'], row.get('response', row.get('error_type')), flush=True)
            if transport_failed:
                raise RuntimeError('Stopped on transport failure; saved record, no retry')
            if replay and not warmup:
                if not row['valid']:
                    raise RuntimeError('Stopped causal replay on invalid response; no invented state or retry')
                history.accept(row['response'], item, window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_command = commands.add_parser('prepare', help='Freeze bounded native-draft diagnostics, no model calls')
    prepare_command.add_argument('--project', type=Path, required=True)
    prepare_command.add_argument('--out', type=Path, required=True)
    prepare_command.add_argument('--limit', type=int, default=20)
    prepare_command.add_argument('--target', help='Select one existing target task without changing its labels')
    prepare_command.add_argument('--period-s', type=float, help='Freeze one chronological replay at this minimum interval')
    run_command = commands.add_parser('run', help='One pass on a checked remote DSPARK service, no retry')
    run_command.add_argument('--out', type=Path, required=True)
    run_command.add_argument('--base-url', default='http://127.0.0.1:30000/v1')
    run_command.add_argument('--prompt', type=Path, help='Explicit frozen comparison prompt; does not change the default')
    run_command.add_argument('--result-name', help='New comparison directory under the frozen inputs')
    run_command.add_argument('--result-out', type=Path, help='New results directory while reusing read-only frozen inputs')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.project, args.out, args.limit, args.target, args.period_s)
    else:
        run(args.out, args.base_url, prompt_path=args.prompt, result_name=args.result_name,
            result_directory=args.result_out)


if __name__ == '__main__':
    main()
