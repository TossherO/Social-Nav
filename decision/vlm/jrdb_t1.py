"""T1 offline workflow: prepare typed inputs, review labels and compare predictions.

Inference lives only in benchmark_vlm. Native JRDB groups are a diagnostic,
never automatic T1 gold. No training, GPU calls or credential access here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image, ImageDraw

from decision.vlm import jrdb_t1_common as t1, jrdb_t1_inputs as inputs
from decision.vlm import vlm_api_common as api
from decision.vlm.project_config import CONFIG_DIR, load_config, project_path
from decision.vlm.visualize_jrdb_t1_gt import load_reviewed_label

CONFIG = load_config()
SEQUENCES = CONFIG['development']['sequences']
DEFAULT_FRAMES = CONFIG['development']['end_frames']


def native_diagnostic(window):
    path = t1.jrdb.label_path(window.root, 'people', window.sequence, '0')
    rows = t1.jrdb.load_json(path)['labels'][window.frames[-1] + '.jpg']
    groups = defaultdict(list)
    for row in rows:
        identity = row['label_id']
        group = (row.get('social_group') or {}).get('cluster_ID')
        if identity in window.active and group is not None:
            groups[str(group)].append(window.pid_map[identity])
    return {'kind': 'native_front_membership_NOT_reviewed_REL_gold',
            'groups': sorted(sorted(g) for g in groups.values() if len(g) >= 2),
            'source_sha256': t1.file_digest(path)}


def prepare(args):
    root, out = args.root.absolute(), args.out.resolve()
    t1.check_output(root, out)
    if out.exists():
        raise ValueError('Use a NEW output directory; never overwrite frozen inputs')
    if len(set(args.scenes)) != len(args.scenes) or len(set(args.frames)) != len(args.frames):
        raise ValueError('Duplicate scene/frame selection')
    cases, files, image_bytes = [], {}, 0
    system = (CONFIG_DIR / 'prompts/jrdb_t1_rel.txt').read_text(encoding='utf-8').strip()
    out.mkdir(parents=True)
    (out / 'references').mkdir()
    (out / 'system_prompt.txt').write_text(system + '\n', encoding='utf-8')
    files['system_prompt.txt'] = t1.file_digest(out / 'system_prompt.txt')
    # Frame-major order makes the default first three cases different scenes.
    for number in args.frames:
        for short in args.scenes:
            sequence = SEQUENCES.get(short, short)
            t1.jrdb.validate_id(sequence, f'{number:06d}')
            available = sorted(p.stem for p in (root / 'images/image_0' / sequence).glob('*.jpg'))
            frames = t1.choose_frames(available, f'{number:06d}', 15)
            window = t1.build_window(root, sequence, frames, scope='front', lidar=False)
            case_id = f'{short}_{number:06d}'
            t1.jrdb.validate_id(case_id, f'{number:06d}')
            folder = out / 'cases' / case_id
            local = inputs.save_bundle(window, 'object', folder, contact_sheet=args.contact_sheet)
            model = t1.jrdb.load_json(folder / 'model_input.json')
            # Geometry-only whitelist; local templates and source IDs are not exported.
            names = [f'cases/{case_id}/model_input.json', *[f'cases/{case_id}/{n}' for n in model['images']]]
            model['images'] = names[1:]
            t1.write_json(folder / 'model_input.json', model)
            # Fingerprint covers the final transport metadata, not pre-rewrite paths.
            local['input_fingerprint'] = t1.digest({'model_input': model, 'images': local['image_sha256'], 'system': system})
            t1.write_json(folder / 'manifest.json', local)
            template = t1.jrdb.load_json(folder / 'annotation_template.json')
            template['input_fingerprint'] = local['input_fingerprint']
            t1.write_json(folder / 'annotation_template.json', template)
            for name in names:
                files[name] = t1.file_digest(out / name)
            image_bytes += sum((out / n).stat().st_size for n in names[1:])
            t1.write_json(out / 'references' / f'{case_id}.native.json', native_diagnostic(window))
            cases.append({'case_id': case_id, 'scene': short, 'sequence': sequence,
                          'frame': f'{number:06d}', 'input_file': names[0],
                          'input_fingerprint': local['input_fingerprint']})
            print(f'PREPARED {case_id}: {len(window.valid_pid)} people', flush=True)
    t1.write_json(out / 'manifest.json', {
        'task': 'JRDB_T1_REL_development', 'profile': 'context_bev_front',
        'prompt_profile': 'object', 'wire_protocol': 'object',
        'native_social_labels_included': False, 'credentials_included': False,
        'image_bytes_total': image_bytes, 'cases': cases, 'files': files,
        'limitations': ['development_NOT_blind_or_reviewed_gold', 'oracle_geometry_and_PID',
                        'nominal_timestamps_no_ego_motion', 'references_and_templates_NOT_HTTP_inputs']})
    api.load_bundle(out)
    return out


def reviewed(args):
    window, model, manifest = inputs.load_window(args.input_dir)
    adapter = {**manifest, 'model': {'VALID_PID': model['fields']['v']}}
    label = load_reviewed_label(args.labels, adapter, args.allow_draft)
    t1.check_output(window.root, args.out, input_dirs=(args.input_dir,))
    if args.out.exists():
        raise ValueError('Use a NEW review output directory')
    args.out.mkdir(parents=True)
    styles, links = t1.relation_layout(label['response'])
    image = inputs.render_focus(window, 2, gallery=False, styles=styles, links=links, context_bev=True)
    gold = label.get('review_status') == 'reviewed'
    name = 'gt' if gold else 'draft'
    inputs.text_tag(ImageDraw.Draw(image), (8, 516), 'T1 REVIEWED GOLD' if gold else 'T1 DRAFT - NOT GOLD', size=18)
    image.save(args.out / f'{name}.png')
    t1.write_json(args.out / f'{name}.json', label)
    if gold:
        t1.write_json(args.out / 'assistant_response.json', label['response'])
    print(args.out)


def compare(args):
    """Render predictions beside native groups; no accuracy claim or gold export."""
    _, _, cases = api.load_bundle(args.bundle)
    indexed = {c['entry']['case_id']: c for c in cases}
    if args.out.exists():
        raise ValueError('Use a NEW comparison directory')
    t1.check_output(None, args.out, input_dirs=(args.bundle,))
    planned, seen = [], set()
    for line in args.requests.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if not row.get('valid'):
            continue
        if row['case_id'] in seen:
            raise ValueError('Duplicate prediction case IDs; do not overwrite comparison images')
        seen.add(row['case_id'])
        case = indexed[row['case_id']]
        if row.get('input_fingerprint') != case['fingerprint']:
            raise ValueError('Prediction was not produced from these exact bundle inputs')
        response = row.get('canonical_rel', row.get('response'))
        api.validate_rel(response, case)
        folder = api.asset_path(args.bundle, case['entry']['input_file']).parent
        window, _, _ = inputs.load_window(folder)
        t1.check_output(window.root, args.out, input_dirs=(args.bundle,))
        reference = native_diagnostic(window)
        native = {'g': [{'m': g, 'r': 'u', 'k': 0} for g in reference['groups']], 'u': []}
        planned.append((row['case_id'], window, response, native))
    # Preflight every accepted prediction before producing any partial output.
    if not planned:
        raise ValueError('No valid predictions to compare')
    args.out.mkdir(parents=True)
    for case_id, window, response, native in planned:
        canvas = Image.new('RGB', (2560, 584), inputs.BG)
        for offset, value, title in ((0, response, 'PREDICTION'), (1280, native, 'JRDB NATIVE MEMBERSHIP - NOT T1 GOLD')):
            styles, links = t1.relation_layout(value)
            panel = inputs.render_focus(window, 2, gallery=False, styles=styles, links=links, context_bev=True)
            canvas.paste(panel, (offset, 40))
            inputs.text_tag(ImageDraw.Draw(canvas), (offset + 10, 9), title, size=18)
        canvas.save(args.out / f"{case_id}.png")
    print(args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--root', type=Path, default=project_path(CONFIG['dataset_roots']['jrdb']))
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--scenes', nargs='+', default=list(SEQUENCES))
    prep.add_argument('--frames', nargs='+', type=int, default=DEFAULT_FRAMES)
    prep.add_argument('--contact-sheet', action='store_true', help='Optional human preview, never an HTTP input')
    gold = commands.add_parser('reviewed')
    gold.add_argument('--input-dir', type=Path, required=True)
    gold.add_argument('--labels', type=Path, required=True)
    gold.add_argument('--out', type=Path, required=True)
    gold.add_argument('--allow-draft', action='store_true')
    vis = commands.add_parser('compare')
    vis.add_argument('--bundle', type=Path, required=True)
    vis.add_argument('--requests', type=Path, required=True)
    vis.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    return {'prepare': prepare, 'reviewed': reviewed, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
