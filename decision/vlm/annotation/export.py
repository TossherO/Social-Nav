"""Export reviewed labels; no VLM execution and no rewrite of source labels."""
import copy
from pathlib import Path

from decision.vlm.scene_inputs import renderer_identity, MODEL_BACKGROUND
from decision.vlm.trajectory_inputs import HISTORY_S, MAX_GAP_S, MAX_POSE_AGE_S
from .rules import dumps, rel_response, require
from .store import Store


def export_project(project, output, inputs=False):
    store = Store(project)
    try:
        return _export_project(store, output, inputs)
    finally:
        store.close()


def _export_project(store, output, inputs):
    events = store.events()
    selected = []
    roots, locations, sequences = {}, {}, {}
    for row in events:
        event = row['document']
        complete = any(s['status'] == 'complete' for s in event.get('t1', {}).values()) or any(
            s['status'] == 'complete' for t in event.get('targets', {}).values() for s in t.get('samples', {}).values())
        if not complete:
            continue
        store.validate_event(event)
        require(event['split'] != 'unassigned', '已标事件尚未设置开发／验证／留出划分：' + event['name'])
        for name, groups in (('root_group', roots), ('location_group', locations), ('source_id', sequences)):
            key = event.get(name)
            if key:
                require(groups.get(key, event['split']) == event['split'], '同源／同地点跨划分')
                groups[key] = event['split']
        selected.append(row)
    require(bool(selected), '没有可导出的已标样本')
    output = Path(output).expanduser().resolve()
    require(not output.exists(), '导出目标已存在，请选择新目录')
    output.mkdir(parents=True)
    t1, t2, sources, rendered = [], [], [], set()
    for row in selected:
        event, revision = row['document'], row['revision']
        sources.append({k: event[k] for k in ('id', 'source_id', 'name', 'start', 'end', 'root_group', 'location_group', 'split')})
        def common(ident, label):
            window, frame = store.window(ident), store.frame(ident)
            if inputs and ident not in rendered:
                directory = output / 'inputs' / event['source_id'] / str(frame['ordinal'])
                directory.mkdir(parents=True, exist_ok=True)
                for i, fid in enumerate(window['frames']):
                    image = store.model_image(window, i)
                    image.save(directory / ('image_' + str(i) + '.png'))
                (directory / 'input.json').write_text(dumps(window), encoding='utf-8')
                rendered.add(ident)
            return dict(event_id=event['id'], event_revision=revision, frame_id=ident, stamp_ns=frame['stamp_ns'],
                        split=event['split'], input_fingerprint=window['fingerprint'], protocol=window['protocol'],
                        pid_to_track={str(pid): key for key, pid in window['mapping'].items()},
                        reviewer=label['reviewer'], reviewed_at=label['reviewed_at']), window, frame
        for ident, label in event.get('t1', {}).items():
            if label['status'] != 'complete':
                continue
            base, window, _ = common(ident, label)
            reference = copy.deepcopy(label.get('reference_groups', []))
            t1.append(dict(base, response=None if reference else rel_response(label, window['mapping']),
                           semantic_eligible=not reference, unrepresented_groups=reference,
                           note=label.get('note', ''), perception_tier='P1_causal_source_tracks'))
        for target in event.get('targets', {}).values():
            for ident, label in target.get('samples', {}).items():
                if label['status'] != 'complete':
                    continue
                base, window, frame = common(ident, label)
                facts = {k: copy.deepcopy(v) for k, v in label.items() if k not in ('binding', 'reviewer', 'reviewed_at', 'status')}
                # These files are gold, never the model request body.
                t2.append(dict(base, target_id=target['id'], description=target['description'],
                               description_source=target.get('description_source', 'current_reference_frame'),
                               description_reference=target['reference_frame'], target_person_id=target.get('person_id'),
                               absent=target.get('absent', False), facts=facts, label_kind='fixed_facts_not_history_state_gold'))
    for name, rows in (('t1.jsonl', t1), ('t2.jsonl', t2)):
        (output / name).write_text(''.join(dumps(r)+'\n' for r in rows), encoding='utf-8')
    source_ids = {row['document']['source_id'] for row in selected}
    observation_sources = [{key: source[key] for key in ('id', 'name', 'bag_fingerprint', 'records_sha256',
                           'calibration_sha256', 'calibration_scope', 'perception_provenance') if key in source}
                           for source in store.sources() if source['id'] in source_ids]
    summary = dict(project_id=store.identity, asset_identity=store.asset_identity, events=sources,
                   rgb_association_identity=store.rgb_cache.identity if store.rgb_cache else None,
                   observation_sources=observation_sources,
                   input_rendering=renderer_identity(),
                   model_input_background=MODEL_BACKGROUND,
                   display_view=store.view_binding,
                   trajectory_policy=dict(history_per_image_s=HISTORY_S,max_gap_s=MAX_GAP_S,
                       max_pose_age_s=MAX_POSE_AGE_S,frame='each_image_lidar',ego_reference='lidar_origin'),
                   counts=dict(t1=len(t1), t2=len(t2)), model_inputs_included=inputs,
                   t2_scope='fixed_facts; no fabricated LAST_TARGET or v/o/a/l/n history gold')
    (output / 'manifest.json').write_text(dumps(summary)+'\n', encoding='utf-8')
    return summary
