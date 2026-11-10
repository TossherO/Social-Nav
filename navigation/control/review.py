"""Summarize recorded navigation decisions and chassis events without replay."""
import argparse
from collections import Counter
import json
from pathlib import Path


def _rows(path):
    if not path.is_file():return None
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(directory):
    root=Path(directory).expanduser().resolve()
    if not (root/'control.jsonl').exists() and (root/'run').is_dir():root=root/'run'
    controls=_rows(root/'control.jsonl')
    safety=_rows(root/'safety.jsonl')
    chassis=_rows(root/'chassis/events.jsonl')
    cycles=_rows(root/'planning/cycles.jsonl')
    telemetry=root/'chassis/telemetry_summary.json'
    quality=json.loads(telemetry.read_text()) if telemetry.is_file() else None
    summary_path=root/'summary.json'
    run_summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if controls is None and safety is None and chassis is None:
        raise ValueError('No control, safety, or chassis event logs found')

    plans={}
    for row in cycles or []:
        if row.get('packet_binding'):
            plans[row['packet_binding']]=row
    proposals={};ambiguous_packets=set()
    source=[]
    for row in controls or []:
        binding=row.get('packet_binding');proposal=row.get('proposal') or {};request=row.get('request')
        plan=plans.get(binding) or {};trajectory=plan.get('trajectory') or {}
        forecast=(trajectory.get('trajectory') or {}).get('people_check')
        item=dict(kind='control',stamp_s=row.get('stamp_s'),clock_domain=row.get('clock_domain'),
            packet_binding=binding,sequence_id=row.get('sequence_id'),request=request,
            status=proposal.get('status'),reason=row.get('reason') or proposal.get('reason'),
            velocity=proposal.get('velocity'),trajectory_binding=proposal.get('trajectory_binding'),
            feedback=row.get('feedback'),people_check=forecast,
            candidates={k:proposal.get(k) for k in ('candidate_count','ranked_candidates','valid_candidates',
                'certification_attempts','unattempted_candidates','selection_stopped_early','rejections')})
        source.append(item)
        if binding:
            if binding in proposals:ambiguous_packets.add(binding)
            else:proposals[binding]=item

    for binding in ambiguous_packets:proposals.pop(binding,None)

    by_sequence={};ambiguous=set()
    for row in safety or []:
        binding=row.get('proposal_binding');linked=proposals.get(binding)
        item=dict(kind='safety',stamp_s=row.get('stamp_s'),clock_domain=row.get('clock_domain'),
            decision_complete_mono=row.get('decision_complete_mono'),sequence_id=row.get('sequence_id'),
            proposal_binding=binding,control_sequence_id=None if linked is None else linked['sequence_id'],
            trajectory_binding=row.get('trajectory_binding'),state=row.get('state'),reason=row.get('reason'),
            proposed_velocity=row.get('proposed_velocity'),velocity=row.get('velocity'),
            people_check=row.get('people_check'),feedback=row.get('feedback'),
            remaining_validity_s=row.get('remaining_validity_s'),check=row.get('check'))
        source.append(item)
        key=row.get('sequence_id')
        if key is not None:
            if key in by_sequence:ambiguous.add(key)
            else:by_sequence[key]=item
    for key in ambiguous:by_sequence.pop(key,None)
    source.sort(key=lambda x:(str(x['clock_domain']),float('inf') if x['stamp_s'] is None else x['stamp_s']))

    output=[];latest_authorization=None
    for row in sorted(chassis or [],key=lambda r:r.get('mono',float('inf'))):
        kind=row.get('kind');item=dict(kind=kind,mono=row.get('mono'))
        if kind=='authorization':
            packet=row.get('packet') or {};diag=packet.get('diagnostics') or {}
            sequence=diag.get('safety_sequence');matched=by_sequence.get(sequence)
            item.update(packet=packet,matched_safety_sequence=None if matched is None else sequence,
                matched_control_sequence=None if matched is None else matched.get('control_sequence_id'))
            latest_authorization=dict(received_mono=row.get('mono'),safety_sequence=sequence,
                allowed=packet.get('allowed'),units=packet.get('units'),expires_mono=packet.get('expires_mono'))
        elif kind=='wire_target':
            item.update(units=row.get('units'),velocity_mps_rps=row.get('velocity_mps_rps'),
                write_started_mono=row.get('write_started_mono'),changed=row.get('changed'),
                preceding_authorization=latest_authorization)
        elif kind=='feedback':
            feedback=row.get('feedback') or {}
            item.update(velocity_mps_rps=feedback.get('velocity_mps_rps'),mode_raw=feedback.get('mode_raw'),
                candidate_checksum_matches=feedback.get('candidate_checksum_matches'),
                timestamp_source=row.get('timestamp_source'),device_measurement_stamp_s=row.get('device_measurement_stamp_s'))
        elif kind=='state':
            item.update(state=row.get('state'),reason=row.get('reason'),arm_revision=row.get('arm_revision'))
        else:
            item['data']={k:v for k,v in row.items() if k not in ('kind','mono')}
        output.append(item)
    drops=[r['telemetry_dropped'] for r in safety or [] if type(r.get('telemetry_dropped')) is int]
    return dict(scope='offline_recorded_events_not_execution_feedback',source_directory=str(root),
        input_available=dict(control=controls is not None,safety=safety is not None,chassis=chassis is not None,
                             planning_cycles=cycles is not None,chassis_telemetry_summary=quality is not None,
                             run_summary=summary_path.is_file()),
        run_quality=dict(success=run_summary.get('success'),error=run_summary.get('error'),
                         planning_record=run_summary.get('planning_record'),
                         safety_telemetry_dropped_max=max(drops) if drops else None),
        chassis_telemetry_quality=quality,ambiguous_packet_bindings=sorted(ambiguous_packets),
        ambiguous_safety_sequences=sorted(ambiguous),
        counts=dict(control=len(controls or []),safety=len(safety or []),chassis=len(chassis or []),
            safety_reasons=dict(Counter(r.get('reason') or ('allowed' if r.get('state')=='allowed' else 'reason_missing') for r in safety or [])),
            chassis_kinds=dict(Counter(r.get('kind','unknown') for r in chassis or []))),
        notes=['源时钟与主机单调时钟分别保留；不通过相减或最近时间推断跨时钟因果。',
            '安全结果通过 proposal_binding 关联控制；底盘授权通过 safety_sequence 关联安全结果，缺失或重复时不关联。',
            '串口写入的 preceding_authorization 仅为之前收到的最近授权上下文，不证明该授权导致此次写入。',
            '底盘反馈时间为主机接收时间，不能视作设备测量时间；空记录和不完整日志不代表成功。'],
        source_clock_events=source,chassis_monotonic_events=output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Run directory or interactive session containing run/')
    parser.add_argument('--output',required=True,help='New JSON file; existing output is never overwritten')
    args=parser.parse_args(argv)
    report=summarize(args.run)
    with Path(args.output).open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(Path(args.output)),counts=report['counts'],
                          input_available=report['input_available']),ensure_ascii=False))


if __name__=='__main__':main()
