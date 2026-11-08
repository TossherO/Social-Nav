"""Estimate video-to-host time from manually annotated screen marker changes.

No video decoding, sensor access or command output. Display latency, exposure,
and input timestamp errors are unknown; this is not a guaranteed time bound.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(markers, marks):
    if len(marks)<3:raise ValueError('At least three marker transitions are required')
    index={r['marker_id']:r for r in markers}
    if len(index)!=len(markers):raise ValueError('Duplicate host marker IDs')
    points=[]
    for row in marks:
        marker_id=int(row['marker_id'])
        before=float(row['video_time_before_s']);after=float(row['video_time_after_s'])
        if not math.isfinite(before) or not math.isfinite(after) or before<0 or not before<after:
            raise ValueError('Each video timestamp pair must be finite, nonnegative and increasing')
        if marker_id not in index:raise ValueError('Video marker ID not in host record')
        if points and (marker_id<=points[-1]['marker_id'] or before<points[-1]['video_after_s']):
            raise ValueError('Marker IDs and video intervals must be strictly ordered')
        source=index[marker_id]
        start=float(source['draw_request_mono']);end=float(source['draw_queued_mono'])
        if not (math.isfinite(start) and math.isfinite(end) and start<=end):
            raise ValueError('Invalid marker draw times')
        if points and start<=points[-1]['host_request_s']:
            raise ValueError('Host marker times must be strictly ordered')
        points.append(dict(marker_id=marker_id,video_before_s=before,video_after_s=after,
            host_request_s=start,host_queued_s=end))
    x=[(p['video_before_s']+p['video_after_s'])/2 for p in points]
    y=[(p['host_request_s']+p['host_queued_s'])/2 for p in points]
    if x[-1]-x[0]<.5:raise ValueError('Markers must span at least 0.5 seconds of video')
    xmean=sum(x)/len(x);ymean=sum(y)/len(y)
    denominator=sum((a-xmean)**2 for a in x)
    if denominator<=0:raise ValueError('Video timestamps have no time span')
    slope=sum((a-xmean)*(b-ymean) for a,b in zip(x,y))/denominator
    if not math.isfinite(slope) or slope<=0:raise ValueError('Invalid relative video clock rate')
    offset=ymean-slope*xmean
    residuals=[b-(slope*a+offset) for a,b in zip(x,y)]
    return dict(host_monotonic_per_video_second=slope,host_offset_s=offset,
        max_abs_fit_residual_s=max(abs(x) for x in residuals),
        max_video_transition_bracket_s=max(p['video_after_s']-p['video_before_s'] for p in points),
        markers=points,physical_display_latency_known=False,
        scope='diagnostic_fit_not_a_calibrated_or_guaranteed_timestamp_bound')


def selected_events(rows,report):
    writes=[e for e in rows if e.get('kind')=='write']
    moving=[e for e in writes if e.get('linear_mps') or e.get('angular_rps')]
    if not moving:raise ValueError('No nonzero command events in probe record')
    first,last=moving[0]['mono'],moving[-1]['mono']
    after=[e['mono'] for e in writes if e['mono']>last and not e['linear_mps'] and not e['angular_rps']]
    if not after:raise ValueError('No recorded active zero after last nonzero')
    silence=report.get('silence_test') or {}
    result=dict(first_nonzero_write=first,last_nonzero_write=last,first_active_zero_after=after[0])
    for source,target in (('started_mono','silence_started'),
                          ('stationary_confirmed_mono','candidate_stationarity_confirmed')):
        if source in silence:result[target]=silence[source]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Completed video_silence run directory')
    parser.add_argument('--marks',required=True,help='CSV with marker_id,video_time_before_s,video_time_after_s')
    args=parser.parse_args(argv)
    run=Path(args.run).expanduser().resolve();marks_path=Path(args.marks).expanduser().resolve()
    marker_path=run/'marker_events.jsonl';events_path=run/'probe/events.jsonl';result_path=run/'probe/result.json'
    markers=[json.loads(s) for s in marker_path.read_text().splitlines()]
    with marks_path.open(newline='') as file:marks=list(csv.DictReader(file))
    fit=align(markers,marks)
    events=[json.loads(s) for s in events_path.read_text().splitlines()]
    report=json.loads(result_path.read_text());items=selected_events(events,report)
    slope=fit['host_monotonic_per_video_second'];offset=fit['host_offset_s']
    fit.update(events={name:dict(host_mono_s=value,estimated_video_time_s=(value-offset)/slope,
        within_marker_range=fit['markers'][0]['host_request_s']<=value<=fit['markers'][-1]['host_queued_s'])
        for name,value in items.items()},inputs_sha256={
            'marker_events':sha(marker_path),'marks':sha(marks_path),'probe_events':sha(events_path),'probe_result':sha(result_path)},
        caveats=['Video times must be presentation timestamps of successive frames, not rounded player seconds.',
            'Display refresh/compositor latency, exposure and rolling shutter are uncalibrated.',
            'Fit residual is not an upper bound on the absolute clock alignment error.',
            'Candidate zero feedback is not a verified physical stop or firmware timeout.'])
    output=run/'video_alignment.json'
    with output.open('x') as file:
        json.dump(fit,file,ensure_ascii=False,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps(fit,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
