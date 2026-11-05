"""Retrack immutable saved detections/poses without reading scans or loading a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .detector import file_sha256
from .geometry import validate_boxes, validate_transform
from .pipeline import Frame, PerceptionPipeline, to_snapshot


class RecordedFrame(Frame):
    """Fingerprint is inherited from a bound source record, not recomputed from points."""
    def __init__(self, row, world_frame):
        fingerprint = row['input_fingerprint']
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('A saved input fingerprint is required for retracking')
        if row['health'] not in ('valid', 'degraded', 'invalid'):
            raise ValueError('Unknown saved observation health')
        localized = row['health'] == 'valid'
        transform = row['world_from_lidar']
        if localized and (transform is None or row['pose_health'] != 'valid'):
            raise ValueError('Saved valid observation has no valid localization')
        if localized and row['pose_source'] not in ('dataset_pose', 'localization', 'stationary_assumption'):
            raise ValueError('Saved observation does not have a trusted localization source')
        if transform is not None:
            validate_transform(transform)
        binding = row.get('localization_binding') or {}
        if localized and binding.get('frame_id', row['frame_id']) != row['frame_id']:
            raise ValueError('Saved localization binding and coordinate frame disagree')
        self.source_record = row
        super().__init__(row['observation_id'], row['sequence_id'], row['stamp_s'], None,
            sensor_frame=row['detection_frame_id'], clock_domain=row['clock_domain'],
            timestamp_source=row['timestamp_source'], world_from_lidar=transform if localized else None,
            world_frame=world_frame, pose_source=row['pose_source'],
            pose_health=row['pose_health'] if localized else 'invalid', pose_stamp_s=row.get('pose_stamp_s'),
            rgb_asset_uri=row.get('rgb_asset_uri'), rgb_stamp_s=row.get('rgb_stamp_s'),
            pose_identity=(row['session_id'], row['generation'], binding.get('session_id'), binding.get('generation')))

    def input_fingerprint(self):
        return self.source_record['input_fingerprint']


class RecordedDetector:
    def predict(self, frame):
        if not isinstance(frame, RecordedFrame):
            raise ValueError('RecordedDetector only accepts explicitly saved observations')
        self.last_input_context = frame.source_record.get('detection_input')
        return validate_boxes(frame.source_record['detections'])


def retrack_rows(rows, tracking, *, session_id=None):
    pipeline = PerceptionPipeline(RecordedDetector(), tracking, session_id)
    for row in rows:
        binding = row.get('localization_binding') or {}
        world = (row['frame_id'] if row['health'] == 'valid' else
                 binding.get('frame_id') or pipeline.world_frame or 'localization_unavailable')
        frame = RecordedFrame(row, world)
        if row['health'] == 'invalid':
            pipeline._begin(frame)
            result = pipeline.failure(frame, row['reason'])
        else:
            result = pipeline.step(frame)
        # Original geometry/scan identity remain immutable; output gets new track identities.
        result['detections'] = row['detections']
        result['detection_input'] = row.get('detection_input')
        result['localization_binding'] = row.get('localization_binding')
        if 'pose_age_s' in row:
            result['pose_age_s'] = row['pose_age_s']
        result['detection_source'] = 'recorded_detections'
        result['tracking_policy'] = pipeline.tracker.association.metadata
        result['retracking_source'] = dict(session_id=row['session_id'], generation=row['generation'])
        to_snapshot(result)
        yield result


def read_people(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def replay(source, output, config):
    from .__main__ import safe_output
    source = Path(source).resolve(strict=True)
    def signature():
        stat = source.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    before = signature()
    fingerprint = file_sha256(source)
    out = safe_output(output, config)
    source_metadata = {}
    for name in ('run.json', 'settings.json', 'manifest.json'):
        path = source.parent/name
        if path.is_file():
            source_metadata[name] = dict(path=str(path), sha256=file_sha256(path))
    manifest = dict(source=dict(path=str(source), sha256=fingerprint, metadata=source_metadata),
        config=config, input_scope='recorded_detections_and_recorded_causal_localization',
        status='running', scan_content_revalidated=False, model_loaded=False, robot_commands_sent=0)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    times, health, identities = [], {}, set()
    with (out/'people.jsonl').open('x', encoding='utf-8') as stream:
        for row in retrack_rows(read_people(source), config['tracking']):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            health[row['health']] = health.get(row['health'], 0)+1
            identities.update(t['track_id'] for t in row['tracks'])
            if row['health'] == 'valid':
                times.append(row['timing_s']['tracking']*1000)
    if signature() != before:
        manifest.update(status='failed', reason='source_changed_during_retracking')
        (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        raise ValueError('Source changed during retracking; use a completed immutable result')
    summary = dict(frames=sum(health.values()), health=health, unique_tracks=len(identities),
        tracking_ms=dict(median=float(np.median(times)), p95=float(np.percentile(times, 95))) if times else None,
        model_loaded=False, identity_accuracy_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    manifest['status'] = 'complete'
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Saved people.jsonl or observations.jsonl')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--local-config', type=Path)
    parser.add_argument('--association-backend', choices=('cpu', 'cuda'), help='Explicit execution override; metric/thresholds still come from config')
    parser.add_argument('--association-device', help='Visible CUDA device, e.g. cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, args.local_config)
    for key in ('association_backend', 'association_device'):
        value = getattr(args, key)
        if value is not None:
            config['tracking'][key] = value
    print(json.dumps(replay(args.source, args.out, config), indent=2))


if __name__ == '__main__':
    main()
