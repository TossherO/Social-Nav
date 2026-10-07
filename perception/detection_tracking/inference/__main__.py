"""Run raw Wheelchair/manifest inference or fingerprint-bound detection replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .config import ROOT, load_config, project_path
from .dataset import WheelchairSource, ManifestSource, CachedDetector
from .detector import SimpleDet3dDetector
from .pipeline import PerceptionPipeline, to_snapshot


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene', help='Relative category/scene below wheelchair_raw')
    source.add_argument('--manifest', type=Path, help='Explicit JSONL frames, e.g. converted CODa/bag exports')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--local-config', type=Path, help='Only this module resource config; never API credentials')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--timestamps', type=Path, help='JSON mapping numeric frame index to real timestamp')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--extension-root', type=Path)
    parser.add_argument('--detections-jsonl', type=Path, help='Replay saved observations.jsonl without Torch/GPU')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Read requested inputs/poses only; no GPU or output writes')
    args = parser.parse_args(argv)
    if args.start < 0 or args.limit < 1:
        parser.error('--start >= 0 and --limit >= 1 are required')
    if not args.dry_run and args.out is None:
        parser.error('Choose a fresh --out directory')
    return args


def safe_output(path, config):
    path = Path(path).absolute()
    resolved = path.resolve()
    aliases = ROOT/'data/datasets'
    protected = [ROOT/'data']
    if aliases.is_dir():
        protected += [p.resolve() for p in aliases.iterdir() if p.is_dir()]
    if config['detector'].get('checkpoint'):
        protected.append(project_path(config['detector']['checkpoint']).resolve().parent.parent)
    if any(resolved.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Output cannot overwrite a dataset/shared resource or training run')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Output already exists: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv=None):
    args = arguments(argv)
    config = load_config(args.config, args.local_config)
    for key in ('checkpoint', 'extension_root'):
        value = getattr(args, key)
        if value is not None:
            config['detector'][key] = str(value.absolute())
    source = (ManifestSource(args.manifest, args.start, args.limit) if args.manifest else
              WheelchairSource(args.data_root or project_path(config['dataset']['root']), args.scene,
                               config['dataset']['nominal_rate_hz'], args.timestamps, args.start, args.limit))
    if args.dry_run:
        rows = []
        for frame in source:
            rows.append({'observation_id': frame.observation_id, 'stamp_s': frame.stamp_s,
                         'points': len(frame.points), 'pose_health': frame.pose_health,
                         'pose_source': frame.pose_source, 'rgb_present': frame.rgb_asset_uri is not None})
        print(json.dumps({'source': source.metadata, 'frames': rows, 'model_loaded': False,
                          'raw_odometry_used': False, 'outputs_written': False}, ensure_ascii=False, indent=2))
        return 0
    # Reject protected/existing output before loading GPU resources.
    out = safe_output(args.out, config)
    startup = time.perf_counter()
    detector = CachedDetector(args.detections_jsonl) if args.detections_jsonl else SimpleDet3dDetector(config['detector'])
    startup_s = time.perf_counter()-startup
    pipeline = PerceptionPipeline(detector, config['tracking'])
    settings = {'config': config, 'source': source.metadata, 'detector': detector.metadata,
                'startup_s': startup_s, 'session_id': pipeline.session_id,
                'checkpoint_loaded': not bool(args.detections_jsonl), 'robot_commands_sent': 0}
    (out/'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = {'valid': 0, 'degraded': 0, 'invalid': 0}
    times = []
    with (out/'observations.jsonl').open('x', encoding='utf-8') as stream:
        for frame in source:
            try:
                result = pipeline.step(frame)
                to_snapshot(result)  # Exercise the actual shared perception adapter, not the VLM wire format.
            except Exception as error:
                result = pipeline.failure(frame, f'{type(error).__name__}: {error}')
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            counts[result['health']] += 1
            if result['timing_s']:
                times.append(result['timing_s']['pipeline'])
            print(f'{frame.observation_id} {result["health"]} detections={len(result["detections"])} '
                  f'tracks={len(result["tracks"])}')
            if result['health'] == 'invalid':
                break  # No hidden retry/fallback; preserve the failed observation and stop.
    summary = {'frames': sum(counts.values()), **counts, 'pipeline_mean_s': sum(times)/len(times) if times else None,
               'pipeline_max_s': max(times) if times else None, 'startup_s': startup_s,
               'semantic_accuracy_evaluated': False, 'robot_commands_sent': 0}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if counts['invalid'] else (2 if counts['degraded'] else 0)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
