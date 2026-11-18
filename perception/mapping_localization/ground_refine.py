"""Experimental offline vertical corrections from tracked floor observations.

This is a one-dimensional soft constraint on keyframe Z, not a full SLAM
factor graph. The horizontal-floor hypothesis and output are not validated for
navigation. Keyframe-local point clouds are unchanged; the global display map
is rebuilt from their corrected poses.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from .level_map import ROOT
from .manual_edit import _poses, _voxel, _world, read_xyzi, source_binding, write_xyzi
from .results import file_fingerprint, map_fingerprint


def estimate_offsets(times, sample_times, floor_heights, *, smooth_sigma=.03, prior_sigma=.35,
                     observation_sigma=.1, robust_m=.15, train=None):
    """Robust 1-D graph, with adjacent-pose, prior and floor residuals."""
    times = np.asarray(times, float); sample_times = np.asarray(sample_times, float)
    floor_heights = np.asarray(floor_heights, float)
    if (len(times) < 2 or len(sample_times) != len(floor_heights) or not len(sample_times) or
        not np.isfinite(times).all() or not np.isfinite(sample_times).all() or
        not np.isfinite(floor_heights).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid floor observations or keyframe times')
    if not np.isfinite([smooth_sigma, prior_sigma, observation_sigma, robust_m]).all() or min(
            smooth_sigma, prior_sigma, observation_sigma, robust_m) <= 0:
        raise ValueError('Floor solver scales must be positive and finite')
    keys = np.searchsorted(times, sample_times, side='right')-1
    if np.any(keys < 0) or np.any(keys >= len(times)):
        raise ValueError('Floor observation is outside keyframe coverage')
    unique = np.unique(keys)
    # Consecutive scans of the same keyframe are correlated; aggregate first.
    heights = np.array([np.median(floor_heights[keys == k]) for k in unique])
    stamps = np.array([np.median(sample_times[keys == k]) for k in unique])
    use = np.ones(len(unique), bool) if train is None else np.asarray(train(stamps), bool)
    if len(unique) < 50 or use.shape != (len(unique),) or use.sum() < 25:
        raise ValueError('Insufficient keyframe floor observations')
    target = float(np.median(heights[use])); n = len(times)
    weight = np.ones(int(use.sum()))
    for _ in range(10):
        # The normal matrix is tridiagonal: an observation affects only its
        # keyframe, and each smoothness factor joins adjacent keyframes.
        diag = np.full(n, 1./prior_sigma**2); off = np.full(n-1, -1./smooth_sigma**2)
        diag[:-1] += 1./smooth_sigma**2; diag[1:] += 1./smooth_sigma**2
        diag[0] += 1./.05**2  # preserve the original map origin approximately
        rhs = np.zeros(n); kk = unique[use]; hh = heights[use]
        np.add.at(diag, kk, weight/observation_sigma**2)
        np.add.at(rhs, kk, weight*(target-hh)/observation_sigma**2)
        # Thomas solve, without requiring SciPy in the ROS Python environment.
        upper = off.copy(); dd = diag.copy(); bb = rhs.copy()
        for i in range(1, n):
            ratio = off[i-1]/dd[i-1]; dd[i] -= ratio*upper[i-1]; bb[i] -= ratio*bb[i-1]
        delta = np.empty(n); delta[-1] = bb[-1]/dd[-1]
        for i in range(n-2, -1, -1): delta[i] = (bb[i]-upper[i]*delta[i+1])/dd[i]
        residual = hh+delta[kk]-target
        weight = np.minimum(1., robust_m/np.maximum(np.abs(residual), 1e-12))
    return delta, dict(target_height_m=target, keyframes_with_floor=len(unique),
                       selected_floor_keyframes=int(use.sum()),
                       correction_range_m=[float(delta.min()), float(delta.max())],
                       max_adjacent_correction_m=float(np.max(np.abs(np.diff(delta))))), (unique, stamps, heights, use)


def collect_floor(source, keys):
    from perception.environment.fusion import _scan_geometry
    from perception.environment.ground import GroundTracker
    from perception.environment.layers import load_config
    from perception.environment.poses import PoseTimeline
    from perception.environment.scans import read_scans, sealed_scan_summary
    journal = source/'environment/scans.sqlite'; sealed_scan_summary(journal)
    rows = [json.loads(line) for line in (source/'localization.jsonl').open() if line.strip()]
    config = load_config(); timeline = PoseTimeline(rows, config['poses'], keys)
    tracker = GroundTracker(config['ground_tracking']); identity = None; previous = None; last = -np.inf
    samples = []; count = 0
    for record, data in read_scans(journal):
        stamp = record['stamp_ns']/1e9
        if stamp-last < .2-1e-6: continue
        last = stamp; count += 1; item = _scan_geometry(record, data, timeline)
        if item is None:
            tracker = GroundTracker(config['ground_tracking']); identity = previous = None
            continue
        _, pose, _, final, cloud, _ = item
        current = (pose['session_id'], pose['generation'], pose['segment'])
        if current != identity or (previous is not None and stamp-previous > 1.):
            tracker = GroundTracker(config['ground_tracking'])
        identity = current; previous = stamp
        _, info = tracker.fit(cloud, final[:3, 3], config)
        if info.get('status') == 'tracked' and info['height_uncertainty_m'] <= .12:
            samples.append([stamp, final[2, 3]-info['measured_height_m']])
        if count % 500 == 0: print('地面约束：%d 帧已采样，%d 帧可用。' % (count, len(samples)), flush=True)
    if not samples: raise ValueError('No reliable floor observations')
    return np.asarray(samples), dict(sampled_scans=count, tracked_scans=len(samples),
                                     bound_keyframes=timeline.bound_keyframes, interval_s=.2)


def create(source, output):
    source = Path(source).resolve(); requested = Path(output).expanduser().absolute()
    if os.path.lexists(requested): raise FileExistsError('Refusing to overwrite output')
    output = requested.resolve()
    if not any(parent in output.parents for parent in (ROOT/'outputs', Path('/tmp').resolve())):
        raise ValueError('Output must be beneath project outputs/ or /tmp/')
    if source == output or source in output.parents: raise ValueError('Output cannot be inside source')
    run = json.loads((source/'run.json').read_text()); old_map = source/'map'
    if not run.get('success') or run.get('mode') != 'mapping': raise ValueError('A mapping run is required')
    if run.get('derived_mapping', {}).get('method') != 'rigid_floor_rotation':
        raise ValueError('First build a rigidly levelled candidate for this experimental solver')
    digest = map_fingerprint(old_map); binding = source_binding(old_map)
    if digest != run['playback']['map_fingerprint']: raise ValueError('Source map binding mismatch')
    journal = source/'environment/scans.sqlite'
    if file_fingerprint(journal) != run.get('environment_scans_fingerprint'):
        raise ValueError('Source scan journal has changed')
    keys, offset = _poses(old_map)
    if offset != 1: raise ValueError('Timestamped ROLL keyframes required')
    samples, collection = collect_floor(source, keys)
    split = lambda stamps: ((stamps-keys[0, 0])//30).astype(int)%2 == 0
    trial, trial_stats, (ids, stamps, heights, train) = estimate_offsets(keys[:, 0], samples[:, 0], samples[:, 1], train=split)
    if (~train).sum() < 20: raise ValueError('Insufficient held-out time intervals')
    base = np.abs(heights[~train]-trial_stats['target_height_m'])
    corrected = np.abs(heights[~train]+trial[ids[~train]]-trial_stats['target_height_m'])
    if np.percentile(corrected, 95) > np.percentile(base, 95)+.02:
        raise ValueError('Ground correction worsens held-out floor consistency; keep the raw map for review')
    delta, stats, _ = estimate_offsets(keys[:, 0], samples[:, 0], samples[:, 1])
    if np.max(np.abs(delta)) > .5: raise ValueError('Ground correction exceeds the 0.5 m experiment limit')
    report = dict(assumption='single_approximately_horizontal_floor', method='robust_vertical_keyframe_constraints',
        collection=collection, solver=stats, split='alternating_30_second_intervals_not_independent_data',
        heldout_count=int((~train).sum()), heldout_abs_floor_deviation_p50_p95_m=np.percentile(base,[50,95]).tolist(),
        heldout_corrected_abs_floor_deviation_p50_p95_m=np.percentile(corrected,[50,95]).tolist(),
        offsets_m=delta.tolist(), source_run=str(source), source_run_fingerprint=file_fingerprint(source/'run.json'),
        source_binding=binding, scans_fingerprint=file_fingerprint(journal), independent_validation=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=output.parent) as temp:
        target = Path(temp)/'candidate'; target.mkdir(); target_map = target/'map'; target_map.mkdir()
        shutil.copytree(old_map/'keyframes', target_map/'keyframes')
        changed = keys.copy(); changed[:, 3] += delta
        np.savetxt(target_map/'keyframes/poses.txt', changed, fmt='%.17g')
        parts = []; total = 0
        for index, row in enumerate(changed):
            for kind in ('corner', 'surf'):
                points = read_xyzi(old_map/'keyframes'/('%s%d.pcd' % (kind, index)))
                total += len(points)
                if total > 8_000_000: raise ValueError('Map reconstruction exceeds point budget')
                parts.append(np.column_stack((_world(points, row[1:7]), points[:, 3])))
        manifest = yaml.safe_load((old_map/'manifest.yaml').read_text())
        write_xyzi(target_map/'GlobalMap.pcd', _voxel(np.concatenate(parts), float(manifest['visualization_resolution'])))
        (target_map/'ground_refinement.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        manifest.update(derived_from_map=digest, derivation='experimental_ground_z_constraints',
                        ground_refinement_fingerprint=file_fingerprint(target_map/'ground_refinement.json'),
                        gravity_alignment_verified=False)
        (target_map/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        (target_map/'COMPLETE').write_text('roll_keyframes_v1\n')
        with (source/'localization.jsonl').open() as old, (target/'localization.jsonl').open('x') as new:
            for line in old:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get('has_map_pose'):
                    index = int(np.searchsorted(keys[:, 0], row['stamp_s'], side='right')-1)
                    if index >= 0:
                        row['map_position'][2] += float(delta[index])
                        row['map_pose_derivation'] = 'offline_keyframe_ground_correction_not_online_pose'
                new.write(json.dumps(row, allow_nan=False)+'\n')
        (target/'environment').mkdir()
        (target/'environment/scans.sqlite').symlink_to(os.path.relpath(journal.resolve(), output/'environment'))
        run.update(map=str(output/'map'), driver='perception.mapping_localization.ground_refine',
                   derived_mapping=dict(method='experimental_ground_z_constraints', source_run=str(source),
                                        source_run_fingerprint=report['source_run_fingerprint'],
                                        gravity_alignment_verified=False))
        run['playback']['map_fingerprint'] = map_fingerprint(target_map)
        (target/'run.json').write_text(json.dumps(run, indent=2, allow_nan=False)+'\n')
        if source_binding(old_map) != binding or file_fingerprint(journal) != report['scans_fingerprint']:
            raise ValueError('Source changed during derivation')
        if run.get('source') == 'bag' and file_fingerprint(run['bag']) != run['playback']['bag_fingerprint']:
            raise ValueError('Source bag changed during derivation')
        if os.path.lexists(output): raise FileExistsError('Output appeared during derivation')
        target.rename(output)
    return {key:value for key,value in report.items() if key != 'offsets_m'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); print(json.dumps(create(args.run, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
