"""Create an experimental rigidly levelled copy of a recorded ROLL map.

The horizontal-floor assumption is explicit. This operation does not correct
non-rigid drift or verify the alignment of the gravity vector for localization.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from .context import pose_matrix
from .initial_pose import quaternion_rpy
from .manual_edit import _poses, read_xyzi, source_binding, write_xyzi
from .results import file_fingerprint, map_fingerprint, rpy_quaternion
from .runtime import quaternion

ROOT = Path(__file__).resolve().parents[2]


def floor_rotation(ground, support, origin, resolution, *, cell_m=1.0, max_angle_deg=8.0):
    """Fit one median per metre cell and robustly estimate a horizontal floor."""
    ground = np.asarray(ground)
    support = np.asarray(support, dtype=bool)
    origin = np.asarray(origin, dtype=float)
    if (ground.ndim != 2 or support.shape != ground.shape or
            origin.shape != (2,) or not np.isfinite(origin).all() or
            not np.isfinite(resolution) or resolution <= 0 or cell_m <= 0):
        raise ValueError('Invalid ground grid')
    rows, cols = np.nonzero(support & np.isfinite(ground))
    if not len(rows):
        raise ValueError('Insufficient spatial floor support for level estimation')
    xy = np.column_stack((origin[0]+(cols+.5)*resolution, origin[1]+(rows+.5)*resolution))
    cells = np.floor(xy/cell_m).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    samples = []
    for number in range(int(inv.max())+1 if len(inv) else 0):
        mask = inv == number
        samples.append([*np.median(xy[mask], axis=0), float(np.median(ground[rows[mask], cols[mask]]))])
    samples = np.asarray(samples, dtype=float).reshape(-1, 3)
    if len(samples) < 50 or np.min(np.ptp(samples[:, :2], axis=0)) < 5:
        raise ValueError('Insufficient spatial floor support for level estimation')
    matrix = np.column_stack((samples[:, :2], np.ones(len(samples))))
    weight = np.ones(len(samples))
    for _ in range(12):
        sw = np.sqrt(weight)
        plane = np.linalg.lstsq(matrix*sw[:, None], samples[:, 2]*sw, rcond=None)[0]
        residual = samples[:, 2]-matrix@plane
        # Huber loss limits individual discontinuities without assuming every
        # supported cell belongs to the same ground surface.
        weight = np.minimum(1., .15/np.maximum(np.abs(residual), 1e-12))
    normal = np.array([-plane[0], -plane[1], 1.]); normal /= np.linalg.norm(normal)
    angle = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    if angle > max_angle_deg:
        raise ValueError('Estimated ground tilt exceeds the experimental limit')
    cross = np.cross(normal, [0., 0., 1.]); sine = np.linalg.norm(cross)
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    rotation = np.eye(3) if sine < 1e-12 else np.eye(3)+skew+skew@skew*((1-normal[2])/(sine*sine))
    result = dict(assumption='single_approximately_horizontal_floor', fit='huber_per_metre_ground_medians',
                  plane_abc=plane.tolist(), rotation_angle_deg=angle, sample_cells=len(samples),
                  residual_p05_p50_p95_m=np.percentile(residual, [5, 50, 95]).tolist(),
                  rotation_new_from_old=rotation.tolist(), translation_new_from_old=[0., 0., 0.],
                  independent_validation=False)
    return rotation, result


def transform_row(row, rotation):
    if row.get('has_map_pose'):
        pose = pose_matrix(row['map_position'], row['map_quaternion_xyzw'])
        row = dict(row, map_position=(rotation@pose[:3, 3]).tolist(),
                   map_quaternion_xyzw=quaternion(rotation@pose[:3, :3]).tolist())
    return row


def _tracked_floor(source, run, map_path, digest):
    """Compute floor support without an unnecessary full static-ray export."""
    from perception.environment.layers import load_config
    from perception.environment.maps import static_geometry
    from perception.environment.offline_ground import scan_ground
    from perception.environment.poses import PoseTimeline
    from perception.environment.scans import sealed_scan_summary
    from .results import read_pcd
    config = load_config()
    keys, offset = _poses(map_path)
    if offset != 1: raise ValueError('A timestamped ROLL map is required')
    rows = [json.loads(line) for line in (source/'localization.jsonl').open() if line.strip()]
    timeline = PoseTimeline(rows, config['poses'], keys)
    journal = source/'environment/scans.sqlite'; sealed_scan_summary(journal)
    scan_digest = file_fingerprint(journal)
    if scan_digest != run.get('environment_scans_fingerprint'):
        raise ValueError('Source scan journal has changed')
    if run.get('source') == 'bag' and file_fingerprint(run['bag']) != run['playback']['bag_fingerprint']:
        raise ValueError('Source bag does not match the recorded mapping run')
    manifest = yaml.safe_load((map_path/'manifest.yaml').read_text())
    points = read_pcd(map_path/'GlobalMap.pcd', config['static']['max_points'], strict=True)
    candidate = static_geometry(points, keys[:, 1:4], config, map_id='ground_seed',
                                frame=manifest['frame'], stamp=keys[-1, 0])
    grid = scan_ground(candidate, journal, timeline, config, scan_digest,
                       progress=lambda message: print(message, flush=True))
    return grid, dict(method='source_bound_tracked_scans', map_fingerprint=digest,
                      scans_fingerprint=scan_digest, records_fingerprint=file_fingerprint(source/'localization.jsonl'),
                      diagnostics=grid.metadata['ground_scan_diagnostics'])


def create(source, static, output):
    from perception.environment.layers import GridSnapshot
    from perception.environment.scans import sealed_scan_summary
    source = Path(source).resolve()
    static = Path(static).resolve() if static is not None else None
    requested = Path(output).expanduser().absolute()
    if os.path.lexists(requested):
        raise FileExistsError('Refusing to overwrite output')
    output = requested.resolve()
    if not any(parent in output.parents for parent in (ROOT/'outputs', Path('/tmp').resolve())):
        raise ValueError('Output must be beneath project outputs/ or /tmp/')
    if source == output or source in output.parents or (static and (static == output or static in output.parents)):
        raise ValueError('Output cannot be inside a source asset')
    run = json.loads((source/'run.json').read_text())
    if not run.get('success') or run.get('mode') != 'mapping':
        raise ValueError('A successful source mapping run is required')
    old_map = source/'map'; digest = map_fingerprint(old_map)
    keyframe_binding = source_binding(old_map)
    if digest != run['playback']['map_fingerprint']:
        raise ValueError('Source run does not match its map')
    if static is not None:
        package = yaml.safe_load((static/'manifest.yaml').read_text())
        if package.get('source', {}).get('map_fingerprint') != digest:
            raise ValueError('Floor grid and source map do not match')
        grid = GridSnapshot.load(static/'layers.npz')
        if (grid.metadata.get('snapshot_id') != package.get('snapshot_id') or
                grid.metadata.get('frame_id') != package.get('frame_id')):
            raise ValueError('Floor grid does not match its package manifest')
        if grid.metadata.get('ground_estimator') != 'consecutive_scan_tracker_cell_median':
            raise ValueError('This method requires the bound tracked-scan floor')
        ground_source = dict(method='static_package', ground_package=str(static),
            ground_package_manifest_fingerprint=file_fingerprint(static/'manifest.yaml'),
            ground_layers_fingerprint=file_fingerprint(static/'layers.npz'))
    else:
        grid, ground_source = _tracked_floor(source, run, old_map, digest)
    rotation, report = floor_rotation(grid.layers['ground_z'], grid.layers['ground_support'],
                                      grid.metadata['origin_xy'], grid.metadata['resolution'])
    poses, offset = _poses(old_map)
    if offset != 1 or np.any(np.diff(poses[:, 0]) <= 0):
        raise ValueError('A timestamped, ordered ROLL map is required')
    journal = source/'environment/scans.sqlite'
    sealed_scan_summary(journal)
    if file_fingerprint(journal) != run.get('environment_scans_fingerprint'):
        raise ValueError('Source scan journal has changed')
    original_cloud = read_xyzi(old_map/'GlobalMap.pcd')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=output.parent) as temp:
        target = Path(temp)/'candidate'; target.mkdir(); target_map = target/'map'; target_map.mkdir()
        shutil.copytree(old_map/'keyframes', target_map/'keyframes')
        changed = poses.copy()
        for index, row in enumerate(poses):
            matrix = pose_matrix(row[1:4], rpy_quaternion(*row[4:7]))
            changed[index, 1:4] = rotation@matrix[:3, 3]
            changed[index, 4:7] = quaternion_rpy(quaternion(rotation@matrix[:3, :3]))
        np.savetxt(target_map/'keyframes/poses.txt', changed, fmt='%.17g')
        transformed = original_cloud.copy()
        transformed[:, :3] = original_cloud[:, :3].astype(float)@rotation.T
        write_xyzi(target_map/'GlobalMap.pcd', transformed)
        report.update(source_run=str(source), source_run_fingerprint=file_fingerprint(source/'run.json'),
                      source_map_fingerprint=digest, source_keyframes_sha256=keyframe_binding['keyframes_sha256'],
                      ground_source=ground_source,
                      note='Experimental coordinate transform; local keyframe clouds remain in their sensor frames.')
        (target_map/'level_transform.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        manifest = yaml.safe_load((old_map/'manifest.yaml').read_text())
        manifest.update(derived_from_map=digest, derivation='experimental_horizontal_floor_rotation',
                        level_transform_fingerprint=file_fingerprint(target_map/'level_transform.json'),
                        gravity_alignment_verified=False)
        (target_map/'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        (target_map/'COMPLETE').write_text('roll_keyframes_v1\n')
        with (source/'localization.jsonl').open() as old, (target/'localization.jsonl').open('x') as new:
            for line in old:
                if line.strip(): new.write(json.dumps(transform_row(json.loads(line), rotation), allow_nan=False)+'\n')
        (target/'environment').mkdir()
        (target/'environment/scans.sqlite').symlink_to(os.path.relpath(journal.resolve(), output/'environment'))
        run.update(map=str(output/'map'), driver='perception.mapping_localization.level_map',
                   derived_mapping=dict(method='rigid_floor_rotation', source_run=str(source),
                                        source_run_fingerprint=report['source_run_fingerprint'],
                                        gravity_alignment_verified=False))
        run['playback']['map_fingerprint'] = map_fingerprint(target_map)
        (target/'run.json').write_text(json.dumps(run, indent=2, allow_nan=False)+'\n')
        if source_binding(old_map) != keyframe_binding or file_fingerprint(journal) != run['environment_scans_fingerprint']:
            raise ValueError('Source changed during derivation')
        if run.get('source') == 'bag' and file_fingerprint(run['bag']) != run['playback']['bag_fingerprint']:
            raise ValueError('Source bag changed during derivation')
        if static is not None and (file_fingerprint(static/'manifest.yaml') != ground_source['ground_package_manifest_fingerprint'] or
                                   file_fingerprint(static/'layers.npz') != ground_source['ground_layers_fingerprint']):
            raise ValueError('Source floor package changed during derivation')
        if os.path.lexists(output): raise FileExistsError('Output appeared during derivation')
        target.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--static-map', help='Optional existing source-bound tracked-scan static map')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.run, args.static_map, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
