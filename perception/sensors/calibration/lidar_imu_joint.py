"""Shared, prior-constrained LiDAR/IMU rotation with explicit clock groups.

Validation observations are loaded only after the training solution is frozen.
Translation is never optimized. Original IMU messages are not resampled online.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .lidar_imu import motion_features, orthogonal, parameter_yaml, write_json


def measured(source, tau, indices):
    query = source['time'][indices] + tau
    return np.column_stack([np.interp(query, source['imu_grid'], source['filtered'][:, k]) for k in range(3)])


def solve_rotation(sources, initial_rotation, *, refine_rotation=True):
    """Fit training sources; freeze rotation when comparing session timing.

    Frozen-rotation comparisons keep each historical spatial solution intact
    while estimating its clock offsets and nuisance biases on training only.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    groups = list(dict.fromkeys(s['clock_group'] for s in sources))
    indices = [np.flatnonzero(s['mask'] & (np.arange(len(s['time'])) % 5 == 0)) for s in sources]
    if len(sources) < 2 or any(len(i) < 100 for i in indices):
        raise ValueError('Joint fitting requires at least two sufficiently covered training sources')
    cov = np.mean([np.cov(s['omega'][i].T) for s, i in zip(sources, indices)], axis=0)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues[-1] < 1e-5:
        raise ValueError('Insufficient angular excitation')
    axis_l = vectors[:, -1]
    basis = orthogonal(initial_rotation @ axis_l)
    count = sum(len(i) for i in indices)
    average_count = count / len(sources)
    weights = [np.sqrt(average_count / len(i)) for i in indices]
    rotation_dofs = 2 if refine_rotation else 0
    tau_start = rotation_dofs + 3 * len(sources)
    group_index = {g: j for j, g in enumerate(groups)}

    def rotation(x):
        if not refine_rotation:
            return initial_rotation
        return Rotation.from_rotvec(basis @ x[:2]).as_matrix() @ initial_rotation

    grid = np.arange(-.25, .2501, .005)
    coarse = {}
    for group in groups:
        costs = []
        for tau in grid:
            errors = []
            for source, selected in zip(sources, indices):
                if source['clock_group'] != group:
                    continue
                delta = measured(source, tau, selected) - source['omega'][selected] @ initial_rotation.T
                errors.append(np.mean((delta - delta.mean(0)) ** 2))
            costs.append(float(np.mean(errors)))
        coarse[group] = dict(tau_s=float(grid[np.argmin(costs)]), grid_s=grid.tolist(), cost=costs)
    initial_biases = [np.clip((measured(s, coarse[s['clock_group']]['tau_s'], i)
                               - s['omega'][i] @ initial_rotation.T).mean(0), -.09, .09)
                      for s, i in zip(sources, indices)]
    x0 = np.r_[np.zeros(rotation_dofs), np.concatenate(initial_biases), [coarse[g]['tau_s'] for g in groups]]

    def residual(x):
        r = rotation(x)
        errors = []
        for j, (source, selected, weight) in enumerate(zip(sources, indices, weights)):
            tau = x[tau_start + group_index[source['clock_group']]]
            bias_start = rotation_dofs + 3*j
            prediction = source['omega'][selected] @ r.T + x[bias_start:bias_start+3]
            errors.append(((prediction - measured(source, tau, selected)) / .02 * weight).ravel())
        regularizer = np.r_[np.sqrt(count / 50.) * x[:rotation_dofs] / np.deg2rad(2.),
                            np.sqrt(average_count / 50.) * x[rotation_dofs:tau_start] / .05]
        return np.concatenate(errors + [regularizer])

    lower = np.r_[[-.12] * rotation_dofs, [-.1] * (3 * len(sources)), [-.25] * len(groups)]
    upper = -lower
    result = least_squares(residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=1., max_nfev=120)
    offsets = {g: float(result.x[tau_start+j]) for j, g in enumerate(groups)}
    return dict(rotation=rotation(result.x),
                biases={s['id']: result.x[rotation_dofs+3*j:rotation_dofs+3*j+3] for j, s in enumerate(sources)},
                taus=offsets, basis=basis, axis_l=axis_l, eigenvalues=eigenvalues,
                correction_deg=float(np.degrees(np.linalg.norm(result.x[:rotation_dofs]))),
                rotation_refined=bool(refine_rotation),
                converged=bool(result.success), nfev=int(result.nfev), message=str(result.message),
                bounded=bool(np.linalg.norm(result.x[:rotation_dofs]) < .115 and all(abs(v) < .245 for v in offsets.values())),
                coarse=coarse, training_samples_20hz=count)


def load_source(row, imu_dir):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
        raise ValueError('Source id must be a plain filename stem')
    source = motion_features(imu_dir / (row['id'] + '.npy'), Path(row['poses']))
    binding = source['binding']
    bag = Path(row['bag']).resolve()
    stat = bag.stat()
    if (bag != Path(binding['bag']).resolve()
            or (stat.st_size, stat.st_mtime_ns) != (binding['size'], binding['mtime_ns'])
            or binding['parameters'].get('calibration/lidar_only') is not True):
        raise ValueError('Source differs from its bound independent pure-LiDAR reference')
    start, end = row['range_s']
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= binding['duration_s']:
        raise ValueError('Invalid source time range')
    mask = (source['time'] >= start) & (source['time'] < end)
    if mask.sum() < 500:
        raise ValueError('Insufficient selected source coverage')
    source.update(id=row['id'], clock_group=row['clock_group'], mask=mask)
    return source


def check_time_separation(training, validation):
    for a in training:
        ta = a['origin'] + a['time'][a['mask']]
        for b in validation:
            tb = b['origin'] + b['time'][b['mask']]
            if max(ta[0], tb[0]) <= min(ta[-1], tb[-1]):
                raise ValueError('Selected training and validation acquisition intervals overlap')


def fit_joint(args, *, evaluate=True):
    import yaml

    out = args.output.resolve()
    if (out / 'optimized.json').exists():
        raise FileExistsError(out / 'optimized.json')
    spec = json.loads(args.sources.read_text())
    rows = spec['train'] + spec['validation']
    if not spec['validation'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Require separate, uniquely identified training and validation sources')
    prior_bytes = args.prior.read_bytes()
    prior = json.loads(prior_bytes)
    if (prior.get('length_unit') != 'meter' or prior.get('lidar_frame') != 'rslidar'
            or prior.get('imu_frame') != 'camera_imu_optical_frame'
            or prior.get('convention') != 'p_camera_imu_optical_frame = T_imu_lidar @ p_rslidar'):
        raise ValueError('Spatial prior frames, units or convention do not match the supported IMU/LiDAR input')
    initial = np.asarray(prior['T_imu_lidar'], dtype=float)
    if (initial.shape != (4, 4) or not np.isfinite(initial).all()
            or not np.allclose(initial[3], [0, 0, 0, 1])
            or not np.allclose(initial[:3, :3].T @ initial[:3, :3], np.eye(3), atol=1e-6)
            or np.linalg.det(initial[:3, :3]) < .999999):
        raise ValueError('Spatial prior is not a finite rigid transform')
    baseline_bytes = args.baseline.read_bytes()
    baseline = yaml.safe_load(baseline_bytes)
    training = [load_source(row, args.imu_dir) for row in spec['train']]
    solved = solve_rotation(training, initial[:3, :3])
    if not solved['converged'] or not solved['bounded']:
        raise RuntimeError('Joint optimization did not converge inside its physical bounds')
    deployment_group = spec['deployment_clock_group']
    if deployment_group not in solved['taus']:
        raise ValueError('Deployment clock group has no training observations')
    transform = initial.copy()
    transform[:3, :3] = solved['rotation']
    record = dict(T_imu_lidar=transform.tolist(), initial_T_imu_lidar=initial.tolist(),
                  convention='p_camera_imu_optical_frame = T_imu_lidar @ p_rslidar', length_unit='meter',
                  lidar_frame='rslidar', imu_frame='camera_imu_optical_frame',
                  method='camera_prior_joint_planar_rotation_and_grouped_time_refinement',
                  imu_time_offset=-solved['taus'][deployment_group],
                  imu_time_offset_by_clock_group={g: -tau for g, tau in solved['taus'].items()},
                  time_convention='gyro_I(t_lidar + tau) = R_IL omega_L(t_lidar) + b_g; corrected IMU time = header - tau',
                  deployment_clock_group=deployment_group,
                  gyro_bias_by_training_bag={k: v.tolist() for k, v in solved['biases'].items()},
                  rotation_correction_deg=solved['correction_deg'], rotation_refinement_accepted=True,
                  translation_refinement_accepted=False, translation_policy='retain_camera_chained_prior',
                  fixed_rotation_axis_lidar=solved['axis_l'].tolist(),
                  fixed_rotation_axis_imu=(initial[:3, :3] @ solved['axis_l']).tolist(),
                  angular_covariance_eigenvalues=solved['eigenvalues'].tolist(),
                  solver=dict(nfev=solved['nfev'], message=solved['message'], loss='soft_l1',
                              gyro_scale_rad_s=.02, rotation_prior_deg=2., bias_prior_rad_s=.05,
                              equal_bag_weight=True, samples_20hz=solved['training_samples_20hz']),
                  prior=str(args.prior.resolve()), prior_sha256=hashlib.sha256(prior_bytes).hexdigest(),
                  sources=spec, source_spec_sha256=hashlib.sha256(args.sources.read_bytes()).hexdigest(),
                  training_bindings=[s['binding'] for s in training],
                  baseline_yaml_snapshot=baseline, baseline_yaml_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
                  selection_split='train_only', status='frozen_training_solution_before_validation',
                  limitations=['Planar observable subspace; rotation about dominant turn axis retains camera prior',
                               'Translation remains the new camera-chain prior, not independently estimated',
                               'Existing validation is a development set, not an unseen blind test',
                               'Clock offsets are session-scoped; per-bag gyro biases are fitting nuisance variables'])
    write_json(out / 'optimized.json', record)
    (out / 'optimized.yaml').write_text(parameter_yaml(record, out / 'optimized.json'))
    if not evaluate:
        print('Joint training solution frozen; validation not loaded', flush=True)
        return record
    print('Joint training solution frozen; now loading validation observations', flush=True)

    validation = [load_source(row, args.imu_dir) for row in spec['validation']]
    check_time_separation(training, validation)
    if any(s['clock_group'] not in solved['taus'] for s in validation):
        raise ValueError('Validation clock group is unseen; do not estimate its offset from validation')
    baseline_r = np.asarray(baseline['mapping']['extrinsic_R']).reshape(3, 3)
    baseline_tau = -float(baseline['common']['imu_time_offset'])
    models = {
        'camera_prior_zero_time': (initial[:3, :3], {g: 0. for g in solved['taus']}),
        'single_bag_default': (baseline_r, {g: baseline_tau for g in solved['taus']}),
        'joint': (solved['rotation'], solved['taus']),
    }
    biases = {}
    for label, (rotation, offsets) in models.items():
        biases[label] = {}
        for source in training:
            selected = np.flatnonzero(source['mask'] & (np.arange(len(source['time'])) % 5 == 0))
            biases[label][source['id']] = (
                solved['biases'][source['id']] if label == 'joint' else
                (measured(source, offsets[source['clock_group']], selected)
                 - source['omega'][selected] @ rotation.T).mean(0))
    scores, arrays = [], []
    for source in training + validation:
        is_training = any(source is item for item in training)
        selected = np.flatnonzero(source['mask'])
        row = dict(id=source['id'], split='train' if is_training else 'validation',
                   clock_group=source['clock_group'], samples=len(selected),
                   relative_s=source['time'][selected[[0, -1]]].tolist(),
                   absolute_s=(source['time'][selected[[0, -1]]] + source['origin']).tolist(), metrics={})
        values = dict(relative_time_s=source['time'][selected],
                      absolute_time_s=source['time'][selected] + source['origin'],
                      omega_lidar=source['omega'][selected],
                      source_id=np.full(len(selected), source['id']), training=np.full(len(selected), is_training))
        for label, (rotation, offsets) in models.items():
            if is_training:
                bias = biases[label][source['id']]
            else:
                bias = np.mean([biases[label][s['id']] for s in training
                                if s['clock_group'] == source['clock_group']], axis=0)
            actual = measured(source, offsets[source['clock_group']], selected)
            prediction = source['omega'][selected] @ rotation.T + bias
            error = prediction - actual
            row['metrics'][label] = dict(rmse_rad_s=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                                         axis_rmse_rad_s=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                                         gyro_bias_from_training=bias.tolist())
            values['gyro_' + label] = actual
            values['prediction_' + label] = prediction
        scores.append(row)
        arrays.append(values)
    packed = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    np.savez_compressed(out / 'evaluation_features.npz', **packed)
    summary = {split: {label: float(np.sqrt(np.mean([s['metrics'][label]['rmse_rad_s']**2
                                                    for s in scores if s['split'] == split])))
                       for label in models} for split in ('train', 'validation')}
    evaluation = dict(frozen_solution_sha256=hashlib.sha256((out / 'optimized.json').read_bytes()).hexdigest(),
                      aggregation='Equal bag mean squared vector error, then square root; per-bag scores use 100Hz filtered samples',
                      bias_policy='All model biases fitted only on training; validation uses equal mean of training biases in its clock group',
                      summary=summary, samples=scores, validation_used_for_optimization=False)
    write_json(out / 'evaluation.json', evaluation)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for values, row in zip(arrays, scores):
        if row['split'] != 'validation':
            continue
        for k, axis in enumerate(axes):
            axis.plot(values['relative_time_s'], values['gyro_joint'][:, k], label='IMU with joint clock offset', lw=.8)
            axis.plot(values['relative_time_s'], values['prediction_joint'][:, k], label='LiDAR with joint rotation', lw=.8)
            axis.set_ylabel('gyro ' + 'xyz'[k] + ' (rad/s)')
            axis.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel('validation bag relative time (s); no validation bias or time fitting')
    figure.tight_layout()
    figure.savefig(out / 'validation_alignment.png', dpi=130)
    plt.close(figure)
    print(json.dumps(dict(rotation_correction_deg=record['rotation_correction_deg'],
                          imu_time_offset_by_clock_group=record['imu_time_offset_by_clock_group'],
                          summary=summary), indent=2), flush=True)
