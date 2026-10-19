"""Compose camera-based spatial priors and refine planar LiDAR/IMU calibration.

Use system ROS Python for compose/extract, and a SciPy Python for fit. Motion
comes from a separately recorded, forced pure-LiDAR LI-Init FAST-LO reference.
The planar mode fixes rotation about the main turn axis and translation along
that axis to the camera prior; it does not claim unconstrained 6-DOF calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def parameter_yaml(record, artifact):
    matrix = np.asarray(record['T_imu_lidar'])
    rotation = json.dumps(matrix[:3, :3].reshape(-1).tolist())
    translation = json.dumps(matrix[:3, 3].tolist())
    policy = 'data_refined_planar_components' if record['translation_refinement_accepted'] else 'retain_camera_chained_prior'
    return ('# p_IMU = R_IL p_LiDAR + t_IL; meters; corrected IMU time = header + offset.\n'
            'common:\n  time_sync_en: false\n  imu_time_offset: ' + repr(record['imu_time_offset']) + '\n'
            'mapping:\n  extrinsic_est_en: false\n  extrinsic_R: ' + rotation + '\n  extrinsic_T: ' + translation + '\n'
            'calibration:\n  lidar_frame: ' + record['lidar_frame'] + '\n  imu_frame: ' + record['imu_frame'] + '\n'
            '  source_artifact: ' + json.dumps(str(artifact)) + '\n'
            '  method: camera_chained_prior_with_planar_rotation_and_time_refinement\n'
            '  translation_policy: ' + policy + '\n  validation_scope: candidate_pending_replay\n')


def compose(args):
    import rosbag
    from tf.transformations import quaternion_matrix
    from perception.detection_tracking.inference.geometry import validate_transform

    if args.tf_bag.resolve() != args.imu_bag.resolve() and not args.same_device:
        raise ValueError('Cross-bag internal TF reuse requires --same-device')
    import yaml
    candidate = yaml.safe_load(args.camera.read_text())
    with rosbag.Bag(str(args.imu_bag)) as bag:
        imu = next(bag.read_messages(topics=['/camera/imu'])).message
    with rosbag.Bag(str(args.tf_bag)) as bag:
        transforms = next(bag.read_messages(topics=['/tf_static'])).message.transforms
        info = next(bag.read_messages(topics=['/camera/color/camera_info'])).message
    graph = {}
    for item in transforms:
        q, p = item.transform.rotation, item.transform.translation
        matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[:3, 3] = [p.x, p.y, p.z]
        parent, child = item.header.frame_id.lstrip('/'), item.child_frame_id.lstrip('/')
        graph.setdefault(child, []).append((parent, matrix))
        graph.setdefault(parent, []).append((child, np.linalg.inv(matrix)))
    source, target = info.header.frame_id.lstrip('/'), imu.header.frame_id.lstrip('/')
    queue, seen = [(source, np.eye(4), [source])], set()
    while queue:
        frame, transform, chain = queue.pop(0)
        if frame == target:
            break
        if frame in seen:
            continue
        seen.add(frame)
        queue.extend((other, matrix @ transform, chain + [other])
                     for other, matrix in graph.get(frame, []) if other not in seen)
    else:
        raise ValueError('Recorded TF does not connect the actual RGB and IMU frames')
    if 'T_camera_lidar' in candidate:
        camera_lidar = candidate['T_camera_lidar']
    else:
        camera_record = candidate['lidar_to_camera_optical']
        if (camera_record['source_frame'].lstrip('/') != 'rslidar'
                or camera_record['target_frame'].lstrip('/') != source
                or camera_record['translation_unit'] != 'meter'
                or camera_record['convention'] != 'column_vector_p_target_equals_T_times_p_source'):
            raise ValueError('Camera calibration frames, units or transform convention do not match the bag')
        camera_lidar = camera_record['matrix']
    initial = validate_transform(transform) @ validate_transform(camera_lidar)
    write_json(args.output, dict(
        T_imu_lidar=initial.tolist(), T_imu_camera=transform.tolist(),
        camera_frame=source, imu_frame=target, lidar_frame='rslidar', internal_tf_chain=chain,
        convention='p_camera_imu_optical_frame = T_imu_lidar @ p_rslidar', length_unit='meter',
        camera_candidate=str(args.camera.resolve()), camera_candidate_sha256=hashlib.sha256(args.camera.read_bytes()).hexdigest(),
        internal_tf_bag=str(args.tf_bag.resolve()), imu_bag=str(args.imu_bag.resolve()),
        hardware_binding='Same-device internal TF reuse explicitly asserted by caller' if args.same_device else 'Internal TF from the IMU source bag',
        status='camera_chained_spatial_prior; temporal offset not calibrated by this composition'))
    print('camera-chained LiDAR->IMU prior:\n', initial, flush=True)


def extract(args):
    import rosbag
    rows = []
    with rosbag.Bag(str(args.bag)) as bag:
        for _, msg, receipt in bag.read_messages(topics=['/camera/imu']):
            w, a = msg.angular_velocity, msg.linear_acceleration
            rows.append([msg.header.stamp.to_sec(), w.x, w.y, w.z, a.x, a.y, a.z, receipt.to_sec()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(rows))
    print('extracted raw IMU samples', len(rows), flush=True)


def orthogonal(axis):
    _, _, vh = np.linalg.svd(np.asarray(axis).reshape(1, 3), full_matrices=True)
    return vh[1:].T


def skew(vector):
    a = np.asarray(vector)
    result = np.zeros(a.shape[:-1] + (3, 3))
    result[..., 0, 1] = -a[..., 2]; result[..., 0, 2] = a[..., 1]
    result[..., 1, 0] = a[..., 2]; result[..., 1, 2] = -a[..., 0]
    result[..., 2, 0] = -a[..., 1]; result[..., 2, 1] = a[..., 0]
    return result


def fit(args):
    from scipy.optimize import least_squares
    from scipy.signal import butter, sosfiltfilt, savgol_filter
    from scipy.spatial.transform import Rotation, Slerp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    prior = json.loads(args.prior.read_text()); initial = np.asarray(prior['T_imu_lidar'])
    r0, t0 = initial[:3, :3], initial[:3, 3]
    raw = np.load(args.imu, allow_pickle=False)
    poses = np.genfromtxt(args.poses, delimiter=',', names=True)
    binding = json.loads(args.poses.with_suffix('.json').read_text())
    origin = binding['record_start_s']
    if (len(raw) != binding['input_messages']['imu']
            or abs(raw[0, 0] - origin) > .3
            or abs(raw[-1, 0] - origin - binding['duration_s']) > .3):
        raise ValueError('IMU observations do not match the pure-LiDAR source bag')
    ti = raw[:, 0] - origin; tp = poses['stamp_s'] - origin
    good = np.isfinite(ti) & np.isfinite(raw[:, 1:7]).all(axis=1)
    raw, ti = raw[good], ti[good]
    goodp = (poses['features'] >= 50) & np.isfinite(tp)
    poses, tp = poses[goodp], tp[goodp]
    if np.any(np.diff(ti) <= 0) or np.any(np.diff(tp) <= 0):
        raise ValueError('Nonmonotonic source observation times')
    if len(ti) < 1000 or len(tp) < 100:
        raise ValueError('Insufficient source observations')
    q = np.column_stack([poses[k] for k in ('qx', 'qy', 'qz', 'qw')])
    rotations = Rotation.from_quat(q)
    mid = (tp[1:] + tp[:-1]) / 2
    angular = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / np.diff(tp)[:, None]
    time = np.arange(max(tp[0], ti[0]) + 2., min(tp[-1], ti[-1]) - 2., .01)
    if len(time) < 1000:
        raise ValueError('Insufficient shared time coverage')
    # Filtering creates fitting features only; original observations remain unchanged.
    imu_grid = np.arange(ti[0], ti[-1], .001)
    signals = np.column_stack([np.interp(imu_grid, ti, raw[:, k]) for k in range(1, 7)])
    filtered = sosfiltfilt(butter(4, 3., fs=1000., output='sos'), signals, axis=0)
    omega = np.column_stack([np.interp(time, mid, angular[:, k]) for k in range(3)])
    omega = sosfiltfilt(butter(4, 3., fs=100., output='sos'), omega, axis=0)
    train = time < args.validation_start - 2.; validation = time >= args.validation_start
    if train.sum() < 500 or validation.sum() < 500:
        raise ValueError('Insufficient training/held-out coverage')
    eig, vectors = np.linalg.eigh(np.cov(omega[train].T)); axis_l = vectors[:, -1]
    if np.dot(r0 @ axis_l, raw[:, 4:7].mean(axis=0)) < 0:
        axis_l = -axis_l
    axis_i = r0 @ axis_l; rotation_basis = orthogonal(axis_i)
    sample = np.arange(0, len(time), 5); tr = sample[train[sample]]

    def measured(tau, columns=slice(0, 3)):
        return np.column_stack([np.interp(time + tau, imu_grid, filtered[:, k]) for k in range(6)])[:, columns]

    def rotation_model(x):
        return Rotation.from_rotvec(rotation_basis @ x[:2]).as_matrix() @ r0

    taus = np.arange(-.25, .2501, .005)
    coarse = []
    for tau in taus:
        delta = measured(tau)[tr] - omega[tr] @ r0.T
        coarse.append(float(np.mean((delta - delta.mean(0)) ** 2)))
    tau0 = float(taus[np.argmin(coarse)])
    bias0 = np.clip((measured(tau0)[tr] - omega[tr] @ r0.T).mean(0), -.09, .09)

    def rotation_residual(x):
        error = (omega[tr] @ rotation_model(x).T + x[2:5] - measured(x[5])[tr]) / .02
        regularizer = np.sqrt(len(tr) / 50.) * np.r_[x[:2] / np.deg2rad(2.), x[2:5] / .05]
        return np.r_[error.ravel(), regularizer]

    fit_r = least_squares(rotation_residual, np.r_[0., 0., bias0, tau0],
                          bounds=(np.r_[[-.12] * 2, [-.1] * 3, -.25], np.r_[[.12] * 2, [.1] * 3, .25]),
                          loss='soft_l1', f_scale=1., max_nfev=120)
    fitted_r = rotation_model(fit_r.x); tau = float(fit_r.x[5]); bias = fit_r.x[2:5]

    def gyro_error(rotation, offset, bg=None):
        actual = measured(offset)
        if bg is None:
            bg = (actual[tr] - omega[tr] @ rotation.T).mean(0)
        return omega @ rotation.T + bg - actual

    before_gyro = gyro_error(r0, 0.)
    timed_gyro = gyro_error(r0, tau)
    after_gyro = gyro_error(fitted_r, tau, bias)
    rmse = lambda error, mask: float(np.sqrt(np.mean(np.sum(error[mask] ** 2, axis=1))))
    rotation_accepted = (rmse(after_gyro, validation) <= 1.02 * rmse(timed_gyro, validation)
                         and np.linalg.norm(fit_r.x[:2]) < .115)
    r = fitted_r if rotation_accepted else r0
    if not rotation_accepted:
        bias = (measured(tau)[tr] - omega[tr] @ r.T).mean(0)
    angular_jacobian = skew(omega[tr]).reshape(-1, 3)
    rotation_singular = np.linalg.svd(angular_jacobian, compute_uv=False)

    world_r = Slerp(tp, rotations)(time).as_matrix()
    p = np.column_stack([np.interp(time, tp, poses[k]) for k in ('x', 'y', 'z')])
    acceleration_world = savgol_filter(p, 101, 3, deriv=2, delta=.01, axis=0)
    wa = sosfiltfilt(butter(4, 1.5, fs=100., output='sos'), omega, axis=0)
    alpha = np.gradient(wa, .01, axis=0)
    lever = skew(wa) @ skew(wa) + skew(alpha)
    target_acc = measured(tau, slice(3, 6))
    target_acc = sosfiltfilt(butter(4, 1.5, fs=100., output='sos'), target_acc, axis=0)
    world_up = np.median(np.einsum('nij,j->ni', world_r[train], axis_l), axis=0)
    world_up /= np.linalg.norm(world_up); gbasis = orthogonal(world_up)
    translation_basis = orthogonal(axis_l); lever0 = -r.T @ t0

    def accel_model(x, move):
        g = Rotation.from_rotvec(gbasis @ x[:2]).apply(-9.81 * world_up)
        arm = lever0 + (translation_basis @ x[5:7] if move else 0)
        local_a = np.einsum('nji,nj->ni', world_r, acceleration_world - g)
        return (local_a + np.einsum('nij,j->ni', lever, arm)) @ r.T + x[2:5]

    def accel_residual(x, move):
        e = (accel_model(x, move)[tr] - target_acc[tr]) / .2
        prior_r = np.r_[x[:2] / np.deg2rad(2.), x[2:5] / 1.5]
        if move:
            prior_r = np.r_[prior_r, x[5:7] / .03]
        return np.r_[e.ravel(), np.sqrt(len(tr) / 50.) * prior_r]

    ba = np.clip((target_acc[tr] - accel_model(np.zeros(5), False)[tr]).mean(0), -1.4, 1.4)
    nuisance = least_squares(lambda x: accel_residual(x, False), np.r_[0., 0., ba],
                             bounds=(np.r_[[-.087] * 2, [-1.5] * 3], np.r_[[.087] * 2, [1.5] * 3]),
                             loss='soft_l1', max_nfev=100)
    fit_t = least_squares(lambda x: accel_residual(x, True), np.r_[nuisance.x, 0., 0.],
                          bounds=(np.r_[[-.087] * 2, [-1.5] * 3, [-.1] * 2], np.r_[[.087] * 2, [1.5] * 3, [.1] * 2]),
                          loss='soft_l1', max_nfev=100)
    before_acc = accel_model(nuisance.x, False) - target_acc
    after_acc = accel_model(fit_t.x, True) - target_acc
    translation_accepted = (rmse(after_acc, validation) < .98 * rmse(before_acc, validation)
                            and np.linalg.norm(fit_t.x[5:7]) < .08)
    arm = lever0 + translation_basis @ fit_t.x[5:7]
    optimized = np.eye(4); optimized[:3, :3] = r
    optimized[:3, 3] = -r @ arm if translation_accepted else t0
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    metrics = {
        'gyro_camera_prior_zero_time': before_gyro,
        'gyro_camera_prior_fitted_time': timed_gyro,
        'gyro_rotation_time_refinement': after_gyro,
        'accel_camera_translation_fitted_nuisance': before_acc,
        'accel_translation_refinement': after_acc,
    }
    if args.baseline_run is not None:
        baseline = json.loads(args.baseline_run.read_text())['algorithm_parameters']
        old_rotation = np.asarray(baseline['mapping']['extrinsic_R']).reshape(3, 3)
        old_offset = baseline['common'].get('imu_time_offset', 0.)
        metrics['gyro_previous_default'] = gyro_error(old_rotation, -old_offset)
    record = dict(
        T_imu_lidar=optimized.tolist(), initial_T_imu_lidar=initial.tolist(),
        convention=prior['convention'], length_unit='meter', imu_frame=prior['imu_frame'], lidar_frame='rslidar',
        imu_header_minus_lidar_time_s=tau, imu_time_offset=-tau,
        time_convention='gyro_I(t_lidar + tau) = R_IL omega_L(t_lidar) + b_g; t_imu_used = t_imu_header - tau',
        gyro_bias_rad_s=bias.tolist(), rotation_refinement_accepted=bool(rotation_accepted),
        translation_refinement_accepted=bool(translation_accepted),
        rotation_correction_deg=float(np.degrees(np.linalg.norm(fit_r.x[:2]))),
        attempted_translation_delta_m=(-r @ arm - t0).tolist(),
        fixed_directions=dict(lidar_main_turn_axis=axis_l.tolist(), imu_main_turn_axis=axis_i.tolist(),
                              rotation='rotation about the main turn axis held to camera prior',
                              translation='lever-arm component along main turn axis held to camera prior'),
        rotation_data_singular_values=rotation_singular.tolist(),
        translation_data_singular_values=np.linalg.svd(lever[tr].reshape(-1, 3), compute_uv=False).tolist(),
        lidar_angular_covariance_eigenvalues=eig.tolist(),
        training_relative_s=[float(time[train][0]), float(time[train][-1])],
        validation_relative_s=[float(time[validation][0]), float(time[validation][-1])],
        metrics={k:dict(train_rmse=rmse(v, train), validation_rmse=rmse(v, validation),
                        unit='rad/s' if k.startswith('gyro') else 'm/s^2') for k, v in metrics.items()},
        raw_imu_samples=len(raw), pose_count=len(poses), prior=str(args.prior.resolve()),
        imu_input=dict(bag=binding['bag'], topic='/camera/imu', frame=prior['imu_frame'],
                       stamp_range_s=[float(raw[0, 0]), float(raw[-1, 0])],
                       source_size=binding['size'], source_mtime_ns=binding['mtime_ns']),
        pose_source=str(args.poses.resolve()), pose_binding=binding,
        filters='Fitting features only: zero-phase gyro 3Hz; acceleration 1.5Hz; position derivative symmetric 1.01s polynomial; raw bag unchanged',
        optimizer='SciPy robust least_squares, soft_l1; planar observable subspaces with camera priors; no free height or gyro-axis yaw',
        acceptance='Rotation held-out consistency; translation adopted only for >=2% held-out acceleration RMSE improvement and bounded displacement',
        status='optimized_candidate_pending_pointcloud_replay',
        limitations=['No independent geometric ground truth; validation is a temporal holdout in the same IMU bag',
                     'Acceleration bias and gravity are nuisance variables, not a standalone IMU intrinsic calibration',
                     'No time drift, per-stream acceleration delay, or full 6-DOF calibration claimed'])
    write_json(out / 'optimized.json', record)
    (out / 'optimized.yaml').write_text(parameter_yaml(record, (out / 'optimized.json').resolve()))
    np.savez_compressed(out / 'fitting_features.npz', relative_time_s=time, omega_lidar=omega,
                        gyro_imu_aligned=measured(tau), gyro_prediction=omega @ r.T + bias,
                        acceleration_error_prior=before_acc, acceleration_error_refined=after_acc,
                        train=train, validation=validation, coarse_tau_s=taus, coarse_cost=coarse)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    actual = measured(tau); predicted = omega @ r.T + bias
    for i in range(3):
        axes[i].plot(time, actual[:, i], label='IMU aligned', lw=1)
        axes[i].plot(time, predicted[:, i], label='LiDAR mapped to IMU', lw=1)
        axes[i].set_ylabel('gyro %s rad/s' % 'xyz'[i]); axes[i].grid(alpha=.3)
    axes[0].legend()
    axes[3].plot(time, np.linalg.norm(before_acc, axis=1), label='camera translation prior')
    axes[3].plot(time, np.linalg.norm(after_acc, axis=1), label='attempted translation refinement')
    axes[3].set_ylabel('accel residual m/s2'); axes[3].legend(); axes[3].grid(alpha=.3)
    for ax in axes:
        ax.axvspan(args.validation_start, time[-1], color='green', alpha=.08)
    axes[-1].set_xlabel('bag relative time (s); green = held out')
    fig.tight_layout(); fig.savefig(out / 'motion_alignment.png', dpi=130); plt.close(fig)
    print(json.dumps({k:record[k] for k in ['imu_time_offset','rotation_correction_deg','rotation_refinement_accepted',
                                          'translation_refinement_accepted','metrics','attempted_translation_delta_m']}, indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('compose')
    p.add_argument('--camera', type=Path, required=True); p.add_argument('--tf-bag', type=Path, required=True)
    p.add_argument('--imu-bag', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--same-device', action='store_true', help='Assert that reused internal TF belongs to this D435i')
    p = sub.add_parser('extract')
    p.add_argument('--bag', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('fit')
    p.add_argument('--prior', type=Path, required=True); p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--imu', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-start', type=float, default=50.)
    p.add_argument('--baseline-run', type=Path, help='Optional prior native run metadata for a fair gyro diagnostic')
    args = parser.parse_args(); globals()[args.stage](args)


if __name__ == '__main__':
    main()
