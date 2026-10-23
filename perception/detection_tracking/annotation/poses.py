"""Bind exported raw scans to recorded, causal localization estimates (CPU only)."""
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np

from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.results import file_fingerprint


class RecordedPoses:
    def __init__(self, root, source, max_age_s):
        root = Path(root)
        stem = Path(source['path']).stem
        candidates = [root / 'localization' / stem, root / 'odometry' / stem, root / stem, root]
        self.directory = next((p for p in candidates if (p / 'run.json').is_file() and
                               (p / 'localization.jsonl').is_file()), None)
        if self.directory is None:
            raise FileNotFoundError('没有该子包的定位结果：' + stem)
        run_path = self.directory / 'run.json'
        states_path = self.directory / 'localization.jsonl'
        run = json.loads(run_path.read_text())
        if (Path(run.get('bag', '')).expanduser().resolve() != Path(source['path']).resolve() or
                run.get('mode') not in ('localization', 'odometry') or not run.get('success')):
            raise ValueError('定位结果的bag绑定、模式或完成状态不匹配：' + str(self.directory))
        if run.get('assume_stationary'):
            raise ValueError('不能用静止假设结果代替实测运动位姿')
        bag_fingerprint = run.get('playback', {}).get('bag_fingerprint')
        if bag_fingerprint is not None and bag_fingerprint != json.loads(source['fingerprint']):
            raise ValueError('定位结果与标注的bag指纹不匹配：' + stem)
        self.rows = [json.loads(line) for line in states_path.read_text().splitlines() if line.strip()]
        self.times = [row['stamp_ns'] for row in self.rows]
        if (not self.times or any(type(t) is not int or t < 0 for t in self.times) or
                any(b <= a for a, b in zip(self.times, self.times[1:]))):
            raise ValueError('定位状态须有严格递增的整数纳秒时间戳')
        self.max_age_ns = round(max_age_s * 1e9)
        self.source_id = source['id']
        self.provenance = dict(run_path=str(self.directory), mode=run['mode'], backend=run.get('backend'),
                               run_fingerprint=file_fingerprint(run_path),
                               states_fingerprint=file_fingerprint(states_path),
                               bag_binding='recorded_fingerprint_and_path' if bag_fingerprint is not None
                                           else 'recorded_path_and_sensor_timestamps',
                               bag_fingerprint=bag_fingerprint,
                               map_fingerprint=run.get('playback', {}).get('map_fingerprint'),
                               geometry_accuracy_verified=bool(run.get('geometry_accuracy_verified', False)),
                               relative_pose_frame='continuous_local', pose_reference='sensor_header',
                               matching='nearest_past', max_age_s=max_age_s)

    def resolve(self, frame):
        result = dict(valid=False, reason='before_first_pose', source_id=self.source_id,
                      query_stamp_ns=frame['stamp_ns'], local_from_lidar=None)
        index = bisect_right(self.times, frame['stamp_ns']) - 1
        if index < 0:
            return result
        row = self.rows[index]
        result.update(stamp_ns=row['stamp_ns'], age_s=(frame['stamp_ns'] - row['stamp_ns']) / 1e9,
                      frame_id=row.get('frame_id'), child_frame_id=row.get('child_frame_id'),
                      session_id=row.get('session_id'), generation=row.get('generation'),
                      estimator=row.get('source'), health=row.get('health'))
        if frame['stamp_ns'] - row['stamp_ns'] > self.max_age_ns:
            result['reason'] = 'pose_too_old'
        elif row.get('health') != 0:
            result['reason'] = 'pose_invalid:' + str(row.get('reason') or row.get('status'))
        elif row.get('child_frame_id') != frame['sensor_frame']:
            result['reason'] = 'sensor_frame_mismatch'
        elif (not row.get('frame_id') or not row.get('session_id') or
              type(row.get('generation')) is not int or row['generation'] < 0):
            result['reason'] = 'missing_pose_identity'
        else:
            # A malformed claimed-valid estimate is an error, not a zero/identity fallback.
            matrix = pose_matrix(row.get('position'), row.get('quaternion_xyzw'))
            result.update(valid=True, reason=None, local_from_lidar=matrix.tolist())
            if row.get('has_map_pose'):
                if not row.get('map_frame'):
                    raise ValueError('地图位姿缺少坐标帧')
                result.update(map_frame=row['map_frame'], map_revision=row.get('map_revision'),
                              map_from_lidar=pose_matrix(row.get('map_position'),
                                                        row.get('map_quaternion_xyzw')).tolist())
        return result


def relative_pose(current, previous):
    """Use one continuous local lifecycle, irrespective of map corrections."""
    if not current['valid']:
        return None, 'current:' + current['reason']
    if not previous['valid']:
        return None, 'history:' + previous['reason']
    identity = ('source_id', 'frame_id', 'child_frame_id', 'session_id', 'generation')
    if any(current[key] != previous[key] for key in identity):
        return None, 'pose_lifecycle_mismatch'
    matrix = np.linalg.solve(np.asarray(current['local_from_lidar']), np.asarray(previous['local_from_lidar']))
    return matrix.tolist(), None


def attach_poses(plan, root, max_age_s):
    providers = {s['id']: RecordedPoses(root, s, max_age_s) for s in plan['sources']}
    for frame in plan['assets'].values():
        frame['pose'] = providers[frame['source_id']].resolve(frame)
    reasons = {}
    ready = partial = unavailable = usable = 0
    for sample in plan['samples']:
        current = plan['assets'][sample['pointcloud_id']]['pose']
        valid_count = 0
        for history in sample['history']:
            previous = plan['assets'][history['pointcloud_id']]['pose']
            transform, reason = relative_pose(current, previous)
            history.update(current_from_history=transform, pose_reason=reason)
            valid_count += int(transform is not None)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        status = ('unavailable' if not current['valid'] else
                  'partial' if valid_count < plan['history']['frames'] else 'ready')
        sample['motion'] = dict(status=status, usable_history_frames=valid_count,
                                current_pose_valid=current['valid'])
        ready += status == 'ready'
        partial += status == 'partial'
        unavailable += status == 'unavailable'
        usable += valid_count
    plan['pose_sources'] = {ident: provider.provenance for ident, provider in providers.items()}
    plan['history'].update(transforms_available=True, compensation='continuous_local_relative_pose_on_load',
                           pose_reference='sensor_header', pose_matching='nearest_past',
                           max_pose_age_s=max_age_s, missing_pose='use_current_and_available_history')
    plan['counts'].update(samples_with_full_motion_history=ready, samples_with_partial_motion_history=partial,
                          samples_without_current_pose=unavailable, usable_history_references=usable,
                          invalid_history_pose_reasons=reasons)
    return plan
