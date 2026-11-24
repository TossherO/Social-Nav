"""Build and select offline route-reference goals for an interactive bag."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading

import numpy as np
import yaml

from perception.environment.context import fingerprint
from perception.mapping_localization.context import pose_matrix
from perception.mapping_localization.results import file_fingerprint, map_fingerprint


ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'recorded_route_planning_review'
MAX_GAP_S = .5
MAX_DEVIATION_M = 1.5


class RecordedRoute:
    def __init__(self, rows, geometry, frame):
        segments, segment = [], []
        last_stamp = -math.inf
        identity = None

        def finish():
            if len(segment) >= 2:
                points = np.asarray(segment)
                distance = np.r_[0., np.linalg.norm(np.diff(points[:, 1:3], axis=0), axis=1).cumsum()]
                if distance[-1] > .35:
                    segments.append((points, distance))
            segment.clear()

        for row in rows:
            stamp = row['stamp_s']
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= last_stamp:
                raise ValueError('Recorded localization must have strictly increasing measurement stamps')
            last_stamp = stamp
            if (row.get('source') != 'roll_fast_lio' or row.get('status') != 'PRIOR_ASSOCIATED' or
                    row.get('health') != 0 or row.get('has_map_pose') is not True or row.get('map_frame') != frame):
                finish()
                continue
            current = tuple(row[k] for k in ('session_id', 'generation', 'frame_id', 'map_revision'))
            matrix = pose_matrix(row['map_position'], row['map_quaternion_xyzw'])
            pose = geometry.pose(matrix)
            z = float(matrix[2, 3])
            if segment:
                dt = stamp - segment[-1][0]
                displacement = np.linalg.norm(pose[:2] - segment[-1][1:3])
                if (current != identity or dt > MAX_GAP_S or displacement > .35 + 2.5*dt or
                        abs(z-segment[-1][4]) > .35 + dt):
                    finish()
            identity = current
            segment.append([stamp, *pose, z])
        finish()
        if not segments:
            raise ValueError('No continuous moving section has trusted map localization')
        self.segments = segments
        self.starts = np.array([points[0, 0] for points, _ in segments])

    def target(self, stamp, current, lookahead):
        index = int(np.searchsorted(self.starts, stamp, side='right'))-1
        if index < 0 or stamp > self.segments[index][0][-1, 0]:
            return None, 'outside_recorded_route_section'
        points, distance = self.segments[index]
        recorded = np.array([np.interp(stamp, points[:, 0], points[:, axis]) for axis in (1, 2)])
        if np.linalg.norm(current[:2]-recorded) > MAX_DEVIATION_M:
            return None, 'current_pose_away_from_recorded_route'
        progress = float(np.interp(stamp, points[:, 0], distance))
        remaining = distance[-1]-progress
        if remaining < .35:
            return None, 'no_forward_target_in_section'
        target = min(progress+lookahead, distance[-1])
        right = min(int(np.searchsorted(distance, target, side='left')), len(distance)-1)
        left = right-1
        alpha = (target-distance[left])/(distance[right]-distance[left])
        xy = points[left, 1:3] + alpha*(points[right, 1:3]-points[left, 1:3])
        yaw = float(points[left, 3] + alpha*math.remainder(points[right, 3]-points[left, 3], 2*math.pi))
        if np.linalg.norm(current[:2]-xy) < .25:
            return None, 'no_forward_target_in_section'
        return (index, [*xy.tolist(), yaw], progress, target), None

    def coverage(self):
        return dict(first_stamp_s=float(self.starts[0]), last_stamp_s=float(self.segments[-1][0][-1, 0]),
                    continuous_sections=len(self.segments))


class BagShadow:
    """Route preparation runs independently of the playback and control clocks."""
    def __init__(self, bag, config, settings, geometry, asset, *, start, duration, cache, route_run, output):
        self.bag, self.config, self.geometry, self.asset = bag, config, geometry, asset
        self.start, self.duration = start, duration
        self.settings = settings['route_replay']
        self.python = settings['input_python']
        self.route_run, self.cache, self.output = route_run, Path(cache), output
        self.frame = asset.snapshot.metadata['frame_id']
        self.lock = threading.RLock()
        self.thread = None
        self.preparation_stop = None
        self.closed = False
        self.route = self.source = self.error = None
        self.enabled = False
        self.state, self.reason = 'disabled', None
        self.request = self.section = self.last_submit = self.target_progress = None
        self.planning_status = self.planning_reason = None
        self.dwa_status = self.dwa_reason = None
        self.revision = 0

    def toggle(self, enabled, *, reason=None):
        with self.lock:
            if enabled == self.enabled:
                return
            self.enabled = enabled
            self.request = self.section = self.last_submit = self.target_progress = None
            self.planning_status = self.planning_reason = None
            self.dwa_status = self.dwa_reason = None
            self.state = 'waiting' if enabled else 'disabled'
            self.reason = reason
            if not enabled:
                if self.preparation_stop is not None:
                    self.preparation_stop.set()
            elif self.route is None:
                self.state, self.error = 'preparing', None
                if self.thread is None:
                    self.preparation_stop = threading.Event()
                    self.thread = threading.Thread(target=self._prepare, name='bag-shadow-route')
                    self.thread.start()

    def _prepare(self):
        while True:
            stop = self.preparation_stop
            try:
                result, error = self._load_route(stop), None
            except Exception as failure:
                result, error = None, str(failure)
            with self.lock:
                if stop.is_set() and self.enabled and not self.closed:
                    self.preparation_stop = threading.Event()
                    continue
                if not stop.is_set():
                    if result is not None:
                        self.route, self.source = result
                        self.state, self.reason = 'waiting', 'current_localization_unavailable'
                    elif error is not None:
                        self.state, self.reason = 'unavailable', error
                    self.error = error
                self.thread = self.preparation_stop = None
                return

    def _load_route(self, stop):
        map_directory = Path(self.config['roll']['saveKeyframeMapDirectory']).parent
        binding = dict(bag_fingerprint=file_fingerprint(self.bag), map_fingerprint=map_fingerprint(map_directory))
        if self.route_run is not None:
            run = Path(self.route_run).resolve()
            metadata = json.loads((run/'run.json').read_text())
            playback = metadata.get('playback') or {}
            records = playback.get('records')
            if (metadata.get('success') is not True or metadata.get('mode') != 'localization' or
                    metadata.get('localization_source') != 'roll_fast_lio' or
                    metadata.get('source') == 'live' or metadata.get('scope') == 'online_topics' or
                    metadata.get('inertial_input') == 'synthetic_stationary_assumption' or
                    any(playback.get(k) != value for k, value in binding.items()) or records != 'localization.jsonl'):
                raise ValueError('Recorded localization must be a successful fixed-map run for this bag and map')
            with (run/records).open() as stream:
                route = RecordedRoute((json.loads(line) for line in stream), self.geometry, self.frame)
            return route, dict(kind='recorded_localization', path=str(run/records), **route.coverage())

        config = copy.deepcopy(self.config)
        config['roll'].pop('saveKeyframeMapDirectory')
        native = ROOT/'outputs/runtime/build/localization'
        libraries = {path.name: file_fingerprint(path) for path in sorted(native.glob('libsocial_nav_*.so'))}
        source_files = ('runtime/bag_shadow.py', 'perception/sensors/observations.py',
                        'perception/mapping_localization/runtime.py', 'perception/mapping_localization/context.py',
                        'perception/environment/configs/environment.yaml')
        binding.update(start_s=self.start, duration_s=self.duration, localization=fingerprint(config),
                       libraries=libraries, sources={name: file_fingerprint(ROOT/name) for name in source_files
                                                     if (ROOT/name).is_file()})
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache/(fingerprint(binding)+'.json')
        reused = path.exists()
        if not reused:
            self._generate(path, binding, stop)
        document = json.loads(path.read_text())
        if document.get('source') != binding or document.get('scope') != SCOPE:
            raise ValueError('Cached route source or map has changed')
        route = RecordedRoute(document['localization'], self.geometry, self.frame)
        return route, dict(kind='recomputed_localization', cache_hit=reused, path=str(path), **route.coverage())

    def _generate(self, path, binding, stop):
        if stop.is_set():
            raise InterruptedError('Route preparation stopped')
        with tempfile.TemporaryDirectory(prefix='bag-shadow-', dir=self.cache) as temp:
            cfg, result = Path(temp)/'localization.yaml', Path(temp)/'route.json'
            cfg.write_text(yaml.safe_dump(self.config))
            command = [self.python, '-B', '-m', 'runtime.bag_shadow', '--bag', str(self.bag), '--config', str(cfg),
                       '--start', str(self.start), '--output', str(result)]
            if self.duration is not None:
                command += ['--duration', str(self.duration)]
            env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=str(ROOT)+':/opt/ros/noetic/lib/python3/dist-packages')
            env['LD_LIBRARY_PATH'] = '/opt/ros/noetic/lib:'+str(ROOT/'outputs/perception/dependencies/gtsam/lib')+':'+env.get('LD_LIBRARY_PATH', '')
            with (self.output/'shadow_route.log').open('a') as log, subprocess.Popen(
                    command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as process:
                while process.poll() is None:
                    if stop.wait(.1):
                        process.terminate()
                        try:
                            process.wait(timeout=2.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise InterruptedError('Route preparation stopped')
                if process.returncode:
                    raise RuntimeError('Map localization for the recorded route failed; see shadow_route.log')
            rows = json.loads(result.read_text())
            # Verify the track before publishing the cache. An unusable result can be retried.
            RecordedRoute(rows, self.geometry, self.frame)
            result.write_text(json.dumps(dict(scope=SCOPE, source=binding, localization=rows), allow_nan=False))
            if stop.is_set():
                raise InterruptedError('Route preparation stopped')
            result.replace(path)

    def suspend(self):
        """The safety worker has retired the last offered route revision."""
        with self.lock:
            self.request = self.last_submit = self.section = self.target_progress = None
            self.planning_status = self.planning_reason = self.dwa_status = self.dwa_reason = None

    def select(self, scene, row, *, now, clock_domain, planning, source_ended=False):
        from navigation.planning.inputs import association_status, scene_status

        with self.lock:
            if not self.enabled:
                return None

            def wait(state, reason):
                self.state, self.reason = state, reason
                self.request = self.last_submit = self.section = self.target_progress = None
                self.planning_status = self.planning_reason = self.dwa_status = self.dwa_reason = None
                return None

            if self.error is not None:
                return wait('unavailable', self.error)
            if self.route is None:
                return wait('preparing', None)
            if source_ended:
                return wait('waiting', 'bag_ended')
            if clock_domain != 'runtime_bag' or row is None or scene is None or row['health'] != 0:
                return wait('waiting', 'current_localization_unavailable')
            if row.get('has_map_pose') is not True or row.get('map_frame') != self.frame:
                return wait('waiting', 'association_unavailable')
            record = scene.record
            stamp = row['stamp_s']
            fields = ('session_id', 'generation', 'frame_id')
            if (tuple(row[k] for k in fields) != tuple(record[k] for k in fields) or
                    stamp < record['as_of'] or
                    not 0 <= now-stamp <= planning['max_geometry_age_s']):
                return wait('waiting', 'current_localization_unavailable')
            try:
                mapped = pose_matrix(row['map_position'], row['map_quaternion_xyzw'])
            except (KeyError, ValueError, TypeError):
                return wait('waiting', 'current_localization_invalid')
            target, reason = self.route.target(stamp, self.geometry.pose(mapped), self.settings['lookahead_m'])
            if target is None:
                return wait('waiting', reason)
            section, goal, progress, target_progress = target
            reason = scene_status(scene, now, clock_domain, planning) or association_status(scene, self.asset, planning)
            if reason:
                if (self.request is not None and section == self.section and
                        reason in ('geometry_future_or_stale','navigation_prior_future_or_stale')):
                    self.state, self.reason = 'waiting', reason
                    return self.request
                return wait('waiting', reason)
            self.state, self.reason = 'planning', None
            if self.request is not None and section == self.section and progress < self.target_progress:
                delta = np.linalg.norm(np.asarray(self.request['goal'][:2])-goal[:2])
                if stamp-self.last_submit < self.settings['min_update_s'] or delta < self.settings['goal_shift_m']:
                    return self.request
            self.revision += 1
            self.section, self.last_submit, self.target_progress = section, stamp, target_progress
            self.planning_status, self.planning_reason = 'pending', None
            self.dwa_status = self.dwa_reason = None
            self.request = dict(request_id=f'bag-shadow:{self.revision}', task_id='bag-shadow', goal_revision=self.revision,
                                mode='mapped', goal=goal, frame_id=self.frame, target_reference='base_link',
                                algorithm='hybrid_astar', goal_behavior='stop', motion_intent='forward_preferred',
                                map_snapshot_id=self.asset.manifest['snapshot_id'], source=SCOPE)
            return self.request

    def note_result(self, request, status, reason, *, proposal, control_reason=None):
        with self.lock:
            if self.enabled and self.request == request:
                self.planning_status, self.planning_reason = status, reason
                self.dwa_status = (proposal or {}).get('status')
                self.dwa_reason = (proposal or {}).get('reason') or control_reason

    def status(self):
        with self.lock:
            return dict(available=True, enabled=self.enabled, state=self.state, reason=self.reason,
                        goal=None if self.request is None else self.request['goal'], route_source=self.source,
                        request_id=None if self.request is None else self.request['request_id'],
                        planning_status=self.planning_status, planning_reason=self.planning_reason,
                        dwa_status=self.dwa_status, dwa_reason=self.dwa_reason,
                        scope=SCOPE, executable=False, execution_feedback=False, task_completed=False)

    def close(self):
        with self.lock:
            self.closed = True
            if self.preparation_stop is not None:
                self.preparation_stop.set()
            thread = self.thread
        if thread is not None:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('Route preparation did not stop')
        return self.status()


def main():
    from perception.mapping_localization.build_map import wait_for_backend
    from perception.mapping_localization.runtime import LocalizationSession
    from perception.sensors.observations import bag_observations, selected_messages

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--duration', type=float)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    source = selected_messages(args.bag, config, start=args.start, duration=args.duration)
    session = LocalizationSession(args.config, mode='localization', allow_local_only=True)
    stop = threading.Event()
    fields = ('stamp_s', 'session_id', 'generation', 'frame_id', 'source', 'status', 'health',
              'has_map_pose', 'map_frame', 'map_revision', 'map_position', 'map_quaternion_xyzw')
    rows = []
    try:
        for batch in bag_observations(None, config, messages=source):
            wait_for_backend(session, stop, 120., max_pending=1)
            row, _, _, _, _ = session.step(batch)
            rows.append({key: row[key] for key in fields if key in row})
    finally:
        session.close()
    if session.error:
        raise RuntimeError(session.error)
    args.output.write_text(json.dumps(rows, allow_nan=False))


if __name__ == '__main__':
    main()
