"""Continuous local pose with a separately timed, optional map association."""
from collections import OrderedDict, deque

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from .results import checked_pose


def pose_matrix(position, quaternion):
    pair = checked_pose(position, quaternion)
    if pair is None:
        raise ValueError('Invalid measured pose')
    p, q = pair
    x, y, z, w = q
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = p
    return validate_transform(matrix)


class LocalizationContext:
    def __init__(self, source, *, requires_map=False, allow_local_only=False,
                 max_map_age=.4, confirm_matches=2, history_size=64, coverage=None):
        if max_map_age <= 0 or confirm_matches < 1 or history_size < 2:
            raise ValueError('Invalid association budget')
        self.source, self.requires_map, self.allow_local_only = source, requires_map, allow_local_only
        self.max_map_age, self.confirm_matches, self.history_size = max_map_age, confirm_matches, history_size
        self.history, self.matches = OrderedDict(), deque(maxlen=history_size)
        self.identity, self.last_stamp = None, None
        self.retired = deque(maxlen=32)
        self.retired_maps = deque(maxlen=32)
        self.frame = None
        self.coverage = coverage
        self.association_generation = -1
        self.active_association = None
        self.confirmation_floor = -np.inf
        self.trusted_correction = None
        self.trusted_map_identity = None

    def add_match(self, row):
        if not np.isfinite(row['stamp_s']):
            raise ValueError('Map observation has no finite acquisition time')
        # Bound memory; a result only becomes eligible once its frontend pose is known.
        if (row['session_id'],row['generation']) not in self.retired_maps:
            self.matches.append(dict(row))

    def update(self, row):
        stamp = float(row['stamp_s'])
        identity = (row['session_id'], row['generation'])
        if not np.isfinite(stamp) or not row['session_id']:
            raise ValueError('Frontend observation has no valid time/session')
        if identity in self.retired:
            return None
        if identity == self.identity and self.last_stamp is not None and stamp <= self.last_stamp:
            return None
        if identity != self.identity:
            if self.identity is not None:
                self.retired.append(self.identity)
                self.retired_maps.extend({(m['session_id'],m['generation']) for m in self.matches})
            self.history.clear()
            self.matches.clear()  # Never bind an old backend result to a new local session.
            self.identity = identity
            self.frame = row['frame_id']
            self.active_association = None
            self.confirmation_floor = -np.inf
            self.trusted_correction = None
            self.trusted_map_identity = None
        if row['frame_id'] != self.frame:
            raise ValueError('Local frame changed without a new localization session')
        local = pose_matrix(row['position'], row['quaternion_xyzw']) if row['health'] == 0 else None
        self.last_stamp = stamp
        self.history[stamp] = local
        while len(self.history) > self.history_size:
            self.history.popitem(last=False)
        candidates = []
        for item in self.matches:
            measured = item['stamp_s']
            if (item['frame_id'] != self.frame or measured > stamp or measured not in self.history
                    or self.history[measured] is None):
                continue
            try:
                backend_local = pose_matrix(item['position'], item['quaternion_xyzw'])
                if not np.allclose(backend_local, self.history[measured], atol=1e-4):
                    continue
                candidates.append(item)
            except ValueError:
                continue
        candidates = sorted({item['stamp_s']: item for item in candidates}.values(), key=lambda x: x['stamp_s'])
        match = candidates[-1] if candidates else None
        accepted = []
        coverage_status = 'NOT_CONFIGURED' if self.coverage is None else 'UNAVAILABLE'
        if match and match['health'] == 0 and match.get('has_map_pose'):
            association_identity = (match['session_id'], match['generation'])
            for item in reversed(candidates):
                if ((item['session_id'], item['generation']) != association_identity or
                        item['health'] != 0 or not item.get('has_map_pose') or item['map_frame'] != match['map_frame'] or
                        item['stamp_s'] <= self.confirmation_floor or
                        (accepted and accepted[-1]['stamp_s']-item['stamp_s'] > self.max_map_age)):
                    break
                if self.coverage and not self.coverage.contains(item['map_position'],item['map_frame']):
                    break
                accepted.append(item)
        age = stamp-match['stamp_s'] if match else None
        map_valid = (local is not None and len(accepted) >= self.confirm_matches and
                     age is not None and age <= self.max_map_age)
        correction = None
        if match and self.trusted_map_identity is not None and (
                match['session_id'], match['generation']) != self.trusted_map_identity[:2]:
            self.trusted_correction = None
            self.trusted_map_identity = None
        if local is not None and match and match['health'] == 0 and match.get('has_map_pose'):
            correction = validate_transform(pose_matrix(match['map_position'], match['map_quaternion_xyzw']) @
                                            np.linalg.inv(self.history[match['stamp_s']]))
            if self.coverage and age <= self.max_map_age:
                covered = self.coverage.contains((correction @ local)[:3,3], match['map_frame'])
                coverage_status = 'COVERED' if covered else 'OUTSIDE_COVERAGE'
                map_valid = map_valid and covered
        if self.coverage and local is not None and self.trusted_correction is not None:
            # Use the last confirmed anchor, including while the matcher is disabled
            # outside coverage. A new false match must not move the region test.
            covered = self.coverage.contains((self.trusted_correction @ local)[:3,3],
                                              self.trusted_map_identity[2])
            coverage_status = 'COVERED' if covered else 'OUTSIDE_COVERAGE'
            map_valid = map_valid and covered
        if self.active_association is not None and not map_valid:
            # Reentry must collect new measurements; old valid matches cannot
            # immediately resurrect an association after loss/coverage exit.
            self.confirmation_floor = (stamp if coverage_status == 'OUTSIDE_COVERAGE' else
                                       match['stamp_s'] if match else stamp)
            self.active_association = None
        if local is None:
            self.confirmation_floor = stamp
        association = dict(status='LOCAL_INVALID' if local is None else 'LOCAL_ONLY',
                           valid=False, measurement_stamp_s=match['stamp_s'] if match else None,
                           age_s=age, map_from_local=None, map_frame='',
                           association_session=match['session_id'] if match else '',
                           association_generation=max(0,self.association_generation),
                           backend_generation=match['generation'] if match else None,
                           map_revision=match['map_revision'] if match else 0,
                           coverage_status=coverage_status,
                           coverage_fingerprint=self.coverage.fingerprint if self.coverage else None,
                           reason=row.get('reason', '') if local is None else (
                               'outside_prior_coverage' if coverage_status == 'OUTSIDE_COVERAGE' else 'prior_association_unavailable'))
        result = dict(row, source=self.source, has_map_pose=False, map_frame='', map_revision=0,
                      status=association['status'])
        if local is not None and match and accepted and not map_valid and coverage_status != 'OUTSIDE_COVERAGE':
            association['status'] = 'ASSOCIATING' if age <= self.max_map_age else 'LOCAL_ONLY'
        if map_valid:
            key = (match['session_id'],match['generation'],match['map_frame'])
            if key != self.active_association:
                self.association_generation += 1
            self.active_association = key
            self.trusted_correction = correction.copy()
            self.trusted_map_identity = key
            association.update(valid=True, status='PRIOR_ASSOCIATED', reason='',
                               map_from_local=correction, map_frame=match['map_frame'],
                               association_generation=self.association_generation)
            result.update(has_map_pose=True, map_frame=match['map_frame'],
                          map_revision=match['map_revision'], map_matrix=correction @ local)
        if local is not None:
            strict_failure = self.requires_map and not map_valid and not self.allow_local_only
            result.update(health=2 if strict_failure else 0,
                          status=association['status'],
                          reason=association['reason'] if strict_failure else '')
        return result, association
