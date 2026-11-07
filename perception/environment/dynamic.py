"""Causal person masks and independent static support; unknown never means free."""
import cv2
import numpy as np


def people_boxes(row, stamp, identity, frame, settings, *, clock_domain=None):
    status = dict(status='unavailable', reason='no_people', selected_tracks=0)
    if row is None:
        return np.empty((0, 8)), status
    status.update(observation_id=row.get('observation_id'), input_fingerprint=row.get('input_fingerprint'),
                  stamp_s=row.get('stamp_s'))
    age = stamp-float(row['stamp_s'])
    binding = row.get('localization_binding') or {}
    if not np.isfinite(age) or not 0 <= age <= settings['max_people_age_s']:
        status['reason'] = 'future_or_stale_people'
    elif clock_domain is not None and row.get('clock_domain') != clock_domain:
        status['reason'] = 'clock_mismatch'
    elif (row.get('health') != 'valid' or row.get('frame_id') != frame or
            (binding.get('session_id'), binding.get('generation'), binding.get('frame_id')) != (*identity, frame)):
        status['reason'] = 'localization_binding_mismatch'
    else:
        boxes = [];track_ids=[]
        for track in row.get('tracks', []):
            box = np.asarray(track['box'], dtype=float)
            observed_age = stamp-float(track['last_observed_stamp_s'])
            if (track['state'] == 'observed' and track['confirmed'] and track['hits'] >= settings['min_hits']
                    and 0 <= observed_age <= settings['max_people_age_s'] and box.shape == (8,)
                    and np.isfinite(box).all() and (box[3:6] > 0).all() and box[7] >= settings['min_score']):
                boxes.append(box)
                track_ids.append(str(track['track_id']) if 'track_id' in track else None)
        status.update(status='valid', reason=None, selected_tracks=len(boxes),track_ids=track_ids)
        return np.asarray(boxes).reshape(-1, 8), status
    return np.empty((0, 8)), status


def floor_at(points, floor, origin, resolution):
    cells = np.floor((points[:, :2]-origin)/resolution).astype(int)
    height, width = floor.shape
    inside = (cells >= 0).all(axis=1) & (cells[:, 0] < width) & (cells[:, 1] < height)
    values = np.full(len(points), np.nan)
    values[inside] = floor[cells[inside, 1], cells[inside, 0]]
    return values


def _extended_person_band(points, box, floor, origin, config, distance):
    """Local, bounded ground extrapolation for a mask only, not free-space evidence."""
    from .geometry import nearest_support
    result = np.zeros(len(points), bool)
    if not len(points) or distance <= 0:
        return result
    resolution = config['resolution']
    radius = config['dynamic']['max_radius_m']+distance+resolution
    lower = np.maximum(0, np.floor((box[:2]-radius-origin)/resolution).astype(int))
    upper = np.minimum(floor.shape[::-1], np.ceil((box[:2]+radius-origin)/resolution).astype(int))
    if (upper <= lower).any():
        return result
    patch = floor[lower[1]:upper[1], lower[0]:upper[0]]
    valid = np.isfinite(patch)
    if valid.sum() < config['ground_min_points']:
        return result
    y, x = np.nonzero(valid)
    xy = (np.c_[x, y]+lower+.5)*resolution+origin-box[:2]
    design = np.c_[xy, np.ones(len(xy))]
    plane, _, rank, _ = np.linalg.lstsq(design, patch[valid], rcond=None)
    if (rank < 3 or np.linalg.norm(plane[:2]) > np.tan(config['ground_max_slope_rad']) or
            np.max(np.abs(design@plane-patch[valid])) > config['ground_tolerance_m']):
        return result
    cells = np.floor((points[:, :2]-origin)/resolution).astype(int)-lower
    inside = (cells >= 0).all(axis=1) & (cells < upper-lower).all(axis=1)
    indices = np.flatnonzero(inside)
    if not len(indices):
        return result
    gaps = nearest_support(valid)[0][cells[indices, 1], cells[indices, 0]]*resolution
    height = points[indices, 2]-(np.c_[points[indices, :2]-box[:2], np.ones(len(indices))]@plane)
    uncertainty = config['ground_tolerance_m']+gaps*np.tan(config['ground_max_slope_rad'])
    result[indices] = ((gaps <= distance) & (height >= config['min_obstacle_height_m']+uncertainty)
                       & (height <= config['max_height_m']-uncertainty))
    return result


def person_points(points, boxes, floor, origin, config, *, ground_extrapolation=0.):
    """Keep ground and disconnected background; weak box sizes are bounded hints."""
    settings, resolution = config['dynamic'], config['resolution']
    mask = np.zeros(len(points), bool)
    if not len(boxes) or not len(points):
        return mask
    eligible = obstacle_band(points, floor, origin, config)
    unsupported = ~np.isfinite(floor_at(points, floor, origin, resolution)) if ground_extrapolation else None
    for box in boxes:
        xy = points[:, :2]-box[:2]
        c, s = np.cos(box[6]), np.sin(box[6])
        local = xy @ np.array([[c, -s], [s, c]])
        half = np.clip(box[3:5]/2+settings['padding_m'], resolution, settings['max_radius_m'])
        inside = (np.abs(local) <= half).all(axis=1)
        selected_band = eligible
        if ground_extrapolation:
            missing = np.flatnonzero(inside & unsupported)
            selected_band = eligible.copy()
            selected_band[missing] = _extended_person_band(points[missing], box, floor, np.asarray(origin),
                                                         config, ground_extrapolation)
        indices = np.flatnonzero(selected_band & inside)
        if len(indices) < settings['min_cluster_points']:
            continue
        distance = np.linalg.norm(xy[indices], axis=1)
        seed = int(np.argmin(distance))
        if distance[seed] > settings['seed_distance_m']:
            continue
        cells = np.floor((local[indices]+half)/resolution).astype(int)
        shape = np.ceil(2*half/resolution).astype(int)+1
        raster = np.zeros((shape[1], shape[0]), np.uint8)
        raster[cells[:, 1], cells[:, 0]] = 1
        _, labels = cv2.connectedComponents(raster, connectivity=8)
        selected = labels[cells[:, 1], cells[:, 0]] == labels[cells[seed, 1], cells[seed, 0]]
        if selected.sum() >= settings['min_cluster_points']:
            mask[indices[selected]] = True
    return mask


def obstacle_band(points, floor, origin, config):
    height = points[:, 2]-floor_at(points, floor, origin, config['resolution'])
    eligible = np.isfinite(height)
    values = height[eligible]
    eligible[eligible] = (values >= config['min_obstacle_height_m']) & (values <= config['max_height_m'])
    return eligible


def voxel_keys(points, resolution):
    # Same world-aligned lattice as OctoMap; cast to float32 at the native boundary.
    cells = np.floor(np.asarray(points, np.float32).astype(np.float64)/resolution).astype(np.int64)
    if (np.abs(cells) >= 2**20).any():
        raise ValueError('Static evidence exceeds voxel-key coordinate bounds')
    cells += 2**20
    return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]


class StaticEvidence:
    """Count time-separated scan support, not points, without retaining full scans."""
    def __init__(self, config):
        self.config, self.settings = config, config['dynamic']
        self.support, self.person = {}, set()

    def add(self, points, person_mask, stamp, floor, origin):
        eligible = obstacle_band(points, floor, origin, self.config)
        keys = voxel_keys(points, self.config['resolution'])
        masked = set(map(int, np.unique(keys[eligible & person_mask])))
        self.person.update(masked)
        # Earlier detector misses must not permanently certify a standing person.
        # Background may be accepted again only after fresh unmasked observations.
        for key in masked:
            self.support.pop(key, None)
        bucket = int(stamp/self.settings['static_observation_interval_s'])
        required = self.settings['static_min_observations']
        for key in np.unique(keys[eligible & ~person_mask]):
            key = int(key)
            if key in masked:
                continue
            old = self.support.get(key)
            count = (old & 255) if old is not None else 0
            if count < required and (old is None or old >> 8 != bucket):
                self.support[key] = (bucket << 8) | min(required, count+1)
        if len(self.support)+len(self.person) > self.config['static']['max_voxels']:
            raise ValueError('Static review voxel budget exceeded')

    def keep(self, points):
        keys = voxel_keys(points, self.config['resolution'])
        required = self.settings['static_min_observations']
        return np.fromiter((((self.support.get(int(key), 0) & 255) >= required) or
                            (int(key) not in self.person) for key in keys), bool, count=len(keys))

    def summary(self):
        return dict(nonperson_voxels=len(self.support), person_voxels=len(self.person),
                    stable_voxels=sum((value & 255) >= self.settings['static_min_observations']
                                      for value in self.support.values()))
