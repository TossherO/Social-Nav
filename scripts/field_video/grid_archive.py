"""Load observation-bound navigation grids retained in a field run."""

from bisect import bisect_right
from functools import lru_cache
import json
import math
from pathlib import Path

import cv2
import numpy as np

from navigation.planning.inputs import scene_status, validate_grid
from perception.detection_tracking.inference.geometry import validate_transform
from perception.environment.navigation import navigation_geometry
from perception.environment.scene import SceneInputs


# The recorder normally saves a scene every half second. Allow for processing
# delay and capture jitter in the video while keeping the observation timestamp.
MAX_SNAPSHOT_AGE_S = 1.0
MAP_BINDING_FIELDS = ('map_id', 'map_frame', 'map_revision', 'association_session',
                      'association_generation')


class SceneGrids:
    def __init__(self, bag, geometry, config):
        self.geometry = geometry
        self.config = config
        items = []
        if bag is not None:
            for cycles in Path(bag).parent.parent.glob('agent/*/run/planning/cycles.jsonl'):
                with cycles.open(encoding='utf-8') as source:
                    for line in source:
                        cycle = json.loads(line)
                        saved = cycle.get('saved_scene')
                        if not saved:
                            # Unsaved cycles usually sit between captures. An
                            # explicit null binding records the loss of a scene.
                            if 'scene_binding' in cycle and cycle['scene_binding'] is None:
                                items.append((float(cycle['stamp_s']), None))
                            continue
                        path = None
                        if (isinstance(saved, str) and saved == cycle.get('scene_binding') and len(saved) == 64
                                and all(c in '0123456789abcdef' for c in saved)):
                            path = cycles.parent / 'scenes' / saved
                        items.append((float(cycle['stamp_s']), path))
        self.items = sorted(items, key=lambda item: item[0])
        self.times = [stamp for stamp, _ in self.items]

    @lru_cache(maxsize=12)
    def _recorded_grid(self, cycle_stamp, path):
        try:
            # Unknown source stamps are represented by NaN in the saved grid.
            with np.errstate(invalid='ignore'):
                scene = SceneInputs.load(path)
            record = scene.record
            prior = record['prior']
            identity = [record[k] for k in ('session_id', 'generation', 'frame_id')]
            measured = prior.get('measurement_stamp_s')
            if (record['input_fingerprint'] != path.name
                    or scene_status(scene, cycle_stamp, 'runtime_live', self.config, check_navigation=False)
                    or record['geometry']['stamp_s'] != record['as_of']
                    or not prior['valid'] or prior['status'] != 'PRIOR_ASSOCIATED'
                    or prior['stamp_s'] != record['as_of']
                    or identity != [prior[k] for k in ('local_session_id', 'local_generation', 'local_frame')]
                    or measured is None or measured > record['as_of']
                    or not 0 <= cycle_stamp - measured <= self.config['max_association_age_s']):
                return None
            local_from_lidar = np.asarray(record['localization']['world_from_lidar'])
            map_from_local = validate_transform(prior['map_from_local'])
            local_base = self.geometry.pose(local_from_lidar)
            map_base = self.geometry.pose(map_from_local @ local_from_lidar)
            with np.errstate(invalid='ignore'):
                grid = navigation_geometry(scene, now=cycle_stamp)
            validate_grid(grid)
            binding = {k: prior[k] for k in MAP_BINDING_FIELDS}
        except (KeyError, OSError, TypeError, ValueError):
            return None
        return record['as_of'], identity, binding, local_base, map_base, grid

    def sample(self, t, pose, map_id, world_xy, *, not_before):
        pose_map = pose.get('map_binding') or {}
        if not map_id or not pose_map.get('valid') or pose_map.get('map_id') != map_id:
            return None
        index = bisect_right(self.times, t) - 1
        if index < 0:
            return None
        cycle_stamp, path = self.items[index]
        if path is None or cycle_stamp < not_before or t - cycle_stamp > MAX_SNAPSHOT_AGE_S:
            return None
        snapshot = self._recorded_grid(cycle_stamp, path)
        if snapshot is None:
            return None
        stamp, identity, binding, local_base, map_base, grid = snapshot
        if (stamp < not_before or not 0 <= t - stamp <= MAX_SNAPSHOT_AGE_S
                or identity != pose.get('local_identity')
                or binding != {k: pose_map.get(k) for k in MAP_BINDING_FIELDS}):
            return None

        # Match the planning replay's map-to-local projection. The display
        # points come from the current pose; the grid stays in its recorded frame.
        turn = local_base[2] - map_base[2]
        c, s = math.cos(turn), math.sin(turn)
        local_xy = (world_xy - map_base[:2]) @ np.array(((c, s), (-s, c))) + local_base[:2]
        coordinates = (local_xy - grid.metadata['origin_xy']) / grid.metadata['resolution'] - .5
        return cv2.remap(grid.layers['occupancy'], coordinates[..., 0].astype(np.float32),
                         coordinates[..., 1].astype(np.float32), cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=-2)
