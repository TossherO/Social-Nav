"""Task-scoped navigation observations; events never command motion."""
import copy
import math

from .tasks import inside


class TaskMonitor:
    def __init__(self, catalog):
        self.catalog = catalog
        self.settings = None
        self.reset_runtime()

    def configure(self, goal_id, focus, interval_s=None, region_ids=None, progress_interval_s=None):
        regions = list(region_ids or [])
        if len(regions) != len(set(regions)) or any(region not in self.catalog.regions for region in regions):
            raise ValueError('Monitor regions must be distinct and present in the current catalog')
        if interval_s is None and not regions and progress_interval_s is None:
            raise ValueError('Monitor requires a review interval, region or progress interval')
        for value in (interval_s, progress_interval_s):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Monitor intervals must be finite and positive')
        self.settings = dict(goal_id=goal_id, focus=focus, interval_s=interval_s,
                             region_ids=regions, progress_interval_s=progress_interval_s)
        self.reset_runtime()

    def reset_runtime(self):
        self.handle = None
        self.last_review = None
        self.regions = {}
        self.progress = None
        self.pending = {}
        self.last_pose_stamp = None

    def snapshot(self):
        return copy.deepcopy(self.settings)

    def update(self, navigation, feedback):
        handle = navigation['handle']
        if (self.settings is None or handle is None or navigation['status'] not in ('running', 'blocked', 'shadow_proposed') or
                navigation['goal_id'] != self.settings['goal_id'] or navigation['context_lost'] or not navigation['connected']):
            self.reset_runtime()
            return
        if feedback.get('handle') != handle:
            self.reset_runtime()
            return
        if handle != self.handle:
            self.reset_runtime()
            self.handle = copy.deepcopy(handle)
        now = feedback.get('stamp_s')
        if (type(now) not in (int, float) or not math.isfinite(now) or feedback.get('input_stale') or
                feedback.get('localization_status', 'valid') != 'valid'):
            self.pending = {}
            self.last_review = None
            self.regions = {}
            self.progress = None
            self.last_pose_stamp = None
            return
        if self.last_review is None:
            self.last_review = now
        elif now < self.last_review:
            self.reset_runtime()
            return
        elif self.settings['interval_s'] is not None and now-self.last_review >= self.settings['interval_s']:
            self.pending['semantic_review_due'] = dict(kind='semantic_review_due', stamp_s=now)
            self.last_review = now

        frame = self.catalog.value['frame_id']
        pose = feedback.get('origin_pose') if feedback.get('origin_frame_id') == frame else (
            feedback.get('pose') if feedback.get('frame_id') == frame else None)
        stamp = feedback.get('origin_stamp_s', now) if feedback.get('origin_frame_id') == frame else now
        if (not isinstance(pose, (list, tuple)) or len(pose) != 3 or
                not all(type(v) in (int, float) and math.isfinite(v) for v in pose) or
                type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now-stamp <= .5):
            self.regions = {}
            self.progress = None
            self.last_pose_stamp = None
            self.pending = {key:value for key,value in self.pending.items() if not key.startswith('region:')
                            and key != 'no_measured_progress'}
            return
        if self.last_pose_stamp is not None and stamp <= self.last_pose_stamp:
            return
        self.last_pose_stamp = stamp
        for region in self.settings['region_ids']:
            is_inside = inside(pose[:2], self.catalog.regions[region]['polygon'])
            if is_inside and self.regions.get(region) is not True:
                self.pending['region:'+region] = dict(kind='region_entered', region_id=region, stamp_s=stamp)
            self.regions[region] = is_inside

        interval = self.settings['progress_interval_s']
        if interval is None:
            return
        route_index = feedback.get('route_index')
        previous = self.progress
        if previous is None or stamp < previous['last_stamp'] or stamp-previous['last_stamp'] > 2:
            self.progress = dict(start=stamp, last_stamp=stamp, pose=list(pose), route_index=route_index)
            return
        if stamp == previous['last_stamp']:
            return
        angle = abs(math.atan2(math.sin(pose[2]-previous['pose'][2]), math.cos(pose[2]-previous['pose'][2])))
        moved = math.hypot(pose[0]-previous['pose'][0], pose[1]-previous['pose'][1]) >= .1 or angle >= .15
        route_advanced = route_index is not None and previous['route_index'] is not None and route_index != previous['route_index']
        if moved or route_advanced:
            self.pending.pop('no_measured_progress', None)
            self.progress = dict(start=stamp, last_stamp=stamp, pose=list(pose), route_index=route_index)
        else:
            previous['last_stamp'] = stamp
            if stamp-previous['start'] >= interval:
                self.pending['no_measured_progress'] = dict(kind='no_measured_progress', stamp_s=stamp,
                    duration_s=stamp-previous['start'], planning_status=feedback.get('planning_status'),
                    planning_reason=feedback.get('planning_reason'))
                previous['start'] = stamp

    def take(self):
        if not self.pending:
            return None
        event = dict(goal_id=self.settings['goal_id'], focus=self.settings['focus'],
                     handle=copy.deepcopy(self.handle), reasons=list(self.pending.values()))
        self.pending = {}
        return event
