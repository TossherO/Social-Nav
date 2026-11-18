"""Task-scoped navigation observations; events never command motion."""
import copy
import math

from .tasks import inside


class TaskMonitor:
    def __init__(self, catalog):
        self.catalog = catalog
        self.settings = None
        self.last_trigger = None
        self.last_review = None
        self.reset_runtime()

    def configure(self, goal_id, focus, interval_s=None, region_ids=None, progress_interval_s=None,
                  distance_m=None, rotation_rad=None, stopped_s=None, timeout_s=None):
        regions = list(region_ids or [])
        if len(regions) != len(set(regions)) or any(region not in self.catalog.regions for region in regions):
            raise ValueError('Monitor regions must be distinct and present in the current catalog')
        if all(v is None for v in (interval_s,progress_interval_s,distance_m,rotation_rad,stopped_s,timeout_s)) and not regions:
            raise ValueError('Monitor requires a review interval, region or progress interval')
        for value in (interval_s, progress_interval_s, distance_m, rotation_rad, stopped_s, timeout_s):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Monitor intervals must be finite and positive')
        self.settings = dict(goal_id=goal_id, focus=focus, interval_s=interval_s,
                             region_ids=regions, progress_interval_s=progress_interval_s,
                             distance_m=distance_m, rotation_rad=rotation_rad, stopped_s=stopped_s, timeout_s=timeout_s)
        self.reset_runtime()

    def watch(self, goal_id, focus):
        """Every stage receives navigation-event review; other triggers are opt-in."""
        self.settings=dict(goal_id=goal_id,focus=focus,interval_s=None,region_ids=[],
            progress_interval_s=None,distance_m=None,rotation_rad=None,stopped_s=None,timeout_s=None)
        self.reset_runtime()

    def reset_runtime(self):
        self.handle = None
        self.last_review_at = None
        self.regions = {}
        self.progress = None
        self.pending = {}
        self.last_pose_stamp = None
        self.travel = None
        self.stop_since = None
        self.stop_notified = False
        self.started_at = None
        self.timeout_notified = False

    def snapshot(self):
        if self.settings is None and self.last_trigger is None and self.last_review is None:
            return None
        value=copy.deepcopy(self.settings) if self.settings is not None else {}
        value.update(last_trigger=copy.deepcopy(self.last_trigger),last_review=copy.deepcopy(self.last_review))
        return value

    def note_trigger(self, event):
        self.last_trigger=copy.deepcopy(event)

    def note_review(self, goal_id, result, evidence, *, source_stamp_s=None):
        self.last_review=dict(goal_id=goal_id,result=result,evidence=evidence,source_stamp_s=source_stamp_s)

    def update_idle(self, goal_id, feedback):
        """Timed review while an observation stage waits without a navigation handle."""
        if self.settings is None or self.settings['goal_id'] != goal_id:return
        now=feedback.get('stamp_s')
        if (type(now) not in (int,float) or not math.isfinite(now) or feedback.get('input_stale') or
                feedback.get('localization_status','valid')!='valid'):
            self.last_review_at=self.started_at=None
            return
        if self.last_review_at is None:self.last_review_at=now
        if self.started_at is None:self.started_at=now
        if now < self.last_review_at or now < self.started_at:
            self.last_review_at=self.started_at=now
            return
        interval=self.settings['interval_s']
        if interval is not None and now-self.last_review_at>=interval:
            self.pending['semantic_review_due']=dict(kind='semantic_review_due',stamp_s=now)
            self.last_review_at=now
        limit=self.settings['timeout_s']
        if limit is not None and not self.timeout_notified and now-self.started_at>=limit:
            self.pending['timeout']=dict(kind='monitor_timeout',stamp_s=now,duration_s=now-self.started_at)
            self.timeout_notified=True

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
            self.last_review_at = None
            self.regions = {}
            self.progress = None
            self.last_pose_stamp = None
            self.travel = None
            self.stop_since = None
            self.stop_notified = False
            return
        if self.last_review_at is None:
            self.last_review_at = now
        elif now < self.last_review_at:
            self.reset_runtime()
            return
        elif self.settings['interval_s'] is not None and now-self.last_review_at >= self.settings['interval_s']:
            self.pending['semantic_review_due'] = dict(kind='semantic_review_due', stamp_s=now)
            self.last_review_at = now
        if self.started_at is None:self.started_at=now
        if (self.settings['timeout_s'] is not None and not self.timeout_notified and
                now-self.started_at >= self.settings['timeout_s']):
            self.pending['timeout'] = dict(kind='monitor_timeout',stamp_s=now,duration_s=now-self.started_at)
            self.timeout_notified=True

        frame = self.catalog.value['frame_id']
        pose = feedback.get('origin_pose') if feedback.get('origin_frame_id') == frame else (
            feedback.get('pose') if feedback.get('frame_id') == frame else None)
        stamp = feedback.get('origin_stamp_s', now) if feedback.get('origin_frame_id') == frame else now
        if (not isinstance(pose, (list, tuple)) or len(pose) != 3 or
                not all(type(v) in (int, float) and math.isfinite(v) for v in pose) or
                type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= now-stamp <= .5):
            self.regions = {}
            self.progress = None
            self.travel = None
            self.stop_since = None
            self.stop_notified = False
            self.last_pose_stamp = None
            self.pending = {key:value for key,value in self.pending.items() if not key.startswith('region:')
                            and key != 'no_measured_progress'}
            return
        if self.last_pose_stamp is not None and stamp <= self.last_pose_stamp:
            return
        if self.last_pose_stamp is not None and stamp-self.last_pose_stamp > .5:
            self.stop_since=None;self.stop_notified=False
        self.last_pose_stamp = stamp
        if self.travel is None or stamp-self.travel['stamp_s'] > 2:
            self.travel=dict(pose=list(pose),stamp_s=stamp,distance=0.,rotation=0.)
        else:
            t=self.travel
            t['distance']+=math.hypot(pose[0]-t['pose'][0],pose[1]-t['pose'][1])
            t['rotation']+=abs(math.atan2(math.sin(pose[2]-t['pose'][2]),math.cos(pose[2]-t['pose'][2])))
            t.update(pose=list(pose),stamp_s=stamp)
            for key,field,kind in (('distance_m','distance','distance_progress'),('rotation_rad','rotation','rotation_progress')):
                limit=self.settings[key]
                if limit is not None and t[field]>=limit:
                    self.pending[kind]=dict(kind=kind,stamp_s=stamp,measured=t[field],threshold=limit)
                    t[field]=0.
        if feedback.get('stopped') is True:
            if self.stop_since is None:self.stop_since=stamp
            if (self.settings['stopped_s'] is not None and not self.stop_notified and
                    stamp-self.stop_since>=self.settings['stopped_s']):
                self.pending['stopped']=dict(kind='stopped_duration',stamp_s=stamp,duration_s=stamp-self.stop_since)
                self.stop_notified=True
        else:
            self.stop_since=None;self.stop_notified=False
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
        self.note_trigger(event)
        return event
