"""Small interactive-session state beside the existing command and control owners."""
import copy
import json
import os
import socket
import threading
import time
import math
import uuid


class InteractiveSession:
    def __init__(self, *, live, safety, output, execute=False, bag_shadow=None):
        self.live, self.safety, self.output, self.execute = live, safety, output, execute
        self.bag_shadow = bag_shadow
        self.lock = threading.RLock()
        self.parent = self.child = None
        if not live:
            self.parent, self.child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.pause_requested = self.paused = self.source_ended = False
        self.epoch = 0
        self.command_generation = 0
        self.preview = self.observation = None
        self.review_source = None
        self.review_displayed = None
        self.goal_intent = None
        self.offset_s = self.stamp_s = None
        self.playback_epoch = None

    def note_command(self, kind, value):
        with self.lock:
            if self.bag_shadow is not None:
                self.bag_shadow.toggle(False, reason='manual_override')
                self.goal_intent = None
            self.command_generation += 1
            self.preview = None

    def handle(self, kind, value):
        if kind == 'save_anomaly_review':
            return self.save_anomaly_review(value)
        if kind == 'shadow_navigation':
            if self.live or self.execute or self.bag_shadow is None:
                raise ValueError('Recorded-route planning requires a bag without hardware output')
            if set(value) != {'enabled'} or type(value['enabled']) is not bool:
                raise ValueError('Shadow navigation requires a Boolean enabled value')
            with self.lock, self.bag_shadow.lock:
                if self.bag_shadow.enabled == value['enabled']:
                    return dict(accepted=True, changed=False, **self.bag_shadow.status())
                self.bag_shadow.toggle(value['enabled'])
                self.preview = None
                self.goal_intent = None
                self.command_generation += 1
                return dict(accepted=True, changed=True, **self.bag_shadow.status())
        if kind == 'session_status':
            if value != {}:
                raise ValueError('Session status takes an empty object')
            return self.status()
        if kind == 'playback':
            if self.live or set(value) != {'paused'} or type(value['paused']) is not bool:
                raise ValueError('Playback pause is available only for a bag')
            with self.lock:
                if self.source_ended and not value['paused']:
                    raise ValueError('End of bag: restart to seek to another interval')
                self.pause_requested = value['paused']
                self.review_source = None
                self.review_displayed = None
                self.preview = None
                self.command_generation += 1
                self.parent.send(json.dumps(value).encode())
            return dict(accepted=True, state='pausing' if value['paused'] else 'resuming', executable=False)
        if kind == 'arm':
            if not self.live or not self.execute:
                raise ValueError('This session has no enabled hardware output')
            if set(value) != {'enabled'} or type(value['enabled']) is not bool:
                raise ValueError('Arming requires a Boolean enabled value')
            self.safety.command('arm', value)
            return dict(accepted=True, execution_feedback=False)
        raise ValueError('Unknown interactive session command')

    def update_playback(self, packet):
        with self.lock:
            if packet['epoch'] != self.playback_epoch or packet['paused'] != self.paused:
                self.epoch += 1
                self.review_source = None
                self.review_displayed = None
                self.preview = None
                self.goal_intent = None
                self.command_generation += 1
            self.playback_epoch = packet['epoch']
            self.paused = self.pause_requested = packet['paused']
            self.source_ended = packet['source_ended']
            self.offset_s, self.stamp_s = packet['offset_s'], packet['stamp_s']

    def publish_observation(self, scene, row, *, goal_intent=None, token=None):
        with self.lock:
            if token is None or token == self.command_generation:
                self.goal_intent = copy.deepcopy(goal_intent)
            self.observation = dict(localization=copy.deepcopy(row),
                prior=None if scene is None else copy.deepcopy(scene.record['prior']),
                scene_stamp_s=None if scene is None else scene.record['as_of'])

    def freeze_review(self, scene, device, row, *, epoch, max_grid_age_s, max_association_age_s):
        """Pin the first eligible scene for this pause, independently of display sampling."""
        with self.lock:
            if self.live or not self.paused or not self.pause_requested or epoch!=self.epoch:
                return None
            if self.review_source is None and scene is not None and device is not None:
                meta=scene.geometry.metadata;prior=scene.record['prior'];stamp=scene.record['as_of']
                measured=prior.get('measurement_stamp_s')
                if (self.stamp_s is not None and stamp<=self.stamp_s and 0<=row['stamp_s']-stamp<=max_grid_age_s
                        and row['health']==0 and row.get('has_map_pose') and prior.get('valid')
                        and row.get('map_frame')==prior.get('map_frame') and row.get('map_revision')==prior.get('map_revision')
                        and measured is not None and 0<=row['stamp_s']-measured<=max_association_age_s
                        and all(meta[k]==row[k] for k in ('session_id','generation','frame_id'))
                        and device.metadata['snapshot_id']==meta['snapshot_id']):
                    self.review_source=(scene,device,copy.deepcopy(row))
            return self.review_source

    def confirm_review_display(self, epoch, snapshot_id):
        with self.lock:
            if (self.paused and self.pause_requested and epoch==self.epoch and self.review_source is not None
                    and self.review_source[0].geometry.metadata['snapshot_id']==snapshot_id):
                self.review_displayed=snapshot_id

    def save_anomaly_review(self, value):
        from .export import materialize_scene

        with self.lock:
            if self.live or not self.paused or not self.pause_requested or self.review_displayed is None:
                raise ValueError('请暂停回放并等待局部地图就绪。')
            if (not isinstance(value,dict) or value.get('epoch')!=self.epoch
                    or value.get('snapshot_id')!=self.review_displayed):
                raise ValueError('暂停画面已变化，请重新圈定区域。')
            scene,device,_ = self.review_source
            if self.stamp_s is None or scene.record['as_of']>self.stamp_s:
                raise ValueError('暂停时刻与局部地图时间尚未对齐。')
            prior = scene.record['prior']
            if not prior.get('valid') or value.get('frame')!=prior.get('map_frame'):
                raise ValueError('请在定位已就绪的地图坐标系中圈定区域。')
            corners = value.get('corners')
            if (not isinstance(corners,list) or len(corners)!=2 or any(not isinstance(p,list) or len(p)!=3
                    or any(type(n) not in (int,float) or not math.isfinite(n) for n in p) for p in corners)):
                raise ValueError('请选择两个有效的矩形对角点。')
            bounds = [[min(a,b),max(a,b)] for a,b in zip(*corners)]
            if any(low==high for low,high in bounds[:2]):
                raise ValueError('矩形需要有宽度和高度，请选择另一个对角点。')
            evidence = dict(frame=value['frame'], corners_xyz=corners,
                            bounds_xy=[bounds[0][0],bounds[1][0],bounds[0][1],bounds[1][1]],
                            playback_offset_s=self.offset_s, playback_stamp_s=self.stamp_s,
                            map_stamp_s=scene.record['as_of'], snapshot_id=scene.geometry.metadata['snapshot_id'])
        bundle = materialize_scene(device,scene,consumer='anomaly_review')
        directory = self.output/'anomaly_reviews'/uuid.uuid4().hex[:12]
        bundle.save(directory/'scene')
        evidence['scene_binding'] = bundle.record['input_fingerprint']
        (directory/'review.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        return dict(accepted=True, path=str(directory), **evidence)

    def clear_preview(self, token):
        with self.lock:
            if token == self.command_generation:
                self.preview = None

    def publish_preview(self, value, token):
        with self.lock:
            if token != self.command_generation or not self.paused:
                return False
            self.preview = copy.deepcopy(value)
            # One current result; immutable source scenes stay in the runtime.
            (self.output/'preview.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            return True

    def status(self):
        with self.lock:
            chassis=None
            if self.execute:
                try:chassis=json.loads((self.output/'chassis/status.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):pass
                if chassis is not None:
                    stamp=chassis.get('status_mono')
                    age=None if type(stamp) not in (int,float) else time.monotonic()-stamp
                    chassis['status_age_s']=age
                    chassis['status_fresh']=age is not None and 0<=age<=.5
                    if not chassis['status_fresh']:
                        chassis.update(reported_state=chassis.get('state'),state='unavailable',reason='chassis_status_stale')
            return dict(live=self.live, hardware_enabled=self.execute,
                        ros_master_uri=os.environ.get('ROS_MASTER_URI','http://localhost:11311'),
                        chassis=chassis,
                        paused=self.paused, pause_requested=self.pause_requested,
                        review_available=self.review_displayed is not None and self.paused and self.pause_requested,
                        review_snapshot_id=self.review_displayed,
                        source_ended=self.source_ended, epoch=self.epoch,
                        stamp_s=self.stamp_s, offset_s=self.offset_s,
                        observation=copy.deepcopy(self.observation), preview=copy.deepcopy(self.preview),
                        goal_intent=copy.deepcopy(self.goal_intent),
                        shadow_navigation=(dict(available=False,enabled=False,state='unavailable')
                                           if self.bag_shadow is None else self.bag_shadow.status()),
                        executable=False, execution_feedback=False)

    def close(self):
        for channel in (self.parent, self.child):
            if channel is not None:
                channel.close()
