"""Bounded measured map poses; never includes planned or interpolated motion."""
import copy
import math
from perception.environment.context import fingerprint


class Trajectory:
    def __init__(self, *, capacity=2048):
        self.segments = []
        self.capacity = capacity
        self.interrupted = True
        self.sample_interval = .1
        self.truncated = False
        self.last_observation_stamp = None

    def update(self, state, frame_id, *, map_snapshot_id=None, localization_map_id=None):
        if map_snapshot_id is not None:
            binding = state.get('map_binding') or {}
            map_id = binding.get('map_id')
            if not (binding.get('valid') and map_id and
                    (map_id == localization_map_id if localization_map_id else
                     map_snapshot_id==map_id or map_snapshot_id.startswith(map_id+':scan_fusion:'))):
                self.segments = []
                self.sample_interval = .1
                self.truncated = False
                self.last_observation_stamp = None
                self.interrupted = True
                return
        pose = state.get('origin_pose') if state.get('origin_frame_id') == frame_id else (
            state.get('pose') if state.get('frame_id') == frame_id else None)
        stamp = state.get('origin_stamp_s') if state.get('origin_frame_id')==frame_id else state.get('stamp_s')
        valid = (state.get('localization_status', 'valid') == 'valid' and not state.get('input_stale') and
                 pose is not None and len(pose) == 3 and all(math.isfinite(v) for v in pose) and
                 isinstance(stamp, (float, int)) and math.isfinite(stamp) and
                 0 <= state['stamp_s']-stamp <= .5)
        if not valid:
            self.interrupted = True
            self.last_observation_stamp = None
            return
        identity = dict(frame_id=frame_id, clock_domain=state['clock_domain'],
                        local_identity=[state.get(k) for k in ('session_id', 'generation', 'frame_id')],
                        map_binding=copy.deepcopy(state.get('map_binding')))
        binding = fingerprint(identity)
        previous = self.segments[-1] if self.segments else None
        if previous and any(previous[k]!=identity[k] for k in ('frame_id','clock_domain','map_binding')):
            self.segments = []
            self.sample_interval = .1
            self.truncated = False
            self.last_observation_stamp = None
            previous = None
        if previous and previous['binding'] != binding:
            self.last_observation_stamp = None
        if self.last_observation_stamp is not None and stamp <= self.last_observation_stamp:
            return
        gap = self.last_observation_stamp is not None and stamp-self.last_observation_stamp > 3
        self.last_observation_stamp=stamp
        if (self.interrupted or previous is None or previous['binding'] != binding or
                gap):
            self.segments.append(dict(identity, binding=binding, samples=[]))
        self.interrupted = False
        samples = self.segments[-1]['samples']
        if samples and stamp-samples[-1]['stamp_s'] < self.sample_interval:
            return
        samples.append(dict(stamp_s=stamp, pose=list(pose), source=state.get('source')))
        while sum(len(s['samples']) for s in self.segments) > self.capacity:
            before=sum(len(s['samples']) for s in self.segments)
            self.sample_interval *= 2
            for segment in self.segments:
                original=segment['samples']
                reduced=original[:1]
                for sample in original[1:-1]:
                    if sample['stamp_s']-reduced[-1]['stamp_s']>=self.sample_interval:
                        reduced.append(sample)
                if len(original)>1:reduced.append(original[-1])
                segment['samples']=reduced
            if sum(len(s['samples']) for s in self.segments)>=before:
                self.segments[0]['samples'].pop(0)
                self.truncated=True
                if not self.segments[0]['samples']:self.segments.pop(0)

    def snapshot(self, *, frame_id, clock_domain, now):
        result = []
        for segment in self.segments:
            if segment['frame_id']!=frame_id or segment['clock_domain']!=clock_domain:
                continue
            samples = [copy.deepcopy(p) for p in segment['samples'] if p['stamp_s']<=now]
            if samples:
                result.append(dict(copy.deepcopy(segment),samples=samples))
        return result
