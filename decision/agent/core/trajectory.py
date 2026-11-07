"""Bounded measured map poses; never includes planned or interpolated motion."""
import copy
import math
from perception.environment.context import fingerprint


class Trajectory:
    def __init__(self, *, capacity=2048):
        self.segments = []
        self.capacity = capacity
        self.interrupted = True

    def update(self, state, frame_id, *, map_snapshot_id=None):
        if map_snapshot_id is not None:
            binding = state.get('map_binding') or {}
            map_id = binding.get('map_id')
            if not (binding.get('valid') and map_id and
                    (map_snapshot_id==map_id or map_snapshot_id.startswith(map_id+':scan_fusion:'))):
                self.segments = []
                self.interrupted = True
                return
        pose = state.get('origin_pose') if state.get('origin_frame_id') == frame_id else (
            state.get('pose') if state.get('frame_id') == frame_id else None)
        stamp = state.get('origin_stamp_s', state.get('stamp_s'))
        valid = (state.get('localization_status', 'valid') == 'valid' and not state.get('input_stale') and
                 pose is not None and len(pose) == 3 and all(math.isfinite(v) for v in pose) and
                 isinstance(stamp, (float, int)) and math.isfinite(stamp) and
                 0 <= state['stamp_s']-stamp <= .5)
        if not valid:
            self.interrupted = True
            return
        identity = dict(frame_id=frame_id, clock_domain=state['clock_domain'],
                        local_identity=[state.get(k) for k in ('session_id', 'generation', 'frame_id')],
                        map_binding=copy.deepcopy(state.get('map_binding')))
        binding = fingerprint(identity)
        previous = self.segments[-1] if self.segments else None
        if previous and any(previous[k]!=identity[k] for k in ('frame_id','clock_domain','map_binding')):
            self.segments = []
            previous = None
        if previous and previous['binding'] == binding and stamp <= previous['samples'][-1]['stamp_s']:
            return
        if (self.interrupted or previous is None or previous['binding'] != binding or
                stamp-previous['samples'][-1]['stamp_s'] > 3):
            self.segments.append(dict(identity, binding=binding, samples=[]))
        self.interrupted = False
        samples = self.segments[-1]['samples']
        if samples and stamp-samples[-1]['stamp_s'] < .1:
            return
        samples.append(dict(stamp_s=stamp, pose=list(pose), source=state.get('source')))
        while sum(len(s['samples']) for s in self.segments) > self.capacity:
            self.segments[0]['samples'].pop(0)
            if not self.segments[0]['samples']:
                self.segments.pop(0)

    def snapshot(self, *, frame_id, clock_domain, now):
        result = []
        for segment in self.segments:
            if segment['frame_id']!=frame_id or segment['clock_domain']!=clock_domain:
                continue
            samples = [copy.deepcopy(p) for p in segment['samples'] if p['stamp_s']<=now]
            if samples:
                result.append(dict(copy.deepcopy(segment),samples=samples))
        return result
