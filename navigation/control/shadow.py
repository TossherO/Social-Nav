"""Shared shadow tracking helpers; observations are not execution feedback."""
import copy
from pathlib import Path
import time
from types import SimpleNamespace
import uuid
import numpy as np
import yaml
from perception.environment.context import fingerprint
from perception.mapping_localization.context import pose_matrix
from navigation.planning.motion import estimate_motion,identity
from .tracking import goal_feedback

TOPICS=dict(observation='/navigation/shadow/observation',proposal='/navigation/shadow/proposal',
    state='/navigation/shadow/state',velocity='/navigation/shadow/velocity',override='/navigation/shadow/override')


def load_config(path=None):
    cfg=yaml.safe_load(Path(path or Path(__file__).with_name('shadow.yaml')).read_text())
    if any(not np.isfinite(v) or v<=0 for v in cfg.values()) or not .01<=cfg['period_s']<=.2:
        raise ValueError('Invalid shadow timing')
    if type(cfg['inbox_size']) is not int or not 2<=cfg['inbox_size']<=64:raise ValueError('Invalid shadow inbox')
    return cfg


class ClockWatch:
    """Monotonic watchdog independent of paused ROS time or incoming heartbeats."""
    def __init__(self,timeout):self.timeout=timeout;self.stamp=self.advanced=None

    def update(self,stamp,mono):
        if not np.isfinite(stamp) or not np.isfinite(mono):return 'clock_invalid'
        if self.stamp is not None and stamp<self.stamp-1e-6:
            self.stamp,self.advanced=stamp,mono;return 'clock_reversed'
        if self.stamp is None or stamp>self.stamp+1e-9:self.stamp,self.advanced=stamp,mono
        return 'clock_stalled' if mono-self.advanced>self.timeout else None


def observed_feedback(rows,scene,geometry,motion,*,now):
    """Latest causal algorithm-pose estimate; never raw ROS odometry or cmd_vel."""
    out=dict(status='unavailable',source='causal_algorithm_poses_not_chassis_feedback',
        stamp_s=None,pose=None,velocity=None,clock_domain=scene.record['clock_domain'],
        local_identity=identity(scene),execution_feedback=False)
    past=[r for r in rows if r['stamp_s']<=now]
    if not past:return dict(out,reason='no_past_localization')
    latest=past[-1];stamp=latest['stamp_s']
    if (latest['health']!=0 or [latest[k] for k in ('session_id','generation','frame_id')]!=identity(scene) or
            not 0<=now-stamp<=motion['state_max_age_s']):return dict(out,reason='localization_invalid_or_stale')
    # This is a motion-estimation context, not a fabricated grid observation.
    context=SimpleNamespace(record={**scene.record,'as_of':stamp})
    estimate=estimate_motion(past,context,geometry,motion)
    if estimate['status']!='valid':return dict(out,reason=estimate.get('reason','motion_unavailable'))
    if abs(estimate['velocity'][1])>motion['max_initial_lateral_mps']:
        return dict(out,reason='lateral_motion_outside_model')
    pose=geometry.pose(pose_matrix(latest['position'],latest['quaternion_xyzw']))
    result=dict(out,status='valid',reason=None,stamp_s=stamp,pose=pose.tolist(),
        velocity=[estimate['velocity'][0],estimate['velocity'][2]],source_binding=estimate['input_fingerprint'])
    if 'stationary_evidence' in estimate:
        result.update(stationary_evidence=estimate['stationary_evidence'],velocity_raw=np.asarray(estimate['velocity_raw'])[[0,2]].tolist())
    return result


class _ShadowOutput:
    """Source and clock envelope for non-executing DWA recommendations."""
    def __init__(self,motion,geometry,asset,settings):
        self.geometry,self.motion,self.asset=geometry,motion,asset;self.settings=settings or load_config()
        self.clock=ClockWatch(self.settings['clock_stall_s']);self.producer=uuid.uuid4().hex;self.sequence=0

    def _output(self,request,scene,began,*,now,clock_domain,**fields):
        self.sequence+=1
        out=dict(producer_id=self.producer,sequence_id=self.sequence,stamp_s=now,clock_domain=clock_domain,
            scene_binding=None if scene is None else scene.record['input_fingerprint'],request=copy.deepcopy(request),
            **fields,compute_ms=(time.perf_counter()-began)*1000,
            scope='shadow_recommendation_not_execution',executable=False,task_completed=False)
        out['packet_binding']=fingerprint(out)
        return out


class DwaShadowController(_ShadowOutput):
    """Direct bounded velocity selection, sharing the existing private wire envelope."""
    def __init__(self,planning,motion,geometry,*,asset=None,settings=None,dwa_settings=None,device_batch=None,execution=False):
        from .dwa import DwaController
        super().__init__(motion,geometry,asset,settings)
        self.controller=DwaController(planning,motion,geometry,asset=asset,settings=dwa_settings,device_batch=device_batch,execution=execution)
        self.last_offered=None

    def accept(self,result,request,scene,*,now,clock_domain):
        if result is None:return False
        binding=result['binding'];changed=binding!=self.last_offered;self.last_offered=binding
        return changed

    def step(self,request,scene,rows,result,*,now,clock_domain,mono=None,reason=None,dt=None):
        mono=time.monotonic() if mono is None else mono;began=time.perf_counter()
        reason=reason or self.clock.update(now,mono)
        if request is None:reason=reason or 'no_active_goal'
        if scene is None:reason=reason or 'observation_unavailable'
        proposal=feedback=diagnostic=None
        if reason:self.controller.clear()
        else:
            feedback=observed_feedback(rows,scene,self.geometry,self.motion,now=now)
            proposal=self.controller.command(request,scene,feedback,result,now=now,clock_domain=clock_domain,dt=self.settings['period_s'] if dt is None else dt)
            diagnostic=goal_feedback(request,scene,feedback,self.controller.goal_settings,now=now,clock_domain=clock_domain,
                active=True,geometry=self.geometry,asset=self.asset,planning=self.controller.planning)
        return self._output(request,scene,began,now=now,clock_domain=clock_domain,
            proposal=proposal,feedback=feedback,reason=reason,result_accepted=self.controller.result is not None,
            recorded_goal_diagnostic=diagnostic)


def publisher(rospy,topic,message_type,**options):
    """Fixed private outputs; refuse remapping a shadow stream to robot topics."""
    if rospy.resolve_name(topic)!=topic:raise ValueError('Shadow output remapping is not allowed: '+topic)
    return rospy.Publisher(topic,message_type,**options)
