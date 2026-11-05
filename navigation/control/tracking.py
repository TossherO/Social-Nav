"""Feedback validity and measured goal completion for the DWA controller."""
from pathlib import Path
import numpy as np
import yaml
from perception.environment.context import fingerprint
from navigation.planning.continuity import context_binding,local_goal
from navigation.planning.footprint import wrap
from navigation.planning.inputs import request_status


def load_config(path=None):
    cfg=yaml.safe_load(Path(path or Path(__file__).with_name('config.yaml')).read_text())
    if any(not np.isfinite(v) or v<=0 for v in cfg.values()):raise ValueError('Invalid tracking setting')
    return cfg


def feedback_valid(feedback,scene,now,clock,max_age):
    if feedback is None or feedback.get('status')!='valid' or not feedback.get('source'):return False
    r=scene.record
    try:
        return (feedback.get('clock_domain')==clock==r['clock_domain'] and
            feedback.get('local_identity')==[r['session_id'],r['generation'],r['frame_id']] and
            np.shape(feedback.get('pose'))==(3,) and np.shape(feedback.get('velocity'))==(2,) and
            np.isfinite(feedback['pose']).all() and np.isfinite(feedback['velocity']).all() and
            0<=now-feedback.get('stamp_s',-np.inf)<=max_age)
    except (ValueError,TypeError):return False


def goal_feedback(request,scene,feedback,settings,*,now,clock_domain,active,geometry=None,asset=None,planning=None):
    result=dict(status='unavailable',task_id=request['task_id'],goal_revision=request['goal_revision'],
        request_binding=fingerprint(request),task_completed=False,source=(feedback or {}).get('source'),executable=False)
    if not active:return dict(result,status='inactive')
    if request_status(request,scene,now,clock_domain):return result
    if not feedback_valid(feedback,scene,now,clock_domain,settings['feedback_max_age_s']):return result
    if (scene.record['localization']['status']!='valid' or scene.record['geometry']['status']!='valid' or
            not 0<=now-scene.record['as_of']<=settings['observation_max_age_s']):return result
    try:goal=local_goal(request,scene,geometry,asset,planning)
    except (ValueError,KeyError,TypeError,AttributeError):return result
    pose=np.asarray(feedback['pose']);velocity=feedback['velocity']
    reached=np.linalg.norm(goal[:2]-pose[:2])<=settings['goal_position_tolerance_m']
    aligned=len(goal)==2 or abs(wrap(goal[2]-pose[2]))<=settings['goal_yaw_tolerance_rad']
    stopped=abs(velocity[0])<=settings['stopped_linear_mps'] and abs(velocity[1])<=settings['stopped_angular_rps']
    complete=bool(reached and aligned and (stopped or request.get('goal_behavior','pass_through')!='stop'))
    if complete:
        try:
            scene.verify()
            if geometry is not None:
                from navigation.planning.footprint import footprint_status
                from perception.environment.navigation import navigation_geometry
                if footprint_status(navigation_geometry(scene,now=now,asset=asset,check_asset=True),geometry,pose,(planning or {}).get('safety_margin_m',.05))!='free':return result
        except (ValueError,KeyError,TypeError):return result
    return dict(result,status='criteria_met' if complete else 'running',criteria_met=complete,
        reached_xy=bool(reached),yaw_aligned=bool(aligned),stopped=bool(stopped),feedback_stamp_s=feedback['stamp_s'])


class GoalMonitor:
    """Arrival requires active, fresh feedback to hold criteria within one lifecycle."""
    def __init__(self,settings=None,*,geometry=None,asset=None,planning=None):
        self.settings=settings or load_config();self.geometry=geometry;self.asset=asset;self.key=None;self.since=None;self.last=None
        self.planning=planning

    def update(self,request,scene,feedback,*,now,clock_domain,active):
        r=goal_feedback(request,scene,feedback,self.settings,now=now,clock_domain=clock_domain,active=active,
            geometry=self.geometry,asset=self.asset,planning=self.planning)
        if not r.get('criteria_met'):
            self.key=self.since=self.last=None
            return r
        key=(context_binding(request,scene,self.asset),clock_domain)
        stamp=None if feedback is None else feedback.get('stamp_s')
        if self.key!=key or (self.last is not None and stamp is not None and stamp<self.last):self.since=None
        self.key=key;self.last=stamp
        if self.since is None:self.since=stamp
        complete=stamp-self.since>=self.settings['goal_dwell_s']-1e-8
        continuous=request.get('completion_policy','goal')=='continuous'
        return dict(r,status=('distance_held' if continuous else 'reached') if complete else 'holding',
            task_completed=complete and not continuous,completion_criteria_sustained=complete,held_s=stamp-self.since)
