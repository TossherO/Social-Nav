"""One owner for simulation tracking, independent safety and measured arrival.

Planning stays in ReferenceRuntime. This adapter has no ROS/chassis publisher and
does not synthesize feedback. Call on every control tick, including outages.
"""
from .tracking import GoalMonitor
from navigation.safety.guard import LocalSafety


class DwaNavigationControl:
    def __init__(self,planning,motion,geometry,*,asset=None,simulation=False):
        from .dwa import DwaController
        self.controller=DwaController(planning,motion,geometry,asset=asset)
        self.safety=LocalSafety(geometry,motion,planning['safety_margin_m'],people_settings=planning['people'],simulation=simulation)
        self.monitor=GoalMonitor(self.controller.goal_settings,geometry=geometry,asset=asset,planning=planning)

    def step(self,request,scene,feedback,*,now,clock_domain,dt,result=None,active=True,manual_override=False,emergency_stop=False):
        if not active or manual_override or emergency_stop:self.controller.clear();proposal=None
        else:proposal=self.controller.command(request,scene,feedback,result,now=now,clock_domain=clock_domain,dt=dt,pending=self.safety.last_target)
        command=self.safety.gate(proposal,request,scene,feedback,now=now,clock_domain=clock_domain,dt=dt,
            manual_override=manual_override,emergency_stop=emergency_stop)
        outcome=self.monitor.update(request,scene,feedback,now=now,clock_domain=clock_domain,
            active=active and not (manual_override or emergency_stop) and command['reason'] in (None,'planner_unavailable'))
        return dict(command=command,proposal=proposal,goal=outcome,result_accepted=self.controller.result is not None,
            result=self.controller.result,executable=False)
