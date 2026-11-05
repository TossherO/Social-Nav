"""Finite-radius native search with shared map fields and independent certification."""
from .backend import FootprintSearch
from .costmap import build_costmap
from .curvature import motion_valid
from .footprint import compact_poses


class PoseGrid:
    def __init__(self,snapshot,geometry,config,*,costs=None,clearance=None,local=False,backend=None):
        self.snapshot,self.geometry,self.config=snapshot,geometry,config
        if costs is None or clearance is None:
            costs,clearance=build_costmap(snapshot,geometry.inscribed_radius,config,return_clearance=True)
        self.costs,self.clearance=costs,clearance
        self.state_resolution=config['pose_search']['local_state_resolution_m' if local else 'state_resolution_m']
        self.backend=backend or FootprintSearch(config['pose_search'].get('native_library',config['native_library']),
            cache_heuristic=config['pose_search'].get('cache_heuristic',True))

    def search(self,start,goal,guide=None):
        # The reverse wavefront uses the full current grid rather than a short
        # list of waypoints; one search can follow arbitrarily many bends.
        poses,detail=self.backend.search(self.snapshot,self.geometry,self.config,self.clearance,self.costs,start,goal,
            state_resolution_m=self.state_resolution)
        if poses is None:return None,detail
        poses=compact_poses(poses)
        import numpy as np
        if np.linalg.norm(poses[-1,:2]-goal[:2])>1e-6:raise RuntimeError('Native endpoint mismatch')
        poses[-1,:2]=goal[:2]
        if len(goal)==3:
            from .footprint import wrap
            if abs(wrap(poses[-1,2]-goal[2]))>1e-6:raise RuntimeError('Native endpoint yaw mismatch')
            poses[-1,2]+=wrap(goal[2]-poses[-1,2])
        if len(poses)==1:
            poses=np.vstack([poses,poses])
        if not motion_valid(poses,self.config['motion_constraints']):
            raise RuntimeError('Native search failed independent curvature certification')
        # Native sweeps certify the curves; the public planner separately checks
        # the returned representation against the original occupancy grid.
        return poses,detail
