"""Native upstream search with a narrow, master-free C ABI."""
import ctypes
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build(prefix=None):
    target = ROOT/'outputs/navigation/build'
    command = ['cmake', '-S', str(Path(__file__).with_name('ros')), '-B', str(target),
               '-DCMAKE_BUILD_TYPE=Release']
    if prefix:
        command.append('-DROS_NAVIGATION_PREFIX='+str(Path(prefix).resolve()))
    subprocess.run(command, check=True)
    subprocess.run(['cmake', '--build', str(target), '-j2'], check=True)


class GlobalPlanner:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.call = self.library.social_nav_plan
        self.call.argtypes = [np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS'),
                              *([ctypes.c_int]*7),
                              np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'),
                              ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.call.restype = ctypes.c_int

    def search(self, costs, start, goal, algorithm):
        if algorithm not in ('dijkstra', 'astar'):
            raise ValueError('algorithm must be dijkstra or astar')
        values = np.ascontiguousarray(costs, dtype=np.uint8)
        height, width = values.shape
        output = np.empty((values.size+1, 2), np.float32)
        elapsed = ctypes.c_double()
        count = self.call(values, width, height, *map(int, start), *map(int, goal),
                          int(algorithm == 'astar'), output, len(output), ctypes.byref(elapsed))
        if count < 0:
            raise RuntimeError('Native GlobalPlanner failed: '+str(count))
        return output[:count].astype(np.float64), elapsed.value


class FootprintSearch:
    """One serial search context with owned cache; independent checks remain mandatory."""
    def __init__(self,library,*,cache_heuristic=True,profile=False):
        self.context = None
        self.cache_heuristic = cache_heuristic
        self.profile = profile
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        if not hasattr(self.library,'social_nav_pose_plan'):raise RuntimeError('Rebuild the native footprint-search adapter')
        f64=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
        self.call=self.library.social_nav_pose_plan
        self.call.argtypes=[np.ctypeslib.ndpointer(np.int8,flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),
            ctypes.c_int,ctypes.c_int,ctypes.c_double,f64,f64,ctypes.c_int,f64,f64,ctypes.c_int,f64,f64,ctypes.c_int,f64]
        self.call.restype=ctypes.c_int
        self.cached_call=None
        if hasattr(self.library,'social_nav_pose_plan_cached'):
            self.library.social_nav_pose_create.argtypes=[]
            self.library.social_nav_pose_create.restype=ctypes.c_void_p
            self.library.social_nav_pose_destroy.argtypes=[ctypes.c_void_p]
            self.library.social_nav_pose_destroy.restype=None
            self.context=self.library.social_nav_pose_create()
            if not self.context:raise RuntimeError('Cannot allocate native footprint-search cache')
            self.cached_call=self.library.social_nav_pose_plan_cached
            self.cached_call.argtypes=[ctypes.c_void_p,*self.call.argtypes,ctypes.c_int]
            self.cached_call.restype=ctypes.c_int

    def close(self):
        if self.context is not None:
            self.library.social_nav_pose_destroy(self.context)
            self.context = None

    def __del__(self):
        self.close()

    def search(self,snapshot,geometry,config,clearance,costs,start,goal,*,state_resolution_m=None):
        cfg=config['pose_search'];policy=config['motion_constraints'];hard=len(goal)==3
        spin=policy.get('allow_in_place_rotation',False)
        if spin and not hasattr(self.library,'social_nav_pose_supports_rotation'):
            raise RuntimeError('Rebuild the native footprint-search library for in-place rotation')
        point=np.r_[goal[:2],goal[2] if hard else 0.].astype(np.float64)
        params=np.array([policy['min_turning_radius_m']+cfg['radius_margin_m'],config['safety_margin_m'],
            cfg['max_states'],cfg['max_expansions'],cfg['max_search_s'],float(int(policy['allow_reverse'])|2*int(spin)),
            cfg['reverse_penalty'],cfg['turn_weight'],cfg['goal_approach_straight_m'],
            cfg['state_resolution_m'] if state_resolution_m is None else state_resolution_m],np.float64)
        meta=snapshot.metadata;h,w=snapshot.layers['occupancy'].shape
        output=np.empty((min(50000,config['max_path_samples']),3),np.float64)
        stats=np.zeros(11 if self.cached_call is not None else 7,np.float64)
        arguments=[np.ascontiguousarray(snapshot.layers['occupancy'],np.int8),np.ascontiguousarray(costs,np.uint8),
            np.ascontiguousarray(clearance,np.float32),w,h,meta['resolution'],np.asarray(meta['origin_xy'],np.float64),
            np.ascontiguousarray(geometry.footprint,np.float64),len(geometry.footprint),np.ascontiguousarray(start,np.float64),
            point,int(hard),params,output,len(output),stats]
        began=time.perf_counter()
        count=(self.call(*arguments) if self.cached_call is None else
               self.cached_call(self.context,*arguments,int(self.cache_heuristic)|int(self.profile)*2))
        native_wall_ms=(time.perf_counter()-began)*1000
        if count<0:raise RuntimeError('Native footprint search failed: '+str(count))
        reasons={0:None,1:'curvature_search_time_budget',2:'curvature_search_state_budget',
            3:'no_curvature_route',4:'invalid_search_start',5:'disconnected_search_region'}
        detail=dict(reason=reasons[int(stats[6])],search_ms=float(stats[0]),native_call_wall_ms=native_wall_ms,
            expanded=int(stats[1]),states=int(stats[2]),
            collision_checks=int(stats[3]),heuristic_ms=float(stats[4]),endpoint_connections_checked=int(stats[5]),
            motion_model='differential_straight_arc_and_rotation' if spin else 'finite_radius_straight_and_arc',search_radius_m=params[0],
            allow_in_place_rotation=spin,
            min_turning_radius_m=policy['min_turning_radius_m'],allow_reverse=policy['allow_reverse'],
            state_resolution_m=max(meta['resolution'],params[9]),
            implementation='native_hybrid_astar_wavefront')
        detail['heuristic_cache_supported']=self.cached_call is not None
        if self.cached_call is not None:
            detail.update(heuristic_cache_reused=bool(stats[10]),profiled=bool(self.profile))
            if self.profile:
                detail.update(endpoint_connection_ms=float(stats[7]),motion_checks_ms=float(stats[8]),
                    state_search_loop_ms=float(stats[9]))
        return (output[:count].copy() if count else None),detail
