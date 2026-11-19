"""One display/navigation proposal from a current RGB-lidar table observation."""
import copy
import threading
import time
from collections import deque

import numpy as np

from decision.vlm.image_inputs import camera_calibration
from decision.vlm.online import OnlineDecision
from decision.vlm.seating.geometry import edge_seeds, fit_edge, segment_candidates
from navigation.tasks import TaskPolicy


class SeatingWorker:
    def __init__(self, geometry, planning, motion):
        self.geometry, self.planning, self.motion = geometry, planning, motion
        self.calibration = camera_calibration()
        self.frames = deque(maxlen=8)
        self.latest_geometry = None
        self.lock = threading.Lock()
        self.thread = None
        self.result = None
        self.closed = False

    def observe(self, scene, observation, image, points, camera, received):
        if points is None or camera is None:
            return
        with self.lock:
            self.frames.append((scene, observation, image, points, camera, received))

    def update_scene(self,scene,measured_mono):
        with self.lock:
            self.latest_geometry=None if scene is None else (scene,measured_mono)

    def select(self, value):
        if set(value)!={'selection_id','observation_binding','instruction','roi'}:
            raise ValueError('Invalid seating selection request')
        if not all(isinstance(value[k],str) and 0<len(value[k])<=2048 for k in ('selection_id','observation_binding','instruction')):
            raise ValueError('Seating selection identifiers and instruction required')
        roi=value['roi']
        if (not isinstance(roi,list) or len(roi)!=4 or any(type(x) not in (int,float) for x in roi) or
                not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1)):
            raise ValueError('Normalized table ROI must lie within the current RGB')
        with self.lock:
            if self.closed or self.thread and self.thread.is_alive():
                raise ValueError('seating_selection_busy')
            original=next((f for f in reversed(self.frames) if f[1].record['input_fingerprint']==value['observation_binding']),None)
            if original is None:
                raise ValueError('seating_rgb_observation_expired')
            self.result=dict(status='selecting',selection_id=value['selection_id'])
            self.thread=threading.Thread(target=self._run,args=(copy.deepcopy(value),original),name='seating-t3',daemon=True)
            self.thread.start()
        return dict(accepted=True,selection_id=value['selection_id'],executable=False,execution_feedback=False)

    def _edge(self,frame,roi):
        scene,obs,rgb,points,camera,_=frame
        if (self.geometry.assumed or scene.record['localization']['status']!='valid' or
                (obs.record['sensors'].get('camera',{}).get('view') or {}).get('status')!='approximate_pinhole'):
            raise ValueError('seating_current_camera_or_vehicle_calibration_missing')
        height,width=rgb.shape[:2]
        pixels=np.asarray(roi)*[width,height,width,height]
        transform=np.asarray(scene.record['localization']['world_from_lidar'])
        support=edge_seeds(points,rgb,transform,self.calibration['matrix'],camera,pixels)
        return fit_edge(support,transform[:2,3])

    def _latest(self):
        with self.lock:
            if self.closed or not self.frames:
                raise ValueError('seating_observation_unavailable')
            return self.frames[-1]

    @staticmethod
    def _binding(scene):
        record=scene.record
        prior=record['prior']
        if (not prior.get('valid') or not prior.get('map_id') or not prior.get('map_frame') or
                [prior.get(k) for k in ('local_session_id','local_generation','local_frame')]!=
                [record.get(k) for k in ('session_id','generation','frame_id')]):
            raise ValueError('seating_map_association_unavailable')
        return ([record[k] for k in ('session_id','generation','frame_id')],record['clock_domain'],
                {k:prior.get(k) for k in ('map_id','map_frame','map_revision','association_session','association_generation')})

    def _run(self,value,original):
        result=dict(selection_id=value['selection_id'],observation_binding=value['observation_binding'])
        try:
            original_edge=self._edge(original,value['roi'])
            frame=self._latest()
            scene,obs=frame[:2]
            binding=self._binding(scene)
            if self._binding(original[0])!=binding:
                raise ValueError('seating_observation_identity_changed')
            edge=self._edge(frame,value['roi'])
            margin=self.planning['safety_margin_m']
            if np.max(np.linalg.norm(np.asarray(original_edge['observed_segment_xyz'])-
                    edge['observed_segment_xyz'],axis=1))>margin:
                raise ValueError('seating_table_edge_changed')
            poses=segment_candidates(edge,self.geometry.footprint,[2*margin],margin,3)
            if not poses:raise ValueError('seating_edge_too_short_for_vehicle')
            r=scene.record
            candidates=dict(instruction=value['instruction'],observation_id=r['observation_id'],stamp_s=r['as_of'],
                frame_id=r['frame_id'],clock_domain=r['clock_domain'],session_id=r['session_id'],generation=r['generation'],
                revision=0,candidates=poses)
            decision=OnlineDecision(TaskPolicy(value['selection_id'],'seating',self.planning,self.geometry))
            request=decision.seating(scene,candidates,observation=obs,now=r['as_of']+time.monotonic()-frame[-1])
            response=decision.infer(request)
            result.update(raw_response=response.get('raw_content'),request_binding=request['request_binding'])
            if response['status']!='complete':raise ValueError(response.get('reason') or 'seating_model_unavailable')
            with self.lock:latest=self.latest_geometry
            if latest is None:raise ValueError('seating_current_geometry_unavailable')
            current,measured_mono=latest
            if self._binding(current)!=binding:
                raise ValueError('seating_selection_superseded')
            now=current.record['as_of']+time.monotonic()-measured_mono
            action=decision.accept(request,response,scene=current,now=now)
            if action.get('operation')!='plan':
                raise ValueError(action.get('reason') or 'seating_candidate_unavailable')
            chosen=action['request']
            from navigation.planning.footprint import footprint_status
            pose=chosen['goal']
            if footprint_status(current.geometry,self.geometry,pose,margin)!='free':
                raise ValueError('seating_local_footprint_not_free')
            from navigation.planning.planner import Planner
            from navigation.planning.reference import planning_config
            policy=planning_config(self.planning)
            policy['motion_constraints']={k:self.motion[k] for k in ('min_turning_radius_m','allow_reverse')}
            path=Planner(policy,self.geometry,require_actual_start=True).query(
                chosen,scene=current,now=now,clock_domain=binding[1])
            if not path.get('current_path_valid') or path['status']!='path_found':
                raise ValueError(path.get('reason') or 'seating_approach_not_feasible')
            matrix=np.asarray(current.record['prior']['map_from_local'])
            xy=(matrix @ [*pose[:2],0.,1.])[:2]
            yaw=pose[2]+np.arctan2(matrix[1,0],matrix[0,0])
            edge_xyz=np.asarray(edge['observed_segment_xyz'])
            endpoints=(edge_xyz@matrix[:3,:3].T+matrix[:3,3])[:,:2]
            mapped_edge=dict(frame_id=binding[2]['map_frame'],endpoints_xy_m=endpoints.tolist(),
                             interior_yaw_rad=float(yaw))
            result.update(status='selected',pose=[*xy.tolist(),float(yaw)],candidate_id=chosen['candidate_id'],
                candidate_binding=chosen['candidate_binding'],edge=edge,source_stamp_s=r['as_of'],
                mapped_edge=mapped_edge,geometry_stamp_s=current.record['as_of'],path_status=path['status'],
                local_identity=binding[0],clock_domain=binding[1],map_binding=binding[2])
        except Exception as error:
            result.update(status='unavailable',reason=str(error)[:240] if isinstance(error,ValueError) else type(error).__name__)
        with self.lock:self.result=result

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.result)

    def close(self):
        with self.lock:
            self.closed=True
            self.frames.clear()
        if self.thread:self.thread.join(timeout=2.)
