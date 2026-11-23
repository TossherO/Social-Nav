"""Continuously select a table approach from current RGB-lidar observations."""
import copy
import threading
import time
from collections import deque

import numpy as np
import cv2

from decision.vlm.image_inputs import camera_calibration
from decision.vlm.project_config import load_config as load_vlm_config
from decision.vlm.t3.online import SeatingDecision
from decision.vlm.t3 import table_keepout_circle
from decision.vlm.t3.geometry import (
    edge_seeds, fit_edge, round_candidates, round_clearance, round_parking_approach,
    segment_candidates, validate_round_table,
)
from navigation.planning.footprint import wrap
from navigation.planning.planner import transform_poses
from navigation.tasks import TaskPolicy


def _transform_pose(pose, transform):
    x, y, yaw = transform_poses([pose], transform)[0]
    return [float(x), float(y), float(wrap(yaw))]


class SeatingWorker:
    def __init__(self, geometry, planning, motion, *, clock_rate=1., asset=None):
        self.geometry, self.planning, self.motion = geometry, planning, motion
        self.asset = asset
        self.clock_rate = clock_rate
        self.calibration = camera_calibration()
        self.seating = load_vlm_config()['tasks']['seating']
        for key in ('table_clearance_m','edge_end_guard_m','edge_match_tolerance_m'):
            value = self.seating[key]
            if type(value) not in (int,float) or not np.isfinite(value) or value<=0.:
                raise ValueError('Invalid T3 '+key)
        self.frames = deque(maxlen=8)
        self.latest_geometry = None
        self.lock = threading.Lock()
        self.thread = None
        self.result = None
        self.closed = False
        self.active = None
        self.epoch = 0
        self.sequence = 0
        self.last_stamp = None
        self.last_candidate = None

    def observe(self, scene, observation, image, points, camera, clock):
        if points is None or camera is None:
            return
        with self.lock:
            self.frames.append((scene, observation, image, points, camera, clock))
            if (self.active is not None and (self.thread is None or not self.thread.is_alive()) and
                    self.last_stamp is not None and scene.record['as_of']>self.last_stamp):
                value,original,original_edge,epoch=self.active
                self.thread=threading.Thread(target=self._run,args=(value,original,original_edge,epoch),
                    name='seating-t3',daemon=True)
                self.thread.start()

    def update_scene(self,scene,clock):
        with self.lock:
            self.latest_geometry=None if scene is None else (scene,clock)

    def _source_now(self, clock):
        source_s, source_mono = clock
        return source_s + (time.monotonic() - source_mono) * self.clock_rate

    def select(self, value):
        fields={'selection_id','observation_binding','instruction'}
        if set(value) not in (fields|{'roi'}, fields|{'round_table'},
                              fields|{'round_table','manual_observation'}):
            raise ValueError('Invalid seating selection request')
        if not all(isinstance(value[k],str) and 0<len(value[k])<=2048 for k in ('selection_id','observation_binding','instruction')):
            raise ValueError('Seating selection identifiers and instruction required')
        roi=value.get('roi')
        if 'round_table' in value:
            validate_round_table(value['round_table'])
        elif (not isinstance(roi,list) or len(roi)!=4 or any(type(x) not in (int,float) for x in roi) or
                not (0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1)):
            raise ValueError('Normalized table ROI must lie within the current RGB')
        observation = value.get('manual_observation')
        if observation is not None:
            if (not isinstance(observation, dict) or
                    set(observation) - {'after_s','visited_xy_m','rejected_views'} or
                    not {'after_s','visited_xy_m'} <= observation.keys() or
                    type(observation['after_s']) not in (int,float) or not np.isfinite(observation['after_s']) or
                    not isinstance(observation['visited_xy_m'],list) or len(observation['visited_xy_m'])>2 or
                    any(not isinstance(xy,list) or len(xy)!=2 or
                        any(type(v) not in (int,float) or not np.isfinite(v) for v in xy)
                        for xy in observation['visited_xy_m'])):
                raise ValueError('Invalid seating observation request')
            rejected_views = observation.get('rejected_views', [])
            if (not isinstance(rejected_views, list) or len(rejected_views) > 8 or
                    any(not isinstance(item, dict) or set(item) != {'pose','reason'} or
                        not isinstance(item['pose'], list) or len(item['pose']) != 3 or
                        any(type(v) not in (int,float) or not np.isfinite(v) for v in item['pose']) or
                        not isinstance(item['reason'], str) or not 0 < len(item['reason']) <= 128
                        for item in rejected_views)):
                raise ValueError('Invalid rejected seating observation poses')
        with self.lock:
            if self.closed or self.active is not None or self.thread and self.thread.is_alive():
                raise ValueError('seating_selection_busy')
            needs_new_view = observation is not None and observation['after_s'] > 0.
            if not needs_new_view:
                original=next((f for f in reversed(self.frames) if f[1].record['input_fingerprint']==value['observation_binding']),None)
            else:
                original=next((f for f in reversed(self.frames)
                               if f[0].record['as_of']>observation['after_s'] and
                               f[1].record['rgb']['stamp_s']>observation['after_s']),None)
            if original is None:
                if not needs_new_view:
                    raise ValueError('seating_rgb_observation_expired')
                return dict(accepted=False, executable=False, execution_feedback=False,
                            error='seating_waiting_for_new_view')
            self.result=dict(status='selecting',selection_id=value['selection_id'])
            self.epoch+=1;self.sequence=0;self.last_stamp=None
            self.last_candidate=None
            value=copy.deepcopy(value)
            self.active=(value,original,None,self.epoch)
            self.thread=threading.Thread(target=self._run,args=(value,original,None,self.epoch),name='seating-t3',daemon=True)
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

    def cancel(self, selection_id):
        with self.lock:
            if self.active is None or self.active[0]['selection_id']!=selection_id:
                return dict(accepted=False,error='seating_selection_not_active')
            self.epoch+=1;self.active=None
            self.result=dict(status='cancelled',selection_id=selection_id)
        return dict(accepted=True,selection_id=selection_id,execution_feedback=False)

    def _project_roi(self,frame,edge):
        scene,_,rgb,_,camera,_=frame
        world=np.asarray(edge['observed_segment_xyz'],dtype=float)
        lidar_from_world=np.linalg.inv(np.asarray(scene.record['localization']['world_from_lidar']))
        transform=np.asarray(self.calibration['matrix'])@lidar_from_world
        optical=world@transform[:3,:3].T+transform[:3,3]
        if np.any(optical[:,2]<=.3):raise ValueError('seating_table_out_of_view')
        uv=cv2.projectPoints(optical,np.zeros(3),np.zeros(3),np.asarray(camera['K']).reshape(3,3),
                             np.asarray(camera['D'],dtype=float))[0].reshape(-1,2)
        h,w=rgb.shape[:2]
        # Expand around the projected supported edge to include nearby surface returns.
        lo=uv.min(axis=0)-[.08*w,.15*h];hi=uv.max(axis=0)+[.08*w,.15*h]
        lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[w,h])
        if np.any(hi-lo<4):raise ValueError('seating_table_out_of_view')
        return [float(lo[0]/w),float(lo[1]/h),float(hi[0]/w),float(hi[1]/h)]

    @staticmethod
    def _same_edge(reference,current,margin):
        a=np.asarray(reference['observed_segment_xyz'],dtype=float)
        b=np.asarray(current['observed_segment_xyz'],dtype=float)
        direction=a[1,:2]-a[0,:2];length=np.linalg.norm(direction)
        other=b[1,:2]-b[0,:2];other_length=np.linalg.norm(other)
        if length<.1 or other_length<.1:return False
        direction/=length;other/=other_length
        normal=np.array([-direction[1],direction[0]])
        along=(b[:,:2]-a[0,:2])@direction
        return bool(abs(direction@other)>.94 and
            np.max(np.abs((b[:,:2]-a[0,:2])@normal))<=margin and
            abs(np.mean(b[:,2])-np.mean(a[:,2]))<=margin and
            max(along.min(),0.)<=min(along.max(),length)+margin)

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

    def _route(self, planner, request, scene, now, *, manual_table=None):
        from navigation.planning.footprint import footprint_status
        from perception.environment.navigation import navigation_geometry
        from navigation.planning.planner import transform_poses

        margin = self.planning['safety_margin_m']
        grid = navigation_geometry(scene, now=now, asset=self.asset, check_asset=True,
                                   include_confirmed_free=manual_table is not None)
        status = footprint_status(grid if manual_table else scene.geometry,self.geometry,request['goal'],margin)
        if status!='free':
            return ('seating_local_footprint_'+status if manual_table else 'seating_local_footprint_not_free'), False
        mapped = manual_table is not None and self.asset is not None
        if mapped:
            goal = transform_poses([request['goal']],self._map_from_local(scene))[0].tolist()
            request = dict(request,mode='mapped',frame_id=scene.record['prior']['map_frame'],goal=goal,
                           map_snapshot_id=self.asset.manifest['snapshot_id'],use_local_approach=True)
        if manual_table is not None:
            request = dict(request,map_keepout_circle=table_keepout_circle(manual_table))
        path = planner.query(request, scene=scene, asset=self.asset, now=now,
                             clock_domain=scene.record['clock_domain'])
        if path['status']!='path_found':
            return path.get('reason') or 'seating_approach_not_feasible', False
        if not path.get('current_path_valid'):
            return 'seating_local_path_'+path.get('local_check',{}).get('status','unavailable'), False
        revised_goal = mapped and footprint_status(self.asset.snapshot,self.geometry,goal,margin) in ('occupied','unknown')
        return None, revised_goal

    def _map_from_local(self, scene):
        from navigation.planning.planner import se2

        record = scene.record
        matrix = np.asarray(record['prior']['map_from_local'])
        lidar = np.asarray(record['localization']['world_from_lidar'])
        return se2(self.geometry.pose(matrix @ lidar)) @ np.linalg.inv(se2(self.geometry.pose(lidar)))

    def _mapped_view_route(self, planner, request, scene, now, pose, *, table=None):
        """Check a display-only view approach against the mapped reference and current hits."""
        from navigation.planning.footprint import swept_check
        from navigation.planning.planner import transform_poses
        from perception.environment.layers import GridSnapshot
        from perception.environment.navigation import navigation_geometry

        if self.asset is None:
            return 'seating_view_requires_static_map'
        mapped = dict(request, mode='mapped', frame_id=scene.record['prior']['map_frame'],
                      goal=pose, map_snapshot_id=self.asset.manifest['snapshot_id'])
        if table is not None:
            mapped['map_keepout_circle'] = table_keepout_circle(table)
        result = planner.query(mapped, scene=scene, asset=self.asset, now=now,
                               clock_domain=scene.record['clock_domain'])
        if result['status']!='path_found':
            return result.get('reason') or 'seating_view_reference_unavailable'
        if result.get('reference_overlay',{}).get('status')!='valid' or result.get('conditional_reference'):
            return 'seating_view_reference_not_supported_by_map'
        if result.get('local_check',{}).get('status')=='occupied':
            return 'seating_view_current_path_occupied'
        # Audit the whole reference: a nearer unknown must not hide a farther hit.
        raw = scene.geometry
        current = navigation_geometry(scene, now=now, asset=self.asset, check_asset=True, include_confirmed_free=True)
        blocked = (current.layers['occupancy']==100)|(raw.layers['occupancy']==100)
        if 'ground_conflict' in raw.layers:
            blocked |= raw.layers['ground_conflict']
        obstacle_grid = GridSnapshot(raw.metadata,dict(occupancy=np.where(blocked,100,0).astype(np.int8)))
        local_path = transform_poses(result['path'],np.linalg.inv(self._map_from_local(scene)))
        audit = swept_check(obstacle_grid,self.geometry,local_path,self.planning['safety_margin_m'],
                            self.planning['max_path_samples'],prefix=False)
        if audit['status']!='free':
            return 'seating_view_current_path_'+audit['status']
        return None

    def _view_search(self, scene, local_table, candidates, value, rejected):
        grid = scene.geometry.metadata
        origin = np.asarray(grid['origin_xy'])
        upper = origin + np.asarray(grid['shape'][::-1])*grid['resolution']
        lidar = np.asarray(scene.record['localization']['world_from_lidar'])
        base = self.geometry.pose(lidar)
        points = np.array([base[:2],local_table['center_xy_m'],*(c['pose'][:2] for c in candidates)])
        lower = np.maximum(origin,points.min(axis=0)-2.)
        upper = np.minimum(upper,points.max(axis=0)+2.)
        visits = value['manual_observation']['visited_xy_m']
        inverse = np.linalg.inv(self._map_from_local(scene))
        local_visits = transform_poses([[*xy,0.] for xy in visits], inverse)[:,:2].tolist() if visits else []
        return dict(bounds_xy_m=[*lower.tolist(),*upper.tolist()],visited_xy_m=local_visits,
                    can_propose=len(visits)<2 and self.last_candidate is None,
                    rejected_views=[dict(pose=transform_poses([item['pose']],inverse)[0].tolist(),
                                         reason=item['reason']) for item in rejected])

    def _view_route(self, planner, chosen, scene, clock, table, search, rejected, *, source_scene=None):
        source_scene = scene if source_scene is None else source_scene
        local = chosen['goal']
        xy = np.asarray(local[:2])
        low, high = np.asarray(search['bounds_xy_m']).reshape(2,2)
        pose = _transform_pose(local, self._map_from_local(source_scene))
        if np.any(xy<low) or np.any(xy>high):
            return pose,'seating_view_outside_displayed_area',False
        if abs(local[2])>np.pi:
            return pose,'seating_view_yaw_must_be_in_radians_between_minus_pi_and_pi',False
        current_local = _transform_pose(pose, np.linalg.inv(self._map_from_local(scene)))
        base = self.geometry.pose(np.asarray(scene.record['localization']['world_from_lidar']))
        if np.linalg.norm(np.asarray(current_local[:2])-base[:2])<.35 and abs(wrap(current_local[2]-base[2]))<np.pi/6:
            return pose,'seating_view_too_similar_to_current',False
        if any(np.linalg.norm(xy-visited)<.35 for visited in search['visited_xy_m']):
            return pose,'seating_view_already_visited',False
        if any(np.linalg.norm(np.asarray(pose[:2])-item['pose'][:2])<.35 for item in rejected):
            return pose,'seating_view_repeats_rejected_pose',False
        if round_clearance(table,self.geometry.polygon(pose))<=self.planning['safety_margin_m']:
            return pose,'seating_view_too_close_to_table',False
        if self.asset is None:
            reason,revised = self._route(planner,dict(chosen, goal=current_local),scene,
                                         self._source_now(clock),manual_table=table)
            return pose,reason,revised
        return pose,self._mapped_view_route(planner,chosen,scene,self._source_now(clock),pose,table=table),False

    def _choose(self, value, binding, original, edge, epoch, result):
        from navigation.planning.footprint import footprint_status
        from navigation.planning.planner import Planner
        from navigation.planning.reference import planning_config
        table = value.get('round_table')
        manual_table = table if 'manual_observation' in value else None
        margin = self.planning['safety_margin_m']
        clearance = self.seating['table_clearance_m']
        deadline = None
        front_allowance = 0.
        candidate_count = 8
        decision_planning = self.planning
        if manual_table:
            settings = self.seating
            deadline = float(settings['manual_round_deadline_s'])
            candidate_count = settings.get('manual_round_candidate_count', 8)
            front_allowance = float(settings.get('manual_round_front_allowance_m', 0.))
            if not 0. <= front_allowance <= .20:
                raise ValueError('seating_manual_round_front_allowance_out_of_range')
            decision_planning = dict(self.planning, people=dict(self.planning['people'], decision_max_age_s=deadline))
        policy = planning_config(self.planning)
        policy['motion_constraints'] = {k:self.motion[k] for k in ('min_turning_radius_m','allow_reverse')}
        planner = Planner(policy, self.geometry, require_actual_start=True)
        rejected = {}
        result['attempts'] = []
        view_checks = []
        previously_rejected = value.get('manual_observation', {}).get('rejected_views', [])

        while True:
            with self.lock:
                if self.closed or epoch != self.epoch:
                    raise ValueError('seating_selection_superseded')
                first_view = self.sequence == 0 and not result['attempts']
            frame = original if first_view else self._latest()
            after = value.get('manual_observation',{}).get('after_s')
            if after is not None and not (frame[0].record['as_of']>after and frame[1].record['rgb']['stamp_s']>after):
                frame = original
            scene,obs=frame[:2]
            r=scene.record
            if self._binding(scene)!=binding:
                raise ValueError('seating_observation_identity_changed')
            instruction = value['instruction']
            local_table = None
            if table:
                if self.geometry.assumed or table['map_id'] != binding[2]['map_id']:
                    raise ValueError('seating_round_table_map_or_vehicle_mismatch')
                map_from_local = self._map_from_local(scene)
                local_from_map = np.linalg.inv(map_from_local)
                local_table = dict(table, center_xy_m=_transform_pose([*table['center_xy_m'],0.],local_from_map)[:2])
                poses = round_candidates(table, self.geometry.footprint, clearance, count=candidate_count)
                from perception.environment.navigation import navigation_geometry
                grid = (navigation_geometry(scene,asset=self.asset,check_asset=self.asset is not None,
                                            include_confirmed_free=True) if manual_table else scene.geometry)
                parking = {}
                for candidate in poses:
                    candidate_id = candidate['id']
                    nominal = _transform_pose(candidate['pose'],local_from_map)
                    candidate['pose'] = nominal
                    if candidate_id in rejected:
                        candidate['geometry'] = 'blocked'
                    else:
                        backoff = 0.
                        if manual_table:
                            pose, status, backoff = round_parking_approach(grid,self.geometry,nominal,margin,front_allowance)
                            candidate['pose'] = pose
                        else:
                            status = footprint_status(grid,self.geometry,nominal,margin)
                        parking[candidate_id] = (_transform_pose(candidate['pose'],map_from_local),nominal,backoff)
                        candidate['geometry'] = {'occupied':'blocked','free':'feasible'}.get(status,'unknown')
                if not manual_table and not any(c['geometry'] != 'blocked' for c in poses):
                    raise ValueError('seating_no_remaining_round_table_candidate')
                if self.last_candidate is not None:
                    instruction += f' Current parking candidate: C{self.last_candidate}.'
                if rejected:
                    instruction += ' Rejected by the current geometry checks: ' + ', '.join(
                        f'C{i} ({reason})' for i,reason in rejected.items())
            else:
                roi=value['roi'] if frame is original else self._project_roi(frame,edge)
                current_edge=self._edge(frame,roi)
                if not self._same_edge(edge,current_edge,self.seating['edge_match_tolerance_m']):
                    raise ValueError('seating_table_edge_changed')
                poses=segment_candidates(current_edge,self.geometry.footprint,[clearance],self.seating['edge_end_guard_m'],3)
                if not poses:raise ValueError('seating_edge_too_short_for_vehicle')
            candidates=dict(instruction=instruction,observation_id=r['observation_id'],stamp_s=r['as_of'],
                frame_id=r['frame_id'],clock_domain=r['clock_domain'],session_id=r['session_id'],generation=r['generation'],
                revision=0,candidates=poses)
            if local_table:
                candidates['round_table'] = local_table
            if manual_table:
                candidates['view_search'] = self._view_search(scene,local_table,poses,value,
                                                              [*previously_rejected,*view_checks])
            source_observation = obs.record['input_fingerprint']
            attempt = dict(source_stamp_s=r['as_of'],source_observation_binding=source_observation,candidates=poses)
            result['attempts'].append(attempt)
            decision=SeatingDecision(TaskPolicy(value['selection_id'],'seating',decision_planning,self.geometry),
                                     deadline_s=deadline)
            request=decision.seating(scene,candidates,observation=obs,now=self._source_now(frame[-1]),
                                     view_prior=self.asset if manual_table else None)
            response=decision.infer(request)
            result.update(raw_response=response.get('raw_content'),request_binding=request['request_binding'],
                          source_observation_binding=source_observation)
            attempt.update(raw_response=result['raw_response'], request_binding=result['request_binding'])
            if 'timing' in response:
                attempt['timing'] = response['timing']
            if response['status']!='complete':raise ValueError(response.get('reason') or 'seating_model_unavailable')
            with self.lock:latest=self.latest_geometry
            if latest is None:raise ValueError('seating_current_geometry_unavailable')
            current,clock=latest
            if self._binding(current)!=binding:
                raise ValueError('seating_selection_superseded')
            now=self._source_now(clock)
            action=decision.accept(request,response,scene=current,now=now)
            if action['operation']=='observe':
                chosen=action['request']
                search=candidates['view_search']
                pose,reason,revised=self._view_route(planner,chosen,current,clock,table,search,
                                                    [*previously_rejected,*view_checks],source_scene=scene)
                view_checks.append(dict(pose=pose,reason=reason,focus=action['focus']))
                result['view_checks']=view_checks
                if reason is None:
                    result['observation_focus']=action['focus']
                    chosen=dict(chosen,goal=_transform_pose(pose,np.linalg.inv(self._map_from_local(current))))
                    return chosen,current,None,r['as_of'],'observe',revised,0.
                if len(view_checks)>=2:
                    raise ValueError('seating_no_reachable_observation_pose')
                continue
            if action.get('operation')!='plan':
                if manual_table:
                    result['model_reason']=decision.policy.raw.get('reason')
                    if self.last_candidate is not None:
                        raise ValueError('seating_reobserve_after_stop')
                raise ValueError(action.get('reason') or 'seating_candidate_unavailable')
            chosen=action['request']
            backoff = 0.
            if table:
                mapped_pose, nominal, backoff = parking[chosen['candidate_id']]
                chosen = dict(chosen,goal=_transform_pose(mapped_pose,np.linalg.inv(self._map_from_local(current))))
                if backoff:
                    attempt['nominal_goal'] = nominal
                    attempt['front_backoff_m'] = backoff
            if table is not None and manual_table is None:
                chosen = dict(chosen,map_keepout_circle=table_keepout_circle(table))
            reason,revised=self._route(planner,chosen,current,now,manual_table=manual_table)
            attempt.update(candidate_id=chosen['candidate_id'], rejection=reason)
            if reason is None:
                return chosen,current,(None if table else current_edge),r['as_of'],'park',revised,backoff
            if table is None:
                raise ValueError(reason)
            rejected[chosen['candidate_id']] = reason

    def _run(self,value,original,original_edge,epoch):
        result=dict(selection_id=value['selection_id'],observation_binding=value['observation_binding'])
        try:
            binding=self._binding(original[0])
            table=value.get('round_table')
            if table is None and original_edge is None:
                original_edge=self._edge(original,value['roi'])
            chosen,current,edge,stamp,phase,revised,backoff=self._choose(value,binding,original,original_edge,epoch,result)
            pose=chosen['goal']
            if table:
                mapped_pose = _transform_pose(pose,self._map_from_local(current))
                result['mapped_round_table'] = dict(table, target_gap_m=self.seating['table_clearance_m']+backoff)
                if backoff:
                    result['front_backoff_m'] = backoff
            else:
                matrix=np.asarray(current.record['prior']['map_from_local'])
                xy=(matrix @ [*pose[:2],0.,1.])[:2]
                heading=matrix[:2,:2] @ [np.cos(pose[2]),np.sin(pose[2])]
                yaw=float(np.arctan2(heading[1],heading[0]))
                mapped_pose=[*xy.tolist(),yaw]
                edge_xyz=np.asarray(edge['observed_segment_xyz'])
                endpoints=(edge_xyz@matrix[:3,:3].T+matrix[:3,3])[:,:2]
                result['edge']=edge
                result['mapped_edge']=dict(frame_id=binding[2]['map_frame'],endpoints_xy_m=endpoints.tolist(),
                                           interior_yaw_rad=yaw)
            result.update(status='selected' if phase=='park' else 'observation',pose=mapped_pose,
                mapped_reference_verified=phase=='observe' and self.asset is not None,
                conditional_prior_cleared=revised,candidate_id=chosen['candidate_id'],
                candidate_binding=chosen['candidate_binding'],source_stamp_s=stamp,
                geometry_stamp_s=current.record['as_of'],path_status='path_found',
                local_identity=binding[0],clock_domain=binding[1],map_binding=binding[2])
        except Exception as error:
            result.update(status='unavailable',reason=str(error)[:240] if isinstance(error,ValueError) else type(error).__name__)
        with self.lock:
            if self.closed or self.active is None or epoch!=self.epoch:return
            self.sequence+=1;result['sequence']=self.sequence
            self.result=result
            if result['status']=='selected':
                self.last_stamp=result['source_stamp_s']
                self.last_candidate=result['candidate_id']
                self.active=(value,original,original_edge,epoch)
                if self.frames and self.frames[-1][0].record['as_of']>self.last_stamp:
                    self.thread=threading.Thread(target=self._run,args=(value,original,original_edge,epoch),
                        name='seating-t3',daemon=True)
                    self.thread.start()
            else:
                self.active=None

    def snapshot(self):
        with self.lock:return copy.deepcopy(self.result)

    def close(self):
        with self.lock:
            self.closed=True
            self.epoch+=1;self.active=None
            self.frames.clear()
        if self.thread:self.thread.join(timeout=2.)
