"""Clocked control scheduling beside perception, sharing sealed GPU observations."""
from collections import deque
import copy
import json
import threading
import time


def agent_mapped_pose(scene, row, geometry, now, max_association_age_s, clock_domain):
    """Bind a fresh localization measurement to the current scene's map association."""
    import math
    import numpy as np
    from perception.mapping_localization.context import pose_matrix

    if scene is None or not isinstance(row, dict):
        return None
    record = scene.record
    prior = record.get('prior') or {}
    identity = [row.get(k) for k in ('session_id', 'generation', 'frame_id')]
    stamp, measured = row.get('stamp_s'), prior.get('measurement_stamp_s')
    if (row.get('health') != 0 or record.get('clock_domain') != clock_domain or
            record.get('localization', {}).get('status') != 'valid' or not prior.get('valid') or
            identity != [record.get(k) for k in ('session_id', 'generation', 'frame_id')] or
            identity != [prior.get(k) for k in ('local_session_id', 'local_generation', 'local_frame')] or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in (stamp, measured, now)) or
            not 0 <= now-stamp <= .5 or not 0 <= stamp-measured or
            not 0 <= now-measured <= max_association_age_s):
        return None
    try:
        local = pose_matrix(row['position'], row['quaternion_xyzw'])
        pose = geometry.pose(np.asarray(prior['map_from_local']) @ local).tolist()
    except (ValueError, TypeError, KeyError, IndexError):
        return None
    binding = {k: copy.deepcopy(prior.get(k)) for k in
               ('valid', 'map_id', 'map_frame', 'map_revision', 'association_session', 'association_generation')}
    return dict(pose=pose, stamp_s=stamp, clock_domain=clock_domain, local_identity=identity,
                map_binding=binding, association_stamp_s=measured, source='current_localization_and_scene_association')


def manual_preview_path(reference, request):
    """Expose a bounded, identity-matched reference for task-level checking."""
    if not isinstance(reference,dict) or not isinstance(request,dict):return None
    result=reference.get('result')
    if not isinstance(result,dict) or result.get('frame_id')!=request.get('frame_id'):
        return None
    from navigation.planning.continuity import goal_binding
    if goal_binding(result.get('request',{}))!=goal_binding(request):return None
    path=result.get('path')
    if not isinstance(path,list) or not 2<=len(path)<=1024:return None
    try:
        import math
        points=[[float(p[0]),float(p[1])] for p in path]
        if not all(math.isfinite(x) and math.isfinite(y) for x,y in points):return None
        return points
    except (TypeError,ValueError,IndexError,KeyError):return None


def control_planning_observation(packet, scene, max_age_s):
    """Tie local control outcomes to a source scan, not to control loop time."""
    request=packet.get('request')
    if request is None or scene is None or packet.get('reason'):
        return None
    record=scene.record
    if (packet.get('scene_binding')!=record['input_fingerprint'] or
            not 0<=packet['stamp_s']-record['as_of']<=max_age_s or
            (packet.get('feedback') or {}).get('status')!='valid'):
        return None
    proposal=packet.get('proposal') or {}
    reason=proposal.get('reason')
    blocked=reason=='no_admissible_velocity' and proposal.get('unattempted_candidates')==0
    velocity=proposal.get('velocity') or [0.,0.]
    moving=proposal.get('status')=='candidate' and (abs(velocity[0])>=.03 or abs(velocity[1])>=.05)
    status='blocked' if blocked else 'clear' if moving else 'unknown'
    from navigation.planning.reference import planning_observation
    return planning_observation(request,record,source='dwa',status=status,reason=reason,max_age_s=max_age_s,
        kind='local_control_blocked' if blocked else None)


class ControlLoop:
    def __init__(self,reference,controller,batch,safety,pose_socket,commands,settings,scene_settings,metrics,record,*,asset=None,initial_goal=None,clock_domain='runtime_bag',display=None,route_replay=None,planning_recorder=None,session=None,follow_decision=None):
        self.reference,self.controller,self.batch,self.safety=reference,controller,batch,safety
        self.pose_socket,self.commands=pose_socket,commands;self.settings=settings;self.scene_settings=scene_settings
        self.metrics,self.record,self.asset=metrics,record,asset;self.request=initial_goal;self.clock_domain=clock_domain;self.display=display
        self.route_replay=route_replay
        self.planning_recorder=planning_recorder;self.offered_mono=None
        self.session=session;self.session_epoch=0;self.preview_token=None
        self.manual_preview_key=None
        self.lock=threading.Lock();self.pending=(None,None);self.rows=deque(maxlen=256)
        self.environment_epoch=0;self.environment_fault=None
        self.device=None;self.scene=None;self.done=threading.Event();self.error=None;self.count=0;self.clock=None
        self.cancelled_request=None
        self.follow_pending=None;self.follow=None
        self.follow_target=None
        self.follow_issued=None
        self.follow_held=False
        self.manual_follow=False
        self.follow_decision=follow_decision
        self.follow_search_anchor=None;self.follow_search_choice=None
        self.follow_search_intent=None
        if commands.follow_enabled:
            from .follow import FollowRunner
            self.follow=FollowRunner(reference.planning,controller.geometry)
        self.reference.worker.scene_factory=self._search_scene
        if initial_goal is not None:self.safety.command('goal',initial_goal)
        self.worker=threading.Thread(target=self._work,name='runtime-control',daemon=True);self.worker.start()

    def offer(self,scene,device):
        with self.lock:
            if self.follow is not None:
                try:self.follow.observe(scene)
                except (ValueError,KeyError,TypeError):self.follow.history.rows.clear()
            self.pending=(scene,device);self.offered_mono=time.monotonic()
            if scene is not None:self.environment_fault=None
            self.metrics.trace('control_scene_offer',scene_binding=None if scene is None else scene.record['input_fingerprint'],
                measurement_stamp_s=None if scene is None else scene.record['as_of'],offered_mono=self.offered_mono)

    def invalidate_environment(self,fault):
        with self.lock:
            self.environment_epoch+=1;self.environment_fault=fault.reason
            self.pending=(None,None);self.offered_mono=time.monotonic()
            if self.follow is not None:self.follow.history.rows.clear()
            self.commands.agent_state.invalidate_environment(fault,self.environment_epoch)

    def _take_environment(self, previous_epoch):
        with self.lock:
            self.scene,self.device=self.pending
            epoch,fault,offered=self.environment_epoch,self.environment_fault,self.offered_mono
        if epoch!=previous_epoch:
            self.reference.invalidate('environment_voxel_budget_exceeded')
            self.controller.controller.clear()
            self.preview_token=None
        return epoch,fault,offered

    def _search_scene(self,scene):
        from perception.environment.scene import assemble_scene,people_observation
        from perception.environment.navigation import CAPSULE_FIELDS
        if self.device.metadata['snapshot_id']!=scene.geometry.metadata['snapshot_id']:raise ValueError('Reference/device observation mismatch')
        if 'navigation_fusion' in self.device.metadata and all(k in scene.geometry.layers for k in CAPSULE_FIELDS):
            # The producer already exported and bound this exact capsule. Its
            # host arrays remain owned by the scene through the search lease.
            return scene
        names=CAPSULE_FIELDS if 'navigation_fusion' in self.device.metadata else ('occupancy','ground_support','coverage','last_observed','near_field_free')
        fields=[k for k in names if k in self.device._layers]
        grid=self.device.materialize(consumer='reference_search',fields=fields,metrics=self.metrics)
        return assemble_scene(grid,clock_domain=scene.record['clock_domain'],people=people_observation(scene),
                              stamp_ns=scene.record.get('stamp_ns'),settings=self.scene_settings)

    def _preview_frozen(self, token, environment_epoch):
        """Publish the search raster only for the preview accepted by the bag session."""
        key=(token,self.scene is not None)
        if key==self.preview_token:
            return
        self.preview_token=key
        if self.display is not None:
            self.display.frozen_basemap(True)
        if self.request is None:
            return
        snapshot=None
        if self.scene is None:
            result=dict(scope='frozen_bag_observation',reason='observation_unavailable',executable=False)
        else:
            from navigation.planning.frozen import preview

            mapped=[]
            def capture(grid,local_from_map):
                mapped.append((grid,local_from_map))

            self.batch.bind(self.device)
            scene=self._search_scene(self.scene)
            result=preview(self.request,scene,list(self.rows),self.controller.controller,
                self.reference.planning,asset=self.asset,dt=self.settings['control_period_s'],
                on_mapped_grid=capture if self.display is not None and self.reference.display_enabled else None)
            if mapped:
                from navigation.planning.runtime import encode_display_map

                grid,local_from_map=mapped[0]
                snapshot=encode_display_map(grid,local_from_map,token,result['reference'])
            if self.display is not None:
                from perception.visualization.interactive import preview_geometry
                from perception.visualization.surface import DisplayGrids

                # A query-only copy avoids sharing the display thread's mutable cache.
                grid=DisplayGrids(self.metrics,asset=self.asset).get(self.device)
                result['display']=preview_geometry(result,grid,self.asset,self.display.buffer.settings)
        with self.lock, self.session.lock:
            if environment_epoch!=self.environment_epoch or not self.session.pause_requested:
                return
            if self.session.publish_preview(result,token) and snapshot is not None:
                self.display.frozen_basemap(True,snapshot)

    def _work(self):
        try:
            import torch
            torch.cuda.set_device(0);stream=torch.cuda.Stream(priority=-1)
            next_tick=time.monotonic();last_tick=None;last_state=None
            previous_environment_epoch=0
            frozen_display=False
            with torch.cuda.stream(stream):
                while not self.done.is_set():
                    while True:
                        try:packet=json.loads(self.pose_socket.recv(262144))
                        except BlockingIOError:break
                        row=packet['state'];self.clock=packet
                        if self.rows and (row['stamp_s']<=self.rows[-1]['stamp_s'] or
                            (row['session_id'],row['generation'])!=(self.rows[-1]['session_id'],self.rows[-1]['generation'])):
                            self.rows.clear();self.reference.invalidate('localization_reset')
                        self.rows.append(row)
                    epoch=self.session_epoch
                    pause_requested=paused=False
                    if self.session is not None:
                        with self.session.lock:
                            epoch=self.session.epoch
                            pause_requested,paused=self.session.pause_requested,self.session.paused
                            # A command offered after this read cannot accept the previous goal's preview.
                            token=self.session.command_generation
                    if epoch!=self.session_epoch:
                        self.session_epoch=epoch
                        self.commands.agent_state.clear_follow_observations()
                        if self.follow is not None:
                            handle=self.follow.handle or (self.follow_pending or {}).get('handle')
                            if handle is not None:
                                stop=self.commands.follow_failed(handle)
                                target=None if stop is None else stop['target']
                                self.commands.agent_state.publish_follow(dict(status='failed',handle=handle,
                                    reason='bag_pause_or_resume',cancelled_request=target))
                                if target is not None and self.request is not None and self.request['task_id']==target['task_id']:
                                    self.cancelled_request=target;self.request=None
                            with self.lock:self.follow.clear();self.follow.history.rows.clear()
                            if self.follow_decision is not None:self.follow_decision.stop()
                            self.follow_pending=None
                            self.follow_target=None
                            self.follow_issued=None;self.follow_held=False
                            self.follow_search_anchor=self.follow_search_choice=None
                            self.follow_search_intent=None
                        if self.request is not None:
                            self.safety.command('cancel',{k:self.request[k] for k in ('task_id','goal_revision')})
                        self.request=None;self.reference.invalidate('bag_pause_or_resume');self.controller.controller.clear()
                    for kind,value in self.commands.drain():
                        if kind=='follow_start':
                            self.follow_pending=value
                            self.follow_target=None
                            self.follow_issued=None;self.follow_held=False
                            self.manual_follow=value.get('manual',False)
                            self.follow_search_anchor=self.follow_search_choice=None
                            self.follow_search_intent=None
                            if self.follow is not None:
                                with self.lock:
                                    self.follow.handle=copy.deepcopy(value['handle'])
                                    if getattr(self.follow_decision,'backend',None)=='reid':
                                        self.follow.reid_reference_id=value['intent']['reference']['reference_id']
                            if self.follow_decision is not None:
                                self.follow_decision.begin(value['handle'],value['intent']['target_description'],
                                    reference=value['intent']['reference'],crop=value['reference_crop'])
                            self.commands.agent_state.publish_follow(dict(status='starting',handle=value['handle'],reason=None))
                        elif kind=='follow_stop':
                            self.follow_target=None
                            self.follow_issued=None;self.follow_held=False
                            evidence=None if self.follow is None else copy.deepcopy(self.follow.last_evidence)
                            if self.follow is not None:
                                with self.lock:self.follow.clear()
                            if self.follow_decision is not None:self.follow_decision.stop()
                            self.follow_pending=None
                            self.follow_search_anchor=self.follow_search_choice=None
                            self.follow_search_intent=None
                            if self.request is not None and self.request['task_id']==value['target']['task_id']:
                                self.reference.cancel(self.request);self.request=None;self.controller.controller.clear()
                            self.cancelled_request=value['target']
                            old=self.commands.agent_state.snapshot().get('follow') or {}
                            self.commands.agent_state.publish_follow(dict(status='failed' if value.get('lost') else 'stopped',
                                handle=old.get('handle'),reason=value['reason'],cancelled_request=value['target'],
                                last_evidence=evidence,recovery_available=bool(value.get('lost') and value.get('retained'))))
                        elif kind=='goal':
                            if self.follow is not None and (self.follow.handle is not None or self.follow_pending is not None):
                                old=self.commands.agent_state.snapshot().get('follow') or {}
                                with self.lock:self.follow.clear()
                                if self.follow_decision is not None:self.follow_decision.stop()
                                self.follow_pending=None
                                self.follow_target=None
                                self.follow_issued=None;self.follow_held=False
                                self.follow_search_anchor=self.follow_search_choice=None
                                self.follow_search_intent=None
                                self.commands.agent_state.publish_follow(dict(status='preempted',handle=old.get('handle'),reason='operator_goal'))
                            self.request=value;self.reference.invalidate('goal_changed');self.controller.controller.clear()
                        elif kind=='cancel' and (self.request is None or
                            value['task_id']==self.request['task_id'] and value['goal_revision']>=self.request['goal_revision']):
                            if self.request is not None:self.reference.cancel(self.request)
                            self.request=None;self.controller.controller.clear();self.cancelled_request=value
                    freezing=pause_requested or paused
                    if freezing!=frozen_display:
                        frozen_display=freezing
                        if self.display is not None:
                            self.display.frozen_basemap(freezing)
                    if freezing:
                        if paused and pause_requested and self.rows:
                            previous_environment_epoch,_,_=self._take_environment(previous_environment_epoch)
                            self.session.publish_observation(self.scene,self.rows[-1],goal_intent=self.request)
                            self._preview_frozen(token,previous_environment_epoch)
                        elif self.preview_token is not None:
                            self.preview_token=None
                            if self.display is not None:
                                self.display.frozen_basemap(True)
                        self.done.wait(.01);continue
                    mono=time.monotonic()
                    if self.clock is None:self.done.wait(.005);continue
                    period=self.settings['control_period_s']
                    fresh=self.rows[-1]['stamp_s']!=last_state
                    earliest=mono if last_tick is None else last_tick+period*.8
                    if not (fresh and mono>=earliest) and mono<next_tick:
                        self.done.wait(min(.005,next_tick-mono));continue
                    dt=period if last_tick is None else min(period,mono-last_tick)
                    last_tick=mono;last_state=self.rows[-1]['stamp_s']
                    # Wait for the next real measurement rather than calculating
                    # just before its arrival. A monotonic watchdog still emits
                    # unavailable proposals when the sensor pipeline stalls.
                    next_tick=mono+max(period,self.controller.motion['state_max_age_s'])
                    previous=self.scene
                    environment_epoch,environment_fault,offered_mono=self._take_environment(previous_environment_epoch)
                    if self.session is not None:self.session.publish_observation(self.scene,self.rows[-1],goal_intent=self.request)
                    if environment_epoch==previous_environment_epoch and previous is not None and self.scene is None:
                        self.reference.invalidate('localization_invalid')
                    previous_environment_epoch=environment_epoch
                    now=self.clock['clock_s']+(mono-self.clock['clock_mono'])*self.clock['clock_rate']
                    self.reference.observe(self.scene,self.asset,now=now)
                    if self.follow is not None and environment_fault is None:
                        self._update_follow(now)
                    self.metrics.add('feedback_age_at_control',max(0.,now-self.rows[-1]['stamp_s']))
                    if self.route_replay is not None:
                        if self.commands.route_preempted:self.route_replay.disable('preempted_by_external_command')
                        action=self.route_replay.step(self.scene,self.rows[-1],now=now,clock_domain=self.clock_domain,
                            failure=self.reference.failure)
                        accepted=False
                        if action is not None:
                            kind,value=action;accepted=self.commands.route_command(kind,value)
                            if accepted:
                                if kind=='goal':
                                    self.request=value;self.reference.invalidate('route_goal_changed')
                                else:
                                    if self.request is not None:self.reference.cancel(self.request)
                                    self.cancelled_request=value;self.request=None
                                self.controller.controller.clear()
                        for event in self.route_replay.drain_events():
                            self.record('route',dict(event,command_kind=None if action is None else action[0],
                                command_accepted=accepted if action is not None else None))
                    if self.scene is not None:
                        self.batch.bind(self.device)
                    preview=self.commands.manual_preview_snapshot() if self.session is not None else None
                    preview_request=None if preview is None or self.request is not None else preview['request']
                    planning_request=self.request if self.request is not None else preview_request
                    target=getattr(self,'follow_target',None)
                    follow_key=(target if target is not None and planning_request is not None and
                                planning_request['request_id']==target[0] and
                                planning_request.get('completion_policy')=='continuous' else None)
                    preview_key=None if preview_request is None else (preview['owner'],preview_request['request_id'],
                        preview_request['goal_revision'] if follow_key is None else follow_key)
                    if self.request is None and preview_key!=self.manual_preview_key:
                        self.reference.invalidate('manual_preview_changed')
                    self.manual_preview_key=preview_key
                    status=None
                    tracing=self.metrics.trace_path is not None
                    if tracing:query_mono=time.monotonic()
                    if planning_request is not None and self.scene is not None:
                        self.reference.update(planning_request,self.scene,None,self.asset,now=now,
                            clock_rate=self.clock['clock_rate'],follow_key=follow_key,
                            locked_follow_goal=getattr(self,'follow_held',False))
                        with self.metrics.span('reference_tick'):status=self.reference.tick(now=now,clock_domain=self.clock_domain)
                        self.record('reference',dict(event='status',as_of=now,reason=status['reason'],blocked=self.reference.blocked,
                            waiting=status['waiting'],start_evidence=status['start_evidence']))
                    ready_request=None if status is None else status.get('request')
                    available_reference=None if status is None else status.get('result',self.reference.current)
                    held=getattr(self,'follow_held',False)
                    from navigation.planning.continuity import goal_binding
                    if held and available_reference is not None:
                        result=available_reference['result']
                        issued,adjustment=self.follow_issued
                        selected=(result.get('follow_goal_adjustment') or {}).get('selected_goal')
                        if goal_binding(result['request'])!=goal_binding(issued) or selected!=adjustment:
                            status=dict(status,result=None,request=None,status='unavailable',
                                        reason='follow_reid_issued_goal_reference_changed')
                            ready_request=available_reference=None
                    elif (preview_request is not None and available_reference is not None and
                            follow_key is not None and getattr(self.follow_decision,'backend',None)=='reid'):
                        result=available_reference['result']
                        if goal_binding(result['request'])==goal_binding(preview_request):
                            selected=(result.get('follow_goal_adjustment') or {}).get('selected_goal')
                            self.follow_issued=(copy.deepcopy(preview_request),copy.deepcopy(selected))
                    display_request=ready_request or planning_request
                    control_request=self.request
                    if follow_key is not None and control_request is not None:
                        control_request=ready_request
                        if control_request is not None and not self.commands.follow_goal(self.follow.handle,control_request):
                            self.reference.invalidate('follow_owner_lost_before_handover')
                            control_request=None
                        elif (control_request is not None and getattr(self.follow_decision,'backend',None)=='reid' and
                                goal_binding(control_request)==goal_binding(planning_request)):
                            adjustment=(available_reference['result'].get('follow_goal_adjustment') or {}).get('selected_goal')
                            self.follow_issued=(copy.deepcopy(control_request),copy.deepcopy(adjustment))
                    control_reference = available_reference if control_request is not None else None
                    if tracing:
                        ready_mono=time.monotonic()
                        reference_request=None if available_reference is None else available_reference['result']['request']
                    with self.metrics.span('dwa_control'):
                        packet=self.controller.step(control_request,self.scene,list(self.rows),control_reference,
                            now=now,clock_domain=self.clock_domain,dt=dt,reason=environment_fault)
                    self.reference.note_control(packet,monotonic_s=mono)
                    self.safety.packet(packet);self.record('control',packet);self.count+=1
                    if tracing:
                        r=None if self.scene is None else self.scene.record
                        self.metrics.trace('control_cycle',sequence_id=packet['sequence_id'],now=now,
                            control_started_mono=mono,scene_offered_mono=offered_mono,
                            reference_epoch=self.reference.epoch,
                            goal_key=None if planning_request is None else [planning_request[k] for k in ('task_id','request_id','goal_revision')],
                            reference_goal_key=None if reference_request is None else [reference_request[k] for k in ('task_id','request_id','goal_revision')],
                            goal_checked_mono=query_mono,reference_checked_mono=ready_mono,
                            scene_binding=None if r is None else r['input_fingerprint'],
                            measurement_stamp_s=None if r is None else r['as_of'],
                            association_stamp_s=None if r is None else r['prior'].get('measurement_stamp_s'),
                            association_valid=False if r is None else r['prior'].get('valid',False),
                            proposal_reason=(packet.get('proposal') or {}).get('reason'),
                            proposal_status=(packet.get('proposal') or {}).get('status'))
                    if self.planning_recorder is not None:
                        r=None if self.scene is None else self.scene.record
                        measured=None if r is None else r['prior'].get('measurement_stamp_s')
                        failure=(packet.get('proposal') or {}).get('reason')
                        capture_required=bool(status and status.get('reason')=='start_footprint_not_free') or failure in (
                            'current_footprint_not_free','certification_time_budget_exceeded','certification_work_budget_exceeded')
                        if self.scene is not None:
                            ground=self.scene.geometry.metadata.get('ground_tracking',{})
                            capture_required|=bool(ground.get('history_suspended') or ground.get('reference_pending'))
                        self.planning_recorder.submit(packet,self.scene,self.device,control_reference,
                            self.controller.controller.result,self.reference.worker.job,self.reference.last_reply,dt=dt,
                            capture_required=capture_required,
                            timing=dict(control_started_mono=mono,scene_offered_mono=offered_mono,
                                scene_age_s=None if r is None else now-r['as_of'],
                                association_age_s=None if measured is None else now-measured,
                                feedback_age_s=None if not packet.get('feedback') or packet['feedback']['stamp_s'] is None else now-packet['feedback']['stamp_s']))
                    prior=None if self.scene is None else self.scene.record['prior']
                    binding=None if prior is None else {k:prior.get(k) for k in
                        ('valid','map_id','map_frame','map_revision','association_session','association_generation')}
                    idle_feedback=None
                    if self.request is None and self.scene is not None and self.commands.agent_state.goal_evidence is not None:
                        from navigation.control.shadow import observed_feedback
                        idle_feedback=observed_feedback(list(self.rows),self.scene,self.controller.geometry,self.controller.motion,now=now)
                    preview_status=None
                    reference_observation=None if status is None else status.get('planning_observation')
                    if preview is not None:
                        preview_status=dict(owner=preview['owner'],request_id=preview['request']['request_id'],
                            status=('blocked_by_observation_goal' if self.request is not None else
                                    'waiting_for_observation' if status is None else status['status']),
                            reason=('another_goal_active' if self.request is not None else
                                    environment_fault or 'observation_unavailable' if status is None else status.get('reason')),
                            reference_stamp_s=None if available_reference is None or self.request is not None
                                              else available_reference['as_of'],executable=False)
                        if self.request is None:
                            preview_status['goal']=preview['request']['goal']
                            preview_status['goal_revision']=preview['request']['goal_revision']
                            preview_status['planning_observation']=reference_observation
                            preview_status['reference_path']=manual_preview_path(available_reference,display_request)
                            preview_status['reference_frame']=preview['request']['frame_id']
                            if preview_status['reference_path'] is not None:
                                preview_status['path_result_binding']=available_reference['result'].get('result_binding')
                    control_observation=None
                    if self.request is not None and self.scene is not None and environment_fault is None:
                        control_observation=control_planning_observation(packet,self.scene,
                            self.reference.planning['max_geometry_age_s'])
                    mapped_pose=agent_mapped_pose(self.scene,self.rows[-1],self.controller.geometry,now,
                        self.scene_settings['max_association_age_s'],self.clock_domain) if self.commands.agent_state.goal_evidence is not None else None
                    self.commands.agent_state.publish_control(packet,self.rows[-1],map_binding=binding,
                        cancelled_request=self.cancelled_request,idle_feedback=idle_feedback,manual_preview=preview_status,
                        reference_observation=reference_observation if self.request is not None else None,
                        control_observation=control_observation,
                        agent_mapped_pose=mapped_pose,environment_epoch=environment_epoch,environment_fault=environment_fault)
                    if self.display is not None:
                        self.display.navigation(self.controller.controller.result,available_reference,
                            display_request,self.scene,self.rows[-1],packet,
                            self.reference.reason if status is None else status['reason'],now=now,clock_domain=self.clock_domain,
                            clock_rate=self.clock['clock_rate'],clock_mono=mono,grid=None if self.scene is None else self.device,
                            search_map=self.reference.display_map if status is not None else None,
                            goal_hint=planning_request if follow_key is not None else None)
                    for event in self.reference.drain_events():self.record('reference',event)
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def follow_context(self):
        with self.lock:
            last=None if self.follow is None else (self.follow.reid_last_evidence
                if getattr(getattr(self,'follow_decision',None),'backend',None)=='reid' else self.follow.last_evidence)
            return dict(last=copy.deepcopy(last),
                        anchor=copy.deepcopy(self.follow_search_anchor))

    def _follow_inputs(self, scene):
        import numpy as np
        from navigation.planning.inputs import association_status
        if scene is None or self.asset is None or association_status(scene,self.asset,self.reference.planning):
            raise ValueError('follow_search_map_unavailable')
        record=scene.record
        local=np.asarray(record['localization']['world_from_lidar'],dtype=float)
        transform=np.asarray(record['prior']['map_from_local'],dtype=float)
        geometry=self.controller.controller.geometry
        return dict(stamp_s=record['as_of'],clock_domain=record['clock_domain'],scene=scene,
            geometry=geometry,global_grid=self.asset.snapshot,local_pose=geometry.pose(local),
            pose=geometry.pose(transform@local))

    def _pause_follow(self, handle):
        self.follow_target=None
        self.follow_issued=None
        self.follow_held=False
        target=self.commands.pause_follow(handle)
        self.follow.revision_floor=max(self.follow.revision_floor,target['goal_revision'])
        if self.request is not None and self.request.get('request_id')==handle['call_id']:
            self.reference.cancel(self.request);self.request=None;self.controller.controller.clear()
        self.cancelled_request=target

    def _checked_follow_search_pose(self,choice,now,*,fixed=False):
        from navigation.search import check_last_position,check_pose,current_search_pose
        if fixed:
            check=check_last_position(self.scene,self.asset,self.controller.controller.geometry,choice['pose'],
                self.reference.planning,now=now,clock_domain=self.clock_domain)
            if 'check' in choice and choice['check'].get('map_context')!=check['map_context']:
                raise ValueError('follow_last_position_map_changed')
            return choice['pose']
        inputs=self._follow_inputs(self.scene)
        if self.follow_search_anchor is None:
            self.follow_search_anchor=list(inputs['pose'][:2])
        target=current_search_pose(choice,inputs['pose'])
        check=check_pose(inputs,target,now=now,clock_domain=self.clock_domain,
                         anchor=self.follow_search_anchor,margin=self.reference.planning['safety_margin_m'])
        return target if check['valid'] else None

    def _follow_search_request(self, handle, choice, binding, now, valid_until, *, fixed=False):
        if choice.get('kind')=='wait' or now>valid_until:return None
        target=self._checked_follow_search_pose(choice,now,fixed=fixed)
        if target is None:return None
        self.follow.revision_floor+=1
        map_id=self.asset.manifest['snapshot_id']
        request=dict(task_id='follow:'+handle['call_id'],request_id=handle['call_id'],
            goal_revision=self.follow.revision_floor,mode='mapped',frame_id=self.asset.snapshot.metadata['frame_id'],
            target_reference='base_link',goal=target,goal_behavior='stop',
            completion_policy='goal',clock_domain=self.clock_domain,
            valid_until_s=valid_until,
            local_identity=[self.scene.record[k] for k in ('session_id','generation','frame_id')],
            map_snapshot_id=map_id)
        if fixed:
            request['follow_last_evidence_binding']=choice['last_evidence_binding']
        else:
            request['search_bounds']=dict(anchor_xy_m=copy.deepcopy(self.follow_search_anchor),radius_m=1.5,
                                          map_snapshot_id=map_id,loss_binding=binding)
        return request

    def _renew_reid_search(self, handle, action, now):
        from navigation.planning.continuity import context_binding

        active=getattr(self,'follow_search_intent',None)
        if active is None or active['continuation']['intent_id']!=action['continuation']['intent_id']:
            if 'continuation_id' in action['candidate']:
                raise ValueError('follow_reid_search_intent_inactive')
            return False
        request=active['request']
        scene=self.scene.record
        if (active['continuation']!=action['continuation'] or now>request['valid_until_s'] or
                action['valid_until_s']<request['valid_until_s'] or self.asset is None or
                request['map_snapshot_id']!=self.asset.manifest['snapshot_id'] or
                request['local_identity']!=[scene[k] for k in ('session_id','generation','frame_id')] or
                context_binding(request,self.scene,self.asset)!=active['context'] or
                action['candidate'].get('continuation_id')!=active['continuation']['intent_id'] or
                self._checked_follow_search_pose(action['candidate'],now,fixed=True)!=request['goal']):
            raise ValueError('follow_reid_search_renewal_unavailable')
        request=dict(request,valid_until_s=action['valid_until_s'])
        if getattr(self,'manual_follow',False):self._publish_follow_preview(handle,request)
        else:self.request=request
        active['request']=request
        active['scene_binding']=scene['input_fingerprint']
        return True

    def _reid_search_stop_reason(self, active, now):
        from navigation.planning.continuity import context_binding, goal_binding

        request=active['request']
        if now>request['valid_until_s']:
            return 'follow_reid_search_observation_expired'
        failure=getattr(self.reference,'failure',None)
        if failure is not None and failure.get('goal_binding')==goal_binding(request):
            return failure['reason']
        if self.scene is None:
            return 'follow_search_scene_unavailable'
        if context_binding(request,self.scene,self.asset)!=active['context']:
            return 'follow_reid_search_context_changed'
        if now-self.scene.record['as_of']>self.reference.planning['max_geometry_age_s']:
            return 'geometry_future_or_stale'
        binding=self.scene.record['input_fingerprint']
        if binding!=active['scene_binding']:
            try:
                self._checked_follow_search_pose(dict(pose=request['goal']),now,fixed=True)
            except (ValueError,KeyError,TypeError) as error:
                return str(error)
            active['scene_binding']=binding
        return None

    def _follow_rotation_ready(self, now):
        from navigation.control.shadow import observed_feedback
        if self.scene is None:return False
        feedback=observed_feedback(list(self.rows),self.scene,self.controller.geometry,self.controller.motion,now=now)
        return feedback['status']=='valid' and 'stationary_evidence' in feedback

    def _wait_reid(self, handle):
        if (self.follow.policy is not None or self.follow_search_choice is not None or
                getattr(self,'follow_search_intent',None) is not None or
                self.request is not None and self.request.get('request_id')==handle['call_id'] or
                getattr(self,'manual_follow',False)):
            self._pause_follow(handle)
        with self.lock:
            if self.follow.policy is not None:self.follow.policy.cancel()
            self.follow.policy=self.follow.request=None
        self.follow_search_choice=None
        self.follow_search_intent=None

    def _hold_reid_goal(self, handle, now):
        issued=getattr(self,'follow_issued',None)
        with self.lock:
            request=self.follow.hold_reid(None if issued is None else issued[0],self.scene,now=now)
        self.follow_held=True
        self.follow_issued=(request,issued[1])
        if getattr(self,'manual_follow',False):
            self._publish_follow_preview(handle,request)
        else:
            self.request=request

    def _update_follow(self, now):
        from decision.vlm.project_config import load_config
        worker=getattr(self,'follow_decision',None)
        pending=self.follow_pending
        handle=pending['handle'] if pending is not None else self.follow.handle
        if handle is None:return
        local=getattr(worker,'backend',None)=='reid'
        search_reason=None
        try:
            if not self.commands.follow_owner_valid(handle):
                raise ValueError('follow_owner_arming_or_agent_heartbeat_lost')
            if worker is None:raise ValueError('follow_t2_worker_unavailable')
            outcome=worker.poll()
            action=None
            if outcome is not None:
                if outcome.get('handle')!=handle:raise ValueError('follow_t2_handle_changed')
                if outcome.get('transient'):
                    search_reason=outcome.get('reason',outcome['error'])
                    self.record('reference',dict(event='follow_t2_input_unavailable',reason=search_reason))
                    if local:self._wait_reid(handle)
                    worker.request_fresh();outcome=None
                elif outcome.get('error'):raise ValueError('follow_t2_'+outcome.get('reason',outcome['error']))
            if outcome is not None and local:
                evidence=outcome['local'];source=evidence['source']
                self.record('reference',dict(event='follow_t2_result',backend='reid',request_binding=evidence['binding'],
                    source_stamp_s=source['stamp_s'],rgb_stamp_ns=source['rgb_stamp_ns'],
                    scene_binding=source['scene_binding'],candidate_binding=source['binding'],
                    reid_model=evidence.get('reid_model'),reid_model_sha256=evidence['reid_model_sha256'],
                    selection=evidence['decision'],loss=evidence['loss'],
                    search_choice=evidence['selected'],timing=outcome['timing']))
                try:
                    with self.lock:action=self.follow.apply_reid(handle,evidence,self.scene,now=now)
                except (ValueError,KeyError,TypeError) as error:
                    search_reason=str(error)
                    self._wait_reid(handle)
                    worker.interrupt_reid()
                    self.record('reference',dict(event='follow_t2_rejected',reason=search_reason,
                        request_binding=evidence['binding'],source_stamp_s=source['stamp_s'],apply_stamp_s=now))
                self.metrics.trace('t2_result_applied',measurement_stamp_s=source['stamp_s'],
                                   apply_stamp_s=now,accepted=action is not None,reason=search_reason)
            elif outcome is not None:
                result=outcome['result'];prepared=outcome['prepared']
                self.record('reference',dict(event='follow_t2_result',request_binding=prepared['binding'],
                    backend=result.get('backend','dspark'),
                    transport_request_binding=result.get('transport_request_binding'),
                    source_stamp_s=prepared['source_stamp_s'],raw_content=result.get('raw_content'),
                    selection=result.get('selection'),
                    clock_domain=prepared['clock_domain'],scene_binding=prepared['scene_binding'],
                    window_binding=prepared['window_binding'],search_binding=prepared['search_binding'],
                    choices=copy.deepcopy(prepared['choices']),images=copy.deepcopy(outcome['window']['images']),
                    status=result.get('status'),reason=result.get('reason'),timing=result.get('timing')))
                if result.get('status')!='complete':raise ValueError('follow_t2_'+str(result.get('reason','unavailable')))
                try:
                    with self.lock:
                        action=self.follow.apply_t2(handle,prepared,result,outcome['window'],self.scene,now=now)
                except ValueError as error:
                    self.metrics.trace('t2_result_applied',measurement_stamp_s=prepared['source_stamp_s'],
                                       apply_stamp_s=now,accepted=False,reason=str(error))
                    self.record('reference',dict(event='follow_t2_rejected',reason=str(error),
                        request_binding=prepared['binding'],source_stamp_s=prepared['source_stamp_s'],apply_stamp_s=now,
                        current_scene_stamp_s=None if self.scene is None else self.scene.record['as_of']))
                    worker.request_fresh()
                else:
                    self.metrics.trace('t2_result_applied',measurement_stamp_s=prepared['source_stamp_s'],
                                       apply_stamp_s=now,accepted=True)
            if action is not None and action['operation']=='plan':
                if local and getattr(self,'follow_search_intent',None) is not None:
                    self._pause_follow(handle)
                self.follow_pending=None;self.follow_search_anchor=self.follow_search_choice=None
                self.follow_search_intent=None
            elif action is not None and action['operation']=='continue':
                worker.request_fresh();action=None
            elif action is not None and action['operation']=='wait':
                search_reason=action['reason']
                self._wait_reid(handle)
            elif action is not None and action['operation']=='search':
                if local:
                    continuation=action['continuation'];old=self.follow_search_choice
                    search_reason=action['reason']
                    try:
                        renewed=continuation is not None and self._renew_reid_search(handle,action,now)
                    except (ValueError,KeyError,TypeError) as error:
                        renewed=True
                        search_reason=str(error)
                        self._wait_reid(handle);worker.interrupt_reid()
                        self.record('reference',dict(event='follow_search_rejected',reason=search_reason))
                    if not renewed:
                        if old is None or len(old)!=4 or old[3]!=continuation:self._wait_reid(handle)
                        self.follow_search_choice=(action['candidate'],action['search_binding'],action['valid_until_s'],continuation)
                else:
                    self._pause_follow(handle)
                    deadline=outcome['window']['stamp_s']+load_config()['tasks']['follow']['decision_deadline_s']
                    self.follow_search_choice=(action['candidate'],action['search_binding'],deadline)
                if not local and self.follow_search_anchor is None:
                    try:self.follow_search_anchor=list(self._follow_inputs(self.scene)['pose'][:2])
                    except (ValueError,KeyError,TypeError):pass
                self.follow_pending=None
            holding=local and (action is not None and action['operation']=='hold' or
                action is None and self.follow.policy is not None)
            if holding:
                try:
                    self._hold_reid_goal(handle,now)
                except (ValueError,KeyError,TypeError) as error:
                    search_reason=str(error)
                    self._wait_reid(handle)
                    worker.request_fresh()
                else:
                    self.commands.agent_state.publish_follow(dict(status='running',handle=handle,
                        target_identity=list(self.follow.policy.target),target_generation=self.follow.policy.tracked_generation,
                        reason='follow_reid_finishing_issued_goal'))
                    return
            elif action is None and self.follow.policy is not None:
                with self.lock:action=self.follow.update(self.scene,now=now)
            if action is not None and action['operation']=='stop':
                if not local and action.get('reason') not in ('follow_track_lost','follow_track_continuity_lost',
                        'follow_visual_confirmation_expired','follow_visual_occlusion_expired'):
                    raise ValueError(action.get('reason','follow_stopped'))
                if local:
                    search_reason=action.get('reason')
                    self._wait_reid(handle);worker.interrupt_reid()
                else:
                    self._pause_follow(handle)
                    with self.lock:self.follow.policy=None;self.follow.request=None
                    self.follow_search_choice=None
                worker.request_fresh();action=None
            if action is not None and action['operation']=='plan':
                from navigation.planning.continuity import goal_binding

                request=action['request']
                target=(handle['call_id'],tuple(action['target_identity']),action.get('target_generation'))
                old_target=getattr(self,'follow_target',None)
                if old_target is not None and target!=old_target:
                    self._pause_follow(handle)
                self.follow_target=target
                self.follow_held=False
                if getattr(self,'manual_follow',False):
                    issued=self._publish_follow_preview(handle,request)
                else:
                    issued=self.request=request
                previous=getattr(self,'follow_issued',None)
                if local:
                    adjustment=previous[1] if previous is not None and goal_binding(previous[0])==goal_binding(issued) else None
                    self.follow_issued=(copy.deepcopy(issued),adjustment)
                self.commands.agent_state.publish_follow(dict(status='running',handle=handle,
                    target_identity=action['target_identity'],target_generation=action.get('target_generation'),reason=None))
                return
            active=getattr(self,'follow_search_intent',None)
            if local and active is not None:
                reason=self._reid_search_stop_reason(active,now)
                if reason is not None:
                    self._wait_reid(handle);worker.interrupt_reid()
                    search_reason=reason
            if self.follow.policy is None and self.follow_search_choice is not None:
                choice,binding,valid_until,*intent=self.follow_search_choice
                if now>valid_until:
                    self.follow_search_choice=None
                    self.record('reference',dict(event='follow_search_rejected',reason='search_observation_expired'))
                elif choice.get('kind')=='rotate' and not self._follow_rotation_ready(now):
                    search_reason='follow_search_waiting_for_stationary'
                    if action is not None and action['operation']=='search':
                        self.record('reference',dict(event='follow_search_deferred',reason=search_reason,as_of=now))
                else:
                    self.follow_search_choice=None
                    try:request=self._follow_search_request(handle,choice,binding,now,valid_until,fixed=local)
                    except (ValueError,KeyError,TypeError) as error:
                        request=None;self.record('reference',dict(event='follow_search_rejected',reason=str(error)))
                    if request is not None:
                        if local:
                            from navigation.planning.continuity import context_binding
                            self.follow_search_intent=dict(continuation=intent[0],request=request,
                                scene_binding=self.scene.record['input_fingerprint'],
                                context=context_binding(request,self.scene,self.asset))
                        if getattr(self,'manual_follow',False):
                            self._publish_follow_preview(handle,request)
                        else:
                            if not self.commands.follow_goal(handle,request):raise ValueError('follow_owner_arming_or_agent_heartbeat_lost')
                            self.request=request;self.reference.invalidate('follow_search_goal');self.controller.controller.clear()
            if self.follow.policy is None and self.request is not None and self.request.get('request_id')==handle['call_id']:
                if not self.commands.follow_goal(handle,self.request):
                    raise ValueError('follow_owner_arming_or_agent_heartbeat_lost')
            status='starting' if pending is not None else 'searching'
            self.commands.agent_state.publish_follow(dict(status=status,handle=handle,reason=search_reason,
                last_evidence=copy.deepcopy(self.follow.last_evidence)))
        except (ValueError,KeyError,TypeError) as error:
            self.follow_target=None
            self.follow_issued=None;self.follow_held=False
            reason=str(error)
            stop=self.commands.follow_failed(handle,recoverable=True)
            target=None if stop is None else stop['target']
            last_evidence=copy.deepcopy(self.follow.last_evidence)
            if self.request is not None and self.request.get('request_id')==handle['call_id']:
                self.reference.cancel(self.request);self.request=None;self.controller.controller.clear()
            if target is not None:self.cancelled_request=target
            if worker is not None:worker.stop()
            with self.lock:self.follow.clear()
            self.follow_pending=None;self.follow_search_anchor=self.follow_search_choice=None
            self.follow_search_intent=None
            self.commands.agent_state.publish_follow(dict(status='failed' if target is not None else 'preempted',
                handle=handle,reason=reason,cancelled_request=target,last_evidence=last_evidence,
                recovery_available=False))

    def _publish_follow_preview(self,handle,request):
        import numpy as np
        import math
        inputs=self._follow_inputs(self.scene)
        scene=self.scene.record
        target=request['goal']
        if request['mode']=='local':
            if request['frame_id']!=scene['frame_id'] or request.get('local_identity')!=[scene[k] for k in ('session_id','generation','frame_id')]:
                raise ValueError('follow_preview_localization_changed')
            matrix=np.asarray(scene['prior']['map_from_local'])
            xy=(matrix @ [target[0],target[1],0.,1.])[:2].tolist()
            target=xy if len(target)==2 else xy+[target[2]+math.atan2(matrix[1,0],matrix[0,0])]
        preview=dict(request,mode='mapped',frame_id=inputs['global_grid'].metadata['frame_id'],
                     map_snapshot_id=self.asset.manifest['snapshot_id'],goal=target)
        self.commands.follow_preview(handle,preview)
        return preview

    def close(self):
        self.done.set();self.worker.join(timeout=10)
        if self.worker.is_alive():raise RuntimeError('Control loop did not stop')
        if self.error:raise RuntimeError('Control loop failed: '+self.error)
        return self.count
