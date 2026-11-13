"""Clocked control scheduling beside perception, sharing sealed GPU observations."""
from collections import deque
import copy
import json
import threading
import time


class ControlLoop:
    def __init__(self,reference,controller,batch,safety,pose_socket,commands,settings,scene_settings,metrics,record,*,asset=None,initial_goal=None,clock_domain='runtime_bag',display=None,route_replay=None,planning_recorder=None,session=None):
        self.reference,self.controller,self.batch,self.safety=reference,controller,batch,safety
        self.pose_socket,self.commands=pose_socket,commands;self.settings=settings;self.scene_settings=scene_settings
        self.metrics,self.record,self.asset=metrics,record,asset;self.request=initial_goal;self.clock_domain=clock_domain;self.display=display
        self.route_replay=route_replay
        self.planning_recorder=planning_recorder;self.offered_mono=None
        self.session=session;self.session_epoch=0;self.preview_token=None
        self.manual_preview_key=None
        self.lock=threading.Lock();self.pending=(None,None);self.rows=deque(maxlen=256)
        self.device=None;self.scene=None;self.done=threading.Event();self.error=None;self.count=0;self.clock=None
        self.cancelled_request=None
        self.follow_pending=None;self.follow_visual_pending=[];self.follow_visual_review=None;self.follow=None
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
            self.metrics.trace('control_scene_offer',scene_binding=None if scene is None else scene.record['input_fingerprint'],
                measurement_stamp_s=None if scene is None else scene.record['as_of'],offered_mono=self.offered_mono)

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
        return assemble_scene(grid,clock_domain=scene.record['clock_domain'],people=people_observation(scene),settings=self.scene_settings)

    def _work(self):
        try:
            import torch
            torch.cuda.set_device(0);stream=torch.cuda.Stream(priority=-1)
            next_tick=time.monotonic();last_tick=None;last_state=None
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
                    if self.session is not None and self.session.epoch!=self.session_epoch:
                        self.session_epoch=self.session.epoch
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
                            self.follow_pending=None
                            self.follow_visual_pending.clear()
                        if self.request is not None:
                            self.safety.command('cancel',{k:self.request[k] for k in ('task_id','goal_revision')})
                        self.request=None;self.reference.invalidate('bag_pause_or_resume');self.controller.controller.clear()
                    for kind,value in self.commands.drain():
                        if kind=='follow_start':
                            self.follow_pending=value
                            self.follow_visual_review=None
                            self.commands.agent_state.publish_follow(dict(status='starting',handle=value['handle'],reason=None))
                        elif kind=='follow_visual':self.follow_visual_pending.append(value)
                        elif kind=='follow_stop':
                            evidence=None if self.follow is None else copy.deepcopy(self.follow.last_evidence)
                            if self.follow is not None:
                                with self.lock:self.follow.clear()
                            self.follow_pending=None
                            self.follow_visual_pending.clear()
                            if self.request is not None and self.request['task_id']==value['target']['task_id']:
                                self.reference.cancel(self.request);self.request=None;self.controller.controller.clear()
                            self.cancelled_request=value['target']
                            old=self.commands.agent_state.snapshot().get('follow') or {}
                            self.commands.agent_state.publish_follow(dict(status='failed' if value.get('lost') else 'stopped',
                                handle=old.get('handle'),reason=value['reason'],cancelled_request=value['target'],
                                last_evidence=evidence,recovery_available=bool(value.get('lost') and value.get('retained'))))
                        elif kind=='goal':
                            if self.follow is not None and (self.follow.policy is not None or self.follow_pending is not None):
                                old=self.commands.agent_state.snapshot().get('follow') or {}
                                with self.lock:self.follow.clear()
                                self.follow_pending=None
                                self.follow_visual_pending.clear()
                                self.commands.agent_state.publish_follow(dict(status='preempted',handle=old.get('handle'),reason='operator_goal'))
                            self.request=value;self.reference.invalidate('goal_changed');self.controller.controller.clear()
                        elif kind=='cancel' and (self.request is None or
                            value['task_id']==self.request['task_id'] and value['goal_revision']>=self.request['goal_revision']):
                            if self.request is not None:self.reference.cancel(self.request)
                            self.request=None;self.controller.controller.clear();self.cancelled_request=value
                    if self.session is not None and (self.session.pause_requested or self.session.paused):
                        if self.session.paused and self.rows:
                            with self.lock:self.scene,self.device=self.pending
                            self.session.publish_observation(self.scene,self.rows[-1],goal_intent=self.request)
                            token=self.session.command_generation
                            if self.request is not None and token!=self.preview_token:
                                self.preview_token=token
                                if self.scene is None:
                                    result=dict(scope='frozen_bag_observation',reason='observation_unavailable',executable=False)
                                else:
                                    from navigation.planning.frozen import preview
                                    self.batch.bind(self.device)
                                    scene=self._search_scene(self.scene)
                                    result=preview(self.request,scene,list(self.rows),self.controller.controller,
                                        self.reference.planning,asset=self.asset,dt=self.settings['control_period_s'])
                                    if self.display is not None:
                                        from perception.visualization.interactive import preview_geometry
                                        from perception.visualization.surface import DisplayGrids
                                        # A query-only copy avoids sharing the display thread's mutable cache.
                                        grid=DisplayGrids(self.metrics,asset=self.asset).get(self.device)
                                        result['display']=preview_geometry(result,grid,self.asset,self.display.buffer.settings)
                                self.session.publish_preview(result,token)
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
                    with self.lock:self.scene,self.device=self.pending;offered_mono=self.offered_mono
                    if self.session is not None:self.session.publish_observation(self.scene,self.rows[-1],goal_intent=self.request)
                    if previous is not None and self.scene is None:self.reference.invalidate('localization_invalid')
                    now=self.clock['clock_s']+(mono-self.clock['clock_mono'])*self.clock['clock_rate']
                    if self.follow is not None:
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
                    preview_key=None if preview_request is None else (preview['owner'],preview_request['request_id'])
                    if self.request is None and preview_key!=self.manual_preview_key:
                        self.reference.invalidate('manual_preview_changed')
                    self.manual_preview_key=preview_key
                    planning_request=self.request if self.request is not None else preview_request
                    status=None
                    if planning_request is not None and self.scene is not None:
                        self.reference.update(planning_request,self.scene,None,self.asset,now=now,clock_rate=self.clock['clock_rate'])
                        with self.metrics.span('reference_tick'):status=self.reference.tick(now=now,clock_domain=self.clock_domain)
                        self.record('reference',dict(event='status',as_of=now,reason=status['reason'],blocked=self.reference.blocked,
                            waiting=status['waiting'],start_evidence=status['start_evidence']))
                    with self.metrics.span('dwa_control'):
                        packet=self.controller.step(self.request,self.scene,list(self.rows),
                            self.reference.reference if self.request is not None else None,
                            now=now,clock_domain=self.clock_domain,dt=dt)
                    self.reference.note_control(packet,monotonic_s=mono)
                    self.safety.packet(packet);self.record('control',packet);self.count+=1
                    if self.metrics.trace_path is not None:
                        r=None if self.scene is None else self.scene.record
                        self.metrics.trace('control_cycle',sequence_id=packet['sequence_id'],now=now,
                            control_started_mono=mono,scene_offered_mono=offered_mono,
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
                        self.planning_recorder.submit(packet,self.scene,self.device,
                            self.reference.reference if self.request is not None else None,
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
                    if preview is not None:
                        preview_status=dict(owner=preview['owner'],request_id=preview['request']['request_id'],
                            status=('blocked_by_observation_goal' if self.request is not None else
                                    'waiting_for_observation' if status is None else status['status']),
                            reason=('another_goal_active' if self.request is not None else
                                    'observation_unavailable' if status is None else status.get('reason')),
                            reference_stamp_s=None if self.reference.current is None or self.request is not None
                                              else self.reference.current['as_of'],executable=False)
                    self.commands.agent_state.publish_control(packet,self.rows[-1],map_binding=binding,
                        cancelled_request=self.cancelled_request,idle_feedback=idle_feedback,manual_preview=preview_status)
                    if self.display is not None:
                        self.display.navigation(self.controller.controller.result,self.reference.current,planning_request,self.scene,
                            self.rows[-1],packet,self.reference.reason,now=now,clock_domain=self.clock_domain,
                            clock_rate=self.clock['clock_rate'],clock_mono=mono,grid=None if self.scene is None else self.device)
                    for event in self.reference.drain_events():self.record('reference',event)
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def _update_follow(self, now):
        from navigation.planning.continuity import goal_binding
        pending=self.follow_pending
        handle=pending['handle'] if pending is not None else self.follow.handle
        if handle is None:return
        try:
            with self.lock:
                for value in self.follow_visual_pending:
                    try:
                        self.follow.visual(value,self.scene,now=now)
                        self.follow_visual_review=dict(status='accepted',state=value['state'],rgb_stamp_s=value['rgb_stamp_s'])
                    except ValueError as error:
                        self.follow_visual_review=dict(status='rejected',state=value['state'],reason=str(error),rgb_stamp_s=value['rgb_stamp_s'])
                    self.record('reference',dict(event='follow_visual_review',**self.follow_visual_review))
                self.follow_visual_pending.clear()
                result=(self.follow.start(handle,pending['reference'],self.scene,now=now) if pending is not None
                        else self.follow.update(self.scene,now=now))
            self.follow_pending=None
            if result is None:return
            if result['operation']!='plan':raise ValueError(result.get('reason','follow_stopped'))
            request=result['request']
            if not self.commands.follow_goal(handle,request):raise ValueError('follow_owner_arming_or_agent_heartbeat_lost')
            if self.request is None or goal_binding(request)!=goal_binding(self.request):
                self.reference.invalidate('follow_goal_changed');self.controller.controller.clear()
            self.request=request
            self.commands.agent_state.publish_follow(dict(status='running',handle=handle,
                target_identity=result['target_identity'],reason=None,visual_review=self.follow_visual_review))
        except (ValueError,KeyError,TypeError) as error:
            reason=str(error)
            recoverable=reason in ('follow_track_lost','follow_track_continuity_lost','follow_visual_occlusion_expired',
                                   'follow_track_continuity_unavailable')
            stop=self.commands.follow_failed(handle,recoverable=recoverable)
            target=None if stop is None else stop['target']
            last_evidence=copy.deepcopy(self.follow.last_evidence)
            if self.request is not None and self.request.get('request_id')==handle['call_id']:
                self.reference.cancel(self.request);self.request=None;self.controller.controller.clear()
            if target is not None:self.cancelled_request=target
            with self.lock:self.follow.clear()
            self.follow_pending=None
            self.commands.agent_state.publish_follow(dict(status='failed' if target is not None else 'preempted',
                handle=handle,reason=reason,cancelled_request=target,last_evidence=last_evidence,
                recovery_available=bool(stop and stop['retained'])))

    def close(self):
        self.done.set();self.worker.join(timeout=10)
        if self.worker.is_alive():raise RuntimeError('Control loop did not stop')
        if self.error:raise RuntimeError('Control loop failed: '+self.error)
        return self.count
