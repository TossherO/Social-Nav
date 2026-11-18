"""Elevator-specific judgment using the generic periodic observation scheduler."""
from .periodic_observation import PeriodicObservation


class ElevatorMonitor:
    def __init__(self, camera, transition, decide, store, *, interval_s=5., history_frames=4, history_policy='agent'):
        if type(history_frames) is not int or not 0<=history_frames<=4:
            raise ValueError('Elevator model supports at most four historical images')
        self.camera,self.transition,self.store=camera,transition,store
        self.periodic=PeriodicObservation(camera,decide,interval_s=interval_s,
                                          history_limit=history_frames,history_policy=history_policy)
        self.model=self.periodic.model
        self.error=None
        self.latest_decision=None

    @property
    def history(self):return self.periodic.history

    def tick(self, *, monotonic_s=None):
        context=dict(status=self.transition.status,task_id=self.transition.task_id,goal_id=self.transition.goal_id)
        result=self.periodic.poll(context)
        if result is not None:
            if result['status']=='discarded':
                self.store.event('elevator_model_discarded',dict(reason=result['reason']))
            elif result['status']=='failed':
                self.error=result['reason']
                self.store.event('elevator_model_failed',dict(reason=self.error,automatically_retried=False))
            else:
                value=result['value'];frame=result['frame']
                try:
                    if not isinstance(value,dict) or set(value)-{'judgment','evidence','retain_observation'} or not {'judgment','evidence'}<=set(value):
                        raise ValueError('Elevator decision requires judgment and evidence')
                    if self.periodic.history_policy=='agent' and type(value.get('retain_observation')) is not bool:
                        raise ValueError('Agent must choose whether to retain the observation')
                    self.transition.observe(camera_record=frame['record'],image=frame['image'],
                                            judgment=value['judgment'],evidence=value['evidence'])
                    observation=self.transition.observations[-1]
                    self.latest_decision=dict(judgment=value['judgment'],evidence=value['evidence'],
                                              source_stamp_s=observation['camera']['stamp_s'])
                    self.periodic.remember(observation,frame['image'],value)
                    # The generic retention policy is the sole owner of visual
                    # history; the transition needs only monotonic source metadata.
                    self.transition.observations.clear()
                    if self.transition.status=='awaiting_destination_localization':
                        self.latest_decision=None
                        self.periodic.clear()
                        self.transition.clear_observations()
                        close=getattr(self.camera,'close',None)
                        if callable(close):close()
                except (ValueError,KeyError,TypeError) as error:
                    self.error='invalid_model_judgment'
                    self.store.event('elevator_model_failed',dict(reason=self.error,error_type=type(error).__name__,
                                                                  automatically_retried=False))
        if self.error is None and self.transition.status in ('awaiting_entry','awaiting_exit'):
            context=dict(status=self.transition.status,task_id=self.transition.task_id,goal_id=self.transition.goal_id)
            self.periodic.submit_if_due(context,monotonic_s=monotonic_s)
        return self.snapshot()

    def snapshot(self):
        return dict(status=self.transition.status,model_busy=self.model.busy,error=self.error,
                    observation_count=self.transition.observation_count,history_frames=len(self.history),
                    latest_decision=self.latest_decision)

    def close(self):
        self.latest_decision=None
        self.transition.observations.clear()
        close=getattr(self.camera,'close',None)
        if callable(close):close()
        return self.periodic.close()
