"""Identity-bound evidence for a manually operated elevator transition.

This object does not start processes, identify an elevator from a map, move a
wheelchair, or make an image interpretation. Callers supply the actual model
judgement and the destination map's measured association.
"""
import copy
import hashlib
import math


class ElevatorTransition:
    def __init__(self, *, task_id, goal_id, from_floor, to_floor, destination_map_id):
        if not all(isinstance(v,str) and v for v in
                   (task_id,goal_id,from_floor,to_floor,destination_map_id)) or from_floor==to_floor:
            raise ValueError('A transition needs a task, goal, two floors and destination map identity')
        self.task_id,self.goal_id=task_id,goal_id
        self.from_floor,self.to_floor=from_floor,to_floor
        self.destination_map_id=destination_map_id
        self.status='awaiting_entry'
        self.observations=[]
        self.last_observation_stamp_s=None
        self.last_observation_sequence=None
        self.camera_frame_id=None
        self.observation_count=0
        self.association=None

    def observe(self, *, camera_record, image, judgment, evidence):
        """Record the Agent's judgment against an actual source camera frame."""
        if self.status not in ('awaiting_entry','awaiting_exit'):
            raise ValueError('No elevator image judgment is currently expected')
        stamp=camera_record.get('stamp_s');sequence=camera_record.get('sequence')
        if (camera_record.get('source')!='ros_camera_independent_of_localization' or
                camera_record.get('clock_domain')!='ros_time' or not camera_record.get('frame_id') or
                type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp<=0 or
                type(sequence) is not int or sequence<0 or judgment not in ('inside','outside','unknown') or
                not isinstance(evidence,str) or not evidence.strip() or not isinstance(image,bytes) or
                not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9')):
            raise ValueError('A source-bound camera observation and reason are required')
        if self.last_observation_sequence is not None:
            if sequence<=self.last_observation_sequence or stamp<self.last_observation_stamp_s:
                raise ValueError('Camera observation must advance within the same stream')
            if camera_record['frame_id']!=self.camera_frame_id:
                raise ValueError('Camera frame identity changed during transition')
        observation=dict(camera=copy.deepcopy(camera_record),judgment=judgment,evidence=evidence.strip(),
                         image_sha256=hashlib.sha256(image).hexdigest())
        self.observations.append(observation)
        self.last_observation_stamp_s=stamp
        self.last_observation_sequence=sequence
        self.camera_frame_id=camera_record['frame_id']
        self.observation_count+=1
        if self.status=='awaiting_entry' and judgment=='inside':self.status='awaiting_exit'
        elif self.status=='awaiting_exit' and judgment=='outside':self.status='awaiting_destination_localization'
        return self.snapshot()

    def accept_localization(self, *, association, now_s, max_age_s, clock_domain):
        """Accept only a fresh destination-map association after the exit observation."""
        if self.status!='awaiting_destination_localization':
            raise ValueError('Destination localization is not expected yet')
        stamp=association.get('measurement_stamp_s')
        if (association.get('valid') is not True or association.get('map_id')!=self.destination_map_id or
                not association.get('map_frame') or not association.get('local_session_id') or
                type(association.get('local_generation')) is not int or not association.get('local_frame') or
                not association.get('association_session') or type(association.get('association_generation')) is not int or
                type(stamp) not in (int,float) or not math.isfinite(stamp) or
                type(now_s) not in (int,float) or not math.isfinite(now_s) or
                type(max_age_s) not in (int,float) or not math.isfinite(max_age_s) or max_age_s<0 or
                clock_domain not in ('ros_time','runtime_live') or not 0<=now_s-stamp<=max_age_s or
                self.last_observation_stamp_s is None or stamp<self.last_observation_stamp_s):
            raise ValueError('A fresh destination-map localization is required')
        self.association=copy.deepcopy(association)
        self.status='completed'
        return self.snapshot()

    def clear_observations(self):
        """Discard per-frame observations after the Agent has detected exit."""
        if self.status not in ('awaiting_destination_localization','completed'):
            raise ValueError('Elevator observations are still needed for the ride')
        self.observations.clear()

    def snapshot(self):
        return dict(task_id=self.task_id,goal_id=self.goal_id,from_floor=self.from_floor,to_floor=self.to_floor,
                    destination_map_id=self.destination_map_id,status=self.status,
                    ride_completed=self.status in ('awaiting_destination_localization','completed'),
                    observations=copy.deepcopy(self.observations),association=copy.deepcopy(self.association),
                    last_observation_stamp_s=self.last_observation_stamp_s,observation_count=self.observation_count,
                    source='agent_visual_judgment_and_destination_localization',physical_elevator_control=False)
