"""Periodic elevator monitoring with synthetic images and decisions."""
import tempfile
import unittest
from pathlib import Path

from decision.agent.core.elevator_monitor import ElevatorMonitor
from decision.agent.core.elevator_transition import ElevatorTransition
from decision.agent.core.store import Store


class Camera:
    def __init__(self):self.sequence=0
    def snapshot(self):
        return dict(status='available',record=dict(source='ros_camera_independent_of_localization',
            clock_domain='ros_time',frame_id='camera',stamp_s=10.+self.sequence,sequence=self.sequence),
            image=b'\xff\xd8synthetic\xff\xd9')


class ElevatorMonitorTests(unittest.TestCase):
    def test_agent_can_choose_not_to_keep_a_frame(self):
        camera=Camera();histories=[]
        transition=ElevatorTransition(task_id='t',goal_id='e',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        def decide(_stage,history,_frame):
            histories.append(len(history))
            return dict(judgment='unknown',evidence='画面不足',retain_observation=False)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                monitor=ElevatorMonitor(camera,transition,decide,store,interval_s=1.)
                monitor.tick(monotonic_s=0.);monitor.model.thread.join(2.);monitor.tick(monotonic_s=.5)
                camera.sequence=1
                monitor.tick(monotonic_s=1.);monitor.model.thread.join(2.);monitor.tick(monotonic_s=1.5)
                self.assertEqual(histories,[0,0])
                self.assertEqual(monitor.history,[])
                self.assertEqual(transition.observations,[])
                monitor.close()
            finally:store.close()

    def test_periodic_decisions_receive_history_and_discard_it_after_exit(self):
        camera=Camera();calls=[]
        transition=ElevatorTransition(task_id='t',goal_id='e',from_floor='floor2',to_floor='floor1',destination_map_id='m')
        def decide(stage,history,frame):
            calls.append((stage['status'],len(history),frame['record']['sequence']))
            return dict(judgment='inside' if len(calls)==1 else 'outside',evidence='可见轿厢或走廊',retain_observation=True)
        with tempfile.TemporaryDirectory() as root:
            store=Store(Path(root)/'session.sqlite')
            try:
                monitor=ElevatorMonitor(camera,transition,decide,store,interval_s=5.)
                monitor.tick(monotonic_s=0.);monitor.model.thread.join(2.)
                monitor.tick(monotonic_s=1.)
                self.assertEqual(transition.status,'awaiting_exit')
                camera.sequence=1
                monitor.tick(monotonic_s=4.9)
                self.assertEqual(len(calls),1)
                monitor.tick(monotonic_s=5.);monitor.model.thread.join(2.)
                monitor.tick(monotonic_s=6.)
                self.assertEqual(transition.status,'awaiting_destination_localization')
                self.assertEqual(calls,[('awaiting_entry',0,0),('awaiting_exit',1,1)])
                self.assertEqual(monitor.history,[])
                self.assertEqual(transition.observations,[])
                self.assertEqual(transition.last_observation_stamp_s,11.)
                self.assertEqual(store.db.execute("select count(*) from events where kind='elevator_observation'").fetchone()[0],0)
                monitor.close()
            finally:store.close()
