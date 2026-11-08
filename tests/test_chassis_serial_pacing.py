"""Exercise serial pacing against fragmented feedback on a pseudoterminal."""
import json
import os
from pathlib import Path
import pty
import select
import tempfile
import threading
import time
import tty
import unittest
from unittest.mock import patch

from navigation.control.chassis import ChassisProcess, COMMAND_PERIOD_S, _command_due, _read_available
from navigation.planning.motion import load_motion_config
from tests.test_ground_probe import frame


class SerialPacingTests(unittest.TestCase):
    def test_read_would_block_is_not_feedback_but_disconnect_remains_fatal(self):
        with patch('navigation.control.chassis.os.read', side_effect=BlockingIOError):
            self.assertIsNone(_read_available(5))
        with patch('navigation.control.chassis.os.read', return_value=b''):
            with self.assertRaisesRegex(RuntimeError, 'Serial input disconnected'):
                _read_available(5)
        with patch('navigation.control.chassis.os.read', return_value=b'abc'):
            self.assertEqual(_read_available(5),b'abc')

    def test_stop_and_disarm_are_due_immediately(self):
        self.assertTrue(_command_due(1., 1.05, (20, 0, 1), (0, 0, 1)))
        self.assertTrue(_command_due(1., 1.05, (0, 0, 1), (0, 0, 3)))
        self.assertFalse(_command_due(1., 1.05, (0, 0, 3), (0, 0, 3)))
        self.assertFalse(_command_due(1., 1.05, (20, 0, 1), (30, 0, 1)))
        self.assertTrue(_command_due(1.05, 1.05, (20, 0, 1), (30, 0, 1)))

    def test_fragmented_feedback_does_not_flood_serial_writes(self):
        master, slave = pty.openpty()
        tty.setraw(slave)
        port = os.ttyname(slave)
        os.close(slave)
        os.set_blocking(master, False)
        stop = threading.Event()

        def controller():
            packet = frame(mode=3)
            index = 0
            while not stop.wait(.001):
                if select.select([master], [], [], 0)[0]:
                    try:
                        os.read(master, 4096)
                    except OSError:
                        pass
                try:
                    os.write(master, packet[index:index+1])
                    index = (index+1) % len(packet)
                except OSError:
                    pass

        worker = threading.Thread(target=controller)
        worker.start()
        driver = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'chassis'
                driver = ChassisProcess(port, output, load_motion_config())
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    packet = dict(mono=now, arm=False, arm_revision=0,
                                  allowed=False, units=[0, 0], expires_mono=now+.1)
                    driver.channel.send(json.dumps(packet).encode())
                    time.sleep(.05)
                cutoff = time.monotonic()
                driver.close()
                driver = None
                rows = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
                writes = [r for r in rows if r['kind']=='wire_target']
                states = [r for r in rows if r['kind']=='state']
                # Shutdown writes use a separate fixed cleanup cadence.
                times = [r['mono'] for r in writes if r['mono'] < cutoff]
                self.assertTrue(any(r['kind']=='feedback' for r in rows))
                self.assertGreaterEqual(len(times), 15)
                self.assertLess(len(times), 65)
                self.assertTrue(all(b-a >= COMMAND_PERIOD_S-.005 for a,b in zip(times,times[1:])))
                self.assertTrue(all(not r['units'][0] and not r['units'][1] for r in writes))
                self.assertFalse(any('BlockingIOError' in str(r.get('reason')) for r in states))
        finally:
            if driver is not None:
                driver.close()
            stop.set()
            worker.join(timeout=1.)
            os.close(master)


if __name__=='__main__':
    unittest.main()
