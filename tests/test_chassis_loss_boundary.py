"""Synthetic PTY check: host process death cannot send a later stop frame."""
import json
import os
from pathlib import Path
import pty
import select
import struct
import tempfile
import threading
import time
import tty
import unittest

from navigation.control.chassis import ChassisProcess
from navigation.planning.motion import load_motion_config
from tests.test_ground_probe import frame


class SerialOwnerLossTests(unittest.TestCase):
    def test_owner_killed_after_motion_cannot_stop_a_controller_without_watchdog(self):
        # This emulator deliberately holds the last command forever: it models
        # one possible controller, not the behavior of the real wheelchair.
        master, slave = pty.openpty()
        tty.setraw(slave)
        port = os.ttyname(slave)
        os.close(slave)
        os.set_blocking(master, False)
        shutdown = threading.Event()
        seen_motion = threading.Event()
        commands = []

        def controller():
            pending = bytearray()
            mode = 3
            while not shutdown.wait(.01):
                if select.select([master], [], [], 0)[0]:
                    try:
                        pending.extend(os.read(master, 4096))
                    except OSError:
                        pass
                while len(pending) >= 12:
                    start = pending.find(b'\xaf\x01')
                    if start < 0:
                        pending.clear()
                        break
                    del pending[:start]
                    if len(pending) < 12:
                        break
                    packet = bytes(pending[:12])
                    del pending[:12]
                    _, _, v, w, mode, _, _, _, _ = struct.unpack('>2B2hBhBBB', packet)
                    commands.append((v, w, mode))
                    if v != 0 or w != 0:
                        seen_motion.set()
                try:
                    os.write(master, frame(mode=mode))
                except OSError:
                    pass

        emulator = threading.Thread(target=controller)
        emulator.start()
        driver = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                motion = load_motion_config()
                motion['controller'] = dict(velocity_limit_field=1500,
                    feedback_max_linear_mps=1.5, feedback_max_angular_rps=1.5)
                driver = ChassisProcess(port, Path(folder) / 'chassis', motion)
                deadline = time.monotonic() + 5.
                while not seen_motion.is_set() and time.monotonic() < deadline:
                    now = time.monotonic()
                    packet = dict(mono=now, arm=True, arm_revision=1, allowed=True,
                                  units=[20, 0], expires_mono=now+.1)
                    driver.channel.send(json.dumps(packet).encode())
                    time.sleep(.025)
                self.assertTrue(seen_motion.is_set(), 'Synthetic controller did not reach motion')
                driver.process.kill()
                driver.process.join(timeout=2.)
                self.assertFalse(driver.process.is_alive())
                time.sleep(.35)
                self.assertEqual(commands[-1], (20, 0, 1),
                                 'No code remains to send a stop after owner death')
                driver.close()
                driver = None
        finally:
            if driver is not None:
                driver.close()
            shutdown.set()
            emulator.join(timeout=1.)
            os.close(master)


if __name__ == '__main__':
    unittest.main()
