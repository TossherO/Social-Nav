import socket
import tempfile
from pathlib import Path
import unittest
import threading
from collections import deque
from types import SimpleNamespace
from runtime.command import CommandInbox
from runtime.transport import Connection


class RuntimeCommandTests(unittest.TestCase):
    def test_external_intent_cannot_be_overwritten_by_recorded_route(self):
        sent=[];inbox=object.__new__(CommandInbox)
        inbox.lock=threading.Lock();inbox.pending=deque();inbox.route_preempted=False
        inbox.session=None
        inbox.safety=SimpleNamespace(command=lambda *v:sent.append(v))
        cancel=dict(task_id='route',goal_revision=1)
        self.assertTrue(inbox.route_command('cancel',cancel))
        override=dict(manual_override=True,emergency_stop=False)
        inbox.offer('override',override)
        self.assertFalse(inbox.route_command('cancel',dict(cancel,goal_revision=2)))
        self.assertEqual(sent,[('cancel',cancel),('override',override)])
        self.assertEqual(inbox.drain(),[('override',override)])

    def test_long_output_path_retains_local_intent_without_execution_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/('long-output-'*12);folder.mkdir()
            sent=[];inbox=CommandInbox(folder/'commands.sock',SimpleNamespace(command=lambda *v:sent.append(v)))
            peer=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);peer.settimeout(2.)
            try:
                peer.connect(str((folder/'commands.sock').resolve()));wire=Connection(peer)
                value=dict(manual_override=True,emergency_stop=False)
                wire.send(dict(kind='override',value=value));reply=wire.recv()
                self.assertEqual(reply,dict(accepted=True,executable=False,execution_feedback=False))
                self.assertEqual(sent,[('override',value)]);self.assertEqual(inbox.drain(),sent)
            finally:peer.close();inbox.close()
            self.assertFalse((folder/'commands.sock').exists())
