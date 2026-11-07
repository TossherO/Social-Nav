"""Explicit bag pause boundaries. Measurement timestamps are never rewritten."""
import json
import socket
import time


class BagPlayback:
    """Input-worker clock; pauses occur between processed scans, not mid-SLAM."""
    def __init__(self, channel, *, rate, pause_at=None):
        self.channel = channel
        self.channel.setblocking(False)
        self.rate = rate
        self.pause_at = pause_at
        self.first = self.anchor = self.last = None
        self.paused = False
        self.epoch = 0

    def _commands(self, *, ended=False):
        while True:
            try:
                value = json.loads(self.channel.recv(4096))
            except BlockingIOError:
                return
            if type(value.get('paused')) is not bool:
                raise ValueError('Playback requires a Boolean paused value')
            if ended and not value['paused']:
                continue
            if self.paused != value['paused']:
                self.paused = value['paused']
                self.epoch += 1

    def boundary(self, notify, stop, *, ended=False):
        self._commands(ended=ended)
        if (self.last is not None and self.pause_at is not None and
                self.last - self.first >= self.pause_at - 1e-6):
            self.paused = True
            self.pause_at = None
            self.epoch += 1
        if ended:
            self.paused = True
            self.epoch += 1
        if not self.paused:
            return
        began = time.monotonic()
        while self.paused and not stop.is_set():
            notify(dict(kind='playback', paused=True, epoch=self.epoch,
                        stamp_s=self.last, offset_s=None if self.last is None else self.last-self.first,
                        source_ended=ended))
            stop.wait(.2)
            self._commands(ended=ended)
        if self.anchor is not None:
            self.anchor += time.monotonic() - began
        if not stop.is_set():
            notify(dict(kind='playback', paused=False, epoch=self.epoch,
                        stamp_s=self.last, offset_s=None if self.last is None else self.last-self.first,
                        source_ended=False))

    def start(self, stamp):
        if self.first is None:
            self.first = stamp
            self.anchor = time.monotonic()

    def close(self):
        self.channel.close()
