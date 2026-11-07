"""Bounded, best-effort chassis recording; disk I/O never runs in the serial loop."""
from collections import Counter
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time


class ChassisTelemetry:
    def __init__(self, directory, *, capacity=2048):
        self.directory = Path(directory)
        self.queue = Queue(maxsize=capacity)
        self.seen = Counter()
        self.dropped = Counter()
        self.written = 0
        self.error = None
        self.done = threading.Event()
        self.latest_status = None
        self.worker = threading.Thread(target=self._run, name='chassis-recording', daemon=True)
        self.worker.start()

    def emit(self, kind, *, mono=None, **values):
        """Caller owns values until submission, then must leave them unchanged."""
        self.seen[kind] += 1
        row = dict(kind=kind, mono=time.monotonic() if mono is None else mono, **values)
        try:
            self.queue.put_nowait(row)
        except Full:
            self.dropped[kind] += 1

    def status(self, row):
        # Coalesced separately so a full event queue cannot hide the latest stop.
        self.latest_status = row

    def summary(self, *, finalized=False):
        seen, dropped = dict(self.seen), dict(self.dropped)
        return dict(scope='host_received_chassis_diagnostics_not_physical_acceptance',
                    finalized=finalized, complete=bool(finalized and not self.error and not dropped and
                                                     self.written == sum(seen.values())),
                    seen=seen, written=self.written, dropped=dropped, error=self.error,
                    queue_capacity=self.queue.maxsize, queued=self.queue.qsize())

    def _save_json(self, name, value):
        target = self.directory/name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, allow_nan=False)+'\n')
        temporary.replace(target)

    def _write_event(self, stream, row):
        stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')

    def _run(self):
        status = None
        last_flush = 0.
        try:
            with (self.directory/'events.jsonl').open('x') as stream:
                self._save_json('telemetry_summary.json', self.summary())
                while not self.done.is_set() or not self.queue.empty():
                    try:
                        row = self.queue.get(timeout=.02)
                    except Empty:
                        row = None
                    if row is not None:
                        self._write_event(stream, row)
                        self.written += 1
                    latest = self.latest_status
                    if latest is not None and latest is not status:
                        self._save_json('status.json', latest)
                        status = latest
                    if time.monotonic()-last_flush >= .5:
                        stream.flush()
                        self._save_json('telemetry_summary.json', self.summary())
                        last_flush = time.monotonic()
                if self.latest_status is not None:
                    self._save_json('status.json', self.latest_status)
        except Exception as error:
            self.error = type(error).__name__+': '+str(error)
        finally:
            try:
                self._save_json('telemetry_summary.json', self.summary(finalized=True))
            except OSError:
                pass  # Missing/incomplete summary also means recording was not verified.

    def close(self, timeout=.5):
        self.done.set()
        self.worker.join(timeout=timeout)
        return self.summary(finalized=not self.worker.is_alive())
