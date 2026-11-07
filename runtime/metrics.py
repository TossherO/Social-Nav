"""Bounded stage samples and explicit CPU/device-copy accounting."""

from collections import Counter, deque
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time

import numpy as np


class Metrics:
    """Store milliseconds; transfer counts attribute consumers, not CUDA API calls."""

    def __init__(self, max_samples=10000, *, trace_path=None, trace_limit=20000):
        self.max_samples = max_samples
        self.samples = {}
        self.bytes = Counter()
        self.counts = Counter()
        self.events = Counter()
        self.trace_path = None if trace_path is None else Path(trace_path)
        self.trace_rows = deque(maxlen=trace_limit)
        self.trace_lock = threading.Lock()
        self.trace_count = 0

    def trace(self, event, **fields):
        """Optional bounded diagnostic metadata; no disk or device access in producers."""
        if self.trace_path is None:return
        with self.trace_lock:
            self.trace_rows.append(dict(event=event,monotonic_s=time.monotonic(),**fields))
            self.trace_count+=1

    def write_trace(self):
        """Call after producers stop; truncation is explicit and never hidden."""
        if self.trace_path is None:return
        with self.trace_lock:
            rows=list(self.trace_rows);dropped=self.trace_count-len(rows)
        with self.trace_path.open('x') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
        report=dict(written=len(rows),dropped=dropped,complete=dropped==0,
            clock='host_monotonic_seconds; measurement_stamp_s remains original source time')
        with self.trace_path.with_suffix('.summary.json').open('x') as stream:json.dump(report,stream,indent=2)

    def event(self, name, count=1):
        self.events[name] += count

    def add(self, name, seconds):
        if name not in self.samples:
            self.samples[name] = deque(maxlen=self.max_samples)
        self.samples[name].append(seconds * 1000)

    @contextmanager
    def span(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def transfer(self, direction, nbytes, reason):
        if (direction not in ('h2d', 'd2h', 'd2d', 'host_copy') or
                not reason or nbytes < 0):
            raise ValueError('Invalid transfer attribution')
        self.bytes[direction + ':' + reason] += int(nbytes)
        self.counts[direction + ':' + reason] += 1

    def summary(self):
        stages = {}
        for name, values in self.samples.items():
            a = np.asarray(values)
            stages[name] = dict(
                samples=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                maximum=float(a.max()),
            )
        return dict(
            stages_ms=stages,
            transfer_bytes=dict(self.bytes),
            transfer_calls=dict(self.counts),
            events=dict(self.events),
            timing_scope='CPU wall time including blocking waits; GPU event timings explicitly named',
        )

