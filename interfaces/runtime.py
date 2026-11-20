"""Concrete local-runtime records; no ROS, CUDA or model imports.

Clock observations are integer nanoseconds. Buffer epochs are ownership
lifecycles, not file-format or project release numbers.
"""
from dataclasses import dataclass
from math import isclose
from typing import Tuple


def source_stamp_ns(stamp_s, stamp_ns=None):
    """Keep the supplied sensor timestamp; seconds-only observations predate it."""
    if stamp_ns is None:
        return round(stamp_s * 1_000_000_000)
    if type(stamp_ns) is not int or not isclose(stamp_ns / 1_000_000_000, stamp_s, rel_tol=0., abs_tol=1e-6):
        raise ValueError('Source nanoseconds do not match the timestamp in seconds')
    return stamp_ns


def rgb_age_s(scan_s, rgb_s, scan_ns=None, rgb_ns=None):
    """Subtract source nanoseconds before conversion when both are available."""
    if rgb_ns is not None:
        source_stamp_ns(rgb_s, rgb_ns)
        if scan_ns is not None:
            return (source_stamp_ns(scan_s, scan_ns) - rgb_ns) / 1_000_000_000
    return scan_s - rgb_s


@dataclass(frozen=True)
class Observation:
    source: str
    stamp_ns: int
    frame_id: str
    clock_domain: str
    session_id: str
    generation: int

    def __post_init__(self):
        if any(not isinstance(x,str) or not x for x in (self.source,self.frame_id,self.clock_domain,self.session_id)):
            raise ValueError('Observation requires explicit source, frame, clock and session')
        if type(self.stamp_ns) is not int or self.stamp_ns<0 or type(self.generation) is not int or self.generation<0:
            raise ValueError('Invalid observation time or lifecycle')


@dataclass(frozen=True)
class ArrayRef:
    owner: str
    slot: int
    epoch: int
    offset: int
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class FrameRef:
    owner: str
    slot: int
    epoch: int
    arrays: Tuple[Tuple[str, ArrayRef], ...]
    content_sha256: str

