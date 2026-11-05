"""Concrete local-runtime records; no ROS, CUDA or model imports.

Clock observations are integer nanoseconds. Buffer epochs are ownership
lifecycles, not file-format or project release numbers.
"""
from dataclasses import dataclass
from typing import Tuple


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

