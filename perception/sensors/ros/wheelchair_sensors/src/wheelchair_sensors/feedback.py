"""32-byte feedback schema extracted from the existing pub_odom.py.

The original receiver did not validate a checksum. The two final byte meanings
are unknown; they are preserved, not treated as a verified checksum. No transmit
or motion-control code is included here.
"""
import struct
from dataclasses import dataclass

HEADER = b'\xbf\x01'
PAYLOAD = struct.Struct('>2h2i2H1B1H1B2h2h1B1B')
FRAME_SIZE = len(HEADER) + PAYLOAD.size

@dataclass(frozen=True)
class Feedback:
    linear_velocity: float
    angular_velocity: float
    x: float
    y: float
    theta: float
    battery: int
    state: int
    joystick_linear: float
    joystick_angular: float
    joystick_x: int
    joystick_y: int
    reserved_byte: int
    reserved_word: int
    trailer: bytes

def decode_frame(frame):
    if not isinstance(frame, bytes):
        raise TypeError('Feedback frames must be bytes')
    if len(frame) != FRAME_SIZE or not frame.startswith(HEADER):
        raise ValueError('Expected a 32-byte BF01 feedback frame')
    v, w, x, y, theta, battery, b, h, state, jv, jw, jx, jy, t1, t2 = PAYLOAD.unpack(frame[2:])
    return Feedback(v / 1000.0, w / 1000.0, x / 1000.0, y / 1000.0,
                    theta / 1000.0, battery, state, jv / 1000.0, jw / 1000.0,
                    jx, jy, b, h, bytes((t1, t2)))

class FrameDecoder:
    """Handle split/coalesced frames and leading noise without splitting payloads."""
    def __init__(self):
        self.buffer = bytearray()
        self.discarded_bytes = 0

    def feed(self, data):
        if not isinstance(data, bytes):
            raise TypeError('Serial input must be bytes')
        self.buffer.extend(data)
        output = []
        while True:
            start = self.buffer.find(HEADER)
            if start < 0:
                keep = 1 if self.buffer.endswith(HEADER[:1]) else 0
                discard = len(self.buffer) - keep
                self.discarded_bytes += discard
                del self.buffer[:discard]
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < FRAME_SIZE:
                break
            frame = bytes(self.buffer[:FRAME_SIZE])
            del self.buffer[:FRAME_SIZE]
            output.append(decode_frame(frame))
        return output
