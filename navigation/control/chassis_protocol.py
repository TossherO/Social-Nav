"""Offline candidate codec from the inspected legacy navigation serial driver.

No device is opened. The spreadsheet and drivers disagree on wire layouts;
feedback checksum, mode semantics and firmware watchdog remain unconfirmed.
Decoded records must not be used as valid execution feedback.
"""
import argparse
import json
import math
from pathlib import Path
import struct


SOURCE=dict(container='nav_wheel',image='wheel_nav:v1',
    file='/home/wheel_nav/src/ros_navigation/move_base/scripts/pub_odom.py',
    sha256='f0c4b5899e4814274bb8d1948103433a670bc6813fbb10e57b4c31fd18bc7a02')


def _integer(value,low,high,name):
    if type(value) is not int or not low<=value<=high:raise ValueError('Invalid '+name)
    return value


def encode_command(linear_mps,angular_rps,*,mode,velocity_limit_mm_s=1500,error=0,led=0):
    """Reproduce the 12-byte driver candidate, including its word-sum checksum."""
    velocities=[]
    for value in (linear_mps,angular_rps):
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Velocity must be finite')
        if not -32.768<=value<=32.767:raise ValueError('Velocity overflows the signed wire field')
        velocities.append(int(value*1000.))
    return encode_command_units(*velocities,mode=mode,velocity_limit_mm_s=velocity_limit_mm_s,error=error,led=led)


def encode_command_units(linear_mm_s,angular_mrad_s,*,mode,velocity_limit_mm_s=1500,error=0,led=0):
    """Encode the exact integer target already checked by independent safety."""
    velocities=[_integer(linear_mm_s,-32768,32767,'linear velocity'),
                _integer(angular_mrad_s,-32768,32767,'angular velocity')]
    mode=_integer(mode,0,255,'raw mode byte')
    limit=_integer(velocity_limit_mm_s,0,32767,'velocity limit')
    error=_integer(error,0,255,'error byte');led=_integer(led,0,255,'LED byte')
    checksum=(0xAF+1+sum(velocities)+mode+limit+error)&255
    return struct.pack('>2B2hBhBBB',0xAF,1,*velocities,mode,limit,error,led,checksum)


def decode_feedback(frame):
    """Decode one exact 32-byte frame; never assert undocumented integrity."""
    if len(frame)!=32 or frame[:2]!=b'\xbf\x01':raise ValueError('Expected one BF01 32-byte candidate frame')
    v,w,x,y,yaw,battery,charging,pitch,mode,jv,jw,jx,jy,reserved,check=struct.unpack('>2h2i2HBHB2h2hBB',frame[2:])
    candidate_matches=(sum(frame[2:-1])&255)==check
    return dict(status='unverified_protocol',execution_feedback=False,device_measurement_stamp_s=None,
        velocity_mps_rps=[v/1000.,w/1000.],encoder_pose_m_m_rad=[x/1000.,y/1000.,yaw/1000.],
        battery_raw=battery,charging_raw=charging,pitch_raw=pitch,mode_raw=mode,
        joystick_mps_rps=[jv/1000.,jw/1000.],joystick_axes_raw=[jx,jy],reserved_raw=reserved,
        checksum_raw=check,checksum_verified=False,candidate_checksum_rule='payload_byte_sum_mod256',
        candidate_checksum_matches=candidate_matches,frame_hex=frame.hex())


class FeedbackFrames:
    """Bounded offline framing with explicit absence of checksum validation."""
    def __init__(self):self.pending=bytearray();self.discarded=0
    def feed(self,data):
        self.pending.extend(data);rows=[]
        while len(self.pending)>=2:
            start=self.pending.find(b'\xbf\x01')
            if start<0:
                keep=1 if self.pending[-1]==0xBF else 0
                self.discarded+=len(self.pending)-keep
                self.pending=self.pending[-1:] if keep else bytearray()
                break
            self.discarded+=start;del self.pending[:start]
            if len(self.pending)<32:break
            rows.append(decode_feedback(bytes(self.pending[:32])));del self.pending[:32]
        return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feedback-file',required=True,help='Existing raw binary capture; no device paths')
    args=p.parse_args(argv);path=Path(args.feedback_file)
    if not path.is_file():p.error('Only regular capture files are accepted; serial devices are not opened')
    frames=FeedbackFrames();count=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4096),b''):
            for row in frames.feed(chunk):print(json.dumps(row,allow_nan=False));count+=1
    print(json.dumps(dict(frames=count,discarded_bytes=frames.discarded,pending_bytes=len(frames.pending),
        checksum_verified=False,execution_feedback=False,protocol_source=SOURCE)))


if __name__=='__main__':main()
