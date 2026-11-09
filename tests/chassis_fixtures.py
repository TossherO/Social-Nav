"""Synthetic chassis feedback shared by serial and navigation tests."""
import struct


def frame(mode=1,velocity=0,angular=0):
    payload=struct.pack('>2h2i2HBHB2h2hB',int(velocity*1000),int(angular*1000),0,0,0,80,0,0,mode,0,0,0,0,0)
    return b'\xbf\x01'+payload+bytes([sum(payload)&255])
