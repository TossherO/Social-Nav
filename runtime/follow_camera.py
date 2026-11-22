"""Optional T2 camera pairing at the sensor edge, independent of scan delivery."""
from collections import deque
import socket
import threading
import time

from interfaces.shared import BufferFull, SharedArrayPool
from perception.sensors.image import rgb_array
from .transport import Connection, frame_ref, frame_wire


PAIR_AGE_NS = 50_000_000
ARRIVAL_WAIT_NS = 30_000_000


class FollowCameraHistory:
    def __init__(self, lidar_topic, *, live):
        self.lidar_topic = lidar_topic
        self.live = live
        self.images = deque(maxlen=96)
        self.intrinsics = deque(maxlen=96)
        self.scans = deque(maxlen=96)
        self.changed = threading.Condition()
        self.watermark = 0

    def observe(self, topic, message, received_ns):
        with self.changed:
            self.watermark = max(self.watermark, received_ns)
            stamp = message.header.stamp.to_nsec()
            if topic == self.lidar_topic:
                self.scans.append((stamp, received_ns))
            elif topic == '/camera/color/image_raw':
                self.images.append((stamp, received_ns, message))
            elif topic == '/camera/color/camera_info':
                self.intrinsics.append((stamp, received_ns, message))
            self.changed.notify_all()

    def set_scan_end(self, header_ns, end_ns):
        """Lidars stamped at scan start use the same receipt deadline at scan end."""
        with self.changed:
            for i in reversed(range(len(self.scans))):
                stamp, received = self.scans[i]
                if stamp == header_ns:
                    self.scans[i] = (end_ns, received)
                    return

    def pair(self, stamp):
        with self.changed:
            receipt = next((t for ns, t in reversed(self.scans) if ns == stamp), None)
            if receipt is None:
                return None
            deadline = receipt + ARRIVAL_WAIT_NS
            wall_end = deadline if self.live else time.monotonic_ns() + ARRIVAL_WAIT_NS
            while True:
                # Both measurements must have arrived within this scan's budget.
                # The newest usable intrinsics may precede the image measurement.
                images = [row for row in self.images
                          if row[1] <= deadline and 0 <= stamp - row[0] <= PAIR_AGE_NS]
                for rgb_ns, _, image in sorted(images, key=lambda row: row[0], reverse=True):
                    camera = max((row for row in self.intrinsics
                                  if row[0] <= rgb_ns and row[1] <= deadline
                                  and row[2].header.frame_id.lstrip('/') == image.header.frame_id.lstrip('/')
                                  and (row[2].width, row[2].height) == (image.width, image.height)),
                                 key=lambda row: row[0], default=None)
                    if camera is not None:
                        return image, camera[2]
                remaining = wall_end - time.monotonic_ns()
                if remaining <= 0 or (not self.live and self.watermark >= deadline):
                    return None
                self.changed.wait(remaining / 1e9)


class CameraPairServer:
    """Only the T2 consumer can wait. The sensor callback just records references."""
    def __init__(self, sock, history, slot_bytes):
        self.wire = Connection(sock)
        self.history = history
        self.pool = SharedArrayPool(1, slot_bytes)
        self.thread = threading.Thread(target=self._serve, name='t2-camera-pairs', daemon=True)
        self.thread.start()

    def _serve(self):
        ref = None
        try:
            self.wire.send_fd(self.pool.fd)
            self.wire.send(dict(owner=self.pool.owner, slots=1, slot_bytes=self.pool.slot_bytes))
            while True:
                stamp = self.wire.recv()
                if ref is not None:
                    self.pool.release(ref, 't2')
                    ref = None
                pair = self.history.pair(stamp)
                if pair is None:
                    self.wire.send(None)
                    continue
                image, info = pair
                try:
                    ref = self.pool.publish(dict(rgb=rgb_array(image)), readers=['t2'])
                except (ValueError, BufferFull):
                    self.wire.send(None)
                    continue
                self.wire.send(dict(
                    scan_stamp_ns=stamp, buffer=frame_wire(ref),
                    rgb_stamp_ns=image.header.stamp.to_nsec(), rgb_frame=image.header.frame_id,
                    camera_info=dict(stamp_ns=info.header.stamp.to_nsec(), frame_id=info.header.frame_id,
                                     width=info.width, height=info.height, K=list(info.K), D=list(info.D),
                                     distortion_model=info.distortion_model)))
        except (EOFError, OSError):
            pass
        finally:
            if ref is not None:
                self.pool.release(ref, 't2')
            self.pool.close()
            self.wire.close()

    def close(self):
        try:
            self.wire.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join()


class CameraPairClient:
    def __init__(self, sock):
        sock.settimeout(1.)
        self.wire = Connection(sock)
        self.reader = None
        self.closed = False
        self.error = None

    def lookup(self, stamp):
        if self.closed:
            return None
        try:
            if self.reader is None:
                self.reader = self.wire.recv_array_reader()
            self.wire.send(stamp)
            reply = self.wire.recv()
            if reply is None:
                return None
            if reply.pop('scan_stamp_ns') != stamp or not 0 <= stamp - reply['rgb_stamp_ns'] <= PAIR_AGE_NS:
                raise ValueError('T2 camera pair does not match requested scan')
            ref = frame_ref(reply.pop('buffer'))
            # The next request releases the producer's slot.
            return self.reader.arrays(ref)['rgb'].copy(), reply
        except (EOFError, OSError, ValueError) as error:
            self.error = type(error).__name__
            self.close()
            return None

    def close(self):
        self.closed = True
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.wire.close()
