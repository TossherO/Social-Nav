"""Local framed JSON control messages and Unix descriptor handoff."""
import array
import dataclasses
import json
import socket
import struct
from interfaces.runtime import ArrayRef,FrameRef


class Connection:
    def __init__(self,sock):self.socket=sock
    def send(self,value):
        payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        if len(payload)>2*1024*1024:raise ValueError('Runtime control message exceeds 2 MiB')
        self.socket.sendall(struct.pack('!I',len(payload))+payload)
    def _read(self,n):
        chunks=[]
        while n:
            part=self.socket.recv(n)
            if not part:raise EOFError('Runtime peer closed')
            chunks.append(part);n-=len(part)
        return b''.join(chunks)
    def recv(self):
        n=struct.unpack('!I',self._read(4))[0]
        if not 0<n<=2*1024*1024:raise ValueError('Invalid runtime message length')
        return json.loads(self._read(n))
    def send_fd(self,fd):
        if self.socket.sendmsg([b'F'],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[fd]))])!=1:
            raise RuntimeError('Descriptor handoff failed')
    def recv_fd(self):
        marker,control,flags,_=self.socket.recvmsg(1,socket.CMSG_SPACE(array.array('i').itemsize))
        fds=array.array('i')
        for level,kind,data in control:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:fds.frombytes(data[:len(data)-len(data)%fds.itemsize])
        if marker!=b'F' or flags&socket.MSG_CTRUNC or len(fds)!=1:raise ValueError('Invalid descriptor handoff')
        return fds[0]
    def close(self):self.socket.close()


def frame_wire(frame):return dataclasses.asdict(frame)


def frame_ref(value):
    arrays=tuple((name,ArrayRef(**dict(a,shape=tuple(a['shape'])))) for name,a in value['arrays'])
    return FrameRef(**dict(value,arrays=arrays))
