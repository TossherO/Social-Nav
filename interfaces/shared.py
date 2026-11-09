"""Single-writer sealed shared slots, with explicit read leases and reuse epochs.

Only small FrameRef descriptors cross the control channel. A reader mapping is
OS read-only. The writer never reuses a sealed slot before every lease is acked.
"""
import hashlib
import mmap
import os
import struct
import uuid
import numpy as np
from interfaces.runtime import ArrayRef,FrameRef

HEADER=64
STATE=struct.Struct('<QQ')


class BufferFull(RuntimeError):pass


class SharedArrayPool:
    def __init__(self,slots=4,slot_bytes=4*1024*1024):
        if type(slots) is not int or slots<1 or type(slot_bytes) is not int or slot_bytes<HEADER+64:
            raise ValueError('Invalid shared-buffer capacity')
        self.owner=uuid.uuid4().hex;self.slots=slots;self.slot_bytes=slot_bytes
        self.fd=os.memfd_create('social-nav-'+self.owner,os.MFD_CLOEXEC)
        os.ftruncate(self.fd,slots*slot_bytes)
        self.mapping=mmap.mmap(self.fd,slots*slot_bytes,access=mmap.ACCESS_WRITE)
        self.epochs=[0]*slots;self.leases={};self.closed=False

    def publish(self,arrays,*,readers):
        if self.closed:raise RuntimeError('Buffer pool closed')
        readers=set(readers)
        if not readers or any(not isinstance(r,str) or not r for r in readers):raise ValueError('Explicit readers required')
        slot=next((i for i in range(self.slots) if i not in self.leases),None)
        if slot is None:raise BufferFull('All shared slots have active leases')
        prepared=[];offset=HEADER
        for name,value in arrays.items():
            if not isinstance(name,str) or not name:raise ValueError('Invalid array name')
            value=np.ascontiguousarray(value)
            if value.dtype.hasobject or not value.size:raise ValueError('Shared arrays require nonempty numeric data')
            offset=(offset+63)//64*64
            if offset+value.nbytes>self.slot_bytes:raise BufferFull('Frame exceeds shared-slot byte budget')
            prepared.append((name,value,offset));offset+=value.nbytes
        if not prepared:raise ValueError('No arrays in frame')
        epoch=self.epochs[slot]+1;base=slot*self.slot_bytes
        STATE.pack_into(self.mapping,base,epoch,0)
        digest=hashlib.sha256();refs=[]
        for name,value,offset in prepared:
            target=np.ndarray(value.shape,value.dtype,buffer=self.mapping,offset=base+offset)
            target[...] = value
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value)
            refs.append((name,ArrayRef(self.owner,slot,epoch,offset,tuple(value.shape),value.dtype.str,value.nbytes)))
            del target
        STATE.pack_into(self.mapping,base,epoch,1)
        self.epochs[slot]=epoch;self.leases[slot]=(epoch,readers)
        return FrameRef(self.owner,slot,epoch,tuple(refs),digest.hexdigest())

    def retain(self,ref,reader):
        """Add a consumer lease while an existing lease still owns the slot."""
        if ref.owner!=self.owner or self.leases.get(ref.slot,(None,None))[0]!=ref.epoch:
            raise ValueError('Cannot retain a stale buffer')
        readers=self.leases[ref.slot][1]
        if not isinstance(reader,str) or not reader or reader in readers:
            raise ValueError('Explicit new reader required')
        readers.add(reader)

    def release(self,ref,reader):
        if ref.owner!=self.owner or self.leases.get(ref.slot,(None,None))[0]!=ref.epoch:raise ValueError('Stale buffer release')
        readers=self.leases[ref.slot][1]
        if reader not in readers:raise ValueError('Unknown or duplicate buffer release')
        readers.remove(reader)
        if not readers:del self.leases[ref.slot]

    def reader(self):return SharedArrayReader(os.dup(self.fd),self.owner,self.slots,self.slot_bytes)

    def close(self):
        if not self.closed:
            if self.leases:raise RuntimeError('Cannot close pool with outstanding leases')
            self.mapping.close();os.close(self.fd);self.closed=True


class SharedArrayReader:
    def __init__(self,fd,owner,slots,slot_bytes):
        self.fd,self.owner,self.slots,self.slot_bytes=fd,owner,slots,slot_bytes
        self.mapping=mmap.mmap(fd,slots*slot_bytes,access=mmap.ACCESS_READ)

    def arrays(self,ref,*,verify=True):
        if (ref.owner!=self.owner or not 0<=ref.slot<self.slots or
                STATE.unpack_from(self.mapping,ref.slot*self.slot_bytes)!=(ref.epoch,1)):
            raise ValueError('Stale, foreign or unsealed shared frame')
        result={};digest=hashlib.sha256();base=ref.slot*self.slot_bytes
        for name,a in ref.arrays:
            dtype=np.dtype(a.dtype)
            if (a.owner!=ref.owner or (a.slot,a.epoch)!=(ref.slot,ref.epoch) or dtype.hasobject or
                    any(type(n) is not int or n<1 for n in a.shape) or a.offset<HEADER or
                    a.nbytes!=int(np.prod(a.shape))*dtype.itemsize or a.offset+a.nbytes>self.slot_bytes or name in result):
                raise ValueError('Invalid shared array descriptor')
            value=np.ndarray(a.shape,dtype,buffer=self.mapping,offset=base+a.offset)
            if verify:
                digest.update(name.encode());digest.update(dtype.str.encode());digest.update(str(a.shape).encode());digest.update(value)
            result[name]=value
        if verify and digest.hexdigest()!=ref.content_sha256:raise ValueError('Shared frame contents changed')
        if STATE.unpack_from(self.mapping,base)!=(ref.epoch,1):raise ValueError('Shared slot changed during acquisition')
        return result

    def close(self):
        self.mapping.close();os.close(self.fd)

