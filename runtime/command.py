"""Local navigation-intent IPC. Commands never bypass the independent shadow gate."""
import argparse
from collections import deque
import copy
import json
import os
from pathlib import Path
import socket
import threading
import tempfile
from .transport import Connection


def validate(kind,value):
    from navigation.planning.online import goal_request
    if kind=='goal':return goal_request(value)
    if kind=='cancel':
        if (not isinstance(value,dict) or set(value)!={'task_id','goal_revision'} or not isinstance(value['task_id'],str) or
                type(value['goal_revision']) is not int or value['goal_revision']<0):raise ValueError('Invalid cancellation')
    elif kind=='override':
        if (not isinstance(value,dict) or set(value)!={'manual_override','emergency_stop'} or
                any(type(v) is not bool for v in value.values())):raise ValueError('Override values must be Boolean')
    else:raise ValueError('Only goal, cancel and override are supported')
    return copy.deepcopy(value)


class CommandInbox:
    def __init__(self,path,safety):
        from .agent_state import AgentState
        self.agent_state=AgentState()
        self.path=Path(path);self.safety=safety;self.pending=deque();self.lock=threading.Lock();self.done=False
        self.route_preempted=False
        if self.path.exists():raise FileExistsError(self.path)
        # Linux sockaddr_un limits the address length even before symlink
        # resolution. The CLI resolves this visible link to its private endpoint.
        self.temporary=tempfile.TemporaryDirectory(prefix='social-nav-command-') if len(os.fsencode(self.path))>=104 else None
        self.address=Path(self.temporary.name)/'command.sock' if self.temporary is not None else self.path
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:self.socket.bind(str(self.address));self.address.chmod(0o600)
        except BaseException:
            self.socket.close()
            if self.temporary is not None:self.temporary.cleanup()
            raise
        if self.temporary is not None:self.path.symlink_to(self.address)
        self.socket.listen(2);self.socket.settimeout(.1)
        self.worker=threading.Thread(target=self._work,name='runtime-commands',daemon=True);self.worker.start()

    def offer(self,kind,value):
        value=validate(kind,value)
        with self.lock:
            if len(self.pending)>=64:raise RuntimeError('Command inbox is full')
            self.route_preempted=True
            self.safety.command(kind,value);self.pending.append((kind,value))

    def route_command(self,kind,value):
        """A manual intent permanently preempts the optional recorded route."""
        value=validate(kind,value)
        with self.lock:
            if self.route_preempted:return False
            self.safety.command(kind,value)
            return True

    def drain(self):
        with self.lock:items=list(self.pending);self.pending.clear();return items

    def _work(self):
        while not self.done:
            try:peer,_=self.socket.accept()
            except socket.timeout:continue
            except OSError:return
            connection=Connection(peer);peer.settimeout(.5)
            try:
                request=connection.recv()
                if set(request)!={'kind','value'}:raise ValueError('Use kind and value')
                if request['kind']=='agent_state':
                    value=request['value']
                    if not isinstance(value,dict) or set(value)!={'with_image'} or type(value['with_image']) is not bool:
                        raise ValueError('State query requires a Boolean with_image')
                    connection.send(self.agent_state.snapshot(with_image=value['with_image']));continue
                self.offer(request['kind'],request['value'])
                connection.send(dict(accepted=True,executable=False,execution_feedback=False))
            except Exception as error:
                try:connection.send(dict(accepted=False,error=str(error),executable=False))
                except OSError:pass
            finally:connection.close()

    def close(self):
        self.done=True;self.socket.close();self.worker.join(timeout=2);self.path.unlink(missing_ok=True)
        if self.temporary is not None:self.temporary.cleanup()


def send_request(path,kind,value,*,timeout=2.):
    """Submit one intent; the reply acknowledges receipt, never execution."""
    value=validate(kind,value)
    if not 0<timeout<=10:raise ValueError('Command timeout must be in (0,10] seconds')
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(timeout)
    wire=Connection(sock)
    try:
        sock.connect(str(Path(path).resolve()))
        wire.send(dict(kind=kind,value=value))
        return wire.recv()
    finally:wire.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--request',required=True);args=p.parse_args()
    request=json.loads(Path(args.request).read_text())
    print(json.dumps(send_request(args.socket,request['kind'],request['value']),ensure_ascii=False))


if __name__=='__main__':main()
