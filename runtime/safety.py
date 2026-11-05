"""Independent CPU shadow safety with a direct localization input channel."""
import copy
import multiprocessing as mp
from multiprocessing.connection import wait
from multiprocessing.reduction import DupFd
import os
import socket
import time
import json
import threading
import signal
from .transport import Connection
from .buffers import SharedArrayReader
from .scenes import ScenePool,read_scene


def _run(control,pose_fd,output_fd,buffer_fd,owner,slots,slot_bytes,planning,motion,geometry,clock_domain,asset_path,shadow_settings):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from navigation.safety.shadow import ShadowSafety
    from perception.environment.storage import StaticEnvironmentMap
    reader=SharedArrayReader(buffer_fd.detach(),owner,slots,slot_bytes)
    pose=Connection(socket.socket(fileno=pose_fd.detach()));output=socket.socket(fileno=output_fd.detach());output.setblocking(False)
    settings=copy.deepcopy(motion);settings['collision_backend']='cpu';settings['collision_library']=None
    asset=None if asset_path is None else StaticEnvironmentMap(asset_path)
    engine=ShadowSafety(planning,settings,geometry,clock_domain=clock_domain,asset=asset,settings=shadow_settings)
    leases={};clock={};stopping=threading.Event();cpu_start=time.process_time()
    def receive():
        pose_alive=control_alive=True
        try:
            while not stopping.is_set():
                available=wait(([control] if control_alive else [])+([pose.socket] if pose_alive else []),timeout=.05)
                if pose_alive and pose.socket in available:
                    try:message=pose.recv()
                    except EOFError:message=dict(kind='end')
                    with engine.lock:
                        if message.get('kind')=='state':
                            engine.push_state(message['state'])
                            clock.update(stamp=message['clock_s'],mono=message['clock_mono'],rate=message['clock_rate'])
                        else:
                            pose_alive=False;engine.rows.clear();engine.barrier='input_disconnected';engine.wake.set()
                if control_alive and control in available:
                    try:message=control.recv()
                    except EOFError:
                        control_alive=False
                        engine.command('override',dict(manual_override=False,emergency_stop=True))
                        continue
                    if message is None:stopping.set();engine.wake.set();return
                    try:
                        with engine.lock:
                            if message['kind']=='scene':
                                description=message['shared_scene'];scene=read_scene(reader,description)
                                leases[description['frame']]=scene.geometry
                                engine.push_grid(scene.geometry);engine.push_record(scene.record)
                                from perception.environment.scene import people_observation
                                people=people_observation(scene)
                                if people is not None:engine.push_people(people)
                            elif message['kind']=='packet':engine.push_packet(message['packet'])
                            elif message['kind']=='command':engine.command(message['command'],message['value'])
                    except (ValueError,KeyError,TypeError) as error:
                        with engine.lock:engine.barrier='input_rejected:'+str(error);engine.wake.set()
        except Exception as error:
            with engine.lock:
                engine.command('override',dict(manual_override=False,emergency_stop=True))
                engine.barrier='receiver_failed:'+str(error)
        finally:engine.wake.set()
    receiver=threading.Thread(target=receive,name='safety-receive',daemon=True);receiver.start()
    dropped=0;next_tick=time.monotonic();control.send(dict(ready=True))
    try:
        while not stopping.is_set():
            engine.wake.wait(max(0.,next_tick-time.monotonic()));engine.wake.clear()
            if stopping.is_set():break
            mono=time.monotonic()
            with engine.lock:source=dict(clock)
            if source:
                # Freeze this conversion for one check; source measurements may
                # update concurrently and are revalidated by ShadowSafety.
                now=lambda:source['stamp']+(time.monotonic()-source['mono'])*source['rate']
                result=engine.tick(now(),mono=mono,clock_now=now)
                result.update(scope='runtime_CPU_shadow',telemetry_dropped=dropped,process_cpu_s=time.process_time()-cpu_start,
                    decision_complete_mono=time.monotonic())
                try:output.send(json.dumps(result,allow_nan=False,separators=(',',':')).encode())
                except (BlockingIOError,BrokenPipeError,ConnectionResetError):dropped+=1
            next_tick=mono+engine.settings['period_s']
            # Only release after a full check, never while its local variables
            # still reference a snapshot removed by the receiver on a reset.
            with engine.lock:
                held={id(x) for x in engine.inbox.grids.values()}
                for scene in (engine.inbox.cached,engine.inbox.current_scene):
                    if scene is not None:held.add(id(scene.geometry))
                if engine.guard._people_cache is not None:held.add(id(engine.guard._people_cache[0].geometry))
                released=[ref for ref,grid in leases.items() if id(grid) not in held]
                for ref in released:del leases[ref]
            for ref in released:
                try:control.send(dict(released=ref))
                except (EOFError,BrokenPipeError):pass
    finally:
        stopping.set();receiver.join(timeout=1)
        if receiver.is_alive():raise RuntimeError('Safety receiver did not stop')
        engine=None;scene=None;grid=None;leases.clear();reader.close();pose.close();output.close();control.close()


class SafetyProcess:
    def __init__(self,pose_socket,planning,motion,geometry,*,clock_domain,asset=None,slots=16,slot_bytes=256*1024,settings=None):
        self.scenes=ScenePool(slots,slot_bytes);pool=self.scenes.pool
        self.out,child_out=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM);self.out.setblocking(False)
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=_run,args=(child,DupFd(pose_socket.fileno()),DupFd(child_out.fileno()),DupFd(pool.fd),
            pool.owner,pool.slots,pool.slot_bytes,planning,motion,geometry,clock_domain,None if asset is None else str(asset.directory),settings),daemon=True)
        self.process.start();child.close();child_out.close();self.ready=False;self.pending={}
        self.send_lock=threading.Lock()

    def _send(self,message):
        with self.send_lock:self.connection.send(message)

    def _poll_control(self):
        while self.connection.poll():
            message=self.connection.recv()
            if message.get('ready'):self.ready=True
            if 'released' in message:
                ref=message['released'];self.scenes.pool.release(ref,'safety');self.pending.pop(ref,None)
    def poll(self):
        self._poll_control();return self.read_output()

    def read_output(self):
        """Telemetry only; lease/control acknowledgements remain with the owner."""
        rows=[]
        while True:
            try:
                row=json.loads(self.out.recv(262144));row['telemetry_receive_mono']=time.monotonic()
                row['telemetry_receive_delay_ms']=(row['telemetry_receive_mono']-row['decision_complete_mono'])*1000
                rows.append(row)
            except BlockingIOError:break
        return rows

    def scene(self,scene):
        self._poll_control();description=self.scenes.publish(scene,'safety');self.pending[description['frame']]=description
        self._send(dict(kind='scene',shared_scene=description))
    def packet(self,packet):self._send(dict(kind='packet',packet=packet))
    def command(self,kind,value):self._send(dict(kind='command',command=kind,value=value))
    def close(self):
        if self.process.is_alive():
            try:self._send(None)
            except (EOFError,BrokenPipeError):pass
        self.process.join(timeout=2)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=1)
        for ref in self.pending:self.scenes.pool.release(ref,'safety')
        self.pending.clear();self.scenes.close();self.connection.close();self.out.close()


class SafetyTelemetry:
    """Record emitted decisions without waiting for a GPU perception iteration."""
    def __init__(self,safety,record):
        self.safety=safety;self.record=record;self.count=0;self.error=None;self.done=threading.Event()
        self.worker=threading.Thread(target=self._work,name='safety-telemetry',daemon=True);self.worker.start()

    def _work(self):
        try:
            while True:
                for row in self.safety.read_output():self.record('safety',row);self.count+=1
                if self.done.is_set():return
                self.done.wait(.002)
        except Exception as error:self.error=type(error).__name__+': '+str(error)

    def close(self):
        self.done.set();self.worker.join(timeout=3)
        if self.worker.is_alive():raise RuntimeError('Safety telemetry failed to drain')
        if self.error:raise RuntimeError('Safety telemetry failed: '+self.error)
