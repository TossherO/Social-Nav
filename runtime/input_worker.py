"""System-Python sensor edge and native localization; arrays leave through memfd."""
import argparse
from collections import deque
import json
import socket
import signal
import threading
import time
import numpy as np
from perception.mapping_localization.runtime import LocalizationSession
from perception.environment.geometry import interpolate_origins
from .inputs import bag_observations
from .buffers import SharedArrayPool
from .transport import Connection,frame_wire
from .metrics import Metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket-fd',type=int,required=True);p.add_argument('--pose-fd',type=int,required=True)
    p.add_argument('--main-pose-fd',type=int,required=True)
    p.add_argument('--association-fd',type=int,required=True)
    p.add_argument('--shared-slots',type=int,default=4);p.add_argument('--slot-bytes',type=int,default=8*1024*1024)
    p.add_argument('--config',required=True);p.add_argument('--bag');p.add_argument('--live',action='store_true');p.add_argument('--mode',required=True)
    p.add_argument('--duration',type=float);p.add_argument('--rate',type=float,default=1.);p.add_argument('--coverage');p.add_argument('--map-output')
    p.add_argument('--with-rgb',action='store_true',help='Read camera input only for explicitly requested display, archive or VLM preparation')
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--origin-gap',type=float,default=.2)
    p.add_argument('--timing-output',help='Optional bounded scalar timing trace; flushed at shutdown')
    args=p.parse_args();wire=Connection(socket.socket(fileno=args.socket_fd));states=Connection(socket.socket(fileno=args.pose_fd))
    stop=threading.Event()
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    main_pose=socket.socket(fileno=args.main_pose_fd);main_pose.setblocking(False)
    pool=SharedArrayPool(args.shared_slots,args.slot_bytes);wire.send_fd(pool.fd)
    wire.send(dict(kind='pool',owner=pool.owner,slots=pool.slots,slot_bytes=pool.slot_bytes))
    metrics=Metrics(trace_path=args.timing_output)
    association_channel=socket.socket(fileno=args.association_fd);association_channel.setblocking(False)
    def association_ready(value):
        try:association_channel.send(json.dumps(value,allow_nan=False,separators=(',',':')).encode())
        except BlockingIOError:metrics.event('association_updates_dropped');return
        metrics.trace('association_publish',measurement_stamp_s=value['stamp_s'],association_stamp_s=value['measurement_stamp_s'],valid=value['valid'])
    session=LocalizationSession(args.config,mode=args.mode,coverage=args.coverage,allow_local_only=True,metrics=metrics,association_ready=association_ready)
    pending={};first=None;anchor=None;count=0;cpu_start=time.process_time();positions=deque(maxlen=64)
    def release():
        ack=wire.recv();key=(ack['slot'],ack['epoch']);ref=pending.pop(key)
        pool.release(ref,'gpu')
    try:
        if args.live:
            from .ros_input import messages
            source=messages(session.config,stop_event=stop,include_rgb=args.with_rgb)
        else:source=None
        iterator=bag_observations(args.bag,session.config,duration=args.duration,start=args.start,messages=source,
            image_topics=dict(rgb='/camera/color/image_raw',camera_info='/camera/color/camera_info') if args.with_rgb else None)
        for batch in iterator:
            if stop.is_set():break
            if first is None:
                first=batch['end'];anchor=time.monotonic();wire.send(dict(kind='ready'))
            if args.rate>0:
                wait=0. if args.live else (batch['end']-first)/args.rate-(time.monotonic()-anchor)
                if wait>0:time.sleep(wait)
            acquired=time.monotonic()
            if args.live:
                import rospy
                acquired_clock=rospy.Time.now().to_sec()
            else:acquired_clock=first+(acquired-anchor)*args.rate
            metrics.trace('input_ready',measurement_stamp_s=batch['end'],source_clock_s=acquired_clock,source_mono=acquired)
            with metrics.span('localization_session'):
                state,association,cloud,front,match=session.step(batch)
            metrics.trace('localization_done',measurement_stamp_s=batch['end'],association_stamp_s=association['measurement_stamp_s'],association_valid=association['valid'])
            for name in ('processing_ms','matching_ms','map_update_ms'):metrics.add('frontend_'+name,front[name]/1000)
            stages=front.get('stages')
            if stages:
                for name,value in stages.items():
                    if name.endswith('_ms'):metrics.add('frontend_'+name,value/1000)
                for direction in ('h2d','d2h'):
                    if stages['deskew_'+direction+'_bytes']:metrics.transfer(direction,stages['deskew_'+direction+'_bytes'],'frontend_cuda_deskew')
            acceleration=front.get('acceleration')
            if acceleration and acceleration['backend']=='cuda':
                for name in ('map_sync_ms','match_wall_ms','query_device_ms'):metrics.add('frontend_cuda_'+name,acceleration[name]/1000)
                for direction in ('h2d','d2h'):metrics.transfer(direction,acceleration[direction+'_bytes'],'frontend_cuda_map_and_matching')
            clock_mono=time.monotonic()
            source_clock=rospy.Time.now().to_sec() if args.live else first+(clock_mono-anchor)*args.rate
            states.send(dict(kind='state',state=state,clock_mono=clock_mono,
                clock_s=source_clock,clock_rate=args.rate))
            try:main_pose.send(json.dumps(dict(state=state,clock_s=source_clock,
                clock_mono=clock_mono,clock_rate=args.rate),allow_nan=False,separators=(',',':')).encode())
            except BlockingIOError:pass  # Bounded latest-state telemetry; safety has its own lossless channel.
            arrays=dict(raw=batch['raw'])
            if state['health']==0:positions.append((state['stamp_s'],state['position']))
            else:positions.clear()
            if len(cloud) and front['reason'] in (0,4):
                arrays['deskewed']=cloud
                origins=interpolate_origins(cloud[:,4],state['stamp_s'],list(positions),args.origin_gap)
                if origins is not None:arrays['ray_origins']=origins
            if batch['rgb'] is not None:arrays['rgb']=batch['rgb']
            while len(pending)>=pool.slots:release()
            with metrics.span('input_shared_publish'):ref=pool.publish(arrays,readers=['gpu'])
            pending[ref.slot,ref.epoch]=ref;count+=1
            metrics.transfer('host_copy',sum(a.nbytes for a in arrays.values()),'sensor_shared_slots')
            metrics.trace('scan_publish',measurement_stamp_s=batch['end'])
            wire.send(dict(kind='scan',buffer=frame_wire(ref),state=state,association=association,
                observation_id=str(count),stamp_s=batch['end'],begin_s=batch['begin'],sensor_frame=batch['sensor_frame'],
                scan_duration_s=float(np.ptp(cloud[:,4]))/1000 if len(cloud) else None,
                source_mono=acquired,source_clock_s=acquired_clock,
                producer_complete_mono=time.monotonic(),rgb_stamp=batch['rgb_stamp'],rgb_stamp_ns=batch['rgb_stamp_ns'],rgb_frame=batch['rgb_frame'],camera_info=batch['camera_info'],
                frontend_timing={k:front[k] for k in ('processing_ms','matching_ms','map_update_ms')},frontend_acceleration=acceleration,
                frontend_valid=front['valid'],backend_valid=None if match is None else match['valid']))
        while pending:release()
        wire.send(dict(kind='draining'))
        if args.map_output:session.save(args.map_output)
        states.send(dict(kind='end'))
        source_metrics=metrics.summary();source_metrics['process_cpu_s']=time.process_time()-cpu_start
        source_metrics['source_wall_s']=None if anchor is None else time.monotonic()-anchor
        source_metrics['stop_requested']=stop.is_set()
        wire.send(dict(kind='end',scans=count,metrics=source_metrics))
    except Exception as error:
        wire.send(dict(kind='error',error=type(error).__name__+': '+str(error)));raise
    finally:
        session.close()
        association_channel.close()
        metrics.write_trace()
        for ref in pending.values():pool.release(ref,'gpu')
        pool.close();wire.close();states.close();main_pose.close()


if __name__=='__main__':main()
