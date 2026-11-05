"""Replay recorded processed geometry on an isolated ROS master; no SLAM or robot control."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from xmlrpc.client import ServerProxy

import numpy as np

from perception.mapping_localization.__main__ import ROOT, stop
from perception.mapping_localization.results import file_fingerprint, load_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Successful run recorded with --record-display')
    parser.add_argument('--start', type=float, default=0, help='Seconds from display.bag start')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--no-rviz', action='store_true', help='Headless playback and message-consumption check')
    args = parser.parse_args(argv)
    if (not math.isfinite(args.start) or args.start < 0 or not math.isfinite(args.rate) or
            not .05 <= args.rate <= 32 or args.duration is not None and
            (not math.isfinite(args.duration) or args.duration <= 0)):
        parser.error('Use finite nonnegative start, positive duration and rate in [0.05, 32]')
    result = load_result(args.result)
    recording = result.path/'display.bag'
    if not result.metadata.get('success') or file_fingerprint(recording) != result.metadata.get('display_recording_fingerprint'):
        raise ValueError('A complete, bound display recording is required')
    if not args.no_rviz and not os.environ.get('DISPLAY'):
        raise RuntimeError('RViz needs a desktop/X11 display; use --no-rviz for headless playback')
    import rosbag
    with rosbag.Bag(str(recording)) as bag:
        if args.start >= bag.get_end_time()-bag.get_start_time():
            raise ValueError('Start exceeds recorded display duration')
    master = playback = view = None
    with tempfile.TemporaryDirectory(prefix='social-nav-geometry-playback-') as scratch:
        env = dict(os.environ, ROS_IP='127.0.0.1', ROS_HOME=scratch, ROS_LOG_DIR=scratch+'/logs',
                   DISABLE_ROS1_EOL_WARNINGS='1')
        env.pop('ROS_HOSTNAME', None)
        os.environ.pop('ROS_HOSTNAME', None)
        with socket.socket() as selector:
            selector.bind(('127.0.0.1', 0))
            port = selector.getsockname()[1]
        env['ROS_MASTER_URI'] = 'http://127.0.0.1:%d'%port
        os.environ.update({k:env[k] for k in ('ROS_MASTER_URI','ROS_IP','ROS_HOME','ROS_LOG_DIR')})
        with (Path(scratch)/'processes.log').open('w') as log:
            try:
                master = subprocess.Popen(['roscore','-p',str(port)],env=env,stdout=log,stderr=log,start_new_session=True)
                proxy = ServerProxy(env['ROS_MASTER_URI']);deadline=time.monotonic()+15
                while True:
                    try:
                        if proxy.getPid('/geometry_playback')[0] == 1: break
                    except OSError: pass
                    if master.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('Isolated playback master did not start')
                    time.sleep(.1)
                import rospy
                import yaml
                from rosgraph_msgs.msg import Clock
                from nav_msgs.msg import OccupancyGrid
                from sensor_msgs.msg import PointCloud2
                from perception.visualization.viewer import Scene
                from perception.visualization.environment import add_environment_display
                proxy.setParam('/geometry_playback','/use_sim_time',True)
                rospy.init_node('geometry_playback',anonymous=True,disable_signals=True)
                scene=Scene(result,'odom',np.empty((0,3)),None,20000,250000)
                received=[0]
                prefix=result.metadata['environment_configuration']['output_topic']
                consumed=dict(scans=0,local_grids=0,prior_grids=0)
                def count(name):
                    def callback(message):consumed[name]+=1
                    return callback
                subscribers=[rospy.Subscriber(prefix+'/scan',PointCloud2,count('scans'),queue_size=8),
                             rospy.Subscriber(prefix+'/occupancy',OccupancyGrid,count('local_grids'),queue_size=8)]
                if result.metadata.get('prior_map'):
                    prior_topic=result.metadata['environment_configuration']['scene']['output_topic']+'/prior/occupancy'
                    subscribers.append(rospy.Subscriber(prior_topic,OccupancyGrid,count('prior_grids'),queue_size=8))
                def clock(message):
                    elapsed=message.clock.to_sec()-result.rows[0]['stamp_s']
                    scene.show(result.index_at(elapsed))
                    received[0]+=1
                subscriber=rospy.Subscriber('/clock',Clock,clock,queue_size=1)
                if not args.no_rviz:
                    config=yaml.safe_load((ROOT/'perception/visualization/configs/results.rviz').read_text())
                    config['Visualization Manager']['Global Options']['Fixed Frame']=result.frame_name('odom')
                    prior=(result.metadata['environment_configuration']['scene']['output_topic']+'/prior/occupancy'
                           if result.metadata.get('prior_map') else None)
                    add_environment_display(config,prefix,prior_topic=prior)
                    config['Visualization Manager']['Displays']=[d for d in config['Visualization Manager']['Displays']
                        if d.get('Name') not in ('Saved map','Recorded observation scan','Recorded people detections and tracks')]
                    config['Visualization Manager']['Displays'].append(dict(Class='rviz/PointCloud2',Name='Recorded deskewed scan',
                        Enabled=True,Topic=prefix+'/scan',**{'Queue Size':1,'Style':'Points','Size (Pixels)':2,
                                                          'Color Transformer':'AxisColor','Axis':'Z','Decay Time':0}))
                    camera=config['Visualization Manager']['Views']['Current'];camera['Distance']=25
                    first=result.rows[result.index_at(args.start)]
                    if first.get('health')!=0:first=next((r for r in result.rows if r.get('health')==0),None)
                    if first:camera['Focal Point']=dict(zip(('X','Y','Z'),first['position']))
                    path=Path(scratch)/'playback.rviz';path.write_text(yaml.safe_dump(config))
                    view=subprocess.Popen(['rviz','-d',str(path)],env=env,stdout=log,stderr=log,start_new_session=True)
                command=['rosbag','play','--quiet','--clock','--delay','2','--rate',str(args.rate),'--start',str(args.start)]
                if args.duration is not None:command+=['--duration',str(args.duration)]
                command.append(str(recording))
                playback=subprocess.Popen(command,env=env,start_new_session=True)
                while playback.poll() is None:
                    if view is not None and view.poll() is not None:break
                    time.sleep(.1)
                if playback.poll() not in (None,0):raise RuntimeError('Display bag playback failed')
                time.sleep(.2)
                print(json.dumps(dict(clock_messages=received[0],consumed=consumed,scope='recorded_geometry_no_robot_control')))
                if args.no_rviz and received[0]==0:raise RuntimeError('No playback clock was consumed')
                while args.duration is None and view is not None and view.poll() is None:time.sleep(.1)
            except KeyboardInterrupt:
                pass
            finally:
                for process in (playback,view):stop(process,first_signal=signal.SIGTERM)
                if 'rospy' in locals():rospy.signal_shutdown('Display playback finished')
                stop(master)


if __name__=='__main__':
    main()
