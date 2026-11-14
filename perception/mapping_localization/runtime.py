"""Owned native localization session shared by bag and sensor-edge adapters."""
import hashlib
import json
from pathlib import Path
import uuid
import queue
import threading
import time
import numpy as np
import yaml
from .context import LocalizationContext
from .native import NativeFrontend,NativeRoll
from .results import map_fingerprint


def quaternion(rotation):
    # Stable symmetric eigensolve, also at pi. No ROS/SciPy dependency.
    r=np.asarray(rotation);k=np.array([[r[0,0]-r[1,1]-r[2,2],r[0,1]+r[1,0],r[0,2]+r[2,0],r[2,1]-r[1,2]],
        [r[0,1]+r[1,0],r[1,1]-r[0,0]-r[2,2],r[1,2]+r[2,1],r[0,2]-r[2,0]],
        [r[0,2]+r[2,0],r[1,2]+r[2,1],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
        [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],np.trace(r)]])/3
    _,vectors=np.linalg.eigh(k);q=vectors[:,-1]
    return q if q[3]>=0 else -q


class LocalizationSession:
    def __init__(self,config_path,*,mode='odometry',coverage=None,allow_local_only=False,metrics=None,association_ready=None,latest_matching=False):
        if mode not in ('mapping','localization','odometry'):raise ValueError('Unsupported native localization mode')
        self.mode=mode;self.latest_matching=bool(latest_matching and mode=='localization');self.session=uuid.uuid4().hex;self.config=yaml.safe_load(Path(config_path).read_text());self.metrics=metrics
        self.map_id='';self.asset_fingerprint=''
        if mode=='localization':
            directory=Path(self.config['roll']['saveKeyframeMapDirectory']).parent
            self.asset_fingerprint=hashlib.sha256(json.dumps(map_fingerprint(directory),sort_keys=True).encode()).hexdigest()
            self.map_id='roll:'+self.asset_fingerprint
        if mode=='localization':
            from .coverage import PriorCoverage
            record=self.config['roll'].get('localization_coverage')
            if not record:
                raise ValueError('Localization configuration must include map-bound coverage')
            frozen=PriorCoverage(record,asset_fingerprint=self.asset_fingerprint)
            if coverage and PriorCoverage.load(coverage,asset_fingerprint=self.asset_fingerprint).fingerprint != frozen.fingerprint:
                raise ValueError('Coverage changed after localization configuration was prepared')
            coverage=frozen
        self.frontend=NativeFrontend(config_path);self.matcher=None
        if mode!='odometry':self.matcher=NativeRoll(config_path)
        from perception.environment.layers import load_config as environment_config
        association_config=environment_config()
        self.context=LocalizationContext('fast_lio' if mode=='odometry' else 'roll_fast_lio',
            requires_map=mode=='localization',allow_local_only=allow_local_only,coverage=coverage,
            max_map_age=association_config['association_max_age_s'],
            confirm_matches=association_config['association_confirm_matches'])
        self.context_lock=threading.Lock();self.association_ready=association_ready
        # Live/fixed-map localization uses the newest causal scans; an old
        # matching backlog cannot become current localization evidence. Mapping
        # keeps its original lossless queue because every scan may affect export.
        self.jobs=queue.Queue(2 if self.latest_matching else 64);self.completed=queue.Queue(64);self.error=None;self.worker=None
        if self.matcher is not None:
            self.worker=threading.Thread(target=self._match,name='native-roll',daemon=True);self.worker.start()

    def _match(self):
        try:
            while True:
                job=self.jobs.get()
                if job is None:return
                if job[0]=='save':
                    _,path,event=job
                    try:
                        if self.error:raise RuntimeError(self.error)
                        self.matcher.save(path)
                    except Exception as error:self.error=type(error).__name__+': '+str(error)
                    finally:event.set()
                    continue
                if self.error:continue
                _,cloud,pose,reference,row=job
                try:
                    if self.metrics is not None:self.metrics.trace('match_start',measurement_stamp_s=row['stamp_s'])
                    begin=time.perf_counter();match=self.matcher.step(cloud[:,:4],pose,reference,row['stamp_s'])
                    if self.metrics is not None:self.metrics.trace('match_done',measurement_stamp_s=row['stamp_s'],
                        available=bool(match['available']),valid=bool(match['valid']),
                        inlier_ratio=match['inlier_ratio'] if np.isfinite(match['inlier_ratio']) else None,
                        registration_error=match['registration_error'] if np.isfinite(match['registration_error']) else None,
                        gate_diagnostics=match.get('gate_diagnostics'))
                    if self.metrics is not None:
                        self.metrics.add('roll_processing',time.perf_counter()-begin)
                        stages=match.get('stages')
                        if stages:
                            for key,value in stages.items():
                                if key.endswith('_ms'):self.metrics.add('roll_'+key,value/1000)
                            if stages['backend']:
                                for direction in ('h2d','d2h'):self.metrics.transfer(direction,stages[direction+'_bytes'],'roll_cuda_per_frame_totals')
                    if match['available']:
                        mapped=match['map_from_lidar']
                        measured=dict(row,session_id=self.session+':map',health=0 if match['valid'] else 2,
                            has_map_pose=match['valid'],map_frame='map',map_position=mapped[:3,3].tolist(),
                            map_quaternion_xyzw=quaternion(mapped[:3,:3]).tolist(),map_revision=match['map_revision'])
                        with self.context_lock:
                            self.context.add_match(measured)
                            refreshed=self.context.refresh()
                            association=None if refreshed is None else self._association(refreshed[1],refreshed[0]['stamp_s'])
                        if association is not None:
                            if self.metrics is not None:self.metrics.trace('match_consumed',measurement_stamp_s=measured['stamp_s'],scene_stamp_s=association['stamp_s'])
                            if self.association_ready is not None:self.association_ready(association)
                        self.completed.put_nowait((measured,match))
                except Exception as error:self.error=type(error).__name__+': '+str(error)
        finally:self.matcher.close()

    def step(self,batch):
        if self.error:raise RuntimeError('Native map matcher failed: '+self.error)
        result,cloud=self.frontend.step(batch['imu'],batch['points'],batch['begin'],batch['end'])
        pose=result['world_from_lidar'];stamp=result['stamp_s'];q=quaternion(pose[:3,:3]).tolist()
        row=dict(stamp_s=stamp,stamp_ns=batch.get('stamp_ns',round(stamp*1e9)),frame_id='odom',child_frame_id=batch['sensor_frame'],
            position=pose[:3,3].tolist(),quaternion_xyzw=q,health=0 if result['valid'] else 2,
            session_id=self.session,generation=0,source='fast_lio',reason='' if result['valid'] else 'frontend_'+str(result['reason']),
            has_map_pose=False,map_frame='',map_revision=0)
        match=None
        while True:
            try:measured,match=self.completed.get_nowait()
            except queue.Empty:break
        with self.context_lock:state,association=self.context.update(row)
        if self.matcher is not None and len(cloud) and result['reason'] in (0,4):
            if self.metrics is not None:self.metrics.trace('match_submit',measurement_stamp_s=stamp)
            self.submit_match(('scan',cloud,pose,result['initial_lidar_pose'],row))
        association=self._association(association,stamp)
        if 'map_matrix' in state:
            mapped=state.pop('map_matrix');state.update(map_position=mapped[:3,3].tolist(),map_quaternion_xyzw=quaternion(mapped[:3,:3]).tolist())
        return state,association,cloud,result,match

    def submit_match(self, job):
        try:
            self.jobs.put_nowait(job)
            return
        except queue.Full:
            if not self.latest_matching:
                raise RuntimeError('Native map matching backlog exceeded 64 scans')
        try:
            replaced=self.jobs.get_nowait()
        except queue.Empty:
            replaced=None  # The worker freed the slot concurrently.
        if replaced is not None:
            if replaced[0]!='scan':
                raise RuntimeError('Cannot replace a non-scan localization job')
            if self.metrics is not None:
                self.metrics.event('localization_match_scans_replaced')
                self.metrics.trace('match_replaced',measurement_stamp_s=replaced[4]['stamp_s'],
                                   replacement_stamp_s=job[4]['stamp_s'])
        self.jobs.put_nowait(job)

    def _association(self,association,stamp):
        association.update(stamp_s=stamp,stamp_ns=round(stamp*1e9),local_frame='odom',local_session_id=self.session,
            local_generation=0,map_id=self.map_id if self.mode!='mapping' else 'mapping:'+self.session+':map',asset_fingerprint=self.asset_fingerprint)
        if isinstance(association.get('map_from_local'),np.ndarray):association['map_from_local']=association['map_from_local'].tolist()
        return association

    def save(self,directory):
        if self.mode!='mapping':raise ValueError('Only mapping sessions produce maps')
        done=threading.Event();self.jobs.put(('save',directory,done),timeout=10)
        if not done.wait(60):raise RuntimeError('Native map export exceeded its drain budget')
        if self.error:raise RuntimeError(self.error)

    def close(self):
        self.frontend.close()
        if self.worker is not None:
            self.jobs.put(None,timeout=10);self.worker.join(timeout=30)
            if self.worker.is_alive():raise RuntimeError('Native map matching worker did not stop')
