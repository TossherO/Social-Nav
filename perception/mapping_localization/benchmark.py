"""Bounded same-input native frontend or fixed-map ROLL comparison."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import yaml
from .native import NativeFrontend,NativeRoll


def statistics(values):
    values=np.asarray(values)
    if not len(values):return dict(samples=0)
    return dict(samples=len(values),mean=float(values.mean()),p50=float(np.median(values)),
        p95=float(np.percentile(values,95)),maximum=float(values.max()))


def run(config,batches,*,backend,plane_fit,warmup,deskew='cpu',linearize='cpu',compact=False):
    config=dict(config,matching_backend=backend,matching_cuda_plane_fit=plane_fit,deskew_backend=deskew,
        linearization_backend=linearize,filter_compact_solve=compact)
    with tempfile.TemporaryDirectory(prefix='social-nav-lio-') as directory:
        path=Path(directory)/'parameters.yaml';path.write_text(yaml.safe_dump(config))
        frontend=NativeFrontend(path);poses=[];states=[];clouds=[];timing=[];acceleration=[];stages=[]
        wall=time.perf_counter();cpu=time.process_time()
        try:
            for batch in batches:
                r,cloud=frontend.step(batch['imu'],batch['points'],batch['begin'],batch['end'])
                poses.append(r['world_from_lidar']);states.append(np.r_[r['velocity'],r['gyro_bias'],r['acc_bias'],
                    r['residual'],r['reason'],r['effective_points'],r['scan_points']])
                clouds.append(hashlib.sha256(cloud.tobytes()).hexdigest())
                timing.append([r[k] for k in ('processing_ms','matching_ms','map_update_ms')])
                acceleration.append(r.get('acceleration'))
                stages.append(r.get('stages'))
        finally:frontend.close()
        wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    poses=np.array(poses);states=np.array(states);timing=np.array(timing)
    steady=np.array([b['end']-batches[0]['end']>=warmup for b in batches])&(states[:,10]==0)
    report=dict(backend=backend,plane_fit=plane_fit,deskew_backend=deskew,linearization_backend=linearize,compact_solve=compact,
        scans=len(batches),valid=int((states[:,10]==0).sum()),
        wall_s=wall,cpu_s=cpu,stages_ms={k:statistics(timing[steady,j]) for j,k in enumerate(('frontend','matching','map_update'))})
    if acceleration and acceleration[0]:
        report['acceleration']={k:statistics(np.array([v[k] for v in acceleration])[steady]) for k in acceleration[0] if k!='backend'}
        report['map_count_mismatches']=sum(v['reference_points']>0 and v['map_points']!=v['reference_points'] for v in acceleration)
    if stages and stages[0]:report['breakdown']={k:statistics(np.array([v[k] for v in stages])[steady]) for k in stages[0]}
    return report,dict(poses=poses,states=states,cloud_hash=np.array(clouds),timing=timing)


def pose_difference(a,b):
    if not len(a['poses']):return dict(position_m=statistics([]),angle_deg=statistics([]),exact_poses=None)
    position=np.linalg.norm(a['poses'][:,:3,3]-b['poses'][:,:3,3],axis=1)
    # Cancel the common, rounded calibration matrix before measuring angle;
    # R.T @ R would mask small differences when that matrix is not exactly SO3.
    relative=b['poses'][:,:3,:3]@np.linalg.inv(a['poses'][:,:3,:3])
    u,_,vt=np.linalg.svd(relative);relative=u@vt
    angle=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    angle[np.all(a['poses'][:,:3,:3]==b['poses'][:,:3,:3],axis=(1,2))]=0
    return dict(position_m=statistics(position),angle_deg=statistics(angle),exact_poses=bool(np.array_equal(a['poses'],b['poses'])))


def difference(a,b):
    return dict(**pose_difference(a,b),exact_states=bool(np.array_equal(a['states'],b['states'])),
        exact_clouds=bool(np.array_equal(a['cloud_hash'],b['cloud_hash'])),
        quality_differences=int(np.count_nonzero(a['states'][:,10]!=b['states'][:,10])),
        max_effective_point_difference=float(np.max(np.abs(a['states'][:,11]-b['states'][:,11]))))


def freeze_roll_input(config,batches):
    """Generate one causal CPU deskew/pose stream, shared by every ROLL variant."""
    config=dict(config,matching_backend='cpu',deskew_backend='cpu',linearization_backend='cpu',filter_compact_solve=False)
    samples=[]
    with tempfile.TemporaryDirectory(prefix='social-nav-roll-input-') as directory:
        path=Path(directory)/'parameters.yaml';path.write_text(yaml.safe_dump(config));frontend=NativeFrontend(path)
        try:
            for batch in batches:
                r,cloud=frontend.step(batch['imu'],batch['points'],batch['begin'],batch['end'])
                if len(cloud) and r['reason'] in (0,4):samples.append((np.ascontiguousarray(cloud[:,:4]),r['world_from_lidar'],r['initial_lidar_pose'],r['stamp_s']))
        finally:frontend.close()
    return samples


def run_roll(config,samples,*,backend,plane_fit,warmup,first_stamp):
    config=copy.deepcopy(config);config['roll'].update(matchingBackend=backend,matchingCudaPlaneFit=plane_fit,localizationAsyncSubmap=False)
    poses=[];states=[];timing=[];stages=[]
    with tempfile.TemporaryDirectory(prefix='social-nav-roll-') as directory:
        path=Path(directory)/'parameters.yaml';path.write_text(yaml.safe_dump(config));matcher=NativeRoll(path)
        wall=time.perf_counter();cpu=time.process_time()
        try:
            for cloud,pose,reference,stamp in samples:
                begin=time.perf_counter();r=matcher.step(cloud,pose,reference,stamp);timing.append((time.perf_counter()-begin)*1000)
                poses.append(r['map_from_lidar']);states.append([r['available'],r['valid'],r['inlier_ratio'],r['registration_error'],r['map_revision']]);stages.append(r.get('stages'))
        finally:matcher.close()
        wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    steady=np.array([s[3]-first_stamp>=warmup for s in samples]);states=np.array(states);timing=np.array(timing)
    report=dict(backend=backend,plane_fit=plane_fit,scans=len(samples),valid=int(states[:,1].sum()),wall_s=wall,cpu_s=cpu,
        stages_ms={'roll':statistics(timing[steady])},configuration=config['roll'])
    if stages and stages[0]:report['breakdown']={k:statistics(np.array([v[k] for v in stages])[steady]) for k in stages[0]}
    return report,dict(poses=np.array(poses),states=states,timing=timing)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True);p.add_argument('--mode',choices=['localization','odometry','mapping'])
    p.add_argument('--duration',type=float,required=True);p.add_argument('--warmup',type=float,default=10)
    p.add_argument('--gpu',help='Physical visible-device selection; otherwise preserve CUDA_VISIBLE_DEVICES')
    p.add_argument('--gpu-plane',action='store_true',help='Also measure the experimental device plane fit')
    p.add_argument('--component',choices=['frontend','roll'],default='frontend')
    p.add_argument('--gpu-deskew',action='store_true');p.add_argument('--gpu-linearize',action='store_true')
    p.add_argument('--output',required=True);args=p.parse_args()
    if not np.isfinite([args.duration,args.warmup]).all() or not 0<=args.warmup<args.duration:
        raise ValueError('Require 0 <= warmup < duration')
    if args.component=='roll' and (args.gpu_deskew or args.gpu_linearize):raise ValueError('Select the frontend component for deskew/linearization trials')
    if args.gpu is not None:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    from perception.mapping_localization.configuration import algorithm_config
    from perception.sensors.observations import bag_observations
    from .results import file_fingerprint
    source=Path(args.source_run).resolve();manifest=json.loads((source/'run.json').read_text())
    mode=args.mode or manifest['mode'];config,_=algorithm_config(mode=mode,source_run=source)
    if args.component=='roll' and mode!='localization':raise ValueError('ROLL numerical comparison requires an immutable localization map')
    # Raw observations are frozen once, then replayed in order for each backend.
    # Bag I/O, CUDA construction and initialization are outside steady timings.
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError(output)
    start=float(manifest.get('start',manifest.get('playback',{}).get('start_offset_s',0.)) or 0.)
    bag=Path(manifest['bag']).resolve();batches=list(bag_observations(bag,config,duration=args.duration,start=start))
    if not batches:raise ValueError('No usable scans in the requested interval')
    output.mkdir(parents=True,exist_ok=False)
    reports={};arrays={}
    variants=[('cpu','cpu',False,{}),('cuda','cuda',False,{})]+([('cuda_plane','cuda',True,{})] if args.gpu_plane else [])
    if args.gpu_deskew:variants.append(('cuda_deskew','cuda',False,dict(deskew='cuda')))
    if args.gpu_linearize:variants.append(('cuda_linearize','cuda',False,dict(linearize='cuda')))
    if args.gpu_deskew and args.gpu_linearize:variants.append(('cuda_combined','cuda',args.gpu_plane,dict(deskew='cuda',linearize='cuda',compact=True)))
    samples=freeze_roll_input(config,batches) if args.component=='roll' else None
    if samples is not None and not samples:raise ValueError('No initialized frontend scans available for ROLL')
    for name,backend,fit,options in variants:
        if samples is None:reports[name],arrays[name]=run(config,batches,backend=backend,plane_fit=fit,warmup=args.warmup,**options)
        else:reports[name],arrays[name]=run_roll(config,samples,backend=backend,plane_fit=fit,warmup=args.warmup,first_stamp=batches[0]['end'])
        print(name,reports[name]['stages_ms'],flush=True)
    differences={}
    for k,v in arrays.items():
        if k=='cpu':continue
        if samples is None:differences[k]=difference(arrays['cpu'],v)
        else:
            available=arrays['cpu']['states'][:,0].astype(bool)&v['states'][:,0].astype(bool)
            differences[k]=dict(**pose_difference({'poses':arrays['cpu']['poses'][available]},{'poses':v['poses'][available]}),
                exact_states=bool(np.array_equal(arrays['cpu']['states'],v['states'])),quality_differences=int(np.count_nonzero(arrays['cpu']['states'][:,:2]!=v['states'][:,:2])))
    report=dict(scope='native_'+args.component+'_only_tight_replay_not_full_load_or_external_accuracy',
        source_run=str(source),bag=str(bag),bag_fingerprint=file_fingerprint(bag),config=config,
        start_s=start,duration_s=args.duration,warmup_s=args.warmup,affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),results=reports,
        differences=differences)
    (output/'results.json').write_text(json.dumps(report,allow_nan=False,indent=2)+'\n')
    np.savez_compressed(output/'traces.npz',**{name+'_'+k:v for name,a in arrays.items() for k,v in a.items()})


if __name__=='__main__':main()
