"""Rebuild selected local scenes causally from raw scans and saved LiDAR poses."""
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import numpy as np

from .context import fingerprint, layer_binding
from .layers import LocalEnvironmentMap, load_config
from .scene import SceneInputs
from .storage import asset_reference
from perception.mapping_localization.results import file_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def extract(run_path, scenes, destination):
    """System ROS subprocess: raw arrays only, no cross-Python ROS extension."""
    import rosbag
    import rospy
    from perception.sensors.pointcloud import pointcloud_fields
    run = json.loads((Path(run_path)/'run.json').read_text())
    if file_fingerprint(run['bag']) != run['playback']['bag_fingerprint']:
        raise ValueError('Raw bag changed after recorded localization')
    windows = json.loads(scenes)
    merged=[]
    for low,high in sorted(windows):
        if not np.isfinite([low,high]).all() or high<low:raise ValueError('Invalid raw scan window')
        if merged and low<=merged[-1][1]+1e-6:merged[-1][1]=max(high,merged[-1][1])
        else:merged.append([low,high])
    db = sqlite3.connect(destination)
    db.execute('CREATE TABLE scans(stamp REAL PRIMARY KEY, payload BLOB)')
    with rosbag.Bag(run['bag']) as bag:
        first = next(bag.read_messages(topics=['/rslidar_points']))
        receipt_offset = first.timestamp.to_sec()-first.message.header.stamp.to_sec()
        for low,high in merged:
            for _, cloud, _ in bag.read_messages(topics=['/rslidar_points'],
                    start_time=rospy.Time.from_sec(low+receipt_offset-.3),
                    end_time=rospy.Time.from_sec(high+receipt_offset+.3)):
                stamp = cloud.header.stamp.to_sec()
                if not low<=stamp<=high+1e-6:continue
                if 'timestamp' not in {f.name for f in cloud.fields}:
                    raise ValueError('Causal deskew requires measured per-point timestamps')
                values = pointcloud_fields(cloud, ['x','y','z','timestamp'])
                stream = io.BytesIO()
                # This cache is disposable. Compression belongs in retained artifacts.
                np.savez(stream, points=np.column_stack([values[k] for k in ['x','y','z']]),
                         times=values['timestamp'], frame=np.asarray(cloud.header.frame_id))
                db.execute('INSERT INTO scans VALUES (?,?)', (stamp,stream.getvalue()))
    db.commit(); db.close()


def _source(run_path):
    from .people import PeopleObservations
    run_path=Path(run_path)
    run=json.loads((run_path/'run.json').read_text())
    if not run.get('success') or run.get('source')!='bag':
        raise ValueError('A successful saved bag run is required')
    blind=run['algorithm_parameters']['preprocess']['blind']
    if not np.isfinite(blind) or blind<0: raise ValueError('Invalid recorded frontend blind range')
    rows=[json.loads(line) for line in (run_path/'localization.jsonl').read_text().splitlines() if line.strip()]
    binding=dict(source_run=asset_reference(run_path),source_run_fingerprint=file_fingerprint(run_path/'run.json'),
                 localization_fingerprint=file_fingerprint(run_path/'localization.jsonl'),
                 bag_fingerprint=run['playback']['bag_fingerprint'])
    people=None
    if (run_path/'people.jsonl').is_file():
        people=PeopleObservations(run_path,run_fingerprint=binding['source_run_fingerprint'],
                                  bag_fingerprint=binding['bag_fingerprint'])
        binding['people_fingerprint']=file_fingerprint(run_path/'people.jsonl')
    return run,rows,people,binding


def _extract(run_path,windows,database):
    environment=dict(os.environ,PYTHONNOUSERSITE='1',
        PYTHONPATH=str(ROOT)+':/opt/ros/noetic/lib/python3/dist-packages')
    environment['LD_LIBRARY_PATH']='/opt/ros/noetic/lib:'+environment.get('LD_LIBRARY_PATH','')
    budget=max(120.,min(900.,30.+2*sum(high-low for low,high in windows)))
    subprocess.run(['/usr/bin/python3','-B','-m',__name__,str(run_path),json.dumps(windows),str(database)],
                   env=environment,cwd=ROOT,check=True,timeout=budget)


def _retention(config):
    return config['voxel_ttl_s']


def replay_scans(database, rows, run, config, start, cutoff, *, people=None, clock_domain=None,sample_times=None,
                 model_factory=None):
    """Stream raw observations once. Invalid poses reset evidence, never bridge it.

    All raw scans are processed unless explicit sampling cutoffs are supplied.
    This does not simulate scheduling, queues or drops in a live ROS process.
    """
    import time
    from scipy.spatial.transform import Rotation,Slerp
    from .poses import PoseTimeline
    timeline=PoseTimeline(rows,config['poses'])
    if sample_times is not None:
        sample_times=np.asarray(sample_times,dtype=float)
        if not len(sample_times) or not np.isfinite(sample_times).all() or np.any(np.diff(sample_times)<=0):
            raise ValueError('Replay sampling cutoffs must be finite and increasing')
    times=timeline.times
    segments={}
    for index,segment in enumerate(timeline.segments):
        if segment is not None: segments.setdefault(segment,[]).append(index)
    interpolation={}
    for segment,indices in segments.items():
        t=times[indices]
        if len(t)<2:continue
        positions=np.asarray([rows[i]['position'] for i in indices])
        rotations=Rotation.from_quat([rows[i]['quaternion_xyzw'] for i in indices])
        interpolation[segment]=(t,positions,rotations,Slerp(t-t[0],rotations),{j:k for k,j in enumerate(indices)})
    model=(model_factory or LocalEnvironmentMap)(config);active=None
    stats=dict(accepted_scans=0,invalid_scans=0,future_firings_excluded=0,environment_ms=0.,deskew_ms=0.)
    try:
        for raw_stamp,payload in database.execute(
                'SELECT stamp,payload FROM scans WHERE stamp>=? AND stamp<=? ORDER BY stamp',(start,cutoff+1e-6)):
            index=int(np.argmin(abs(times-raw_stamp)))
            if abs(times[index]-raw_stamp)>1e-6:continue
            row=rows[index];stamp=float(times[index]);segment=timeline.segments[index]
            if segment is None or segment not in interpolation:
                model.reset();active=None;stats['invalid_scans']+=1
                yield None,row,dict(stats)
                continue
            if segment!=active:
                model.reset();active=segment
            if sample_times is not None and np.min(abs(sample_times-stamp))>1e-6:continue
            begin=time.perf_counter()
            t,positions,rotations,interpolator,indices=interpolation[segment]
            end_index=indices[index]
            with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
                points,firing,frame=arrays['points'],arrays['times'],arrays['frame'].item()
            if frame.lstrip('/')!=row['child_frame_id'].lstrip('/'):
                raise ValueError('Raw scan frame does not match saved localization')
            keep=(np.isfinite(points).all(axis=1)&np.isfinite(firing)&(firing<=stamp)&(firing>=t[0])&
                  (np.linalg.norm(points,axis=1)>=run['algorithm_parameters']['preprocess']['blind']))
            stats['future_firings_excluded']+=int(np.count_nonzero(np.isfinite(firing)&(firing>stamp)))
            points,firing=points[keep],firing[keep]
            if not len(points):continue
            hi=end_index+1
            lo=max(0,np.searchsorted(t,firing.min(),side='right')-1)
            if np.any(np.diff(t[lo:hi])>config['max_origin_interpolation_gap_s']):
                model.reset();active=None;stats['invalid_scans']+=1
                yield None,row,dict(stats)
                continue
            world=interpolator(firing-t[0]).apply(points)
            origins=np.column_stack([np.interp(firing,t[:hi],positions[:hi,k]) for k in range(3)])
            world+=origins
            rotation=rotations[end_index].as_matrix();position=positions[end_index]
            deskewed=(world-position)@rotation
            transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=position
            stats['deskew_ms']+=(time.perf_counter()-begin)*1000
            begin=time.perf_counter()
            snapshot=model.update(deskewed,transform,stamp,(row['session_id'],row['generation']),frame=row['frame_id'],
                scan_duration=float(np.ptp(firing)),ray_origins=origins,people=people.at(stamp) if people else None,
                clock_domain=clock_domain)
            snapshot.metadata.update(localization_source=row.get('source',run.get('localization_source','recorded_localization')),
                                     sensor_frame=row['child_frame_id'])
            stats['accepted_scans']+=1;stats['environment_ms']+=(time.perf_counter()-begin)*1000
            yield snapshot,row,dict(stats)
    finally:model.close()


def _derive(snapshot,original,binding,config,run,start,stats,*,row):
    from scipy.spatial.transform import Rotation
    from perception.mapping_localization.context import pose_matrix
    stored=pose_matrix(row['position'],row['quaternion_xyzw'])
    if (abs(original['as_of']-row['stamp_s'])>1e-6 or
            (original['session_id'],original['generation'],original['frame_id'])!=
            (row['session_id'],row['generation'],row['frame_id']) or
            not np.allclose(original['localization']['world_from_lidar'],stored,atol=1e-6,rtol=0)):
        raise ValueError('Original scene does not match the recorded source pose')
    # Verify the original raw-quaternion conversion first. Normalize exactly as
    # causal Slerp does, rather than loosening a pose tolerance to accept drift.
    expected=stored.copy();expected[:3,:3]=Rotation.from_quat(row['quaternion_xyzw']).as_matrix()
    if not np.allclose(expected,snapshot.metadata['world_from_lidar'],atol=1e-6,rtol=0):
        raise ValueError('Original scene and saved localization cutoff poses disagree')
    derivation=dict(method='causal_raw_scan_pose_replay',source_scene_fingerprint=original['input_fingerprint'],
        **binding,window_s=float(snapshot.metadata['stamp_s']-start),configuration=config,
        recorded_frontend_blind_m=run['algorithm_parameters']['preprocess']['blind'],
        pose_conversion='unit_quaternion_for_causal_slerp',
        source_quaternion_norm=float(np.linalg.norm(row['quaternion_xyzw'])),
        **{k:v for k,v in stats.items() if not k.endswith('_ms')})
    snapshot.metadata.update(snapshot_id='causal_rebuild:'+fingerprint(derivation),association=copy.deepcopy(original['prior']))
    return derivation


def rebuild_scenes(run_path, scenes, output, *, config=None, window_s=8.):
    """No future poses/points, no final-map corrections, no model inference."""
    run_path,output=Path(run_path),Path(output)
    if output.exists():raise FileExistsError(output)
    if not np.isfinite(window_s) or window_s<=0:raise ValueError('Invalid replay window')
    run,rows,people,binding=_source(run_path)
    sources=[SceneInputs.load(p) for p in scenes]
    config=copy.deepcopy(config or load_config())
    window_s=max(window_s,_retention(config))
    windows=[[s.record['as_of']-window_s-.25,s.record['as_of']] for s in sources]
    result=[]
    with tempfile.TemporaryDirectory(prefix='environment-causal-') as scratch:
        database=Path(scratch)/'raw.sqlite';_extract(run_path,windows,database)
        db=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for scene_path,original,(start,cutoff) in zip(scenes,sources,windows):
                snapshot=None
                for snapshot,row,stats in replay_scans(db,rows,run,config,start,cutoff,people=people,
                                                       clock_domain=original.record['clock_domain']):pass
                if snapshot is None or abs(snapshot.metadata['stamp_s']-cutoff)>1e-6:
                    raise ValueError('Replay did not reach the exact original scene cutoff')
                derivation=_derive(snapshot,original.record,binding,config,run,start,stats,row=row)
                derivation['source_scene']=asset_reference(scene_path)
                record=copy.deepcopy(original.record);record.pop('input_fingerprint');record.pop('prior_geometry',None)
                record['geometry'].update(snapshot_id=snapshot.metadata['snapshot_id'],layer_binding=layer_binding(snapshot),
                                          geometry_status=snapshot.metadata['geometry_status'])
                record['observation_id']='scene:'+snapshot.metadata['snapshot_id'];record['derivation']=derivation
                record['input_fingerprint']=fingerprint(record)
                target=output/Path(scene_path).name
                SceneInputs(record,snapshot,original.rgb).save(target)
                result.append(dict(path=str(target),accepted_scans=stats['accepted_scans'],
                                   future_firings_excluded=stats['future_firings_excluded']))
                print(json.dumps(result[-1]),flush=True)
        finally:db.close()
    return result


def replay_window(run_path,output,*,start_s,end_s,config=None,warmup_s=None,sampling='all'):
    """Rebuild a continuous window at saved scene cutoffs in a single raw pass.

    Saved metadata supplies the exact association, never historical grid arrays.
    Geometry and source-bound people are replayed; RGB is explicitly unavailable.
    """
    from .scene import assemble_scene
    run_path,output=Path(run_path),Path(output)
    if output.exists():raise FileExistsError(output)
    if not np.isfinite([start_s,end_s]).all() or start_s<0 or end_s<=start_s:
        raise ValueError('A finite, increasing relative window is required')
    if sampling not in ('all','recorded'):raise ValueError('Unknown replay scan sampling')
    run,rows,people,binding=_source(run_path)
    metadata_path=run_path/'scene/scenes.jsonl'
    records=[json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    sample_times=[r['as_of'] for r in records] if sampling=='recorded' else None
    base=rows[0]['stamp_s']
    records=[r for r in records if base+start_s<=r['as_of']<=base+end_s]
    if not records or np.any(np.diff([r['as_of'] for r in records])<=0):
        raise ValueError('Window needs strictly ordered saved scene cutoffs')
    if len(records)>1000:raise ValueError('Replay window exceeds the 1000-scene budget')
    for record in records:
        if fingerprint({k:v for k,v in record.items() if k!='input_fingerprint'})!=record['input_fingerprint']:
            raise ValueError('Saved scene metadata changed after binding')
    config=copy.deepcopy(config or load_config())
    retention=_retention(config)
    warmup_s=retention if warmup_s is None else warmup_s
    if not np.isfinite(warmup_s) or warmup_s<retention:
        raise ValueError('Warmup must cover the evidence retention interval')
    binding['scene_records_fingerprint']=file_fingerprint(metadata_path)
    binding['scan_sampling']=sampling
    begin=records[0]['as_of']-warmup_s-.25;end=records[-1]['as_of']
    output.mkdir(parents=True)
    times=np.asarray([r['as_of'] for r in records]);index=[];seen=set()
    with tempfile.TemporaryDirectory(prefix='environment-window-') as scratch:
        database=Path(scratch)/'raw.sqlite';_extract(run_path,[[begin,end]],database)
        db=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for snapshot,row,stats in replay_scans(db,rows,run,config,begin,end,people=people,
                    clock_domain=records[0]['clock_domain'],sample_times=sample_times):
                nearest=int(np.argmin(abs(times-row['stamp_s'])))
                if abs(times[nearest]-row['stamp_s'])>1e-6:continue
                original=records[nearest];seen.add(nearest)
                elapsed=row['stamp_s']-base
                if snapshot is None:
                    index.append(dict(relative_s=elapsed,status='unavailable',reason='localization_invalid_or_pose_gap'))
                    continue
                derivation=_derive(snapshot,original,binding,config,run,begin,stats,row=row)
                derived=assemble_scene(snapshot,clock_domain=original['clock_domain'],
                    people=people.at(row['stamp_s']) if people else None,settings=config['scene'])
                derived.record['derivation']=derivation
                derived.record['scope']='geometry_only_causal_replay'
                derived.record['input_fingerprint']=fingerprint({k:v for k,v in derived.record.items() if k!='input_fingerprint'})
                name='at_%09.3fs'%elapsed;derived.save(output/name)
                index.append(dict(relative_s=elapsed,status='available',scene=name,statistics=stats))
        finally:db.close()
    if len(seen)!=len(records):raise ValueError('Some saved scene cutoffs have no matching raw scan')
    manifest=dict(source=binding,start_s=start_s,end_s=end_s,warmup_s=warmup_s,sampling=sampling,configuration=config,
                  scope='causal_geometry_and_recorded_people_not_live_scheduling',frames=index)
    (output/'window.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return manifest


if __name__=='__main__': extract(*sys.argv[1:])
