"""Read-only timeline of recorded reference, DWA and independent safety outcomes."""
import argparse
import base64
from bisect import bisect_right
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile


REASONS = {
    'candidate': ('已生成DWA候选', '速度与预测轨迹已经生成；仍需独立安全复核，不代表真实执行。'),
    'no_active_goal': ('没有活动目标', '当前控制周期没有活动导航目标，不能计为搜索失败。结合路线状态判断初始化、取消或暂停。'),
    'start_footprint_not_free': ('参考搜索起点足迹非自由', '车体完整足迹及安全余量未全部落在自由格中，参考搜索受到阻止。具体状态以本周期足迹证据为准；缺失时不推断。'),
    'current_footprint_not_free': ('DWA当前足迹非自由', '当前车体足迹检查未通过，DWA在候选评估前退出。'),
    'waiting_for_background_reference': ('等待后台参考路径', 'DWA没有可用参考。应先看同周期参考原因和最近搜索事件；等待本身不等于搜索已判无路。'),
    'reference_has_no_path': ('参考结果没有可用路径', '后台返回的结果未通过路径检查。旧日志只保存这一汇总原因；搜索内部更具体的拒绝原因可能没有记录。'),
    'reference_timeout': ('参考搜索超时', '后台查询超过运行时预算。'),
    'reference_worker_startup_timeout': ('参考进程启动超时', '后台参考进程未在启动预算内就绪。'),
    'reference_future_or_stale': ('参考时间失效', '参考输入时刻与当前场景不符或已超过允许年龄。'),
    'reference_context_mismatch': ('参考上下文已改变', '目标、定位生命周期或地图关联改变后，旧参考不能继续使用。'),
    'reference_goal_mismatch': ('参考目标不一致', '返回参考与当前活动目标不一致。'),
    'navigation_prior_future_or_stale': ('静态支持关联时间失效', '用于导航静态自由支持的地图关联测量过旧或来自未来；不是单纯没有地图文件。'),
    'geometry_future_or_stale': ('局部地图时间失效', '控制时读取的局部地图不满足测量时效。'),
    'observation_wall_stale': ('观察到达后等待过久', '最新观察在运行时墙钟上已过期。'),
    'no_admissible_velocity': ('DWA没有合格速度', '实际检查的候选均未通过。下方列出几何、人物预测和制动检查的拒绝计数。'),
    'feedback_invalid': ('位姿或运动反馈无效', '控制反馈不满足时间、坐标或有效性要求。'),
    'feedback_outside_motion_model': ('反馈超出运动模型', '根据定位估计的运动状态超出控制器当前模型允许范围。'),
    'planner_unavailable': ('安全层没有可用控制候选', '上游未提供可复核的候选，不表示安全层发现了新的几何障碍。'),
    'stopping_sweep_unknown': ('制动扫掠进入未知', '完整制动过程覆盖了未知空间。'),
    'stopping_sweep_occupied': ('制动扫掠碰到占据', '完整制动过程与占据空间相交。'),
    'people_prediction_horizon_exceeded': ('人物预测时域不足', '待复核的运动或制动时间超出人物预测支持范围。'),
    'active_goal_mismatch': ('安全目标已更新', '提议对应的目标与安全层当前目标不一致。'),
    'scene_binding_invalid': ('安全场景对应关系失效', '提议与安全层持有的观察不匹配。'),
    'goal_changed_during_gate': ('复核期间目标改变', '安全检查结束前目标已变化。'),
    'feedback_expired_during_gate': ('复核期间反馈过期', '安全检查完成时原反馈已超过时效。'),
    'grid_wall_timeout': ('安全地图到达超时', '安全层没有及时收到新的局部地图。'),
    'waiting_for_route_observation': ('路线等待有效观察', '路线调度尚无可用定位或场景。'),
    'waiting_for_route_window': ('尚未进入路线时间段', '该时间没有配置录制路线目标。'),
    'goal': ('已提交路线目标', '录制路线调度提交了当前开发回放目标。'),
    'route_goal_changed': ('路线更新目标', '目标修订改变，旧参考被撤回。'),
    'cancel': ('目标取消', '当前目标被取消。'),
    'record_missing': ('日志没有记录', '不能从缺失记录推断成功或失败。'),
}


def read_rows(path):
    rows=[]
    with Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            if line.strip():
                try:rows.append(json.loads(line))
                except ValueError as error:raise ValueError(f'{path}:{number}: invalid JSON') from error
    return rows


def previous(rows,stamps,stamp):
    index=bisect_right(stamps,stamp)-1
    return None if index<0 else rows[index]


def reference_records(rows):
    # Runtime writes each status before draining the events from that same tick.
    # Event.as_of is the original scene time, NOT query completion time.
    statuses={};attempts=[];anchor=None
    for row in rows:
        if row.get('event')=='status':anchor=row['as_of'];statuses[anchor]=row
        elif row.get('event') in ('submitted','reference_ready','discarded','timeout'):
            attempts.append(dict(row,recorded_tick_s=anchor))
    return statuses,attempts


def compact_safety(row,base):
    if row is None:return None
    clearance=(row.get('people_check') or {}).get('minimum_clearance_m')
    return dict(t=row['stamp_s']-base,state=row['state'],reason=row.get('reason'),
        trajectory_binding=row.get('trajectory_binding'),sequence=row.get('sequence_id'),
        minimum_clearance=clearance if clearance is not None and math.isfinite(clearance) else None,
        check=(row.get('check') or {}).get('status'))


def classify(control,reference):
    proposal=control.get('proposal') or {}
    if control.get('request') is None:return 'idle','no_active_goal'
    if proposal.get('status')=='candidate':return 'generated','candidate'
    reason=control.get('reason') or proposal.get('reason') or 'record_missing'
    if reason=='record_missing':return 'unrecorded',reason
    if reason=='waiting_for_background_reference':
        return 'reference',((reference or {}).get('reason') or reason)
    if reason in ('no_admissible_velocity','current_footprint_not_free'):
        return 'dwa',reason
    return 'input',reason


def assemble(run,controls,references,safety,routes,positions):
    base=run['route_replay']['source']['source_start_s']
    controls=sorted(controls,key=lambda x:x['stamp_s'])
    safety=sorted(safety,key=lambda x:x['stamp_s']);safe_times=[x['stamp_s'] for x in safety]
    routes=sorted(routes,key=lambda x:x['source_offset_s']);route_times=[x['source_offset_s']+base for x in routes]
    positions=sorted(positions,key=lambda x:x['stamp_s']);pose_times=[x['stamp_s'] for x in positions]
    for i,row in enumerate(positions):row['index']=i
    statuses,attempts=reference_records(references)
    by_time={x['stamp_s']:x for x in controls};jobs={}
    for event in attempts:
        owner=by_time.get(event['recorded_tick_s'],{}).get('request') or {}
        if event['event']=='submitted':jobs[event['job_id']]=owner
        event['goal_revision']=(jobs.get(event.get('job_id')) or owner).get('goal_revision')
        event['t']=None if event['recorded_tick_s'] is None else event['recorded_tick_s']-base
        event['source_t']=None if 'as_of' not in event else event['as_of']-base
    ordered=[(i,x) for i,x in enumerate(attempts) if x['t'] is not None]
    ordered.sort(key=lambda x:x[1]['t']);cursor=0;latest={}
    safe_by_binding={}
    for row in safety:
        binding=row.get('trajectory_binding')
        if binding:safe_by_binding.setdefault(binding,[]).append(row)
    frames=[]
    for control in controls:
        stamp=control['stamp_s'];t=stamp-base;request=control.get('request') or {};p=control.get('proposal') or {}
        reference=statuses.get(stamp)
        while cursor<len(ordered) and ordered[cursor][1]['t']<=t:
            index,event=ordered[cursor]
            if event['goal_revision'] is not None:latest[event['goal_revision']]=index
            cursor+=1
        active=bool(request);stage,reason=classify(control,reference)
        observed=previous(positions,pose_times,stamp);feedback=control.get('feedback') or {}
        last=previous(safety,safe_times,stamp);route=previous(routes,route_times,stamp)
        matching=next((row for row in safe_by_binding.get(p.get('trajectory_binding'),[]) if row['stamp_s']>=stamp),None)
        geometry_stamp=(p.get('navigation_geometry') or {}).get('as_of')
        frames.append(dict(t=t,sequence=control['sequence_id'],active=active,stage=stage,reason=reason,
            reference_available=(reference.get('reason') is None) if reference is not None else None,
            reference_reason=None if reference is None else reference.get('reason'),
            reference_blocked=None if reference is None else reference.get('blocked'),
            start_evidence=None if reference is None else reference.get('start_evidence'),
            reference_waiting=None if reference is None else reference.get('waiting'),sampling=p.get('sampling'),
            generated=p.get('status')=='candidate',dwa_reason=control.get('reason') or p.get('reason'),
            candidates=p.get('candidate_count'),checked=p.get('certification_attempts'),rejections=p.get('rejections',{}),
            velocity=p.get('velocity') if p.get('status')=='candidate' else None,compute_ms=control.get('compute_ms'),
            reference_age=None if p.get('reference_as_of') is None else stamp-p['reference_as_of'],
            geometry_age=None if geometry_stamp is None else stamp-geometry_stamp,
            feedback_age=None if feedback.get('stamp_s') is None else stamp-feedback['stamp_s'],
            scene_binding=control.get('scene_binding'),trajectory_binding=p.get('trajectory_binding'),
            pose=feedback.get('pose') if feedback.get('status')=='valid' else (observed or {}).get('pose'),
            pose_stamp=(feedback.get('stamp_s') if feedback.get('status')=='valid' else (observed or {}).get('stamp_s')),
            pose_index=None if observed is None else observed['index'],goal=request.get('goal'),
            goal_revision=request.get('goal_revision'),mode=request.get('mode'),segment=request.get('route_segment_id'),
            route_status=None if route is None else route.get('status'),
            safety=compact_safety(last,base),matched_safety=compact_safety(matching,base),
            attempt=latest.get(request.get('goal_revision')) if active else None))
    summary=dict(control_ticks=len(frames),active_ticks=sum(f['active'] for f in frames),
        reference_available_ticks=sum(f['reference_available'] is True for f in frames),
        generated_ticks=sum(f['generated'] for f in frames),
        no_goal_ticks=sum(not f['active'] for f in frames),
        reference_events=dict(Counter(x.get('event') for x in references)),
        stage_counts=dict(Counter(f['stage'] for f in frames)),
        cause_counts=dict(Counter(f['reason'] for f in frames if f['active'] and not f['generated'] and f['stage']!='unrecorded')),
        dwa_reason_counts=dict(Counter(f['dwa_reason'] for f in frames if f['dwa_reason'])),
        safety_states=dict(Counter(x['state'] for x in safety)),
        rejected_candidates=dict(sum((Counter(f['rejections']) for f in frames),Counter())))
    segments=[]
    for i,f in enumerate(frames):
        key=(f['stage'],f['reason'],f['segment'])
        gap=bool(i and f['t']-frames[i-1]['t']>.5)
        if not segments or segments[-1]['key']!=key or gap:
            segments.append(dict(key=key,start=i,end=i,t0=f['t'],t1=f['t'],ticks=1))
        else:segments[-1].update(end=i,t1=f['t'],ticks=segments[-1]['ticks']+1)
    return dict(frames=frames,attempts=attempts,safety=[compact_safety(x,base) for x in safety],
        positions=positions,segments=segments,summary=summary,base=base)


def localization_positions(rows):
    from .footprint import VehicleGeometry
    from .inputs import load_config
    from perception.mapping_localization.context import pose_matrix
    geometry=VehicleGeometry.historical(load_config()['geometry_reference'])
    result=[]
    for i,row in enumerate(rows):
        pose=None;mapped=None
        if row.get('health')==0:
            pose=geometry.pose(pose_matrix(row['position'],row['quaternion_xyzw'])).tolist()
            if row.get('has_map_pose'):
                mapped=geometry.pose(pose_matrix(row['map_position'],row['map_quaternion_xyzw'])).tolist()
        result.append(dict(index=i,stamp_s=row['stamp_s'],pose=pose,map_pose=mapped,
            identity=[row['session_id'],row['generation'],row['frame_id']],association=row.get('status')))
    return result,geometry.footprint.tolist()


def extract_rgb(request_path,output):
    """System ROS interpreter; latest past RGB for each requested sampling time."""
    import genpy
    import rosbag
    from PIL import Image
    from perception.sensors.image import rgb_array
    from perception.mapping_localization.results import file_fingerprint
    request=json.loads(Path(request_path).read_text());samples=[];last=None
    if file_fingerprint(Path(request['bag']))!=request['bag_fingerprint']:
        raise ValueError('RGB bag differs from the recorded navigation input')
    target=request['start'];step=1/request['hz'];end=request['end'];previous_stamp=None
    def save(at):
        if last is None or not 0<=at-last.header.stamp.to_sec()<=.2:return
        image=Image.fromarray(rgb_array(last));image.thumbnail((480,270))
        stream=io.BytesIO();image.save(stream,format='JPEG',quality=65)
        samples.append(dict(t=last.header.stamp.to_sec()-request['base'],image=base64.b64encode(stream.getvalue()).decode()))
    with rosbag.Bag(request['bag'],'r') as bag:
        duration=bag.get_end_time()-bag.get_start_time()
        for _,message,_ in bag.read_messages(topics=['/camera/color/image_raw'],
                start_time=genpy.Time.from_sec(target-.25),end_time=genpy.Time.from_sec(end+.25)):
            stamp=message.header.stamp.to_sec()
            if previous_stamp is not None and stamp<previous_stamp:raise ValueError('RGB header time reversed')
            previous_stamp=stamp
            while target<stamp and target<=end:save(target);target+=step
            last=message
            if target>end:break
        while target<=end:save(target);target+=step
    Path(output).write_text(json.dumps(dict(images=samples,bag_duration_s=duration,source=request),separators=(',',':')))
    print(json.dumps(dict(rgb_samples=len(samples),bag_duration_s=duration)))


def build(run_path,output,*,rgb=None,static_map=None):
    import numpy as np
    from PIL import Image
    source=Path(run_path).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Choose a new review output directory: '+str(out))
    names=['run.json','summary.json','control.jsonl','reference.jsonl','safety.jsonl','route.jsonl','localization.jsonl']
    if (source/'planning').exists():
        names+=sorted(str(p.relative_to(source)) for p in (source/'planning').rglob('*') if p.is_file())
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    run=json.loads((source/'run.json').read_text());recorded=json.loads((source/'summary.json').read_text())
    if not run.get('success') or not recorded.get('success'):raise ValueError('Use a complete successful source run')
    positions,footprint=localization_positions(read_rows(source/'localization.jsonl'))
    data=assemble(run,read_rows(source/'control.jsonl'),read_rows(source/'reference.jsonl'),
        read_rows(source/'safety.jsonl'),read_rows(source/'route.jsonl'),positions)
    window_end=run['start']+run['duration'] if run.get('duration') is not None else positions[-1]['stamp_s']-data['base']
    images=json.loads(Path(rgb).read_text()) if rgb else dict(images=[],bag_duration_s=None)
    if rgb:
        provenance=images.get('source',{})
        if (provenance.get('bag_fingerprint')!=run['playback']['bag_fingerprint'] or
                provenance.get('base')!=data['base']):
            raise ValueError('RGB observations do not match the recorded bag/time origin')
        times=[v['t'] for v in images['images']]
        if times!=sorted(times):raise ValueError('RGB observation times are not ordered')
    data.update(images=images['images'],bag_duration_s=images['bag_duration_s'],footprint=footprint,
        reasons=REASONS,source=str(source),bag=Path(run['bag']).name,source_hashes=hashes,
        start=run['start'],window_end=window_end,
        end=max(window_end,data['frames'][-1]['t'],max((x['t'] for x in data['safety']),default=window_end)),floor=run['route_replay']['floor_id'],
        bag_fingerprint=run['playback']['bag_fingerprint'],static_map=None)
    if static_map:
        from perception.environment.storage import StaticEnvironmentMap
        asset=StaticEnvironmentMap(static_map);cells=asset.snapshot.layers['occupancy']
        requested={r.get('request',{}).get('map_snapshot_id') for r in read_rows(source/'control.jsonl') if r.get('request')}
        if asset.manifest['snapshot_id'] not in requested:raise ValueError('Static map is not the recorded goal asset')
        palette=np.array([[226,222,211],[249,250,247],[62,73,82]],np.uint8)
        pixels=palette[np.where(cells==0,1,np.where(cells==100,2,0))]
        stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
        data['static_map']=dict(image=base64.b64encode(stream.getvalue()).decode(),
            resolution=asset.snapshot.metadata['resolution'],origin=asset.snapshot.metadata['origin_xy'],
            shape=list(cells.shape),source=str(Path(static_map).resolve()),manifest=asset.manifest)
    if (source/'planning/cycles.jsonl').exists():
        attach_planning(data,source)
    # Each status keeps its own source time. HTML interpolation is only a cursor.
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    template=Path(__file__).with_suffix('.html').read_text()
    out.mkdir(parents=True,exist_ok=False)
    (out/'review.html').write_text(template.replace('__TIMELINE_DATA__',encoded))
    (out/'summary.json').write_text(json.dumps(data['summary'],ensure_ascii=False,indent=2)+'\n')
    manifest={k:data[k] for k in ('source','source_hashes','bag','bag_fingerprint','start','window_end','end','floor','bag_duration_s')}
    manifest.update(rgb_samples=len(data['images']),control_ticks=len(data['frames']),
        recorded_navigation_maps=len(data.get('navigation_maps',{})),
        scope='Recorded outcomes only; maps and curves shown only when archived with matching identity')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if hashes!={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}:
        raise RuntimeError('Source files changed during rendering')
    return data['summary']


def attach_planning(data,source):
    """Only embed actual recorded maps/results, with exact cycle and scene matches."""
    import numpy as np
    from PIL import Image
    from perception.environment.scene import SceneInputs
    from .continuity import content_valid
    from .footprint import VehicleGeometry
    from .planner import se2
    from perception.environment.context import fingerprint
    directory=source/'planning';maps={};references={}
    configuration=json.loads((directory/'config.json').read_text())['geometry']
    geometry=VehicleGeometry(configuration['footprint'],configuration['base_from_lidar'],source=configuration['source'])
    controls={r['sequence_id']:r for r in read_rows(source/'control.jsonl')}
    frames={f['sequence']:f for f in data['frames']}
    for row in read_rows(directory/'cycles.jsonl'):
        control=controls[row['sequence_id']];f=frames[row['sequence_id']]
        if row['packet_binding']!=control['packet_binding'] or row['scene_binding']!=control.get('scene_binding'):
            raise ValueError('Planning evidence does not match the control cycle')
        f['planning_timing']=row['timing'];f['reference_record']=row['reference_binding']
        trajectory=row.get('trajectory')
        if trajectory is not None:
            if not content_valid(trajectory) or trajectory['result_binding']!=control['proposal'].get('trajectory_binding'):
                raise ValueError('Recorded DWA curve differs from its proposal')
            f['trajectory_points']=trajectory['trajectory']['points']
        binding=row['reference_binding']
        if binding and binding not in references:
            ref=json.loads((directory/'references'/(binding+'.json')).read_text())
            if fingerprint({k:v for k,v in ref.items() if k!='binding'})!=binding or not content_valid(ref['result']):
                raise ValueError('Recorded reference changed')
            references[binding]=dict(path=ref['result']['path'],frame_id=ref['result']['frame_id'],as_of=ref['as_of'])
        name=row['scene_binding']
        if name and name not in maps and (directory/'scenes'/name/'scene.json').exists():
            scene=SceneInputs.load(directory/'scenes'/name)
            if scene.record['input_fingerprint']!=name:raise ValueError('Recorded navigation map changed')
            from perception.environment.navigation import navigation_geometry
            grid=scene.geometry;cells=navigation_geometry(scene).layers['occupancy']
            palette=np.array([[200,159,80],[215,221,227],[240,88,72]],np.uint8)
            pixels=palette[np.where(cells==0,1,np.where(cells==100,2,0))]
            stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
            maps[name]=dict(image=base64.b64encode(stream.getvalue()).decode(),shape=list(cells.shape),
                resolution=grid.metadata['resolution'],origin=grid.metadata['origin_xy'],as_of=scene.record['as_of'],
                frame_id=grid.metadata['frame_id'],odom_from_map=None)
            if scene.record['prior'].get('valid'):
                world=np.asarray(scene.record['localization']['world_from_lidar'])
                pose=geometry.pose(world);mapped=geometry.pose(np.asarray(scene.record['prior']['map_from_local'])@world)
                maps[name]['odom_from_map']=(se2(pose)@np.linalg.inv(se2(mapped))).tolist()
    data['navigation_maps']=maps;data['recorded_references']=references
    data['planning_record']=json.loads((source/'summary.json').read_text()).get('planning_record')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run');parser.add_argument('--output',required=True)
    images=parser.add_mutually_exclusive_group()
    images.add_argument('--rgb');images.add_argument('--with-rgb',action='store_true')
    parser.add_argument('--static-map');parser.add_argument('--extract-rgb')
    args=parser.parse_args()
    if args.extract_rgb:extract_rgb(args.extract_rgb,args.output)
    elif args.run:
        if Path(args.output).exists():raise FileExistsError('Choose a new review output directory')
        with tempfile.TemporaryDirectory(prefix='social-nav-planning-review-') as temporary:
            if args.with_rgb:
                run=json.loads((Path(args.run)/'run.json').read_text());base=run['route_replay']['source']['source_start_s']
                end=base+run['start']+run['duration'] if run.get('duration') is not None else read_rows(Path(args.run)/'localization.jsonl')[-1]['stamp_s']
                plan=Path(temporary)/'request.json';args.rgb=str(Path(temporary)/'rgb.json')
                plan.write_text(json.dumps(dict(bag=run['bag'],bag_fingerprint=run['playback']['bag_fingerprint'],
                    base=base,start=base+run['start'],end=end,hz=2)))
                subprocess.run(['/usr/bin/python3','-B','-m','navigation.planning.timeline',
                    '--extract-rgb',str(plan),'--output',args.rgb],check=True)
            print(json.dumps(build(args.run,args.output,rgb=args.rgb,static_map=args.static_map),ensure_ascii=False))
    else:parser.error('--run or --extract-rgb is required')


if __name__=='__main__':main()
