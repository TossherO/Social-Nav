"""Inspect recorded shadow people refusals, without models or robot execution.

ROS bag I/O runs in the system interpreter. Numeric diagnostics and rendering
run in the project interpreter; recorded sources are read-only throughout.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def read_assets(output):
    import numpy as np
    import rosbag,genpy
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from perception.sensors.pointcloud import pointcloud_fields
    from perception.sensors.image import rgb_array
    request=json.load(sys.stdin);arrays={};meta=[]
    with rosbag.Bag(request['bag'],'r') as bag:
        for i,sample in enumerate(request['samples']):
            stamp=sample['stamp_s'];cloud=None;image=None;image_stamp=None
            for topic,message,_ in bag.read_messages(topics=['/rslidar_points','/camera/color/image_raw'],
                    start_time=genpy.Time.from_sec(stamp-.5),end_time=genpy.Time.from_sec(stamp+.5)):
                t=message.header.stamp.to_sec()
                if topic=='/rslidar_points' and abs(t-stamp)<1e-6:cloud=message
                if topic.endswith('image_raw') and 0<=stamp-t<=.1 and (image_stamp is None or t>image_stamp):
                    image=message;image_stamp=t
            if cloud is None:raise ValueError('Exact recorded point cloud is missing')
            if cloud.header.frame_id.lstrip('/')!=sample['sensor_frame'].lstrip('/'):raise ValueError('Point cloud frame changed')
            xyz=pointcloud_fields(cloud,('x','y','z'));points=np.column_stack([xyz[k] for k in ('x','y','z')])
            points=points[np.isfinite(points).all(1)]
            arrays['points_'+str(i)]=points.astype(np.float32)
            if image is not None:arrays['rgb_'+str(i)]=rgb_array(image)
            meta.append(dict(cloud_stamp_s=stamp,sensor_frame=cloud.header.frame_id,rgb_stamp_s=image_stamp))
    arrays['metadata']=np.asarray(json.dumps(meta));np.savez_compressed(output,**arrays)


def inspect_people(run,output,samples=5,assets=True,compare_config=None):
    import base64,html,io
    from collections import Counter
    from types import SimpleNamespace
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    from .people import PeopleModel
    from .footprint import VehicleGeometry
    from .motion import motion_uncertainty_margin
    from navigation.safety.guard import stopping_trajectory
    from perception.environment.context import fingerprint
    import yaml
    source=Path(run).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1<=samples<=8:raise ValueError('Choose at most eight representative cases')
    recorded=json.loads((source/'results.json').read_text())
    configuration=source.parent/'comparison.json'
    cfg=recorded.get('navigation_config')
    if cfg is None:
        if not configuration.exists():raise ValueError('Need the recorded run configuration')
        cfg=json.loads(configuration.read_text())['configuration']
    planning,motion=cfg['planning'],cfg['motion']
    comparison=planning
    if compare_config is not None:
        from .inputs import load_config
        comparison=load_config(compare_config)
    safety=cfg.get('safety') or yaml.safe_load((Path(__file__).resolve().parents[1]/'safety/config.yaml').read_text())
    geometry=VehicleGeometry.historical(planning['geometry_reference'])
    people={p['input_fingerprint']:p for p in map(json.loads,(source/'perception/people.jsonl').open())}
    events=list(map(json.loads,(source/'metrics.jsonl').open()))
    first=min(r['stamp_s'] for r in events if r['kind']=='raw_scan')
    metrics=[r for r in events if r['kind']=='shadow']

    records=[];working=[]
    for entry in metrics:
        s=entry['row'];check=s.get('people_check')
        if not check or check.get('source') not in people:continue
        data=people[check['source']];identity=s['feedback']['local_identity']
        # A diagnostic people-only context, not a reconstructed navigation scene.
        context=SimpleNamespace(record=dict(as_of=s['as_of'],frame_id=identity[2],session_id=identity[0],generation=identity[1],
            people=dict(status='valid',data=data)))
        model=PeopleModel(context,comparison['people']);old=PeopleModel(context,planning['people'])
        if model.status!='valid':raise ValueError(model.reason)
        if not model.tracks:continue
        sweep=s.get('stopping_sweep')
        points=(np.asarray(sweep['points']) if sweep else stopping_trajectory(s['feedback'],s['proposed_velocity'],motion,safety,cfg['shadow']['period_s']))
        if 'prediction_start_s' in check:
            anchor=check['prediction_start_s'];anchors=[anchor];anchor_scope='recorded_exact'
        else:
            anchor=s['stamp_s']+s['timing_ms'].get('source_join',0.)/1000
            low,high=s['stamp_s'],entry['receive_ros_s'];anchor=float(np.clip(anchor,low,high))
            anchors=np.unique(np.r_[anchor,np.linspace(low,high,max(2,int(np.ceil((high-low)/.01))+1))])
            anchor_scope='recorded_start_to_receipt_interval_not_exact'
        margin=check.get('spatial_margin_m',planning['safety_margin_m']+motion_uncertainty_margin(s['feedback'],geometry,float(points[-1,0])))
        previous=old.check(points,geometry,margin,start_time=anchor)
        current=model.check(points,geometry,margin,start_time=anchor)
        outcomes=[model.check(points,geometry,margin,start_time=float(t))['valid'] for t in anchors]
        probes=[]
        for scale in (.5,0.):
            alternative=stopping_trajectory(s['feedback'],np.asarray(s['proposed_velocity'])*scale,motion,safety,cfg['shadow']['period_s'])
            extra=motion_uncertainty_margin(s['feedback'],geometry,float(alternative[-1,0]))
            probe=model.check(alternative,geometry,planning['safety_margin_m']+extra,start_time=anchor)
            checks=[model.check(alternative,geometry,planning['safety_margin_m']+extra,start_time=float(t))['valid'] for t in anchors]
            probes.append(dict(scale=scale,people_valid_at_anchor=probe['valid'],people_valid_across_sampled_anchors=all(checks),
                geometry_rechecked=False,scope='people_only_counterfactual_not_navigation_permission',prediction=probe))
        recorded_critical=tuple(check.get('critical_identity',[]))
        known=None if recorded_critical not in model.ids else bool(model.known[model.ids.index(recorded_critical)])
        row=dict(sequence_id=s['sequence_id'],stamp_s=s['stamp_s'],offset_s=s['stamp_s']-first,recorded_critical_velocity_known=known,
            scene_binding_reference=s['scene_binding'],trajectory_binding_reference=s['trajectory_binding'],
            people_input_fingerprint=data['input_fingerprint'],people_record_fingerprint=fingerprint(data),
            anchor_scope=anchor_scope,anchor_s=anchor,anchor_interval_s=[float(min(anchors)),float(max(anchors))],
            recorded_gate_reason=s['reason'],recorded_people_check=check,
            action_probes=probes,recorded_velocity_filter=s.get('velocity_filter'),
            old_prediction=previous,new_prediction=current,
            new_valid_across_sampled_anchor_window=all(outcomes),new_rejected_across_sampled_anchor_window=not any(outcomes),
            anchor_sensitive=any(outcomes) and not all(outcomes),stopping_duration_s=float(points[-1,0]),
            classification=('prediction_unavailable' if 'minimum_nominal_clearance_m' not in current else
                'nominal_path_overlap' if current['minimum_nominal_clearance_m']<=0 else 'deterministic_clear'))
        records.append(row);working.append((entry,data,model,old,points,margin))
    if not records:raise ValueError('No recorded people checks')
    selected=[]
    pools=[lambda r:not r['new_prediction']['valid'] and r['action_probes'][0]['people_valid_across_sampled_anchors'],
        lambda r:r['recorded_gate_reason']=='predicted_people_collision' and r['new_valid_across_sampled_anchor_window'],
        lambda r:r['recorded_critical_velocity_known'] is False,
        lambda r:r['recorded_critical_velocity_known'] is True,
        lambda r:r['classification']=='nominal_path_overlap',lambda r:r['recorded_gate_reason'] is None]
    for choose in pools:
        eligible=[i for i,r in enumerate(records) if i not in selected and choose(r)]
        spaced=[i for i in eligible if all(abs(records[i]['offset_s']-records[j]['offset_s'])>1. for j in selected)]
        eligible=spaced or eligible  # Cover distinct mechanisms even in nearby observations.
        if eligible:selected.append(max(eligible,key=lambda i:records[i]['new_prediction'].get('minimum_clearance_m',-1e9)-records[i]['old_prediction'].get('minimum_clearance_m',-1e9)))
    selected=selected[:samples]
    if len(selected)<samples:
        for i in range(len(records)):
            if all(abs(records[i]['offset_s']-records[j]['offset_s'])>1. for j in selected):selected.append(i)
            if len(selected)==samples:break
    source_run=json.loads((Path(recorded['source_run'])/'run.json').read_text());bag=source_run['bag']
    from perception.mapping_localization.results import file_fingerprint
    if file_fingerprint(bag)!=recorded['bag_fingerprint']:raise ValueError('Source bag fingerprint changed')
    asset_data=None;cards=[]
    with tempfile.TemporaryDirectory(prefix='people-review-') as temporary:
        if assets:
            file=Path(temporary)/'assets.npz'
            request=dict(bag=bag,samples=[dict(stamp_s=working[i][1]['stamp_s'],sensor_frame=working[i][1]['detection_frame_id']) for i in selected])
            subprocess.run(['/usr/bin/python3','-B',str(Path(__file__)),'--read-assets',str(file)],
                input=json.dumps(request),text=True,check=True,timeout=60)
            asset_data=np.load(file,allow_pickle=False)
        for rank,index in enumerate(selected):
            row=records[index];entry,data,model,old,points,margin=working[index];s=entry['row'];anchor=row['anchor_s']
            row['assets']=None if asset_data is None else json.loads(str(asset_data['metadata']))[rank]
            fresh=model.check(points,geometry,margin,start_time=anchor,details=True);detail=fresh['diagnostic']
            ids=[tuple(k) for k in detail['identities']];critical=ids.index(tuple(fresh['critical_identity']))
            t=fresh['critical_relative_time_s'];focus='Minimum envelope clearance'
            recorded_identity=tuple(row['recorded_people_check'].get('critical_identity',[]))
            if compare_config is not None and recorded_identity in ids:
                critical=ids.index(recorded_identity);t=row['recorded_people_check']['critical_relative_time_s']
                focus='Recorded check critical identity'
            elif row['classification']=='nominal_path_overlap':
                i,critical=np.unravel_index(np.argmin(detail['nominal_clearance']),np.shape(detail['nominal_clearance']))
                t=detail['mid_times'][i];focus='Minimum nominal clearance'
            oldcentre,oldrad=old.positions([anchor+t]);newcentre,newrad=model.positions([anchor+t])
            pose=np.asarray(s['feedback']['pose']);c,sn=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,-sn],[sn,c]])
            local=lambda xy:(np.asarray(xy)-pose[:2])@rotation
            fig,axes=plt.subplots(2,2,figsize=(13,8));ax=axes[0,0]
            if asset_data is not None:
                cloud=asset_data['points_'+str(rank)];transform=np.asarray(data['world_from_lidar'])
                world=cloud@transform[:3,:3].T+transform[:3,3];cloudxy=local(world[:,:2])
                keep=(abs(cloudxy[:,0])<16)&(abs(cloudxy[:,1])<12)
                dots=cloudxy[keep][::max(1,int(keep.sum()/9000))]
                ax.scatter(dots[:,0],dots[:,1],s=.8,c='#9ba8b5',alpha=.55,rasterized=True)
            for j,track in enumerate(model.tracks):
                xy=local(track['xy']);ax.add_patch(Circle(xy,track['radius'],fill=False,color='#cb3e42' if j==critical else '#929aa4'))
                ax.text(*xy,str(track['identity'][-1]).split(':')[-1],fontsize=8)
            path=local(points[:,1:3]);ax.plot(path[:,0],path[:,1],color='#bd7d09',lw=2,label='Hypothetical stop path')
            ax.add_patch(Polygon(geometry.footprint,facecolor='#378b58',alpha=.8));ax.set(xlim=(-4,15),ylim=(-10,10),aspect='equal',title='Source scan / tracks and observed wheelchair')
            ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Forward (m)');ax.set_ylabel('Left (m)')
            ax=axes[0,1];ax.plot(path[:,0],path[:,1],color='#bd7d09',lw=2)
            from navigation.kinematics import sample_trajectory
            body,_=sample_trajectory(points,[t]);bodyxy=local(body[0,:2])
            for radius,color,label in [(oldrad[0,critical],'#cb3e42','Recorded model envelope'),(newrad[0,critical],'#286cbb','Compared model envelope')]:
                ax.add_patch(Circle(local(newcentre[0,critical]),radius,fill=False,color=color,lw=2,label=f'{label}: {radius:.2f} m'))
            bodypolygon=geometry.polygon(body[0]);ax.add_patch(Polygon(local(bodypolygon),facecolor='#378b58',alpha=.8))
            ax.scatter(*bodyxy,s=20,c='#378b58');ax.set(xlim=(-4,15),ylim=(-10,10),aspect='equal',title=f'Model at +{t:.2f}s (not future truth)');ax.legend(fontsize=8);ax.grid(alpha=.2)
            ax=axes[1,0];times=np.linspace(0,points[-1,0],70)
            _,a=old.positions(anchor+times);_,b,parts=model.positions(anchor+times,components=True)
            ax.plot(times,a[:,critical],color='#cb3e42',label='Recorded model radius');ax.plot(times,b[:,critical],color='#286cbb',label='Compared model radius')
            for key in parts:ax.plot(times,parts[key][:,critical],ls='--',label=key)
            probe=row['action_probes'][0]
            if probe['people_valid_at_anchor']:
                end=probe['prediction']['prediction_end_s']-anchor
                ax.axvline(end,color='#378b58',ls=':',label='Lower-speed stop time (people-only check)')
            ax.set(xlabel='Relative prediction time (s)',ylabel='Envelope contribution (m)',title=focus+' | identity '+str(model.ids[critical][-1]).split(':')[-1]);ax.legend(fontsize=8);ax.grid(alpha=.2)
            ax=axes[1,1]
            if asset_data is not None and 'rgb_'+str(rank) in asset_data:ax.imshow(asset_data['rgb_'+str(rank)])
            else:ax.text(.5,.5,'RGB unavailable',ha='center')
            ax.axis('off');ax.set_title('Latest past raw RGB; no identity projection or ground truth')
            fig.suptitle(f"{row['offset_s']:.2f}s | {row['classification']} | recorded={row['recorded_gate_reason']} | new people valid={fresh['valid']}",fontsize=12)
            fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',dpi=120);plt.close(fig)
            preview=Path('/tmp')/('people_prediction_case_'+str(rank)+'.png');preview.write_bytes(buf.getvalue())
            encoded=base64.b64encode(buf.getvalue()).decode()
            cards.append(f'<article><h2>{rank+1}. {row["offset_s"]:.2f}s</h2><p>{html.escape(row["classification"])} · 锚点：{row["anchor_scope"]} · 半速人物复核通过：{probe["people_valid_across_sampled_anchors"]}（尚未重查几何）</p><img src="data:image/png;base64,{encoded}"></article>')
            row['selected_for_review']=True
        if asset_data is not None:asset_data.close()
    refused=[r for r in records if r['recorded_gate_reason']=='predicted_people_collision']
    summary=dict(recorded_people_checks=len(records),recorded_people_refusals=len(refused),
        critical_known=sum(r['recorded_critical_velocity_known'] is True for r in refused),
        critical_unknown=sum(r['recorded_critical_velocity_known'] is False for r in refused),
        previously_refused_now_valid_across_sampled_anchors=sum(r['new_valid_across_sampled_anchor_window'] for r in refused),
        anchor_sensitive=sum(r['anchor_sensitive'] for r in refused),classifications=dict(Counter(r['classification'] for r in refused)))
    summary['half_speed_people_valid_across_sampled_anchors']=sum(r['action_probes'][0]['people_valid_across_sampled_anchors'] for r in refused)
    summary['stop_people_valid_across_sampled_anchors']=sum(r['action_probes'][1]['people_valid_across_sampled_anchors'] for r in refused)
    result=dict(scope='recorded_people_prediction_diagnostic_not_full_navigation_replay',source_run=str(source),bag=bag,
        summary=summary,selected_sequences=[records[i]['sequence_id'] for i in selected],records=records,
        configuration=dict(planning=planning,comparison_people=comparison['people'],motion=motion,safety=safety),
        safety_configuration_source='recorded' if cfg.get('safety') else 'current_configuration_not_captured_in_legacy_record',
        executable=False,task_completed=False)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    page='<!doctype html><meta charset="utf-8"><title>人物预测拒绝审阅</title><style>body{font:16px sans-serif;margin:24px auto;max-width:1600px;background:#f3f5f7}article{background:white;padding:20px;margin:24px 0}img{width:100%}p{line-height:1.6}</style>'
    page+='<h1>人物预测拒绝：集中审阅</h1><p>点云／RGB只使用原观测及过去图像；停车路径与未来圆均为模型预测，未被执行，也不是人工真值。历史记录未保存精确检查起点时，以原起点至接收时间窗口采样比较；不能把重新预测通过当作导航可放行。</p>'
    (out/'review.html').write_text(page+''.join(cards));return result['summary']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--read-assets',required=True)
    read_assets(parser.parse_args().read_assets)
