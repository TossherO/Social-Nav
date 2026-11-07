"""Fixed-step closed-loop development simulation; no bag views or chassis output."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from perception.environment.context import fingerprint
from perception.environment.layers import GridSnapshot,load_config as environment_config
from perception.environment.scene import assemble_scene
from navigation.kinematics import advance,velocity_step,turning_residual
from navigation.planning.inputs import load_config
from navigation.planning.motion import load_motion_config
from navigation.planning.footprint import VehicleGeometry,swept_check
from navigation.planning.motion import interpolate_se2
from navigation.planning.continuity import reference_envelope
from navigation.control.runtime import DwaNavigationControl


class PlanarWorld:
    """Synthetic finite 2D sensor: current occupancy within a radius, unknown outside."""
    resolution=.1
    def __init__(self,kind):
        self.kind=kind;self.shape=(100,140);self.origin=np.array([0.,0.])
        y,x=np.indices(self.shape);self.xy=np.stack([x+.5,y+.5],axis=-1)*self.resolution
        self.static=np.zeros(self.shape,np.int8)
        self.observed_free=np.zeros(self.shape,bool)
        self.static[:2]=self.static[-2:]=100;self.static[:,:2]=self.static[:,-2:]=100
        if kind in ('turn','turn_stop'):self.static[15:45,45:55]=100
        if kind in ('dynamic','infeasible'):
            self.static[10:12,:]=100;self.static[28:30,:]=100
        if kind=='narrow':
            self.static[:]=100
            self.static[18:42,10:67]=0;self.static[18:90,43:67]=0

    def occupancy(self,t):
        cells=self.static.copy()
        actor=self.actor_position(t)
        if actor is not None:cells[np.linalg.norm(self.xy-actor,axis=2)<.35]=100
        return cells

    def actor_position(self,t):
        if self.kind=='crossing':return [5.,5.-.6*t] if t<8. else None
        if self.kind=='follow':return [5.+min(t,12.)*.25,2.]
        if self.kind!='dynamic':return None
        # Current world evolution only; planner never sees future positions.
        y=5.0 if t<2 else max(2.,5.-(t-2)*3.) if t<3 else 2. if t<12 else min(5.,2.+(t-12)*3.)
        return [6.,y]

    def scene(self,t,pose,geometry):
        cells=self.occupancy(t);outside=np.linalg.norm(self.xy-pose[:2],axis=2)>8.
        cells[outside]=-1
        self.observed_free|=cells==0
        c,s=np.cos(pose[2]),np.sin(pose[2]);transform=np.eye(4)
        transform[:2,:2]=[[c,-s],[s,c]];transform[:2,3]=pose[:2]
        transform=transform@geometry.base_from_lidar
        m=dict(shape=list(self.shape),resolution=self.resolution,origin_xy=self.origin.tolist(),frame_id='sim_odom',
            snapshot_id='synthetic:'+self.kind+':'+str(t),stamp_s=10.+t,session_id=self.kind,generation=0,
            geometry_status='OBSERVED',world_from_lidar=transform.tolist(),sensor_frame='rslidar',
            localization_source='synthetic_differential_plant',association=dict(valid=False),
            source='synthetic_2d_radius_sensor_no_occlusion_model_not_bag_replay')
        people=None;actor=self.actor_position(t)
        if self.kind in ('crossing','follow'):
            tracks=[]
            if actor is not None:
                # Velocity comes only from past/current synthetic measurements.
                previous=self.actor_position(max(0.,t-.1));velocity=None if t<.2 else ((np.asarray(actor)-previous)/.1).tolist()+[0.]
                tracks=[dict(track_id=self.kind+'_person',box=actor+[0.,.5,.5,1.7,0.],confirmed=t>=.2,
                    velocity_xyz_mps=velocity,last_observed_stamp_s=10.+t,
                    covariance=np.diag([.02**2]*3+[0.]*4+[.03**2]*3).ravel().tolist())]
            people=dict(observation_id='synthetic_people:'+str(t),stamp_s=10.+t,clock_domain='simulation',health='valid',
                frame_id='sim_odom',session_id='synthetic_tracker',generation=0,sequence_id=self.kind,
                pose_stamp_s=10.+t,world_from_lidar=transform.tolist(),input_fingerprint=fingerprint(tracks),
                localization_binding=dict(session_id=self.kind,generation=0,frame_id='sim_odom'),tracks=tracks,
                covariance_order=['x','y','z','dx','dy','dz','yaw','vx','vy','vz'],velocity_semantics='world_motion',detections=[],detection_frame_id='rslidar')
        layers=dict(occupancy=cells)
        if people is not None:
            # Synthetic sensor class labels, never a future free-space query.
            # Only cells observed free in this feedback run can supply background.
            layers['person_background_free']=self.observed_free&(cells==100)&(self.static==0)
            m['dynamic_filter']=dict(status='valid',input_fingerprint=people['input_fingerprint'])
            m['dynamic_navigation']=dict(policy='synthetic_person_on_past_observed_free')
        return assemble_scene(GridSnapshot(m,layers),clock_domain='simulation',settings=environment_config()['scene'],people=people)


def run_case(kind,planning,motion,geometry,*,max_s=75.,progress=False,global_period_s=None,plant_delay_s=.1,plant_acceleration_scale=1.,feedback_noise_m=0.,feedback_noise_rad=0.):
    planning=copy.deepcopy(planning)
    planning['motion_constraints']={k:motion[k] for k in ('min_turning_radius_m','allow_reverse','allow_in_place_rotation')}
    global_period_s=3. if global_period_s is None else global_period_s
    world=PlanarWorld(kind);pose=np.array([2.,2.,0.]);velocity=np.zeros(2);dt=.1
    goal=[8.,6.,np.pi/2] if kind in ('turn','turn_stop') else [8.,2.,0.]
    if kind=='heading':pose=np.array([2.,4.,np.pi/2]);goal=[8.,4.,0.]
    if kind=='rotation':goal=[2.,2.,np.pi/2]
    if kind=='narrow':pose=np.array([2.,3.,0.]);goal=[5.5,7.,np.pi/2]
    if kind=='infeasible':goal=[4.,2.,np.pi/2];max_s=min(max_s,3.)
    request=dict(request_id=kind,task_id=kind,goal_revision=1,mode='local',frame_id='sim_odom',target_reference='base_link',
        goal=goal,goal_behavior='stop')
    from navigation.planning.planner import Planner
    from navigation.planning.reference import planning_config
    planning=planning_config(planning)
    planner=Planner(planning,geometry,require_actual_start=True)
    policy=None
    if kind=='follow':
        from navigation.tasks import TaskPolicy
        policy=TaskPolicy(kind,'follow',planning,geometry)
    control=DwaNavigationControl(planning,motion,geometry,simulation=True)
    tracker,safety=control.controller,control.safety
    goal_settings=tracker.goal_settings
    if goal_settings['goal_reaction_s']+1e-8<safety.settings['reaction_s']+safety.settings['actuator_delay_s']:
        raise ValueError('Goal approach delay must cover the assumed watchdog and actuator delay')
    reference=None;last_reference=-np.inf;scene=None;history=[];references=[];reasons=Counter();compute=[];control_compute=[];policy_compute=[]
    delay_ticks=int(np.ceil(plant_delay_s/dt));queue=[np.zeros(2) for _ in range(delay_ticks)];cancelled=False;resume=False;follow_held=False
    plant_motion=dict(motion)
    for key in ('max_acceleration_mps2','max_angular_acceleration_rps2'):plant_motion[key]*=plant_acceleration_scale
    rng=np.random.default_rng(7)
    for index in range(int(max_s/dt)):
        t=round(index*dt,8);now=10.+t
        dropout=kind=='faults' and 20.<=t<24.5
        if not dropout:scene=world.scene(t,pose,geometry)
        feedback=dict(status='valid',source='synthetic_differential_drive_plant',pose=pose.tolist(),velocity=velocity.tolist(),
            stamp_s=now,local_identity=[kind,0,'sim_odom'],clock_domain='simulation')
        if feedback_noise_m or feedback_noise_rad:
            scale=np.array([feedback_noise_m,feedback_noise_m,feedback_noise_rad])
            feedback['pose']=(pose+rng.uniform(-scale,scale)).tolist()
            feedback['source']='synthetic_plant_with_bounded_pose_noise'
        if kind=='faults' and 2<=t<6:cancelled=True;tracker.clear()
        elif kind=='faults' and t>=6 and not resume:
            request=dict(request,goal_revision=2,request_id=kind+':resume');resume=True;cancelled=False;reference=None
        manual=(kind=='faults' and 8<=t<12) or (kind=='turn_stop' and 6<=t<10)
        emergency=kind=='faults' and 14<=t<18
        policy_active=True;policy_began=time.perf_counter()
        if policy is not None:
            # Explicit synthetic semantic responses test navigation policy, not
            # VLM recognition. No future actor positions enter the policy.
            if index%5==0:
                ticket=policy.observe(scene,pid_to_track={0:['synthetic_tracker',0,'follow_person']})
                state_name=('visible' if not 6<=t<10 else
                    'occluded' if policy.last_visual is not None and now-policy.last_visual<=policy.hold_s else 'lost')
                response=dict(id=-1 if state_name=='lost' else 0,state=state_name)
                packet=policy.respond(ticket,response,scene=scene,now=now)
            else:packet=policy.update(scene,now=now)
            policy_active=packet['operation']=='plan'
            if policy_active:
                from navigation.planning.continuity import goal_binding
                updated=packet['request']
                if goal_binding(updated)!=goal_binding(request):reference=None
                request=updated
        policy_compute.append((time.perf_counter()-policy_began)*1000)
        plan=None;began=time.perf_counter()
        if policy_active and not cancelled and not dropout and not manual and not emergency:
            from navigation.planning.continuity import context_binding
            if (reference is None or t-last_reference>=global_period_s-1e-8 or
                    reference['context_binding']!=context_binding(request,scene,None)):
                r=planner.query(request,scene=scene,now=now,clock_domain='simulation')
                reference=reference_envelope(r,request,scene,None);last_reference=t
                references.append(dict(t=t,result=r))
            plan=reference
        compute.append((time.perf_counter()-began)*1000)
        control_began=time.perf_counter()
        packet=control.step(request,scene,feedback,now=now,clock_domain='simulation',dt=dt,result=plan,
            active=policy_active and not (cancelled or dropout),manual_override=manual,emergency_stop=emergency)
        proposal,command=packet['proposal'],packet['command']
        plan=packet['result']
        control_compute.append((time.perf_counter()-control_began)*1000)
        reasons[command['reason'] or 'allowed']+=1
        # One real simulation tick of actuator delay and an independently stored plant state.
        queue.append(np.array(command['velocity']));desired=queue.pop(0)
        after=velocity_step(velocity,desired,plant_motion,dt)
        previous=pose.copy();pose=advance(pose,(velocity+after)/2,dt);velocity=after
        truth=GridSnapshot(scene.geometry.metadata,dict(occupancy=np.maximum(world.occupancy(t),world.occupancy(t+dt))))
        dense,error=interpolate_se2(np.array([previous,pose]),world.resolution,2000)
        collision=swept_check(truth,geometry,dense,error,2000,prefix=False)['status']!='free'
        outcome=packet['goal']
        if kind=='follow' and t>=10 and outcome.get('completion_criteria_sustained'):
            observed=scene.record['people']['data']['tracks'][0]['velocity_xyz_mps']
            follow_held=bool(observed is not None and np.linalg.norm(observed[:2])<.03)
        history.append(dict(t=t+dt,pose=pose.tolist(),velocity=velocity.tolist(),command=command['velocity'],
            safety=command['reason'] or 'allowed',plan=None if plan is None else dict(status=plan['status'],reason=plan['reason'],
                progress=(plan.get('trajectory') or {}).get('route_progress',{}).get('progress_type')),
            continuation=bool(proposal and proposal.get('continuation')),dropout=dropout,cancelled=cancelled,
            emergency=emergency,manual=manual,policy_active=policy_active,collision=collision,goal=outcome))
        if progress and index and index%100==0:
            print(json.dumps(dict(case=kind,simulation_time_s=t,pose=pose.tolist(),
                last_plan=history[-1]['plan'],safety=command['reason'] or 'allowed')),flush=True)
        if collision or outcome['task_completed'] or follow_held:break
    stop_checks={}
    if kind in ('faults','turn_stop','follow'):
        checks=([('cancel',5.9),('manual_override',11.9),('emergency_stop',17.9),('input_dropout',24.4)]
            if kind=='faults' else [('follow_visual_loss',9.9)] if kind=='follow' else [('turning_manual_stop',9.9)])
        for name,stamp in checks:
            sample=min(history,key=lambda h:abs(h['t']-stamp))
            stop_checks[name]=dict(stopped=bool(abs(sample['velocity'][0])<=goal_settings['stopped_linear_mps'] and
                abs(sample['velocity'][1])<=goal_settings['stopped_angular_rps']),stamp_s=sample['t'],velocity=sample['velocity'])
    speeds=np.array([h['velocity'] for h in history]);poses=np.array([h['pose'] for h in history])
    moving=abs(speeds[:,1])>1e-6
    near_goal=np.linalg.norm(poses[:,:2]-np.asarray(goal[:2]),axis=1)<1.
    turn=np.abs(np.diff(np.r_[0.,poses[:,2]]))
    safe_refusal=bool(kind=='infeasible' and not any(h['collision'] for h in history) and
        np.max(abs(speeds))<1e-9 and not history[-1]['goal']['task_completed'])
    return dict(case=kind,scope='offline_fixed_step_synthetic_sensor_and_delayed_differential_plant',
        expected='remain_stopped' if kind=='infeasible' else 'hold_follow_distance_task_stays_active' if kind=='follow' else 'reach',safe_refusal=safe_refusal,
        scenario_passed=bool(safe_refusal or follow_held or history[-1]['goal']['task_completed']),follow_distance_held=follow_held,
        motion_metrics=dict(turning_constraint_violations=int(np.count_nonzero(turning_residual(speeds,motion)<-1e-7)),
            stationary_spin_steps=int(np.count_nonzero((abs(speeds[:,0])<1e-5)&(abs(speeds[:,1])>1e-5))),
            minimum_actual_turning_radius_m=float(np.min(abs(speeds[moving,0]/speeds[moving,1]))) if moving.any() else None,
            near_goal_angular_travel_rad=float(turn[near_goal].sum())),
        completed=history[-1]['goal']['task_completed'],collision=history[-1]['collision'],simulated_s=history[-1]['t'],
        final_pose=history[-1]['pose'],final_velocity=history[-1]['velocity'],request=request,history=history,references=references,stop_checks=stop_checks,
        tracking_config=goal_settings,dwa_config=tracker.settings,safety_config=safety.settings,dt_s=dt,actuator_delay_s=delay_ticks*dt,
        requested_actuator_delay_s=plant_delay_s,plant_acceleration_scale=plant_acceleration_scale,
        feedback_noise_m=feedback_noise_m,feedback_noise_rad=feedback_noise_rad,
        global_period_s=global_period_s,
        safety_counts=dict(reasons),compute_ms=dict(p50=float(np.percentile(compute,50)),p95=float(np.percentile(compute,95))),
        control_ms=dict(p50=float(np.percentile(control_compute,50)),p95=float(np.percentile(control_compute,95))),
        navigation_tick_ms=dict(p50=float(np.percentile(np.array(compute)+control_compute+policy_compute,50)),
            p95=float(np.percentile(np.array(compute)+control_compute+policy_compute,95))),
        compute_scope='planning_only_with_periodic_global; navigation_tick_adds_policy_tracking_safety_excludes_synthetic_sensor_and_truth_checks',
        wall_timing_is_not_simulated_sensor_latency=True,geometry_assumed=True,motion_parameters_assumed=True)


def write_review(data,out,geometry):
    """One self-contained replay and one exportable figure; no model reruns."""
    import base64,io
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    columns=min(4,len(data['cases']));groups=int(np.ceil(len(data['cases'])/columns))
    fig,axes=plt.subplots(2*groups,columns,figsize=(4*columns,8*groups),squeeze=False,constrained_layout=True)
    payload=[]
    for index,case in enumerate(data['cases']):
        world=PlanarWorld(case['case']);history=case['history'];poses=np.array([h['pose'] for h in history]);times=np.array([h['t'] for h in history])
        row=2*(index//columns);column=index%columns
        ax=axes[row,column];ax.imshow(world.static,origin='lower',extent=[0,14,0,10],cmap=ListedColormap(['white','#334b5a']),vmin=0,vmax=100)
        ax.plot(poses[:,0],poses[:,1],color='#168147',lw=2);ax.scatter(*poses[0,:2],c='#168147',marker='o');ax.scatter(*case['request']['goal'][:2],c='#d45c25',marker='*',s=100)
        goal=case['request']['goal']
        if len(goal)==3:ax.arrow(*goal[:2],.6*np.cos(goal[2]),.6*np.sin(goal[2]),color='#d45c25',width=.025)
        for j in np.linspace(0,len(poses)-1,6).astype(int):
            p=geometry.polygon(poses[j]);ax.plot(*np.vstack([p,p[0]]).T,color='#2178b4',alpha=.5,lw=.8)
        if case['case']=='dynamic':
            ax.add_patch(plt.Circle((6,2),.35,color='#d64b4b',alpha=.6))
            ax.text(4,3.5,'Blocked at 3-12s; passed after withdrawal',fontsize=7)
        label='distance held; task active' if case.get('follow_distance_held') else 'safe refusal' if case['safe_refusal'] else 'reached' if case['completed'] else 'incomplete'
        ax.set(xlim=(.5,10),ylim=(.5,8),aspect='equal',title=case['case']+' | '+label,xlabel='X (m)',ylabel='Y (m)');ax.grid(alpha=.15)
        ax=axes[row+1,column];v=np.array([h['velocity'] for h in history]);ax.plot(times,v[:,0],label='actual v (m/s)');ax.plot(times,v[:,1],label='actual omega (rad/s)')
        for field in ['cancelled','manual','emergency','dropout']:
            active=np.array([h[field] for h in history]);ax.fill_between(times,-.4,.9,where=active,alpha=.10,label=field)
        ax.set(title='Actual plant feedback | %.1fs'%case['simulated_s'],xlabel='simulation time (s)',ylim=(-.4,.9));ax.grid(alpha=.2);ax.legend(fontsize=7)
        pixels=np.where(world.static[...,None]==100,np.array([51,75,90]),np.array([255,255,255])).astype(np.uint8)
        stream=io.BytesIO();Image.fromarray(pixels[::-1]).save(stream,format='PNG')
        payload.append(dict(case=case['case'],history=history,goal=case['request']['goal'],
            actors=[world.actor_position(h['t']) for h in history],background=base64.b64encode(stream.getvalue()).decode()))
    fig.suptitle('Synthetic closed-loop feedback; assumed geometry / braking; NOT real robot execution')
    fig.savefig(Path(out)/'overview.png',dpi=140);plt.close(fig)
    encoded=json.dumps(dict(cases=payload,footprint=geometry.footprint.tolist()),ensure_ascii=False,allow_nan=False).replace('</','<'+chr(92)+'/')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>有反馈运动学仿真</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:20px;background:#eef3f6;color:#223540}canvas{width:100%;background:white}button,select{padding:8px}input{width:60%}pre{white-space:pre-wrap}</style>
<h1>有反馈运动学仿真</h1><p>绿色为实际积分轨迹，蓝色为当前假设足迹，红色为当前圆形障碍。背景为合成世界；算法只接收当前半径内观测。没有使用bag扫描或驱动真实底盘。</p>
<select id="case"></select> <button id="play">播放／暂停</button><input id="time" type="range" min="0" value="0"><canvas id="map" width="1000" height="650"></canvas><pre id="details"></pre>
<details><summary>整段路径与速度</summary><img src="overview.png" style="width:100%"></details>
<script>const D=PAYLOAD,select=document.getElementById('case'),slider=document.getElementById('time'),canvas=document.getElementById('map'),ctx=canvas.getContext('2d');let frame=0,playing=false,last=0;
D.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.case;select.append(o)});const images=D.cases.map(c=>{const i=new Image();i.src='data:image/png;base64,'+c.background;return i});
function draw(){const c=D.cases[Number(select.value)],h=c.history[frame];slider.max=c.history.length-1;slider.value=frame;const scale=60,xy=(x,y)=>[60+x*scale,620-y*scale];ctx.fillStyle='#e2e9ed';ctx.fillRect(0,0,1000,650);ctx.imageSmoothingEnabled=false;ctx.drawImage(images[Number(select.value)],60,20,840,600);
function line(p,color,width){ctx.beginPath();p.forEach((q,j)=>j?ctx.lineTo(...xy(...q)):ctx.moveTo(...xy(...q)));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
line(c.history.slice(0,frame+1).map(h=>h.pose.slice(0,2)),'#168147',3);const p=h.pose,co=Math.cos(p[2]),si=Math.sin(p[2]);const poly=D.footprint.map(q=>[p[0]+co*q[0]-si*q[1],p[1]+si*q[0]+co*q[1]]);line([...poly,poly[0]],'#2278b5',3);line([p.slice(0,2),[p[0]+.7*co,p[1]+.7*si]],'#2278b5',3);
ctx.beginPath();ctx.arc(...xy(...c.goal.slice(0,2)),7,0,2*Math.PI);ctx.fillStyle='#d5781e';ctx.fill();const a=c.actors[frame];if(a){ctx.beginPath();ctx.arc(...xy(...a),.35*scale,0,2*Math.PI);ctx.fillStyle='#d64b4b';ctx.fill()}
if(c.goal.length===3){line([c.goal.slice(0,2),[c.goal[0]+.7*Math.cos(c.goal[2]),c.goal[1]+.7*Math.sin(c.goal[2])]],'#d5781e',3)}
document.getElementById('details').textContent=JSON.stringify(h,null,2)}
select.onchange=()=>{frame=0;draw()};slider.oninput=()=>{frame=Number(slider.value);draw()};document.getElementById('play').onclick=()=>{playing=!playing};
function animate(t){if(playing&&t-last>=100){frame=(frame+1)%D.cases[Number(select.value)].history.length;draw();last=t}requestAnimationFrame(animate)}images[0].onload=draw;setTimeout(draw,100);requestAnimationFrame(animate);
</script></html>'''.replace('PAYLOAD',encoded)
    (Path(out)/'review.html').write_text(html)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--case',choices=['straight','turn','dynamic','crossing','follow','faults','heading','rotation','narrow','turn_stop','infeasible'],action='append')
    p.add_argument('--plant-delay-s',type=float,default=.1)
    p.add_argument('--plant-acceleration-scale',type=float,default=1.)
    p.add_argument('--feedback-noise-m',type=float,default=0.)
    p.add_argument('--feedback-noise-rad',type=float,default=0.)
    p.add_argument('--min-turning-radius-m',type=float,help='Explicit assumed radius for sensitivity comparison')
    p.add_argument('--collision-backend',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--max-s',type=float,default=75.,help='Logical simulation time budget per case')
    args=p.parse_args()
    if not np.isfinite(args.max_s) or not 0<args.max_s<=300:p.error('Simulation budget must be in (0,300] seconds')
    if (not np.isfinite([args.plant_delay_s,args.plant_acceleration_scale,args.feedback_noise_m,args.feedback_noise_rad]).all() or
            not 0<=args.plant_delay_s<=1 or not 0<args.plant_acceleration_scale<=2 or not 0<=args.feedback_noise_m<=.1 or
            not 0<=args.feedback_noise_rad<=.1):p.error('Invalid development plant perturbation')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    planning=load_config();motion=load_motion_config();geometry=VehicleGeometry.historical(planning['geometry_reference'])
    motion['collision_backend']=args.collision_backend
    if args.min_turning_radius_m is not None:
        if not np.isfinite(args.min_turning_radius_m) or args.min_turning_radius_m<=0:p.error('Turning radius must be positive and finite')
        motion['min_turning_radius_m']=args.min_turning_radius_m
        motion['turning_radius_source']='explicit_simulation_comparison_not_measured'
    planning['motion_constraints']={k:motion[k] for k in ('min_turning_radius_m','allow_reverse','allow_in_place_rotation')}
    results=[];data=dict(scope='simulation_only_no_chassis',planning=planning,motion=motion,controller='dwa',cases=results,run_complete=False)
    for kind in args.case or ['straight','turn','dynamic','faults']:
        r=run_case(kind,planning,motion,geometry,max_s=args.max_s,progress=True,plant_delay_s=args.plant_delay_s,
            plant_acceleration_scale=args.plant_acceleration_scale,feedback_noise_m=args.feedback_noise_m,feedback_noise_rad=args.feedback_noise_rad);results.append(r)
        (out/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False))
        print(json.dumps({k:r[k] for k in ['case','completed','collision','simulated_s','final_pose','final_velocity','safety_counts','compute_ms']}),flush=True)
    data['run_complete']=True
    (out/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False));write_review(data,out,geometry)


if __name__=='__main__':main()
