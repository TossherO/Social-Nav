"""Bag observations, frozen model images and assisted offline review preparation.

ROS is imported only by extract(); model/rendering dependencies are imported by
their own operations so extraction remains usable with the system interpreter.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def select_poses(rows,start,offsets):
    if (not offsets or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in offsets)
            or any(b<=a for a,b in zip(offsets,offsets[1:]))):
        raise ValueError('Ordered finite bag offsets required')
    selected=[]
    for offset in offsets:
        available=[row for row in rows if row['stamp_s']<=start+offset]
        if not available or available[-1]['health']!=0:
            raise ValueError('Current localization invalid; no fallback to an older valid pose')
        pose=available[-1]
        if start+offset-pose['stamp_s']>.2:raise ValueError('Current localization stale')
        if selected and pose['stamp_ns']<=selected[-1]['stamp_ns']:raise ValueError('Distinct ordered samples required')
        selected.append(pose)
    return selected


def observation_clock(metadata, poses, explicit=None):
    """Keep recorded domains; infer only documented native/legacy run formats."""
    domains={r['clock_domain'] for r in poses if r.get('clock_domain')}
    if metadata.get('clock_domain'):domains.add(metadata['clock_domain'])
    if 'runtime' in metadata and metadata.get('backend')=='native_roll_fast_lio' and metadata.get('source') in ('bag','live'):
        domains.add('runtime_live' if metadata['source']=='live' else 'runtime_bag')
    if not domains and metadata.get('source')=='bag' and 'playback' in metadata:
        domains.add('ros_sim_time')  # Recorded ROS bag diagnostic contract.
    if explicit:domains.add(explicit)
    if len(domains)!=1:raise ValueError('Missing or conflicting source clock domain; provide --clock-domain only for an unspecified source')
    return domains.pop()


def bind_scene_observation(observation, scene):
    """Derive an observation bound to an existing scene, without editing sources."""
    from copy import deepcopy
    scene.verify();r=scene.record
    if (any(observation[k]!=r[k] for k in ('frame_id','session_id','generation','clock_domain')) or
            abs(observation['stamp_s']-r['as_of'])>1e-6 or
            r['localization']['status']!='valid' or
            not np.allclose(observation['world_from_lidar'],r['localization']['world_from_lidar'],atol=1e-6,rtol=0)):
        raise ValueError('Scene and raw observation time/pose/lifecycle disagree')
    result=deepcopy(observation)
    result['source']['raw_observation_id']=observation['observation_id']
    result['source']['scene_input_fingerprint']=r['input_fingerprint']
    result['observation_id']=r['observation_id']
    return result


def extract(run,bag_path,output,offsets,*,clock_domain=None,scenes=None):
    import rosbag
    from perception.sensors.image import rgb_array
    from perception.sensors.pointcloud import pointcloud_fields
    from perception.mapping_localization.results import file_fingerprint
    from perception.detection_tracking.inference.geometry import validate_transform
    run,bag_path,output=Path(run),Path(bag_path),Path(output)
    meta=json.loads((run/'run.json').read_text())
    expected=meta['playback']['bag_fingerprint']
    if file_fingerprint(bag_path)!=expected:raise ValueError('Bag differs from the localization source')
    poses=[json.loads(line) for line in (run/'localization.jsonl').read_text().splitlines()]
    clock=observation_clock(meta,poses,clock_domain)
    bound_scenes=[]
    if scenes:
        from perception.environment.scene import SceneInputs
        if len(scenes) not in (1,len(offsets)):raise ValueError('Provide the current scene only, or one scene per offset')
        loaded=[SceneInputs.load(p) for p in scenes]
        bound_scenes=([None]*(len(offsets)-1)+loaded) if len(scenes)==1 else loaded
    if output.resolve()==run.resolve() or run.resolve() in output.resolve().parents or bag_path.resolve().parent in output.resolve().parents:
        raise ValueError('Do not write into source bag or perception directories')
    output.mkdir(parents=True,exist_ok=False)
    samples=[];last_rgb=None
    with rosbag.Bag(str(bag_path),'r') as bag:
        start=bag.get_start_time()
        selected=select_poses(poses,start,offsets)
        index=0
        for topic,message,_ in bag.read_messages(topics=['/camera/color/image_raw','/rslidar_points']):
            if topic=='/camera/color/image_raw':
                last_rgb=message;continue
            target=selected[index]
            if message.header.stamp.to_nsec()!=target['stamp_ns']:continue
            if message.header.frame_id!=target['child_frame_id']:raise ValueError('Scan/pose sensor frame mismatch')
            stamp=message.header.stamp.to_sec()
            if last_rgb is None or not 0<=stamp-last_rgb.header.stamp.to_sec()<=.15:
                raise ValueError('Missing past RGB; no future pairing')
            fields=pointcloud_fields(message,('x','y','z','intensity'))
            points=np.stack([fields[k] for k in ('x','y','z','intensity')],axis=1).astype('<f4')
            rgb=rgb_array(last_rgb)
            q=np.asarray(target['quaternion_xyzw'],dtype=float);q/=np.linalg.norm(q)
            x,y,z,w=q
            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=target['position'];validate_transform(transform)
            directory=output/('O%02d'%index);directory.mkdir()
            np.save(directory/'scan.npy',points,allow_pickle=False);Image.fromarray(rgb).save(directory/'rgb.png')
            row=dict(observation_id='seating_scan_'+str(target['stamp_ns']),stamp_s=stamp,elapsed_s=stamp-start,
                frame_id=target['frame_id'],sensor_frame=message.header.frame_id,clock_domain=clock,
                session_id=target['session_id'],generation=target['generation'],world_from_lidar=transform.tolist(),
                rgb_stamp_s=last_rgb.header.stamp.to_sec(),rgb_age_s=stamp-last_rgb.header.stamp.to_sec(),
                scope='raw_single_scan_not_deskewed_not_free_space_or_reachability',
                source=dict(bag=str(bag_path.resolve()),bag_fingerprint=expected,run=str(run.resolve()),
                    localization_row_sha256=hashlib.sha256(json.dumps(target,sort_keys=True).encode()).hexdigest()),
                assets={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['rgb.png','scan.npy']})
            if bound_scenes and bound_scenes[index] is not None:
                row=bind_scene_observation(row,bound_scenes[index])
                row['source']['scene_directory']=str(Path(scenes[-1] if len(scenes)==1 else scenes[index]).resolve())
            (directory/'observation.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
            samples.append(dict(path=directory.name,elapsed_s=row['elapsed_s']));index+=1
            if index==len(selected):break
    if len(samples)!=len(offsets):raise ValueError('Missing exact scan observations')
    (output/'index.json').write_text(json.dumps(samples,indent=2)+'\n')
    print(json.dumps(samples),flush=True)


def validate_window(rows):
    from .. import scene_inputs as visual
    from .geometry import IDENTITY
    if len(rows) != 3:
        raise ValueError('Exactly three distinct causal observations required')
    current = rows[-1]
    if any(any(row[k] != current[k] for k in IDENTITY) for row in rows):
        raise ValueError('Observation window crosses a localization reset')
    if any(b['stamp_s'] <= a['stamp_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Window must be strictly chronological')
    expected = np.asarray(visual.OFFSETS_NS)/1e9
    actual = np.array([r['stamp_s']-current['stamp_s'] for r in rows])
    if np.any(np.abs(actual-expected) > .15):
        raise ValueError('Expected T1/T2 offsets -1, -0.5, 0 seconds within 150ms')
    for row in rows:
        if not 0 <= row['stamp_s']-row['rgb_stamp_s'] <= .15:
            raise ValueError('RGB must be current or past, within 150ms')


def render_window(directory, observation, candidate_set, *, elapsed, extent, current):
    """Past panels show only their scan, body and prior; goals on current only."""
    from PIL import ImageDraw
    from .. import scene_inputs as visual
    from .geometry import footprint_world, base_pose
    directory = Path(directory)
    cloud = np.load(directory/'scan.npy', allow_pickle=False)
    frame = dict(people=[], image_size=[640,480])
    image = visual.render_observation(frame, (directory/'rgb.png').read_bytes(), {}, elapsed, extent,
        bev_background=visual.pointcloud_background(cloud, extent))
    layer = image.copy(); draw = ImageDraw.Draw(layer)
    context = candidate_set['context']; inverse = np.linalg.inv(np.asarray(observation['world_from_lidar']))
    def local(world_xy):
        # XY display uses each actual sensor frame, not uncorrected cross-frame deltas.
        p = np.c_[np.asarray(world_xy), np.full(len(world_xy), context['table_prior']['edge_z_m'])]
        return (p @ inverse[:3,:3].T+inverse[:3,3])[:,:2]
    def pixels(world_xy): return [visual.bev_xy(*p,extent) for p in local(world_xy)]
    polygon = pixels(context['table_prior']['polygon_xy_m'])
    draw.line(polygon+[polygon[0]], fill='#17865b', width=3)
    base = base_pose(observation['world_from_lidar'], context['base_from_lidar'])
    body = pixels(footprint_world(context['footprint_xy_m'], [*base[:2,3],math.atan2(base[1,0],base[0,0])]))
    draw.line(body+[body[0]], fill='#d96b22', width=2)
    if current:
        markers=[]; bounds=visual.BEV_RECT
        for c in candidate_set['candidates']:
            x,y,yaw=c['pose']; center=pixels([[x,y]])[0]
            shape=pixels(footprint_world(context['footprint_xy_m'], c['pose']))
            draw.line(shape+[shape[0]],fill='#9eafc5',width=1)
            if not (bounds[0] <= center[0] <= bounds[2] and bounds[1] <= center[1] <= bounds[3]): continue
            tip=pixels([[x+.35*math.cos(yaw),y+.35*math.sin(yaw)]])[0]
            draw.line([center,tip],fill='#184fa0',width=3)
            v=np.asarray(tip)-center; v=v/max(np.linalg.norm(v),1e-6); side=np.array([-v[1],v[0]])
            draw.polygon([tuple(tip),tuple(np.asarray(tip)-v*8+side*4),tuple(np.asarray(tip)-v*8-side*4)],fill='#184fa0')
            markers.append((c['id'],*center))
        face=visual.font(24); sizes={i:(draw.textbbox((0,0),f'C{i}',font=face)[2]+10,32) for i,_,_ in markers}
        placements=visual.place_bev_labels(markers,sizes,(bounds[0]+3,bounds[1]+3,bounds[2]-3,bounds[3]-3))
        anchors={i:(x,y) for i,x,y in markers}
        for i,(left,top,right,bottom) in placements:
            x,y=anchors[i]
            draw.line([(x,y),(max(left,min(x,right)),max(top,min(y,bottom)))],fill='#728ba9',width=1)
        for _,x,y in markers:draw.ellipse((x-7,y-7,x+7,y+7),fill='#184fa0')
        for i,rect in placements:
            draw.rounded_rectangle(rect,radius=3,fill='white',outline='#184fa0')
            draw.text((rect[0]+5,rect[1]+1),f'C{i}',fill='#184fa0',font=face)
    image.paste(layer.crop(visual.BEV_RECT),visual.BEV_RECT[:2])
    draw=ImageDraw.Draw(image)
    draw.rectangle((visual.BEV_RECT[0],0,1280,31),fill=visual.BACKGROUND)
    draw.text((visual.BEV_RECT[0]+6,3),'BEV: table prior / '+('current goals' if current else 'past scan'),
              font=visual.font(21),fill='#213448')
    return image


def freeze_window(observations, candidate_path, output, *, current_only=False):
    """current_only is an explicit, same-layout ablation, not an automatic fallback."""
    from .. import scene_inputs as visual
    from ..annotation.rules import digest
    from ..vlm_api_common import read_json, save_json
    from .geometry import read_observation, IDENTITY, model_context, selection_context
    from .inference import validate_candidate_set, binding, selection_schema
    from ..project_config import load_config
    paths=[Path(p) for p in observations]; rows=[read_observation(p) for p in paths]
    validate_window(rows)
    candidates=candidate_path if isinstance(candidate_path,dict) else read_json(candidate_path)
    validate_candidate_set(candidates); current=rows[-1]
    for k in (*IDENTITY,'observation_id','stamp_s'):
        if candidates[k]!=current[k]:raise ValueError('Candidate/current observation mismatch')
    context=candidates['context']
    if context['kind']!='table_outline_hypothesis':raise ValueError('Window currently requires an explicit table prior')
    if context['table_prior']['available_after_s'] > rows[0]['stamp_s']:
        raise ValueError('Prior must already be available in the oldest displayed observation')
    output=Path(output)
    if any(output.resolve().is_relative_to(p.resolve()) for p in paths):raise ValueError('Do not write into source observations')
    output.mkdir(parents=True,exist_ok=False)
    task=load_config()['tasks']['seating']; extent=task['window_extent_m']
    selected=selection_context(context,current['world_from_lidar'],candidates['candidates'],task['adjustment_gate'])
    images=[]; sources=[]
    used=list(zip(paths,rows))[-1:] if current_only else list(zip(paths,rows))
    for index,(path,row) in enumerate(used):
        name=f'frame_{index}.png'
        render_window(path,row,candidates,elapsed=row['stamp_s']-current['stamp_s'],extent=extent,
               current=row['observation_id']==current['observation_id']).save(output/name)
        images.append(dict(path=name,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()))
        sources.append(dict(path=str(path.resolve()),observation_sha256=hashlib.sha256((path/'observation.json').read_bytes()).hexdigest(),
                            observation_id=row['observation_id'],stamp_s=row['stamp_s'],rgb_stamp_s=row['rgb_stamp_s'],
                            world_from_lidar=row['world_from_lidar']))
    fields=dict(instruction=candidates['instruction'],units=dict(position='m',yaw='rad'),frame_id=current['frame_id'],
        ego_pose=current['world_from_lidar'],candidates=candidates['candidates'],
        dt=[r['stamp_s']-current['stamp_s'] for _,r in used],
        images='chronological RGB+BEV; each BEV in its own LiDAR frame; current goals only in last panel',
        **model_context(context,current['world_from_lidar'],candidates['candidates'],include_approach=selected['mode']=='adjust_nearby_pose'))
    record=dict(candidate_set=candidates,candidate_binding=binding(candidates),fields=fields,images=images,
                sources=sources,selection_context=selected,response_schema=selection_schema(candidates),
                rendering=dict(kind='seating_rgb_bev_window',extent=extent,shared=visual.renderer_identity(),
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                experiment='current_only_ablation' if current_only else 'three_causal_frames')
    record['input_fingerprint']=digest(record);save_json(output/'input.json',record)
    return record


def render_scene(scene,candidate_set,path,*,observation=None):
    """Neutral RGB+BEV, known/unknown geometry retained; no answers or endpoint."""
    from .inference import validate_candidate_set
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    scene.verify();validate_candidate_set(candidate_set)
    record=scene.record
    for name in ['observation_id','frame_id','clock_domain','session_id','generation']:
        if candidate_set[name]!=record[name]:raise ValueError('Candidates belong to another observation')
    if candidate_set['stamp_s']!=record['as_of']:raise ValueError('Candidate cutoff mismatch')
    if record['localization']['status']!='valid':raise ValueError('Valid RGB/localization required')
    if observation is None:
        if record['rgb']['status']!='valid':raise ValueError('Valid RGB/localization required')
        pixels=scene.rgb
    else:
        import io
        from PIL import Image
        observation.verify();source=observation.record
        if (source['rgb']['status']!='valid' or source['scene_binding']!=record['input_fingerprint'] or
                source['stamp_s']!=record['as_of'] or source['clock_domain']!=record['clock_domain'] or
                source['local_identity']!=[record[k] for k in ('session_id','generation','frame_id')]):
            raise ValueError('RGB observation does not bind the seating scene')
        with Image.open(io.BytesIO(observation.image)) as image:
            pixels=np.asarray(image.convert('RGB'))
    fig=Figure(figsize=(12,5),dpi=130,constrained_layout=True);FigureCanvasAgg(fig)
    rgb,bev=fig.subplots(1,2);rgb.imshow(pixels);rgb.axis('off');rgb.set_title('Current RGB')
    meta=scene.geometry.metadata;cells=scene.geometry.layers['occupancy'];h,w=cells.shape;x,y=meta['origin_xy'];r=meta['resolution']
    bev.imshow(np.where(cells==-1,0,np.where(cells==0,1,2)),origin='lower',extent=[x,x+w*r,y,y+h*r],
               cmap=ListedColormap(['#b6c0c9','#f8f8f8','#34414d']),vmin=0,vmax=2,interpolation='nearest')
    pose=np.asarray(meta['world_from_lidar']);ex,ey=pose[:2,3]
    bev.plot(ex,ey,'o',color='#df531b',ms=6);bev.arrow(ex,ey,pose[0,0]*.5,pose[1,0]*.5,width=.035,color='#df531b')
    for candidate in candidate_set['candidates']:
        px,py,yaw=candidate['pose'];bev.plot(px,py,'o',color='#145dc0',ms=5)
        bev.arrow(px,py,.3*math.cos(yaw),.3*math.sin(yaw),width=.02,color='#145dc0')
        bev.annotate('C'+str(candidate['id']),(px,py),xytext=(6,5),textcoords='offset points',fontsize=11,
                     bbox=dict(facecolor='white',edgecolor='#145dc0',pad=1.5),zorder=10)
    positions=np.array([[ex,ey]]+[c['pose'][:2] for c in candidate_set['candidates']])
    low=positions.min(axis=0)-2;high=positions.max(axis=0)+2
    bev.set_xlim(low[0],high[0]);bev.set_ylim(low[1],high[1]);bev.set_aspect('equal');bev.set_facecolor('#b6c0c9')
    bev.set_title('C: candidate pose | gray: unknown');bev.set_xlabel(candidate_set['frame_id']+' x (m)');bev.set_ylabel('y (m)')
    fig.savefig(path)


def freeze_scene(scene_directory,candidate_path,output):
    from .. import vlm_api_common as common
    from ..annotation.rules import digest
    from .inference import validate_candidate_set, binding, selection_schema
    from perception.environment.scene import SceneInputs
    scene=SceneInputs.load(scene_directory);data=common.read_json(candidate_path)
    if 'context' in data:raise ValueError('Historical geometry review currently requires a raw observation')
    output=Path(output)
    if output.resolve().is_relative_to(Path(scene_directory).resolve()):raise ValueError('Output must be outside source scene')
    output.mkdir(parents=True,exist_ok=False)
    render_scene(scene,data,output/'image.png')
    pose=scene.geometry.metadata['world_from_lidar']
    fields=dict(instruction=data['instruction'],units=dict(position='m',yaw='rad'),
                frame_id=data['frame_id'],ego_pose=pose,candidates=data['candidates'])
    record=dict(candidate_set=data,candidate_binding=binding(data),scene_input_fingerprint=scene.record['input_fingerprint'],
                scene_source=str(Path(scene_directory).resolve()),image_sha256=hashlib.sha256((output/'image.png').read_bytes()).hexdigest(),
                fields=fields,response_schema=selection_schema(data))
    record['input_fingerprint']=digest(record)
    common.save_json(output/'input.json',record)
    return record


def freeze_scan(observation_directory,candidate_path,output):
    from .. import vlm_api_common as common
    from ..annotation.rules import digest
    from .inference import validate_candidate_set, binding, selection_schema
    from ..project_config import load_config
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory=Path(observation_directory);observation=common.read_json(directory/'observation.json')
    for name,expected in observation['assets'].items():
        if common.asset_path(directory,name).name!=name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Raw observation asset changed')
    candidates=candidate_path if isinstance(candidate_path,dict) else common.read_json(candidate_path)
    validate_candidate_set(candidates)
    if candidates.get('context',{}).get('kind')=='table_outline_hypothesis':
        raise ValueError('Table outline inputs use prepare-window; current-only is an explicit ablation')
    for name in ['observation_id','stamp_s','frame_id','clock_domain','session_id','generation']:
        if candidates[name]!=observation[name]:raise ValueError('Candidate observation mismatch')
    output=Path(output)
    if output.resolve().is_relative_to(directory.resolve()):raise ValueError('Output must be outside source observation')
    output.mkdir(parents=True,exist_ok=False)
    points=np.load(directory/'scan.npy',allow_pickle=False);pose=np.array(observation['world_from_lidar'])
    valid=np.isfinite(points).all(axis=1);points=points[valid]
    points=points[np.linalg.norm(points[:,:2],axis=1)<12]
    world=points[:,:3]@pose[:3,:3].T+pose[:3,3]
    fig=Figure(figsize=(12,5),dpi=130,constrained_layout=True);FigureCanvasAgg(fig);rgb,bev=fig.subplots(1,2)
    rgb.imshow(Image.open(directory/'rgb.png'));rgb.axis('off');rgb.set_title('Current RGB')
    scatter=bev.scatter(world[:,0],world[:,1],c=points[:,2],s=1.2,cmap='turbo',vmin=-1.5,vmax=1.5,linewidths=0)
    fig.colorbar(scatter,ax=bev,label='LiDAR height (m)',shrink=.7)
    ex,ey=pose[:2,3];bev.plot(ex,ey,'o',ms=7,color='#e15016',zorder=5)
    bev.arrow(ex,ey,.5*pose[0,0],.5*pose[1,0],color='#e15016',width=.025,zorder=5)
    for candidate in candidates['candidates']:
        x,y,yaw=candidate['pose'];bev.plot(x,y,'o',ms=5,color='#153f9d',zorder=6)
        bev.arrow(x,y,.3*math.cos(yaw),.3*math.sin(yaw),width=.02,color='#153f9d',zorder=6)
        offset=(0,19 if candidate['id']<4 else -23) if 'context' in candidates else (7,7)
        bev.annotate('C'+str(candidate['id']),(x,y),xytext=offset,textcoords='offset points',fontsize=11,
                     bbox=dict(facecolor='white',edgecolor='#153f9d',pad=1.5),zorder=10)
    geometry_fields={};selection=None
    if 'context' in candidates:
        from matplotlib.patches import Polygon
        from .geometry import footprint_world,model_context,selection_context
        context=candidates['context']
        if 'sampling_policy' in context:
            selection=selection_context(context,pose,candidates['candidates'],load_config()['tasks']['seating']['adjustment_gate'])
        geometry_fields=model_context(context,pose,candidates['candidates'],
            include_approach=selection is not None and selection['mode']=='adjust_nearby_pose')
        a,b=np.array(context['edge']['observed_segment_xy']);normal=np.array(context['edge']['estimated_interior_normal_xy'])
        bev.plot([a[0],b[0]],[a[1],b[1]],color='#06854c',lw=2,zorder=7)
        mid=(a+b)/2;bev.arrow(*mid,*(normal*.35),width=.02,color='#06854c',zorder=7)
        for candidate in candidates['candidates']:
            bev.add_patch(Polygon(footprint_world(context['footprint_xy_m'],candidate['pose']),
                closed=True,fill=False,lw=.8,edgecolor='#153f9d',alpha=.65,zorder=4))
        base=np.array(geometry_fields['ego_base_pose_hypothesis'])
        bev.add_patch(Polygon(footprint_world(context['footprint_xy_m'],
            [*base[:2,3],math.atan2(base[1,0],base[0,0])]),closed=True,fill=False,lw=2,edgecolor='#e15016',zorder=6))
        bev.plot(*base[:2,3],'+',color='#e15016',ms=10,zorder=7)
    positions=np.array([[ex,ey]]+[c['pose'][:2] for c in candidates['candidates']])
    low=positions.min(axis=0)-1.8;high=positions.max(axis=0)+1.8
    bev.set_xlim(low[0],high[0]);bev.set_ylim(low[1],high[1]);bev.set_aspect('equal')
    bev.set_title('Current scan + candidates; no free-space claim');bev.set_xlabel(candidates['frame_id']+' x (m)');bev.set_ylabel('y (m)')
    fig.savefig(output/'image.png')
    record=dict(candidate_set=candidates,candidate_binding=binding(candidates),
        source_observation_sha256=hashlib.sha256((directory/'observation.json').read_bytes()).hexdigest(),source=str(directory.resolve()),
        image_sha256=hashlib.sha256((output/'image.png').read_bytes()).hexdigest(),
        fields=dict(instruction=candidates['instruction'],units=dict(position='m',yaw='rad'),frame_id=candidates['frame_id'],
                    ego_pose=observation['world_from_lidar'],candidates=candidates['candidates'],
                    **geometry_fields),response_schema=selection_schema(candidates))
    if selection is not None:record['selection_context']=selection
    record['input_fingerprint']=digest(record);common.save_json(output/'input.json',record)
    return record


def prepare_review(observations, rois, reference_path, calibration_path, camera, output, *,
                   front_gaps_m, end_guard_m, reviewed_edge_path=None):
    """Prepare a small ordered, assisted dataset; no model calls or control.

    rois maps observation directory names to current-image XYXY regions. A null
    region may consume a separately reviewed edge only after its source cutoff.
    Values here are diagnostic parameters, not deployment defaults.
    """
    from .geometry import edge_seeds, fit_edge, edge_candidates, overlap_diagnostics
    import yaml
    from .. import vlm_api_common as common

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    observations, output = Path(observations), Path(output)
    if not isinstance(rois,dict) or not rois or any(not name or Path(name).name != name for name in rois):
        raise ValueError('Ordered observation names and current-image regions required')
    if output.resolve().is_relative_to(observations.resolve()):
        raise ValueError('Output must be outside frozen observations')
    output.mkdir(parents=True, exist_ok=False)
    reference = common.read_json(reference_path)
    calibration = yaml.safe_load(Path(calibration_path).read_text())
    transform = calibration['lidar_to_camera_optical']['matrix']
    base = np.eye(4)
    offset = reference['jetson_urdf']['base_from_rslidar']
    if offset['rpy_rad'] != [0., 0., 0.]:
        raise ValueError('Historical transform requires an explicit rotation implementation')
    base[:3, 3] = offset['translation_m']
    ground_config = yaml.safe_load(Path('perception/environment/configs/environment.yaml').read_text())
    seed_clouds, sources, cases, past_scans = [], [], [], []
    first = None; previous = -math.inf; identity = None
    for name, roi in rois.items():
        directory = observations/name
        obs_path = directory/'observation.json'; obs = common.read_json(obs_path)
        if obs['stamp_s'] <= previous:
            raise ValueError('Observations must be strictly ordered')
        previous = obs['stamp_s']
        current_identity = tuple(obs[k] for k in ['frame_id', 'clock_domain', 'session_id', 'generation'])
        if identity is not None and current_identity != identity:
            raise ValueError('Cannot accumulate edges across localization resets')
        identity = current_identity
        for asset, expected in obs['assets'].items():
            if common.asset_path(directory, asset).name != asset or sha(directory/asset) != expected:
                raise ValueError('Frozen source changed')
        cloud = np.load(directory/'scan.npy', allow_pickle=False)
        pose = np.asarray(obs['world_from_lidar'])
        if first is None:
            first = pose[:2, 3]
        if roi is not None:
            seed_clouds.append(edge_seeds(cloud, np.asarray(Image.open(directory/'rgb.png')),
                pose, transform, camera, roi))
            sources.append(dict(observation_id=obs['observation_id'], stamp_s=obs['stamp_s'],
                observation_sha256=sha(obs_path), rgb_roi_xyxy=roi))
            edge = fit_edge(np.concatenate(seed_clouds), first)
            used_sources = list(sources)
        else:
            if reviewed_edge_path is None:
                raise ValueError('Null ROI requires a source-bound reviewed edge')
            reviewed = common.read_json(reviewed_edge_path)
            if any(reviewed[k] != obs[k] for k in ['frame_id', 'session_id', 'generation']):
                raise ValueError('Reviewed edge belongs to another localization')
            if reviewed['source_cutoff_s'] > obs['stamp_s']:
                raise ValueError('Reviewed whole-demo edge is future information here')
            used_sources = []
            for source in reviewed['source_observations']:
                path = Path(reviewed_edge_path).parent.parent/source['path']/'observation.json'
                source_obs = common.read_json(path)
                if (source_obs['assets'] != source['assets'] or source_obs['stamp_s'] != source['stamp_s'] or
                        source_obs['observation_id'] != source['observation_id'] or
                        any(source_obs[k] != obs[k] for k in ['frame_id', 'clock_domain', 'session_id', 'generation'])):
                    raise ValueError('Reviewed edge source binding changed')
                used_sources.append(dict(observation_id=source['observation_id'], stamp_s=source['stamp_s'],
                    observation_sha256=sha(path)))
            edge = dict(reviewed['edge'], status=reviewed['status'], reviewed_asset_sha256=sha(reviewed_edge_path))
        context = dict(kind='historical_footprint_edge_hypothesis', pose_reference='base_link_hypothesis',
            reference_sha256=sha(reference_path), reference_status=reference['status'],
            footprint_xy_m=reference['costmap']['footprint_xy_m'], base_from_lidar=base.tolist(),
            edge=edge, source_cutoff_s=max(s['stamp_s'] for s in used_sources),
            frame_id=obs['frame_id'], session_id=obs['session_id'], generation=obs['generation'],
            source_bindings=used_sources, front_gaps_m=list(front_gaps_m), end_guard_m=end_guard_m)
        candidates = edge_candidates(context, obs)
        world = cloud[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        world = world[np.isfinite(world).all(axis=1)]
        data = {k: obs[k] for k in ['observation_id', 'stamp_s', 'frame_id', 'clock_domain', 'session_id', 'generation']}
        data.update(instruction='在房间中这张桌子旁选择方便的轮椅就座位置。', revision=1,
                    candidates=candidates, context=context)
        row = dict(case=name, elapsed_s=obs['elapsed_s'], edge=edge, candidate_set=data,
            status='prepared' if candidates else 'insufficient_observed_edge_span',
            diagnostics=overlap_diagnostics(world, pose[:3, 3], pose, context, candidates, ground_config))
        row['past_terminal_evidence'] = []
        for past_name, past_obs, past_world, past_pose in past_scans:
            past = overlap_diagnostics(past_world,past_pose[:3,3],past_pose,context,candidates,ground_config)
            row['past_terminal_evidence'].append(dict(source_case=past_name,
                age_s=obs['stamp_s']-past_obs['stamp_s'],
                candidates=[dict(id=c['id'],terminal_nonself_voxels=c['terminal_nonself_voxels'])
                            for c in past['candidates']]))
        if candidates:
            record = freeze_scan(directory, data, output/name)
            row['input_fingerprint'] = record['input_fingerprint']
        cases.append(row)
        past_scans.append((name,obs,world,pose))
    summary = dict(scope='assisted_offline_geometry_diagnostic_not_navigation_or_gold',
        source_observations=str(observations.resolve()), camera=camera,
        calibration_source=dict(path=str(calibration_path), sha256=sha(calibration_path)),
        reference_source=dict(path=str(reference_path), sha256=sha(reference_path)),
        ground_parameters=ground_config, rois=rois, cases=cases)
    common.save_json(output/'geometry.json', summary)
    return summary


def refine_review(previous_review, policy_path, output):
    """Reuse frozen causal edge fits and observations, not whole-demo endpoints."""
    from .geometry import validate_sampling_policy, refine_candidates
    from .. import vlm_api_common as common
    from ..annotation.rules import digest
    previous_review,policy_path,output=Path(previous_review),Path(policy_path),Path(output)
    original=common.read_json(previous_review);policy=common.read_json(policy_path)
    validate_sampling_policy(policy)
    observations=Path(original['source_observations'])
    if output.resolve().is_relative_to(observations.resolve()) or output.resolve().is_relative_to(previous_review.parent.resolve()):
        raise ValueError('New refinement output must be outside frozen sources')
    output.mkdir(parents=True,exist_ok=False)
    scans=[];cases=[];previous=-math.inf
    for case in original['cases']:
        name=case['case']
        if Path(name).name!=name:raise ValueError('Expected a source observation name')
        directory=observations/name;obs_path=directory/'observation.json'
        obs=common.read_json(obs_path)
        if obs['stamp_s']<=previous:raise ValueError('Ordered causal observations required')
        previous=obs['stamp_s']
        for asset,sha in obs['assets'].items():
            if common.asset_path(directory,asset).name!=asset or hashlib.sha256((directory/asset).read_bytes()).hexdigest()!=sha:
                raise ValueError('Frozen observation asset changed')
        if case['status']=='prepared':
            base=common.read_json(previous_review.parent/name/'input.json')
            if (base['input_fingerprint']!=case['input_fingerprint'] or base['candidate_set']!=case['candidate_set'] or
                    digest({k:v for k,v in base.items() if k!='input_fingerprint'})!=base['input_fingerprint']):
                raise ValueError('Original geometry input binding changed')
        cloud=np.load(directory/'scan.npy',allow_pickle=False);pose=np.asarray(obs['world_from_lidar'])
        scans.append(dict(observation=obs,world_points=cloud[:,:3]@pose[:3,:3].T+pose[:3,3],
            observation_sha256=hashlib.sha256(obs_path.read_bytes()).hexdigest()))
        data={k:obs[k] for k in ['observation_id','stamp_s','frame_id','clock_domain','session_id','generation']}
        source_context=case['candidate_set']['context']
        source_map={s['observation_id']:s for s in source_context['source_bindings']}
        for scan in scans:
            source=source_map.get(scan['observation']['observation_id'])
            if source is None or source['observation_sha256']!=scan['observation_sha256']:
                raise ValueError('Past scan not bound to original geometry sources')
        context,candidates,diagnostics=refine_candidates(source_context,obs,scans,policy)
        data.update(instruction='在这张桌子旁选择方便的轮椅就座位置，允许略斜停靠，脚踏可以进入桌下。',
            revision=case['candidate_set']['revision']+1,context=context,candidates=candidates)
        row=dict(case=name,elapsed_s=obs['elapsed_s'],status='prepared' if candidates else diagnostics['status'],
                 candidate_set=data,diagnostics=diagnostics)
        if candidates:
            record=freeze_scan(directory,data,output/name)
            row['input_fingerprint']=record['input_fingerprint']
        cases.append(row)
    summary=dict(scope='causal_relaxed_seating_development_not_clearance_or_gold',
        previous_review=dict(path=str(previous_review.resolve()),sha256=hashlib.sha256(previous_review.read_bytes()).hexdigest()),
        policy_source=dict(path=str(policy_path.resolve()),sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest()),
        source_observations=str(observations.resolve()),cases=cases)
    common.save_json(output/'geometry.json',summary)
    return summary
