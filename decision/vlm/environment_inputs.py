"""Neutral environment context; separate from frozen REL inputs and model calls."""
import hashlib
import copy
from pathlib import Path

import numpy as np

from perception.detection_tracking.inference.geometry import validate_transform
from perception.environment.layers import load_config
from perception.environment.context import layer_binding
from .scene_inputs import canonical


def prepare_scene_context(bundle, *, settings=None):
    """Consume one bound live/replayed scene; never resolve modalities by 'latest'."""
    bundle.verify()
    scene = bundle.record
    if scene['rgb']['status'] != 'valid' or scene['localization']['status'] != 'valid':
        raise ValueError('A model scene requires valid local geometry and current RGB')
    context = prepare_environment_context(bundle.geometry,scene['memory']['observations'],bundle.rgb,
        rgb_stamp_s=scene['rgb']['stamp_s'],as_of=scene['as_of'],observation_id=scene['observation_id'],
        clock_domain=scene['clock_domain'],rgb_reference=scene['rgb']['reference'],settings=settings)
    context.update(protocol='environment_scene_context',scene_input_fingerprint=scene['input_fingerprint'],
        people=copy.deepcopy(scene['people']),prior_association=copy.deepcopy(scene['prior']),scene_health=scene['health'],
        memory_status=scene['memory']['status'],scope='causal_geometry_rgb_people_and_optional_prior_memory')
    if 'prior_geometry' in scene:
        context['prior_geometry'] = copy.deepcopy(scene['prior_geometry'])
    tracks = scene['people'].get('data',{}).get('tracks',[])
    context['person_mapping'] = [dict(label='P%d'%(index+1),track_id=row['track_id'],
        lifecycle_generation=row['lifecycle_generation']) for index,row in
        enumerate(sorted(tracks,key=lambda row:row['track_id']))]
    context.pop('input_fingerprint')
    context['input_fingerprint'] = hashlib.sha256(canonical(context).encode()).hexdigest()
    return context


def prepare_environment_context(snapshot, observations, rgb, *, rgb_stamp_s, as_of,
                                observation_id, clock_domain, rgb_reference, settings=None):
    settings = settings or load_config()['consumer']
    meta = snapshot.metadata
    if not observation_id or not clock_domain or not np.isfinite([as_of, rgb_stamp_s, meta['stamp_s']]).all():
        raise ValueError('An observation identity, explicit clock domain and finite sensor times are required')
    if not isinstance(rgb_reference,dict) or not rgb_reference:
        raise ValueError('An explicit RGB asset/observation source is required')
    if meta.get('time_scope') == 'offline_final_prior' or meta['geometry_status'] not in ('OBSERVED','GROUND_UNAVAILABLE'):
        raise ValueError('Current context requires a valid causal local observation, not a final prior')
    if not 0 <= as_of-meta['stamp_s'] <= settings['max_geometry_age_s'] or not 0 <= as_of-rgb_stamp_s <= settings['max_rgb_age_s']:
        raise ValueError('Current RGB/geometry is future or stale at this observation cutoff')
    pose = validate_transform(meta['world_from_lidar'])
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('Expected a decoded uint8 RGB image')
    history = []
    for row in sorted(observations, key=lambda item:item['stamp_s'], reverse=True):
        if ((row['session_id'],row['generation'],row['frame_id']) !=
                (meta['session_id'],meta['generation'],meta['frame_id'])):
            raise ValueError('Historical coordinates belong to another local lifecycle')
        if row['stamp_s'] > as_of or row['pose_stamp_s'] > row['stamp_s']:
            raise ValueError('Future observations or poses cannot enter model context')
        if as_of-row['stamp_s'] > settings['history_age_s']:
            continue
        matrix = validate_transform(row['world_from_lidar'])
        history.append(dict(id=row['id'],stamp_s=row['stamp_s'],age_s=as_of-row['stamp_s'],
            pose_stamp_s=row['pose_stamp_s'],position=matrix[:3,3].tolist(),
            image_sha256=row['image_sha256'],calibration_sha256=row.get('calibration_sha256')))
        if len(history) >= settings['history_limit']:
            break
    context = dict(protocol='environment_context',observation_id=observation_id,clock_domain=clock_domain,
        as_of=as_of,frame_id=meta['frame_id'],session_id=meta['session_id'],generation=meta['generation'],
        geometry=dict(snapshot_id=meta['snapshot_id'],stamp_s=meta['stamp_s'],age_s=as_of-meta['stamp_s'],
            layer_binding=layer_binding(snapshot),origin_xy=meta['origin_xy'],resolution=meta['resolution'],
            shape=meta['shape'],geometry_status=meta['geometry_status'],
            cell_counts={str(k):int((snapshot.layers['occupancy']==k).sum()) for k in [-1,0,100]}),
        reference_from_lidar=pose.tolist(),
        rgb=dict(stamp_s=rgb_stamp_s,age_s=as_of-rgb_stamp_s,shape=list(rgb.shape),
                 sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),reference=rgb_reference),history=history,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prior_association=meta.get('association',{}),
        scope='causal_local_geometry_and_historical_observation_references',
        navigation_reachability_verified=False)
    context['input_fingerprint'] = hashlib.sha256(canonical(context).encode()).hexdigest()
    return context


def render_environment_context(snapshot, context, rgb, *, prior_geometry=None):
    content = {key:value for key,value in context.items() if key != 'input_fingerprint'}
    if (hashlib.sha256(canonical(content).encode()).hexdigest() != context['input_fingerprint'] or
            context['renderer_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            layer_binding(snapshot) != context['geometry']['layer_binding'] or
            hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest() != context['rgb']['sha256']):
        raise ValueError('Renderer inputs no longer match their source binding')
    prior = context.get('prior_geometry',{})
    if prior.get('status') == 'valid':
        if prior_geometry is None or layer_binding(prior_geometry) != prior['layer_binding']:
            raise ValueError('Renderer prior no longer matches its scene binding')
    elif prior_geometry is not None:
        raise ValueError('Cannot render layers for an unavailable prior')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    has_prior = 'prior_geometry' in context
    figure = Figure(figsize=(18 if has_prior else 12,5),dpi=110,constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(1,3 if has_prior else 2)
    camera, bev = axes[:2]
    camera.imshow(rgb); camera.axis('off'); camera.set_title('Current RGB / t=%.3f'%context['rgb']['stamp_s'])
    meta = snapshot.metadata
    h,w = meta['shape']; x,y = meta['origin_xy']; r = meta['resolution']
    cells = snapshot.layers['occupancy']
    classes = np.where(cells == -1,0,np.where(cells == 0,1,2))
    bev.imshow(classes,origin='lower',extent=[x,x+w*r,y,y+h*r],vmin=0,vmax=2,
               cmap=ListedColormap(['#b8c4cf','#ffffff','#263b4d']),interpolation='nearest')
    pose = np.asarray(context['reference_from_lidar'])
    bev.plot(pose[0,3],pose[1,3],'o',color='#db551e')
    bev.arrow(pose[0,3],pose[1,3],pose[0,0],pose[1,0],color='#db551e',width=.05)
    for index,row in enumerate(context['history']):
        px,py = row['position'][:2]
        bev.plot(px,py,'x',color='#313f56')
        bev.annotate('H%d'%(index+1),(px,py),fontsize=8)
    if 'people' in context:
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        tracks = {row['track_id']:row for row in context['people'].get('data',{}).get('tracks',[])}
        for item in context['person_mapping']:
            row = tracks[item['track_id']]
            px,py,_,dx,dy,_,yaw,_ = row['box']
            color = '#137bbb' if row['state'] == 'observed' else '#be6b10'
            patch = Rectangle((-dx/2,-dy/2),dx,dy,fill=False,edgecolor=color,
                              linestyle='-' if row['state'] == 'observed' else '--',linewidth=1.5)
            patch.set_transform(Affine2D().rotate(yaw).translate(px,py)+bev.transData)
            bev.add_patch(patch)
            bev.annotate(item['label'],(px,py),color=color,fontsize=9)
        prior_state = context['prior_association'].get('status','UNAVAILABLE')
        figure.suptitle('Scene: %s | prior: %s | people: %s'%(context['scene_health'],prior_state,context['people']['status']))
        bev.text(.01,.99,'Blue: observed | orange dashed: predicted | H: history',transform=bev.transAxes,
                 va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    bev.set_title('Gray unknown / white observed free / dark occupied')
    # People/history can expand the axes beyond the observed raster. Keep
    # that uncovered margin unknown rather than the axes' default white.
    bev.set_facecolor('#b8c4cf')
    bev.set_xlabel(context['frame_id']+' x (m)');bev.set_ylabel('y (m)')
    if has_prior:
        view = axes[2]
        view.set_facecolor('#b8c4cf')
        if prior_geometry is None:
            view.text(.5,.5,'Prior withdrawn\n'+prior.get('reason','unavailable'),
                      ha='center',va='center',transform=view.transAxes)
            view.set_xticks([]);view.set_yticks([])
        else:
            pm = prior_geometry.metadata
            ph,pw = pm['shape'];px,py = pm['origin_xy'];pr = pm['resolution']
            pc = prior_geometry.layers['occupancy']
            view.imshow(np.where(pc == -1,0,np.where(pc == 0,1,2)),origin='lower',
                extent=[px,px+pw*pr,py,py+ph*pr],vmin=0,vmax=2,
                cmap=ListedColormap(['#b8c4cf','#f4ead5','#775c43']),interpolation='nearest')
            mp = np.asarray(prior['map_from_lidar'])
            view.plot(mp[0,3],mp[1,3],'o',color='#db551e')
            view.arrow(mp[0,3],mp[1,3],mp[0,0],mp[1,0],color='#db551e',width=.08)
            for index,row in enumerate(prior['observations']):
                view.plot(*row['position'][:2],'+',color='#7350a2')
                view.annotate('S%d'%(index+1),row['position'][:2],fontsize=8,color='#7350a2')
            transform = np.asarray(prior['map_from_local'])
            for item in context.get('person_mapping',[]):
                row = tracks[item['track_id']]
                point = transform @ np.r_[row['box'][:3],1.]
                color = '#137bbb' if row['state'] == 'observed' else '#be6b10'
                view.plot(*point[:2],marker='o' if row['state'] == 'observed' else 'x',color=color)
                view.annotate(item['label'],point[:2],color=color,fontsize=8)
            view.set_xlim(px,px+pw*pr);view.set_ylim(py,py+ph*pr)
            view.set_xlabel(prior['frame_id']+' x (m)');view.set_ylabel('y (m)')
        view.set_title('Historical prior / S: stored views\nNot current free space or verified reachability')
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:,:,:3].copy()


def main():
    import argparse
    import json
    import cv2
    from perception.environment.scene import SceneInputs
    parser = argparse.ArgumentParser(description='Render one saved causal scene, without a model request')
    parser.add_argument('--scene',required=True,help='Run scene directory containing scene.json/layers.npz/rgb.png')
    parser.add_argument('--out',required=True,help='New directory for context.json and preview.png')
    parser.add_argument('--prior-map',help='Explicit static package: create a new binding without changing the saved source scene')
    args = parser.parse_args()
    source,target = Path(args.scene),Path(args.out)
    if target.exists():
        raise FileExistsError('Select a new scene rendering output directory')
    bundle = SceneInputs.load(source)
    if args.prior_map:
        from perception.environment.prior import PriorMap
        bundle = PriorMap(args.prior_map).attach(bundle,load_config()['scene'])
    context = prepare_scene_context(bundle)
    preview = render_environment_context(bundle.geometry,context,bundle.rgb,prior_geometry=bundle.prior_geometry)
    target.mkdir(parents=True)
    (target/'context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    success,data = cv2.imencode('.png',preview[:,:,::-1])
    if not success:
        raise ValueError('Preview encoding failed')
    (target/'preview.png').write_bytes(data.tobytes())
    print(json.dumps(dict(observation_id=context['observation_id'],input_fingerprint=context['input_fingerprint'],
                         people=context['people']['status'],out=str(target))))


if __name__ == '__main__':
    main()
