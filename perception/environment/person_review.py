"""Build an offline, rotatable review of captured mapper inputs and point ownership.

This reruns the current ownership function on frozen observations, not detection,
tracking or temporal map fusion. Original point order and source times are kept.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np

def encoded(array,dtype):
    return base64.b64encode(np.asarray(array,dtype=dtype).tobytes()).decode('ascii')


def extract_rgb(request_path,output):
    """System ROS interpreter: match each exact cloud time to the latest past RGB."""
    import genpy
    import rosbag
    from PIL import Image
    from perception.sensors.image import rgb_array
    from perception.mapping_localization.results import file_fingerprint
    request=json.loads(Path(request_path).read_text())
    if file_fingerprint(Path(request['bag']))!=request['bag_fingerprint']:
        raise ValueError('RGB bag differs from the selected source run')
    segments=[];images={}
    for stamp in request['stamps']:
        if not segments or stamp-segments[-1][-1]>.6:segments.append([])
        segments[-1].append(stamp)
    with rosbag.Bag(request['bag'],'r') as bag:
        for segment in segments:
            index=0;last=None;previous=None
            def save(stamp):
                if last is None:return
                age=stamp-last.header.stamp.to_sec()
                if not 0<=age<=.2:return
                image=Image.fromarray(rgb_array(last));image.thumbnail((640,360))
                stream=io.BytesIO();image.save(stream,format='JPEG',quality=76)
                images[str(stamp)]=dict(stamp_s=last.header.stamp.to_sec(),age_s=age,
                    image=base64.b64encode(stream.getvalue()).decode())
            for _,message,_ in bag.read_messages(topics=['/camera/color/image_raw'],
                    start_time=genpy.Time.from_sec(segment[0]-.25),end_time=genpy.Time.from_sec(segment[-1]+.1)):
                stamp=message.header.stamp.to_sec()
                if previous is not None and stamp<previous:raise ValueError('RGB time moved backwards')
                previous=stamp
                while index<len(segment) and segment[index]<stamp:save(segment[index]);index+=1
                last=message
                if index==len(segment):break
            while index<len(segment):save(segment[index]);index+=1
    Path(output).write_text(json.dumps(dict(images=images,bag_fingerprint=request['bag_fingerprint']),separators=(',',':')))


def matched_image(images,stamp):
    image=images.get('images',{}).get(str(stamp))
    if image and not (0<=stamp-image['stamp_s']<=.2 and abs(stamp-image['stamp_s']-image['age_s'])<1e-6):
        raise ValueError('RGB must be a recent past observation')
    return image


def build(archive,prefix,output,*,config_path=None,rgb=None,rgb_run=None):
    from .dynamic import people_boxes
    from .layers import load_config
    from .person_objects import ownership_record,point_owners
    archive=Path(archive);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new review output directory')
    config=load_config(config_path) if config_path else load_config()
    images={} if rgb is None else json.loads(Path(rgb).read_text())
    prefix=prefix.rstrip('/')+'/'
    frames=[];clips=[];identities={};sources={};previous=None
    with ZipFile(archive) as source:
        index=json.loads(source.read(prefix+'index.json'))
        for entry in index['frames']:
            name=entry['file'];raw=source.read(prefix+name+'.json');cloud=source.read(prefix+name+'.npz')
            sources[name]=dict(metadata=hashlib.sha256(raw).hexdigest(),points=hashlib.sha256(cloud).hexdigest())
            record=json.loads(raw);m=record['metadata'];people=record['people'];stamp=m['stamp_s']
            if stamp!=record['stamp_s'] or entry['snapshot_id']!=m['snapshot_id'] or entry['offset_s']!=record['offset_s']:
                raise ValueError('Captured frame index differs from its source metadata')
            if previous is not None and stamp<=previous[0]:raise ValueError('Capture times must increase')
            with np.load(io.BytesIO(cloud),allow_pickle=False) as arrays:
                points=arrays['endpoints'];old=arrays['transient_mask'].astype(bool)
            if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid captured points')
            if old.shape!=(len(points),):raise ValueError('Captured mask has changed point order')
            boxes,status=people_boxes(people,stamp,(m['session_id'],m['generation']),m['frame_id'],
                config['dynamic'],clock_domain=people.get('clock_domain'))
            if status['status']!='valid':raise ValueError('This review requires source-valid captured people observations')
            if abs(people['stamp_s']-stamp)>1e-6:raise ValueError('Captured people are not from the same measurement')
            sensor=np.asarray(m['world_from_lidar'])[:3,3];ground=m.get('ground_tracking',{})
            plane=ground.get('plane_world') if ground.get('status')=='tracked' else None
            owners=point_owners(points,boxes,config,sensor,plane)
            objects=ownership_record(points,owners,boxes,status['track_ids'],people,stamp,
                (m['session_id'],m['generation']),m['frame_id'])['objects']
            circles={tuple(o['identity']):o['radius_m'] for o in objects};tracks=[]
            for j,(box,track) in enumerate(zip(boxes,status['track_ids'])):
                key=(people['session_id'],people['generation'],track)
                if key not in identities:identities[key]=len(identities)+1
                tracks.append(dict(pid=identities[key],identity=list(key),box=box.tolist(),
                    points=int((owners==j).sum()),radius_m=circles.get(key)))
            assigned=np.zeros(len(points),np.uint16)
            for j,track in enumerate(tracks):assigned[owners==j]=track['pid']
            # Amber only means unassigned inside an accepted box, never ground truth.
            inside=np.zeros(len(points),bool)
            for box in boxes:
                xy=points[:,:2]-box[:2];c,s=np.cos(box[6]),np.sin(box[6])
                local=xy@np.array([[c,-s],[s,c]])
                inside|=(abs(local)<=box[3:5]/2+.12).all(1)&(abs(points[:,2]-box[2])<=box[5]/2+.12)
            stamp_offset=float(entry['offset_s'])
            identity=(m['session_id'],m['generation'],m['frame_id'])
            if previous is None or stamp-previous[0]>.6 or previous[1]!=identity:
                clips.append(dict(start=len(frames),end=len(frames),start_s=stamp_offset,end_s=stamp_offset))
            clips[-1].update(end=len(frames),end_s=stamp_offset);previous=(stamp,identity)
            image=matched_image(images,stamp)
            frames.append(dict(id=name,stamp_s=stamp,t=stamp_offset,clip=len(clips)-1,snapshot_id=m['snapshot_id'],
                people_source=people['input_fingerprint'],local_identity=list(identity),count=len(points),
                xyz=encoded(points,'<f4'),owner=encoded(assigned,'<u2'),old=encoded(old,'u1'),inside=encoded(inside,'u1'),
                sensor=sensor.tolist(),forward=np.asarray(m['world_from_lidar'])[:3,0].tolist(),
                plane=plane,ground_status=ground.get('status'),tracks=tracks,rgb=image,
                counts=dict(current=int((owners>=0).sum()),previous=int(old.sum()),
                    shared=int((old&(owners>=0)).sum()),added=int((~old&(owners>=0)).sum()),removed=int((old&(owners<0)).sum()))))
    if not frames:raise ValueError('No captured frames')
    if rgb_run is not None:
        run=json.loads((Path(rgb_run)/'run.json').read_text())
        base=run['route_replay']['source']['source_start_s']
        if any(abs(f['stamp_s']-f['t']-base)>1e-5 for f in frames):
            raise ValueError('RGB source run and captured timeline differ')
        with tempfile.TemporaryDirectory(prefix='social-nav-person-rgb-') as directory:
            request=Path(directory)/'request.json';result=Path(directory)/'rgb.json'
            request.write_text(json.dumps(dict(bag=run['bag'],bag_fingerprint=run['playback']['bag_fingerprint'],
                stamps=[f['stamp_s'] for f in frames])))
            subprocess.run(['/usr/bin/python3','-B','-m','perception.environment.person_review',
                '--extract-rgb',str(request),'--output',str(result)],check=True)
            images=json.loads(result.read_text())
        for f in frames:f['rgb']=matched_image(images,f['stamp_s'])
    root=Path(__file__).resolve().parent
    provenance=dict(archive=str(archive),prefix=prefix,source_frames=sources,
        algorithm_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('person_objects.py','dynamic.py','layers.py')},config=config,
        rgb_bag_fingerprint=images.get('bag_fingerprint'),
        scope='current ownership on frozen mapper inputs; recorded old mask; no map replay or point-level truth')
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    data=dict(fingerprint=fingerprint,frames=frames,clips=clips,source=provenance)
    template=(root/'person_review.html').read_text()
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    if template.count('/*__DATA__*/')!=1:raise ValueError('Invalid viewer template')
    output.mkdir(parents=True)
    (output/'review.html').write_text(template.replace('/*__DATA__*/',payload))
    manifest=dict(fingerprint=fingerprint,frames=len(frames),points=sum(f['count'] for f in frames),
        owned_points=sum(f['counts']['current'] for f in frames),rgb_frames=sum(f['rgb'] is not None for f in frames),
        identities=len(identities),clips=clips,source=provenance)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return {k:v for k,v in manifest.items() if k!='source'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-archive')
    parser.add_argument('--capture-prefix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--config');parser.add_argument('--extract-rgb',help=argparse.SUPPRESS)
    images=parser.add_mutually_exclusive_group();images.add_argument('--rgb');images.add_argument('--rgb-run')
    args=parser.parse_args()
    if args.extract_rgb:extract_rgb(args.extract_rgb,args.output)
    else:
        if not args.capture_archive or not args.capture_prefix:parser.error('--capture-archive and --capture-prefix are required')
        print(json.dumps(build(args.capture_archive,args.capture_prefix,args.output,
            config_path=args.config,rgb=args.rgb,rgb_run=args.rgb_run),ensure_ascii=False))


if __name__=='__main__':main()
