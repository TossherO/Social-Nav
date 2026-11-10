"""Metric map views and objective coordinates; never generates target poses."""
from dataclasses import dataclass
import copy
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from perception.environment.context import fingerprint, layer_binding


def se2(pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.array([[c,-s,pose[0]],[s,c,pose[1]],[0,0,1.]])


def extent(grid):
    m = grid.metadata
    x, y = m['origin_xy']
    h, w = grid.layers['occupancy'].shape
    return [x, y, x+w*m['resolution'], y+h*m['resolution']]


def display_extent(grid, annotations, *, extras=(), padding_m=1.):
    """Crop unknown-only borders without hiding fixed annotations or live poses.

    Return map coordinates, not a new raster or a new occupancy classification.
    An entirely unknown map retains its original extent.
    """
    asset=np.asarray(extent(grid),dtype=float)
    cells=grid.layers['occupancy'];known=(cells==0)|(cells==100)
    rows=np.flatnonzero(known.any(axis=1));cols=np.flatnonzero(known.any(axis=0))
    if len(rows) and len(cols):
        origin=np.asarray(grid.metadata['origin_xy']);res=grid.metadata['resolution']
        bounds=np.r_[origin+np.array([cols[0],rows[0]])*res-padding_m,
                     origin+np.array([cols[-1]+1,rows[-1]+1])*res+padding_m]
        bounds[:2]=np.maximum(bounds[:2],asset[:2]);bounds[2:]=np.minimum(bounds[2:],asset[2:])
        policy='known_occupancy_with_padding'
    else:
        bounds=asset.copy();policy='all_unknown_original_extent'
    fixed=[]
    for point in annotations.get('points',[]):
        if point.get('place_id')!='mission_start' and point.get('pose') is not None:
            fixed.append(point['pose'][:2])
    for region in annotations.get('regions',[]):fixed.extend(region['polygon'])
    def include(points):
        if not points:return False
        xy=np.asarray(points,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Invalid display positions')
        low,high=xy.min(0),xy.max(0)
        expand=bool((low<bounds[:2]).any() or (high>bounds[2:]).any())
        if expand:
            bounds[:2]=np.minimum(bounds[:2],low-padding_m)
            bounds[2:]=np.maximum(bounds[2:],high+padding_m)
        return expand
    annotation_expanded=include(fixed)
    live_expanded=include(list(extras))
    return dict(bounds=bounds.tolist(),asset_bounds=asset.tolist(),policy=policy,padding_m=padding_m,
                annotation_expanded=annotation_expanded,live_positions_expanded=live_expanded)


def spatial_source_binding(inputs):
    scene = inputs.get('scene')
    geometry = inputs.get('geometry')
    return fingerprint(dict(global_binding=layer_binding(inputs['global_grid']),
        scene_binding=None if scene is None else scene.record['input_fingerprint'],
        local_grid_binding=None if inputs.get('local_grid') is None else layer_binding(inputs['local_grid']),
        local_grid_map_from_local=inputs.get('local_grid_map_from_local'),
        pose=inputs.get('pose'),local_pose=inputs.get('local_pose'),stamp_s=inputs.get('stamp_s'),
        clock_domain=inputs.get('clock_domain'),
        footprint=None if geometry is None else geometry.footprint.tolist(),sensors=inputs.get('sensors')))


def sample(grid, xy, transform=None):
    if transform is not None:
        xy = xy@transform[:2,:2].T+transform[:2,2]
    cell = np.floor((xy-grid.metadata['origin_xy'])/grid.metadata['resolution']).astype(int)
    data = grid.layers['occupancy']
    valid = (cell[:,0]>=0)&(cell[:,1]>=0)&(cell[:,0]<data.shape[1])&(cell[:,1]<data.shape[0])
    values = np.full(len(xy), -1, dtype=np.int8)
    values[valid] = data[cell[valid,1],cell[valid,0]]
    return values, valid


def map_font(size):
    for path,index in (('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',2),
                       ('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',0),
                       ('C:/Windows/Fonts/msyh.ttc',0)):
        if Path(path).is_file():
            try:return ImageFont.truetype(path,size,index=index)
            except OSError:continue
    return ImageFont.truetype('DejaVuSans.ttf', size)


def wrap_map_text(text, font, width):
    if not text.strip():
        return []
    if font.getlength(text)<=width:
        return [text]
    lines, current = [], ''
    for char in text:
        if current and font.getlength(current+char)>width:
            lines.append(current.rstrip())
            current = ''
        current += char
    if current.strip():
        lines.append(current.rstrip())
    return lines


def reference_layout(bounds, captions, *, max_edge=2048):
    """Fit the metric map and readable text, keeping the longest edge bounded."""
    span_x,span_y = bounds[2]-bounds[0],bounds[3]-bounds[1]
    if not all(math.isfinite(v) and v>0 for v in (span_x,span_y)):
        raise ValueError('Reference map requires finite nonempty bounds')
    ratio = max_edge/1024
    font,small = map_font(round(12*ratio)),map_font(round(12*ratio))
    title = 'MAP XY metres | x right, y up | yaw CCW radians'
    minimum_width = min(max_edge,round(320*ratio))

    def text_layout(width):
        headers = wrap_map_text(title,font,width-36*ratio)
        lines = [line for caption in captions for line in wrap_map_text(caption,small,width-36*ratio)]
        top = round((34+20*len(headers))*ratio)
        legend_height = round((30+20*len(lines))*ratio)
        overhead = top+30*ratio+legend_height
        return headers,lines,top,legend_height,overhead

    # Start at full width. Portrait maps reduce width while retaining the
    # longest height and recomputing wrapping; a minimum width keeps text legible.
    width = max_edge
    full_width_layout = None
    while True:
        headers,lines,top,legend_height,overhead = text_layout(width)
        if overhead>=max_edge:
            if full_width_layout is not None:
                return full_width_layout
            raise ValueError('Reference map legend cannot fit within the image limit')
        if full_width_layout is None:
            full_width_layout = dict(size=(max_edge,max_edge),ratio=ratio,headers=headers,lines=lines,
                                     top=top,legend_height=legend_height)
        height = overhead+(width-136*ratio)*span_y/span_x
        if height<=max_edge:
            return dict(size=(width,math.ceil(height)),ratio=ratio,headers=headers,lines=lines,
                        top=top,legend_height=legend_height)
        fitted = max(minimum_width,min(width-1,math.floor((max_edge-overhead)*span_x/span_y+136*ratio)))
        if fitted>=width:
            return dict(size=(width,max_edge),ratio=ratio,headers=headers,lines=lines,
                        top=top,legend_height=legend_height)
        width = fitted


def semantic_label_layout(entries, font, frame, ratio):
    """Place point and region labels around nearby anchors with short leaders."""
    if not entries:return []
    left,top,right,bottom=frame
    anchors=np.asarray([e['anchor'] for e in entries],float)
    parents=list(range(len(entries)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i in range(len(entries)):
        for j in range(i):
            if np.linalg.norm(anchors[i]-anchors[j])<=55*ratio:
                parents[root(i)]=root(j)
    groups={}
    for i in range(len(entries)):groups.setdefault(root(i),[]).append(i)
    def crossed(a,b,c,d):
        def cross(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        return cross(a,b,c)*cross(a,b,d)<-1e-8 and cross(c,d,a)*cross(c,d,b)<-1e-8
    def hits_box(a,b,box):
        lo,hi=0.,1.
        for p,d,low,high in zip(a,(b[0]-a[0],b[1]-a[1]),box[:2],box[2:]):
            if abs(d)<1e-9:
                if p<low or p>high:return False
            else:
                t1,t2=(low-p)/d,(high-p)/d
                lo=max(lo,min(t1,t2));hi=min(hi,max(t1,t2))
                if lo>hi:return False
        return True
    def near_line(a,b,p):
        delta=np.asarray(b)-a;length=float(delta@delta)
        t=0. if length<1e-10 else np.clip((np.asarray(p)-a)@delta/length,0.,1.)
        return np.linalg.norm(np.asarray(p)-(np.asarray(a)+t*delta))<5*ratio
    result=[]
    # Top-to-bottom ordering preserves the spatial order of adjacent leaders.
    for index in sorted(range(len(entries)),key=lambda i:(anchors[i,1],anchors[i,0],i)):
        e=entries[index];x,y=anchors[index];size=font.getbbox(e['text']);tw=size[2]-size[0];th=size[3]-size[1]
        if tw>right-left-8:
            e=dict(e,text=e['text'].split(' ',1)[0]);size=font.getbbox(e['text']);tw=size[2]-size[0];th=size[3]-size[1]
        group=anchors[groups[root(index)]];low=group.min(0);high=group.max(0);mid=(low[0]+high[0])/2
        sides=('left','right') if x<mid else ('right','left')
        offsets=(0.,-1.,1.,-2.,2.,-3.,3.)
        region=e.get('region_bounds')
        if region is not None:
            low=np.minimum(low,region[:2]);high=np.maximum(high,region[2:])
        proposals=[]
        for side in sides:
            tx=low[0]-10*ratio-tw-size[0] if side=='left' else high[0]+10*ratio-size[0]
            for offset in offsets:
                proposals.append((tx,y-(size[1]+size[3])/2+offset*(th+5*ratio)))
        if region is not None:
            for side in ('top','bottom'):
                ty=region[1]-10*ratio-size[3] if side=='top' else region[3]+10*ratio-size[1]
                for offset in offsets:
                    proposals.append((x-(size[0]+size[2])/2+offset*(tw+5*ratio),ty))
        options=[]
        for tx,ty in proposals:
            tx1=min(max(tx,left+3-size[0]),right-4-size[2])
            ty1=min(max(ty,top+3-size[1]),bottom-4-size[3])
            box=(tx1+size[0],ty1+size[1],tx1+size[2],ty1+size[3])
            end=(max(box[0],min(x,box[2])),max(box[1],min(y,box[3])))
            penalty=0.
            for p in anchors:
                if box[0]-4*ratio<=p[0]<=box[2]+4*ratio and box[1]-4*ratio<=p[1]<=box[3]+4*ratio:penalty+=10000
            for j,p in enumerate(anchors):
                if j!=index and near_line((x,y),end,p):penalty+=3000
                other_region=entries[j].get('region_bounds')
                if j!=index and other_region is not None and not (
                        other_region[0]<=x<=other_region[2] and other_region[1]<=y<=other_region[3]):
                    if hits_box((x,y),end,other_region):penalty+=3000
            for old in result:
                other=old['box'];pad=3*ratio
                if box[0]<other[2]+pad and box[2]>other[0]-pad and box[1]<other[3]+pad and box[3]>other[1]-pad:penalty+=10000
                if crossed((x,y),end,old['anchor'],old['end']):penalty+=10000
                if hits_box((x,y),end,other) or hits_box(old['anchor'],old['end'],box):penalty+=5000
            penalty+=float(np.hypot(end[0]-x,end[1]-y))*.1
            options.append((penalty,dict(e,index=index,anchor=(x,y),end=end,box=box,text_xy=(tx1,ty1))))
        result.append(min(options,key=lambda item:item[0])[1])
    return result


def render(catalog, global_grid, bounds, pose, *, local_grid=None, map_from_local=None, tracks=(), geometry=None,
           size=(896,768), detailed=False, trajectory=(), local_view=False, sensors=None):
    annotations = catalog.annotations()
    legend = []
    if detailed:
        for kind in ('points','regions'):
            for item in annotations[kind]:
                if kind=='points' and item.get('pose') is None:
                    continue
                legend.append(item['display_id']+' '+item.get('name',item.get('description','')))
    captions = ['Pale: prior free | cyan: observed free | dark: occupied | grey: unknown',
                'Orange robot | blue trail | red P/people | green R']
    layout = reference_layout(bounds,[*captions,'    '.join(legend)]) if size is None else None
    width,height = layout['size'] if layout else size
    ratio = layout['ratio'] if layout else width/896
    canvas = Image.new('RGB', (width,height), '#f2f4f7')
    draw = ImageDraw.Draw(canvas)
    font = map_font(round((12 if detailed else 16)*ratio))
    label_font = map_font(max(10,round((10 if detailed else 14)*ratio)))
    small = map_font(round(12*ratio))
    lines = []
    for item in legend:
        candidate = (lines[-1]+'    '+item) if lines else item
        if lines and draw.textlength(candidate,font=small) <= width-50*ratio:
            lines[-1] = candidate
        else:
            lines.append(item)
    legend_height = layout['legend_height'] if layout else (80+len(lines)*24)*ratio
    x0,y0,x1,y1 = map(float,bounds)
    top = layout['top'] if layout else round(58*ratio)
    if local_view and sensors and (sensors.get('camera') or {}).get('view',{}).get('status') == 'approximate_pinhole':
        top = max(top,round(72*ratio))
    scale = min((width-136*ratio)/(x1-x0), (height-top-30*ratio-legend_height)/(y1-y0))
    w,h = max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    left = (width-w)//2
    yy,xx = np.mgrid[:h,:w]
    xy = np.column_stack((x0+(xx.ravel()+.5)/scale, y1-(yy.ravel()+.5)/scale))
    values,_ = sample(global_grid,xy)
    rgb = np.full((len(xy),3), [169,177,186], dtype=np.uint8)
    rgb[values==0] = [246,246,239]
    rgb[values==100] = [58,63,71]
    if local_grid is not None:
        inverse = None if map_from_local is None else np.linalg.inv(map_from_local)
        local,valid = sample(local_grid,xy,inverse)
        rgb[valid&(local==0)] = [208,239,243]
        rgb[valid&(local==100)] = [26,45,64]
        rgb[valid&(local==-1)] = [130,151,168]
    canvas.paste(Image.fromarray(rgb.reshape(h,w,3)), (left,top))
    def pixel(point):
        return (left+(point[0]-x0)*scale, top+(y1-point[1])*scale)
    if local_view and pose is not None and sensors and sensors.get('status') == 'configured':
        camera = sensors.get('camera') or {}
        view = camera.get('view') or {}
        if view.get('status') == 'approximate_pinhole':
            offset = np.asarray(camera['position_m'][:2])
            rotation = np.array([[math.cos(pose[2]),-math.sin(pose[2])],
                                 [math.sin(pose[2]),math.cos(pose[2])]])
            origin = np.asarray(pose[:2])+rotation@offset
            for yaw in view['edge_yaw_rad']:
                direction = pose[2]+yaw
                end = origin+2.5*np.array([math.cos(direction),math.sin(direction)])
                draw.line([pixel(origin),pixel(end)],fill='#7848bf',width=max(1,round(2*ratio)))
            cx,cy = pixel(origin)
            draw.ellipse((cx-3*ratio,cy-3*ratio,cx+3*ratio,cy+3*ratio),fill='#7848bf')
    occupied_labels = []
    def label(point,text,color,below=False):
        x,y = pixel(point)
        if not (left<=x<=left+w and top<=y<=top+h):
            return
        size = draw.textbbox((0,0),text,font=label_font)
        tw,th = size[2]-size[0],size[3]-size[1]
        if tw>w-8 and detailed:
            text = text.split(' ',1)[0]
            size = draw.textbbox((0,0),text,font=label_font)
            tw,th = size[2]-size[0],size[3]-size[1]
        candidates = [(x-35*ratio if below else x+10*ratio,y+12*ratio if below else y-23*ratio)]
        for step in (1,2,3):
            distance = step*(th+8*ratio)
            candidates += [(x+10*ratio,y-distance),(x+10*ratio,y+distance),
                           (x-tw-10*ratio,y-distance),(x-tw-10*ratio,y+distance)]
        for choice,(tx,ty) in enumerate(candidates):
            tx = min(max(tx,left+3),left+w-tw-4)
            ty = min(max(ty,top+2-size[1]),top+h-size[3]-3)
            box = draw.textbbox((tx,ty),text,font=label_font)
            if not any(box[0]<old[2]+4 and box[2]>old[0]-4 and box[1]<old[3]+4 and box[3]>old[1]-4
                       for old in occupied_labels):
                break
        occupied_labels.append(box)
        if choice>0:
            draw.line([(x,y),(max(box[0],min(x,box[2])),max(box[1],min(y,box[3])))],fill=color,width=max(1,round(ratio)))
        draw.text((tx,ty),text,fill=color,font=label_font,stroke_width=1,stroke_fill='#d4d8d9')
    def title(item):
        return item['display_id']+(' '+item.get('name',item.get('description','')) if detailed else '')
    for segment in trajectory:
        points = [pixel(p['pose']) for p in segment['samples']]
        if len(points)>1:
            draw.line(points,fill='#1778b5',width=max(2,round(2*ratio)))
        elif points:
            x,y = points[0]
            draw.ellipse((x-2*ratio,y-2*ratio,x+2*ratio,y+2*ratio),fill='#1778b5')
    semantic=[]
    for region in annotations['regions']:
        vertices=np.asarray([pixel(p) for p in region['polygon']])
        draw.line([*map(tuple,vertices),tuple(vertices[0])],fill='#269255',width=max(2,round(3*ratio)))
        anchor=tuple(vertices[0]);region_bounds=np.r_[vertices.min(0),vertices.max(0)]
        if left<=anchor[0]<=left+w and top<=anchor[1]<=top+h:
            semantic.append(dict(anchor=anchor,text=title(region),color='#14723b',point=False,region_bounds=region_bounds))
    for point in annotations['points']:
        if point.get('pose') is None:
            continue
        x,y = pixel(point['pose'])
        if left<=x<=left+w and top<=y<=top+h:
            semantic.append(dict(anchor=(x,y),text=title(point),color='#a41b3d',point=True))
    placed=semantic_label_layout(semantic,label_font,(left,top,left+w,top+h),ratio)
    for item in placed:
        draw.line([item['anchor'],item['end']],fill=item['color'],width=max(1,round(ratio)))
    for item in placed:
        x,y=item['anchor']
        if item['point']:
            radius=3.2*ratio;draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#bc3053')
        draw.text(item['text_xy'],item['text'],fill=item['color'],font=label_font,stroke_width=1,stroke_fill='#d4d8d9')
        occupied_labels.append(item['box'])
    for track in tracks:
        if not track.get('confirmed') or track.get('state') != 'observed':
            continue
        point = np.array(track['box'][:2])
        if map_from_local is not None:
            point = map_from_local[:2,:2]@point+map_from_local[:2,2]
        x,y = pixel(point)
        if left<=x<=left+w and top<=y<=top+h:
            r = .3*scale
            draw.ellipse((x-r,y-r,x+r,y+r),outline='#dd4168',width=3)
            label(point,'T'+str(track['track_id']),'#ac2149')
    if pose is not None:
        x,y = pixel(pose)
        if geometry is not None:
            draw.line([pixel(p) for p in [*geometry.polygon(pose),geometry.polygon(pose)[0]]],fill='#f08b19',width=3)
        draw.ellipse((x-7*ratio,y-7*ratio,x+7*ratio,y+7*ratio),fill='#ed8819')
        direction = np.array([math.cos(pose[2]),-math.sin(pose[2])])
        tip = np.array([x,y])+max(.6*scale,20*ratio)*direction
        draw.line([(x,y),tuple(tip)],fill='#cc6600',width=max(2,round(4*ratio)))
        rear = tip-8*ratio*direction
        side = 4*ratio*np.array([-direction[1],direction[0]])
        draw.polygon([tuple(tip),tuple(rear+side),tuple(rear-side)],fill='#cc6600')
        label(pose,'ROBOT','#ad5200',below=math.sin(pose[2])>=0)
    clipped = Image.new('RGB',(width,height),'#f2f4f7')
    left,top = int(left),int(top)
    clipped.paste(canvas.crop((left,top,left+w,top+h)),(left,top))
    canvas = clipped
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((left,top,left+w,top+h),outline='#526274',width=1)
    for value in np.linspace(x0,x1,5):
        x,_ = pixel([value,y0])
        draw.text((x-24*ratio,top+h+8*ratio),f'{value:.1f}',font=small,fill='#223548')
    for value in np.linspace(y0,y1,5):
        _,y = pixel([x0,value])
        draw.text((max(2,left-55*ratio),y-10*ratio),f'{value:.1f}',font=small,fill='#223548')
    headers = layout['headers'] if layout else ['MAP XY metres | x right, y up | yaw CCW radians']
    for i,line in enumerate(headers):
        draw.text((18*ratio,(10+i*(20 if layout else 24))*ratio),line,font=font,fill='#162c43')
    if local_view and local_grid is None:
        draw.text((18*ratio,34*ratio),'STATIC PRIOR ONLY | no current local grid',font=small,fill='#8b4c15')
    if local_view and sensors and (sensors.get('camera') or {}).get('view',{}).get('status') == 'approximate_pinhole':
        draw.text((18*ratio,48*ratio),'Purple: approximate camera FOV, not visibility',font=small,fill='#7848bf')
    footer = height-legend_height+15*ratio
    for i,line in enumerate(layout['lines'] if layout else [*captions,*lines]):
        draw.text((18*ratio,footer+i*(20 if layout else 24)*ratio),line,font=small,fill='#263b51')
    buffer = io.BytesIO()
    canvas.save(buffer,format='PNG')
    return buffer.getvalue()


@dataclass(frozen=True)
class ReferenceMap:
    record: dict
    image: bytes

    def verify(self):
        if (fingerprint({k:v for k,v in self.record.items() if k!='reference_map_id'}) != self.record['reference_map_id'] or
                hashlib.sha256(self.image).hexdigest() != self.record['image_sha256']):
            raise ValueError('Reference map changed after preparation')
        return self

    @classmethod
    def build(cls, catalog, grid):
        from .places import Catalog
        value = copy.deepcopy(catalog.value)
        # Task-local start is not a fixed annotation and would change the prefix.
        for point in value['places']:
            if point.get('place_id')=='mission_start':
                point['pose'] = None
                point.pop('task_origin',None)
        static = Catalog(value)
        if grid.metadata['frame_id'] != static.value['frame_id']:
            raise ValueError('Reference map and annotations use different frames')
        display=display_extent(grid,static.annotations());bounds=display['bounds']
        image = render(static,grid,bounds,None,size=None,detailed=True)
        with Image.open(io.BytesIO(image)) as picture:
            image_size = list(picture.size)
        record = dict(frame_id=static.value['frame_id'],units=dict(position='metres',angle='radians'),
                      map_condition='complete',bounds=bounds,asset_bounds=display['asset_bounds'],
                      display_crop=display,global_binding=layer_binding(grid),
                      annotations=static.annotations(),image_role='static_annotated_global_map',
                      image_sha256=hashlib.sha256(image).hexdigest(),image_size=image_size,
                      image_layout='map_aspect_and_wrapped_legend',max_image_edge=2048,
                      scope='static_prior_not_current_observation',task_start_in_current_context=True)
        record['reference_map_id'] = fingerprint(record)
        return cls(record,image).verify()


@dataclass(frozen=True)
class SpatialView:
    record: dict
    images: tuple

    def verify(self):
        expected = fingerprint({k:v for k,v in self.record.items() if k!='view_id'})
        if expected != self.record['view_id'] or [hashlib.sha256(i).hexdigest() for i in self.images] != self.record['image_sha256']:
            raise ValueError('Spatial view changed after binding')
        return self

    def current(self, stamp, max_age=3.):
        measured = self.record['source_stamp_s']
        return measured is None or 0 <= stamp-measured <= max_age

    @classmethod
    def build(cls, catalog, inputs, bounds=None, *, trajectory=()):
        catalog.require_complete()
        global_grid = inputs['global_grid']
        pose = inputs.get('pose')
        scene = inputs.get('scene')
        local = transform = None
        tracks = []
        if scene is not None:
            scene.verify()
            local = scene.geometry
            if local.metadata['frame_id'] != catalog.value['frame_id']:
                if pose is None or inputs.get('local_pose') is None:
                    raise ValueError('Local map needs a measured map transform')
                transform = se2(pose)@np.linalg.inv(se2(inputs['local_pose']))
            people = scene.record['people']
            if people['status']=='valid':
                tracks = people['data']['tracks']
        elif inputs.get('local_grid') is not None:
            local=inputs['local_grid']
            matrix=np.asarray(inputs.get('local_grid_map_from_local'),dtype=float)
            if matrix.shape != (4,4) or not np.isfinite(matrix).all():
                raise ValueError('Local occupancy requires its measured map transform')
            transform=matrix[np.ix_([0,1,3],[0,1,3])]
        extras=[]
        if pose is not None:extras.append(pose[:2])
        for segment in trajectory:
            extras.extend(p['pose'][:2] for p in segment['samples'])
        display=display_extent(global_grid,catalog.annotations(),extras=extras)
        full=display['bounds']
        if bounds is None:
            centre = np.array(pose[:2]) if pose is not None else (np.array(full[:2])+full[2:])/2
            bounds = [*(centre-3.),*(centre+3.)]
        if np.shape(bounds)!=(4,) or not np.isfinite(bounds).all() or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            raise ValueError('Expected finite nonempty metric crop bounds')
        if global_grid.metadata['frame_id'] != catalog.value['frame_id']:
            raise ValueError('Map view and annotation coordinate frames differ')
        images = (render(catalog,global_grid,full,pose,geometry=inputs.get('geometry'),size=(768,640),trajectory=trajectory),
                  render(catalog,global_grid,bounds,pose,local_grid=local,map_from_local=transform,
                         tracks=tracks,geometry=inputs.get('geometry'),local_view=True,sensors=inputs.get('sensors')))
        record = dict(catalog_fingerprint=catalog.fingerprint,map_condition=catalog.condition,
            frame_id=catalog.value['frame_id'],target_reference='base_link',units=dict(position='metres',angle='radians'),
            full_bounds=display['asset_bounds'],overview_bounds=full,display_crop=display,
            local_bounds=list(bounds),pose=copy.deepcopy(pose),input_source_binding=spatial_source_binding(inputs),
            source_stamp_s=inputs.get('stamp_s'),clock_domain=inputs.get('clock_domain'),
            global_binding=layer_binding(global_grid),local_binding=None if local is None else layer_binding(local),
            local_observation_id=inputs.get('local_grid_observation_id') if scene is None else scene.record['observation_id'],
            map_from_local=None if transform is None else transform.tolist(),
            local_source='current_measurement' if local is not None else 'static_prior_only',
            annotations=catalog.annotations(),
            robot_footprint=None if inputs.get('geometry') is None else inputs['geometry'].footprint.tolist(),
            geometry_assumed=None if inputs.get('geometry') is None else inputs['geometry'].assumed,
            heading_reference='base_link_x',camera_extrinsics_status=(inputs.get('sensors') or {}).get('status','not_provided'),
            camera_view=copy.deepcopy(((inputs.get('sensors') or {}).get('camera') or {}).get('view',dict(status='unavailable'))),
            trajectory=dict(segment_count=len(trajectory),sample_count=sum(len(s['samples']) for s in trajectory),
                            source_binding=fingerprint(trajectory),scope='measured_poses_only_not_planned_route'),
            image_roles=['trajectory_global_overview','target_or_robot_local_map'],
            image_sha256=[hashlib.sha256(i).hexdigest() for i in images])
        record['view_id'] = fingerprint(record)
        return cls(record,images).verify()
