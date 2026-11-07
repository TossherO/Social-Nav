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


def spatial_source_binding(inputs):
    scene = inputs.get('scene')
    geometry = inputs.get('geometry')
    return fingerprint(dict(global_binding=layer_binding(inputs['global_grid']),
        scene_binding=None if scene is None else scene.record['input_fingerprint'],
        pose=inputs.get('pose'),local_pose=inputs.get('local_pose'),stamp_s=inputs.get('stamp_s'),
        clock_domain=inputs.get('clock_domain'),
        footprint=None if geometry is None else geometry.footprint.tolist()))


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
    for path in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                 'C:/Windows/Fonts/msyh.ttc'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
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
    font,small = map_font(round(15*ratio)),map_font(round(16*ratio))
    title = 'MAP XY metres | x right, y up | yaw CCW radians'
    minimum_width = min(max_edge,round(320*ratio))

    def text_layout(width):
        headers = wrap_map_text(title,font,width-36*ratio)
        lines = [line for caption in captions for line in wrap_map_text(caption,small,width-36*ratio)]
        top = round((34+24*len(headers))*ratio)
        legend_height = round((32+24*len(lines))*ratio)
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


def render(catalog, global_grid, bounds, pose, *, local_grid=None, map_from_local=None, tracks=(), geometry=None,
           size=(896,768), detailed=False, trajectory=(), local_view=False):
    annotations = catalog.annotations()
    legend = []
    if detailed:
        for kind in ('points','regions','landmarks'):
            for item in annotations[kind]:
                if kind=='points' and item.get('pose') is None:
                    continue
                legend.append(item['display_id']+' '+item.get('name',item.get('description','')))
    captions = ['Pale: prior free | cyan: observed free | dark: occupied | grey: unknown',
                'Orange robot | blue trail | red P/people | green R | purple L']
    layout = reference_layout(bounds,[*captions,'    '.join(legend)]) if size is None else None
    width,height = layout['size'] if layout else size
    ratio = layout['ratio'] if layout else width/896
    canvas = Image.new('RGB', (width,height), '#f2f4f7')
    draw = ImageDraw.Draw(canvas)
    font = map_font(round((15 if detailed else 19)*ratio))
    small = map_font(round(16*ratio))
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
    occupied_labels = []
    def label(point,text,color,below=False):
        x,y = pixel(point)
        if not (left<=x<=left+w and top<=y<=top+h):
            return
        size = draw.textbbox((0,0),text,font=font)
        tw,th = size[2]-size[0],size[3]-size[1]
        if tw>w-8 and detailed:
            text = text.split(' ',1)[0]
            size = draw.textbbox((0,0),text,font=font)
            tw,th = size[2]-size[0],size[3]-size[1]
        candidates = [(x-35*ratio if below else x+10*ratio,y+12*ratio if below else y-23*ratio)]
        for step in (1,2,3):
            distance = step*(th+8*ratio)
            candidates += [(x+10*ratio,y-distance),(x+10*ratio,y+distance),
                           (x-tw-10*ratio,y-distance),(x-tw-10*ratio,y+distance)]
        for choice,(tx,ty) in enumerate(candidates):
            tx = min(max(tx,left+3),left+w-tw-4)
            ty = min(max(ty,top+2-size[1]),top+h-size[3]-3)
            box = draw.textbbox((tx,ty),text,font=font)
            if not any(box[0]<old[2]+4 and box[2]>old[0]-4 and box[1]<old[3]+4 and box[3]>old[1]-4
                       for old in occupied_labels):
                break
        occupied_labels.append(box)
        if choice>0:
            draw.line([(x,y),(max(box[0],min(x,box[2])),max(box[1],min(y,box[3])))],fill=color,width=max(1,round(ratio)))
        draw.rectangle((box[0]-2,box[1]-2,box[2]+2,box[3]+2),fill='#ffffff')
        draw.text((tx,ty),text,fill=color,font=font)
    def title(item):
        return item['display_id']+(' '+item.get('name',item.get('description','')) if detailed else '')
    for segment in trajectory:
        points = [pixel(p['pose']) for p in segment['samples']]
        if len(points)>1:
            draw.line(points,fill='#1778b5',width=max(2,round(2*ratio)))
        elif points:
            x,y = points[0]
            draw.ellipse((x-2*ratio,y-2*ratio,x+2*ratio,y+2*ratio),fill='#1778b5')
    for region in annotations['regions']:
        draw.line([pixel(p) for p in [*region['polygon'],region['polygon'][0]]],fill='#269255',width=max(2,round(3*ratio)))
        label(region['polygon'][0],title(region),'#14723b')
    for point in annotations['points']:
        if point.get('pose') is None:
            continue
        x,y = pixel(point['pose'])
        if left<=x<=left+w and top<=y<=top+h:
            radius = 6*ratio
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#bc3053')
            label(point['pose'],title(point),'#a41b3d')
    for landmark in annotations['landmarks']:
        x,y = pixel(landmark['position'])
        if left<=x<=left+w and top<=y<=top+h:
            radius = 5*ratio
            draw.rectangle((x-radius,y-radius,x+radius,y+radius),fill='#7040a0')
            label(landmark['position'],title(landmark),'#7040a0')
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
        draw.text((18*ratio,(10+i*24)*ratio),line,font=font,fill='#162c43')
    if local_view and local_grid is None:
        draw.text((18*ratio,34*ratio),'STATIC PRIOR ONLY | no current local grid',font=small,fill='#8b4c15')
    footer = height-legend_height+15*ratio
    for i,line in enumerate(layout['lines'] if layout else [*captions,*lines]):
        draw.text((18*ratio,footer+i*24*ratio),line,font=small,fill='#263b51')
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
        bounds = extent(grid)
        image = render(static,grid,bounds,None,size=None,detailed=True)
        with Image.open(io.BytesIO(image)) as picture:
            image_size = list(picture.size)
        record = dict(frame_id=static.value['frame_id'],units=dict(position='metres',angle='radians'),
                      map_condition='complete',bounds=bounds,global_binding=layer_binding(grid),
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
        full = extent(global_grid)
        if bounds is None:
            centre = np.array(pose[:2]) if pose is not None else (np.array(full[:2])+full[2:])/2
            bounds = [*(centre-3.),*(centre+3.)]
        if np.shape(bounds)!=(4,) or not np.isfinite(bounds).all() or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
            raise ValueError('Expected finite nonempty metric crop bounds')
        if global_grid.metadata['frame_id'] != catalog.value['frame_id']:
            raise ValueError('Map view and annotation coordinate frames differ')
        images = (render(catalog,global_grid,full,pose,geometry=inputs.get('geometry'),size=(768,640),trajectory=trajectory),
                  render(catalog,global_grid,bounds,pose,local_grid=local,map_from_local=transform,
                         tracks=tracks,geometry=inputs.get('geometry'),local_view=True))
        record = dict(catalog_fingerprint=catalog.fingerprint,map_condition=catalog.condition,
            frame_id=catalog.value['frame_id'],target_reference='base_link',units=dict(position='metres',angle='radians'),
            full_bounds=full,local_bounds=list(bounds),pose=copy.deepcopy(pose),input_source_binding=spatial_source_binding(inputs),
            source_stamp_s=inputs.get('stamp_s'),clock_domain=inputs.get('clock_domain'),
            global_binding=layer_binding(global_grid),local_binding=None if local is None else layer_binding(local),
            local_observation_id=None if scene is None else scene.record['observation_id'],
            map_from_local=None if transform is None else transform.tolist(),
            local_source='current_measurement' if local is not None else 'static_prior_only',
            annotations=catalog.annotations(),
            robot_footprint=None if inputs.get('geometry') is None else inputs['geometry'].footprint.tolist(),
            geometry_assumed=None if inputs.get('geometry') is None else inputs['geometry'].assumed,
            heading_reference='base_link_x',camera_extrinsics_status='not_provided',
            trajectory=dict(segment_count=len(trajectory),sample_count=sum(len(s['samples']) for s in trajectory),
                            source_binding=fingerprint(trajectory),scope='measured_poses_only_not_planned_route'),
            image_roles=['trajectory_global_overview','target_or_robot_local_map'],
            image_sha256=[hashlib.sha256(i).hexdigest() for i in images])
        record['view_id'] = fingerprint(record)
        return cls(record,images).verify()
