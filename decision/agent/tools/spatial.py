"""Metric map views and objective coordinates; never generates target poses."""
from dataclasses import dataclass
import copy
import hashlib
import io
import math
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


def sample(grid, xy, transform=None):
    if transform is not None:
        xy = xy@transform[:2,:2].T+transform[:2,2]
    cell = np.floor((xy-grid.metadata['origin_xy'])/grid.metadata['resolution']).astype(int)
    data = grid.layers['occupancy']
    valid = (cell[:,0]>=0)&(cell[:,1]>=0)&(cell[:,0]<data.shape[1])&(cell[:,1]<data.shape[0])
    values = np.full(len(xy), -1, dtype=np.int8)
    values[valid] = data[cell[valid,1],cell[valid,0]]
    return values, valid


def render(catalog, global_grid, bounds, pose, *, local_grid=None, map_from_local=None, tracks=(), geometry=None):
    width, height = 896, 768
    canvas = Image.new('RGB', (width,height), '#f2f4f7')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 19)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    x0,y0,x1,y1 = map(float,bounds)
    scale = min(760/(x1-x0), 600/(y1-y0))
    w,h = max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    left,top = (width-w)//2,58+(600-h)//2
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
    def label(point,text,color,below=False):
        x,y = pixel(point)
        if not (left<=x<=left+w and top<=y<=top+h):
            return
        size = draw.textbbox((0,0),text,font=font)
        tx = min(max(x-35 if below else x+10,left+3),left+w-(size[2]-size[0])-4)
        ty = min(max(y+12 if below else y-23,top+2),top+h-24)
        box = draw.textbbox((tx,ty),text,font=font)
        draw.rectangle((box[0]-2,box[1]-2,box[2]+2,box[3]+2),fill='#ffffff')
        draw.text((tx,ty),text,fill=color,font=font)
    for i,region in enumerate(catalog.regions.values()):
        draw.line([pixel(p) for p in [*region['polygon'],region['polygon'][0]]],fill='#269255',width=3)
        label(region['polygon'][0],'R'+str(i+1),'#14723b')
    for i,point in enumerate(catalog.places.values()):
        if point.get('pose') is None:
            continue
        x,y = pixel(point['pose'])
        if left<=x<=left+w and top<=y<=top+h:
            draw.ellipse((x-6,y-6,x+6,y+6),fill='#bc3053')
            label(point['pose'],'P'+str(i+1),'#a41b3d')
    for i,landmark in enumerate(catalog.landmarks.values()):
        x,y = pixel(landmark['position'])
        if left<=x<=left+w and top<=y<=top+h:
            draw.rectangle((x-5,y-5,x+5,y+5),fill='#7040a0')
            label(landmark['position'],'L'+str(i+1),'#7040a0')
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
        draw.ellipse((x-7,y-7,x+7,y+7),fill='#ed8819')
        tip = pixel([pose[0]+.6*math.cos(pose[2]),pose[1]+.6*math.sin(pose[2])])
        draw.line([(x,y),tip],fill='#cc6600',width=4)
        label(pose,'ROBOT','#ad5200',below=True)
    clipped = Image.new('RGB',(width,height),'#f2f4f7')
    clipped.paste(canvas.crop((left,top,left+w,top+h)),(left,top))
    canvas = clipped
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((left,top,left+w,top+h),outline='#526274',width=1)
    for value in np.linspace(x0,x1,5):
        x,_ = pixel([value,y0])
        draw.text((x-24,top+h+8),f'{value:.1f}',font=small,fill='#223548')
    for value in np.linspace(y0,y1,5):
        _,y = pixel([x0,value])
        draw.text((max(2,left-55),y-10),f'{value:.1f}',font=small,fill='#223548')
    draw.text((18,10),'MAP XY metres | x right, y up | yaw CCW radians',font=font,fill='#162c43')
    draw.text((18,690),'Pale: static prior | blue: current free | dark: occupied | grey: unknown',font=small,fill='#263b51')
    draw.text((18,720),'Orange: robot | red: points / people | green: regions | purple: landmarks',font=small,fill='#263b51')
    buffer = io.BytesIO()
    canvas.save(buffer,format='PNG')
    return buffer.getvalue()


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
    def build(cls, catalog, inputs, bounds=None):
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
        images = (render(catalog,global_grid,full,pose,geometry=inputs.get('geometry')),
                  render(catalog,global_grid,bounds,pose,local_grid=local,map_from_local=transform,
                         tracks=tracks,geometry=inputs.get('geometry')))
        record = dict(catalog_fingerprint=catalog.fingerprint,map_condition=catalog.condition,
            frame_id=catalog.value['frame_id'],target_reference='base_link',units=dict(position='metres',angle='radians'),
            full_bounds=full,local_bounds=list(bounds),pose=copy.deepcopy(pose),
            source_stamp_s=inputs.get('stamp_s'),clock_domain=inputs.get('clock_domain'),
            global_binding=layer_binding(global_grid),local_binding=None if local is None else layer_binding(local),
            local_observation_id=None if scene is None else scene.record['observation_id'],
            map_from_local=None if transform is None else transform.tolist(),
            local_source='current_measurement' if local is not None else 'static_prior_only',
            annotations=catalog.annotations(),
            robot_footprint=None if inputs.get('geometry') is None else inputs['geometry'].footprint.tolist(),
            geometry_assumed=None if inputs.get('geometry') is None else inputs['geometry'].assumed,
            heading_reference='base_link_x',camera_extrinsics_status='not_provided',
            image_roles=['annotated_global_map','target_or_robot_local_map'],
            image_sha256=[hashlib.sha256(i).hexdigest() for i in images])
        record['view_id'] = fingerprint(record)
        return cls(record,images).verify()
