"""Causal real-robot observation windows. No human annotation is accepted here."""
import hashlib
import io
import json
import math
from pathlib import Path
from functools import lru_cache

import PIL
from PIL import Image, ImageDraw, ImageFont


PROTOCOL = 'wheelchair_rel_front_rgb_bev'
OFFSETS_NS = (-1_000_000_000, -500_000_000, 0)
MODEL_SIZE = (1280, 576)
RGB_RECT = (0, 32, 736, 576)
BEV_RECT = (736, 32, 1280, 576)
BEV_SIZE = (544, 544)
BACKGROUND = '#f5f7fa'
PID_FONT_SIZE = 24
PERSON_RADIUS = 7
MODEL_BACKGROUND = 'full_height_single_scan'
CLOUD_HEIGHT_RANGE = (-1.5, 2.5)
CLOUD_HEIGHT_PALETTE = ((83,129,191),(61,151,155),(123,165,101),(197,157,91),(197,117,115))
BEV_HISTORY = '#477caa'
BEV_CURRENT = '#df701e'
BEV_EGO_HISTORY = '#697b8c'
TRAJECTORY_WIDTH = 3


def place_bev_labels(markers, sizes, bounds, *, marker_radius=8, obstacles=()):
    """Place neutral IDs near their markers, avoiding points and earlier labels."""
    def overlap(a, b):
        return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))

    occupied = []
    points = [(x-marker_radius, y-marker_radius, x+marker_radius, y+marker_radius) for _, x, y in markers]
    result = []
    for key, x, y in markers:
        width, height = sizes[key]
        candidates = []
        for gap in (9, 20, 34, 52, 76, 104):
            offsets = ((gap, -height/2), (-gap-width, -height/2), (-width/2, -gap-height), (-width/2, gap),
                       (gap, -gap-height), (-gap-width, -gap-height), (gap, gap), (-gap-width, gap))
            for dx, dy in offsets:
                left = max(bounds[0], min(x+dx, bounds[2]-width))
                top = max(bounds[1], min(y+dy, bounds[3]-height))
                rect = (left, top, left+width, top+height)
                distance = math.hypot(x-max(left, min(x, rect[2])), y-max(top, min(y, rect[3])))
                collision = sum(overlap(rect, other) for other in occupied+points+list(obstacles))
                candidates.append((collision*1000+distance+len(candidates)*.001, rect))
        rect = min(candidates, key=lambda value: value[0])[1]
        occupied.append((rect[0]-2, rect[1]-2, rect[2]+2, rect[3]+2))
        result.append((key, rect))
    return result


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def renderer_identity():
    face = font()
    path = getattr(face, 'path', None)
    return dict(implementation=implementation_digest(), pillow=PIL.__version__,
                trajectory_implementation=hashlib.sha256(Path(__file__).with_name('trajectory_inputs.py').read_bytes()).hexdigest(),
                font_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else 'pillow-default')


def model_eligible(person):
    """A current confirmed 3D track is valid independently of RGB coverage."""
    if person.get('kind') == 'track' and 'confirmed' in person and 'observed' in person:
        return bool(person.get('box') is not None and person['confirmed'] and person['observed'])
    # Old inputs without lifecycle evidence retain their recorded eligibility.
    return bool(person.get('eligible'))


def build_window(rows, current, asset_identity, *, extent):
    """Use the latest *past* sample within 150ms of each requested instant."""
    end = current['stamp_ns']
    frames = []
    for offset in OFFSETS_NS:
        candidates = [r for r in rows if r['stamp_ns'] <= end + offset]
        frame = max(candidates, key=lambda r: r['stamp_ns']) if candidates else None
        if frame and end + offset - frame['stamp_ns'] > 150_000_000:
            frame = None
        frames.append(frame)
    valid = sorted(p['key'] for p in current['people'] if p.get('eligible'))
    if len(valid) != len(set(valid)):
        raise ValueError('同一观察含重复源轨迹身份')
    if len(valid) > 20:
        raise ValueError('当前有效人物超过20；尚未定义中性选人规则，不能静默截断')
    mapping = {key: i for i, key in enumerate(valid)}
    observations = []
    for key, pid in mapping.items():
        positions, boxes, heights = [], [], []
        for frame in frames:
            person = next((p for p in frame['people'] if p['key'] == key and p.get('eligible')), None) if frame else None
            xy = person['box'][:2] if person else None
            box = person.get('rgb_box') if person else None
            positions.append([round(v, 3) for v in xy] if xy else None)
            boxes.append([round(v * 1000 / frame['image_size'][j % 2]) for j, v in enumerate(box)] if box else None)
            heights.append(round(box[3] - box[1]) if box else None)
        observations.append([pid, positions, boxes, heights])
    distances = []
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            values = [round(math.dist(x, y), 3) if x is not None and y is not None else None
                      for x, y in zip(a[1], b[1])]
            if any(v is not None and v <= 3.5 for v in values):
                distances.append([a[0], b[0], *values])
    xmin, xmax, ymin, ymax = checked_extent(extent)
    fields = dict(v=list(mapping.values()), dt=[(r['stamp_ns'] - end) / 1e9 if r else None for r in frames],
                  ts={'source': current.get('timestamp_source', 'sensor'), 'clock_domain': current.get('clock_domain')},
                  e=[xmin, xmax, ymin, ymax], o=observations, d=distances)
    ready = all(r and r.get('has_rgb') for r in frames)
    signature = dict(protocol=PROTOCOL, renderer=renderer_identity(), assets=asset_identity,
                     frames=[r['fingerprint'] if r else None for r in frames], mapping=mapping, fields=fields)
    return dict(protocol=PROTOCOL, frames=[r['id'] if r else None for r in frames], mapping=mapping,
                fields=fields, ready=ready, fingerprint=hashlib.sha256(canonical(signature).encode()).hexdigest())


@lru_cache(maxsize=8)
def font(size=18):
    for name in ('DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def checked_extent(extent):
    values = tuple(float(v) for v in extent)
    if (len(values) != 4 or not all(math.isfinite(v) for v in values) or
            values[1] <= values[0] or values[3] <= values[2]):
        raise ValueError('Expected finite increasing BEV XY bounds')
    return values


def rgb_geometry(size):
    width, height = size
    scale = min((RGB_RECT[2]-RGB_RECT[0])/width, (RGB_RECT[3]-RGB_RECT[1])/height)
    w, h = round(width*scale), round(height*scale)
    return scale, RGB_RECT[0]+(RGB_RECT[2]-RGB_RECT[0]-w)//2, RGB_RECT[1]+(RGB_RECT[3]-RGB_RECT[1]-h)//2, w, h


def bev_xy(x, y, extent):
    xmin, xmax, ymin, ymax = checked_extent(extent)
    scale = min(BEV_SIZE[0]/(ymax-ymin), BEV_SIZE[1]/(xmax-xmin))
    return ((BEV_RECT[0]+BEV_RECT[2])/2-scale*(y-(ymin+ymax)/2),
            (BEV_RECT[1]+BEV_RECT[3])/2-scale*(x-(xmin+xmax)/2))


def pointcloud_background(points, extent):
    """Full-height single scan, pale Z colors and 2px points; display only.

    Coordinates are this observation's LiDAR XYZ. No ground/person/height
    filter or sampling is applied. Same-pixel collisions keep the highest Z.
    Callers bind the original scan separately when creating derived inputs.
    """
    import numpy as np
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] not in (3,4):
        raise ValueError('Expected sensor-frame Nx3 or Nx4 point cloud')
    points = points[np.isfinite(points[:,:3]).all(axis=1),:3]
    xmin,xmax,ymin,ymax = checked_extent(extent)
    points = points[(points[:,0]>=xmin)&(points[:,0]<=xmax)&(points[:,1]>=ymin)&(points[:,1]<=ymax)]
    scale = min(BEV_SIZE[0]/(ymax-ymin),BEV_SIZE[1]/(xmax-xmin))
    x = np.floor(BEV_SIZE[0]/2-scale*(points[:,1]-(ymin+ymax)/2)).astype(int)
    y = np.floor(BEV_SIZE[1]/2-scale*(points[:,0]-(xmin+xmax)/2)).astype(int)
    x,y = np.clip(x,0,BEV_SIZE[0]-1),np.clip(y,0,BEV_SIZE[1]-1)
    heights = np.full(BEV_SIZE[::-1],-np.inf,np.float32)
    for dx in (-1,0):
        for dy in (-1,0):
            xx,yy = x+dx,y+dy
            inside = (xx>=0)&(xx<BEV_SIZE[0])&(yy>=0)&(yy<BEV_SIZE[1])
            np.maximum.at(heights,(yy[inside],xx[inside]),points[:,2][inside])
    observed = np.isfinite(heights)
    ramp = np.clip((heights[observed]-CLOUD_HEIGHT_RANGE[0])/(CLOUD_HEIGHT_RANGE[1]-CLOUD_HEIGHT_RANGE[0]),0,1)*4
    index = np.minimum(ramp.astype(int),3);weight = (ramp-index)[:,None]
    palette = np.asarray(CLOUD_HEIGHT_PALETTE)
    color = palette[index]*(1-weight)+palette[index+1]*weight
    bg = np.asarray(Image.new('RGB',BEV_SIZE,BACKGROUND)).copy()
    bg[observed] = np.rint(color*.45+np.array([245,247,250])*.55).astype(np.uint8)
    image = Image.fromarray(bg)
    image.info['color_encoding'] = 'lidar_z'
    return image


def draw_height_scale(canvas):
    """Fixed pale height/color key in the header, outside RGB and BEV content."""
    draw = ImageDraw.Draw(canvas)
    left, right, top, bottom = 365, 695, 3, 12
    colors = CLOUD_HEIGHT_PALETTE
    bg = (245,247,250)
    for x in range(left,right+1):
        value = (x-left)/(right-left)*(len(colors)-1)
        i = min(int(value),len(colors)-2);weight = value-i
        rgb = tuple(round((colors[i][k]*(1-weight)+colors[i+1][k]*weight)*.45+bg[k]*.55) for k in range(3))
        draw.line([(x,top),(x,bottom)],fill=rgb)
    draw.rectangle((left,top,right,bottom),outline='#aab7c3',width=1)
    draw.text((225,4),'LiDAR Z[m]',fill='#213448',font=font(18))
    low,high = CLOUD_HEIGHT_RANGE
    for i in range(len(colors)):
        x = left+(right-left)*i/(len(colors)-1)
        value = low+(high-low)*i/(len(colors)-1)
        label = ('≤' if i==0 else '≥' if i==len(colors)-1 else '')+f'{value:g}'
        face = font(20);box = draw.textbbox((0,0),label,font=face)
        draw.line([(x,bottom),(x,bottom+3)],fill='#526678',width=1)
        draw.text((x-(box[2]-box[0])/2,14-box[1]),label,fill='#213448',font=face)


def clip_segment(a, b, extent):
    """Clip an edge at view boundaries without joining across outside samples."""
    xmin, xmax, ymin, ymax = checked_extent(extent)
    dx, dy = b[0]-a[0], b[1]-a[1]
    lo, hi = 0., 1.
    for p, q in ((-dx,a[0]-xmin), (dx,xmax-a[0]), (-dy,a[1]-ymin), (dy,ymax-a[1])):
        if abs(p) < 1e-12:
            if q < 0:return None
        elif p < 0:lo = max(lo,q/p)
        else:hi = min(hi,q/p)
        if lo > hi:return None
    return ((a[0]+lo*dx,a[1]+lo*dy),(a[0]+hi*dx,a[1]+hi*dy))


def continuous_history(segments):
    """Merge shared endpoints across state changes, keeping true gaps separate.

    This is a display of tracked estimates, including predictions. Original
    observation flags remain unchanged in the input trajectory metadata.
    """
    run=[]
    for segment in segments:
        points=segment['points']
        if not points:continue
        if run and run[-1][:3]==points[0][:3]:run.extend(points[1:])
        else:
            if run:yield run
            run=list(points)
    if run:yield run


def stroke_history(layer, points, color, width=TRAJECTORY_WIDTH):
    """Antialiased round joins, bounded to the stroke's crop, without point glyphs."""
    if len(points)<2:return
    pad=math.ceil(width/2)+2
    left=max(BEV_RECT[0],math.floor(min(p[0] for p in points))-pad)
    top=max(BEV_RECT[1],math.floor(min(p[1] for p in points))-pad)
    right=min(BEV_RECT[2],math.ceil(max(p[0] for p in points))+pad)
    bottom=min(BEV_RECT[3],math.ceil(max(p[1] for p in points))+pad)
    if right<=left or bottom<=top:return
    factor=4
    mask=Image.new('L',((right-left)*factor,(bottom-top)*factor),0)
    path=[((x-left)*factor,(y-top)*factor) for x,y in points]
    ImageDraw.Draw(mask).line(path,fill=255,width=width*factor,joint='curve')
    mask=mask.resize((right-left,bottom-top),Image.Resampling.LANCZOS)
    layer.paste(color,(left,top,right,bottom),mask)


def draw_rgb_people(canvas, frame, mapping):
    """Shared model/browse RGB geometry, with larger contrast-backed neutral IDs."""
    draw = ImageDraw.Draw(canvas)
    if not frame.get('image_size'):return
    scale, ox, oy, w, h = rgb_geometry(frame['image_size'])
    markers, boxes = [], []
    for person in frame['people']:
        box = person.get('rgb_box')
        if not box or person['key'] not in mapping or not person.get('eligible'):continue
        rect = (max(ox,ox+box[0]*scale),max(oy,oy+box[1]*scale),
                min(ox+w-1,ox+box[2]*scale),min(oy+h-1,oy+box[3]*scale))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:continue
        draw.rectangle(rect, outline='#00bfd6', width=2)
        markers.append((mapping[person['key']],rect[0],rect[1]));boxes.append(rect)
    face = font(PID_FONT_SIZE)
    glyphs = {pid:draw.textbbox((0,0),f'P{pid}',font=face) for pid,_,_ in markers}
    sizes = {pid:(r[2]-r[0]+12,r[3]-r[1]+10) for pid,r in glyphs.items()}
    anchors = {pid:(x,y) for pid,x,y in markers}
    for pid,rect in place_bev_labels(markers,sizes,(RGB_RECT[0]+3,RGB_RECT[1]+3,RGB_RECT[2]-3,RGB_RECT[3]-3),
                                     marker_radius=0,obstacles=boxes):
        x,y = anchors[pid]
        draw.line([(x,y),(max(rect[0],min(x,rect[2])),max(rect[1],min(y,rect[3])))],fill='#00bfd6',width=2)
        draw.rounded_rectangle(rect,radius=4,fill='#173b4d',outline='#00bfd6',width=1)
        b = glyphs[pid]
        draw.text((rect[0]+6-b[0],rect[1]+5-b[1]),f'P{pid}',fill='white',font=face)


def render_observation(frame, rgb, mapping, elapsed, extent, trajectory=None, *, bev_background=None):
    """Neutral model image. Human group/target colors never enter this function."""
    checked_extent(extent)
    canvas = Image.new('RGB', MODEL_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    if frame is None or rgb is None:
        draw.text((25, 30), 'MISSING OBSERVATION', fill='#a32323', font=font(24))
        return canvas
    image = Image.open(io.BytesIO(rgb)).convert('RGB')
    _, ox, oy, w, h = rgb_geometry(image.size)
    canvas.paste(image.resize((w, h)), (ox, oy))
    draw_rgb_people(canvas,frame,mapping)
    # Raster clipping also prevents thick strokes at the boundary from
    # spilling into RGB or the header, beyond the geometric edge clipping.
    layer = Image.new('RGB',MODEL_SIZE,BACKGROUND)
    if bev_background is not None:
        if bev_background.mode != 'RGB' or bev_background.size != BEV_SIZE:
            raise ValueError('BEV background must match the current model layout')
        layer.paste(bev_background,BEV_RECT[:2])
    draw = ImageDraw.Draw(layer)
    xmin, xmax, ymin, ymax = extent
    def bev(x, y):
        return bev_xy(x,y,extent)
    def trail(points, color):
        run=[]
        def flush():
            stroke_history(layer,run,color)
            run.clear()
        for a,b in zip(points,points[1:]):
            clipped=clip_segment(a[1:3],b[1:3],extent)
            if clipped is None:
                flush()
                continue
            ca,cb=bev(*clipped[0]),bev(*clipped[1])
            if run and math.dist(run[-1],ca)>1e-6:flush()
            if not run:run.append(ca)
            run.append(cb)
        flush()
    available=trajectory and trajectory['status']=='available'
    if available:
        for segment in trajectory['ego']:trail(segment,BEV_EGO_HISTORY)
        for segments in trajectory['people'].values():
            for points in continuous_history(segments):trail(points,BEV_HISTORY)
    ex, ey = bev(0, 0)
    ahead=bev(1,0)
    draw.line([(ex,ey),ahead],fill='#293d52',width=3)
    draw.polygon([(ahead[0],ahead[1]-5),(ahead[0]-6,ahead[1]+7),(ahead[0]+6,ahead[1]+7)],fill='#293d52')
    markers=[(-1,ex,ey)]
    for person in frame['people']:
        if person['key'] not in mapping or not person.get('eligible'):
            continue
        pid = mapping[person['key']]
        x, y = bev(*person['box'][:2])
        if xmin<=person['box'][0]<=xmax and ymin<=person['box'][1]<=ymax:
            markers.append((pid,x,y))
    markers.sort()
    face=font(PID_FONT_SIZE)
    names={pid:'EGO' if pid==-1 else f'P{pid}' for pid,_,_ in markers}
    boxes={pid:draw.textbbox((0,0),names[pid],font=face) for pid,_,_ in markers}
    sizes={pid:(b[2]-b[0]+10,b[3]-b[1]+8) for pid,b in boxes.items()}
    ppm=min(BEV_SIZE[0]/(ymax-ymin),BEV_SIZE[1]/(xmax-xmin))
    bar_m=max((v for v in (.1,.2,.5,1,2,5,10,20) if v*ppm<=150),default=.1)
    placements=place_bev_labels(markers,sizes,(BEV_RECT[0]+4,BEV_RECT[1]+4,BEV_RECT[2]-4,BEV_RECT[3]-4),
                                marker_radius=PERSON_RADIUS+2)
    anchors={pid:(x,y) for pid,x,y in markers}
    for pid,(left,top,right,bottom) in placements:
        x,y=anchors[pid]
        draw.line([(x,y),(max(left,min(x,right)),max(top,min(y,bottom)))],fill='#9daab6',width=1)
    for pid,x,y in markers:
        draw.ellipse((x-PERSON_RADIUS,y-PERSON_RADIUS,x+PERSON_RADIUS,y+PERSON_RADIUS),
                     fill='#293d52' if pid==-1 else BEV_CURRENT)
    # All labels go on top of all trajectories, leaders and current points.
    for pid,rect in placements:
        draw.rounded_rectangle(rect,radius=4,fill='white',outline='#b7c4d1',width=1)
        b=boxes[pid]
        draw.text((rect[0]+5-b[0],rect[1]+4-b[1]),names[pid],fill='#17324d',font=face)
    canvas.paste(layer.crop(BEV_RECT),BEV_RECT[:2])
    draw = ImageDraw.Draw(canvas)
    by=25;bx=BEV_RECT[2]-14
    draw.line([(bx-bar_m*ppm,by),(bx,by)],fill='#526678',width=3)
    for px in (bx-bar_m*ppm,bx):draw.line([(px,by-5),(px,by+5)],fill='#526678',width=2)
    draw.text((bx-bar_m*ppm,0),f'{bar_m:g} m',fill='#213448',font=font(20))
    draw.text((8, 2), f'RGB   t={elapsed:+.2f}s', fill='#213448', font=font(24))
    draw.text((BEV_RECT[0]+8, 3),'BEV [m] / x forward, y left' if available else 'BEV [m] / NO HISTORY',
              fill='#213448',font=font(22))
    if bev_background is not None and bev_background.info.get('color_encoding')=='lidar_z':
        draw_height_scale(canvas)
    return canvas
