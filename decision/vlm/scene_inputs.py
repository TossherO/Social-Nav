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
EXTENT = (-2., 16., -9., 9.)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def implementation_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def renderer_identity():
    face = font()
    path = getattr(face, 'path', None)
    return dict(implementation=implementation_digest(), pillow=PIL.__version__,
                font_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else 'pillow-default')


def build_window(rows, current, asset_identity):
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
    positions = [xy for observation in observations for xy in observation[1] if xy is not None]
    xmin, xmax, ymin, ymax = EXTENT
    if positions:
        xmin, xmax = min(xmin, math.floor(min(p[0] for p in positions))-1), max(xmax, math.ceil(max(p[0] for p in positions))+1)
        limit = max(ymax, math.ceil(max(abs(p[1]) for p in positions))+1)
        ymin, ymax = -limit, limit
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


def render_observation(frame, rgb, mapping, elapsed, extent=EXTENT):
    """Neutral model image. Human group/target colors never enter this function."""
    canvas = Image.new('RGB', (1280, 544), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    if frame is None or rgb is None:
        draw.text((25, 30), 'MISSING OBSERVATION', fill='#a32323', font=font(24))
        return canvas
    image = Image.open(io.BytesIO(rgb)).convert('RGB')
    scale = min(800 / image.width, 512 / image.height)
    w, h = round(image.width * scale), round(image.height * scale)
    ox, oy = (800 - w) // 2, 24 + (512 - h) // 2
    canvas.paste(image.resize((w, h)), (ox, oy))
    draw = ImageDraw.Draw(canvas)
    xmin, xmax, ymin, ymax = extent
    def bev(x, y):
        return (800 + 480 * (ymax - y) / (ymax - ymin), 536 - 480 * (x - xmin) / (xmax - xmin))
    for x in range(math.ceil(xmin), math.floor(xmax) + 1):
        draw.line([bev(x, ymin), bev(x, ymax)], fill='#d4dce5')
    for y in range(math.ceil(ymin), math.floor(ymax) + 1):
        draw.line([bev(xmin, y), bev(xmax, y)], fill='#d4dce5')
    ex, ey = bev(0, 0)
    draw.ellipse((ex-4, ey-4, ex+4, ey+4), fill='#293d52')
    for person in frame['people']:
        if person['key'] not in mapping or not person.get('eligible'):
            continue
        pid = mapping[person['key']]
        b = person['rgb_box']
        rect = (ox+b[0]*scale, oy+b[1]*scale, ox+b[2]*scale, oy+b[3]*scale)
        draw.rectangle(rect, outline='#00bfd6', width=2)
        draw.text((rect[0], max(24, rect[1]-21)), f'P{pid}', fill='#00b9dd', font=font())
        x, y = bev(*person['box'][:2])
        if 800 <= x < 1280 and 56 <= y <= 536:
            draw.ellipse((x-5, y-5, x+5, y+5), fill='#008498')
            draw.text((x+6, y-9), f'P{pid}', fill='#163547', font=font())
    draw.text((8, 1), f't={elapsed:+.3f}s', fill='#213448', font=font())
    draw.text((805, 1), 'BEV [m]  x forward / y left', fill='#213448', font=font(16))
    return canvas
