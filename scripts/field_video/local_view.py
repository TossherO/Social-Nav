"""Vehicle-centred perspective view of measured geometry and map context."""

from bisect import bisect_right
from functools import lru_cache
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from navigation.planning.footprint import VehicleGeometry
from perception.environment.geometry import distance_to_zero
from perception.environment.ground import GroundTracker

from .grid_archive import SceneGrids


ROOT = Path(__file__).resolve().parents[2]
BACKGROUND = (13, 23, 34)
PANEL = (16, 28, 41)
RULE = (53, 73, 90)
TEXT = (231, 241, 248)
MUTED = (166, 187, 203)
TRAIL = (91, 236, 211)
REFERENCE = (171, 195, 255)
GOAL = (255, 207, 132)
BLOCKED = (255, 131, 110)
HEIGHTS = ((65, 139, 152), (77, 185, 195), (105, 187, 249), (194, 190, 255))
X_MIN, X_MAX, Y_LIMIT = -4., 15., 9.


def _yaml(relative):
    return yaml.safe_load((ROOT / relative).read_text(encoding='utf-8'))


@lru_cache(maxsize=16)
def _font(size, bold=False):
    weight = 'Bold' if bold else 'Regular'
    return ImageFont.truetype(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc', size, index=2)


def _text_width(font, value):
    return round(font.getlength(value)) if hasattr(font, 'getlength') else font.getsize(value)[0]


def _to_world(xy, pose):
    c, s = math.cos(pose['yaw']), math.sin(pose['yaw'])
    return xy @ np.array(((c, s), (-s, c)), dtype=np.float32) + (pose['x'], pose['y'])


def _to_base(xy, pose):
    c, s = math.cos(pose['yaw']), math.sin(pose['yaw'])
    return (xy - (pose['x'], pose['y'])) @ np.array(((c, -s), (s, c)), dtype=np.float32)


def _same_segment(left, right):
    return (left['floor'], left.get('segment')) == (right['floor'], right.get('segment'))


class _Perspective:
    def __init__(self, size):
        self.position = np.array((-5.6, -1.28, 6.77))
        forward = np.array((4.2, 0.1, 0.)) - self.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0, 0, 1))
        right /= np.linalg.norm(right)
        self.axes = np.column_stack((right, np.cross(right, forward), forward))
        self.center = np.array((size[0] * .53, size[1] * .43))
        self.focal = size[0] * 1.03

    def __call__(self, xyz):
        camera = (xyz - self.position) @ self.axes
        depth = camera[:, 2]
        pixels = self.center + camera[:, :2] / np.maximum(depth[:, None], .2) * (self.focal, -self.focal)
        return pixels, depth

    def ground(self, xy, height=0.):
        pixels, _ = self(np.column_stack((xy, np.broadcast_to(height, len(xy)))))
        return np.rint(pixels).astype(np.int32)


class _Map:
    def __init__(self, config):
        self.origin = np.array(config['origin'][:2])
        yaw = config['origin'][2]
        c, s = math.cos(yaw), math.sin(yaw)
        self.axes = np.array(((c, -s), (s, c)))
        self.resolution = config['resolution']
        self.map_id = config.get('map_id')
        with Image.open(config['image']) as image:
            gray = np.asarray(image.convert('L'), dtype=np.float32) / 255
        probability = gray if config.get('negate', 0) else 1. - gray
        self.cells = np.ones(gray.shape, np.uint8)
        self.cells[probability < config.get('free_thresh', .196)] = 0
        self.cells[probability > config.get('occupied_thresh', .65)] = 2

    def at(self, world_xy):
        coordinates = (world_xy - self.origin) @ self.axes / self.resolution
        columns = np.floor(coordinates[..., 0]).astype(np.float32)
        rows = (self.cells.shape[0] - 1 - np.floor(coordinates[..., 1])).astype(np.float32)
        return cv2.remap(self.cells, columns, rows, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=1)


class LocalView:
    def __init__(self, data, size=(1200, 1000)):
        self.size = tuple(size)
        self.scale = self.size[0] / 1200
        self.project = _Perspective(size)
        self.poses = data.get('poses', [])
        self.pose_times = [p['t'] for p in self.poses]
        self.maps = {floor: _Map(config) for floor, config in data.get('maps', {}).items()}
        profile = _yaml('configs/wheelchair_vehicle.yaml')['geometry']
        self.footprint = np.array(profile['footprint'])
        self.extrinsics = np.array(profile['base_from_lidar'])
        self.env = _yaml('perception/environment/configs/environment.yaml')
        self.planning = _yaml('navigation/planning/configs/planning.yaml')
        geometry = VehicleGeometry(self.footprint, self.extrinsics, source='wheelchair_vehicle.yaml')
        self.scenes = SceneGrids(data.get('bag'), geometry, self.planning)
        self.res = float(self.env['resolution'])
        x = np.arange(X_MIN + self.res / 2, X_MAX, self.res, dtype=np.float32)
        y = np.arange(-Y_LIMIT + self.res / 2, Y_LIMIT, self.res, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.grid_xy = np.stack((xx, yy), axis=2)
        self.grid_shape = xx.shape
        source = np.float32(((0, 0), (len(y) - 1, 0), (0, len(x) - 1), (len(y) - 1, len(x) - 1)))
        # The raster spans the projected centres of the display grid cells.
        destination, _ = self.project(np.array(((x[0], y[0], 0), (x[0], y[-1], 0),
                                                (x[-1], y[0], 0), (x[-1], y[-1], 0))))
        self.homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        self.scale_bars = self.project.ground(np.array(((.6, 2.4), (.6, 3.4))))
        lines = [((x, -Y_LIMIT), (x, Y_LIMIT)) for x in range(int(X_MIN), int(X_MAX) + 1)]
        lines += [((X_MIN, y), (X_MAX, y)) for y in range(-int(Y_LIMIT), int(Y_LIMIT) + 1)]
        self.grid_lines = self.project.ground(np.reshape(lines, (-1, 2))).reshape((-1, 2, 2)).tolist()
        self.scan_key = None
        self.scan = (None, None)

    def _stroke(self, width):
        return max(1, round(width * self.scale))

    def _pose_at(self, stamp, pose, t):
        if abs(pose['t'] - stamp) < 1e-6:
            return pose
        index = bisect_right(self.pose_times, stamp)
        older = self.poses[index - 1] if index else None
        newer = self.poses[index] if index < len(self.poses) else None
        if (older is not None and newer is not None and _same_segment(older, pose)
                and _same_segment(newer, pose) and newer['t'] <= t
                and newer['t'] - older['t'] <= .45
                and math.hypot(newer['x'] - older['x'], newer['y'] - older['y']) < 1.):
            weight = (stamp - older['t']) / (newer['t'] - older['t'])
            turn = math.remainder(newer['yaw'] - older['yaw'], 2 * math.pi)
            return dict(older, x=older['x'] + weight * (newer['x'] - older['x']),
                        y=older['y'] + weight * (newer['y'] - older['y']), yaw=older['yaw'] + weight * turn)
        if older is not None and _same_segment(older, pose) and stamp - older['t'] <= .3:
            return older
        return None

    def _prepare_cloud(self, scan, frame, stamp, t):
        if scan is None or stamp is None or not 0 <= t - stamp <= .26:
            return None, None
        frame = (frame or '').lstrip('/')
        if frame not in ('rslidar', 'base_link'):
            return None, None
        key = (frame, stamp)
        if self.scan_key == key:
            return self.scan
        points = np.asarray(scan, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return None, None
        points = points[np.isfinite(points).all(axis=1)]
        points = points[(np.abs(points) < 30.).all(axis=1)]
        if frame == 'rslidar':
            points = points @ self.extrinsics[:3, :3].T + self.extrinsics[:3, 3]
        sensor = self.extrinsics[:3, 3]
        distance = np.linalg.norm(points[:, :2] - sensor[:2], axis=1)
        points = points[distance > .38].astype(np.float32)
        # A single independently fitted scan is suitable for display only.
        tracker = GroundTracker(dict(self.env['ground_tracking'], min_observations=1,
                                     max_extrapolation_error_m=.20))
        plane, _ = tracker.fit(points, sensor, self.env)
        if len(points):
            if plane is not None:
                points[:, 2] -= points[:, :2] @ plane[:2] + plane[2]
            else:
                points[:, 2] -= sensor[2]
            # Keep external returns and suppress echos inside the verified body outline.
            within_body = ((points[:, :2] >= self.footprint.min(axis=0))
                           & (points[:, :2] <= self.footprint.max(axis=0))).all(axis=1)
            points = points[~within_body]
        self.scan_key = key
        self.scan = points, plane
        return self.scan

    def _scan_hits(self, points, plane):
        hits = np.zeros(self.grid_shape, np.uint8)
        if points is None or plane is None:
            return hits.astype(bool)
        low = self.env['min_obstacle_height_m'] + self.env['ground_tolerance_m']
        selected = points[(points[:, 2] >= low) & (points[:, 2] <= self.env['max_height_m'])]
        xy = np.floor((selected[:, :2] - (X_MIN, -Y_LIMIT)) / self.res).astype(np.int32)
        inside = ((xy >= 0) & (xy < self.grid_shape)).all(axis=1)
        x, y = xy[inside].T
        counts = np.zeros(self.grid_shape, np.uint16)
        np.add.at(counts, (x, y), 1)
        supported = cv2.boxFilter(counts.astype(np.float32), -1, (3, 3), normalize=False)
        return (counts > 0) & (supported >= 4)

    def _ground(self, pose, points, plane, t, snapshot_not_before):
        floor = self.maps.get(pose['floor']) if pose else None
        cells = np.ones(self.grid_shape, np.uint8)
        source = '无关联地图'
        if floor is not None:
            world_xy = _to_world(self.grid_xy, pose)
            cells = floor.at(world_xy)
            source = '回放估算代价 · 静态地图'
            archive = self.scenes.sample(t, pose, floor.map_id, world_xy, not_before=snapshot_not_before)
        else:
            archive = None
        terrain = np.empty(self.grid_shape + (3,), np.uint8)
        terrain[:] = (25, 37, 50)
        terrain[cells == 2] = (75, 88, 105)
        if archive is not None:
            covered = archive >= -1
            cells[covered] = 1
            cells[archive == 0] = 0
            cells[archive == 100] = 2
            terrain[covered & (cells == 1)] = (25, 37, 50)
            terrain[covered & (cells == 2)] = (150, 99, 86)
            source = '现场导航栅格快照 · 外围静态图'
        if archive is None:
            hits = self._scan_hits(points, plane)
            if hits.any():
                terrain[hits] = (191, 103, 69)
                source = ('回放估算代价 · 静态地图＋当前激光' if floor is not None
                          else '当前激光回波 · 车体坐标')
            cells[hits] = 2
        free = cells == 0
        if free.any():
            # Same blocked/unknown distance and exponential fall-off as the
            # navigation clearance cost, using the OpenCV EDT implementation.
            distance = distance_to_zero(np.pad(free, 1))[1:-1, 1:-1] * self.res
            clearance = distance - math.sqrt(2.) * self.res - self.planning['safety_margin_m']
            cost = np.exp(-np.maximum(clearance, 0) / self.planning['clearance_cost_scale_m'])
            near, far = np.array((152, 88, 57)), np.array((37, 75, 86))
            terrain[free] = far + cost[free, None] * (near - far)
        pixels = cv2.warpPerspective(terrain, self.homography, self.size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND)
        for start, end in self.grid_lines:
            cv2.line(pixels, tuple(start), tuple(end), (61, 80, 93), self._stroke(1), cv2.LINE_AA)
        return pixels, source

    def _polyline(self, pixels, xy, color, *, dashed=False):
        if len(xy) < 2:
            return
        # Clip in 3D before perspective division to avoid lines behind the camera.
        left, right = xy[:-1].copy(), xy[1:].copy()
        _, a = self.project(np.c_[left, np.zeros(len(left))])
        _, b = self.project(np.c_[right, np.zeros(len(right))])
        keep = (a >= 1.) | (b >= 1.)
        left, right, a, b = left[keep], right[keep], a[keep], b[keep]
        near = a < 1.
        left[near] += ((1. - a[near]) / (b[near] - a[near]))[:, None] * (right[near] - left[near])
        near = b < 1.
        right[near] += ((1. - b[near]) / (a[near] - b[near]))[:, None] * (left[near] - right[near])
        first, second = self.project.ground(left, .04), self.project.ground(right, .04)
        segments = []
        for start, end in zip(first, second):
            visible, start, end = cv2.clipLine((0, 0, *self.size), tuple(start), tuple(end))
            if visible:
                segments.append((start, end))
        halo, width = self._stroke(8), self._stroke(4)
        for start, end in segments:
            cv2.line(pixels, start, end, BACKGROUND, halo, cv2.LINE_AA)
        phase = 0.
        period = 23. * self.scale
        for start, end in segments:
            if not dashed:
                cv2.line(pixels, start, end, color, width, cv2.LINE_AA)
                continue
            vector = np.subtract(end, start)
            length = np.hypot(*vector)
            if length > .1:
                for offset in np.arange(-phase, length, period):
                    until = min(length, offset + period * .6)
                    if until > max(0., offset):
                        p0 = tuple(np.rint(start + vector * max(0., offset) / length).astype(int))
                        p1 = tuple(np.rint(start + vector * until / length).astype(int))
                        cv2.line(pixels, p0, p1, color, width, cv2.LINE_AA)
                phase = (phase + length) % period

    def _cloud(self, pixels, points, plane):
        if points is None or not len(points):
            return
        limit = (points[:, 0] > X_MIN) & (points[:, 0] < X_MAX) & (np.abs(points[:, 1]) < Y_LIMIT)
        points = points[limit & (points[:, 2] > -.75) & (points[:, 2] < 3.1)]
        screen, depth = self.project(points)
        visible = ((screen > 1) & (screen < np.array(self.size) - 2)).all(axis=1) & (depth > 1.)
        points, screen, depth = points[visible], screen[visible], depth[visible]
        # Uniform projected bins reduce bright overdrawing of nearby surfaces.
        # A larger bin at floor level keeps map shading and paths legible.
        ground = (points[:, 2] < .12) if plane is not None else np.zeros(len(points), bool)
        step = np.where(ground, 6., 2.7)[:, None] * self.scale
        bins = np.floor(screen / step).astype(np.int32)
        keys = (bins[:, 0] << 13) + bins[:, 1] + (ground.astype(np.int32) << 27)
        nearest = np.argsort(depth)
        _, unique = np.unique(keys[nearest], return_index=True)
        indices = nearest[np.sort(unique)[::-1]]
        levels = np.digitize(points[:, 2], (.12, .75, 1.55) if plane is not None else (-.35, .25, .85))
        levels[ground] = len(HEIGHTS)
        palette = HEIGHTS + ((76, 130, 145),)
        positions = np.rint(screen[indices]).astype(int).tolist()
        radii = np.where(ground[indices] | (depth[indices] > 14.), self._stroke(1), self._stroke(2)).tolist()
        for position, level, radius in zip(positions, levels[indices].tolist(), radii):
            cv2.circle(pixels, tuple(position), radius, palette[level], -1, cv2.LINE_AA)

    def _vehicle(self, pixels):
        outline = self.project.ground(self.footprint)
        roof = self.project.ground(self.footprint, .5)
        cv2.fillConvexPoly(pixels, outline, (14, 24, 36), cv2.LINE_AA)
        for index, color in ((3, (36, 89, 107)), (0, (40, 111, 128))):
            other = (index + 1) % len(roof)
            face = np.array((outline[index], outline[other], roof[other], roof[index]))
            cv2.fillConvexPoly(pixels, face, color, cv2.LINE_AA)
            cv2.polylines(pixels, [face], True, (120, 197, 201), self._stroke(1), cv2.LINE_AA)
        cv2.fillConvexPoly(pixels, roof, (62, 144, 157), cv2.LINE_AA)
        cv2.polylines(pixels, [roof], True, (186, 244, 241), self._stroke(2), cv2.LINE_AA)
        arrow = self.project.ground(np.array(((.70, 0.), (.32, -.18), (.40, 0.), (.32, .18))), .51)
        cv2.fillPoly(pixels, [arrow], TEXT, cv2.LINE_AA)

    def _goals(self, pixels, route, pose):
        labels = []
        if route is not None and route.get('points') and route.get('floor') == pose['floor']:
            xy = _to_base(np.asarray(route['points']), pose)
            anchors, depth = self.project(np.c_[xy, np.zeros(len(xy))])
            for i, ((x, y), z) in enumerate(zip(anchors, depth)):
                if (i < route.get('waypoint_index', 0) or z < 1.
                        or not 24 * self.scale < x < self.size[0] - 24 * self.scale
                        or not self.size[1] * .12 < y < self.size[1] * .92):
                    continue
                color = BLOCKED if i == route.get('blocked_index') else GOAL
                p = (int(round(x)), int(round(y)))
                cv2.circle(pixels, p, self._stroke(9), BACKGROUND, -1, cv2.LINE_AA)
                cv2.circle(pixels, p, self._stroke(8), color, self._stroke(2), cv2.LINE_AA)
                cv2.circle(pixels, p, self._stroke(2), color, -1, cv2.LINE_AA)
                name = route['labels'][i]
                if i == route.get('blocked_index'):
                    name += '受阻'
                labels.append((p, name, color))
        return labels

    def _hud(self, pixels, pose, plane, cloud, source, labels, reference_name):
        result = Image.fromarray(pixels)
        draw = ImageDraw.Draw(result)
        def px(x, y):
            return round(x * self.scale), round(y * self.size[1] / 1000)

        def text(x, y, value, size, fill=TEXT, bold=False):
            draw.text(px(x, y), value, fill=fill, font=_font(max(10, self._stroke(size)), bold))

        def tag(x, y, value, fill):
            font = _font(self._stroke(22), True)
            left, top = px(x, y)
            width = _text_width(font, value)
            draw.rectangle((left, top, left + width + self._stroke(24), top + self._stroke(38)),
                           fill=PANEL, outline=RULE)
            draw.text((left + self._stroke(12), top + self._stroke(3)), value, font=font, fill=fill)

        tag(25, 66, source if pose else '无地图定位 · ' + source, TEXT)
        for (x, y), name, color in labels:
            font = _font(self._stroke(21), True)
            width = _text_width(font, name) + self._stroke(15)
            offset = self._stroke(14)
            left = x + offset if x + offset + width < self.size[0] else x - offset - width
            origin = left, y - self._stroke(16)
            draw.rectangle((*origin, origin[0] + width, origin[1] + self._stroke(30)), fill=PANEL)
            draw.text((origin[0] + self._stroke(7), origin[1]), name, font=font, fill=color)
        start, end = [tuple(p) for p in self.scale_bars]
        draw.line((start, end), fill=MUTED, width=self._stroke(2))
        for x, y in (start, end):
            draw.line(((x, y - self._stroke(4)), (x, y + self._stroke(4))), fill=MUTED, width=self._stroke(2))
        draw.text((end[0], end[1] - self._stroke(26)), '1 m', font=_font(self._stroke(17)), fill=MUTED)

        legend = ImageDraw.Draw(result, 'RGBA')
        left = 25
        width = 13 + 163 * cloud + 221 * (pose is not None)
        if reference_name:
            width += 36 + _text_width(_font(self._stroke(18)), reference_name) / self.scale
        if pose:
            width += 145
        if cloud or pose:
            legend.rectangle((*px(25, 943), *px(left + width, 977)), fill=(*PANEL, 229))
        if cloud:
            for i, color in enumerate(HEIGHTS):
                x, y = px(left + 12 + i * 14, 961)
                r = self._stroke(4)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            text(left + 66, 948, '离地高度' if plane is not None else '相对高度', 18, MUTED)
            left += 163
        if pose:
            draw.line((px(left + 12, 961), px(left + 33, 961)), fill=TRAIL, width=self._stroke(3))
            text(left + 40, 948, '实测轨迹', 18, MUTED)
            x, y = px(left + 126, 961)
            r = self._stroke(5)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=GOAL, width=self._stroke(2))
            text(left + 138, 948, '任务目标', 18, MUTED)
            left += 221
        if reference_name:
            for offset in ((0, 28),) if reference_name == '现场参考路径' else ((0, 10), (17, 28)):
                draw.line((px(left + offset[0], 961), px(left + offset[1], 961)),
                          fill=REFERENCE, width=self._stroke(3))
            text(left + 36, 948, reference_name, 18, MUTED)
            left += 47 + _text_width(_font(self._stroke(18)), reference_name) / self.scale
        if pose:
            draw.rectangle((*px(left + 1, 956), *px(left + 11, 966)), fill=(37, 75, 86))
            draw.rectangle((*px(left + 14, 956), *px(left + 24, 966)), fill=(152, 88, 57))
            text(left + 32, 948, '代价低 → 高', 18, MUTED)
        if not cloud:
            tag(25, 112, '无同时间对齐的激光点云', MUTED)
        return result

    def render(self, t, pose, trail, route, scan_xyz, scan_frame, scan_stamp, reference, *, snapshot_not_before):
        points, plane = self._prepare_cloud(scan_xyz, scan_frame, scan_stamp, t)
        if pose is not None and points is not None:
            scan_pose = self._pose_at(scan_stamp, pose, t)
            if scan_pose is None:
                points, plane = None, None
            elif scan_pose is not pose:
                points = points.copy()
                points[:, :2] = _to_base(_to_world(points[:, :2], scan_pose), pose)
        pixels, source = self._ground(pose, points, plane, t, snapshot_not_before)
        self._cloud(pixels, points, plane)
        labels = []
        reference_name = None
        if pose is not None:
            if trail:
                history = [(p['x'], p['y']) for p in trail if _same_segment(p, pose) and p['t'] <= pose['t']]
                history.append((pose['x'], pose['y']))
                self._polyline(pixels, _to_base(np.asarray(history), pose), TRAIL)
            if (reference and reference.get('floor') == pose['floor']
                    and reference['t'] <= t < reference['until']):
                archived = reference['source'] == 'archived'
                self._polyline(pixels, _to_base(np.asarray(reference['points']), pose), REFERENCE, dashed=not archived)
                reference_name = '现场参考路径' if archived else '静态图离线重算参考'
            labels = self._goals(pixels, route, pose)
        self._vehicle(pixels)
        return self._hud(pixels, pose, plane, points is not None and len(points) > 0, source, labels, reference_name)
