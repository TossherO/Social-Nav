"""Remove selected map-frame volumes from every ROLL keyframe in a new map."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml

from perception.environment.storage import asset_reference, content_hash
from perception.visualization.edit_geometry import polygon_mask, validate_polygon
from .results import map_fingerprint


def _poses(directory):
    path = Path(directory)/'keyframes/poses.txt'
    rows = np.loadtxt(path, ndmin=2)
    if (not len(rows) or rows.shape[1] not in (7, 8, 9) or not np.isfinite(rows).all()):
        raise ValueError('ROLL 地图关键帧位姿无效')
    offset = 1 if rows.shape[1] == 9 else 0
    if not np.array_equal(rows[:, offset+6], np.arange(len(rows))):
        raise ValueError('ROLL 地图关键帧编号不连续')
    return rows, offset


def source_binding(directory):
    directory = Path(directory).resolve()
    fingerprint = map_fingerprint(directory)
    rows, _ = _poses(directory)
    manifest = yaml.safe_load((directory/'manifest.yaml').read_text())
    if manifest.get('format') != 'roll_keyframes_v1' or manifest.get('keyframes') != len(rows):
        raise ValueError('仅支持完整 ROLL 关键帧地图')
    digest = hashlib.sha256()
    for index in range(len(rows)):
        for kind in ('corner', 'surf'):
            file = directory/'keyframes'/f'{kind}{index}.pcd'
            digest.update(f'{kind}{index}.pcd\0'.encode())
            digest.update(content_hash(file).encode())
    return dict(map_fingerprint=fingerprint, keyframes_sha256=digest.hexdigest())


def read_xyzi(path):
    """Read the PCL XYZI keyframe format without losing fields or sampling."""
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            if stream.tell() > 65536:
                raise ValueError('PCD 文件头过长')
            line = stream.readline()
            if not line:
                raise ValueError('PCD 文件不完整')
            words = line.decode('ascii').strip().split()
            if words and not words[0].startswith('#'):
                header[words[0]] = words[1:]
                if words[0] == 'DATA':
                    break
        if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity'] or
                header.get('SIZE') != ['4']*4 or header.get('TYPE') != ['F']*4 or
                header.get('COUNT', ['1']*4) != ['1']*4):
            raise ValueError('仅支持完整的 PCL XYZI 浮点关键帧：'+str(path))
        count = int(header['POINTS'][0])
        if (count < 0 or count > 8_000_000 or
                int(header['WIDTH'][0]) * int(header['HEIGHT'][0]) != count):
            raise ValueError('关键帧点数无效或超出预算')
        if header['DATA'] == ['binary']:
            raw = stream.read()
            if len(raw) != count*16:
                raise ValueError('PCD 二进制长度与点数不一致：'+str(path))
            points = np.frombuffer(raw, dtype='<f4').reshape(count, 4).copy()
        elif header['DATA'] == ['ascii']:
            points = np.loadtxt(stream, dtype=np.float32, ndmin=2) if count else np.empty((0, 4), np.float32)
            if points.shape != (count, 4):
                raise ValueError('PCD 文本长度与点数不一致：'+str(path))
        else:
            raise ValueError('不支持压缩或未知 PCD 格式：'+str(path))
    if not np.isfinite(points).all():
        raise ValueError('关键帧包含非有限坐标或强度：'+str(path))
    return points


def write_xyzi(path, points):
    points = np.ascontiguousarray(points, dtype='<f4')
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError('输出关键帧点云无效')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
              'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n'
              'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n') % (len(points), len(points))
    with Path(path).open('xb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(points.tobytes())


def _world(points, pose):
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    return points[:, :3].astype(float) @ rotation.T + [x, y, z]


def _remove(world, selections):
    selected = np.zeros(len(world), bool)
    for item in selections:
        polygon = validate_polygon(item['vertices_xy'])
        low, high = item['z_min_m'], item['z_max_m']
        if not np.isfinite([low, high]).all() or low > high:
            raise ValueError('三维选区高度范围无效')
        box = ((world[:, :2] >= polygon.min(axis=0)).all(axis=1) &
               (world[:, :2] <= polygon.max(axis=0)).all(axis=1) &
               (world[:, 2] >= low) & (world[:, 2] <= high))
        indices = np.flatnonzero(box & ~selected)
        selected[indices] = polygon_mask(world[indices, :2], polygon)
    return selected


def _voxel(points, resolution):
    if len(points) > 8_000_000:
        raise ValueError('地图点数超过显示点云重建预算')
    keys = np.floor(points[:, :3]/resolution).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    output = np.column_stack([np.bincount(inverse, weights=points[:, i])/counts for i in range(4)])
    return output.astype(np.float32)


def save_manual_map(source, output, selections, *, expected_binding):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('输出目录已存在，不能覆盖')
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError('输出不能位于源地图内部')
    if not selections:
        raise ValueError('没有需要删除的三维选区')
    for item in selections:
        validate_polygon(item['vertices_xy'])
        if not np.isfinite([item['z_min_m'], item['z_max_m']]).all() or item['z_min_m'] > item['z_max_m']:
            raise ValueError('三维选区高度范围无效')
    if source_binding(source) != expected_binding:
        raise ValueError('源定位地图已改变，请重新打开编辑器')
    rows, offset = _poses(source)
    manifest = yaml.safe_load((source/'manifest.yaml').read_text())
    resolution = manifest.get('visualization_resolution', .1)
    if not isinstance(resolution, (float, int)) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('地图显示分辨率无效')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=output.parent) as temp:
        target = Path(temp)/'map'; (target/'keyframes').mkdir(parents=True)
        shutil.copyfile(source/'keyframes/poses.txt', target/'keyframes/poses.txt')
        world_parts = []
        removed = dict(corner=0, surf=0)
        remaining = 0
        for index, row in enumerate(rows):
            for kind in ('corner', 'surf'):
                cloud = read_xyzi(source/'keyframes'/f'{kind}{index}.pcd')
                world = _world(cloud, row[offset:offset+6])
                mask = _remove(world, selections)
                filtered = cloud[~mask]
                if kind == 'surf' and not len(filtered):
                    raise ValueError(f'选区会清空表面关键帧 {index}；请缩小选区')
                write_xyzi(target/'keyframes'/f'{kind}{index}.pcd', filtered)
                world_parts.append(np.c_[world[~mask], filtered[:, 3]])
                removed[kind] += int(mask.sum())
                remaining += len(filtered)
                if remaining > 8_000_000:
                    raise ValueError('地图点数超过显示点云重建预算')
        if not sum(removed.values()):
            raise ValueError('选区没有命中任何关键帧点')
        display = _voxel(np.concatenate(world_parts), float(resolution))
        write_xyzi(target/'GlobalMap.pcd', display)
        record = dict(source=asset_reference(source), source_binding=expected_binding,
                      selections=selections, removed_points=removed,
                      global_map='rebuilt_from_edited_keyframes', frame=manifest['frame'])
        (target/'manual_edit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
        manifest = dict(manifest, visualization_resolution=float(resolution), reference_bundle=False,
                        manual_edit_sha256=content_hash(target/'manual_edit.json'),
                        manual_edit_source='keyframe_point_removal')
        (target/'manifest.yaml').write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))
        (target/'COMPLETE').write_text('roll_keyframes_v1\n')
        if source_binding(source) != expected_binding:
            raise ValueError('源定位地图在编辑过程中发生变化')
        if output.exists() or output.is_symlink():
            raise FileExistsError('输出目录已出现，不能覆盖')
        target.rename(output)
    return record
