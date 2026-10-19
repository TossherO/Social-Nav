"""Bounded, pickle-free asset transport and shared remote identities (no ROS/Qt)."""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from .data import input_fingerprint, validate_review


FORMAT = 'social-nav-people-annotation'
FEATURES = ['xy-centers', 'npy-xyzi', 'png-rgb', 'jpeg-preview', 'review-cas', 'mutation-receipts', 'writer-lease']
MAX_BUNDLE = 64 * 1024 * 1024
MAX_POINTS = 2000000
MAX_JSON = 2 * 1024 * 1024
MAX_PIXELS = 4096 * 2160


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def public_source(source):
    return {key: (Path(value).name if key == 'path' else value) for key, value in source.items()
            if key in ('id', 'path', 'topic', 'rgb_topic', 'root_group', 'location_group', 'split', 'indexed', 'epoch', 'revision')}


def public_frame(frame):
    result = {key: copy.deepcopy(frame[key]) for key in
              ('id', 'source_id', 'ordinal', 'stamp_ns', 'record_ns', 'sensor_frame', 'fingerprint',
               'candidates', 'review', 'revision', 'epoch') if key in frame}
    result['model'] = {key: copy.deepcopy(value) for key, value in (frame.get('model') or {}).items()
                       if key in ('backend', 'checkpoint_sha256', 'config_sha256', 'point_cloud_range',
                                  'single_frame', 'input_sweeps', 'precision')}
    return result


def validate_payload(review, bounds, candidates):
    """Network input gets structural and candidate-reference checks as well as XY rules."""
    if not isinstance(review, dict) or set(review) != {'status', 'decisions', 'centers', 'note'}:
        raise ValueError('标注字段不完整或含未知字段')
    if not isinstance(review['note'], str) or len(review['note']) > 4096:
        raise ValueError('备注过长或不是文本')
    if not isinstance(review['centers'], list) or len(review['centers']) > 1000 or not isinstance(review['decisions'], dict):
        raise ValueError('中心／候选决定格式错误')
    valid = {str(i) for i in range(len(candidates or []))}
    if not set(review['decisions']).issubset(valid):
        raise ValueError('标注引用不存在的候选')
    for center in review['centers']:
        if (not isinstance(center, dict) or set(center) != {'id', 'xy', 'candidate'} or
                not isinstance(center['id'], str) or not 1 <= len(center['id']) <= 128 or
                center['candidate'] is not None and center['candidate'] not in valid):
            raise ValueError('人工中心格式或候选引用错误')
    validate_review(review, bounds)


def pack_assets(frame, points, image, image_ns, notice='', encoding='jpeg'):
    points = np.ascontiguousarray(points, dtype='<f4')
    if points.ndim != 2 or points.shape[1] != 4 or not 0 < len(points) <= MAX_POINTS:
        raise ValueError('点云不是有界的 N×4 XYZI')
    if not frame.get('fingerprint') or input_fingerprint(points, frame['source_id'], frame['id'],
            frame['stamp_ns'] / 1e9, frame['sensor_frame']) != frame['fingerprint']:
        raise ValueError('点云未绑定或原始输入指纹不匹配；请先完成本机预计算／绑定')
    raw = io.BytesIO()
    np.save(raw, points, allow_pickle=False)
    entries = {'points.npy': raw.getvalue()}
    manifest = dict(format=FORMAT, frame=public_frame(frame), image_ns=image_ns, notice=notice,
                    encoding=encoding, image=None)
    if image is not None:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.shape[0] * image.shape[1] > MAX_PIXELS:
            raise ValueError('RGB图像格式／尺寸错误')
        name = 'image.png' if encoding == 'png' else 'image.jpg'
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format='PNG' if encoding == 'png' else 'JPEG',
                                   **({'compress_level': 1} if encoding == 'png' else {'quality': 90}))
        entries[name] = stream.getvalue()
        manifest['image'] = dict(name=name, shape=list(image.shape), sha256=hashlib.sha256(entries[name]).hexdigest(),
                                 preview=encoding != 'png')
    manifest['assets_id'] = digest(dict(fingerprint=frame['fingerprint'], image_ns=image_ns,
                                      encoding=encoding, image=manifest['image']))
    entries['manifest.json'] = canonical(manifest).encode('utf-8')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, data in entries.items():
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED if name.endswith('.npy') else zipfile.ZIP_STORED)
    result = output.getvalue()
    if len(result) > MAX_BUNDLE:
        raise ValueError('资产超过传输预算')
    return result


def unpack_assets(payload, expected=None, expected_encoding=None):
    if len(payload) > MAX_BUNDLE:
        raise ValueError('响应超过资产预算')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if (len(set(names)) != len(names) or not {'manifest.json', 'points.npy'}.issubset(names) or
                not set(names).issubset({'manifest.json', 'points.npy', 'image.png', 'image.jpg'}) or
                sum(item.file_size for item in archive.infolist()) > MAX_BUNDLE or
                archive.getinfo('manifest.json').file_size > MAX_JSON):
            raise ValueError('资产条目或解压体积错误')
        manifest = json.loads(archive.read('manifest.json'))
        frame = manifest['frame']
        if manifest['format'] != FORMAT or expected is not None and frame['id'] != expected:
            raise ValueError('资产不属于请求的格式／帧')
        if expected_encoding is not None and manifest['encoding'] != expected_encoding:
            raise ValueError('图像编码与请求不一致')
        raw = io.BytesIO(archive.read('points.npy'))
        magic = np.lib.format.read_magic(raw)
        if magic != (1, 0):
            raise ValueError('不支持的点云NPY格式')
        position = raw.tell()
        if int.from_bytes(raw.read(2), 'little') > 10000:
            raise ValueError('点云数组头过长')
        raw.seek(position)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(raw)
        if (len(shape) != 2 or shape[1] != 4 or not 0 < shape[0] <= MAX_POINTS or fortran or dtype != np.dtype('<f4') or
                raw.getbuffer().nbytes - raw.tell() != shape[0] * 16):
            raise ValueError('点云数组形状、类型或长度错误')
        points = np.frombuffer(raw.read(), dtype='<f4').reshape(shape).copy()
        if input_fingerprint(points, frame['source_id'], frame['id'], frame['stamp_ns'] / 1e9,
                             frame['sensor_frame']) != frame['fingerprint']:
            raise ValueError('下载点云与原始输入绑定不一致')
        image, info = None, manifest['image']
        if info is not None:
            if info['name'] not in ('image.png', 'image.jpg') or info['name'] not in names:
                raise ValueError('图像条目错误')
            encoded = archive.read(info['name'])
            if hashlib.sha256(encoded).hexdigest() != info['sha256']:
                raise ValueError('图像内容身份错误')
            with Image.open(io.BytesIO(encoded)) as original:
                if original.width * original.height > MAX_PIXELS or [original.height, original.width, 3] != info['shape']:
                    raise ValueError('图像尺寸超限或不一致')
                image = np.asarray(original.convert('RGB')).copy()
    identity = digest(dict(fingerprint=frame['fingerprint'], image_ns=manifest['image_ns'],
                           encoding=manifest['encoding'], image=info))
    if identity != manifest['assets_id']:
        raise ValueError('资产身份不一致')
    return frame, points, image, manifest['image_ns'], manifest['notice']
