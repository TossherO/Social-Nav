"""Shared scalar PointCloud2 decoding; NumPy only, including organized row padding."""
import hashlib

import numpy as np


def input_fingerprint(points, sequence_id, observation_id, stamp_s, sensor_frame):
    """Same observation binding for ROS, annotation and detector cache replay."""
    points = np.ascontiguousarray(points, dtype='<f4')
    digest = hashlib.sha256()
    digest.update(f'{sequence_id}\n{observation_id}\n{stamp_s:.17g}\n{sensor_frame}\n{points.shape}\n'.encode())
    digest.update(points)
    return digest.hexdigest()


def pointcloud_fields(message, selected):
    codes = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4', 8:'f8'}
    fields = {}
    for item in message.fields:
        if item.name in fields:
            raise ValueError('Duplicate PointCloud2 field')
        fields[item.name] = item
    if len(set(selected)) != len(selected) or any(name not in fields for name in selected):
        raise ValueError('Missing/duplicate requested PointCloud2 fields')
    byteorder = '>' if message.is_bigendian else '<'
    formats, offsets = [], []
    for name in selected:
        item = fields[name]
        if item.count != 1 or item.datatype not in codes:
            raise ValueError('Unsupported PointCloud2 field type/count')
        dtype = np.dtype(byteorder+codes[item.datatype])
        if item.offset < 0 or item.offset+dtype.itemsize > message.point_step:
            raise ValueError('PointCloud2 field exceeds point_step')
        formats.append(dtype)
        offsets.append(item.offset)
    if (message.width < 1 or message.height < 1 or message.point_step < 1
            or message.row_step < message.width*message.point_step
            or len(message.data) != message.row_step*message.height):
        raise ValueError('Invalid/empty PointCloud2 layout or payload')
    dtype = np.dtype({'names': list(selected), 'formats': formats, 'offsets': offsets,
                     'itemsize': message.point_step})
    rows = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.row_step, message.point_step))
    return {name: rows[name].reshape(-1) for name in selected}
