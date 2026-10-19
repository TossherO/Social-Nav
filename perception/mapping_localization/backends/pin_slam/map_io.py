"""Complete PIN neural maps: tensor-only payload, portable config, atomic completion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

SOURCE_COMMIT = 'e2b7fbf328e5fac1d638e217d70a69cbe9f30a52'
FORMAT = 'pin_slam_neural_map'
POINT_FIELDS = ('neural_points', 'point_orientations', 'geo_features',
                'point_ts_create', 'point_ts_update', 'point_certainties')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def portable_config(config):
    excluded = {'run_path', 'output_root', 'pc_path', 'pose_path', 'calib_path',
                'label_path', 'model_path', 'sensor_cad_path'}
    result = {}
    for name, value in vars(config).items():
        if name not in excluded and (value is None or type(value) in (str, bool, int, float, list)):
            result[name] = value
    return result


def save_map(path, neural, decoder, config, metadata):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('Map destination already exists')
    if neural.count() == 0:
        raise ValueError('Cannot complete an empty neural map')
    neural.assign_local_to_global()
    payload = {name: getattr(neural, name).detach().cpu().clone() for name in POINT_FIELDS}
    payload.update(decoder={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                   encoder_geo={k: v.detach().cpu() for k, v in neural.position_encoder_geo.state_dict().items()},
                   encoder_color={k: v.detach().cpu() for k, v in neural.position_encoder_color.state_dict().items()},
                   cur_ts=int(neural.cur_ts), max_ts=int(neural.max_ts), reboot_ts=int(neural.reboot_ts),
                   after_pgo=bool(neural.after_pgo),
                   travel_dist=neural.travel_dist.detach().cpu(),
                   config=portable_config(config), format=FORMAT, source_commit=SOURCE_COMMIT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.pin-map-', dir=path.parent))
    try:
        torch.save(payload, temporary / 'map.pt')
        checksum = digest(temporary / 'map.pt')
        manifest = dict(metadata, format=FORMAT, source_commit=SOURCE_COMMIT,
                        tensor_sha256=checksum, neural_points=neural.count(),
                        after_pgo=payload['after_pgo'],
                        reference='scan_end', units='metres_radians_seconds')
        (temporary / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        (temporary / 'COMPLETE').write_text(checksum + '\n')
        # The caller owns this fresh run directory; never replace an existing result.
        if path.exists() or path.is_symlink():
            raise FileExistsError('Map destination appeared during saving')
        os.rename(temporary, path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def load_map(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('format') != FORMAT or manifest.get('source_commit') != SOURCE_COMMIT:
        raise ValueError('Unsupported neural map format/source; ROLL and object-pickle maps are not inputs')
    checksum = manifest['tensor_sha256']
    if (path / 'COMPLETE').read_text().strip() != checksum or digest(path / 'map.pt') != checksum:
        raise ValueError('Incomplete or changed neural map')
    payload = torch.load(path / 'map.pt', map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT or payload.get('source_commit') != SOURCE_COMMIT:
        raise ValueError('Neural map payload disagrees with its manifest')
    n, features = manifest['neural_points'], payload['config']['feature_dim']
    shapes = dict(neural_points=(n, 3), point_orientations=(n, 4), geo_features=(n+1, features),
                  point_ts_create=(n,), point_ts_update=(n,), point_certainties=(n,))
    if n <= 0:
        raise ValueError('Empty neural map')
    for name, shape in shapes.items():
        value = payload[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('Invalid neural map tensor: ' + name)
    for name in ('point_ts_create', 'point_ts_update'):
        if payload[name].is_floating_point() or (payload[name] < 0).any() or (payload[name] > payload['max_ts']).any():
            raise ValueError('Invalid neural point frame association')
    norms = payload['point_orientations'].norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise ValueError('Invalid neural point orientation')
    # Earlier project maps retained the rotated point frames but omitted the
    # upstream switch that applies them during feature queries. Recover that
    # switch from their recorded PGO revision, without rewriting the input map.
    if 'after_pgo' not in payload:
        revision = manifest.get('map_revision', 0)
        if type(revision) is not int or revision < 0:
            raise ValueError('Invalid neural map correction history')
        payload['after_pgo'] = revision > 0
    if type(payload['after_pgo']) is not bool:
        raise ValueError('Invalid neural map point-frame state')
    if 'after_pgo' in manifest and manifest['after_pgo'] != payload['after_pgo']:
        raise ValueError('Neural map point-frame state disagrees with manifest')
    if not payload['after_pgo'] and (payload['point_orientations'][:, 1:].abs() > 1e-6).any():
        raise ValueError('Rotated neural point frames require a recorded PGO state')
    if payload['config']['color_on'] or payload['config']['semantic_on']:
        raise ValueError('This adapter supports geometry-only maps')
    return payload, manifest
