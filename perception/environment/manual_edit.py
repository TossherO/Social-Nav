"""Save manually edited static occupancy as a new source-bound map package."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .layers import GridSnapshot, finish_evidence
from .storage import StaticEnvironmentMap, asset_reference, content_hash
from .export import _save_package


def source_binding(directory):
    directory = Path(directory)
    return {name: content_hash(directory/name) for name in ('manifest.yaml', 'layers.npz')}


def save_manual_map(source, output, occupancy, *, expected_binding, selection_provenance=None):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('输出目录已存在，不能覆盖')
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError('输出不能位于源地图内部')
    if source_binding(source) != expected_binding:
        raise ValueError('源静态地图已改变，请重新打开编辑器')
    asset = StaticEnvironmentMap(source)
    original = asset.snapshot
    cells = np.asarray(occupancy)
    if cells.shape != tuple(original.metadata['shape']) or not np.isin(cells, (-1, 0, 100)).all():
        raise ValueError('静态地图状态或尺寸无效')
    changed = cells != original.layers['occupancy']
    if not changed.any():
        raise ValueError('没有需要保存的地图状态修改')
    layers = {key: value.copy() for key, value in original.layers.items()}
    layers['occupancy'] = cells.astype(np.int8, copy=True)
    for name, value in (('observed', False), ('last_observed', np.nan), ('coverage', 0.),
                        ('obstacle_min_z', np.nan), ('obstacle_max_z', np.nan),
                        ('endpoint_count', 0), ('conditional_obstacle', False)):
        if name in layers:
            layers[name][changed] = value
    manual = layers.get('manual_edit', np.zeros(cells.shape, bool)).astype(bool, copy=True)
    manual[changed] = True
    assumed = layers.get('assumed_free', np.zeros(cells.shape, bool)).astype(bool, copy=True)
    assumed[changed] = cells[changed] == 0
    layers.update(manual_edit=manual, assumed_free=assumed)
    if 'survey_index' in layers:
        layers['survey_index'][changed] = -2
    derived = finish_evidence(layers['occupancy'], layers['observed'], layers['last_observed'],
        layers['coverage'], layers['obstacle_min_z'], layers['obstacle_max_z'],
        layers['ground_z'], layers['ground_support'], original.metadata['resolution'])
    if original.metadata.get('projection_strategy') in ('docker_assume_free', 'ground_relative_assume_free'):
        # The projection has no ray-observed free-space evidence.
        derived['unknown_distance'][:] = 0
        derived['frontier'][:] = False
    layers.update(derived)
    record = dict(source_snapshot_id=original.metadata['snapshot_id'], source_files=expected_binding,
                  source=asset_reference(source), edited_cells=int(changed.sum()),
                  manual_free_cells=int((changed & (cells == 0)).sum()),
                  manual_occupied_cells=int((changed & (cells == 100)).sum()),
                  manual_unknown_cells=int((changed & (cells == -1)).sum()),
                  cell_selection='map_cell_center', manual_free_is_observation=False)
    if selection_provenance is not None:
        record['selection_provenance'] = copy.deepcopy(selection_provenance)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()+cells.astype(np.int8).tobytes()).hexdigest()
    metadata = dict(original.metadata, snapshot_id='manual_static:'+digest, manual_edit=record,
                    manual_free_cells_total=int((manual & (cells == 0)).sum()))
    if 'survey_index' in layers:
        metadata['survey_index_meaning'] = 'zero_based_manifest_sources_minus_one_final_map_endpoints_minus_two_manual_edit'
    if 'conditional_obstacle' in layers:
        metadata['conditional_obstacle_cells'] = int(layers['conditional_obstacle'].sum())
    snapshot = GridSnapshot(metadata, layers)
    source_info = copy.deepcopy(asset.manifest['source'])
    source_info['manual_edit'] = record
    limitations = list(asset.manifest.get('limitations', []))
    if 'manual_free_not_sensor_observation' not in limitations:
        limitations.append('manual_free_not_sensor_observation')
    if source_binding(source) != expected_binding:
        raise ValueError('源静态地图在编辑过程中发生变化')
    return _save_package(snapshot, output, source_info, asset.manifest['configuration'],
                         metadata['frame_id'], asset.manifest.get('bound_final_keyframes', 0), True, limitations)
