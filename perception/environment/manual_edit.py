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


def fit_manual_ground(layers, resolution):
    """Fit a local plane to existing free ground for each unsupported manual-free cell.

    Fitted heights remain separate from measured ground support and observation time.
    """
    from scipy.spatial import cKDTree
    cells=layers['occupancy'];ground=layers['ground_z'];supported=layers['ground_support']
    manual=layers['manual_edit'] & (cells==0)
    estimated=layers.get('ground_estimated',np.zeros(cells.shape,bool)).copy()
    target=manual & ~(supported & np.isfinite(ground)) & ~estimated
    if not target.any():return None
    reference=(cells==0)&supported&np.isfinite(ground)&~layers['manual_edit']&~estimated
    references=np.argwhere(reference);targets=np.argwhere(target)
    if len(references)<3:raise ValueError('有效地面参考点不足，无法拟合人工空闲区域')
    coordinates=references.astype(float)*resolution
    positions=targets.astype(float)*resolution
    distances,indices=cKDTree(coordinates).query(positions,k=min(96,len(references)),workers=1)
    offset=coordinates[indices]-positions[:,None,:]
    values=ground[tuple(references[indices].transpose(2,0,1))].astype(float)
    scale=np.maximum(.5,np.median(distances,axis=1))
    weights=np.exp(-distances/scale[:,None])
    design=np.concatenate((offset,np.ones((*offset.shape[:2],1))),axis=2)
    coefficients=None
    for _ in range(3):
        matrix=np.einsum('nki,nk,nkj->nij',design,weights,design)
        # Dampen unstable slopes when all samples lie on one side of a gap.
        matrix[:,0,0]+=.1;matrix[:,1,1]+=.1
        rhs=np.einsum('nki,nk,nk->ni',design,weights,values)
        coefficients=np.linalg.solve(matrix,rhs)
        residual=values-np.einsum('nki,ni->nk',design,coefficients)
        center=np.median(residual,axis=1)
        robust=np.maximum(.03,1.4826*np.median(np.abs(residual-center[:,None]),axis=1))
        weights=np.exp(-distances/scale[:,None])*np.minimum(1.,1.5*robust[:,None]/np.maximum(np.abs(residual-center[:,None]),1e-12))
    predictions=coefficients[:,2]
    rmse=np.sqrt(np.mean(residual**2,axis=1))
    if not np.isfinite(predictions).all() or not np.isfinite(rmse).all():
        raise ValueError('地面拟合结果无效，未保存地图')
    yy,xx=targets.T
    invalid_support=int(supported[yy,xx].sum())
    supported[yy,xx]=False
    ground[yy,xx]=predictions.astype(ground.dtype)
    estimated[yy,xx]=True
    layers['ground_estimated']=estimated
    for name,data in (('ground_estimate_distance_m',distances[:,0]),('ground_estimate_rmse_m',rmse)):
        layer=layers.get(name,np.full(cells.shape,np.nan,np.float32)).copy()
        layer[yy,xx]=data.astype(np.float32);layers[name]=layer
    report=dict(method='robust_weighted_local_plane',estimated_cells=int(len(targets)),
        neighbor_count=int(indices.shape[1]),slope_regularization=.1,iterations=3,
        max_nearest_distance_m=float(distances[:,0].max()),
        max_neighbor_distance_m=float(distances[:,-1].max()),max_fit_rmse_m=float(rmse.max()),
        max_slope=float(np.linalg.norm(coefficients[:,:2],axis=1).max()),
        reference='previously_supported_nonmanual_free_cells',ground_support_changed=bool(invalid_support),
        estimates_are_observations=False)
    if invalid_support:report['invalid_support_cleared']=invalid_support
    return report


def save_manual_map(source, output, occupancy, *, expected_binding, selection_provenance=None, fit_ground=False):
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
    existing_manual=original.layers.get('manual_edit',np.zeros(cells.shape,bool))
    missing_ground=(existing_manual&(cells==0)&~(original.layers['ground_support']&np.isfinite(original.layers['ground_z']))&
                    ~original.layers.get('ground_estimated',np.zeros(cells.shape,bool)))
    if not changed.any() and not (fit_ground and missing_ground.any()):
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
    if 'ground_estimated' in layers:
        invalid=layers['ground_estimated'] & (cells!=0)
        layers['ground_z'][invalid]=np.nan
        layers['ground_estimated'][invalid]=False
        for name in ('ground_estimate_distance_m','ground_estimate_rmse_m'):
            if name in layers:layers[name][invalid]=np.nan
    estimation=fit_manual_ground(layers,float(original.metadata['resolution'])) if fit_ground else None
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
    if estimation is not None:record['ground_estimation']=estimation
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()+cells.astype(np.int8).tobytes()+
                            layers['ground_z'].tobytes()).hexdigest()
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
    if 'ground_estimated' in layers:
        metadata['ground_estimated_cells']=int(layers['ground_estimated'].sum())
    if 'ground_estimated' in layers and layers['ground_estimated'].any():
        if 'estimated_ground_not_sensor_observation' not in limitations:
            limitations.append('estimated_ground_not_sensor_observation')
    if source_binding(source) != expected_binding:
        raise ValueError('源静态地图在编辑过程中发生变化')
    return _save_package(snapshot, output, source_info, asset.manifest['configuration'],
                         metadata['frame_id'], asset.manifest.get('bound_final_keyframes', 0), True, limitations)
