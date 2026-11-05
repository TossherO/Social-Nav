"""Causal table-edge and outline candidates; all automatic poses remain hypotheses.

Historical vehicle geometry and partial returns never certify clearance or
reachability. These helpers neither call a model nor issue navigation commands.
"""
import copy
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


IDENTITY = ('frame_id', 'clock_domain', 'session_id', 'generation')


def footprint_world(footprint, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(footprint) @ np.array([[c, s], [-s, c]]) + [x, y]


def base_pose(world_from_lidar, base_from_lidar):
    return np.asarray(world_from_lidar) @ np.linalg.inv(np.asarray(base_from_lidar))


def inside_footprint(points_xy, footprint, pose):
    footprint=np.asarray(footprint);x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
    local=(np.asarray(points_xy)-[x,y]) @ np.array([[c,-s],[s,c]])
    return np.all((local>=footprint.min(axis=0)) & (local<=footprint.max(axis=0)),axis=1)


def validate_sampling_policy(policy):
    required={'scope','footrest_under_table_allowed','front_gaps_m','yaw_offsets_deg',
        'along_step_m','min_along_separation_m','lateral_zones','end_guard_m','evidence_window_s','evidence_decay_s',
        'below_edge_band_m','voxel_m','ranking_weights'}
    if set(policy)!=required or policy['scope']!='offline_historical_geometry_hypothesis':
        raise ValueError('Explicit offline sampling policy required')
    def finite(value):return type(value) in (int,float) and math.isfinite(value)
    if type(policy['footrest_under_table_allowed']) is not bool:
        raise ValueError('Explicit under-table permission required')
    gaps=policy['front_gaps_m'];angles=policy['yaw_offsets_deg'];band=policy['below_edge_band_m']
    if (not isinstance(gaps,list) or len(gaps)!=2 or not all(finite(v) for v in gaps) or gaps[0]>=gaps[1] or
            (gaps[0]<0 and not policy['footrest_under_table_allowed'])):
        raise ValueError('Two ordered, explicit signed front gaps required')
    if (not isinstance(angles,list) or not angles or not all(finite(v) and abs(v)<90 for v in angles) or
            len(set(angles))!=len(angles)):
        raise ValueError('Explicit facing hypotheses required')
    if (not isinstance(band,list) or len(band)!=2 or not all(finite(v) for v in band) or not 0<band[0]<band[1]):
        raise ValueError('Explicit below-edge evidence band required')
    for key in ['along_step_m','min_along_separation_m','evidence_window_s','evidence_decay_s','voxel_m']:
        if not finite(policy[key]) or policy[key]<=0:raise ValueError('Positive diagnostic scales required')
    if (not finite(policy['end_guard_m']) or policy['end_guard_m']<0 or
            type(policy['lateral_zones']) is not int or not 1<=policy['lateral_zones']<=4):
        raise ValueError('At most eight final hypotheses and a nonnegative guard required')
    weights=policy['ranking_weights']
    if set(weights)!={'observed_returns','approach_m','turn_rad'} or any(not finite(v) or v<0 for v in weights.values()):
        raise ValueError('Explicit nonnegative ranking weights required')


def edge_seeds(points, rgb, world_from_lidar, camera_from_lidar, camera, roi):
    """Current-image manual ROI + light surface + nearest depth band.

    ROI is an assistant annotation, not an automatic semantic segmentation.
    No whole-demo world crop or final table coordinates enter this operation.
    """
    p = np.asarray(points)[:, :3]
    t = np.asarray(camera_from_lidar)
    optical = p @ t[:3, :3].T + t[:3, 3]
    finite = np.isfinite(optical).all(axis=1) & (optical[:, 2] > .3) & (optical[:, 2] < 5.)
    indices = np.flatnonzero(finite)
    pixels = cv2.projectPoints(optical[finite], np.zeros(3), np.zeros(3),
        np.asarray(camera['K']).reshape(3, 3), np.asarray(camera['D'], dtype=float))[0].reshape(-1, 2)
    x0, y0, x1, y1 = roi
    h, w = rgb.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI outside current image')
    keep = ((pixels[:, 0] >= x0) & (pixels[:, 0] < x1) &
            (pixels[:, 1] >= y0) & (pixels[:, 1] < y1))
    indices = indices[keep]
    uv = pixels[keep].astype(int)
    colors = rgb[uv[:, 1], uv[:, 0]].astype(float)
    indices = indices[(colors.min(axis=1) > 85) & (np.ptp(colors, axis=1) < 65)]
    if len(indices) < 3:
        raise ValueError('Insufficient current tabletop seeds')
    near = float(np.quantile(optical[indices, 2], .1))
    indices = indices[optical[indices, 2] < near + .4]
    pose = np.asarray(world_from_lidar)
    return p[indices] @ pose[:3, :3].T + pose[:3, 3]


def fit_edge(points, observer_xy):
    """Fit an observed segment; endpoints are support quantiles, not corners."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20 or not np.isfinite(points).all():
        raise ValueError('Insufficient finite edge support')
    rng = np.random.default_rng(123)
    best = None
    for _ in range(800):
        a, b = points[rng.choice(len(points), 2, replace=False), :2]
        direction = b-a
        length = np.linalg.norm(direction)
        if length < .2:
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        mask = np.abs((points[:, :2]-a) @ normal) < .035
        if best is None or int(mask.sum()) > best[0]:
            best = (int(mask.sum()), mask)
    if best is None or best[0] < 20:
        raise ValueError('No supported edge line')
    cloud = points[best[1]]
    center = cloud[:, :2].mean(axis=0)
    direction = np.linalg.svd(cloud[:, :2]-center, full_matrices=False)[2][0]
    if direction[0] < 0:
        direction = -direction
    along = (cloud[:, :2]-center) @ direction
    limits = np.quantile(along, [.02, .98])
    ends = center + limits[:, None]*direction
    normal = np.array([-direction[1], direction[0]])
    if (center-np.asarray(observer_xy)) @ normal < 0:
        normal = -normal
    # Height only supports review projection; it is not a measured floor height.
    height = np.linalg.lstsq(np.c_[along, np.ones(len(along))], cloud[:, 2], rcond=None)[0]
    z = np.c_[limits, np.ones(2)] @ height
    return dict(observed_segment_xy=ends.tolist(), observed_segment_xyz=np.c_[ends, z].tolist(),
        estimated_interior_normal_xy=normal.tolist(), inlier_count=len(cloud),
        observed_length_m=float(limits[1]-limits[0]),
        line_residual_rmse_m=float(np.sqrt(np.mean(((cloud[:, :2]-center) @ normal)**2))),
        endpoints_are_confirmed_corners=False, status='assistant_prefix_fit_not_user_reviewed')


def validate_context(context, observation):
    if context.get('kind') == 'table_outline_hypothesis':
        return _validate_prior_context(context, observation)
    required = {'kind', 'pose_reference', 'reference_sha256', 'reference_status', 'footprint_xy_m',
                'base_from_lidar', 'edge', 'source_cutoff_s', 'frame_id', 'session_id', 'generation',
                'source_bindings', 'front_gaps_m', 'end_guard_m'}
    optional={'sampling_policy','table_band_evidence'}
    if not required<=set(context) or set(context)-required-optional or context['kind'] != 'historical_footprint_edge_hypothesis':
        raise ValueError('Explicit offline geometry context required')
    if context['pose_reference'] != 'base_link_hypothesis' or context['reference_status'] != 'historical_image_reference_current_platform_not_confirmed':
        raise ValueError('Historical geometry must remain an unconfirmed hypothesis')
    if (not isinstance(context['reference_sha256'], str) or len(context['reference_sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in context['reference_sha256'])):
        raise ValueError('Geometry asset digest required')
    for k in ['frame_id', 'session_id', 'generation']:
        if context[k] != observation[k]:
            raise ValueError('Geometry context belongs to another localization')
    cutoff = context['source_cutoff_s']
    if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff > observation['stamp_s']:
        raise ValueError('Future table geometry is not a current input')
    sources = context['source_bindings']
    if not sources or any(type(s['stamp_s']) not in (int, float) or not math.isfinite(s['stamp_s']) or
                          s['stamp_s'] > cutoff or not s.get('observation_sha256') for s in sources):
        raise ValueError('Causal source bindings required')
    if max(s['stamp_s'] for s in sources) != cutoff:
        raise ValueError('Geometry cutoff must match actual sources')
    matrix = np.asarray(context['base_from_lidar'], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1.)):
        raise ValueError('Rigid base-from-LiDAR hypothesis required')
    footprint = np.asarray(context['footprint_xy_m'], dtype=float)
    # This offline sampler handles a forward-aligned rectangular reference.
    if (footprint.shape != (4, 2) or not np.isfinite(footprint).all() or
            set(map(tuple, footprint)) != {(x, y) for x in [footprint[:, 0].min(), footprint[:, 0].max()]
                                         for y in [footprint[:, 1].min(), footprint[:, 1].max()]} or
            np.any(np.ptp(footprint, axis=0) <= 0)):
        raise ValueError('Nondegenerate rectangular footprint required')
    edge = context['edge']
    ends = np.asarray(edge['observed_segment_xy'], dtype=float)
    normal = np.asarray(edge['estimated_interior_normal_xy'], dtype=float)
    if ends.shape != (2, 2) or normal.shape != (2,) or not np.isfinite(ends).all() or not np.isfinite(normal).all():
        raise ValueError('Finite observed edge required')
    direction = ends[1]-ends[0]
    if np.linalg.norm(direction) < 1e-6 or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6) or abs(direction @ normal) > 1e-5:
        raise ValueError('Unit perpendicular table normal required')
    gaps = context['front_gaps_m']
    signed_gaps=False
    if 'sampling_policy' in context:
        validate_sampling_policy(context['sampling_policy'])
        if context['sampling_policy']['front_gaps_m']!=gaps or context['sampling_policy']['end_guard_m']!=context['end_guard_m']:
            raise ValueError('Sampling policy and context disagree')
        signed_gaps=context['sampling_policy']['footrest_under_table_allowed']
    if (not isinstance(gaps, list) or not gaps or len(gaps) > 8 or
            any(type(g) not in (int, float) or not math.isfinite(g) or (g <= 0 and not signed_gaps) for g in gaps) or
            len(set(gaps)) != len(gaps) or type(context['end_guard_m']) not in (int, float) or
            not math.isfinite(context['end_guard_m']) or context['end_guard_m'] < 0):
        raise ValueError('Explicit permitted front gaps and endpoint guard required')
    if 'table_band_evidence' in context:
        if 'sampling_policy' not in context:raise ValueError('Evidence requires its sampling policy')
        evidence=context['table_band_evidence'];policy=context['sampling_policy']
        if evidence['below_edge_band_m']!=policy['below_edge_band_m'] or evidence['voxel_m']!=policy['voxel_m']:
            raise ValueError('Evidence diagnostic scales changed')
        source_map={s['observation_id']:s for s in sources}
        seen_sources=set()
        for source in evidence['sources']:
            bound=source_map.get(source['observation_id'])
            if (bound is None or source['stamp_s']!=bound['stamp_s'] or source['stamp_s']>observation['stamp_s'] or
                    not 0<=observation['stamp_s']-source['stamp_s']<=policy['evidence_window_s'] or
                    not math.isclose(source['age_s'],observation['stamp_s']-source['stamp_s'],abs_tol=1e-6)):
                raise ValueError('Evidence not bound to causal source observations')
            if source['observation_id'] in seen_sources:raise ValueError('Repeated evidence source')
            seen_sources.add(source['observation_id'])
            ids=[]
            for c in source['candidates']:
                if (type(c['id']) is not int or not 0<=c['id']<=7 or
                        any(type(c[k]) is not int or c[k]<0 for k in ['total_voxels','nonself_voxels']) or
                        c['nonself_voxels']>c['total_voxels']):
                    raise ValueError('Invalid observed support')
                ids.append(c['id'])
            if len(ids)!=len(set(ids)):raise ValueError('Repeated evidence candidate')
            if 'candidates' in observation and set(ids)!={c['id'] for c in observation['candidates']}:
                raise ValueError('Observed support does not match current candidate IDs')


def edge_candidates(context, observation, *, stations=4):
    """Perpendicular targets inside supported along-edge span; never feasible."""
    validate_context(context, observation)
    if type(stations) is not int or stations < 1 or stations*len(context['front_gaps_m']) > 8:
        raise ValueError('At most eight candidate hypotheses')
    footprint = np.asarray(context['footprint_xy_m'])
    front = float(footprint[:, 0].max())
    a, b = np.asarray(context['edge']['observed_segment_xy'])
    length = float(np.linalg.norm(b-a)); direction = (b-a)/length
    normal = np.asarray(context['edge']['estimated_interior_normal_xy'])
    # Asymmetric Y extents also fit inside support; do not assume a centered body.
    yaw = math.atan2(normal[1], normal[0])
    local_y = np.array([-normal[1], normal[0]])
    along_extents = footprint[:, 1]*(local_y @ direction)
    lo = context['end_guard_m']-along_extents.min()
    hi = length-context['end_guard_m']-along_extents.max()
    if hi < lo:
        return []
    along = [float((lo+hi)/2)] if stations == 1 else np.linspace(lo, hi, stations)
    result = []
    for gap in context['front_gaps_m']:
        for distance in along:
            center = a+direction*distance-normal*(front+gap)
            result.append(dict(id=len(result), pose=[float(center[0]), float(center[1]), yaw],
                               motion='forward', geometry='unknown'))
    return result


def selection_context(context, world_from_lidar, candidates, gate):
    """Route before inference using current pose, never a previous response."""
    if (set(gate)!={'distance_m','heading_deg'} or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in gate.values())):
        raise ValueError('Explicit positive prompt-routing gate required')
    ego=base_pose(world_from_lidar,context['base_from_lidar']);yaw=math.atan2(ego[1,0],ego[0,0])
    nearby=[]
    for c in candidates:
        distance=float(np.linalg.norm(np.asarray(c['pose'][:2])-ego[:2,3]))
        angle=math.degrees(abs(math.atan2(math.sin(c['pose'][2]-yaw),math.cos(c['pose'][2]-yaw))))
        if c['geometry']!='blocked' and distance<=gate['distance_m'] and angle<=gate['heading_deg']:
            nearby.append(c['id'])
    return dict(mode='adjust_nearby_pose' if nearby else 'select_location',gate=dict(gate),nearby_ids=nearby,
        scope='Prompt routing only; no geometry validation or completion feedback')


def model_context(context, world_from_lidar, candidates=None, *, include_approach=False):
    """Only geometry and its limitations, never human answers or final pose."""
    if context.get('kind') == 'table_outline_hypothesis':
        fields = _prior_model_context(context, world_from_lidar)
        if include_approach and candidates is not None:
            ego = base_pose(world_from_lidar, context['base_from_lidar'])
            yaw = math.atan2(ego[1,0], ego[0,0])
            fields['candidate_approach'] = [dict(id=c['id'], straight_distance_m=round(float(np.linalg.norm(np.asarray(c['pose'][:2])-ego[:2,3])),3),
                heading_change_deg=round(math.degrees(abs(math.atan2(math.sin(c['pose'][2]-yaw),math.cos(c['pose'][2]-yaw)))),1)) for c in candidates]
        return fields
    fields=dict(candidate_pose_reference=context['pose_reference'],
        historical_footprint_xy_m=context['footprint_xy_m'],
        ego_base_pose_hypothesis=base_pose(world_from_lidar, context['base_from_lidar']).tolist(),
        observed_table_edge_xy=context['edge']['observed_segment_xy'],
        table_interior_normal_xy=context['edge']['estimated_interior_normal_xy'],
        limitation='Manual-assisted table edge and unconfirmed historical vehicle geometry; no verified clearance or route')
    if 'sampling_policy' in context:
        fields['seating_relation']=dict(footrest_under_table_allowed=context['sampling_policy']['footrest_under_table_allowed'],
            exact_perpendicular_required=False,front_gap_convention='signed outermost historical-footprint gap to edge; negative extends under table')
    if 'table_band_evidence' in context:
        fields['observed_subtable_returns']=context['table_band_evidence']
    if 'sampling_policy' in context and candidates is not None and include_approach:
        ego=base_pose(world_from_lidar,context['base_from_lidar'])
        yaw=math.atan2(ego[1,0],ego[0,0])
        fields['candidate_approach']=[dict(id=c['id'],
            straight_distance_m=round(float(np.linalg.norm(np.asarray(c['pose'][:2])-ego[:2,3])),3),
            heading_change_deg=round(math.degrees(abs(math.atan2(math.sin(c['pose'][2]-yaw),math.cos(c['pose'][2]-yaw)))),1))
            for c in candidates]
        fields['approach_scope']='Current hypothetical base only; straight distance is not a verified route cost'
    return fields


def refine_candidates(context, observation, scans, policy):
    """Dense causal hypotheses -> spatially diverse pairs; no endpoint input.

    A negative gap permits an XY footprint to enter below the edge. No 3D
    passenger/footrest envelope is inferred and no target becomes feasible.
    """
    context=copy.deepcopy(context)
    context.pop('table_band_evidence',None)
    context.update(sampling_policy=copy.deepcopy(policy),front_gaps_m=list(policy['front_gaps_m']),end_guard_m=policy['end_guard_m'])
    validate_context(context,observation)
    ends=np.asarray(context['edge']['observed_segment_xyz'],dtype=float)
    if ends.shape!=(2,3) or not np.isfinite(ends).all() or not np.allclose(ends[:,:2],context['edge']['observed_segment_xy']):
        raise ValueError('Bound 3D edge support required for floor-independent diagnostics')
    a,b=ends[:,:2];length=np.linalg.norm(b-a);direction=(b-a)/length
    normal=np.asarray(context['edge']['estimated_interior_normal_xy']);normal_yaw=math.atan2(normal[1],normal[0])
    footprint=np.asarray(context['footprint_xy_m'])
    current=base_pose(observation['world_from_lidar'],context['base_from_lidar'])
    ego=[*current[:2,3],math.atan2(current[1,0],current[0,0])]
    sources=[];seen=set();source_map={s['observation_id']:s for s in context['source_bindings']}
    for scan in scans:
        obs=scan['observation'];age=observation['stamp_s']-obs['stamp_s']
        if age<0 or any(obs[k]!=observation[k] for k in ['frame_id','clock_domain','session_id','generation']):
            raise ValueError('Future or incompatible scan evidence')
        if age>policy['evidence_window_s']:continue
        if obs['observation_id'] in seen:raise ValueError('Repeated source scan')
        seen.add(obs['observation_id'])
        if obs['observation_id'] not in source_map or source_map[obs['observation_id']]['stamp_s']!=obs['stamp_s']:
            raise ValueError('Unbound scan evidence')
        points=np.asarray(scan['world_points']);points=points[np.isfinite(points).all(axis=1)]
        along=(points[:,:2]-a)@direction
        edge_z=ends[0,2]+along/length*(ends[1,2]-ends[0,2])
        below=edge_z-points[:,2];lo,hi=policy['below_edge_band_m']
        slab=points[(below>=lo)&(below<=hi)]
        # Voxel representatives keep return density from dominating this diagnostic.
        _,indices=np.unique(np.floor(slab/policy['voxel_m']).astype(np.int64),axis=0,return_index=True)
        slab=slab[indices]
        source_base=base_pose(obs['world_from_lidar'],context['base_from_lidar'])
        self_mask=inside_footprint(slab[:,:2],footprint,[*source_base[:2,3],math.atan2(source_base[1,0],source_base[0,0])])
        sources.append(dict(observation_id=obs['observation_id'],stamp_s=obs['stamp_s'],age_s=age,
                            points=slab,nonself=~self_mask))
    if observation['observation_id'] not in seen:
        raise ValueError('Current scan required; missing evidence is not zero')
    headings=[]
    for angle in policy['yaw_offsets_deg']:
        yaw=normal_yaw+math.radians(angle)
        rotated=footprint_world(footprint,[0,0,yaw]);extent=rotated@direction
        headings.append(dict(yaw=yaw,front=float((rotated@normal).max()),
            lo=policy['end_guard_m']-float(extent.min()),hi=length-policy['end_guard_m']-float(extent.max())))
    lo=max(h['lo'] for h in headings);hi=min(h['hi'] for h in headings)
    if hi<lo:return context,[],dict(status='insufficient_observed_edge_span',pool_size=0)
    zones=min(policy['lateral_zones'],max(1,int((hi-lo)/policy['along_step_m'])))
    boundaries=np.linspace(lo,hi,zones+1)
    weights=policy['ranking_weights'];pool=[];selected=[]
    def describe(pose):
        counts=[];support=0.
        for source in sources:
            mask=inside_footprint(source['points'][:,:2],footprint,pose)
            count=int((mask&source['nonself']).sum())
            support+=math.exp(-source['age_s']/policy['evidence_decay_s'])*math.log1p(count)
            counts.append(dict(total_voxels=int(mask.sum()),nonself_voxels=count))
        distance=float(np.linalg.norm(np.asarray(pose[:2])-ego[:2]))
        turn=abs(math.atan2(math.sin(pose[2]-ego[2]),math.cos(pose[2]-ego[2])))
        score=weights['observed_returns']*support+weights['approach_m']*distance+weights['turn_rad']*turn
        return dict(pose=pose,counts=counts,score=score)
    for zone,(left,right) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        count=max(1,int(math.ceil((right-left)/policy['along_step_m'])))
        anchors=left+(np.arange(count)+.5)*(right-left)/count
        pairs=[]
        for along in anchors:
            for heading in headings:
                pair=[]
                for gap in policy['front_gaps_m']:
                    xy=a+direction*along-normal*(heading['front']+gap)
                    pair.append(describe([float(xy[0]),float(xy[1]),heading['yaw']]))
                record=dict(zone=zone,along_m=float(along),pair=pair)
                pairs.append(record);pool.append(record)
        diverse=[p for p in pairs if all(abs(p['along_m']-s['along_m'])>=policy['min_along_separation_m'] for s in selected)]
        if diverse:
            selected.append(min(diverse,key=lambda p:(min(c['score'] for c in p['pair']),sum(c['score'] for c in p['pair']),p['along_m'])))
    # IDs: first near-depth row, then farther-depth row; stable within this set only.
    chosen=[pair['pair'][depth] for depth in range(2) for pair in selected]
    candidates=[dict(id=i,pose=c['pose'],motion='forward',geometry='unknown') for i,c in enumerate(chosen)]
    evidence=dict(kind='raw_returns_below_observed_edge',below_edge_band_m=policy['below_edge_band_m'],
        voxel_m=policy['voxel_m'],sources=[],
        limitation='Partial vertical band only; historical footprint and self mask unconfirmed; no return is not free; old objects may move')
    for index,source in enumerate(sources):
        evidence['sources'].append({**{k:source[k] for k in ['observation_id','stamp_s','age_s']},
            'candidates':[dict(id=i,**c['counts'][index]) for i,c in enumerate(chosen)]})
    context['table_band_evidence']=evidence
    validate_context(context,observation)
    return context,candidates,dict(status='refined_unknown_hypotheses',pool_size=len(pool)*2,
        selected=selected,scope='causal_ranking_not_path_or_clearance',
        selection_uses_demo_endpoint=False,ground_plane_used=False)


def overlap_diagnostics(world_points, sensor_xyz, world_from_lidar, context, candidates, ground_config):
    """Observed occupied columns in hypothetical terminal / straight swept body.

    Straight interpolation is a diagnostic corridor, not a kinematic planner.
    Zero overlaps and sparse floor evidence both leave reachability unknown.
    """
    from perception.environment.geometry import ground_plane
    points = np.asarray(world_points)
    plane = ground_plane(points, np.asarray(sensor_xyz), ground_config)
    if plane is None:
        return dict(status='ground_unavailable', candidates=[], reachability='unknown')
    height = points[:, 2]-(np.c_[points[:, :2], np.ones(len(points))] @ plane)
    occupied = points[(height >= .12) & (height <= 1.8)]
    footprint = np.asarray(context['footprint_xy_m'])
    current = base_pose(world_from_lidar, context['base_from_lidar'])
    ego = [*current[:2, 3], math.atan2(current[1, 0], current[0, 0])]
    low, high = footprint.min(axis=0), footprint.max(axis=0)
    def inside(points_xy, pose):
        x, y, yaw = pose; c, s = math.cos(yaw), math.sin(yaw)
        local = (points_xy-[x, y]) @ np.array([[c, -s], [s, c]])
        return np.all((local >= low) & (local <= high), axis=1)
    self_mask = inside(occupied[:, :2], ego)
    # Report self-model overlaps separately. Do not silently delete those returns.
    rows = []
    for candidate in candidates:
        pose = candidate['pose']; terminal = inside(occupied[:, :2], pose)
        dyaw = math.atan2(math.sin(pose[2]-ego[2]), math.cos(pose[2]-ego[2]))
        n = max(2, int(math.ceil(max(np.linalg.norm(np.array(pose[:2])-ego[:2])/.05, abs(dyaw)/.05)))+1)
        swept = np.zeros(len(occupied), dtype=bool)
        for alpha in np.linspace(0, 1, n):
            swept |= inside(occupied[:, :2], [ego[0]+alpha*(pose[0]-ego[0]),
                            ego[1]+alpha*(pose[1]-ego[1]), ego[2]+alpha*dyaw])
        def voxels(mask):
            return int(len(np.unique(np.floor(occupied[mask]/.05).astype(np.int64), axis=0)))
        rows.append(dict(id=candidate['id'], terminal_occupied_voxels=voxels(terminal),
            terminal_nonself_voxels=voxels(terminal & ~self_mask),
            straight_sweep_nonself_voxels=voxels(swept & ~self_mask), geometry='unknown'))
    edge=context['edge'].get('observed_segment_xyz')
    edge_height=None
    if edge is not None:
        edge=np.asarray(edge,dtype=float)
        edge_height=(edge[:,2]-(np.c_[edge[:,:2],np.ones(len(edge))] @ plane)).tolist()
    return dict(status='conditional_observed_overlap_only', ground_status='unreviewed_plane_candidate',
        table_edge_height_above_plane_m=edge_height, ground_plane_z_equals_ax_by_c=plane.tolist(),
        obstacle_height_band_m=[.12, 1.8], voxel_m=.05, occupied_points=len(occupied),
        self_model_points=int(self_mask.sum()), candidates=rows, reachability='unknown',
        limitation='Unconfirmed footprint and floor plane; raw scan not deskewed; no return is not free; tabletop and chairs are not segmented')


def table_candidates(table,*,stand_off_m,spacing_m):
    """Manual/observed rectangle -> neutral hypotheses; no reachability assertion."""
    if set(table)!={'center_xy','yaw_rad','length_m','width_m','source_observation_id','source_stamp_s'}:
        raise ValueError('Bound observed table rectangle required')
    numbers=[*table['center_xy'],table['yaw_rad'],table['length_m'],table['width_m'],stand_off_m,spacing_m]
    if len(table['center_xy'])!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in numbers):
        raise ValueError('Invalid table geometry')
    if min(table['length_m'],table['width_m'],stand_off_m,spacing_m)<=0:raise ValueError('Positive dimensions required')
    cx,cy=table['center_xy'];yaw=table['yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    positions=[]
    for side in [-1,1]:
        for along in [-min(spacing_m,table['length_m']/4),min(spacing_m,table['length_m']/4)]:
            positions.append((along,side*(table['width_m']/2+stand_off_m)))
    for side in [-1,1]:positions.append((side*(table['length_m']/2+stand_off_m),0.))
    result=[]
    for i,(x,y) in enumerate(positions):
        px,py=cx+c*x-s*y,cy+s*x+c*y
        result.append(dict(id=i,pose=[px,py,math.atan2(cy-py,cx-px)],motion='forward',geometry='unknown'))
    return result


def read_observation(directory):
    from ..vlm_api_common import read_json, asset_path
    directory = Path(directory)
    row = read_json(directory/'observation.json')
    for name, expected in row['assets'].items():
        if hashlib.sha256(asset_path(directory, name).read_bytes()).hexdigest() != expected:
            raise ValueError('Observation asset changed')
    from perception.detection_tracking.inference.geometry import validate_transform
    validate_transform(row['world_from_lidar'])
    return row


def validate_prior(prior, observation):
    """The test prior is available only after all its real source observations."""
    if prior['kind'] != 'approximate_table_outline' or prior['metric_verified'] is not False:
        raise ValueError('This sampler requires an explicitly approximate table')
    if any(prior[k] != observation[k] for k in IDENTITY):
        raise ValueError('Table and observation belong to different local frames')
    stamps = [s['stamp_s'] for s in prior['sources']]
    if not stamps or not np.isfinite(stamps).all() or max(stamps) != prior['available_after_s']:
        raise ValueError('Table availability must bind its actual sources')
    if prior['available_after_s'] > observation['stamp_s']:
        raise ValueError('Future table prior')
    if any(not s.get('observation_sha256') for s in prior['sources']):
        raise ValueError('Table source fingerprints required')
    p = np.asarray(prior['polygon_xy_m'], dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all() or not np.isfinite(prior['edge_z_m']):
        raise ValueError('Finite rectangular table required')
    edges = np.roll(p, -1, axis=0)-p
    cross = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if (not np.all(cross > 1e-6) or not np.allclose(edges[:2], -edges[2:]) or
            abs(float(edges[0] @ edges[1])) > 1e-6):
        raise ValueError('Counterclockwise rectangle required')


def _validate_prior_context(context, observation):
    required = {'kind', 'pose_reference', 'reference_sha256', 'reference_status', 'footprint_xy_m',
                'base_from_lidar', 'table_prior', 'sampling_policy', 'evidence_sources', 'candidate_evidence'}
    if set(context) != required or context['kind'] != 'table_outline_hypothesis':
        raise ValueError('Unexpected table hypothesis context')
    if (context['pose_reference'] != 'base_link_hypothesis' or
            context['reference_status'] != 'historical_image_reference_current_platform_not_confirmed'):
        raise ValueError('Historical body must remain an explicit hypothesis')
    sha = context['reference_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Vehicle reference fingerprint required')
    from navigation.planning.footprint import VehicleGeometry
    VehicleGeometry(context['footprint_xy_m'], context['base_from_lidar'], source=sha)
    validate_prior(context['table_prior'], observation)
    validate_sampling_policy(context['sampling_policy'])
    ids = set()
    for source in context['evidence_sources']:
        if (source['observation_id'] in ids or not source.get('observation_sha256') or
                not 0 <= observation['stamp_s']-source['stamp_s'] <= context['sampling_policy']['evidence_window_s'] or
                any(source[k] != observation[k] for k in IDENTITY)):
            raise ValueError('Invalid causal return evidence')
        ids.add(source['observation_id'])
    if observation['observation_id'] not in ids:
        raise ValueError('Current evidence required')
    evidence = context['candidate_evidence']
    if len({r['id'] for r in evidence}) != len(evidence):
        raise ValueError('Duplicate evidence candidate')
    if 'candidates' in observation and {r['id'] for r in evidence} != {r['id'] for r in observation['candidates']}:
        raise ValueError('Candidate evidence identity mismatch')
    for row in evidence:
        if len(row['nonself_voxels']) != len(context['evidence_sources']) or any(type(v) is not int or v < 0 for v in row['nonself_voxels']):
            raise ValueError('Invalid return counts')


def table_edges(prior):
    p = np.asarray(prior['polygon_xy_m'])
    for i, (a, b) in enumerate(zip(p, np.roll(p, -1, axis=0))):
        direction = (b-a)/np.linalg.norm(b-a)
        yield i, a, b, direction, np.array([-direction[1], direction[0]])


def generate(prior, observations, reference_path, policy):
    """All four sides; per-heading limits and individual, diverse goal ranking."""
    from navigation.planning.footprint import VehicleGeometry
    observations = [Path(p) for p in observations]
    rows = [read_observation(p) for p in observations]
    current = rows[-1]
    validate_prior(prior, current); validate_sampling_policy(policy)
    if any(b['stamp_s'] <= a['stamp_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Strictly ordered observations required')
    vehicle = VehicleGeometry.historical(reference_path)
    sources, clouds = [], []
    for directory, row in zip(observations, rows):
        if any(row[k] != current[k] for k in IDENTITY):
            raise ValueError('Cannot accumulate across localization resets')
        age = current['stamp_s']-row['stamp_s']
        if age > policy['evidence_window_s']: continue
        points = np.load(directory/'scan.npy', allow_pickle=False)[:, :3]
        t = np.asarray(row['world_from_lidar']); world = points @ t[:3, :3].T+t[:3, 3]
        below = prior['edge_z_m']-world[:, 2]; lo, hi = policy['below_edge_band_m']
        world = world[np.isfinite(world).all(axis=1) & (below >= lo) & (below <= hi)]
        _, keep = np.unique(np.floor(world/policy['voxel_m']).astype(np.int64), axis=0, return_index=True)
        world = world[keep]
        world = world[~inside_footprint(world[:, :2], vehicle.footprint, vehicle.pose(t))]
        clouds.append((age, world))
        sources.append(dict(observation_id=row['observation_id'], stamp_s=row['stamp_s'],
            observation_sha256=hashlib.sha256((directory/'observation.json').read_bytes()).hexdigest(),
            **{k: row[k] for k in IDENTITY}))
    ego = vehicle.pose(current['world_from_lidar']); pool = []
    weights = policy['ranking_weights']
    for edge_id, a, b, direction, normal in table_edges(prior):
        for angle in policy['yaw_offsets_deg']:
            yaw = math.atan2(normal[1], normal[0])+math.radians(angle)
            rotated = footprint_world(vehicle.footprint, [0, 0, yaw])
            extent = rotated @ direction
            lo = policy['end_guard_m']-extent.min()
            hi = np.linalg.norm(b-a)-policy['end_guard_m']-extent.max()
            if hi < lo: continue
            for along in np.linspace(lo, hi, max(1, math.ceil((hi-lo)/policy['along_step_m'])+1)):
                for gap in policy['front_gaps_m']:
                    xy = a+direction*along-normal*((rotated @ normal).max()+gap)
                    pose = [float(xy[0]), float(xy[1]), yaw]
                    counts = [int(inside_footprint(p[:, :2], vehicle.footprint, pose).sum()) for _, p in clouds]
                    support = sum(math.exp(-age/policy['evidence_decay_s'])*math.log1p(n) for (age, _), n in zip(clouds, counts))
                    distance = float(np.linalg.norm(xy-ego[:2]))
                    turn = abs(math.atan2(math.sin(yaw-ego[2]), math.cos(yaw-ego[2])))
                    score = weights['observed_returns']*support+weights['approach_m']*distance+weights['turn_rad']*turn
                    pool.append(dict(edge_id=edge_id, along_m=float(along), gap_m=gap, pose=pose,
                                     nonself_voxels=counts, score=score))
    selected = []
    def separated(row):
        return all(np.linalg.norm(np.asarray(row['pose'][:2])-other['pose'][:2]) >= policy['min_along_separation_m']
                   for other in selected)
    # Preserve side diversity first; no side is declared usable by this heuristic.
    for edge_id in range(4):
        choices = [r for r in pool if r['edge_id'] == edge_id and separated(r)]
        if choices: selected.append(min(choices, key=lambda r: r['score']))
    for row in sorted(pool, key=lambda r: r['score']):
        if len(selected) >= 8: break
        if separated(row): selected.append(row)
    selected.sort(key=lambda r: (r['edge_id'], r['along_m'], r['gap_m']))
    candidates = [dict(id=i, pose=r['pose'], motion='forward', geometry='unknown') for i, r in enumerate(selected)]
    context = dict(kind='table_outline_hypothesis', pose_reference='base_link_hypothesis',
        reference_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        reference_status='historical_image_reference_current_platform_not_confirmed',
        footprint_xy_m=vehicle.footprint.tolist(), base_from_lidar=vehicle.base_from_lidar.tolist(),
        table_prior=copy.deepcopy(prior), sampling_policy=copy.deepcopy(policy), evidence_sources=sources,
        candidate_evidence=[dict(id=i, edge_id=r['edge_id'], front_gap_m=r['gap_m'], nonself_voxels=r['nonself_voxels']) for i, r in enumerate(selected)])
    validate_context(context, dict(current, candidates=candidates))
    return context, candidates, dict(pool_size=len(pool), selected=selected,
        ranking='return evidence + straight distance + heading; not route cost', executable=False)


def _prior_model_context(context, world_from_lidar):
    table = context['table_prior']
    return dict(candidate_pose_reference=context['pose_reference'],
        historical_footprint_xy_m=context['footprint_xy_m'],
        ego_base_pose_hypothesis=base_pose(world_from_lidar, context['base_from_lidar']).tolist(),
        table_prior=dict(id=table['id'], polygon_xy_m=table['polygon_xy_m'], status='approximate_not_measured',
                         available_after_s=table['available_after_s']),
        seating_relation=dict(footrest_under_table_allowed=context['sampling_policy']['footrest_under_table_allowed'],
                              exact_perpendicular_required=False),
        observed_subtable_returns=dict(sources=[dict(observation_id=s['observation_id'], stamp_s=s['stamp_s'])
                                               for s in context['evidence_sources']],
            below_edge_band_m=context['sampling_policy']['below_edge_band_m'], candidates=context['candidate_evidence']),
        limitation='Approximate prior; unconfirmed vehicle; counts are partial evidence, zero is not free; route and clearance unknown')


def observation_goals(context, observation, camera, camera_from_lidar):
    """Potential viewing poses only: FOV prediction is not occlusion/route proof."""
    prior = context['table_prior']; validate_context(context, observation)
    footprint = context['footprint_xy_m']; base_from_lidar = np.asarray(context['base_from_lidar'])
    ego = base_pose(observation['world_from_lidar'], base_from_lidar)
    targets = []
    for _, a, b, _, _ in table_edges(prior):
        targets.extend(np.c_[np.linspace(a, b, 9), np.full(9, prior['edge_z_m'])])
    targets = np.asarray(targets); result = []
    k = np.asarray(camera['K']).reshape(3, 3)
    for side, a, b, _, normal in table_edges(prior):
        yaw = math.atan2(normal[1], normal[0]); front = (footprint_world(footprint, [0, 0, yaw]) @ normal).max()
        for clearance in (.6, 1.0):
            xy = (a+b)/2-normal*(front+clearance)
            hypothetical = np.eye(4); c, s = math.cos(yaw), math.sin(yaw)
            hypothetical[:2, :2] = [[c, -s], [s, c]]; hypothetical[:2, 3] = xy; hypothetical[2, 3] = ego[2, 3]
            camera_from_world = np.asarray(camera_from_lidar) @ np.linalg.inv(hypothetical @ base_from_lidar)
            optical = targets @ camera_from_world[:3, :3].T+camera_from_world[:3, 3]
            pixels = optical @ k.T; pixels = pixels[:, :2]/np.maximum(optical[:, 2:3], 1e-6)
            visible = ((optical[:, 2] > .2) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera['width']) &
                       (pixels[:, 1] >= 0) & (pixels[:, 1] < camera['height']))
            distance = float(np.linalg.norm(xy-ego[:2, 3])); coverage = float(visible.mean())
            result.append(dict(id='V'+str(len(result)), edge_id=side, pose=[*map(float, xy), yaw],
                assumed_front_clearance_m=clearance,
                predicted_fov_coverage=coverage, straight_distance_m=distance, rank_cost=distance+2*(1-coverage),
                geometry='unknown', occlusion='unknown', executable=False))
    return sorted(result, key=lambda r: r['rank_cost'])


def check_navigation(candidate_set,scene,planning,motion,vehicle,*,now):
    """Bounded offline preflight of at most eight goals, without changing candidates."""
    from navigation.tasks import TaskPolicy
    from navigation.planning.reference import planning_config
    from navigation.planning.planner import Planner
    from .inference import binding
    task=TaskPolicy('seating-preflight','seating',planning,vehicle)
    task.observe(scene,candidates=candidate_set)  # Require the actual scene identity/clock.
    context=candidate_set.get('context')
    if context and (not np.allclose(context['footprint_xy_m'],vehicle.footprint) or
                    not np.allclose(context['base_from_lidar'],vehicle.base_from_lidar)):
        raise ValueError('Seating and navigation vehicle hypotheses differ')
    cfg=planning_config(planning)
    cfg['motion_constraints']={k:motion[k] for k in ('min_turning_radius_m','allow_reverse')}
    planner=Planner(cfg,vehicle,require_actual_start=True);results=[]
    for candidate in candidate_set['candidates']:
        if candidate['geometry']=='blocked' or candidate['motion']=='wait' or (candidate['motion']=='reverse' and not motion['allow_reverse']):
            results.append(dict(id=candidate['id'],status='not_planned',reason='blocked_wait_or_reverse_not_permitted'))
            continue
        request=dict(task_id='seating-preflight',request_id='candidate-'+str(candidate['id']),goal_revision=candidate_set['revision'],
            mode='local',frame_id=scene.record['frame_id'],target_reference='base_link',goal=candidate['pose'],
            goal_behavior='stop',motion_intent='reverse_required' if candidate['motion']=='reverse' else 'forward_preferred',
            candidate_binding=binding(candidate_set),candidate_id=candidate['id'])
        planner.reset()  # Candidate order must not supply a warm path to a different goal.
        result=planner.query(request,scene=scene,now=now,clock_domain=scene.record['clock_domain'])
        results.append(dict(id=candidate['id'],status=result['status'],result=result))
    return dict(candidate_binding=binding(candidate_set),scene_binding=scene.record['input_fingerprint'],
        motion_constraints=cfg['motion_constraints'],candidates=results,executable=False,
        scope='offline_current_geometry_path_diagnostic_not_execution_or_seating_completion')
