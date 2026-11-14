"""Create a reviewable ROLL map candidate from a registered local bag survey.

The input mapping and fixed-map localization runs must use the same bag.  The
source maps and runs are read-only; this command never changes active assets.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from perception.environment.fusion import _scan_geometry
from perception.environment.layers import load_config
from perception.environment.occupancy import OccupancyTree
from perception.environment.poses import PoseTimeline
from perception.environment.scans import read_scans, sealed_scan_summary
from perception.environment.storage import asset_reference, content_hash
from .context import pose_matrix
from .manual_edit import _poses, _voxel, _world, read_xyzi, source_binding, write_xyzi
from .results import file_fingerprint, map_fingerprint


def _load_run(path, mode):
    path = Path(path).resolve()
    run = json.loads((path/'run.json').read_text())
    if not run.get('success') or run.get('source') != 'bag' or run.get('mode') != mode:
        raise ValueError('A successful bag '+mode+' run is required: '+str(path))
    bag = Path(run['bag']).resolve()
    if file_fingerprint(bag) != run['playback']['bag_fingerprint']:
        raise ValueError('Recorded bag no longer matches its run')
    rows = [json.loads(line) for line in (path/'localization.jsonl').read_text().splitlines() if line.strip()]
    return path,run,rows


def _sensor_parameters(run):
    parameters=run.get('algorithm_parameters') or {}
    mapping=parameters.get('mapping') or {};common=parameters.get('common') or {}
    rotation=np.asarray(mapping.get('extrinsic_R'),dtype=float)
    translation=np.asarray(mapping.get('extrinsic_T'),dtype=float)
    offset=common.get('imu_time_offset')
    if (rotation.shape not in ((9,),(3,3)) or translation.shape!=(3,) or
            not np.isfinite(rotation).all() or not np.isfinite(translation).all() or
            type(offset) not in (int,float) or not np.isfinite(offset) or
            type(mapping.get('extrinsic_est_en')) is not bool or type(common.get('time_sync_en')) is not bool):
        raise ValueError('Both runs must record sensor extrinsics and IMU timing parameters')
    if mapping['extrinsic_est_en'] or common['time_sync_en']:
        raise ValueError('Online extrinsic estimation or heuristic time sync is not supported for map update')
    return rotation.reshape(3,3),translation,float(offset)


def _registered_keyframes(local_map, timeline, frame, *, tolerance):
    rows,offset = _poses(local_map)
    if offset != 1:
        raise ValueError('New local map requires timestamped keyframes')
    times = timeline.times
    registered=[]
    for index,row in enumerate(rows):
        at = int(np.argmin(np.abs(times-row[0])))
        if abs(float(times[at]-row[0])) > tolerance:
            continue
        pose = timeline.at(float(times[at]),exact_ns=timeline.rows[at].get('stamp_ns'))
        target = timeline.registered.get(timeline.rows[at].get('stamp_ns'))
        if pose is None or target.get('map_frame') != frame:
            continue
        matrix = np.asarray(pose['world_from_lidar'])
        rpy = Rotation.from_matrix(matrix[:3,:3]).as_euler('xyz')
        registered.append((index,float(times[at]),np.r_[matrix[:3,3],rpy],int(row[-1])))
    return registered


def _prepare_evidence(old_map, mapping_run, reference_run):
    old_map = Path(old_map).resolve()
    old_binding = source_binding(old_map)
    old_manifest = yaml.safe_load((old_map/'manifest.yaml').read_text())
    source,run,rows = _load_run(mapping_run,'mapping')
    reference,ref,targets = _load_run(reference_run,'localization')
    digest = map_fingerprint(old_map)
    if run['playback']['bag_fingerprint'] != ref['playback']['bag_fingerprint']:
        raise ValueError('Mapping and localization runs must use the same bag')
    first,second=_sensor_parameters(run),_sensor_parameters(ref)
    if any(not np.allclose(a,b,atol=1e-9,rtol=0.) for a,b in zip(first,second)):
        raise ValueError('Mapping and localization runs use different sensor extrinsics or IMU timing')
    if ref['playback']['map_fingerprint'] != digest:
        raise ValueError('Localization run must use the selected unchanged map')
    local_map = source/'map'
    if map_fingerprint(local_map) != run['playback']['map_fingerprint']:
        raise ValueError('Local mapping result no longer matches its run')
    scans = source/'environment/scans.sqlite'
    sealed_scan_summary(scans)
    if file_fingerprint(scans) != run.get('environment_scans_fingerprint'):
        raise ValueError('Sealed scan evidence does not match its run')
    config=load_config()
    timeline=PoseTimeline(rows,config['poses'],registered_rows=targets)
    frame=old_manifest['frame']
    for row in targets:
        if row.get('has_map_pose') and row.get('map_frame') != frame:
            raise ValueError('Registered observations have a different map frame')
    keys=_registered_keyframes(local_map,timeline,frame,tolerance=config['poses']['keyframe_time_tolerance_s'])
    local_rows,_=_poses(local_map)
    if len(keys)<2 or len(keys)/len(local_rows)<.5:
        raise ValueError('Insufficient same-observation registered local keyframes')
    old_rows,offset=_poses(old_map)
    if offset!=1 or np.any(np.diff(old_rows[:,0])<=0):
        raise ValueError('The old map must have strictly increasing acquisition timestamps')
    if keys[0][1]<=old_rows[-1,0] or any(b[1]<=a[1] for a,b in zip(keys,keys[1:])):
        raise ValueError('Cross-session keyframe acquisition times overlap or are not increasing; no timestamps are rewritten')
    inputs=dict(mapping_run=file_fingerprint(source/'run.json'),mapping_poses=file_fingerprint(source/'localization.jsonl'),
                localization_run=file_fingerprint(reference/'run.json'),localization_poses=file_fingerprint(reference/'localization.jsonl'),
                scans=file_fingerprint(scans))
    return dict(old=old_map,old_binding=old_binding,manifest=old_manifest,old_rows=old_rows,inputs=inputs,
                local=local_map,local_binding=source_binding(local_map),source=source,run=run,reference=reference,ref=ref,
                scans=scans,timeline=timeline,keys=keys,local_keyframes=len(local_rows),config=config)


def _pose_innovation(previous_map, previous_local, current_map, current_local):
    """Compare current mapped pose with local-motion propagation at the LiDAR."""
    predicted=previous_map @ np.linalg.inv(previous_local) @ current_local
    delta=np.linalg.inv(predicted) @ current_map
    return float(np.linalg.norm(delta[:3,3])),float(Rotation.from_matrix(delta[:3,:3]).magnitude())


def _evidence(data):
    config=data['config']; timeline=data['timeline']
    old_cloud=read_xyzi(data['old']/'GlobalMap.pcd')
    if not len(old_cloud):raise ValueError('Old map has no global geometry')
    # Bound the survey by the existing map plus a small margin; this first
    # implementation does not claim new rooms outside the original map.
    lo=old_cloud[:,:3].min(axis=0)-1.;hi=old_cloud[:,:3].max(axis=0)+1.
    bounds=[float(value) for value in (*lo,*hi)]
    settings=dict(config,voxel_ttl_s=max(config['voxel_ttl_s'],float(timeline.times[-1]-timeline.times[0]+1)),
                  max_rays=config['static']['max_rays'],max_voxels=config['static']['max_voxels'],
                  hit_invalidates_free=False)
    settings['obstacle_ttl_s']=settings['voxel_ttl_s']
    tree=OccupancyTree(config['resolution'],backend='cpu')
    try:
        tree.enable_static_retention()
        tree.enable_static_review(config['dynamic']['static_observation_interval_s'],config['static']['ray_end_margin_m'])
        used=skipped=rays=0;positions=[];previous=None;max_translation=max_rotation=0.
        for record,arrays in read_scans(data['scans']):
            scan=_scan_geometry(record,arrays,timeline)
            if scan is None:
                skipped+=1;continue
            stamp,pose,_,final,points,origins=scan
            if not record['rays_used'] or origins is None:
                skipped+=1;continue
            local=np.asarray(pose['local_from_lidar'])
            if previous is not None and stamp-previous[0] <= 2.:
                distance,angle=_pose_innovation(previous[1],previous[2],final,local)
                max_translation=max(max_translation,distance);max_rotation=max(max_rotation,angle)
                if distance>.5 or angle>.35:
                    raise ValueError('Consecutive registered poses contain a large map correction jump: '
                        f'run_elapsed_s={stamp-timeline.times[0]:.3f}, stamp_s={stamp:.6f}, '
                        f'dt_s={stamp-previous[0]:.3f}, position_innovation_m={distance:.3f}, '
                        f'angle_innovation_deg={np.degrees(angle):.2f}; limits are 0.5 m and 20.05 deg')
            previous=(stamp,final,local)
            tree.update(points,final[:3,3],bounds,stamp,settings,rays=True,ray_origins=origins,read=False)
            used+=1;rays+=len(points);positions.append(final[:3,3])
        if used<3 or used/(used+skipped)<.5:
            raise ValueError('At least three and half of the sealed scans need registered moving-origin evidence')
        positions=np.asarray(positions)
        if np.linalg.norm(positions.max(axis=0)-positions.min(axis=0)) < .3:
            raise ValueError('Insufficient survey viewpoint displacement')
        voxels=tree.snapshot(previous[0])
        occupied=voxels[voxels[:,3]>=config['occupied_probability'],:3]
        supported=occupied[_masks(tree,occupied,config,new=True)] if len(occupied) else occupied
        # Nearby persistent returns veto deletion even when small registration
        # errors put a ray in the neighbouring voxel of an unchanged wall.
        hit_tree=cKDTree(supported) if len(supported) else None
        return tree,hit_tree,dict(scans_used=used,scans_skipped=skipped,endpoint_count_before_sampling=rays,
                         trajectory_extent_m=float(np.linalg.norm(positions.max(axis=0)-positions.min(axis=0))),
                         bounds=bounds,observation_interval_s=config['dynamic']['static_observation_interval_s'],
                         max_pose_innovation_m=max_translation,max_pose_innovation_rad=max_rotation,
                         stable_hit_voxels=len(supported),deletion_hit_guard_m=.25)
    except BaseException:
        tree.close();raise


def _masks(tree,world,config,*,new=False,hit_tree=None):
    votes=tree.static_votes(world)
    hits,free=votes[:,0],votes[:,1]
    minimum=config['static']['free_min_observations']
    ratio=config['static']['free_to_hit_ratio']
    if new:
        return (hits>=minimum)&(hits>=np.maximum(free,1)*ratio)
    removed=(free>=minimum)&(free>=np.maximum(hits,1)*ratio)
    if hit_tree is not None and removed.any():
        indices=np.flatnonzero(removed)
        distance,_=hit_tree.query(world[indices],distance_upper_bound=.25)
        removed[indices[np.isfinite(distance)]]=False
    return removed


def _plot(path, before, removed, added):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(12,9),constrained_layout=True)
    def draw(parts,color,label,size):
        points=np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.empty((0,3))
        if len(points):
            points=points[::max(1,int(np.ceil(len(points)/120000)))]
            ax.scatter(points[:,0],points[:,1],s=size,c=color,label=label,rasterized=True)
    draw([before[:,:3]],'#b8b8b8','Old map',.2)
    draw(removed,'#df3333','Removed candidate',1.)
    draw(added,'#1777db','Retained new features',.5)
    ax.set_aspect('equal');ax.set_xlabel('Map X (m)');ax.set_ylabel('Map Y (m)')
    ax.set_title('3D map update candidate: XY overview');ax.legend(loc='best',markerscale=5)
    fig.savefig(path,dpi=160)
    changed=[p for p in (*removed,*added) if len(p)]
    if changed:
        xy=np.concatenate(changed)[:,:2]
        low=xy.min(axis=0)-2.;high=xy.max(axis=0)+2.
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1])
        ax.set_title('3D map update candidate: changed area')
        fig.savefig(path.with_name('comparison_detail.png'),dpi=160)
    plt.close(fig)


def update(old_map,mapping_run,reference_run,output):
    output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Output already exists')
    inputs=[Path(p).resolve() for p in (old_map,mapping_run,reference_run)]
    destination=output.resolve()
    if any(destination==p or p in destination.parents or destination in p.parents for p in inputs):
        raise ValueError('Output must not contain or be inside an input')
    data=_prepare_evidence(*inputs)
    tree,hit_tree,review=_evidence(data)
    try:
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.'+output.name+'-',dir=output.parent) as temp:
            package=Path(temp)/'candidate';keydir=package/'map/keyframes';keydir.mkdir(parents=True)
            rows=[];world_parts=[];removed=[];added=[];counts=dict(old_removed=0,new_kept=0,new_skipped_keyframes=0)
            resolution=float(data['manifest'].get('visualization_resolution',.2))
            for index,row in enumerate(data['old_rows']):
                clouds=[];worlds=[]
                for kind in ('corner','surf'):
                    cloud=read_xyzi(data['old']/'keyframes'/f'{kind}{index}.pcd')
                    world=_world(cloud,row[1:7]);mask=_masks(tree,world,data['config'],hit_tree=hit_tree)
                    clouds.append(cloud[~mask]);worlds.append(world[~mask]);removed.append(world[mask])
                    counts['old_removed']+=int(mask.sum())
                if not len(clouds[1]):raise ValueError('Update would empty old surface keyframe '+str(index))
                for kind,cloud,world in zip(('corner','surf'),clouds,worlds):
                    write_xyzi(keydir/f'{kind}{len(rows)}.pcd',cloud)
                    world_parts.append(np.c_[world,cloud[:,3]])
                rows.append(row.copy())
            for index,stamp,pose,indoor in data['keys']:
                clouds=[];worlds=[]
                for kind in ('corner','surf'):
                    cloud=read_xyzi(data['local']/'keyframes'/f'{kind}{index}.pcd')
                    world=_world(cloud,pose);mask=_masks(tree,world,data['config'],new=True)
                    clouds.append(cloud[mask]);worlds.append(world[mask])
                if not len(clouds[1]):
                    counts['new_skipped_keyframes']+=1;continue
                for kind,cloud,world in zip(('corner','surf'),clouds,worlds):
                    write_xyzi(keydir/f'{kind}{len(rows)}.pcd',cloud)
                    world_parts.append(np.c_[world,cloud[:,3]]);added.append(world)
                    counts['new_kept']+=len(cloud)
                rows.append(np.r_[stamp,pose,len(rows),indoor])
            if not counts['old_removed'] and not counts['new_kept']:
                raise ValueError('No changes have sufficient recorded evidence')
            if sum(len(p) for p in world_parts)>8_000_000:raise ValueError('Map display reconstruction exceeds point budget')
            np.savetxt(keydir/'poses.txt',np.asarray(rows),fmt='%.17g')
            write_xyzi(package/'map/GlobalMap.pcd',_voxel(np.concatenate(world_parts),resolution))
            _plot(package/'comparison.png',read_xyzi(data['old']/'GlobalMap.pcd'),removed,added)
            def combine(parts):
                nonempty=[p for p in parts if len(p)]
                return np.concatenate(nonempty).astype(np.float32) if nonempty else np.empty((0,3),np.float32)
            np.savez_compressed(package/'changes.npz',removed_map_xyz=combine(removed),added_map_xyz=combine(added))
            report=dict(status='candidate_not_accepted',old_map=asset_reference(data['old']),
                old_binding=data['old_binding'],local_mapping=asset_reference(data['source']),
                fixed_localization=asset_reference(data['reference']),bag_fingerprint=data['run']['playback']['bag_fingerprint'],
                evidence=review,counts=counts,registered_keyframes=len(data['keys']),
                local_keyframes=data['local_keyframes'],
                local_map_binding=data['local_binding'],input_fingerprints=data['inputs'],
                evidence_policy=dict(resolution_m=data['config']['resolution'],
                    free_min_observations=data['config']['static']['free_min_observations'],
                    free_to_hit_ratio=data['config']['static']['free_to_hit_ratio'],
                    ray_end_margin_m=data['config']['static']['ray_end_margin_m'],
                    max_rays_per_scan=data['config']['static']['max_rays']),
                changes_sha256=content_hash(package/'changes.npz'),
                limitations=['temporal_votes_do_not_prove_independent_viewpoints','dynamic_objects_may_persist',
                             'global_registration_accuracy_not_independently_verified','static_map_and_annotations_not_migrated'])
            report_text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
            (package/'report.json').write_text(report_text)
            (package/'map/update_source.json').write_text(report_text)
            old_id=source_binding(data['old'])
            if old_id!=data['old_binding']:raise ValueError('Source map changed during update')
            if source_binding(data['local'])!=data['local_binding']:raise ValueError('Local map changed during update')
            for key,path in (('mapping_run',data['source']/'run.json'),('mapping_poses',data['source']/'localization.jsonl'),
                             ('localization_run',data['reference']/'run.json'),('localization_poses',data['reference']/'localization.jsonl'),
                             ('scans',data['scans'])):
                if file_fingerprint(path)!=data['inputs'][key]:raise ValueError('Input changed during update: '+key)
            if file_fingerprint(data['run']['bag'])!=data['run']['playback']['bag_fingerprint']:
                raise ValueError('Input bag changed during update')
            manifest=dict(format='roll_keyframes_v1',frame=data['manifest']['frame'],units='metres_radians',
                          keyframes=len(rows),visualization_resolution=resolution,source='local_bag_update_candidate',
                          update_source_sha256=content_hash(package/'map/update_source.json'))
            (package/'map/manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
            (package/'map/COMPLETE').write_text('roll_keyframes_v1\n')
            if output.exists() or output.is_symlink():raise FileExistsError('Output appeared during update')
            package.rename(output)
        return report
    finally:tree.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-map',required=True)
    parser.add_argument('--mapping-run',required=True,help='Same-bag local mapping with sealed scan journal')
    parser.add_argument('--localization-run',required=True,help='Same-bag fixed old-map localization')
    parser.add_argument('--out',required=True,help='New candidate directory; never the active map')
    args=parser.parse_args()
    print(json.dumps(update(args.old_map,args.mapping_run,args.localization_run,args.out),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
