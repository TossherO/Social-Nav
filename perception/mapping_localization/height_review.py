"""Read-only height and revisit diagnostics for an offline ROLL map candidate."""
import argparse
import json
from pathlib import Path

import numpy as np


def _nearby_pairs(xy, radius):
    """Find 2-D neighbours with a grid, without an extra SciPy dependency."""
    buckets = {}
    pairs = []
    for index, point in enumerate(xy):
        cell = tuple(np.floor(point / radius).astype(np.int64))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((cell[0]+dx, cell[1]+dy), ()):
                    if np.linalg.norm(point - xy[other]) <= radius:
                        pairs.append((other, index))
        buckets.setdefault(cell, []).append(index)
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def review(map_path, *, radius_m=1., separation_s=60., log_path=None):
    path=Path(map_path)
    if not (path/'COMPLETE').is_file():raise ValueError('A complete ROLL map is required')
    poses=np.loadtxt(path/'keyframes/poses.txt',ndmin=2)
    if poses.shape[1]!=9 or not len(poses) or not np.isfinite(poses).all() or np.any(np.diff(poses[:,0])<=0):
        raise ValueError('Invalid ordered keyframe poses')
    if not np.isfinite([radius_m,separation_s]).all() or radius_m<=0 or separation_s<=0:
        raise ValueError('Positive finite revisit radius and time separation are required')
    pairs=_nearby_pairs(poses[:,1:3],radius_m)
    pairs=pairs[(poses[pairs[:,1],0]-poses[pairs[:,0],0])>=separation_s]
    dz=poses[pairs[:,1],3]-poses[pairs[:,0],3]
    distance=np.linalg.norm(poses[pairs[:,1],1:3]-poses[pairs[:,0],1:3],axis=1)
    top=[]
    for index in np.argsort(np.abs(dz))[::-1][:12]:
        a,b=pairs[index];top.append(dict(keyframes=[int(poses[a,7]),int(poses[b,7])],
            stamps_s=[float(poses[a,0]),float(poses[b,0])],xy_m=[poses[a,1:3].tolist(),poses[b,1:3].tolist()],
            distance_xy_m=float(distance[index]),separation_s=float(poses[b,0]-poses[a,0]),
            signed_height_difference_m=float(dz[index])))
    matrix=np.column_stack((poses[:,1:3],np.ones(len(poses))))
    plane=np.linalg.lstsq(matrix,poses[:,3],rcond=None)[0]
    residual=poses[:,3]-matrix@plane
    result=dict(keyframes=len(poses),scope='optimized_keyframe_lidar_poses_not_ground_truth',
        keyframe_height_range_m=[float(poses[:,3].min()),float(poses[:,3].max())],
        start_end=dict(horizontal_distance_m=float(np.linalg.norm(poses[-1,1:3]-poses[0,1:3])),
            signed_height_difference_m=float(poses[-1,3]-poses[0,3]),
            heading_difference_deg=float(np.degrees(abs(np.arctan2(np.sin(poses[-1,6]-poses[0,6]),
                                                                   np.cos(poses[-1,6]-poses[0,6]))))),
            assumed_same_location=False),
        trajectory_plane=dict(coefficients=plane.tolist(),inclination_deg=float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))),
            residual_p05_p50_p95_m=np.percentile(residual,[5,50,95]).tolist()),
        revisit=dict(radius_m=radius_m,min_separation_s=separation_s,pairs=len(pairs),
            p50_p95_max_abs_height_m=np.percentile(np.abs(dz),[50,95,100]).tolist() if len(dz) else None,
            pairs_over_0_2m=int((np.abs(dz)>.2).sum()),pairs_over_0_5m=int((np.abs(dz)>.5).sum()),
            highest=top,pair_count_is_not_independent_locations=True),
        limitations=['horizontal_proximity_does_not_prove_same_place','no_independent_ground_truth'])
    if log_path is not None:
        from collections import Counter
        import re
        stages=Counter()
        for line in Path(log_path).read_text(errors='replace').splitlines():
            match=re.search(r'loop_check stage=([a-z_]+)',line)
            if match:stages[match.group(1)]+=1
        result['loop_trace']=dict(stages=dict(stages),source=str(log_path),
            note='Trace describes this replay only; absence does not prove no loop in another run')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--radius-m',type=float,default=1.);parser.add_argument('--separation-s',type=float,default=60.)
    parser.add_argument('--loop-log')
    args=parser.parse_args();out=Path(args.out)
    if out.exists() or out.is_symlink():raise FileExistsError('Refusing to overwrite an existing review')
    value=review(args.map,radius_m=args.radius_m,separation_s=args.separation_s,log_path=args.loop_log)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(out),keyframes=value['keyframes'],revisit=value['revisit']['pairs']),ensure_ascii=False))


if __name__=='__main__':main()
