"""Convex footprint/grid intersection and bounded swept-pose checking."""
import json
from pathlib import Path
import numpy as np
from perception.detection_tracking.inference.geometry import validate_transform


class VehicleGeometry:
    def __init__(self, footprint, base_from_lidar, *, source, assumed=True):
        self.footprint = np.array(footprint, dtype=float, copy=True)
        self.base_from_lidar = validate_transform(base_from_lidar).copy()
        p = self.footprint
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('Expected a finite convex footprint polygon')
        edges = np.roll(p, -1, axis=0)-p
        crosses = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
        if not (np.all(crosses > 0) or np.all(crosses < 0)):
            raise ValueError('Footprint must be strictly convex and ordered')
        self.source, self.assumed = source, bool(assumed)
        self.radius = float(np.linalg.norm(p, axis=1).max())
        self.inscribed_radius = float(np.min(np.abs(edges[:,0]*p[:,1]-edges[:,1]*p[:,0])/np.linalg.norm(edges,axis=1)))
        self.footprint.flags.writeable=False
        self.base_from_lidar.flags.writeable=False
        self.lidar_from_base=np.linalg.inv(self.base_from_lidar)
        self.lidar_from_base.flags.writeable=False

    @classmethod
    def historical(cls, path):
        x = json.loads(Path(path).read_text())
        transform = x['jetson_urdf']['base_from_rslidar']
        if transform['rpy_rad'] != [0., 0., 0.]:
            raise ValueError('Historical geometry loader requires its documented zero rotation')
        matrix = np.eye(4); matrix[:3, 3] = transform['translation_m']
        return cls(x['costmap']['footprint_xy_m'], matrix,
                   source=dict(path=str(path), image=x['source_image'], status=x['status']))

    def pose(self, world_from_lidar):
        transform = validate_transform(world_from_lidar) @ self.lidar_from_base
        return np.array([*transform[:2, 3], np.arctan2(transform[1, 0], transform[0, 0])])

    def polygon(self, pose):
        x, y, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        return self.footprint @ np.array([[c, s], [-s, c]])+np.array([x, y])


def footprint_status(snapshot, geometry, pose, margin):
    """SAT intersection includes touched cells and a conservative square margin."""
    return polygon_status(snapshot, geometry.polygon(pose), margin)


def polygon_status(snapshot, polygon, margin):
    """Classify an entire convex polygon, including touched grid cells."""
    res = float(snapshot.metadata['resolution'])
    origin = np.asarray(snapshot.metadata['origin_xy'])
    occupancy = snapshot.layers['occupancy']; height, width = occupancy.shape
    lower = np.floor((polygon.min(axis=0)-margin-origin)/res).astype(int)
    upper = np.floor((polygon.max(axis=0)+margin-origin)/res).astype(int)
    if np.any(lower < 0) or upper[0] >= width or upper[1] >= height:
        return 'outside'
    if np.all(occupancy[lower[1]:upper[1]+1, lower[0]:upper[0]+1] == 0):
        return 'free'
    yy, xx = np.mgrid[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
    centres = np.column_stack((xx.ravel(), yy.ravel()))*res+origin+res/2
    overlap = polygon_cells(polygon,centres,res/2+margin)
    values = occupancy[yy.ravel()[overlap], xx.ravel()[overlap]]
    if np.any(values == 100): return 'occupied'
    if np.any(values == -1): return 'unknown'
    return 'free'


def polygon_cells(polygon, centres, half_size):
    """Conservative SAT intersections shared by individual poses and kernels."""
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    edges = np.roll(polygon, -1, axis=0)-polygon
    axes.extend(np.column_stack((-edges[:, 1], edges[:, 0])))
    overlap = np.ones(len(centres), bool)
    for axis in axes:
        projection = polygon @ axis
        radius = half_size*np.abs(axis).sum()
        points = centres @ axis
        overlap &= (points+radius >= projection.min()-1e-10) & (points-radius <= projection.max()+1e-10)
    return overlap


def footprint_details(snapshot, geometry, pose, margin):
    """Explain unavailable evidence without relabeling any unknown cell."""
    polygon=geometry.polygon(pose);m=snapshot.metadata;r=m['resolution'];origin=np.asarray(m['origin_xy'])
    low=np.floor((polygon.min(0)-margin-origin)/r).astype(int)
    high=np.floor((polygon.max(0)+margin-origin)/r).astype(int)
    height,width=snapshot.layers['occupancy'].shape
    if (low<0).any() or high[0]>=width or high[1]>=height:return dict(status='outside')
    y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
    keep=polygon_cells(polygon,np.column_stack((x.ravel(),y.ravel()))*r+origin+r/2,r/2+margin)
    x,y=x.ravel()[keep],y.ravel()[keep];values=snapshot.layers['occupancy'][y,x];unknown=values==-1
    result=dict(cells=int(len(values)),free=int((values==0).sum()),occupied=int((values==100).sum()),unknown=int(unknown.sum()))
    centres=np.column_stack((x,y))*r+origin+r/2
    physical=polygon_cells(polygon,centres,r/2)
    result['occupied_intersecting_footprint']=int(((values==100)&physical).sum())
    result['occupied_only_in_margin']=int(((values==100)&~physical).sum())
    # This is a geometric diagnostic, never permission to clear the robot area.
    edges=np.roll(polygon,-1,axis=0)-polygon
    sign=np.sign(np.cross(edges[0],edges[1]))
    inside=np.ones(len(centres),bool)
    for point,edge in zip(polygon,edges):
        normal=sign*np.array([-edge[1],edge[0]])
        inside&=((centres-point)@normal-r/2*np.abs(normal).sum()>=1e-10)
    result['unknown_fully_inside_footprint']=int((unknown&inside).sum())
    result['unknown_on_footprint_boundary']=int((unknown&~inside).sum())
    if 'ground_support' in snapshot.layers:
        support=snapshot.layers['ground_support'][y,x]
        result['unknown_without_ground_support']=int((unknown&~support).sum())
    if 'coverage' in snapshot.layers and unknown.any():
        coverage=snapshot.layers['coverage'][y[unknown],x[unknown]]
        coverage=coverage[np.isfinite(coverage)]
        if len(coverage): result['unknown_vertical_coverage_range']=[float(coverage.min()),float(coverage.max())]
    if 'observed' in snapshot.layers:
        result['unknown_with_observed_evidence']=int((unknown&snapshot.layers['observed'][y,x]).sum())
    return result


def wrap(angle):
    return (angle+np.pi) % (2*np.pi)-np.pi


def path_poses(xy, start_yaw, goal_yaw=None):
    """Explicit rotations at vertices plus straight segments; geometry only."""
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-10]]
    if len(xy) > 2:
        a, b = np.diff(xy, axis=0)[:-1], np.diff(xy, axis=0)[1:]
        straight = (np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]) < 1e-10) & ((a*b).sum(axis=1) > 0)
        xy = xy[np.r_[True, ~straight, True]]
    poses = [[*xy[0], float(start_yaw)]]
    for a, b in zip(xy[:-1], xy[1:]):
        if np.linalg.norm(b-a) < 1e-10: continue
        heading = float(np.arctan2(*(b-a)[::-1]))
        poses.extend([[*a, heading], [*b, heading]])
    if goal_yaw is not None: poses.append([*xy[-1], float(goal_yaw)])
    # Ensure even a zero-length route checks the actual starting footprint.
    if len(poses) == 1: poses.append(poses[0])
    return np.asarray(poses)


def compact_poses(poses):
    """Remove redundant straight states without changing rotations or reversing."""
    poses=np.asarray(poses,dtype=float)
    if len(poses)<3:return poses
    delta=np.diff(poses[:,:2],axis=0);turn=wrap(np.diff(poses[:,2]))
    keep=np.r_[True,(np.linalg.norm(delta,axis=1)>1e-10)|(np.abs(turn)>1e-10)]
    poses=poses[keep]
    if len(poses)<3:return poses
    a,b=np.diff(poses[:,:2],axis=0)[:-1],np.diff(poses[:,:2],axis=0)[1:]
    straight=(np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])<1e-10)&((a*b).sum(axis=1)>0)
    same=(np.abs(wrap(np.diff(poses[:,2])))<1e-10)
    return poses[np.r_[True,~(straight&same[:-1]&same[1:]),True]]


def swept_check(snapshot, geometry, poses, margin, max_samples, *, prefix=True):
    """Certify whole translation sweeps; subdivide only blocked or uncertain intervals."""
    from scipy.spatial import ConvexHull
    poses=np.asarray(poses,dtype=float)
    checked = 1; distance = 0.; last = None
    status=footprint_status(snapshot,geometry,poses[0],margin)
    if status!='free':return dict(status=status,checked_samples=checked,valid_distance_m=0.,failed_segment=0,
                                  failed_pose=poses[0].tolist(),last_valid_pose=None)
    last=poses[0].tolist()
    for index, (a, b) in enumerate(zip(poses[:-1], poses[1:])):
        delta = b[:2]-a[:2]; turn = wrap(b[2]-a[2])
        motion=float(np.linalg.norm(delta))+geometry.radius*abs(turn)
        pending=[(0.,1.)]
        verified_t=0.;known_failure=None
        while pending:
            low,high=pending.pop();t=(low+high)/2
            pose=np.r_[a[:2]+delta*t,a[2]+turn*t]
            if checked>=max_samples:
                status='budget_exceeded'
            else:
                enclosure=motion*(high-low)/2
                translation=abs(turn)<1e-12 and np.linalg.norm(delta)>1e-12
                if translation:
                    endpoints=np.vstack([geometry.polygon(np.r_[a[:2]+delta*u,a[2]]) for u in (low,high)])
                    polygon=endpoints[ConvexHull(endpoints).vertices]
                    status=polygon_status(snapshot,polygon,margin)
                else:
                    status=footprint_status(snapshot,geometry,pose,margin+enclosure)
                checked+=1
                if status=='free':
                    verified_t=high;last=np.r_[a[:2]+delta*high,a[2]+turn*high].tolist();continue
                if translation and not prefix:
                    return dict(status=status,checked_samples=checked,valid_distance_m=distance,
                        failed_segment=index,failed_pose=None,last_valid_pose=last,failure_location='convex_translation_sweep')
                actual='free'
                if checked<max_samples:
                    actual=footprint_status(snapshot,geometry,pose,margin);checked+=1
                    if actual!='free':known_failure=(actual,pose.copy())
                    if actual!='free' and not prefix:
                        return dict(status=actual,checked_samples=checked,valid_distance_m=distance,
                            failed_segment=index,failed_pose=pose.tolist(),last_valid_pose=last)
                if enclosure>1e-4 and checked<max_samples:
                    pending.extend([(t,high),(low,t)]);continue
                status=actual if actual!='free' else 'uncertified_clearance'
                if status=='uncertified_clearance' and known_failure is not None:
                    status,pose=known_failure
            return dict(status=status,checked_samples=checked,
                valid_distance_m=distance+float(np.linalg.norm(delta))*verified_t,
                failed_segment=index,failed_pose=pose.tolist(),last_valid_pose=last)
        distance += float(np.linalg.norm(delta))
    return dict(status='free', checked_samples=checked, valid_distance_m=distance, last_valid_pose=last)
