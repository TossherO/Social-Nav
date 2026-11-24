"""Terrain projection for frozen previews; planning outputs remain unchanged."""
import numpy as np

from navigation.planning.inputs import navigation_grid

from .surface import (DisplaySurface,SearchMapSurface,annotation_height,
    complete_display_heights,line_segments,sample_ground,sample_poses)


def goal_geometry(request,grid):
    """Show the accepted XY/yaw intent, with no claim of reachability or execution."""
    if request is None or request['frame_id']!=grid.metadata['frame_id']:return None
    goal=request['goal'];z=sample_ground(grid,[goal[:2]],offset=.10)[0][0]
    if not np.isfinite(z):return None
    return dict(position=[float(goal[0]),float(goal[1]),float(z)],
        yaw=None if len(goal)==2 else float(goal[2]))


def preview_geometry(preview,grid,asset,settings):
    """Serialize small display geometry, never the scene or invented zero heights."""
    stamp=preview['as_of'];reference=(preview.get('reference') or {}).get('result') or {}
    trajectory=preview.get('trajectory') or {};row=dict(grid.metadata,stamp_s=stamp)
    surface=DisplaySurface(grid,row,offset=settings['path_z_m'],mode=settings['height_mode'],
        plane_radius=settings['ground_plane_radius_m'])
    static=None if asset is None else navigation_grid(asset)

    def heights(xy,frame,lift=0.):
        if frame==grid.metadata['frame_id']:
            z,source=surface.sample(xy)
        elif static is not None and frame==static.metadata['frame_id']:
            z,source=sample_ground(static,xy,settings['path_z_m'])
        else:
            z=np.full(len(xy),np.nan);source=np.zeros(len(xy),np.uint8)
        return z+lift,source

    curves={}
    for name,result,poses,lift in (
        ('reference',reference,reference.get('path',[]),0.),
        ('trajectory',trajectory,[p[1:] for p in trajectory.get('points',[])],.02)):
        frame=result.get('frame_id',grid.metadata['frame_id'])
        mapped=name=='reference' and static is not None and frame==static.metadata['frame_id']
        points=sample_poses(poses,(static if mapped else grid).metadata['resolution'])
        z,height_source=heights(points[:,:2],frame,lift)
        approximate=height_source==5
        if mapped and settings['height_mode']=='ground':
            missing=~np.isfinite(z)
            if missing.any():
                z[missing]=SearchMapSurface(asset).path_points(points[missing,:2],settings['path_z_m'])[:,2]
                approximate|=missing & np.isfinite(z)
        elif name=='reference':
            fallback=float(np.asarray(grid.metadata['world_from_lidar'])[2,3])+settings['path_z_m']
            z,filled=complete_display_heights(z,fallback)
            approximate|=filled
        xyz=np.c_[points[:,:2],z];valid=np.isfinite(z)
        end=int(np.flatnonzero(~valid)[0]) if (~valid).any() else len(z)
        curves[name]=dict(frame_id=frame,stamp_s=stamp,segments=line_segments(xyz).tolist(),
            prefix=np.c_[xyz[:end],points[:end,2]].tolist(),height_count=int(valid.sum()),point_count=len(z),
            approximate_count=int(approximate.sum()))

    frame=reference.get('frame_id',grid.metadata['frame_id'])
    start=reference.get('start')
    if start is None:
        # Failed references may not contain a start. Anchor status at the measured
        # LiDAR in its real local frame instead of assuming map coordinates.
        frame=grid.metadata['frame_id'];start=np.asarray(grid.metadata['world_from_lidar'])[:3,3]
    z=heights(np.asarray([start[:2]]),frame)[0][0]
    if not np.isfinite(z):
        fallback=float(np.asarray(grid.metadata['world_from_lidar'])[2,3])
        if static is not None and frame==static.metadata['frame_id']:
            floor=static.layers.get('ground_z')
            support=static.layers.get('ground_support')
            # Map-frame label fallback is a map statistic, never a local Z value.
            known=floor[support & np.isfinite(floor)] if floor is not None and support is not None else []
            z=annotation_height(static,start[:2],float(np.median(known)) if len(known) else 0.)
        else:z=fallback
    return dict(display_only=True,curves=curves,label_frame=frame,
        label_position=[float(start[0]),float(start[1])-2.,float(z)+2.2])
