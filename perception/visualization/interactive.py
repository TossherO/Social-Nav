"""Terrain projection for frozen previews; planning outputs remain unchanged."""
import numpy as np

from .surface import DisplaySurface,annotation_height,line_segments,sample_ground,sample_poses,complete_display_heights


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

    def heights(xy,frame,lift=0.):
        if frame==grid.metadata['frame_id']:
            z,source=surface.sample(xy)
        elif asset is not None and frame==asset.manifest['frame_id']:
            z,source=sample_ground(asset.snapshot,xy,settings['path_z_m'])
        else:
            z=np.full(len(xy),np.nan);source=np.zeros(len(xy),np.uint8)
        return z+lift,source

    curves={}
    for name,result,poses,lift in (
        ('reference',reference,reference.get('path',[]),0.),
        ('trajectory',trajectory,[p[1:] for p in trajectory.get('points',[])],.02)):
        frame=result.get('frame_id',grid.metadata['frame_id'])
        points=sample_poses(poses,grid.metadata['resolution'])
        z,_=heights(points[:,:2],frame,lift)
        approximate=np.zeros(len(z),bool)
        if name=='reference':
            fallback=(annotation_height(asset.snapshot,points[0,:2],0.)+settings['path_z_m']
                      if len(points) and asset is not None and frame==asset.manifest['frame_id']
                      else float(np.asarray(grid.metadata['world_from_lidar'])[2,3])+settings['path_z_m'])
            z,approximate=complete_display_heights(z,fallback)
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
        if asset is not None and frame==asset.manifest['frame_id']:
            floor=asset.snapshot.layers['ground_z'];known=asset.snapshot.layers['ground_support']&np.isfinite(floor)
            # Map-frame label fallback is a map statistic, never a local Z value.
            z=annotation_height(asset.snapshot,start[:2],float(np.median(floor[known])) if known.any() else 0.)
        else:z=fallback
    return dict(display_only=True,curves=curves,label_frame=frame,
        label_position=[float(start[0]),float(start[1])-2.,float(z)+2.2])
