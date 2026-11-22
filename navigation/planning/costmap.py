"""Private costs from an observed grid or an explicit conditional-reference grid."""
import numpy as np
from scipy.ndimage import distance_transform_edt
try:
    import cv2
except ImportError:
    cv2 = None


def build_costmap(snapshot, radius, config, *, return_clearance=False):
    cells = snapshot.layers['occupancy']
    error = 'Unsupported occupancy or grid budget exceeded'
    if cells.size > config['max_cells']:
        raise ValueError(error)
    free_cells = cells == 0
    if not ((cells == -1) | free_cells | (cells == 100)).all():
        raise ValueError(error)
    resolution = float(snapshot.metadata['resolution'])
    # Exterior is blocked. Subtract both cell half-diagonals so an accepted cell
    # contains only circle-safe positions, not just a safe cell centre.
    costs = np.full(cells.shape, 254, np.uint8)
    distance=np.zeros(cells.shape,np.float32)
    rows=np.flatnonzero(free_cells.any(axis=1))
    if not len(rows):return (costs,distance) if return_clearance else costs
    columns=np.flatnonzero(free_cells.any(axis=0))
    # Large exported maps include extensive unknown padding. EDT outside the
    # free-space bounding box is identically zero. Compute only its exact ROI,
    # keeping a halo for nearby conditional obstacles and the blocked exterior.
    halo=int(np.ceil((radius+config['safety_margin_m']+resolution)/resolution))+1
    region=np.s_[max(0,int(rows[0])-halo):min(cells.shape[0],int(rows[-1])+halo+1),
                 max(0,int(columns[0])-halo):min(cells.shape[1],int(columns[-1])+halo+1)]
    local_free=free_cells[region]
    field=distance_transform_edt(np.pad(local_free,1))[1:-1,1:-1]*resolution
    distance[region]=field
    available=field-np.sqrt(2)*resolution-radius-config['safety_margin_m']
    free=local_free&(available>1e-8)
    values=costs[region]
    values[free]=np.rint(180*config.get('clearance_cost_gain',1.)*
                         np.exp(-available[free]/config['clearance_cost_scale_m'])).astype(np.uint8)
    conditional = snapshot.layers.get('conditional_obstacle')
    if conditional is not None:
        conditional = conditional[region]
    if conditional is not None and conditional.any():
        # Propagate risk to proposals whose circumscribed body can touch a
        # conditional cell; the exact route still gets a footprint audit.
        limit = radius+config['safety_margin_m']+resolution
        if cv2 is not None and conditional.dtype == np.bool_ and limit <= 8*resolution:
            # Only the set inside the threshold is needed. A small exact disk
            # avoids constructing a second floating-point distance field.
            span=int(np.ceil(limit/resolution))
            y,x=np.ogrid[-span:span+1,-span:span+1]
            disk=(np.sqrt(x*x+y*y)*resolution <= limit).astype(np.uint8)
            near=cv2.dilate(conditional.view(np.uint8),disk,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        else:
            near=distance_transform_edt(~conditional)*resolution <= limit
        penalized=free & (near!=0)
        values[penalized] = np.maximum(values[penalized],config['conditional_obstacle_cost'])
    return (costs,distance) if return_clearance else costs
