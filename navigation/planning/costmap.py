"""Private costs from an observed grid or an explicit conditional-reference grid."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def build_costmap(snapshot, radius, config, *, return_clearance=False):
    cells = snapshot.layers['occupancy']
    if cells.size > config['max_cells'] or not np.isin(cells, [-1, 0, 100]).all():
        raise ValueError('Unsupported occupancy or grid budget exceeded')
    resolution = float(snapshot.metadata['resolution'])
    # Exterior is blocked. Subtract both cell half-diagonals so an accepted cell
    # contains only circle-safe positions, not just a safe cell centre.
    costs = np.full(cells.shape, 254, np.uint8)
    distance=np.zeros(cells.shape,np.float32)
    y,x=np.nonzero(cells==0)
    if not len(x):return (costs,distance) if return_clearance else costs
    # Large exported maps include extensive unknown padding. EDT outside the
    # free-space bounding box is identically zero. Compute only its exact ROI,
    # keeping a halo for nearby conditional obstacles and the blocked exterior.
    halo=int(np.ceil((radius+config['safety_margin_m']+resolution)/resolution))+1
    region=np.s_[max(0,int(y.min())-halo):min(cells.shape[0],int(y.max())+halo+1),
                 max(0,int(x.min())-halo):min(cells.shape[1],int(x.max())+halo+1)]
    local=cells[region]
    field=distance_transform_edt(np.pad(local==0,1))[1:-1,1:-1]*resolution
    distance[region]=field
    available=field-np.sqrt(2)*resolution-radius-config['safety_margin_m']
    free=(local==0)&(available>1e-8)
    values=costs[region]
    values[free]=np.rint(180*config.get('clearance_cost_gain',1.)*
                         np.exp(-available[free]/config['clearance_cost_scale_m'])).astype(np.uint8)
    conditional = snapshot.layers.get('conditional_obstacle')
    if conditional is not None and conditional[region].any():
        penalty = config['conditional_obstacle_cost']
        # Propagate risk to proposals whose circumscribed body can touch a
        # conditional cell; the exact route still gets a footprint audit.
        near = distance_transform_edt(~conditional[region])*resolution <= radius+config['safety_margin_m']+resolution
        values[free & near] = np.maximum(values[free & near],penalty)
    return (costs,distance) if return_clearance else costs
