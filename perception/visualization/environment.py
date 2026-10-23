"""Read-only layer overview, including unknown and incomplete vertical coverage."""
import numpy as np


def add_environment_display(configuration, topic='/perception/environment', *, prior_topic=None, target_frame=None):
    if target_frame:
        configuration['Visualization Manager'].setdefault('Views',{}).setdefault('Current',{})['Target Frame'] = target_frame
    for display in configuration['Visualization Manager']['Displays']:
        if display.get('Name') == 'Current scan (observation-bound)':
            display['Topic'] = topic+'/scan'
    if prior_topic:
        configuration['Visualization Manager']['Displays'].append({
            'Class':'rviz/Map','Name':'Historical prior (withdrawn when unassociated)',
            'Enabled':True,'Topic':prior_topic,'Alpha':.3,
            'Color Scheme':'map','Draw Behind':True})
    configuration['Visualization Manager']['Displays'].append({
        'Class':'rviz/Map','Name':'Observed environment (gray=unknown)',
        'Enabled':True,'Topic':topic+'/occupancy','Alpha':.65,
        'Color Scheme':'map','Draw Behind':True})
    return configuration


def plot_environment(snapshot):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    meta,layers = snapshot.metadata,snapshot.layers
    height,width = meta['shape']
    x,y = meta['origin_xy']
    r = meta['resolution']
    extent = [x,x+width*r,y,y+height*r]
    figure,axes = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    categories = np.where(layers['occupancy']==-1,0,np.where(layers['occupancy']==0,1,2))
    axes[0].imshow(categories,origin='lower',extent=extent,vmin=0,vmax=2,
                   cmap=ListedColormap(['#b8c4cf','#ffffff','#263b4d']),interpolation='nearest')
    axes[0].set_title('Occupancy: gray unknown / white free / dark occupied')
    ground = axes[1].imshow(layers['ground_z'],origin='lower',extent=extent,cmap='viridis',interpolation='nearest')
    axes[1].set_title('Observed ground support (m)')
    figure.colorbar(ground,ax=axes[1],shrink=.7)
    coverage = axes[2].imshow(layers['coverage'],origin='lower',extent=extent,vmin=0,vmax=1,
                             cmap='YlOrRd',interpolation='nearest')
    axes[2].set_title('Free evidence in collision-height column')
    figure.colorbar(coverage,ax=axes[2],shrink=.7)
    pose = meta.get('world_from_lidar')
    if pose is not None:
        pose = np.asarray(pose)
        for ax in axes:
            ax.plot(pose[0,3],pose[1,3],'o',color='#ef571f',markersize=6)
            ax.arrow(pose[0,3],pose[1,3],pose[0,0],pose[1,0],color='#ef571f',width=.05)
    for ax in axes:
        ax.set_xlabel(meta['frame_id']+' x (m)')
        ax.set_ylabel('y (m)')
    figure.suptitle('t=%.3f | %s | rays: %s' % (meta['stamp_s'],meta['geometry_status'],meta['rays_used']))
    return figure


def plot_memory(snapshot, observations):
    """Overlay only recorded observations; breaks never become invented routes."""
    figure = plot_environment(snapshot)
    axes = figure.axes[:3]
    previous = None
    positions = []
    ordered = sorted(observations, key=lambda item: item['stamp_s'])
    labelled = set(np.linspace(0, len(ordered)-1, min(16, len(ordered)), dtype=int))
    for index, row in enumerate(ordered):
        camera = row.get('world_from_camera')
        pose = np.asarray(camera if camera is not None else row['world_from_lidar'])
        point = pose[:2, 3]
        positions.append(point)
        forward = pose[:2, 2 if camera is not None else 0]
        length = np.linalg.norm(forward)
        if length > 1e-6:
            forward = forward/length*.6
        for axis in axes:
            axis.plot(*point, 'o', color='#d44716', markersize=4)
            if index in labelled:
                axis.annotate(str(index+1), point, xytext=(4, 4), textcoords='offset points', color='#a43712', fontsize=8)
            if length > 1e-6 and index in labelled:
                axis.arrow(*point, *forward, color='#d44716', width=.025, length_includes_head=True)
            if previous is not None and previous[0] == row['segment']:
                axis.plot([previous[1][0], point[0]], [previous[1][1], point[1]], '--', color='#d44716', linewidth=1)
        previous = row['segment'], point
    supported = np.argwhere(snapshot.layers['ground_support'] | (snapshot.layers['occupancy'] == 100))
    if len(supported):
        region = supported[:, ::-1]*snapshot.metadata['resolution']+snapshot.metadata['origin_xy']
        if positions:
            region = np.r_[region, positions]
        low, high = region.min(axis=0)-2, region.max(axis=0)+2
        for axis in axes:
            axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1])
    figure.suptitle('Supported-region view / historical observations (orange); no reachability claim')
    return figure
