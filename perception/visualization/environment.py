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
            'Class':'rviz/Map','Name':'Static prior view (withdrawn when unassociated)',
            'Enabled':True,'Topic':prior_topic,'Alpha':.3,
            'Color Scheme':'map','Draw Behind':True})
    configuration['Visualization Manager']['Displays'].append({
        'Class':'rviz/Map','Name':'Local environment map (gray=unknown)',
        'Enabled':True,'Topic':topic+'/occupancy','Alpha':.65,
        'Color Scheme':'map','Draw Behind':True})
    for label,suffix in [('Historical free inference (provenance)','historical_free'),
                         ('Recent obstacle coverage (provenance)','recent_obstacles')]:
        configuration['Visualization Manager']['Displays'].append({
            'Class':'rviz/Map','Name':label,'Enabled':False,'Topic':topic+'/'+suffix,
            'Alpha':.5,'Color Scheme':'costmap','Draw Behind':False})
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
    if 'conditional_obstacle' in layers: categories[layers['conditional_obstacle']]=3
    if 'assumed_free' in layers:categories[layers['assumed_free']]=4
    axes[0].imshow(categories,origin='lower',extent=extent,vmin=0,vmax=4,
                   cmap=ListedColormap(['#b8c4cf','#ffffff','#263b4d','#e7a343','#f7e8a4']),interpolation='nearest')
    axes[0].set_title('Gray unknown / white observed free / dark occupied\nYellow: assumed free; orange: conditional occupied')
    for name,color in [('near_field_free','#138a67'),('near_field_obstacle','#b34a24')]:
        if name in layers and layers[name].any():
            axes[0].contour(x+(np.arange(width)+.5)*r,y+(np.arange(height)+.5)*r,layers[name],
                levels=[.5],colors=[color],linewidths=.6)
    if 'near_field_free' in layers:
        axes[0].set_title('Green: reused historical free\nRed: recent obstacle coverage; gray: unknown')
    ground = axes[1].imshow(layers['ground_z'],origin='lower',extent=extent,cmap='viridis',interpolation='nearest')
    axes[1].set_title('Supported ground height (m; history explicitly tagged)')
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
    role = meta.get('view_type') or meta.get('map_type') or 'Geometry snapshot (legacy metadata)'
    figure.suptitle('%s | t=%.3f | %s | rays: %s' % (role,meta['stamp_s'],meta['geometry_status'],meta['rays_used']))
    return figure
