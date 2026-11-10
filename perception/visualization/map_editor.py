"""Interactive manual editor for a copied ROLL map or static occupancy map."""
import argparse
from pathlib import Path

import numpy as np

from .edit_geometry import grid_polygon_mask, polygon_mask


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--localization-map', help='完整 ROLL 定位地图')
    group.add_argument('--static-map', help='静态环境地图包')
    p.add_argument('--out', required=True, help='新输出目录；不覆盖已有地图')
    return p


def make_figure(source, output, *, localization=False):
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import font_manager
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    from PyQt5 import QtCore, QtGui, QtWidgets
    from perception.environment.storage import StaticEnvironmentMap
    from perception.environment import manual_edit as static_editor
    from perception.mapping_localization import manual_edit as roll_editor
    from perception.mapping_localization.results import read_pcd

    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('输出目录已存在，不能覆盖')
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError('输出不能位于源地图内部')
    binding = roll_editor.source_binding(source) if localization else static_editor.source_binding(source)
    if localization:
        points = read_pcd(source/'GlobalMap.pcd', 250000)
        if not len(points):
            raise ValueError('显示点云为空')
        rows = np.loadtxt(source/'keyframes/poses.txt', ndmin=2)
        offset = 1 if rows.shape[1] == 9 else 0
        trajectory = rows[:, offset:offset+2]
        data = None
    else:
        asset = StaticEnvironmentMap(source)
        snapshot = asset.snapshot
        data = snapshot.layers['occupancy'].copy()
        origin = np.asarray(snapshot.metadata['origin_xy'])
        resolution = snapshot.metadata['resolution']
        height, width = data.shape
        points = trajectory = None
    vertices, selections, undo = [], [], []
    artists = []
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title('三维定位地图编辑' if localization else '静态地图编辑')
    if localization:
        cloud = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=.5, cmap='viridis', rasterized=True)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color='#ec8c18', lw=.8)
        lower, upper = np.percentile(points[:, :2], [1, 99], axis=0)
        pad = np.maximum((upper-lower)*.08, 2.)
        ax.set_xlim(lower[0]-pad[0], upper[0]+pad[0]);ax.set_ylim(lower[1]-pad[1], upper[1]+pad[1])
        fig.colorbar(cloud, ax=ax, label='地图 Z（米）')
    else:
        ax.set_facecolor('#64717e')  # Outside the map, distinct from unknown and free cells.
        colors = ListedColormap(['#9ca9b7', '#fafafa', '#242c36'])
        display = np.where(data == -1, 0, np.where(data == 0, 1, 2))
        raster = ax.imshow(display, origin='lower', interpolation='nearest', cmap=colors, vmin=0, vmax=2,
                           extent=[origin[0], origin[0]+width*resolution, origin[1], origin[1]+height*resolution])
    # Keep metres equally scaled while letting the axes fill the available window.
    ax.set_aspect('equal', adjustable='datalim');ax.set_xlabel('地图 X（米）');ax.set_ylabel('地图 Y（米）')
    status = fig.text(.02, .02, '')
    window = fig.canvas.manager.window
    if fig.canvas.manager.toolbar is not None:
        fig.canvas.manager.toolbar.hide()
    toolbar = QtWidgets.QToolBar('人工编辑', window)
    toolbar.setObjectName('manual_map_editor')
    mode = QtWidgets.QComboBox();mode.setObjectName('edit_mode')
    if localization:
        mode.addItem('多边形选择三维删除区域', 'polygon')
    else:
        mode.addItem('画笔', 'brush');mode.addItem('多边形', 'polygon')
    toolbar.addWidget(mode)
    value = None
    if not localization:
        value = QtWidgets.QComboBox();value.setObjectName('edit_state')
        for label, cell in (('空闲（人工假定）', 0), ('占据', 100), ('未知', -1)):
            value.addItem(label, cell)
        toolbar.addWidget(value)
        toolbar.addWidget(QtWidgets.QLabel('半径（米）'))
        radius = QtWidgets.QDoubleSpinBox();radius.setRange(.05, 20.);radius.setDecimals(2);radius.setValue(.3)
        radius.setObjectName('brush_radius');toolbar.addWidget(radius)
    else:
        toolbar.addWidget(QtWidgets.QLabel('高度 Z 范围（米）'))
        zmin = QtWidgets.QDoubleSpinBox();zmax = QtWidgets.QDoubleSpinBox()
        for box in (zmin,zmax):box.setRange(-10000.,10000.);box.setDecimals(2);box.setSingleStep(.1);toolbar.addWidget(box)
        zmin.setValue(float(np.floor(points[:, 2].min()*100)/100));zmax.setValue(float(np.ceil(points[:, 2].max()*100)/100))
        zmin.setObjectName('z_min');zmax.setObjectName('z_max')
    back = toolbar.addAction('撤销');back.setShortcut(QtGui.QKeySequence('Ctrl+Z'));back.setObjectName('edit_undo')
    clear = toolbar.addAction('取消未完成多边形');clear.setObjectName('edit_cancel_polygon')
    save = toolbar.addAction('保存为新地图');save.setShortcut(QtGui.QKeySequence('Ctrl+S'));save.setObjectName('edit_save')
    reset_view = toolbar.addAction('恢复视图');reset_view.setObjectName('reset_view')
    window.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)
    fig.canvas.draw()
    initial_view = (ax.get_xlim(), ax.get_ylim())
    stroke = [None]
    pan = [None]
    pointer = [None]
    brush_outline = []
    if not localization:
        # Two contrasting outlines remain visible over both free and occupied cells.
        for color, linewidth in (('white', 3), ('black', 1)):
            circle = Circle((0, 0), radius.value(), fill=False, edgecolor=color,
                            linewidth=linewidth, visible=False, zorder=20)
            ax.add_patch(circle);brush_outline.append(circle)

    def update_pointer(event=None):
        if event is not None:
            pointer[0] = ((event.xdata, event.ydata) if event.inaxes is ax and
                          event.xdata is not None and event.ydata is not None else None)
        visible = not localization and mode.currentData() == 'brush' and pointer[0] is not None and pan[0] is None
        for circle in brush_outline:
            circle.set_visible(visible)
            if visible:
                circle.center = pointer[0]
                circle.set_radius(radius.value())
        fig.canvas.draw_idle()

    def redraw(message=''):
        for artist in artists:artist.remove()
        artists.clear()
        if localization:
            visible = (points[:, 2] >= zmin.value()) & (points[:, 2] <= zmax.value())
            cloud.set_offsets(points[visible, :2]);cloud.set_array(points[visible, 2])
            selected = np.zeros(len(points), bool)
            for item in selections:
                band = (points[:, 2] >= item['z_min_m']) & (points[:, 2] <= item['z_max_m'])
                selected[band] |= polygon_mask(points[band, :2], item['vertices_xy'])
            if selected.any():
                artists.append(ax.scatter(points[selected, 0], points[selected, 1], s=3, c='#e12c49', rasterized=True))
            for item in selections:
                closed = np.r_[np.asarray(item['vertices_xy']), np.asarray(item['vertices_xy'])[:1]]
                line, = ax.plot(closed[:, 0], closed[:, 1], color='#e12c49', lw=1);artists.append(line)
            default = f'{len(selections)} 个删除区域，预览命中 {selected.sum()} 个显示点；保存时处理所有关键帧。'
        else:
            raster.set_data(np.where(data == -1, 0, np.where(data == 0, 1, 2)))
            default = f'空闲 {int((data==0).sum())}　占据 {int((data==100).sum())}　未知 {int((data==-1).sum())}；人工空闲不等于传感器观测。'
        if vertices:
            line, = ax.plot(*np.asarray(vertices).T, 'o-', color='#e12c49', markersize=4);artists.append(line)
        status.set_text(message or default)
        fig.canvas.draw_idle()

    def remember():
        undo.append(data.copy())
        if len(undo)>30:undo.pop(0)

    def paint(start, end):
        # Fill the swept circle between motion events so fast drags leave no gaps.
        start, end = np.asarray(start), np.asarray(end)
        r=radius.value();lo=np.maximum(0,np.floor((np.minimum(start,end)-r-origin)/resolution).astype(int))
        hi=np.minimum([width,height],np.ceil((np.maximum(start,end)+r-origin)/resolution).astype(int))
        if (hi<=lo).any():return
        y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        dx=origin[0]+(x+.5)*resolution-start[0];dy=origin[1]+(y+.5)*resolution-start[1]
        delta=end-start;length2=float(delta @ delta)
        t=np.clip((dx*delta[0]+dy*delta[1])/length2,0,1) if length2 else 0
        inside=((dx-t*delta[0])**2+(dy-t*delta[1])**2<=r*r)
        data[y[inside],x[inside]]=int(value.currentData());redraw()

    def click(event):
        if event.inaxes is not ax or event.xdata is None:return
        if event.button in (2, 3):
            stroke[0]=None
            pan[0]=(event.button,event.x,event.y,ax.get_xlim(),ax.get_ylim())
            update_pointer(event)
            return
        if event.button != 1 or pan[0] is not None:return
        if not localization and mode.currentData() == 'brush':
            remember();stroke[0]=(event.xdata,event.ydata);paint(stroke[0],stroke[0]);update_pointer(event)
        else:
            vertices.append([float(event.xdata),float(event.ydata)]);redraw('继续左键添加顶点，按 Enter 完成。')

    def motion(event):
        if pan[0] is not None:
            _, x, y, xlim, ylim = pan[0]
            dx=(event.x-x)*(xlim[1]-xlim[0])/ax.bbox.width
            dy=(event.y-y)*(ylim[1]-ylim[0])/ax.bbox.height
            ax.set_xlim(xlim[0]-dx,xlim[1]-dx);ax.set_ylim(ylim[0]-dy,ylim[1]-dy)
            fig.canvas.draw_idle()
            return
        update_pointer(event)
        if stroke[0] is not None and not localization:
            if pointer[0] is None:
                stroke[0] = ()
            else:
                current = pointer[0]
                paint(stroke[0] if len(stroke[0]) else current, current)
                stroke[0] = current

    def release(event):
        if event.button == 1:stroke[0]=None
        if pan[0] is not None and event.button == pan[0][0]:
            pan[0]=None;update_pointer(event)

    def leave(event):
        stroke[0]=None;pan[0]=None;pointer[0]=None;update_pointer()

    def scroll(event):
        if pan[0] is not None or event.inaxes is not ax or event.xdata is None or not event.step:return
        stroke[0]=None
        scale=1.2**(-event.step)
        xlim,ylim=ax.get_xlim(),ax.get_ylim()
        if not 1e-4 < (xlim[1]-xlim[0])*scale < 1e7:return
        ax.set_xlim(event.xdata+(xlim[0]-event.xdata)*scale,event.xdata+(xlim[1]-event.xdata)*scale)
        ax.set_ylim(event.ydata+(ylim[0]-event.ydata)*scale,event.ydata+(ylim[1]-event.ydata)*scale)
        update_pointer(event)

    def finish_polygon():
        if len(vertices)<3:
            redraw('多边形至少需要三个顶点。');return
        try:
            from .edit_geometry import validate_polygon
            polygon = validate_polygon(vertices).tolist()
            if localization:
                if zmin.value()>zmax.value():raise ValueError('最小高度不能大于最大高度')
                selections.append(dict(vertices_xy=polygon,z_min_m=zmin.value(),z_max_m=zmax.value()))
            else:
                mask=grid_polygon_mask(data.shape,origin,resolution,polygon)
                if not mask.any():raise ValueError('选区未覆盖任何栅格中心')
                remember();data[mask]=int(value.currentData())
            vertices.clear();redraw()
        except ValueError as error:redraw(str(error))

    def undo_edit():
        if vertices:vertices.pop()
        elif localization and selections:selections.pop()
        elif not localization and undo:data[:]=undo.pop()
        redraw()

    def save_edit():
        if vertices:
            redraw('请完成或取消当前多边形。');return
        try:
            result = (roll_editor.save_manual_map(source,output,selections,expected_binding=binding) if localization
                      else static_editor.save_manual_map(source,output,data,expected_binding=binding))
            text = ('已保存新地图：'+str(output))
            if localization:
                text += '\n已删除关键帧点：'+str(result['removed_points'])+'。原地图对应的静态地图和定位区域须重新生成或绑定。'
            else:
                text += '\n人工空闲不构成传感器观测；请检查局部障碍及地图来源。'
            redraw(text);QtWidgets.QMessageBox.information(window,'保存成功',text)
        except (OSError, ValueError, KeyError) as error:
            redraw(str(error));QtWidgets.QMessageBox.warning(window,'保存失败',str(error))

    def key(event):
        if event.key=='enter':finish_polygon()
        elif event.key=='backspace' and vertices:vertices.pop();redraw()
        elif event.key=='escape':vertices.clear();redraw()

    matplotlib.rcParams['keymap.save'] = []
    matplotlib.rcParams['keymap.pan'] = []
    matplotlib.rcParams['keymap.zoom'] = []
    fig.canvas.mpl_connect('button_press_event', click)
    fig.canvas.mpl_connect('motion_notify_event', motion)
    fig.canvas.mpl_connect('button_release_event', release)
    fig.canvas.mpl_connect('figure_leave_event', leave)
    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('key_press_event', key)
    back.triggered.connect(undo_edit);clear.triggered.connect(lambda: (vertices.clear(),redraw()))
    save.triggered.connect(save_edit)
    reset_view.triggered.connect(lambda: (ax.set_xlim(*initial_view[0]), ax.set_ylim(*initial_view[1]), fig.canvas.draw_idle()))
    if localization:
        zmin.valueChanged.connect(lambda _:redraw());zmax.valueChanged.connect(lambda _:redraw())
        ax.set_title('左键绘制多边形并按 Enter；滚轮缩放，右键或中键拖动平移。红色为删除预览。')
    else:
        mode.currentIndexChanged.connect(lambda _: (stroke.__setitem__(0,None),update_pointer()))
        radius.valueChanged.connect(lambda _:update_pointer())
        ax.set_title('左键画笔或多边形；滚轮缩放，右键或中键拖动平移。')
    redraw()
    return fig


def main():
    args = parser().parse_args()
    make_figure(args.localization_map or args.static_map, args.out, localization=bool(args.localization_map))
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    main()
