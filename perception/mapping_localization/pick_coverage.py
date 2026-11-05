"""Draw map-localization coverage polygons without ROS or changing map geometry."""
import argparse
from pathlib import Path

import numpy as np

from .coverage import PriorCoverage, asset_digest, save_coverage
from .results import read_pcd


def make_figure(map_path, output):
    """Edit loaded and newly drawn regions through the same visible controls."""
    map_path, output = Path(map_path), Path(output)
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.path import Path as PolygonPath
    from PyQt5 import QtCore, QtGui, QtWidgets

    map_digest = asset_digest(map_path)
    points = read_pcd(map_path / 'GlobalMap.pcd', 180000)
    polygons = []
    if output.exists():
        polygons = [p.tolist() for p in PriorCoverage.load(output, asset_fingerprint=map_digest).regions]
    loaded = [True] * len(polygons)
    selected = 0 if polygons else None
    vertices, artists = [], []
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title('Localization coverage (XY metres)')
    ax.scatter(points[:, 0], points[:, 1], s=.5, c='#657888', alpha=.4, rasterized=True)
    poses = map_path / 'keyframes' / 'poses.txt'
    if poses.is_file():
        trajectory = np.loadtxt(poses, ndmin=2)
        ax.plot(trajectory[:, 1], trajectory[:, 2], color='#d77a00', linewidth=1, label='Mapping keyframes')
        ax.legend(loc='upper right')
    focus = np.concatenate(polygons) if polygons else np.percentile(points[:, :2], [1, 99], axis=0)
    lower, upper = focus.min(axis=0), focus.max(axis=0)
    padding = np.maximum((upper-lower)*.15, 2)
    ax.set_xlim(lower[0]-padding[0], upper[0]+padding[0])
    ax.set_ylim(lower[1]-padding[1], upper[1]+padding[1])
    ax.set_aspect('equal'); ax.set_xlabel('map X (m)'); ax.set_ylabel('map Y (m)')
    ax.grid(alpha=.2)
    ax.set_title('Left click: vertex | Enter: close | Backspace: undo vertex\n'
                 'Right click: select region | Delete: remove selected | Ctrl+S: save')
    status = fig.text(.02, .02, '')
    window = fig.canvas.manager.window
    controls = QtWidgets.QToolBar('区域编辑', window)
    controls.setObjectName('coverage_editor')
    controls.addWidget(QtWidgets.QLabel('区域：'))
    selector = QtWidgets.QComboBox()
    selector.setObjectName('coverage_regions')
    selector.setMinimumContentsLength(14)
    controls.addWidget(selector)
    delete_action = controls.addAction('删除选中区域')
    delete_action.setObjectName('coverage_delete')
    delete_action.setShortcut(QtGui.QKeySequence('Delete'))
    clear_action = controls.addAction('清空全部区域')
    clear_action.setObjectName('coverage_clear')
    save_action = controls.addAction('保存区域')
    save_action.setObjectName('coverage_save')
    save_action.setShortcut(QtGui.QKeySequence('Ctrl+S'))
    window.addToolBar(QtCore.Qt.BottomToolBarArea, controls)

    def redraw(message=''):
        for artist in artists:
            artist.remove()
        artists.clear()
        for index, poly in enumerate(polygons):
            patch = Polygon(poly, facecolor='#3bb273', alpha=.3,
                            edgecolor='#cc5400' if index == selected else '#147d46',
                            linewidth=2.5 if index == selected else 1)
            ax.add_patch(patch); artists.append(patch)
        if vertices:
            line, = ax.plot(*np.asarray(vertices).T, 'o-', color='#cc5400', markersize=5)
            artists.append(line)
        blocked = selector.blockSignals(True)
        selector.clear()
        for index, original in enumerate(loaded):
            selector.addItem('区域 %d（%s）' % (index+1, '原有' if original else '新绘制'))
        selector.setCurrentIndex(selected if selected is not None else -1)
        selector.blockSignals(blocked)
        delete_action.setEnabled(selected is not None)
        clear_action.setEnabled(bool(polygons or vertices))
        save_action.setEnabled(bool(polygons) and not vertices)
        status.set_text(message or f'{len(polygons)} regions; {len(vertices)} pending vertices. Orange border: selected. Coverage is not free space.')
        fig.canvas.draw_idle()

    def select(index):
        nonlocal selected
        selected = index if 0 <= index < len(polygons) else None
        redraw()

    def delete_selected():
        nonlocal selected
        if selected is None:
            return
        polygons.pop(selected)
        loaded.pop(selected)
        selected = min(selected, len(polygons)-1) if polygons else None
        redraw('Removed selected region. Draw a replacement or save remaining regions.')

    def clear_all():
        nonlocal selected
        if QtWidgets.QMessageBox.question(window, '清空全部区域',
                '清空原有和新绘制的全部区域及未闭合顶点？保存前不会修改文件。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        polygons.clear(); loaded.clear(); vertices.clear(); selected = None
        redraw('All regions cleared. Draw at least one region, close it with Enter, then save.')

    def save():
        if vertices:
            redraw('Close the pending polygon with Enter before saving.'); return
        if not polygons:
            redraw('Draw at least one region before saving.'); return
        if output.exists() and QtWidgets.QMessageBox.question(window, 'Replace coverage?', str(output),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        try:
            if asset_digest(map_path) != map_digest:
                raise ValueError('Map changed; reopen the editor before selecting coverage.')
            save_coverage(output, map_path, polygons)
            redraw(f'Saved: {output}')
        except ValueError as error:
            redraw(str(error))

    def click(event):
        if event.inaxes is not ax or fig.canvas.manager.toolbar.mode:
            return
        if event.button == 1:
            vertices.append([event.xdata, event.ydata]); redraw()
        elif event.button == 3:
            for index in reversed(range(len(polygons))):
                if PolygonPath(polygons[index]).contains_point((event.xdata, event.ydata)):
                    select(index)
                    break

    def key(event):
        nonlocal selected
        if event.key == 'enter' and len(vertices) >= 3:
            polygons.append(list(vertices)); loaded.append(False)
            vertices.clear(); selected = len(polygons)-1; redraw()
        elif event.key == 'backspace' and vertices:
            vertices.pop(); redraw()
        elif event.key == 'delete':
            delete_selected()
        elif event.key == 'ctrl+s':
            save()
    # Matplotlib's default Ctrl+S saves an image; reserve it for the coverage YAML.
    matplotlib.rcParams['keymap.save'] = []
    matplotlib.rcParams['keymap.back'] = ['alt+left']
    fig.canvas.mpl_connect('button_press_event', click)
    fig.canvas.mpl_connect('key_press_event', key)
    selector.currentIndexChanged.connect(select)
    delete_action.triggered.connect(delete_selected)
    clear_action.triggered.connect(clear_all)
    save_action.triggered.connect(save)
    redraw()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--localization-map','--map',dest='map', type=Path, required=True,help='3D localization map bundle; output is a trusted-region configuration')
    parser.add_argument('--out', type=Path, required=True, help='Coverage YAML; an existing file is loaded for editing')
    args = parser.parse_args()
    make_figure(args.map, args.out)
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    main()
