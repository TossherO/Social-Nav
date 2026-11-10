"""Interactive semantic annotation of an existing static environment map."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from perception.environment.storage import StaticEnvironmentMap, asset_reference, content_hash
from perception.visualization.edit_geometry import validate_polygon
from .tools.places import Catalog


KINDS = {'点': ('places', 'place_id'), '区域': ('regions', 'region_id'), '地标': ('landmarks', 'landmark_id')}


class Annotation:
    def __init__(self, map_path, floor_id, catalog_path=None):
        self.map_path = Path(map_path).expanduser().resolve()
        self.asset = StaticEnvironmentMap(self.map_path)
        self.source_binding = {name:content_hash(self.map_path/name) for name in ('manifest.yaml','layers.npz')}
        m = self.asset.manifest
        metadata = self.asset.snapshot.metadata
        self.map_id = metadata.get('map_id')
        if not self.map_id:
            raise ValueError('静态地图缺少定位地图身份，不能建立语义对应关系')
        if catalog_path:
            self.value = json.loads(Path(catalog_path).read_text())
            if (self.value.get('map_snapshot_id') != m['snapshot_id'] or
                    self.value.get('frame_id') != m['frame_id'] or
                    self.value.get('localization_map_id',self.map_id) != self.map_id or
                    self.value.get('floor_id') != floor_id):
                raise ValueError('语义目录与静态地图或定位地图不一致')
            self.value['map_asset']=asset_reference(self.map_path)['path']
            self.value['localization_map_id']=self.map_id
        else:
            ref = asset_reference(self.map_path)
            self.value = dict(scope='development_prior',map_condition='complete',floor_id=floor_id,
                frame_id=m['frame_id'],map_asset=ref['path'],map_snapshot_id=m['snapshot_id'],
                localization_map_id=self.map_id,source='manual_semantic_annotation',
                access={},places=[],regions=[],landmarks=[])
        self.value.setdefault('regions',[]);self.value.setdefault('landmarks',[])
        self.value.setdefault('access',{})
        Catalog(self.value)
        self.undo = []

    def update(self, kind, identity, name, aliases, coordinates, description='', previous=None):
        group, id_field = KINDS[kind]
        identity, name = identity.strip(), name.strip()
        if not identity or not name or identity == 'mission_start':
            raise ValueError('请填写非空 ID 和名称；mission_start 由任务运行时生成')
        value = copy.deepcopy(self.value)
        if previous is not None:
            old_group, old_id = previous
            value[old_group] = [p for p in value[old_group] if p.get(KINDS_BY_GROUP[old_group]) != old_id]
        if any(p.get(field)==identity for items,field in KINDS.values() for p in value[items]):
            raise ValueError('ID 已被其他元素使用')
        coordinates = np.asarray(coordinates,dtype=float)
        if kind == '区域':
            polygon = validate_polygon(coordinates)
            if len(polygon)>1000:raise ValueError('区域顶点过多')
            if len({tuple(p) for p in polygon}) != len(polygon):raise ValueError('区域顶点不能重复')
            def cross(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
            for i in range(len(polygon)):
                a,b=polygon[i],polygon[(i+1)%len(polygon)]
                for j in range(i+1,len(polygon)):
                    if j==i+1 or (i==0 and j==len(polygon)-1):continue
                    c,d=polygon[j],polygon[(j+1)%len(polygon)]
                    if ((np.maximum(np.minimum(a,b),np.minimum(c,d)) <= np.minimum(np.maximum(a,b),np.maximum(c,d))).all() and
                            cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0):
                        raise ValueError('区域边不能相交')
            coordinates = polygon.tolist()
        elif coordinates.shape not in ([(2,), (3,)] if kind=='点' else [(2,)]):
            raise ValueError('点使用 [x,y] 或 [x,y,yaw]，地标使用 [x,y]')
        elif not np.isfinite(coordinates).all():
            raise ValueError('坐标必须为有限数值')
        else:
            coordinates = coordinates.tolist()
        origin=np.asarray(self.asset.snapshot.metadata['origin_xy']);res=self.asset.snapshot.metadata['resolution']
        height,width=self.asset.snapshot.layers['occupancy'].shape
        positions=np.asarray(coordinates if kind=='区域' else [coordinates[:2]])
        if ((positions<origin).any() or (positions>=origin+np.array([width,height])*res).any()):
            raise ValueError('标注坐标超出静态地图范围')
        item = {id_field:identity,'name':name,'aliases':[a.strip() for a in aliases if a.strip()],
                'source':'manual_map_annotation'}
        if description.strip():item['description'] = description.strip()
        if kind=='点':item.update(pose=coordinates,reference='base_link',status='manual_annotation',allow_shadow=True)
        elif kind=='区域':item['polygon'] = coordinates
        else:item['position'] = coordinates
        value[group].append(item)
        Catalog(value)
        self.undo.append(self.value);self.value=value

    def delete(self, group, identity):
        if identity=='mission_start':raise ValueError('任务起点由运行时生成')
        value=copy.deepcopy(self.value)
        value[group]=[p for p in value[group] if p.get(KINDS_BY_GROUP[group])!=identity]
        self.undo.append(self.value);self.value=value

    def save(self, path, expected=None):
        path=Path(path).expanduser().absolute()
        if self.source_binding != {name:content_hash(self.map_path/name) for name in self.source_binding}:
            raise ValueError('源静态地图已改变，请重新打开标注器')
        if self.map_path == path or self.map_path in path.resolve().parents:
            raise ValueError('语义目录不能写入源地图内部')
        value=copy.deepcopy(self.value)
        if not any(p['place_id']=='mission_start' for p in value['places']):
            value['places'].append(dict(place_id='mission_start',name='本次出发点',aliases=['起点'],
                pose=None,reference='base_link',status='bind_at_task_start',allow_shadow=False))
        Catalog(value)
        data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        path.parent.mkdir(parents=True,exist_ok=True)
        if expected is None:
            with path.open('xb') as stream:stream.write(data)
        else:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('输出文件已被外部修改，拒绝覆盖')
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
                temporary=Path(stream.name);stream.write(data)
            try:temporary.replace(path)
            finally:temporary.unlink(missing_ok=True)
        return hashlib.sha256(data).hexdigest()


KINDS_BY_GROUP = {group:field for group,field in KINDS.values()}


def make_figure(annotation, output, geometry=None):
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus']=False
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PyQt5 import QtCore,QtWidgets

    grid=annotation.asset.snapshot
    cells=grid.layers['occupancy'];origin=grid.metadata['origin_xy'];res=grid.metadata['resolution']
    height,width=cells.shape
    fig,ax=plt.subplots(figsize=(12,8));window=fig.canvas.manager.window
    window.setWindowTitle('Agent 地图语义标注')
    if fig.canvas.manager.toolbar:fig.canvas.manager.toolbar.hide()
    ax.imshow(np.where(cells==-1,0,np.where(cells==0,1,2)),origin='lower',interpolation='nearest',
        cmap=ListedColormap(['#9ca9b7','#fafafa','#242c36']),vmin=0,vmax=2,
        extent=[origin[0],origin[0]+width*res,origin[1],origin[1]+height*res])
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('地图 X（米）');ax.set_ylabel('地图 Y（米）')
    ax.set_title('左键放点并可拖动朝向；区域按 Enter 完成，滚轮缩放，右键平移')
    dock=QtWidgets.QDockWidget('标注',window);dock.setObjectName('semantic_annotation')
    panel=QtWidgets.QWidget();layout=QtWidgets.QVBoxLayout(panel)
    mode=QtWidgets.QComboBox();mode.addItems(list(KINDS));layout.addWidget(mode)
    listing=QtWidgets.QListWidget();layout.addWidget(listing)
    form=QtWidgets.QFormLayout();layout.addLayout(form)
    identity=QtWidgets.QLineEdit();name=QtWidgets.QLineEdit();aliases=QtWidgets.QLineEdit()
    coords=QtWidgets.QLineEdit();description=QtWidgets.QLineEdit()
    for key,widget in [('identity',identity),('name',name),('aliases',aliases),('coordinates',coords),('description',description)]:widget.setObjectName(key)
    for label,widget in [('ID',identity),('名称',name),('别名（逗号分隔）',aliases),('坐标 JSON',coords),('说明',description)]:form.addRow(label,widget)
    buttons=QtWidgets.QHBoxLayout();layout.addLayout(buttons)
    apply=QtWidgets.QPushButton('应用');delete=QtWidgets.QPushButton('删除');undo=QtWidgets.QPushButton('撤销')
    for button in (apply,delete,undo):buttons.addWidget(button)
    actions=QtWidgets.QHBoxLayout();layout.addLayout(actions)
    clear=QtWidgets.QPushButton('新建');save=QtWidgets.QPushButton('保存');actions.addWidget(clear);actions.addWidget(save)
    status=QtWidgets.QLabel('坐标可直接编辑；导航点可填写 [x,y,yaw]，区域用多边形。');status.setWordWrap(True);layout.addWidget(status)
    dock.setWidget(panel);window.addDockWidget(QtCore.Qt.RightDockWidgetArea,dock)
    selected=[None];vertices=[];artists=[];pan=[None];point_drag=[None];saved=[None]

    def redraw():
        for artist in artists:artist.remove()
        artists.clear()
        for kind,(group,field) in KINDS.items():
            for item in annotation.value[group]:
                point=item.get('pose') if kind=='点' else item.get('position') if kind=='地标' else item.get('polygon')
                if point is None:continue
                if kind=='区域':
                    p=np.asarray(point);line,=ax.plot(*np.vstack([p,p[0]]).T,color='#168a55',lw=2);artists.append(line);xy=p.mean(axis=0)
                else:
                    xy=point[:2];artists.append(ax.scatter(*xy,c='#d94b39' if kind=='点' else '#7853b8',s=42,zorder=5))
                    if kind=='点' and len(point)==3:
                        artists.append(ax.arrow(xy[0],xy[1],.7*np.cos(point[2]),.7*np.sin(point[2]),width=.03,color='#d94b39',zorder=5))
                        if geometry is not None and selected[0]==(group,item[field]):
                            polygon=geometry.polygon(point);line,=ax.plot(*np.vstack([polygon,polygon[0]]).T,color='#eb8b00',lw=2);artists.append(line)
                artists.append(ax.text(*xy,item['name'],fontsize=9,color='#111111',bbox=dict(facecolor='white',alpha=.8,edgecolor='none')))
        if vertices:
            line,=ax.plot(*np.asarray(vertices).T,'o-',color='#168a55');artists.append(line)
        listing.blockSignals(True);listing.clear()
        for kind,(group,field) in KINDS.items():
            for item in annotation.value[group]:
                if item[field]=='mission_start':continue
                row=QtWidgets.QListWidgetItem(kind+' · '+item['name']+' · '+item[field]);row.setData(QtCore.Qt.UserRole,(kind,item[field]));listing.addItem(row)
                if selected[0]==(group,item[field]):listing.setCurrentItem(row)
        listing.blockSignals(False);fig.canvas.draw_idle()

    def new():
        selected[0]=None;vertices.clear()
        for field in (identity,name,aliases,coords,description):field.clear()
        listing.clearSelection();redraw()

    def choose(row):
        if row is None:return
        kind,ident=row.data(QtCore.Qt.UserRole);group,field=KINDS[kind]
        item=next(p for p in annotation.value[group] if p[field]==ident)
        selected[0]=(group,ident);mode.setCurrentText(kind);identity.setText(ident);name.setText(item['name'])
        aliases.setText(','.join(item.get('aliases',[])));description.setText(item.get('description',''))
        coords.setText(json.dumps(item.get({'点':'pose','区域':'polygon','地标':'position'}[kind]),ensure_ascii=False));vertices.clear();redraw()

    def apply_edit():
        try:
            annotation.update(mode.currentText(),identity.text(),name.text(),aliases.text().replace('，',',').split(','),
                              json.loads(coords.text()),description.text(),selected[0])
            status.setText('已应用；点击保存写入语义目录。');new()
        except (ValueError,TypeError,KeyError) as error:status.setText(str(error))

    def delete_edit():
        if selected[0] is not None:
            annotation.delete(*selected[0]);new();status.setText('已删除，尚未保存。')

    def undo_edit():
        if vertices:vertices.pop()
        elif annotation.undo:annotation.value=annotation.undo.pop();selected[0]=None
        redraw()

    def save_edit():
        if vertices:status.setText('请先按 Enter 完成或取消多边形。');return
        try:
            saved[0]=annotation.save(output,saved[0]);status.setText('已保存：'+str(output))
        except (OSError,ValueError,KeyError) as error:status.setText('保存失败：'+str(error))

    def click(event):
        if event.inaxes is not ax or event.xdata is None:return
        if event.button in (2,3):pan[0]=(event.button,event.x,event.y,ax.get_xlim(),ax.get_ylim());return
        if event.button!=1:return
        point=[float(event.xdata),float(event.ydata)]
        if mode.currentText()=='区域':vertices.append(point);redraw()
        else:
            coords.setText(json.dumps(point));point_drag[0]=point if mode.currentText()=='点' else None
            status.setText('位置已选；点可拖动设置朝向，填写属性后点击应用。')

    def motion(event):
        if pan[0] is None:
            if point_drag[0] is not None and event.inaxes is ax and event.xdata is not None:
                x,y=point_drag[0];dx,dy=event.xdata-x,event.ydata-y
                if np.hypot(dx,dy)>.1:coords.setText(json.dumps([x,y,float(np.arctan2(dy,dx))]))
            return
        _,x,y,xlim,ylim=pan[0];dx=(event.x-x)*(xlim[1]-xlim[0])/ax.bbox.width;dy=(event.y-y)*(ylim[1]-ylim[0])/ax.bbox.height
        ax.set_xlim(xlim[0]-dx,xlim[1]-dx);ax.set_ylim(ylim[0]-dy,ylim[1]-dy);fig.canvas.draw_idle()

    def release(event):
        if event.button==1:point_drag[0]=None
        if pan[0] and event.button==pan[0][0]:pan[0]=None

    def scroll(event):
        if event.inaxes is not ax or event.xdata is None or not event.step:return
        scale=1.2**(-event.step);xlim=ax.get_xlim();ylim=ax.get_ylim()
        ax.set_xlim(event.xdata+(xlim[0]-event.xdata)*scale,event.xdata+(xlim[1]-event.xdata)*scale)
        ax.set_ylim(event.ydata+(ylim[0]-event.ydata)*scale,event.ydata+(ylim[1]-event.ydata)*scale);fig.canvas.draw_idle()

    def key(event):
        if event.key=='enter' and vertices:
            try:coords.setText(json.dumps(validate_polygon(vertices).tolist()));vertices.clear();redraw();status.setText('区域已绘制；填写属性后点击应用。')
            except ValueError as error:status.setText(str(error))
        elif event.key=='escape':vertices.clear();redraw()
        elif event.key=='backspace' and vertices:vertices.pop();redraw()

    listing.currentItemChanged.connect(lambda current,previous:choose(current))
    apply.clicked.connect(apply_edit);delete.clicked.connect(delete_edit);undo.clicked.connect(undo_edit)
    clear.clicked.connect(new);save.clicked.connect(save_edit)
    for event,handler in [('button_press_event',click),('motion_notify_event',motion),('button_release_event',release),('scroll_event',scroll),('key_press_event',key)]:fig.canvas.mpl_connect(event,handler)
    redraw();return fig


def main():
    p=argparse.ArgumentParser(description='对项目静态环境地图标注 Agent 点、区域与地标')
    p.add_argument('--static-map',required=True);p.add_argument('--floor-id',required=True)
    p.add_argument('--catalog',help='只读导入已有语义目录，必须与地图匹配')
    p.add_argument('--vehicle-profile',help='可选轮椅档案，用于显示已指定朝向的车体轮廓')
    p.add_argument('--out',required=True,help='新语义目录 JSON 文件')
    args=p.parse_args()
    if Path(args.out).exists():raise FileExistsError('输出已存在，请使用新路径')
    geometry=None
    if args.vehicle_profile:
        import yaml
        from navigation.planning.footprint import VehicleGeometry
        profile=yaml.safe_load(Path(args.vehicle_profile).read_text())
        geometry=VehicleGeometry(**profile['geometry'],source={'path':str(Path(args.vehicle_profile).resolve())},assumed=False)
    make_figure(Annotation(args.static_map,args.floor_id,args.catalog),args.out,geometry)
    import matplotlib.pyplot as plt
    plt.show()


if __name__=='__main__':main()
