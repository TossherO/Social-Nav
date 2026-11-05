"""Light Qt graphics; independent of the detector annotation UI."""
import io
import math
from bisect import bisect_left

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from decision.vlm.scene_inputs import RGB_RECT,BEV_RECT,MODEL_SIZE,rgb_geometry,bev_xy,PERSON_RADIUS

STATUS_COLORS = {'draft': '#d6a52c', 'skipped': '#87939e', 'complete': '#2b9c68'}
BEV_COLORS = dict(background='#111923', grid='#273648', major_grid='#3a5068',
                  target='#55dbea', reference='#ffad66', selected='#ff68bf',
                  observed='#ffd071', predicted='#b5a1ff', ego='#77e2ba',
                  camera='#79aaff', text='#e3edf7', label_background='#19283b', leader='#8196b0')
BEV_HEIGHT_RANGE = (-1.5, 2.5)
BEV_HEIGHT_PALETTE = ((83,129,191), (61,151,155), (123,165,101), (197,157,91), (197,117,115))


def place_context_labels(markers, sizes, bounds):
    """The same collision scoring as model labels, vectorized for interactive scans."""
    if not markers:
        return []
    obstacles = np.asarray([(x-8,y-8,x+8,y+8) for _,x,y in markers],dtype=float)
    occupied,result = [],[]
    for key,x,y in markers:
        width,height = sizes[key]
        offsets = np.asarray([(dx,dy) for gap in (9,20,34,52,76,104) for dx,dy in
            ((gap,-height/2),(-gap-width,-height/2),(-width/2,-gap-height),(-width/2,gap),
             (gap,-gap-height),(-gap-width,-gap-height),(gap,gap),(-gap-width,gap))])
        left = np.maximum(bounds[0],np.minimum(x+offsets[:,0],bounds[2]-width))
        top = np.maximum(bounds[1],np.minimum(y+offsets[:,1],bounds[3]-height))
        rects = np.column_stack((left,top,left+width,top+height))
        others = np.concatenate((np.asarray(occupied).reshape(-1,4),obstacles))
        area = np.maximum(0,np.minimum(rects[:,None,2:],others[None,:,2:])-
                          np.maximum(rects[:,None,:2],others[None,:,:2]))
        collisions = (area[:,:,0]*area[:,:,1]).sum(axis=1)
        distance = np.hypot(x-np.maximum(left,np.minimum(x,left+width)),
                            y-np.maximum(top,np.minimum(y,top+height)))
        choice = int(np.argmin(collisions*1000+distance+np.arange(len(rects))*.001))
        rect = tuple(rects[choice])
        occupied.append((rect[0]-2,rect[1]-2,rect[2]+2,rect[3]+2))
        result.append((key,rect))
    return result


def bev_corners(box):
    x, y, _, length, width, _, yaw = box[:7]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*dx-s*dy, y+s*dx+c*dy) for dx,dy in
            ((-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2))]


def scan_image(raw, bounds, factor, point_size):
    """Thread-safe display raster; QPixmap conversion stays on the GUI thread."""
    width,height = math.ceil(bounds.width()),math.ceil(bounds.height())
    pixels=np.full((height,width,3),QtGui.QColor(BEV_COLORS['background']).getRgb()[:3],dtype=np.uint8)
    if raw:
        cloud=np.load(io.BytesIO(raw),allow_pickle=False)
        if cloud.ndim!=2 or cloud.shape[1]!=4 or len(cloud)>100000 or cloud.dtype.kind!='f':
            raise ValueError('点云浏览资产布局错误')
        cloud=cloud[np.isfinite(cloud).all(axis=1)]
        xx=np.rint(-cloud[:,1]*factor-bounds.left()).astype(int)
        yy=np.rint(-cloud[:,0]*factor-bounds.top()).astype(int)
        valid=(xx>=3)&(xx<width-3)&(yy>=3)&(yy<height-3)
        xx,yy,zz=xx[valid],yy[valid],cloud[valid,2]
        heights=np.full((height,width),-np.inf,dtype=np.float32)
        diameter=max(1,point_size-1);start=-(diameter//2)
        for dx in range(start,start+diameter):
            for dy in range(start,start+diameter):
                np.maximum.at(heights,(yy+dy,xx+dx),zz)
        occupied=np.isfinite(heights)
        low,high=BEV_HEIGHT_RANGE
        ramp=np.clip((heights[occupied]-low)/(high-low),0,1)*(len(BEV_HEIGHT_PALETTE)-1)
        index=np.minimum(ramp.astype(int),len(BEV_HEIGHT_PALETTE)-2)
        weight=(ramp-index)[:,None];palette=np.asarray(BEV_HEIGHT_PALETTE)
        pixels[occupied]=np.rint(palette[index]*(1-weight)+palette[index+1]*weight).astype(np.uint8)
    return QtGui.QImage(pixels.data,width,height,width*3,QtGui.QImage.Format_RGB888).copy()


def prepare_view(view, point_size):
    images=[QtGui.QImage.fromData(raw) for raw in view['histories']]
    context=bev_context(view['context_frames'], view['window']['mapping'])
    scans=[]
    if context:
        xmin,xmax,ymin,ymax=context_extent(context)
        factor=min(20.,1800/max(xmax-xmin,ymax-ymin))
        bounds=QtCore.QRectF(-ymax*factor,-xmax*factor,math.ceil((ymax-ymin)*factor),math.ceil((xmax-xmin)*factor))
        key=(bounds.x(),bounds.y(),bounds.width(),bounds.height(),factor,point_size)
        scans=[(key,scan_image(row['cloud'],bounds,factor,point_size)) if row else None for row in context]
    return dict(images=images,scans=scans)


class PersonRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, key, rect, signal, polygon=None):
        super().__init__(QtCore.QRectF(*rect))
        self.key, self.signal = key, signal
        self.polygon = polygon
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.setZValue(2)

    def mousePressEvent(self, event):
        self.signal.emit(self.key, bool(event.modifiers() & QtCore.Qt.ControlModifier))
        event.accept()

    def shape(self):
        if self.polygon is None:
            return super().shape()
        path = QtGui.QPainterPath()
        path.addPolygon(self.polygon)
        path.closeSubpath()
        return path

    def paint(self, painter, option, widget=None):
        if self.polygon is None:
            return super().paint(painter, option, widget)
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPolygon(self.polygon)


class PersonLabel(QtWidgets.QGraphicsTextItem):
    def __init__(self, key, text, signal):
        super().__init__(text)
        self.key, self.signal = key, signal
        self.document().setDocumentMargin(4)
        self.border = QtGui.QColor('#506883')
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

    def paint(self, painter, option, widget=None):
        painter.setPen(QtGui.QPen(self.border))
        painter.setBrush(QtGui.QColor(BEV_COLORS['label_background']))
        painter.drawRoundedRect(self.boundingRect(), 4, 4)
        super().paint(painter, option, widget)

    def mousePressEvent(self, event):
        self.signal.emit(self.key, bool(event.modifiers() & QtCore.Qt.ControlModifier))
        event.accept()


class PersonView(QtWidgets.QGraphicsView):
    picked = QtCore.pyqtSignal(str, bool)
    box_drawn = QtCore.pyqtSignal(object)

    def __init__(self, bev=False):
        super().__init__()
        self.bev = bev
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setBackgroundBrush(QtGui.QColor(BEV_COLORS['background'] if bev else '#edf3f8'))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(self.ScrollHandDrag)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setMinimumSize(230, 180)
        self.items_by_key, self.texts = {}, {}
        self.label_leaders = []
        self.initial = True
        self.auto_fit = True
        self.draw_box, self.draw_start, self.rubber = False, None, None
        self.cloud, self.people, self.labels = None, [], {}
        self.point_size = 2
        self.factor = 20.
        self.image_size = None
        self.group_colors, self.selected = {}, set()
        self.inspector = None
        self.bev_title = None
        self.layout_hold = 0

    def wheelEvent(self, event):
        self.auto_fit = False
        self.scale(1.18 if event.angleDelta().y() > 0 else 1/1.18, 1.18 if event.angleDelta().y() > 0 else 1/1.18)
        self.layout_bev_labels()
        event.accept()

    def fit(self):
        self.auto_fit = True
        self.layout_hold += 1
        try:
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        finally:
            self.layout_hold -= 1
        self.layout_bev_labels()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.initial or self.auto_fit:
            self.fit()
        else:
            self.layout_bev_labels()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.layout_bev_labels()

    def layout_bev_labels(self):
        if self.layout_hold or not self.bev or not getattr(self, 'texts', None):
            return
        for item in self.label_leaders:
            self.scene().removeItem(item)
        self.label_leaders = []
        bounds = self.viewport().rect().adjusted(3, 3, -3, -3)
        visible_bounds = QtCore.QRect(bounds)
        if self.bev_title is not None:
            title = self.bev_title.deviceTransform(self.viewportTransform()).mapRect(self.bev_title.boundingRect())
            if title.intersects(QtCore.QRectF(bounds)):
                bounds.setTop(min(bounds.bottom()-30, max(bounds.top(), math.ceil(title.bottom())+3)))
        markers, sizes = [], {}
        for key, item in self.items_by_key.items():
            point = self.mapFromScene(item.rect().center())
            visible = visible_bounds.contains(point)
            self.texts[key].setVisible(visible)
            if visible:
                markers.append((key, point.x(), point.y()))
                rect = self.texts[key].boundingRect()
                sizes[key] = (rect.width(), rect.height())
        anchors = {key: (x, y) for key, x, y in markers}
        placements = place_context_labels(sorted(markers), sizes, (bounds.left(), bounds.top(), bounds.right(), bounds.bottom()))
        for key, (left, top, right, bottom) in placements:
            self.texts[key].setPos(self.mapToScene(QtCore.QPoint(round(left), round(top))))
            x, y = anchors[key]
            end = self.mapToScene(QtCore.QPoint(round(max(left, min(x, right))), round(max(top, min(y, bottom)))))
            pen = QtGui.QPen(QtGui.QColor(BEV_COLORS['leader']), 1)
            pen.setCosmetic(True)
            line = self.scene().addLine(QtCore.QLineF(self.items_by_key[key].rect().center(), end), pen)
            line.setZValue(1.5)
            self.label_leaders.append(line)

    def show_frame(self, image, cloud, people, labels, trajectory=None, extent=None, camera_fov=None):
        # fitInView emits several scroll events; lay out labels once per complete scan.
        self.layout_hold += 1
        try:
            self._show_frame(image,cloud,people,labels,trajectory,extent,camera_fov)
        finally:
            self.layout_hold -= 1
        self.layout_bev_labels()

    def draw_camera_fov(self, camera_fov):
        if camera_fov is None:
            return False
        origin = np.asarray(camera_fov['origin_xy_m'])
        rays = np.asarray(camera_fov['boundary_directions_xy'])
        forward = np.asarray(camera_fov['forward_xy'])
        bounds = self.sceneRect()
        reach = (math.hypot(bounds.width(), bounds.height())/self.factor + np.linalg.norm(origin)+1)
        reach /= min(rays @ forward)
        def point(xy):
            return QtCore.QPointF(-xy[1]*self.factor, -xy[0]*self.factor)
        a = point(origin)
        ends = [point(origin+ray*reach) for ray in rays]
        clip = self.scene().addRect(bounds, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtCore.Qt.NoBrush))
        clip.setFlag(QtWidgets.QGraphicsItem.ItemClipsChildrenToShape)
        clip.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        clip.setZValue(-.5)
        fill = QtGui.QColor(BEV_COLORS['camera'])
        fill.setAlpha(18)
        area = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([a, *ends]), clip)
        area.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        area.setBrush(QtGui.QBrush(fill))
        area.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        for end in ends:
            line = QtWidgets.QGraphicsLineItem(QtCore.QLineF(a, end), clip)
            pen = QtGui.QPen(QtGui.QColor(BEV_COLORS['camera']), 1.5, QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        return True

    def _show_frame(self, image, cloud, people, labels, trajectory=None, extent=None, camera_fov=None):
        if self.bev:
            people = [p for p in people if p.get('box') is not None and
                      (p.get('bev_current_target', p.get('eligible')) or p.get('bev_reference'))]
        self.items_by_key, self.texts, self.label_leaders = {}, {}, []
        self.bev_title = None
        self.scene().clear()
        self.draw_start, self.rubber, self.draw_box = None, None, False
        self.items_by_key, self.texts = {}, {}
        self.people, self.labels, self.cloud = people, labels, cloud
        if self.bev:
            positions = [p['box'][:2] for p in people if p.get('box') is not None]
            if trajectory and trajectory['status']=='available':
                positions += [p[1:] for segment in trajectory['ego'] for p in segment]
                positions += [p[1:] for segments in trajectory['people'].values() for segment in segments for p in segment['points']]
            xs, ys = [-5., 25.]+[p[0] for p in positions], [-15., 15.]+[p[1] for p in positions]
            xmin, xmax = math.floor(min(xs))-1, math.ceil(max(xs))+1
            ymin, ymax = math.floor(min(ys))-1, math.ceil(max(ys))+1
            if extent is not None:
                xmin, xmax, ymin, ymax = extent
            self.factor = min(20., 1800/max(xmax-xmin, ymax-ymin))
            self.setSceneRect(-ymax*self.factor, -xmax*self.factor,
                              math.ceil((ymax-ymin)*self.factor), math.ceil((xmax-xmin)*self.factor))
            pix = self.cloud_pixmap(cloud)
            background = self.scene().addPixmap(pix)
            background.setZValue(-2)
            bounds = self.sceneRect()
            background.setPos(bounds.left(), bounds.top())
            step = max(1, math.ceil(max(xmax-xmin, ymax-ymin)/80))
            def grid_pen(value):
                color = QtGui.QColor(BEV_COLORS['major_grid'] if value%5==0 else BEV_COLORS['grid'])
                color.setAlpha(145)
                pen = QtGui.QPen(color, 1)
                pen.setCosmetic(True)
                return pen
            for y in range(ymin, ymax+1, step):
                self.scene().addLine(-y*self.factor, bounds.top(), -y*self.factor, bounds.bottom(), grid_pen(y)).setZValue(-1)
            for x in range(xmin, xmax+1, step):
                self.scene().addLine(bounds.left(), -x*self.factor, bounds.right(), -x*self.factor, grid_pen(x)).setZValue(-1)
            has_fov = self.draw_camera_fov(camera_fov)
            ego = QtGui.QColor(BEV_COLORS['ego'])
            self.scene().addEllipse(-4, -4, 8, 8, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(ego)).setZValue(1)
            self.scene().addLine(0, 0, 0, -20, QtGui.QPen(ego, 2)).setZValue(1)
            arrow = QtGui.QPolygonF([QtCore.QPointF(0,-23),QtCore.QPointF(-4,-15),QtCore.QPointF(4,-15)])
            self.scene().addPolygon(arrow, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(ego)).setZValue(1)
            title = self.bev_title = self.scene().addText('')
            legend = [('target','■ 当前目标'),('observed','━ 观测'),
                      ('predicted','┄ 预测'),('ego','● 自车'),('selected','■ 选中')]
            if any(p.get('bev_reference') for p in people):
                legend.insert(1, ('reference', '┄ 历史参考'))
            if has_fov:
                legend.append(('camera', '⋁ 相机水平视野（示意）'))
            heights = np.linspace(*BEV_HEIGHT_RANGE, len(BEV_HEIGHT_PALETTE))
            height_legend = 'Z[m]: '+''.join(
                f'<span style="color:{QtGui.QColor(*rgb).name()}">■ {"≤" if i==0 else "≥" if i==len(heights)-1 else ""}{z:g}</span>&nbsp;'
                for i,(z,rgb) in enumerate(zip(heights,BEV_HEIGHT_PALETTE)))
            title.setHtml(f'点云BEV · 前↑左← · {step}m'+(' · 缺扫描' if cloud is None else '')+
                          ('' if has_fov else ' · 相机视野不可用')+'<br>'+''.join(
                f'<span style="color:{BEV_COLORS[key]}">{text}&nbsp;</span>' for key,text in legend)+'<br>'+height_legend)
            title.setToolTip('颜色表示雷达坐标系Z高度，非离地高度。蓝色区域为相机主点行左右边缘射线的XY投影，'
                             '不表示垂直视野、遮挡或可见距离，也不筛选目标。历史框使用当时记录的位置，'
                             '蓝色实框属于当前有效目标，橙色虚框为历史参考；观测／预测由轨迹线区分。')
            title.setPos(bounds.left()+5, bounds.top()+3)
            title.setDefaultTextColor(QtGui.QColor(BEV_COLORS['text']))
            title.setFont(QtGui.QFont(self.font().family(), 8))
            title.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            title.setZValue(4)
            if trajectory and trajectory['status']=='available':
                def line(points,color,predicted=False):
                    path=QtGui.QPainterPath()
                    for i,point in enumerate(points):
                        x,y=-point[2]*self.factor,-point[1]*self.factor
                        path.moveTo(x,y) if i==0 else path.lineTo(x,y)
                    pen=QtGui.QPen(QtGui.QColor(color),2)
                    pen.setCosmetic(True)
                    if predicted:pen.setStyle(QtCore.Qt.DashLine)
                    self.scene().addPath(path,pen)
                for segment in trajectory['ego']:line(segment,BEV_COLORS['ego'])
                for segments in trajectory['people'].values():
                    for segment in segments:line(segment['points'],BEV_COLORS['observed'] if segment['observed'] else BEV_COLORS['predicted'],not segment['observed'])
        else:
            pix = QtGui.QPixmap()
            if image:
                pix.loadFromData(image)
            if pix.isNull():
                self.scene().addText('无因果配对RGB')
                self.setSceneRect(0, 0, 640, 480)
            else:
                item = self.scene().addPixmap(pix)
                width, height = self.image_size or [pix.width(), pix.height()]
                item.setScale(width/pix.width())
                self.setSceneRect(0, 0, width, height)
        for index, person in enumerate(people):
            key = person['key']
            polygon = None
            if self.bev:
                if person.get('box') is None:
                    continue
                corners = bev_corners(person['box'])
                polygon = QtGui.QPolygonF([QtCore.QPointF(-cy*self.factor,-cx*self.factor) for cx,cy in corners])
                bounds = polygon.boundingRect()
                rect = [bounds.x(), bounds.y(), bounds.width(), bounds.height()]
            else:
                b = person.get('rgb_box')
                if not b:
                    continue
                rect = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
            item = PersonRect(key, rect, self.picked, polygon)
            normal = person.get('bev_current_target', person.get('eligible'))
            color = BEV_COLORS['target' if normal else 'reference'] if self.bev else ('#008aac' if normal else '#8d7f66')
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.scene().addItem(item)
            self.items_by_key[key] = item
            if self.bev:
                name = labels.get(key, str(index+1))
                short = name.split(' / ')[0]
                if short.startswith('参考'):
                    short = 'R'+short[2:]
                text = PersonLabel(key, short, self.picked)
                text.setFont(QtGui.QFont(self.font().family(), 9))
                text.setToolTip(name)
                self.scene().addItem(text)
            else:
                text = self.scene().addText(labels.get(key, str(index+1)))
            text.setDefaultTextColor(QtGui.QColor(BEV_COLORS['text'] if self.bev else '#004d68'))
            text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(rect[0], rect[1]-20)
            text.setZValue(3)
            self.texts[key] = text
        if self.initial or self.auto_fit:
            self.fit()
            self.initial = False
        self.highlight(self.selected)
        self.layout_bev_labels()

    def cloud_pixmap(self, raw):
        bounds = self.sceneRect()
        key = (bounds.x(), bounds.y(), bounds.width(), bounds.height(), self.factor, self.point_size)
        cached = getattr(self, '_cloud_cache', None)
        if cached and cached[0] is raw and cached[1] == key:
            return cached[2]
        prepared=getattr(self,'prepared_cloud',None)
        image=prepared[2] if prepared and prepared[0] is raw and prepared[1]==key else scan_image(raw,bounds,self.factor,self.point_size)
        pixmap = QtGui.QPixmap.fromImage(image)
        self._cloud_cache = (raw, key, pixmap)
        return pixmap

    def highlight(self, selected):
        self.selected = set(selected)
        eligible = {p['key'] for p in self.people if
                    (p.get('bev_current_target', p.get('eligible')) if self.bev else p.get('eligible'))}
        for key, item in self.items_by_key.items():
            normal = BEV_COLORS['target' if key in eligible else 'reference'] if self.bev else ('#008aac' if key in eligible else '#8d7f66')
            normal = self.group_colors.get(key, normal)
            color = QtGui.QColor(BEV_COLORS['selected'] if self.bev and key in selected else '#f06418' if key in selected else normal)
            if self.bev and key not in selected and key in self.group_colors:
                h,s,l,a = color.getHslF()
                color.setHslF(h,s,max(l,.67),a)
            pen = QtGui.QPen(color, 3 if key in selected else 2 if self.bev else 1.5)
            pen.setCosmetic(True)
            if self.bev and key not in eligible:
                pen.setStyle(QtCore.Qt.DashLine)
            item.setPen(pen)
            fill = QtGui.QColor(color) if self.bev else QtGui.QColor(255, 157, 65)
            fill.setAlpha(65 if self.bev else 35)
            item.setBrush(QtGui.QBrush(fill) if key in selected else QtGui.QBrush(QtCore.Qt.NoBrush))
            if self.bev and key in self.texts:
                text = self.texts[key]
                text.border = color
                text.setDefaultTextColor(color if key not in eligible or key in selected else QtGui.QColor(BEV_COLORS['text']))
                text.update()

    def mousePressEvent(self, event):
        if self.draw_box and event.button() == QtCore.Qt.LeftButton:
            self.draw_start = self.mapToScene(event.pos())
            self.rubber = self.scene().addRect(QtCore.QRectF(self.draw_start, self.draw_start), QtGui.QPen(QtGui.QColor('#e66b20'), 2))
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.inspector is not None:
            self.inspector()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() and not self.draw_box:
            self.auto_fit = False
        if self.draw_start is not None:
            self.rubber.setRect(QtCore.QRectF(self.draw_start, self.mapToScene(event.pos())).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.draw_start is not None:
            rect = self.rubber.rect().intersected(self.sceneRect())
            self.scene().removeItem(self.rubber)
            self.rubber, self.draw_start, self.draw_box = None, None, False
            if rect.width() >= 5 and rect.height() >= 5:
                self.box_drawn.emit([rect.left(), rect.top(), rect.right(), rect.bottom()])
            return
        super().mouseReleaseEvent(event)


def bev_context(context, mapping):
    """Show current identities and confirmed historical references without changing evidence."""
    def historical_reference(person):
        return (person['key'] not in mapping and person.get('kind') == 'track' and
                person.get('confirmed') and person.get('box') is not None)
    reference_keys = sorted({p['key'] for row in context[:-1] if row is not None
                             for p in row['frame']['people'] if historical_reference(p)})
    references = {key: i+1 for i, key in enumerate(reference_keys)}
    result = []
    for index, row in enumerate(context):
        if row is None:
            result.append(None)
            continue
        # The style flag never changes source eligibility, confirmation or observation state.
        people = [dict(p, bev_current_target=True) for p in row['frame']['people']
                  if p['key'] in mapping and p.get('box') is not None]
        if index < len(context)-1:
            people.extend(dict(p, bev_current_target=False, bev_reference=references[p['key']])
                          for p in row['frame']['people'] if historical_reference(p))
        present = {p['key'] for p in people}
        trace = row['trajectory']
        trace = dict(trace, people={k:v for k,v in trace['people'].items() if k in present})
        result.append(dict(row, frame=dict(row['frame'], people=people), trajectory=trace))
    return result


def context_extent(context):
    positions = []
    for row in context:
        if row is None:
            continue
        positions.extend(corner for p in row['frame']['people'] if p.get('box') is not None for corner in bev_corners(p['box']))
        trace = row['trajectory']
        positions.extend(p[1:3] for segment in trace['ego'] for p in segment)
        positions.extend(p[1:3] for segments in trace['people'].values() for segment in segments for p in segment['points'])
    xs, ys = [-5.,16.]+[p[0] for p in positions], [-10.,10.]+[p[1] for p in positions]
    return (math.floor(min(xs))-2, math.ceil(max(xs))+2, math.floor(min(ys))-2, math.ceil(max(ys))+2)


def observation_regions(frame, window):
    """Current-frame hit areas use the same layout and scale as model rendering."""
    if not frame.get('has_rgb'):
        return []
    scale,ox,oy,_,_ = rgb_geometry(frame['image_size'])
    xmin, xmax, ymin, ymax = window['fields']['e']
    regions = []
    for person in frame['people']:
        if not person.get('eligible') or person['key'] not in window['mapping']:
            continue
        box = person.get('rgb_box')
        if box:
            rect = QtCore.QRectF(ox+box[0]*scale, oy+box[1]*scale, (box[2]-box[0])*scale, (box[3]-box[1])*scale)
            rect = rect.intersected(QtCore.QRectF(RGB_RECT[0],RGB_RECT[1],RGB_RECT[2]-RGB_RECT[0],RGB_RECT[3]-RGB_RECT[1]))
            if not rect.isEmpty():
                regions.append(dict(key=person['key'], kind='rgb', rect=rect))
        x, y = person['box'][:2]
        u,v = bev_xy(x,y,window['fields']['e'])
        if xmin<=x<=xmax and ymin<=y<=ymax:
            regions.append(dict(key=person['key'], kind='bev', rect=QtCore.QRectF(u-PERSON_RADIUS,v-PERSON_RADIUS,2*PERSON_RADIUS,2*PERSON_RADIUS)))
    return regions


class ImageThumb(QtWidgets.QLabel):
    picked = QtCore.pyqtSignal(str, bool)

    def __init__(self, region=None):
        super().__init__('无历史')
        self.setMinimumSize(120, 85)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet('background:#e8eef4;border:1px solid #cbd5df')
        self.data = None
        self.region = region
        self.source_size = QtCore.QSize()
        self.original_pixmap = QtGui.QPixmap()
        self.original_loader = None
        self.regions, self.selected, self.group_colors = [], set(), {}
        self.show_overlays = False
        self.setMouseTracking(True)

    def show_data(self, data, pixmap=None):
        changed = data is not self.data
        self.data = data
        if pixmap is not None:
            self.original_pixmap = pixmap
        elif changed:
            self.original_pixmap = QtGui.QPixmap()
            if data:
                self.original_pixmap.loadFromData(data)
        self.draw_pixmap()

    def draw_pixmap(self):
        pix = self.original_pixmap
        if pix.isNull():
            self.clear()
            self.setText('历史缺失')
        else:
            self.source_size = pix.size()
            if self.region is not None:
                pix = pix.copy(QtCore.QRect(*self.region))
            self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def set_interaction(self, regions=(), selected=(), colors=None, overlays=False):
        self.regions, self.selected = list(regions), set(selected)
        self.group_colors, self.show_overlays = colors or {}, overlays
        self.setFocusPolicy(QtCore.Qt.ClickFocus if self.regions else QtCore.Qt.NoFocus)
        self.update()

    def image_rects(self):
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        region = QtCore.QRectF(*self.region) if self.region else QtCore.QRectF(0, 0, self.source_size.width(), self.source_size.height())
        width, height = pix.width()/pix.devicePixelRatioF(), pix.height()/pix.devicePixelRatioF()
        center = QtCore.QRectF(self.contentsRect()).center()
        return region, QtCore.QRectF(center.x()-width/2, center.y()-height/2, width, height)

    def image_to_widget(self, point):
        source, target = self.image_rects()
        return QtCore.QPointF(target.left()+(point.x()-source.left())*target.width()/source.width(),
                             target.top()+(point.y()-source.top())*target.height()/source.height())

    def target_at(self, point):
        rects = self.image_rects()
        if rects is None or not rects[1].contains(point):
            return None
        source, _ = rects
        hits = []
        for region in self.regions:
            rect = region['rect']
            if not source.intersects(rect):
                continue
            a, b = self.image_to_widget(rect.topLeft()), self.image_to_widget(rect.bottomRight())
            screen = QtCore.QRectF(a, b)
            if region['kind'] == 'rgb':
                if screen.contains(point):
                    hits.append((screen.width()*screen.height(), region['key']))
            else:
                distance = QtCore.QLineF(screen.center(), point).length()
                if distance <= max(9, screen.width()/2+3):
                    hits.append((distance, region['key']))
        return min(hits)[1] if hits else None

    def mousePressEvent(self, event):
        key = self.target_at(QtCore.QPointF(event.pos()))
        if key is not None and event.button() == QtCore.Qt.LeftButton:
            self.setFocus(QtCore.Qt.MouseFocusReason)
            self.picked.emit(key, bool(event.modifiers() & QtCore.Qt.ControlModifier))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.setCursor(QtCore.Qt.PointingHandCursor if self.target_at(QtCore.QPointF(event.pos())) is not None else QtCore.Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_overlays or self.image_rects() is None:
            return
        source, target = self.image_rects()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setClipRect(target.intersected(QtCore.QRectF(self.contentsRect())))
        for region in self.regions:
            key, rect = region['key'], region['rect']
            color = self.group_colors.get(key)
            if not source.intersects(rect) or (color is None and key not in self.selected):
                continue
            screen = QtCore.QRectF(self.image_to_widget(rect.topLeft()), self.image_to_widget(rect.bottomRight()))
            if region['kind'] == 'bev':
                radius = max(6, screen.width()/2+2)
                screen = QtCore.QRectF(screen.center()-QtCore.QPointF(radius, radius), QtCore.QSizeF(radius*2, radius*2))
            def outline(pen, bounds):
                painter.setPen(pen)
                painter.drawEllipse(bounds) if region['kind']=='bev' else painter.drawRect(bounds)
            if color:
                outline(QtGui.QPen(QtGui.QColor(color), 2), screen)
            if key in self.selected:
                bounds = screen if region['kind']=='bev' else screen.adjusted(-3, -3, 3, 3)
                outline(QtGui.QPen(QtGui.QColor('white'), 4), bounds)
                outline(QtGui.QPen(QtGui.QColor('#b53672'), 2, QtCore.Qt.DashLine), bounds)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw_pixmap()

    def mouseDoubleClickEvent(self, event):
        self.open_original()

    def open_original(self):
        if self.original_loader is not None:
            return self.original_loader()
        self.show_original(self.data)

    def show_original(self, data):
        if not data:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('观察图 · 滚轮缩放／拖动平移')
        dialog.resize(1100, 640)
        view = PersonView()
        view.show_frame(data, None, [], {})
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(view)
        QtCore.QTimer.singleShot(0, view.fit)
        dialog.exec_()


class ObservationCard(QtWidgets.QStackedWidget):
    """Rearrange display pixels; double-click always opens the original model image."""
    picked = QtCore.pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.data = None
        self.compact, self.context = False, None
        self.whole = ImageThumb()
        self.whole.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Ignored)
        self.addWidget(self.whole)
        divided = self.divided = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.rgb = ImageThumb((0,0,RGB_RECT[2],MODEL_SIZE[1]))
        self.bev = ImageThumb((BEV_RECT[0],0,BEV_RECT[2]-BEV_RECT[0],MODEL_SIZE[1]))
        for view in (self.whole, self.rgb, self.bev):
            view.picked.connect(self.picked)
        self.context_bev = PersonView(bev=True)
        self.context_bev.inspector = lambda: self.bev.open_original()
        self.context_bev.setMinimumSize(120,85)
        self.context_bev.picked.connect(self.picked)
        self.bev_stack = QtWidgets.QStackedWidget()
        self.bev_stack.addWidget(self.bev)
        self.bev_stack.addWidget(self.context_bev)
        for view in (self.rgb,self.bev_stack):
            view.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Ignored)
            divided.addWidget(view)
        divided.setSizes([280,320])
        self.addWidget(divided)
        self.setCurrentIndex(1)

    def set_compact(self, compact):
        self.compact = compact
        self.update_layout()
        self.show_data(self.data)
        self.updateGeometry()

    def update_layout(self):
        self.setCurrentIndex(0 if self.compact and self.context is None else 1)
        self.divided.setOrientation(QtCore.Qt.Horizontal if self.compact else QtCore.Qt.Vertical)
        self.bev_stack.setCurrentIndex(1 if self.context is not None else 0)

    def show_context(self, context, people=None, labels=None, extent=None):
        self.context = context
        if context is not None:
            row = context['frame']
            self.context_bev.show_frame(None, context['cloud'], people if people is not None else row['people'],
                                        labels or {}, context['trajectory'], extent, row.get('camera_fov'))
        self.update_layout()

    def minimumSizeHint(self):
        return QtCore.QSize(120,85 if self.compact else 175)

    def show_data(self, data, pixmap=None):
        self.data = data
        if pixmap is None:
            pixmap = QtGui.QPixmap()
            if data:
                pixmap.loadFromData(data)
        for view in (self.whole,self.rgb,self.bev):
            view.show_data(data,pixmap)

    def set_interaction(self, *args, **kwargs):
        for view in (self.whole, self.rgb, self.bev):
            view.set_interaction(*args, **kwargs)

    def highlight(self, selected):
        self.context_bev.highlight(selected)
        for view in (self.whole, self.rgb, self.bev):
            view.selected = set(selected)
            view.update()


class Timeline(QtWidgets.QWidget):
    jumped = QtCore.pyqtSignal(float)
    range_changed = QtCore.pyqtSignal(float, float)
    range_selected = QtCore.pyqtSignal(float, float)
    event_selected = QtCore.pyqtSignal(str)
    event_edited = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(82)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.frames, self.event, self.target, self.current = [], None, None, 0
        self.events, self.times, self.lookup = [], [], {}
        self.selection, self.press, self.drag = None, None, None

    def show_state(self, frames, event, target, current, events=()):
        if frames is not self.frames:
            self.times = [r['elapsed'] for r in frames]
            self.lookup = {r['id']: r['elapsed'] for r in frames}
        self.frames, self.event, self.target, self.current = frames, event, target, current
        self.events = list(events)
        self.update()

    def set_selection(self, selection):
        self.selection = selection
        self.update()

    def xpos(self, elapsed):
        a, b = (self.frames[0]['elapsed'], self.frames[-1]['elapsed']) if self.frames else (0, 1)
        return 35 + (self.width()-48)*(elapsed-a)/max(b-a, .001)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#edf2f7'))
        if not self.frames:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for event in sorted(self.events,key=lambda e:(-(e['end']-e['start']),e['id']==(self.event or {}).get('id'),e['id'])):
            a, b = self.xpos(event['start']), self.xpos(event['end'])
            active = event['id']==(self.event or {}).get('id')
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor('#7caacb' if active else '#c1cbd4'))
            painter.drawRoundedRect(QtCore.QRectF(a,2,max(3,b-a),17),3,3)
            painter.setPen(QtGui.QColor('#193953'))
            text = painter.fontMetrics().elidedText(event['name'],QtCore.Qt.ElideRight,max(0,int(b-a)-6))
            painter.drawText(QtCore.QRectF(a+3,2,max(0,b-a-6),17),QtCore.Qt.AlignVCenter,text)
        if self.event:
            a, b = self.xpos(self.event['start']), self.xpos(self.event['end'])
            painter.fillRect(QtCore.QRectF(a,21,b-a,self.height()-34),QtGui.QColor('#dae9f5'))
        if self.selection is not None:
            a,b = [self.xpos(value) for value in self.selection]
            painter.fillRect(QtCore.QRectF(a,21,max(2,b-a),self.height()-34),QtGui.QColor(235,170,40,65))
            painter.setPen(QtGui.QPen(QtGui.QColor('#bb7c0b'),2))
            for x in (a,b):
                painter.drawLine(QtCore.QPointF(x,20),QtCore.QPointF(x,self.height()-12))
                painter.fillRect(QtCore.QRectF(x-3,43,6,10),QtGui.QColor('#bb7c0b'))
        painter.setPen(QtGui.QColor('#4b6274'))
        painter.drawText(3,15,'事件')
        painter.drawText(3,39,'T1')
        painter.drawText(3,61,'T2')
        for y, labels in ((36, self.event.get('t1', {}) if self.event else {}),
                          (58, self.target.get('samples', {}) if self.target else {})):
            for ident, label in labels.items():
                if ident not in self.lookup:
                    continue
                x = self.xpos(self.lookup[ident])
                painter.setBrush(QtGui.QColor(STATUS_COLORS[label['status']]))
                painter.setPen(QtCore.Qt.NoPen)
                if label['status'] == 'draft':
                    painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y-5), QtCore.QPointF(x-4, y+4), QtCore.QPointF(x+4, y+4)]))
                elif label['status'] == 'skipped':
                    painter.drawRect(QtCore.QRectF(x-3, y-3, 6, 6))
                else:
                    painter.drawEllipse(QtCore.QPointF(x, y), 4, 4)
        painter.setPen(QtGui.QPen(QtGui.QColor('#193953'), 2))
        x = int(self.xpos(self.current))
        painter.drawLine(x, 0, x, self.height()-10)
        painter.setPen(QtGui.QColor('#4b6274'))
        painter.drawText(35, self.height()-1, f'{self.frames[0]["elapsed"]:.1f}s')
        painter.drawText(self.width()-60, self.height()-1, f'{self.frames[-1]["elapsed"]:.1f}s')

    def elapsed_at(self, x):
        a, b = self.frames[0]['elapsed'], self.frames[-1]['elapsed']
        return a + min(1, max(0, (x-35)/max(1, self.width()-48)))*(b-a)

    def snap_at(self, x):
        elapsed = self.elapsed_at(x)
        index = bisect_left(self.times,elapsed)
        choices = self.times[max(0,index-1):min(len(self.times),index+1)]
        return max(0,round(min(choices,key=lambda t:abs(t-elapsed)),3))

    def event_at(self, x):
        elapsed = self.elapsed_at(x)
        choices = [e for e in self.events if e['start']-.001<=elapsed<=e['end']+.001]
        return min(choices,key=lambda e:(e['end']-e['start'],e['id']!=(self.event or {}).get('id'),e['id'])) if choices else None

    def mousePressEvent(self, event):
        if self.frames and event.button() == QtCore.Qt.LeftButton:
            self.setFocus(QtCore.Qt.MouseFocusReason)
            handle = None
            if self.selection is not None:
                distances = [(abs(self.xpos(value)-event.x()),index) for index,value in enumerate(self.selection)]
                distance,index = min(distances)
                if distance<=7:
                    handle=index
            self.press = (event.pos(),self.snap_at(event.x()),handle,self.selection)
            self.drag = None
            event.accept()

    def mouseMoveEvent(self, event):
        if self.frames:
            if self.press and event.buttons() & QtCore.Qt.LeftButton and abs(event.x()-self.press[0].x())>=5:
                _,anchor,handle,original = self.press
                current = self.snap_at(event.x())
                if handle is None:
                    selected = tuple(sorted((anchor,current)))
                elif handle==0:
                    selected = (min(current,original[1]),original[1])
                else:
                    selected = (original[0],max(current,original[0]))
                if selected != self.drag:
                    self.drag=selected
                    self.set_selection(selected)
                    self.range_changed.emit(*selected)
                return
            matched = self.event_at(event.x()) if event.y()<21 else None
            near_edge = self.selection is not None and any(abs(self.xpos(t)-event.x())<=7 for t in self.selection)
            self.setCursor(QtCore.Qt.SizeHorCursor if near_edge else QtCore.Qt.PointingHandCursor if matched else QtCore.Qt.CrossCursor)
            message = f'{matched["name"]} · {matched["start"]:.3f}–{matched["end"]:.3f}s\n' if matched else ''
            self.setToolTip(message+f'{self.elapsed_at(event.x()):.2f}s · 拖动选区／拖边界调整；点击浏览；双击事件条编辑')

    def mouseReleaseEvent(self, event):
        if not self.frames or self.press is None or event.button()!=QtCore.Qt.LeftButton:
            return
        selected = self.drag
        self.press,self.drag = None,None
        if selected is not None:
            self.range_selected.emit(*selected)
        elif event.modifiers() & QtCore.Qt.ShiftModifier:
            elapsed=self.snap_at(event.x())
            self.range_selected.emit(elapsed,elapsed)
        else:
            matched = self.event_at(event.x()) if event.y()<21 else None
            if matched:
                self.event_selected.emit(matched['id'])
            self.jumped.emit(self.snap_at(event.x()))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.frames and event.button()==QtCore.Qt.LeftButton and event.y()<21:
            matched=self.event_at(event.x())
            if matched:
                self.event_edited.emit(matched['id'])
                self.press,self.drag=None,None
                event.accept()
                return
        super().mouseDoubleClickEvent(event)
