"""Light Qt graphics; independent of the detector annotation UI."""
import io
import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


STATUS_COLORS = {'draft': '#d6a52c', 'skipped': '#87939e', 'complete': '#2b9c68'}


class PersonRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, key, rect, signal):
        super().__init__(QtCore.QRectF(*rect))
        self.key, self.signal = key, signal
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.setZValue(2)

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
        self.setBackgroundBrush(QtGui.QColor('#edf3f8'))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(self.ScrollHandDrag)
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setMinimumSize(230, 180)
        self.items_by_key, self.texts = {}, {}
        self.initial = True
        self.draw_box, self.draw_start, self.rubber = False, None, None
        self.cloud, self.people, self.labels = None, [], {}
        self.point_size = 2
        self.factor = 20.
        self.image_size = None

    def wheelEvent(self, event):
        self.scale(1.18 if event.angleDelta().y() > 0 else 1/1.18, 1.18 if event.angleDelta().y() > 0 else 1/1.18)
        event.accept()

    def fit(self):
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.initial:
            self.fit()

    def show_frame(self, image, cloud, people, labels):
        self.scene().clear()
        self.draw_start, self.rubber, self.draw_box = None, None, False
        self.items_by_key, self.texts = {}, {}
        self.people, self.labels, self.cloud = people, labels, cloud
        if self.bev:
            positions = [p['box'][:2] for p in people if p.get('box') is not None]
            xs, ys = [-5., 25.]+[p[0] for p in positions], [-15., 15.]+[p[1] for p in positions]
            xmin, xmax = math.floor(min(xs))-1, math.ceil(max(xs))+1
            ymin, ymax = math.floor(min(ys))-1, math.ceil(max(ys))+1
            self.factor = min(20., 1800/max(xmax-xmin, ymax-ymin))
            self.setSceneRect(-ymax*self.factor, -xmax*self.factor,
                              math.ceil((ymax-ymin)*self.factor), math.ceil((xmax-xmin)*self.factor))
            pix = self.cloud_pixmap(cloud)
            background = self.scene().addPixmap(pix)
            bounds = self.sceneRect()
            background.setPos(bounds.left(), bounds.top())
            step = max(1, math.ceil(max(xmax-xmin, ymax-ymin)/80))
            for y in range(ymin, ymax+1, step):
                self.scene().addLine(-y*self.factor, bounds.top(), -y*self.factor, bounds.bottom(), QtGui.QPen(QtGui.QColor('#d2dde8')))
            for x in range(xmin, xmax+1, step):
                self.scene().addLine(bounds.left(), -x*self.factor, bounds.right(), -x*self.factor, QtGui.QPen(QtGui.QColor('#d2dde8')))
            self.scene().addEllipse(-4, -4, 8, 8, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(QtGui.QColor('#203b52')))
            self.scene().addLine(0, 0, 0, -20, QtGui.QPen(QtGui.QColor('#203b52'), 2))
            title = self.scene().addText(f'米制BEV · 前方↑ · 左侧← · 网格{step}m')
            title.setPos(bounds.left()+5, bounds.top()+3)
            title.setDefaultTextColor(QtGui.QColor('#294459'))
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
            if self.bev:
                if person.get('box') is None:
                    continue
                x, y = person['box'][:2]
                rect = [-y*self.factor-7, -x*self.factor-7, 14, 14]
            else:
                b = person.get('rgb_box')
                if not b:
                    continue
                rect = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
            item = PersonRect(key, rect, self.picked)
            color = '#008aac' if person.get('eligible') else '#8d7f66'
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.scene().addItem(item)
            self.items_by_key[key] = item
            text = self.scene().addText(labels.get(key, str(index+1)))
            text.setDefaultTextColor(QtGui.QColor('#004d68'))
            text.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(rect[0], rect[1]-20)
            text.setZValue(3)
            self.texts[key] = text
        if self.initial:
            self.fit()
            self.initial = False

    def cloud_pixmap(self, raw):
        bounds = self.sceneRect()
        width, height = math.ceil(bounds.width()), math.ceil(bounds.height())
        pixels = np.full((height, width, 3), [237, 243, 248], dtype=np.uint8)
        if raw:
            cloud = np.load(io.BytesIO(raw), allow_pickle=False)
            if cloud.ndim != 2 or cloud.shape[1] != 4 or len(cloud) > 100000 or cloud.dtype.kind != 'f':
                raise ValueError('点云浏览资产布局错误')
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            xx = np.rint(-cloud[:, 1]*self.factor-bounds.left()).astype(int)
            yy = np.rint(-cloud[:, 0]*self.factor-bounds.top()).astype(int)
            valid = (xx >= 3) & (xx < width-3) & (yy >= 3) & (yy < height-3)
            xx, yy, zz = xx[valid], yy[valid], cloud[valid, 2]
            ramp = np.clip((zz+1.5)/4, 0, 1)
            colors = np.column_stack([40+90*ramp, 119-40*ramp, 170+45*ramp]).astype(np.uint8)
            radius = (self.point_size-1)//2
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    pixels[yy+dy, xx+dx] = colors
        return QtGui.QPixmap.fromImage(QtGui.QImage(pixels.data, width, height, width*3, QtGui.QImage.Format_RGB888).copy())

    def highlight(self, selected):
        for key, item in self.items_by_key.items():
            pen = QtGui.QPen(QtGui.QColor('#f06418' if key in selected else '#008aac'), 3 if key in selected else 1.5)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 157, 65, 35)) if key in selected else QtGui.QBrush(QtCore.Qt.NoBrush))

    def mousePressEvent(self, event):
        if self.draw_box and event.button() == QtCore.Qt.LeftButton:
            self.draw_start = self.mapToScene(event.pos())
            self.rubber = self.scene().addRect(QtCore.QRectF(self.draw_start, self.draw_start), QtGui.QPen(QtGui.QColor('#e66b20'), 2))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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


class ImageThumb(QtWidgets.QLabel):
    def __init__(self):
        super().__init__('无历史')
        self.setMinimumSize(120, 85)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet('background:#e8eef4;border:1px solid #cbd5df')
        self.data = None

    def show_data(self, data):
        self.data = data
        pix = QtGui.QPixmap()
        if data:
            pix.loadFromData(data)
        if pix.isNull():
            self.clear()
            self.setText('历史缺失')
        else:
            self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_data(self.data)

    def mouseDoubleClickEvent(self, event):
        if not self.data:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('观察图 · 滚轮缩放／拖动平移')
        dialog.resize(1100, 640)
        view = PersonView()
        view.show_frame(self.data, None, [], {})
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(view)
        QtCore.QTimer.singleShot(0, view.fit)
        dialog.exec_()


class Timeline(QtWidgets.QWidget):
    jumped = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setMouseTracking(True)
        self.frames, self.event, self.target, self.current = [], None, None, 0

    def show_state(self, frames, event, target, current):
        self.frames, self.event, self.target, self.current = frames, event, target, current
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
        if self.event:
            a, b = self.xpos(self.event['start']), self.xpos(self.event['end'])
            painter.fillRect(QtCore.QRectF(a, 0, b-a, self.height()), QtGui.QColor('#dae9f5'))
        painter.setPen(QtGui.QColor('#4b6274'))
        painter.drawText(3, 21, 'T1')
        painter.drawText(3, 43, 'T2')
        lookup = {r['id']: r['elapsed'] for r in self.frames}
        for y, labels in ((18, self.event.get('t1', {}) if self.event else {}),
                          (40, self.target.get('samples', {}) if self.target else {})):
            for ident, label in labels.items():
                if ident not in lookup:
                    continue
                x = self.xpos(lookup[ident])
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

    def mousePressEvent(self, event):
        if self.frames and event.button() == QtCore.Qt.LeftButton:
            self.jumped.emit(self.elapsed_at(event.x()))

    def mouseMoveEvent(self, event):
        if self.frames:
            self.setToolTip(f'{self.elapsed_at(event.x()):.2f}s · 黄△待标／灰□跳过／绿●已标；点击仅浏览')
