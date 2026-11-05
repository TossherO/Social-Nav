"""Interactive CPU-rendered 3D scan; no OpenGL/ROS dependency on either client."""
import io
import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


class CloudView(QtWidgets.QWidget):
    picked = QtCore.pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.people, self.labels, self.selected, self.hits = [], {}, set(), []
        self.point_size, self.revision, self.cached = 2, 0, None
        self.press = self.previous = None
        self.dragged, self.button = False, None
        self.fit()

    def fit(self):
        self.azimuth, self.elevation, self.distance = -140., 52., 25.
        self.center = np.array([5., 0., -.3])
        self.cached = None
        self.update()

    def top_view(self):
        self.azimuth, self.elevation = 180., 89.5
        self.cached = None
        self.update()

    def set_point_size(self, value):
        self.point_size = int(value)
        self.cached = None
        self.update()

    def show_frame(self, raw, people, labels):
        if raw is not getattr(self, 'raw', None):
            self.raw = raw
            cloud = np.load(io.BytesIO(raw), allow_pickle=False) if raw else np.empty((0,4))
            if cloud.ndim != 2 or cloud.shape[1] != 4 or len(cloud)>100000:
                raise ValueError('三维点云须为有界Nx4数组')
            self.points = np.asarray(cloud[np.isfinite(cloud).all(axis=1), :3], dtype=np.float32)
            level = np.clip((self.points[:,2]+1.3)/3.5, 0, 1)
            palette = np.array([[104,169,255],[65,215,224],[126,235,146],[255,213,101],[255,133,132]])
            index = np.minimum((level*4).astype(int),3)
            weight = (level*4-index)[:,None]
            self.colors = np.rint(palette[index]*(1-weight)+palette[index+1]*weight).astype(np.uint8)
            self.revision += 1
            self.cached = None
        self.people, self.labels = people, labels
        self.update()

    def highlight(self, keys):
        self.selected = set(keys)
        self.update()

    def basis(self):
        az, el = np.deg2rad([self.azimuth, self.elevation])
        backward = np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        forward = -backward
        right = np.cross(forward,[0.,0.,1.]);right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        return self.center+self.distance*backward,right,up,forward

    def project(self, xyz):
        eye,right,up,forward = self.basis()
        relative = np.asarray(xyz)-eye
        depth = relative@forward
        focal = max(1,min(self.width(),self.height()))*1.25
        safe = np.maximum(depth,.05)
        screen = np.column_stack((self.width()/2+relative@right*focal/safe,
                                  self.height()/2-relative@up*focal/safe))
        return screen,depth

    def cloud_image(self):
        key = (self.revision,self.width(),self.height(),self.point_size,self.azimuth,self.elevation,
               self.distance,tuple(self.center))
        if self.cached and self.cached[0] == key:
            return self.cached[1]
        width,height = max(1,self.width()),max(1,self.height())
        pixels = np.empty((height,width,3),dtype=np.uint8);pixels[:]=[19,25,36]
        if len(self.points):
            screen,depth = self.project(self.points)
            uv = np.rint(screen).astype(int)
            valid = (depth>.2)&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
            uv,z,colors = uv[valid],depth[valid],self.colors[valid]
            # Nearest-depth point wins, independently of height/color order.
            order = np.argsort(z)
            _,first = np.unique((uv[:,1]*width+uv[:,0])[order],return_index=True)
            chosen = order[first]
            zbuffer = np.full((height,width),np.inf,dtype=np.float32)
            for dx in range(-(self.point_size//2), (self.point_size+1)//2):
                for dy in range(-(self.point_size//2), (self.point_size+1)//2):
                    pos = uv[chosen]+[dx,dy]
                    inside=(pos[:,0]>=0)&(pos[:,0]<width)&(pos[:,1]>=0)&(pos[:,1]<height)
                    candidate=chosen[inside];pos=pos[inside]
                    closer=z[candidate]<=zbuffer[pos[:,1],pos[:,0]]
                    pos=pos[closer];candidate=candidate[closer]
                    zbuffer[pos[:,1],pos[:,0]]=z[candidate]
                    pixels[pos[:,1],pos[:,0]]=colors[candidate]
        image = QtGui.QImage(pixels.data,width,height,pixels.strides[0],QtGui.QImage.Format_RGB888).copy()
        self.cached=(key,image)
        return image

    def line(self, painter, a, b, color, width=1):
        xy,depth = self.project([a,b])
        if (depth>.2).all():
            painter.setPen(QtGui.QPen(QtGui.QColor(color),width))
            painter.drawLine(QtCore.QPointF(*xy[0]),QtCore.QPointF(*xy[1]))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0,0,self.cloud_image())
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for value in range(-10,26,2):
            self.line(painter,[value,-15,0],[value,15,0],'#664c617a')
        for value in range(-14,16,2):
            self.line(painter,[-10,value,0],[25,value,0],'#664c617a')
        for end,color in [([2,0,0],'#ff847e'),([0,2,0],'#75e3a4'),([0,0,2],'#8dbfff')]:
            self.line(painter,[0,0,0],end,color,2)
        self.hits=[]
        for person in self.people:
            if person.get('box') is None:
                continue
            box=np.asarray(person['box']);c,s=math.cos(box[6]),math.sin(box[6])
            corners=np.array([[x,y,z] for z in (-.5,.5) for x,y in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))])*box[3:6]
            corners=corners@np.array([[c,s,0],[-s,c,0],[0,0,1]])+box[:3]
            xy,depth=self.project(corners)
            if not (depth>.2).all():
                continue
            key=person['key']
            color='#ff709a' if key in self.selected else '#55dbea' if person.get('observed') else '#ffba5a'
            pen=QtGui.QPen(QtGui.QColor(color),3 if key in self.selected else 1.6)
            if not person.get('observed'):pen.setStyle(QtCore.Qt.DashLine)
            painter.setPen(pen)
            for a,b in EDGES:painter.drawLine(QtCore.QPointF(*xy[a]),QtCore.QPointF(*xy[b]))
            lo,hi=xy.min(0),xy.max(0)
            rect=QtCore.QRectF(lo[0]-5,lo[1]-20,hi[0]-lo[0]+10,hi[1]-lo[1]+25)
            self.hits.append((key,rect,float(depth.mean())))
            text=self.labels.get(key,key.rsplit(':',1)[-1])
            font=painter.font();font.setPointSize(10);font.setBold(key in self.selected);painter.setFont(font)
            bounds=painter.fontMetrics().boundingRect(text)
            label=QtCore.QRectF(lo[0],lo[1]-22,bounds.width()+8,21)
            painter.fillRect(label,QtGui.QColor('#d9162132'))
            painter.drawText(label.adjusted(4,0,0,0),QtCore.Qt.AlignVCenter,text)
        painter.setPen(QtGui.QColor('#dce8f5'))
        painter.drawText(12,23,f'当前扫描 · {len(self.points):,}点 · 雷达坐标／米 · z=0参考网格2m')
        painter.drawText(12,self.height()-14,'左拖旋转 · 右拖平移 · 滚轮缩放 · 点击框选人 · Ctrl多选')
        painter.end()

    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton,QtCore.Qt.RightButton):
            self.press=self.previous=event.pos();self.button=event.button();self.dragged=False
            self.setFocus();event.accept()

    def mouseMoveEvent(self, event):
        if self.previous is None:
            return
        delta=event.pos()-self.previous;self.previous=event.pos()
        self.dragged=self.dragged or (event.pos()-self.press).manhattanLength()>4
        if self.button==QtCore.Qt.LeftButton:
            self.azimuth-=delta.x()*.4
            self.elevation=float(np.clip(self.elevation+delta.y()*.35,-10,89.5))
        else:
            _,right,up,_=self.basis()
            self.center+=(-right*delta.x()+up*delta.y())*self.distance/max(1,min(self.width(),self.height()))/1.25
        self.cached=None;self.update();event.accept()

    def mouseReleaseEvent(self, event):
        if self.press is not None and not self.dragged and self.button==QtCore.Qt.LeftButton:
            matches=[(depth,key) for key,rect,depth in self.hits if rect.contains(event.pos())]
            if matches:
                self.picked.emit(min(matches)[1],bool(event.modifiers()&QtCore.Qt.ControlModifier))
        self.press=self.previous=None;self.button=None
        event.accept()

    def wheelEvent(self, event):
        self.distance=float(np.clip(self.distance*math.exp(-event.angleDelta().y()/120*.15),2.,120.))
        self.cached=None;self.update();event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit();event.accept()
