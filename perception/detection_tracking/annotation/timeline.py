"""Time-proportional bag browsing with three annotation-state lanes; no I/O in paint."""
from PyQt5 import QtCore, QtGui, QtWidgets

from .data import ANNOTATION_STATES, FRAME_STATUS_NAMES
from .appearance import ANNOTATION_COLORS, review_display_status


STATE_COLORS = ANNOTATION_COLORS
LANES = dict(draft=24, skipped=43, complete=62)


class BagTimeline(QtWidgets.QWidget):
    seek = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows, self.by_id, self.allowed = [], {}, set()
        self.current, self.preview = None, None
        self.begin, self.end, self.dragging = 0, 0, False
        self.setMinimumHeight(103)
        self.setMaximumHeight(125)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def set_frames(self, rows, allowed=None):
        self.rows = [dict(row) for row in rows]
        self.by_id = {row['id']: row for row in self.rows}
        self.allowed = set(self.by_id) if allowed is None else set(allowed)
        self.begin = min((row['record_ns'] for row in self.rows), default=0)
        self.end = max((row['record_ns'] for row in self.rows), default=self.begin)
        self.current = self.preview = None
        self.update()

    def set_current(self, ident):
        self.current = ident if ident in self.by_id else None
        self.update()

    def set_review(self, ident, review):
        if ident in self.by_id:
            self.by_id[ident]['status'] = review['status']
            self.by_id[ident]['display_status'] = review_display_status(review)
            self.update()

    def x_at(self, stamp):
        return 61 + (stamp - self.begin) / max(1, self.end - self.begin) * max(1, self.width() - 83)

    def frame_at(self, position):
        if not self.rows:
            return None
        lane = next((status for status, y in LANES.items() if abs(position.y() - y) <= 7), None)
        markers = [row for row in self.rows if row['status'] in ANNOTATION_STATES and
                   (lane is None or row['status'] == lane) and abs(self.x_at(row['record_ns']) - position.x()) <= 6]
        if markers:
            return min(markers, key=lambda row: (abs(self.x_at(row['record_ns']) - position.x()), row['ordinal']))
        fraction = min(1., max(0., (position.x() - 61) / max(1, self.width() - 83)))
        target = self.begin + round(fraction * (self.end - self.begin))
        return min(self.rows, key=lambda row: (abs(row['record_ns'] - target), row['ordinal']))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#172029'))
        painter.setPen(QtGui.QColor('#c6d4de'))
        painter.drawText(8, 12, 'ROS包')
        for x, state, label in [(72, 'complete', '已标有人'), (160, 'complete_empty', '已标无人')]:
            painter.setPen(QtGui.QColor(STATE_COLORS[state]))
            painter.drawText(x, 12, label)
        for state, y in LANES.items():
            painter.setPen(QtGui.QColor(STATE_COLORS[state]))
            painter.drawText(8, y + 4, FRAME_STATUS_NAMES[state])
            painter.setPen(QtGui.QPen(QtGui.QColor('#34434e'), 1))
            painter.drawLine(QtCore.QPointF(61, y), QtCore.QPointF(self.width() - 22, y))
            # Dim the range excluded by --start/--duration, keeping every bag marker visible.
            kinds = ('complete', 'complete_empty') if state == 'complete' else (state,)
            for kind in kinds:
                for allowed, size in [(True, 5), (False, 3)]:
                    color = QtGui.QColor(STATE_COLORS[kind])
                    painter.setPen(QtGui.QPen(color if allowed else color.darker(240), 2))
                    lines = [QtCore.QLineF(self.x_at(row['record_ns']), y - size, self.x_at(row['record_ns']), y + size)
                             for row in self.rows if row.get('display_status', row['status']) == kind
                             and (row['id'] in self.allowed) == allowed]
                    if lines:
                        painter.drawLines(lines)
        painter.setPen(QtGui.QColor('#c6d4de'))
        for fraction in (0., .25, .5, .75, 1.):
            stamp = self.begin + round(fraction * (self.end - self.begin))
            x = self.x_at(stamp)
            text = '%.1fs' % ((stamp - self.begin) / 1e9)
            left = max(4, min(self.width() - 68, x - 32))
            painter.drawText(QtCore.QRectF(left, 80, 64, 18), QtCore.Qt.AlignCenter, text)
        for ident, color, style in [(self.current, '#ffffff', QtCore.Qt.SolidLine),
                                     (self.preview, '#9fc9f4', QtCore.Qt.DashLine)]:
            if ident in self.by_id:
                x = self.x_at(self.by_id[ident]['record_ns'])
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 1, style))
                painter.drawLine(QtCore.QPointF(x, 15), QtCore.QPointF(x, 74))
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = True
            self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        row = self.frame_at(event.pos())
        if row is None:
            return
        name = ('已标（无人）' if row.get('display_status') == 'complete_empty'
                else FRAME_STATUS_NAMES[row['status']])
        text = '%.3f s · 原始帧 %d · %s' % ((row['record_ns'] - self.begin) / 1e9,
                row['ordinal'], name)
        if row['id'] not in self.allowed:
            text += ' · 不在当前 --start/--duration 范围内'
        self.setToolTip(text)
        if self.dragging:
            self.preview = row['id']
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.dragging:
            self.dragging = False
            row = self.frame_at(event.pos())
            self.preview = None
            self.update()
            if row is not None:
                self.seek.emit(row['id'])
