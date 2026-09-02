"""Top-down (ENU x-y) 2D view of the scene: obstacles, drone, goal, trail."""

import math

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget


class ObstacleView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self._pos = None            # (x, y) ENU, m
        self._yaw = 0.0             # rad
        self._goal = None           # (x, y)
        self._obstacles = []        # [(x, y, r)] detected
        self._expected = []         # [(x, y, r)] frozen-scene ground truth
        self._trail = []            # [(x, y)]
        self._ref_trail = []        # [(x, y)] exported SITL reference path
        self._min_dist = None

    # data in
    def set_detected_obstacles(self, obs):
        self._obstacles = list(obs)
        self.update()

    def set_expected_obstacles(self, obs):
        self._expected = list(obs)
        self.update()

    def set_reference_trail(self, pts):
        """[(x, y), ...] -- the exported SITL twin's path, drawn as a faint
        dashed ghost for the operator to compare the live run against."""
        self._ref_trail = [(p[0], p[1]) for p in (pts or [])]
        self.update()

    def set_goal(self, xy):
        self._goal = xy
        self.update()

    def set_min_dist(self, d):
        self._min_dist = d
        self.update()

    def set_pose(self, x, y, yaw):
        self._pos = (x, y)
        self._yaw = yaw
        self._trail.append((x, y))
        if len(self._trail) > 2000:
            self._trail = self._trail[-2000:]
        self.update()

    def clear_trail(self):
        self._trail = []
        self.update()

    # world <-> screen
    def _bounds(self):
        xs, ys = [], []
        for (x, y, r) in list(self._obstacles) + list(self._expected):
            xs += [x - r, x + r]
            ys += [y - r, y + r]
        for pt in ([self._pos, self._goal] + self._trail + self._ref_trail):
            if pt is not None:
                xs.append(pt[0])
                ys.append(pt[1])
        if not xs:
            return (-3.0, 3.0, -1.0, 7.0)
        pad = 1.0
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def _make_tf(self):
        x0, x1, y0, y1 = self._bounds()
        w = max(self.width() - 20, 1)
        h = max(self.height() - 20, 1)
        sx = w / max(x1 - x0, 1e-3)
        sy = h / max(y1 - y0, 1e-3)
        s = min(sx, sy)
        # centre the content
        cx = 10 + (w - s * (x1 - x0)) / 2.0
        cy = 10 + (h - s * (y1 - y0)) / 2.0

        def tf(x, y):
            return QPointF(cx + (x - x0) * s, self.height() - (cy + (y - y0) * s))

        return tf, s

    # paint
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(20, 22, 26))
        tf, s = self._make_tf()

        # grid every 1 m
        x0, x1, y0, y1 = self._bounds()
        p.setPen(QPen(QColor(45, 48, 54), 1))
        gx = math.floor(x0)
        while gx <= x1:
            p.drawLine(tf(gx, y0), tf(gx, y1))
            gx += 1
        gy = math.floor(y0)
        while gy <= y1:
            p.drawLine(tf(x0, gy), tf(x1, gy))
            gy += 1

        # expected (ground-truth) obstacles: dashed outline
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(120, 120, 130), 1, Qt.DashLine))
        for (x, y, r) in self._expected:
            c = tf(x, y)
            p.drawEllipse(c, r * s, r * s)

        # detected obstacles: filled
        p.setPen(QPen(QColor(255, 140, 0), 2))
        p.setBrush(QBrush(QColor(255, 140, 0, 90)))
        for (x, y, r) in self._obstacles:
            c = tf(x, y)
            p.drawEllipse(c, r * s, r * s)

        # SITL reference path: faint dashed ghost
        if len(self._ref_trail) > 1:
            p.setPen(QPen(QColor(150, 150, 160, 120), 1, Qt.DashLine))
            p.drawPolyline(QPolygonF([tf(x, y) for (x, y) in self._ref_trail]))

        # trail
        if len(self._trail) > 1:
            p.setPen(QPen(QColor(90, 170, 255, 160), 2))
            poly = QPolygonF([tf(x, y) for (x, y) in self._trail])
            p.drawPolyline(poly)

        # goal
        if self._goal is not None:
            g = tf(*self._goal)
            p.setPen(QPen(QColor(60, 220, 120), 2))
            p.setBrush(Qt.NoBrush)
            p.drawLine(QPointF(g.x() - 8, g.y()), QPointF(g.x() + 8, g.y()))
            p.drawLine(QPointF(g.x(), g.y() - 8), QPointF(g.x(), g.y() + 8))
            p.drawEllipse(g, 6, 6)

        # drone: triangle pointing along yaw
        if self._pos is not None:
            c = tf(*self._pos)
            L = 12
            nose = QPointF(c.x() + L * math.cos(self._yaw), c.y() - L * math.sin(self._yaw))
            left = QPointF(c.x() + L * math.cos(self._yaw + 2.6),
                           c.y() - L * math.sin(self._yaw + 2.6))
            right = QPointF(c.x() + L * math.cos(self._yaw - 2.6),
                            c.y() - L * math.sin(self._yaw - 2.6))
            p.setPen(QPen(QColor(120, 200, 255), 2))
            p.setBrush(QBrush(QColor(120, 200, 255, 180)))
            p.drawPolygon(QPolygonF([nose, left, right]))

        # min-dist readout
        if self._min_dist is not None and self._min_dist < 1e8:
            p.setPen(QColor(220, 220, 220))
            p.drawText(QRectF(8, 6, self.width() - 16, 18),
                       Qt.AlignLeft, f'min obstacle dist: {self._min_dist:.2f} m')
        p.setPen(QColor(120, 120, 130))
        p.drawText(QRectF(8, self.height() - 20, self.width() - 16, 16),
                   Qt.AlignRight, '1 m grid  ·  ENU top-down')
        p.end()
