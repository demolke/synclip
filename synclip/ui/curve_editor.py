"""
CurveEditor - a small freeform response-curve widget.

Maps an input blendshape weight (x, 0..1) to an output weight (y, 0..1) via
draggable control points. The shape is baked into a lookup table the capture
filter samples per frame, so you can e.g. boost low values while letting large
ones fall off toward 1.0.

Interaction:
    left-drag a point   move it (endpoints keep x at 0 / 1)
    left-click empty     add a new point there
    right-click a point  remove it (endpoints can't be removed)
    right-click empty     menu -> reset the curve to the identity line
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

# Curve baking is pure math and shared with the headless modifier pipeline, so it
# lives in the engine layer (curve_lut) to keep that layer Qt-free.
from ..curve_lut import LUT_SIZE as _LUT_SIZE, build_lut, clamp01 as _clamp01

_HIT_RADIUS = 12  # px


class CurveEditor(QWidget):
    """Draggable 0..1 -> 0..1 response curve. Emits the baked LUT on change."""

    curve_changed = Signal(list)  # list[float] of length _LUT_SIZE

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 160)
        # Default = identity line (no change).
        self._points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._drag_idx: int | None = None
        self.setMouseTracking(True)

    # ------------------------------------------------------------------

    def lut(self) -> list[float]:
        return build_lut(self._points)

    def points(self) -> list[list[float]]:
        """Control points as JSON-friendly [x, y] pairs."""
        return [[x, y] for (x, y) in self._points]

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = sorted(points)
        self.update()
        self.curve_changed.emit(self.lut())

    def reset_to_identity(self) -> None:
        """Restore the default identity (no-change) curve."""
        self.set_points([(0.0, 0.0), (1.0, 1.0)])

    # ------------------------------------------------------------------
    # Coordinate mapping (curve space <-> widget pixels)
    # ------------------------------------------------------------------

    def _margin(self) -> int:
        return 10

    def _to_px(self, x: float, y: float) -> QPointF:
        m = self._margin()
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        return QPointF(m + x * w, m + (1.0 - y) * h)

    def _to_curve(self, px: float, py: float) -> tuple[float, float]:
        m = self._margin()
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        x = (px - m) / w if w else 0.0
        y = 1.0 - (py - m) / h if h else 0.0
        return _clamp01(x), _clamp01(y)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 34))

        # Grid + identity reference line.
        p.setPen(QPen(QColor(60, 60, 66), 1))
        for f in (0.25, 0.5, 0.75):
            a = self._to_px(f, 0.0)
            b = self._to_px(f, 1.0)
            p.drawLine(a, b)
            c = self._to_px(0.0, f)
            d = self._to_px(1.0, f)
            p.drawLine(c, d)
        p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DashLine))
        p.drawLine(self._to_px(0.0, 0.0), self._to_px(1.0, 1.0))

        # The baked curve.
        lut = self.lut()
        p.setPen(QPen(QColor(120, 200, 120), 2))
        prev = None
        for j, y in enumerate(lut):
            x = j / (len(lut) - 1)
            pt = self._to_px(x, y)
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        # Control points.
        p.setPen(QPen(QColor(220, 220, 120), 1))
        p.setBrush(QColor(220, 220, 120))
        for (x, y) in self._points:
            p.drawEllipse(self._to_px(x, y), 4, 4)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _hit_test(self, pos) -> int | None:
        for i, (x, y) in enumerate(self._points):
            if (self._to_px(x, y) - QPointF(pos)).manhattanLength() <= _HIT_RADIUS:
                return i
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        idx = self._hit_test(pos)

        if event.button() == Qt.MouseButton.RightButton:
            # On a point: remove it (never the two endpoints).
            if idx is not None and 0 < idx < len(self._points) - 1:
                del self._points[idx]
                self.update()
                self.curve_changed.emit(self.lut())
            elif idx is None:
                # On empty space: offer a reset so accidental points can be
                # cleared without removing them one by one.
                menu = QMenu(self)
                act_reset = menu.addAction("Reset curve to identity")
                if menu.exec(event.globalPosition().toPoint()) == act_reset:
                    self.reset_to_identity()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if idx is None:
                # Add a new point at the click and start dragging it.
                cx, cy = self._to_curve(pos.x(), pos.y())
                self._points.append((cx, cy))
                self._points.sort()
                idx = self._points.index((cx, cy))
            self._drag_idx = idx
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_idx is None:
            return
        i = self._drag_idx
        cx, cy = self._to_curve(event.position().x(), event.position().y())
        last = len(self._points) - 1
        if i == 0:
            cx = 0.0  # first endpoint stays at x = 0
        elif i == last:
            cx = 1.0  # last endpoint stays at x = 1
        else:
            # Keep x strictly between neighbours so the curve stays a function.
            lo = self._points[i - 1][0] + 1e-3
            hi = self._points[i + 1][0] - 1e-3
            cx = min(max(cx, lo), hi)
        self._points[i] = (cx, cy)
        self.update()
        self.curve_changed.emit(self.lut())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_idx = None
