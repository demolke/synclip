"""
CurveEditor - a small freeform curve widget with draggable control points.

Two roles, selected by ``eval_mode``:

* ``"lut"`` (default) - a **response curve**: maps an input weight (x, 0..1) to
  an output weight (y, 0..1), baked into a monotone lookup table the capture
  filter samples per frame (e.g. boost low values while letting large ones fall
  off toward 1.0).
* ``"animation"`` - a **timeline curve**: x is normalised clip position (0..1),
  y is a free value within ``y_range`` sampled piecewise-linearly (non-monotone,
  unclamped). Used for a modifier's time-varying influence. A playhead marker can
  be drawn at the current scrub position.

Interaction:
    left-drag a point   move it (endpoints keep x at 0 / 1)
    left-click empty     add a new point there
    right-click a point  remove it (endpoints can't be removed)
    right-click empty     menu -> reset the curve to its default
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget

# Curve baking is pure math and shared with the headless modifier pipeline, so it
# lives in the engine layer (curve_lut) to keep that layer Qt-free.
from ..curve_lut import build_lut, clamp01 as _clamp01, sample_curve

_HIT_RADIUS = 12  # px


class CurveEditor(QWidget):
    """Draggable curve editor. Emits the baked LUT (lut mode) or the raw control
    points (animation mode) on every change."""

    # Payload is the baked LUT (lut mode) or the list of [x, y] points (animation
    # mode). Existing response-curve consumers read .points()/.lut() directly and
    # only use this as a change trigger, so the payload type can vary by mode.
    curve_changed = Signal(list)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        y_range: tuple[float, float] = (0.0, 1.0),
        eval_mode: str = "lut",
        reference: str | float | None = "identity",
        default_points: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 120)
        self._ymin, self._ymax = float(y_range[0]), float(y_range[1])
        self._eval_mode = eval_mode
        # Reference line: "identity" diagonal, a horizontal line at a y value, or
        # None. Default depends on the role.
        self._reference = reference
        self._default_points: list[tuple[float, float]] = list(
            default_points if default_points is not None else [(0.0, 0.0), (1.0, 1.0)]
        )
        self._points: list[tuple[float, float]] = list(self._default_points)
        self._drag_idx: int | None = None
        self._playhead: float | None = None
        self.setMouseTracking(True)

    # ------------------------------------------------------------------

    def lut(self) -> list[float]:
        return build_lut(self._points)

    def points(self) -> list[list[float]]:
        """Control points as JSON-friendly [x, y] pairs."""
        return [[x, y] for (x, y) in self._points]

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = sorted((float(x), float(y)) for x, y in points)
        self.update()
        self._emit()

    def reset_to_default(self) -> None:
        """Restore the configured default curve (identity for response curves,
        the neutral flat line for an animation curve)."""
        self.set_points(list(self._default_points))

    # Back-compat alias (response-curve callers / context menu wording).
    reset_to_identity = reset_to_default

    def set_playhead(self, t: float | None) -> None:
        """Draw a vertical marker at normalised position *t* (0..1), or clear it
        with None. Cheap repaint only."""
        self._playhead = t
        self.update()

    def _emit(self) -> None:
        self.curve_changed.emit(
            self.points() if self._eval_mode == "animation" else self.lut()
        )

    # ------------------------------------------------------------------
    # Coordinate mapping (curve space <-> widget pixels)
    # ------------------------------------------------------------------

    def _margin(self) -> int:
        return 10

    def _norm_y(self, y: float) -> float:
        span = self._ymax - self._ymin
        return (y - self._ymin) / span if span else 0.0

    def _to_px(self, x: float, y: float) -> QPointF:
        m = self._margin()
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        return QPointF(m + x * w, m + (1.0 - self._norm_y(y)) * h)

    def _to_curve(self, px: float, py: float) -> tuple[float, float]:
        m = self._margin()
        w = self.width() - 2 * m
        h = self.height() - 2 * m
        x = (px - m) / w if w else 0.0
        ny = 1.0 - (py - m) / h if h else 0.0
        y = self._ymin + ny * (self._ymax - self._ymin)
        y = min(max(y, self._ymin), self._ymax)
        return _clamp01(x), y

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 34))

        # Grid (quarter divisions in normalised space).
        p.setPen(QPen(QColor(60, 60, 66), 1))
        for f in (0.25, 0.5, 0.75):
            yv = self._ymin + f * (self._ymax - self._ymin)
            p.drawLine(self._to_px(f, self._ymin), self._to_px(f, self._ymax))
            p.drawLine(self._to_px(0.0, yv), self._to_px(1.0, yv))

        # Reference / neutral line.
        p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DashLine))
        if self._reference == "identity":
            p.drawLine(self._to_px(0.0, 0.0), self._to_px(1.0, 1.0))
        elif isinstance(self._reference, (int, float)):
            yv = float(self._reference)
            p.drawLine(self._to_px(0.0, yv), self._to_px(1.0, yv))

        # The curve itself.
        p.setPen(QPen(QColor(120, 200, 120), 2))
        if self._eval_mode == "animation":
            # Piecewise-linear straight through the control points (matches
            # sample_curve), so no monotone-LUT assumptions.
            pts = sorted(self._points)
            prev = None
            for (x, y) in pts:
                pt = self._to_px(x, y)
                if prev is not None:
                    p.drawLine(prev, pt)
                prev = pt
        else:
            lut = self.lut()
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

        # Playhead marker (animation mode).
        if self._playhead is not None:
            x = _clamp01(self._playhead)
            p.setPen(QPen(QColor(230, 120, 120), 1))
            p.drawLine(self._to_px(x, self._ymin), self._to_px(x, self._ymax))

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
                self._emit()
            elif idx is None:
                # On empty space: offer a reset so accidental points can be
                # cleared without removing them one by one.
                menu = QMenu(self)
                act_reset = menu.addAction("Reset curve")
                if menu.exec(event.globalPosition().toPoint()) == act_reset:
                    self.reset_to_default()
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
        self._emit()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_idx = None
