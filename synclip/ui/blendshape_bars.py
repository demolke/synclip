"""
BlendshapeBars - custom QPainter widget showing 52 horizontal bars.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..arkit_names import BLENDSHAPE_NAMES

_NUM_SHAPES = 52
assert len(BLENDSHAPE_NAMES) == _NUM_SHAPES

_ROW_H = 12       # bar height in pixels
_ROW_GAP = 2      # vertical gap between bars
_ROW_STRIDE = _ROW_H + _ROW_GAP   # 14 px per row

_NAME_W = 120     # left label area
_GAP1 = 4         # gap between name and bar
_BAR_W = 200      # bar area width
_GAP2 = 4         # gap between bar and number
_NUM_W = 40       # numeric value area

_TOTAL_W = _NAME_W + _GAP1 + _BAR_W + _GAP2 + _NUM_W
_TOTAL_H = _NUM_SHAPES * _ROW_STRIDE

_COL_BG_BAR   = QColor(40, 40, 40)
_COL_DIM      = QColor(70, 70, 70)
_COL_BRIGHT   = QColor(60, 200, 60)
_COL_TEXT     = QColor(200, 200, 200)
_COL_NUM      = QColor(180, 220, 180)

_THRESHOLD = 0.05


class BlendshapeBars(QWidget):
    """52 horizontal bars drawn with QPainter (not QProgressBar widgets)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: list[float] = [0.0] * _NUM_SHAPES
        self.setMinimumSize(self.sizeHint())

        self._font = QFont("Monospace", 7)
        self._font.setStyleHint(QFont.StyleHint.TypeWriter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_values(self, values: list[float]) -> None:
        if len(values) != _NUM_SHAPES:
            return
        self._values = list(values)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(_TOTAL_W, _TOTAL_H)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setFont(self._font)

        fm = painter.fontMetrics()
        text_h = fm.height()
        text_baseline_offset = (_ROW_H + text_h) // 2 - fm.descent()

        for i, (name, val) in enumerate(zip(BLENDSHAPE_NAMES, self._values)):
            y_top = i * _ROW_STRIDE

            # --- Name label (right-aligned) ---
            painter.setPen(QPen(_COL_TEXT))
            name_rect_x = 0
            name_rect_w = _NAME_W
            text_y = y_top + text_baseline_offset
            painter.drawText(
                name_rect_x,
                y_top,
                name_rect_w,
                _ROW_H,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                name,
            )

            # --- Bar background ---
            bar_x = _NAME_W + _GAP1
            painter.fillRect(bar_x, y_top, _BAR_W, _ROW_H, _COL_BG_BAR)

            # --- Filled portion ---
            clamped = max(0.0, min(1.0, val))
            fill_w = int(clamped * _BAR_W)
            if fill_w > 0:
                colour = _COL_BRIGHT if val >= _THRESHOLD else _COL_DIM
                painter.fillRect(bar_x, y_top, fill_w, _ROW_H, colour)

            # --- Numeric value ---
            num_x = bar_x + _BAR_W + _GAP2
            painter.setPen(QPen(_COL_NUM))
            painter.drawText(
                num_x,
                y_top,
                _NUM_W,
                _ROW_H,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{val:.2f}",
            )

        painter.end()
