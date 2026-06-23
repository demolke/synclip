"""
ValueEditor - editable per-channel blendshape value sliders for REVIEW mode.

Shows all 52 ARKit channels as labelled 0..1 sliders, grouped by facial region
and laid out in several columns so the long list is easy to scan. While a take
plays back the sliders follow the interpolated values; dragging one edits the
actual value at the current playback position (best done while paused), which
MainWindow writes back into the take's frames.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..arkit_names import BLENDSHAPE_NAMES

# Topic groups (by channel-name prefix). Order defines display order.
_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Brows", ("brow",)),
    ("Eyes", ("eye",)),
    ("Jaw", ("jaw",)),
    ("Mouth", ("mouth",)),
    ("Cheeks / Nose", ("cheek", "nose")),
]

# How the groups are distributed across columns (so no single column is huge).
_COLUMNS: list[list[str]] = [
    ["Brows", "Jaw", "Cheeks / Nose"],
    ["Eyes"],
    ["Mouth"],
]

_LABEL_STYLE = "font-size: 13px;"
_HEADER_STYLE = "font-size: 15px; font-weight: bold; color: #cda;"

# Read-only "meter" look: a thin groove with a coloured fill and no handle, so
# the live value display doesn't masquerade as an interactive slider.
_METER_STYLE = """
QSlider::groove:horizontal { height: 6px; background: #2c2c30; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #6aa0d8; border-radius: 3px; }
QSlider::add-page:horizontal { background: #2c2c30; border-radius: 3px; }
QSlider::handle:horizontal { width: 0px; height: 0px; margin: 0; background: transparent; }
"""


class _ValueSlider(QSlider):
    """Slider that ignores the mouse wheel (so scrolling the panel doesn't
    change values), and optionally ignores all user interaction when read-only
    (used for the live display view)."""

    def __init__(self, *args, read_only: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._read_only = read_only

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        event.ignore()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._read_only:
            event.ignore()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._read_only:
            event.ignore()
        else:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._read_only:
            event.ignore()
        else:
            super().keyPressEvent(event)


def _group_for(name: str) -> str:
    for title, prefixes in _GROUPS:
        if name.startswith(prefixes):
            return title
    return "Mouth"


class ValueEditor(QWidget):
    """Grid of editable per-channel value sliders, grouped by region."""

    value_changed = Signal(int, float)  # (channel index, value) - user edits only

    def __init__(self, parent: QWidget | None = None, read_only: bool = False) -> None:
        super().__init__(parent)
        self._read_only = read_only
        self._sliders: dict[int, QSlider] = {}
        self._value_labels: dict[int, QLabel] = {}

        # Channel indices per group (skip index 0 = _neutral).
        grouped: dict[str, list[int]] = {title: [] for title, _ in _GROUPS}
        for i, name in enumerate(BLENDSHAPE_NAMES):
            if i == 0:
                continue
            grouped[_group_for(name)].append(i)

        columns = QHBoxLayout(self)
        columns.setContentsMargins(8, 4, 8, 4)
        columns.setSpacing(18)

        for col_titles in _COLUMNS:
            col = QVBoxLayout()
            col.setSpacing(6)
            for title in col_titles:
                col.addWidget(self._build_group(title, grouped[title]))
            col.addStretch(1)
            columns.addLayout(col)

    # ------------------------------------------------------------------

    def _build_group(self, title: str, indices: list[int]) -> QWidget:
        box = QWidget(self)
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(3)
        grid.setHorizontalSpacing(6)

        header = QLabel(title, box)
        header.setStyleSheet(_HEADER_STYLE)
        grid.addWidget(header, 0, 0, 1, 3)

        for row, idx in enumerate(indices, start=1):
            name_lbl = QLabel(BLENDSHAPE_NAMES[idx], box)
            name_lbl.setStyleSheet(_LABEL_STYLE)
            name_lbl.setMinimumWidth(150)

            slider = _ValueSlider(
                Qt.Orientation.Horizontal, box, read_only=self._read_only
            )
            slider.setRange(0, 100)   # 0.00 .. 1.00
            slider.setValue(0)
            slider.setMinimumWidth(110)
            if self._read_only:
                # Style as a read-only meter (filled groove, no grab handle) so
                # it doesn't imply the live values are draggable.
                slider.setStyleSheet(_METER_STYLE)
            else:
                slider.valueChanged.connect(
                    lambda v, i=idx: self._on_slider(i, v)
                )

            val_lbl = QLabel("0.00", box)
            val_lbl.setStyleSheet(_LABEL_STYLE)
            val_lbl.setFixedWidth(40)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

            grid.addWidget(name_lbl, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val_lbl, row, 2)

            self._sliders[idx] = slider
            self._value_labels[idx] = val_lbl

        return box

    def _on_slider(self, idx: int, value: int) -> None:
        v = value / 100.0
        self._value_labels[idx].setText(f"{v:.2f}")
        self.value_changed.emit(idx, v)

    # ------------------------------------------------------------------

    def set_read_only(self, read_only: bool) -> None:
        """Dynamically switch between editable and read-only (meter) mode."""
        if read_only == self._read_only:
            return
        self._read_only = read_only
        for idx, slider in self._sliders.items():
            slider._read_only = read_only
            if read_only:
                slider.setStyleSheet(_METER_STYLE)
                try:
                    slider.valueChanged.disconnect()
                except RuntimeError:
                    pass
            else:
                slider.setStyleSheet("")
                slider.valueChanged.connect(
                    lambda v, i=idx: self._on_slider(i, v)
                )

    def values(self) -> list[float]:
        """Return current slider values as a 52-element list."""
        out = [0.0] * 52
        for idx, slider in self._sliders.items():
            out[idx] = slider.value() / 100.0
        return out

    def set_values(self, values: list[float]) -> None:
        """Reflect *values* (length 52) without emitting value_changed."""
        for idx, slider in self._sliders.items():
            v = values[idx] if idx < len(values) else 0.0
            slider.blockSignals(True)
            slider.setValue(int(round(v * 100)))
            slider.blockSignals(False)
            self._value_labels[idx].setText(f"{v:.2f}")
