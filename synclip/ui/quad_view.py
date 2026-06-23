"""
QuadView - the 2x2 preview grid.

    +-----------------+-----------------+
    |  video (webcam) |  view 0         |
    +-----------------+-----------------+
    |  view 1         |  view 2         |
    +-----------------+-----------------+

Top-left hosts the existing WebcamView (the real camera/video picture). The
other three cells are MeshRenderers, each driven by its own per-view pipeline
(see view_pipeline.ViewPipeline). Clicking a mesh cell selects it and emits
``view_selected`` so the right-dock panel can bind to that view's settings.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .mesh_renderer import MeshRenderer, MeshPlaceholder, OPENGL_AVAILABLE

MESH_VIEW_COUNT = 3


class _Cell(QFrame):
    """A titled container around one preview widget."""

    def __init__(self, title: str, body: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        header = QWidget(self)
        header.setStyleSheet("background:#222;")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(6, 2, 6, 2)
        hlay.setSpacing(4)
        self._name_label = QLabel(title, header)
        self._name_label.setStyleSheet("color:#bbb;")
        self._meta_label = QLabel("", header)
        self._meta_label.setStyleSheet("color:#888; font-size:10px;")
        self._meta_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hlay.addWidget(self._name_label)
        hlay.addStretch(1)
        hlay.addWidget(self._meta_label)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)
        lay.addWidget(header)
        lay.addWidget(body, stretch=1)
        self._selected = False
        self._refresh_style()

    def set_title(self, name: str, meta: str = "") -> None:
        self._name_label.setText(name)
        self._meta_label.setText(meta)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh_style()

    def _refresh_style(self) -> None:
        self.setStyleSheet(
            "QFrame { border: 2px solid #3ab; }" if self._selected
            else "QFrame { border: 1px solid #333; }"
        )


class QuadView(QWidget):
    """2x2 grid: webcam + 3 mesh preview views."""

    view_selected = Signal(int)  # index 0..MESH_VIEW_COUNT-1

    def __init__(self, webcam_view: QWidget, labels: list[str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        self._webcam_view = webcam_view
        grid.addWidget(_Cell("Video", webcam_view, self), 0, 0)

        self._renderers: list[QWidget] = []
        self._cells: list[_Cell] = []
        positions = [(0, 1), (1, 0), (1, 1)]
        for i in range(MESH_VIEW_COUNT):
            if OPENGL_AVAILABLE:
                r = MeshRenderer(labels[i], self)
            else:
                r = MeshPlaceholder("3D preview\n(OpenGL unavailable)", self)
            r.clicked.connect(lambda idx=i: self._on_cell_clicked(idx))
            cell = _Cell(labels[i], r, self)
            self._renderers.append(r)
            self._cells.append(cell)
            grid.addWidget(cell, *positions[i])

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self._selected = 0
        self._cells[0].set_selected(True)
        self._renderers[0].set_selected(True)

    # ------------------------------------------------------------------

    def _on_cell_clicked(self, index: int) -> None:
        self.select(index)
        self.view_selected.emit(index)

    def select(self, index: int) -> None:
        self._selected = index
        for i, (cell, r) in enumerate(zip(self._cells, self._renderers)):
            cell.set_selected(i == index)
            r.set_selected(i == index)

    def selected_index(self) -> int:
        return self._selected

    def set_weights(self, index: int, weights) -> None:
        self._renderers[index].set_weights(weights)

    def set_head_pose(self, index: int, pose: dict | None) -> None:
        if hasattr(self._renderers[index], "set_head_pose"):
            self._renderers[index].set_head_pose(pose)

    def set_neck_anchor(self, index: int, value: float) -> None:
        if hasattr(self._renderers[index], "set_neck_anchor"):
            self._renderers[index].set_neck_anchor(value)

    def set_mesh(self, index: int, mesh) -> None:
        self._renderers[index].set_mesh(mesh)

    def set_title(self, index: int, name: str, meta: str = "") -> None:
        self._cells[index].set_title(name, meta)

    def renderer(self, index: int) -> QWidget:
        return self._renderers[index]

    def get_camera(self, index: int) -> dict:
        r = self._renderers[index]
        if hasattr(r, "get_camera"):
            return r.get_camera()
        return {}

    def set_camera(self, index: int, state: dict) -> None:
        r = self._renderers[index]
        if hasattr(r, "set_camera"):
            r.set_camera(state)
