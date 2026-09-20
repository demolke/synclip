"""GNM head viewer/exporter"""

from __future__ import annotations

import os
import pathlib
import sys
import time
import traceback

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except Exception:  # pragma: no cover - GL-less boxes
    QOpenGLWidget = QWidget  # type: ignore
    GL = None
    OPENGL_AVAILABLE = False

import gnm_model as model


# ---------------------------------------------------------------------------
# Grid / axes shaders (verbatim from the original tool's main(); the head
# shader + debug-triangle shader live in gnm_head_model, since they're plain
# GLSL strings with no moderngl dependency and are shared as-is).
# ---------------------------------------------------------------------------

_GRID_VERT = """
#version 330
in vec3 in_position;
uniform mat4 view; uniform mat4 proj; uniform mat4 model;
void main(){ gl_Position = proj*view*model*vec4(in_position,1.0); }
"""
_GRID_FRAG = """
#version 330
out vec4 fragColor;
uniform vec3 color;
void main(){ fragColor = vec4(color,1.0); }
"""


# ---------------------------------------------------------------------------
# 3D viewport
# ---------------------------------------------------------------------------

class GNMViewport(QOpenGLWidget):
    """Renders the live GNM head. Camera/mouse behaviour matches
    synclip.ui.mesh_renderer.MeshRenderer exactly - no invert-mouse options.
    """

    camera_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if OPENGL_AVAILABLE:
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            self.setFormat(fmt)
        self.setMinimumSize(480, 360)

        self._verts = model.template_verts
        self._tris = model.triangles
        self.camera = model.Camera(self._verts)

        # Toggles (mirrors the old D / G / W keys + checkboxes).
        self.wireframe = False
        self.show_grid = True
        self.show_debug_tri = False

        # Mouse drag state - same conventions as MeshRenderer.
        self._drag_pos = None
        self._pan_drag_pos = None

        self._gpu_ready = False
        self._prog = 0
        self._prog_grid = 0
        self._prog_debug = 0
        self._vao = 0
        self._vbo_pos = 0
        self._vbo_norm = 0
        self._ibo = 0
        self._index_count = 0
        self._vao_grid = 0
        self._vbo_grid = 0
        self._grid_vert_count = 0
        self._vao_axes = 0
        self._vbo_axes = 0
        self._vao_debug = 0
        self._vbo_debug = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_verts(self, verts: np.ndarray) -> None:
        self._verts = verts
        if OPENGL_AVAILABLE and self._gpu_ready and self.isValid():
            self.makeCurrent()
            try:
                self._upload_verts()
            finally:
                self.doneCurrent()
        self.update()

    def reset_view(self) -> None:
        self.camera.yaw = 90.0
        self.camera.pitch = 10.0
        self.camera.focus(self._verts)
        self.camera_changed.emit()
        self.update()

    def focus_view(self) -> None:
        self.camera.focus(self._verts)
        self.camera_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Mouse / wheel interaction (copied from MeshRenderer - left-drag
    # orbit, middle-drag pan, wheel zoom, right-click context menu).
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position().toPoint()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_drag_pos = event.position().toPoint()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self._drag_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = pos - self._drag_pos
            self._drag_pos = pos
            self.camera.yaw += delta.x() * 0.35
            self.camera.pitch += delta.y() * 0.35
            self.camera.pitch = float(np.clip(self.camera.pitch, -89.0, 89.0))
            self.camera_changed.emit()
            self.update()
        if self._pan_drag_pos is not None and (event.buttons() & Qt.MouseButton.MiddleButton):
            delta = pos - self._pan_drag_pos
            self._pan_drag_pos = pos
            eye = self.camera.eye()
            fwd = self.camera.target - eye
            fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
            right = np.cross(fwd, np.array([0, 1, 0], dtype=np.float32))
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, fwd)
            factor = self.camera.distance * 0.002
            self.camera.target = self.camera.target - right * delta.x() * factor
            self.camera.target = self.camera.target + up * delta.y() * factor
            self.camera_changed.emit()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_drag_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        self.camera.distance = max(
            0.05, self.camera.distance - (delta / 120.0) * 0.08 * self.camera.distance
        )
        self.camera_changed.emit()
        self.update()

    def _show_context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        reset_act = QAction("Reset view", self)
        reset_act.triggered.connect(self.reset_view)
        menu.addAction(reset_act)
        menu.exec(global_pos)

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.12, 0.12, 0.14, 1.0)

        self._prog = self._link(model.VERT, model.FRAG)
        self._prog_grid = self._link(_GRID_VERT, _GRID_FRAG)
        self._prog_debug = self._link(model.VERT_DEBUG, model.FRAG_DEBUG)

        self._init_mesh()
        self._init_grid()
        self._init_debug_tri()
        self._gpu_ready = True

    def _link(self, vsrc: str, fsrc: str) -> int:
        def compile_shader(src, kind):
            sid = GL.glCreateShader(kind)
            GL.glShaderSource(sid, src)
            GL.glCompileShader(sid)
            if not GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sid).decode(errors="ignore"))
            return sid

        vs = compile_shader(vsrc, GL.GL_VERTEX_SHADER)
        fs = compile_shader(fsrc, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode(errors="ignore"))
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def _init_mesh(self) -> None:
        verts = self._verts.astype(np.float32)
        normals = model.compute_normals(verts, self._tris)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        pos_loc = GL.glGetAttribLocation(self._prog, "in_position")
        norm_loc = GL.glGetAttribLocation(self._prog, "in_normal")

        self._vbo_pos = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self._vbo_norm = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_norm)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(norm_loc)
        GL.glVertexAttribPointer(norm_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        idx = self._tris.astype(np.uint32)
        self._ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        self._index_count = idx.size

        GL.glBindVertexArray(0)

    def _upload_verts(self) -> None:
        verts = self._verts.astype(np.float32)
        normals = model.compute_normals(verts, self._tris)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_pos)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_norm)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, normals.nbytes, normals)

    def _init_grid(self) -> None:
        floor_y = float(model.template_verts[:, 1].min() - 0.05)
        grid_verts = []
        for i in range(-5, 6):
            grid_verts.extend([i, floor_y, -5, i, floor_y, 5, -5, floor_y, i, 5, floor_y, i])
        grid_verts = np.array(grid_verts, dtype=np.float32)
        self._grid_vert_count = grid_verts.shape[0] // 3

        c = model.template_verts.mean(axis=0)
        axes_verts = np.array([
            c[0], c[1], c[2], c[0] + 0.1, c[1], c[2],
            c[0], c[1], c[2], c[0], c[1] + 0.1, c[2],
            c[0], c[1], c[2], c[0], c[1], c[2] + 0.1,
        ], dtype=np.float32)

        pos_loc = GL.glGetAttribLocation(self._prog_grid, "in_position")

        self._vao_grid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao_grid)
        self._vbo_grid = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_grid)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, grid_verts.nbytes, grid_verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)

        self._vao_axes = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao_axes)
        self._vbo_axes = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_axes)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, axes_verts.nbytes, axes_verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)

    def _init_debug_tri(self) -> None:
        tri = np.array([-0.9, -0.9, 0.9, -0.9, 0.0, 0.9], dtype=np.float32)
        pos_loc = GL.glGetAttribLocation(self._prog_debug, "in_pos")
        self._vao_debug = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao_debug)
        self._vbo_debug = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_debug)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, tri.nbytes, tri, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Resize / paint
    # ------------------------------------------------------------------

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE or not self._gpu_ready:
            return
        w, h = max(1, self.width()), max(1, self.height())
        aspect = w / float(h)
        view = self.camera.view()
        proj = self.camera.proj(aspect)
        model_mat = np.eye(4, dtype=np.float32)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_CULL_FACE)

        if self.show_debug_tri:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glUseProgram(self._prog_debug)
            GL.glUniform3f(GL.glGetUniformLocation(self._prog_debug, "color"), 1.0, 0.0, 0.0)
            GL.glBindVertexArray(self._vao_debug)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glBindVertexArray(0)
            GL.glEnable(GL.GL_DEPTH_TEST)

        if self.show_grid:
            GL.glUseProgram(self._prog_grid)
            loc = GL.glGetUniformLocation
            GL.glUniformMatrix4fv(loc(self._prog_grid, "view"), 1, GL.GL_TRUE, view)
            GL.glUniformMatrix4fv(loc(self._prog_grid, "proj"), 1, GL.GL_TRUE, proj)
            GL.glUniformMatrix4fv(loc(self._prog_grid, "model"), 1, GL.GL_TRUE, model_mat)
            GL.glUniform3f(loc(self._prog_grid, "color"), 0.3, 0.3, 0.3)
            GL.glBindVertexArray(self._vao_grid)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_vert_count)
            GL.glBindVertexArray(0)
            GL.glUniform3f(loc(self._prog_grid, "color"), 1.0, 0.2, 0.2)
            GL.glBindVertexArray(self._vao_axes)
            GL.glDrawArrays(GL.GL_LINES, 0, 6)
            GL.glBindVertexArray(0)

        GL.glUseProgram(self._prog)
        loc = GL.glGetUniformLocation
        GL.glUniformMatrix4fv(loc(self._prog, "model"), 1, GL.GL_TRUE, model_mat)
        GL.glUniformMatrix4fv(loc(self._prog, "view"), 1, GL.GL_TRUE, view)
        GL.glUniformMatrix4fv(loc(self._prog, "proj"), 1, GL.GL_TRUE, proj)
        GL.glUniform3f(loc(self._prog, "light_pos"), 1.0, 1.0, 2.0)
        eye = self.camera.eye()
        GL.glUniform3f(loc(self._prog, "cam_pos"), float(eye[0]), float(eye[1]), float(eye[2]))
        GL.glUniform3f(loc(self._prog, "base_color"), 0.96, 0.78, 0.74)
        GL.glUniform1i(loc(self._prog, "wireframe"), 1 if self.wireframe else 0)

        GL.glBindVertexArray(self._vao)
        if self.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)
        if self.wireframe:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


class ViewportPlaceholder(QWidget):
    """Shown instead of GNMViewport when OpenGL is unavailable."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        lay = QVBoxLayout(self)
        lbl = QLabel("OpenGL is not available in this environment.", self)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #a88; font-size: 14px;")
        lay.addWidget(lbl)

    def set_verts(self, verts) -> None:
        pass

    def reset_view(self) -> None:
        pass

    def focus_view(self) -> None:
        pass

    camera_changed = Signal()
    wireframe = False
    show_grid = True
    show_debug_tri = False
    camera = None


# ---------------------------------------------------------------------------
# Small reusable widgets: a lazily-built collapsible section (imgui's
# collapsing_header, retained-mode) and a labelled float slider row.
# ---------------------------------------------------------------------------

class CollapsibleSection(QWidget):
    """A header button that reveals a body widget the first time it is
    expanded, built by *build_fn(body_layout)*. Mirrors imgui's
    ``collapsing_header`` but only pays the widget-construction cost once,
    and only for sections the user actually opens - important here since a
    couple of these sections hold hundreds of sliders.
    """

    def __init__(
        self,
        title: str,
        build_fn,
        parent: QWidget | None = None,
        start_expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._build_fn = build_fn
        self._built = False

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(start_expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if start_expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 4px; }"
        )
        self._toggle.clicked.connect(self._on_toggle)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(16, 2, 2, 6)
        self._body.setVisible(start_expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._body)

        if start_expanded:
            self._build_fn(self._body_layout)
            self._built = True

    def _on_toggle(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if checked and not self._built:
            self._build_fn(self._body_layout)
            self._built = True
        self._body.setVisible(checked)


_SLIDER_STEPS = 2000


def _add_float_slider(
    layout: QVBoxLayout,
    label_text: str,
    value: float,
    lo: float,
    hi: float,
    on_change,
    tooltip: str | None = None,
) -> QSlider:
    """Add one label + slider + value row to *layout*; returns the QSlider
    (float range [lo, hi] mapped onto an int slider, same convention as
    synclip.ui.modifier_stack._ParamEditor's float controls).
    """
    row = QWidget()
    row_lay = QHBoxLayout(row)
    row_lay.setContentsMargins(0, 0, 0, 0)
    row_lay.setSpacing(6)

    name_lbl = QLabel(label_text, row)
    name_lbl.setMinimumWidth(170)
    name_lbl.setStyleSheet("font-size: 12px;")
    if tooltip:
        name_lbl.setToolTip(tooltip)

    slider = QSlider(Qt.Orientation.Horizontal, row)
    slider.setRange(0, _SLIDER_STEPS)
    slider.setValue(int(round((value - lo) / (hi - lo) * _SLIDER_STEPS)))

    val_lbl = QLabel(f"{value:.3f}", row)
    val_lbl.setFixedWidth(48)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    val_lbl.setStyleSheet("font-size: 12px;")

    def _on_value(v: int) -> None:
        val = lo + (v / _SLIDER_STEPS) * (hi - lo)
        val_lbl.setText(f"{val:.3f}")
        on_change(val)

    slider.valueChanged.connect(_on_value)

    row_lay.addWidget(name_lbl)
    row_lay.addWidget(slider, stretch=1)
    row_lay.addWidget(val_lbl)
    layout.addWidget(row)
    return slider


def _sync_slider(slider: QSlider, lo: float, hi: float, value: float) -> None:
    slider.blockSignals(True)
    slider.setValue(int(round((value - lo) / (hi - lo) * _SLIDER_STEPS)))
    slider.blockSignals(False)


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GNMHeadViewerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "GNM Head Viewer - identity/expression/pose + glTF export "
            "(quads + UVs + FACS)"
        )
        self.resize(1600, 1000)

        # --- Parameters (same shapes/semantics as the original tool) ----
        self.identity = np.zeros(model.identity_dim, dtype=np.float32)
        self.expression = np.zeros(model.expression_dim, dtype=np.float32)
        self.rotations = np.zeros((model.num_joints, 3), dtype=np.float32)
        self.translation = np.zeros(3, dtype=np.float32)

        self._dirty = True
        self._last_verts = model.template_verts
        self._log_file = pathlib.Path(os.getcwd()) / "gnm_debug.log"
        self._last_log_text = "Press Log (or L) to log debug info"
        self._last_frame_t = time.perf_counter()

        # Slider registries, so Reset/Randomize/Preview can refresh whatever
        # sections happen to already be built.
        self._id_sliders: dict[int, QSlider] = {}
        self._ex_sliders: dict[int, QSlider] = {}
        self._pose_sliders: dict[tuple[int, int], QSlider] = {}
        self._trans_sliders: dict[int, QSlider] = {}

        self._dim_to_facs = self._build_dim_to_facs()

        # --- Viewport (central widget) -----------------------------------
        if OPENGL_AVAILABLE:
            self.viewport = GNMViewport(self)
        else:
            self.viewport = ViewportPlaceholder(self)
        self.viewport.camera_changed.connect(self._refresh_info_labels)
        self.setCentralWidget(self.viewport)

        # --- Right dock: everything the old imgui window held -----------
        self._build_side_panel()

        # --- Bottom dock: log -------------------------------------------
        self._build_log_dock()

        self._build_shortcuts()

        self._do_log()

        # Drive mesh regeneration + info refresh at ~60 Hz, same cadence
        # as the old glfw swap-interval-1 loop, but only recompute the mesh
        # when something actually changed (mesh_dirty).
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        print(
            "\n=== CONTROLS ===\n"
            "Left drag orbit, Middle drag pan, Scroll zoom, Right click for menu\n"
            "R reset view, F focus, L log\n"
        )

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def _build_dim_to_facs(self) -> dict[int, list[str]]:
        mapping: dict[int, list[str]] = {}
        try:
            for facs_name in model.ALL_FACS[:52]:  # ARKit-52 only, as before
                expr = model.facs_to_expr(facs_name, 1.0)
                nonzero = np.where(np.abs(expr) > 1e-6)[0]
                for d in nonzero:
                    mapping.setdefault(int(d), [])
                    if facs_name not in mapping[int(d)]:
                        mapping[int(d)].append(facs_name)
        except Exception:
            traceback.print_exc()
        return mapping

    def _mark_dirty(self) -> None:
        self._dirty = True

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _build_side_panel(self) -> None:
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        outer.addWidget(self._build_info_box())
        outer.addWidget(self._build_view_controls_box())
        outer.addWidget(self._build_reset_box())
        outer.addWidget(self._build_export_box())
        outer.addWidget(_hline())

        outer.addWidget(QLabel("<b>ALL EXPORTED PROPERTIES (full sliders with names)</b>"))

        outer.addWidget(CollapsibleSection(
            "Identity 253 (Head 0-169, Eyeball 170-172, Teeth 173-252)",
            self._build_identity_section,
        ))
        outer.addWidget(CollapsibleSection(
            "Expression 383 (Left Eye 0-99, Right Eye 100-199, "
            "Lower Face 200-349, Tongue 350-381, Iris 382)",
            self._build_expression_section,
        ))
        outer.addWidget(CollapsibleSection(
            "Pose 12 (4 joints x 3 axis-angle)",
            self._build_pose_section,
        ))
        outer.addWidget(CollapsibleSection(
            "Translation 3",
            self._build_translation_section,
        ))
        outer.addWidget(CollapsibleSection(
            f"Exported Blend Shapes Preview {len(model.ALL_FACS)} "
            "(exact GNM deltas)",
            self._build_facs_preview_section,
        ))
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        dock = QDockWidget("Parameters", self)
        dock.setWidget(scroll)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setMinimumWidth(420)

    def _build_info_box(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lbl_counts = QLabel()
        self._lbl_center = QLabel()
        self._lbl_camera = QLabel()
        self._lbl_frustum = QLabel()
        for lbl in (self._lbl_counts, self._lbl_center, self._lbl_camera,
                    self._lbl_frustum):
            lbl.setStyleSheet("font-size: 12px;")
            lay.addWidget(lbl)

        n_quads = model.quads.shape[0] if model.quads is not None else 0
        self._lbl_counts.setText(
            f"FULL RES: {model.template_verts.shape[0]} verts, "
            f"{model.triangles.shape[0]} tris, Quads {n_quads}"
        )
        self._refresh_info_labels()
        return box

    def _build_view_controls_box(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset view [R]")
        btn_reset.clicked.connect(self._on_reset_view)
        btn_focus = QPushButton("Focus [F]")
        btn_focus.clicked.connect(self._on_focus_view)
        btn_log = QPushButton("Log [L]")
        btn_log.clicked.connect(self._on_log_clicked)
        for b in (btn_reset, btn_focus, btn_log):
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        chk_row = QHBoxLayout()
        self.chk_debug_tri = QCheckBox("Debug tri [D]")
        self.chk_debug_tri.toggled.connect(self._on_debug_tri_toggled)
        self.chk_grid = QCheckBox("Grid [G]")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._on_grid_toggled)
        self.chk_wireframe = QCheckBox("Wireframe [W]")
        self.chk_wireframe.toggled.connect(self._on_wireframe_toggled)
        for c in (self.chk_debug_tri, self.chk_grid, self.chk_wireframe):
            chk_row.addWidget(c)
        lay.addLayout(chk_row)
        return box

    def _build_reset_box(self) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        btn_reset_params = QPushButton("Reset Params")
        btn_reset_params.clicked.connect(self._on_reset_params)
        btn_randomize = QPushButton("Randomize")
        btn_randomize.clicked.connect(self._on_randomize)
        lay.addWidget(btn_reset_params)
        lay.addWidget(btn_randomize)
        return box

    def _build_export_box(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("glTF Export - shared verts + exact GNM deltas"))

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Path"))
        self.gltf_path_edit = QLineEdit(
            os.path.join(os.getcwd(), "gnm_head_quads_uvs_facs.gltf")
        )
        path_row.addWidget(self.gltf_path_edit, stretch=1)
        lay.addLayout(path_row)

        lay.addWidget(QLabel(
            f"Will export {len(model.ALL_FACS)} blend shapes: "
            "52 ARKit + 33 FACS + 21 semantic"
        ))

        btn_export_full = QPushButton(
            "EXPORT glTF - ALL FACS + QUADS + UVs (full pose)"
        )
        btn_export_full.clicked.connect(self._on_export_full)
        lay.addWidget(btn_export_full)

        btn_export_base = QPushButton("EXPORT glTF - BASE ONLY (shared verts)")
        btn_export_base.clicked.connect(self._on_export_base)
        lay.addWidget(btn_export_base)

        return box

    # -- lazily-built sections ------------------------------------------

    def _build_identity_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel(
            "Identity controls head shape. Each dim is a PCA component."
        ))
        groups = [
            ("Head 0-169 (face shape)", 0, 170),
            ("Eyeball 170-172 (eyeball shape)", 170, 173),
            ("Teeth 173-252 (teeth shape)", 173, 253),
        ]
        for group_name, s, e in groups:
            def build_group(sub_layout: QVBoxLayout, s=s, e=e) -> None:
                for i in range(s, e):
                    slider = _add_float_slider(
                        sub_layout, f"ID {i}", float(self.identity[i]),
                        -3.0, 3.0,
                        lambda val, i=i: self._on_identity_changed(i, val),
                    )
                    self._id_sliders[i] = slider
            layout.addWidget(CollapsibleSection(group_name, build_group))

    def _build_expression_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel(
            "Expression controls FACS. Shows which ARKit blend shapes "
            "affect each dim."
        ))
        groups = [
            ("Left Eye 0-99 (brows, eyelids, gaze)", 0, 100),
            ("Right Eye 100-199 (brows, eyelids, gaze)", 100, 200),
            ("Lower Face 200-349 (jaw, mouth, cheeks, nose)", 200, 350),
            ("Tongue 350-381", 350, 382),
            ("Iris 382 (iris size)", 382, 383),
        ]
        for group_name, s, e in groups:
            def build_group(sub_layout: QVBoxLayout, s=s, e=e, group_name=group_name) -> None:
                for i in range(s, e):
                    facs_list = self._dim_to_facs.get(i, [])
                    if facs_list:
                        label = f"EX {i} [{','.join(facs_list[:2])}]"
                        tooltip = f"Dim {i} affects: {', '.join(facs_list)}"
                    else:
                        words = group_name.split()
                        label = f"EX {i} [{words[0]} {words[1] if len(words) > 1 else ''}]"
                        tooltip = None
                    slider = _add_float_slider(
                        sub_layout, label, float(self.expression[i]),
                        -3.0, 3.0,
                        lambda val, i=i: self._on_expression_changed(i, val),
                        tooltip=tooltip,
                    )
                    self._ex_sliders[i] = slider
            layout.addWidget(CollapsibleSection(group_name, build_group))

    def _build_pose_section(self, layout: QVBoxLayout) -> None:
        joint_names = ["neck", "jaw", "left_eye", "right_eye"]
        for j in range(model.num_joints):
            name = joint_names[j] if j < len(joint_names) else f"joint{j}"

            def build_joint(sub_layout: QVBoxLayout, j=j, name=name) -> None:
                for axis, axis_label in enumerate(["X", "Y", "Z"]):
                    slider = _add_float_slider(
                        sub_layout, f"{name} {axis_label}",
                        float(self.rotations[j, axis]), -1.5, 1.5,
                        lambda val, j=j, axis=axis: self._on_pose_changed(j, axis, val),
                    )
                    self._pose_sliders[(j, axis)] = slider
            layout.addWidget(CollapsibleSection(name, build_joint))

    def _build_translation_section(self, layout: QVBoxLayout) -> None:
        for i, axis_label in enumerate(["TX", "TY", "TZ"]):
            slider = _add_float_slider(
                layout, axis_label, float(self.translation[i]), -0.3, 0.3,
                lambda val, i=i: self._on_translation_changed(i, val),
            )
            self._trans_sliders[i] = slider

    def _build_facs_preview_section(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel(
            "Preview each FACS - this is what glTF will export as delta"
        ))
        grid = QGridLayout()
        grid.setSpacing(4)
        cols = 3
        for idx, name in enumerate(model.ALL_FACS):
            btn = QPushButton(name)
            btn.setToolTip(f"Preview {name}")
            btn.clicked.connect(lambda _checked=False, name=name: self._on_preview_facs(name))
            grid.addWidget(btn, idx // cols, idx % cols)
        layout.addLayout(grid)

        btn_neutral = QPushButton("Reset to Neutral")
        btn_neutral.clicked.connect(self._on_reset_to_neutral)
        layout.addWidget(btn_neutral)

    # ------------------------------------------------------------------
    # Log dock
    # ------------------------------------------------------------------

    def _build_log_dock(self) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlainText(self._last_log_text)
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log_view)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setMinimumHeight(140)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence("R"), self, activated=self._on_reset_view)
        QShortcut(QKeySequence("F"), self, activated=self._on_focus_view)
        QShortcut(QKeySequence("L"), self, activated=self._on_log_clicked)
        QShortcut(QKeySequence("D"), self, activated=self.chk_debug_tri.toggle)
        QShortcut(QKeySequence("W"), self, activated=self.chk_wireframe.toggle)
        QShortcut(QKeySequence("G"), self, activated=self.chk_grid.toggle)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def _on_identity_changed(self, i: int, val: float) -> None:
        self.identity[i] = val
        self._mark_dirty()

    def _on_expression_changed(self, i: int, val: float) -> None:
        self.expression[i] = val
        self._mark_dirty()

    def _on_pose_changed(self, j: int, axis: int, val: float) -> None:
        self.rotations[j, axis] = val
        self._mark_dirty()

    def _on_translation_changed(self, i: int, val: float) -> None:
        self.translation[i] = val
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_reset_view(self) -> None:
        self.viewport.reset_view()
        self._do_log()

    def _on_focus_view(self) -> None:
        self.viewport.focus_view()
        self._do_log()

    def _on_log_clicked(self) -> None:
        self._do_log()

    def _on_debug_tri_toggled(self, checked: bool) -> None:
        self.viewport.show_debug_tri = checked
        self.viewport.update()

    def _on_grid_toggled(self, checked: bool) -> None:
        self.viewport.show_grid = checked
        self.viewport.update()

    def _on_wireframe_toggled(self, checked: bool) -> None:
        self.viewport.wireframe = checked
        self.viewport.update()

    def _on_reset_params(self) -> None:
        self.identity[:] = 0
        self.expression[:] = 0
        self.rotations[:] = 0
        self.translation[:] = 0
        self._sync_all_sliders()
        self._mark_dirty()

    def _on_randomize(self) -> None:
        self.identity[:] = (np.random.randn(model.identity_dim) * 0.8).astype(np.float32)
        self._sync_identity_sliders()
        self._mark_dirty()

    def _on_preview_facs(self, name: str) -> None:
        expr, rots, trans = model.facs_to_full_params(name, 1.0)
        self.expression[:] = 0
        self.rotations[:] = 0
        self.translation[:] = 0
        self.expression[:] = expr
        self.rotations[:] = rots
        self.translation[:] = trans
        self._sync_expression_sliders()
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._mark_dirty()

    def _on_reset_to_neutral(self) -> None:
        self.expression[:] = 0
        self.rotations[:] = 0
        self.translation[:] = 0
        self._sync_expression_sliders()
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._mark_dirty()

    def _on_export_full(self) -> None:
        try:
            base_v = model.gen_verts(
                self.identity, np.zeros(model.expression_dim, dtype=np.float32),
                self.rotations, self.translation,
            )
            facs_full = [model.facs_to_full_params(name, 1.0) for name in model.ALL_FACS]
            out_gltf, out_bin = model.build_gltf_with_quads_and_facs(
                base_verts=base_v, base_tris=model.triangles, base_quads=model.quads,
                quad_uvs_arr=model.quad_uvs, tri_uvs_arr=model.tri_uvs,
                facs_names=model.ALL_FACS, facs_full_params=facs_full,
                identity=self.identity, rotations=self.rotations,
                translation=self.translation, out_path=self.gltf_path_edit.text(),
            )
            self._set_log_text(
                f"Exported glTF {out_gltf}\nBin {out_bin}\n"
                f"{len(model.ALL_FACS)} blend shapes, shared verts"
            )
        except Exception as e:
            traceback.print_exc()
            self._set_log_text(f"glTF export failed: {e}")

    def _on_export_base(self) -> None:
        try:
            base_v = model.gen_verts(
                self.identity, np.zeros(model.expression_dim, dtype=np.float32),
                self.rotations, self.translation,
            )
            out_gltf, out_bin = model.build_gltf_with_quads_and_facs(
                base_verts=base_v, base_tris=model.triangles, base_quads=model.quads,
                quad_uvs_arr=model.quad_uvs, tri_uvs_arr=model.tri_uvs,
                facs_names=[], facs_exprs=[], identity=self.identity,
                rotations=self.rotations, translation=self.translation,
                out_path=self.gltf_path_edit.text(),
            )
            self._set_log_text(f"Exported BASE glTF {out_gltf}\nBin {out_bin}")
        except Exception as e:
            traceback.print_exc()
            self._set_log_text(f"BASE export failed: {e}")

    # ------------------------------------------------------------------
    # Slider sync helpers (numpy arrays -> already-built Qt widgets)
    # ------------------------------------------------------------------

    def _sync_identity_sliders(self) -> None:
        for i, slider in self._id_sliders.items():
            _sync_slider(slider, -3.0, 3.0, float(self.identity[i]))

    def _sync_expression_sliders(self) -> None:
        for i, slider in self._ex_sliders.items():
            _sync_slider(slider, -3.0, 3.0, float(self.expression[i]))

    def _sync_pose_sliders(self) -> None:
        for (j, axis), slider in self._pose_sliders.items():
            _sync_slider(slider, -1.5, 1.5, float(self.rotations[j, axis]))

    def _sync_translation_sliders(self) -> None:
        for i, slider in self._trans_sliders.items():
            _sync_slider(slider, -0.3, 0.3, float(self.translation[i]))

    def _sync_all_sliders(self) -> None:
        self._sync_identity_sliders()
        self._sync_expression_sliders()
        self._sync_pose_sliders()
        self._sync_translation_sliders()

    # ------------------------------------------------------------------
    # Per-tick update + info panel
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_t
        self._last_frame_t = now

        if self._dirty:
            try:
                verts = model.gen_verts(
                    self.identity, self.expression, self.rotations, self.translation
                )
                self._last_verts = verts
                self.viewport.set_verts(verts)
            except Exception:
                traceback.print_exc()
            self._dirty = False
        else:
            self.viewport.update()

        self._refresh_info_labels()

    def _refresh_info_labels(self) -> None:
        bmin, bmax, center, size, diag = model.get_bounds(self._last_verts)
        self._lbl_center.setText(f"Head center {center} diag {diag:.4f}")

        cam = self.viewport.camera
        if cam is None:
            return
        eye = cam.eye()
        self._lbl_camera.setText(
            f"Cam eye {eye} target {cam.target} dist {cam.distance:.3f} "
            f"yaw {cam.yaw:.1f} pitch {cam.pitch:.1f}"
        )

        try:
            w = max(1, self.viewport.width())
            h = max(1, self.viewport.height())
            aspect = w / float(h)
            view = cam.view()
            proj = cam.proj(aspect)
            model_mat = np.eye(4, dtype=np.float32)
            frustum = model.frustum_check(self._last_verts, model_mat, view, proj)
            in_frustum = (
                frustum["num_inside"] > 0
                or (
                    (np.abs(frustum["center_ndc"]) <= 1).all()
                    and frustum["center_clip"][3] > 0
                )
            )
            colour = "#33cc33" if in_frustum else "#cc3333"
            self._lbl_frustum.setText(
                f"Frustum: {frustum['num_inside']}/8 inside, "
                f"Center NDC {frustum['center_ndc']} - "
                f"<span style='color:{colour}'>"
                f"{'IN FRUSTUM' if in_frustum else 'NOT IN FRUSTUM'}</span>"
            )
            self._lbl_frustum.setTextFormat(Qt.TextFormat.RichText)
        except Exception as e:
            self._lbl_frustum.setText(f"Frustum err {e}")

    def _set_log_text(self, text: str) -> None:
        self._last_log_text = text
        self.log_view.setPlainText(text[-4000:])

    def _do_log(self):
        cam = self.viewport.camera
        if cam is None:
            return None
        w = max(1, self.viewport.width())
        h = max(1, self.viewport.height())
        aspect = w / float(h)
        view = cam.view()
        proj = cam.proj(aspect)
        model_mat = np.eye(4, dtype=np.float32)
        v = model.gen_verts(self.identity, self.expression, self.rotations, self.translation)
        text, frustum = model.log_debug_info(cam, v, model_mat, view, proj, log_file=self._log_file)
        self._set_log_text(text)
        return text, frustum


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("GNM Head Viewer")
    app.setOrganizationName("minigltf")

    # Reuse synclip's own dark Fusion palette so this tool looks like the
    # rest of the app rather than the OS default.
    try:
        from ..main import _build_dark_palette
        app.setPalette(_build_dark_palette())
        app.setStyle("Fusion")
    except Exception:
        pass

    window = GNMHeadViewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Allow ``python gnm_head_viewer.py`` as well as
    # ``python -m synclip.tools.gnm_head_viewer``.
    if __package__ in (None, ""):
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _synclip_dir = os.path.dirname(_pkg_dir)
        sys.path.insert(0, os.path.dirname(_synclip_dir))
        __package__ = f"{os.path.basename(_synclip_dir)}.tools"
    main()
