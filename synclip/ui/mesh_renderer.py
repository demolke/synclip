"""
MeshRenderer - a QOpenGLWidget that draws a morph-target head on the GPU.

The base mesh and all 52 ARKit morph-offset arrays are uploaded to the GPU
*once* (offsets live in a float texture). Each frame we only push the 52 morph
weights as a uniform and a vertex shader does
``pos = base + sum_i(weight_i * offset_i)`` -- so animating the head costs one
52-float upload, not a per-vertex CPU pass per view.

There is no CPU blending fallback: morphing happens entirely in the vertex
shader. If PyOpenGL isn't importable at all the module still imports with
``OPENGL_AVAILABLE`` False so callers can show a placeholder instead.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

try:
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except Exception:  # pragma: no cover - GL-less boxes
    QOpenGLWidget = QWidget  # type: ignore
    GL = None
    OPENGL_AVAILABLE = False

_MAX_MORPHS = 52

_VERT_SHADER = """
#version 330
layout(location = 0) in vec3 aBase;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat3 uNormalMat;
uniform float uWeights[52];
uniform int uVertexCount;
uniform int uTexWidth;
uniform sampler2D uOffsets;   // morph position offsets, morph-major

out vec3 vNormal;

vec3 fetchOffset(int morph, int vid) {
    int linear = morph * uVertexCount + vid;
    int tx = linear % uTexWidth;
    int ty = linear / uTexWidth;
    return texelFetch(uOffsets, ivec2(tx, ty), 0).rgb;
}

void main() {
    vec3 pos = aBase;
    for (int m = 0; m < 52; ++m) {
        float w = uWeights[m];
        if (w != 0.0) {
            pos += w * fetchOffset(m, gl_VertexID);
        }
    }
    vNormal = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(pos, 1.0);
}
"""

_FRAG_SHADER = """
#version 330
in vec3 vNormal;
out vec4 fragColor;
const vec3 LIGHT = normalize(vec3(0.4, 0.6, 1.0));
const vec3 SKIN = vec3(0.82, 0.78, 0.74);
void main() {
    float d = max(dot(normalize(vNormal), LIGHT), 0.0);
    vec3 c = SKIN * (0.35 + 0.75 * d);
    fragColor = vec4(c, 1.0);
}
"""

_BORDER_VERT = """
#version 330
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0); }
"""

_BORDER_FRAG = """
#version 330
out vec4 fragColor;
void main() { fragColor = vec4(0.25, 0.7, 1.0, 1.0); }
"""


# ---------------------------------------------------------------------------
# Small matrix helpers (numpy, row-major; uploaded with transpose=GL_TRUE)
# ---------------------------------------------------------------------------

def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _translate(x, y, z) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def _rotate_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def _rotate_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _rotate_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


# MediaPipe's facial-transform translation is in a large camera-space metric
# (tz is the head's distance from the camera, tens of units). Applied raw it
# throws the preview head far off screen, so we scale it down to mesh units.
_HEAD_POS_SCALE = 0.05

# How far below the head centre the neck pivot sits (mesh units) at neck_anchor
# = 1. The head is roughly one unit tall, so the neck is ~one unit down.
_NECK_OFFSET = 1.0


def head_model_matrix(rot, pos, pos_scale: float = _HEAD_POS_SCALE,
                      neck_anchor: float = 0.0) -> np.ndarray:
    """Model matrix for a head pose: rotate (about a pivot), then translate.

    *rot* is XYZ Euler degrees, *pos* is the MediaPipe translation. A zero pose
    returns identity, so disabling the head pose (all axes zeroed) leaves the
    head exactly at the original base view.

    *neck_anchor* (0..1) moves the rotation pivot from the head's own centre
    (0) down to the neck (1): with the neck pivot, the neck stays put and the
    head swings against it. Rotation about a pivot p is T(p) R T(-p).
    """
    rx, ry, rz = rot
    tx, ty, tz = pos
    rot_m = _rotate_x(rx) @ _rotate_y(ry) @ _rotate_z(rz)
    if neck_anchor:
        py = -neck_anchor * _NECK_OFFSET  # pivot below the head centre
        rot_m = _translate(0.0, py, 0.0) @ rot_m @ _translate(0.0, -py, 0.0)
    return _translate(tx * pos_scale, ty * pos_scale, tz * pos_scale) @ rot_m


def _vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    if tris.size == 0:
        normals[:, 2] = 1.0
        return normals
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tris[:, 0], fn)
    np.add.at(normals, tris[:, 1], fn)
    np.add.at(normals, tris[:, 2], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (normals / lengths).astype(np.float32)


class MeshRenderer(QOpenGLWidget):
    """Draws a HeadMesh deformed by 52 ARKit weights (GPU vertex-shader morphing)."""

    clicked = Signal()
    camera_changed = Signal()  # emitted when yaw/pitch/zoom change

    _DEFAULT_YAW = 0.0
    _DEFAULT_PITCH = 0.0
    _DEFAULT_ZOOM = 0.0

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if OPENGL_AVAILABLE:
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            self.setFormat(fmt)
        self._label = label
        self._mesh = None
        self._weights = np.zeros(_MAX_MORPHS, dtype=np.float32)
        self._selected = False
        self.setMinimumSize(160, 120)

        # Camera state (interactive).
        self._yaw = self._DEFAULT_YAW      # degrees, horizontal orbit
        self._pitch = self._DEFAULT_PITCH  # degrees, vertical orbit
        self._zoom = self._DEFAULT_ZOOM    # units added to translate-z
        self._pan_x = 0.0                  # horizontal pan
        self._pan_y = 0.0                  # vertical pan
        self._drag_pos = None              # QPoint when left-dragging
        self._pan_drag_pos = None          # QPoint when middle-dragging

        # Head pose from MediaPipe ({"rot": [x,y,z] deg, "pos": [x,y,z]}).
        self._head_rot = [0.0, 0.0, 0.0]
        self._head_pos = [0.0, 0.0, 0.0]
        self._neck_anchor = 0.0  # 0 = pivot at head centre, 1 = pivot at neck

        # GPU resources / state.
        self._gpu_ready = False
        self._program = 0
        self._border_program = 0
        self._border_vao = 0
        self._vao = 0
        self._base_vbo = 0
        self._normal_vbo = 0
        self._ebo = 0
        self._offset_tex = 0
        self._tex_width = 0
        self._index_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mesh(self, mesh) -> None:
        self._mesh = mesh
        if OPENGL_AVAILABLE and self.isValid():
            self.makeCurrent()
            try:
                self._upload_mesh()
            finally:
                self.doneCurrent()
        self.update()

    def set_weights(self, weights) -> None:
        self._weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.update()

    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self.update()

    def set_head_pose(self, pose: dict | None) -> None:
        if pose:
            self._head_rot = list(pose.get("rot", [0.0, 0.0, 0.0]))
            self._head_pos = list(pose.get("pos", [0.0, 0.0, 0.0]))
            if "neck_anchor" in pose:
                self._neck_anchor = max(0.0, min(1.0, float(pose["neck_anchor"])))
        else:
            self._head_rot = [0.0, 0.0, 0.0]
            self._head_pos = [0.0, 0.0, 0.0]
        self.update()

    def set_neck_anchor(self, value: float) -> None:
        self._neck_anchor = max(0.0, min(1.0, float(value)))
        self.update()

    def has_mesh(self) -> bool:
        return self._mesh is not None

    def get_camera(self) -> dict:
        return {"yaw": self._yaw, "pitch": self._pitch, "zoom": self._zoom,
                "pan_x": self._pan_x, "pan_y": self._pan_y}

    def set_camera(self, state: dict) -> None:
        self._yaw = float(state.get("yaw", self._DEFAULT_YAW))
        self._pitch = float(state.get("pitch", self._DEFAULT_PITCH))
        self._zoom = float(state.get("zoom", self._DEFAULT_ZOOM))
        self._pan_x = float(state.get("pan_x", 0.0))
        self._pan_y = float(state.get("pan_y", 0.0))
        self.update()

    def reset_camera(self) -> None:
        self._yaw = self._DEFAULT_YAW
        self._pitch = self._DEFAULT_PITCH
        self._zoom = self._DEFAULT_ZOOM
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.camera_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Mouse / wheel interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.position().toPoint()
            self.clicked.emit()
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
            self._yaw += delta.x() * 0.5
            self._pitch += delta.y() * 0.5
            self._pitch = max(-89.0, min(89.0, self._pitch))
            self.camera_changed.emit()
            self.update()
        if self._pan_drag_pos is not None and (event.buttons() & Qt.MouseButton.MiddleButton):
            delta = pos - self._pan_drag_pos
            self._pan_drag_pos = pos
            self._pan_x += delta.x() * 0.004
            self._pan_y -= delta.y() * 0.004
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
        self._zoom -= delta / 120.0 * 0.15
        self._zoom = max(-2.0, min(3.0, self._zoom))
        self.camera_changed.emit()
        self.update()

    def _show_context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        reset_act = QAction("Reset view", self)
        reset_act.triggered.connect(self.reset_camera)
        menu.addAction(reset_act)
        menu.exec(global_pos)

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.10, 0.10, 0.12, 1.0)
        self._program = self._link(_VERT_SHADER, _FRAG_SHADER)
        self._border_program = self._link(_BORDER_VERT, _BORDER_FRAG)
        self._init_border()
        if self._mesh is not None:
            self._upload_mesh()

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

    def _init_border(self) -> None:
        # A line-loop just inside the viewport edge, in NDC-ish [0,1] space.
        verts = np.array([0.01, 0.01, 0.99, 0.01, 0.99, 0.99, 0.01, 0.99],
                         dtype=np.float32)
        self._border_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._border_vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # GPU upload
    # ------------------------------------------------------------------

    def _upload_mesh(self) -> None:
        if self._program == 0 or self._mesh is None:
            return
        mesh = self._mesh
        n = mesh.vertex_count
        base = np.ascontiguousarray(mesh.base, dtype=np.float32)
        tris = (mesh.indices.reshape(-1, 3).astype(np.int64)
                if mesh.indices.size else np.zeros((0, 3), np.int64))
        normals = _vertex_normals(base, tris)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._base_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._base_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, base.nbytes, base, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        self._normal_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._normal_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        if mesh.indices.size:
            idx = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
            self._ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
            self._index_count = idx.size
        GL.glBindVertexArray(0)

        # Morph offsets -> a morph-major RGB32F texture (52*N texels).
        total = _MAX_MORPHS * n
        max_w = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
        self._tex_width = min(total, max(1, min(max_w, 4096)))
        rows = (total + self._tex_width - 1) // self._tex_width
        padded = np.zeros((rows * self._tex_width, 3), dtype=np.float32)
        padded[:total] = mesh.arkit_offsets.reshape(total, 3)
        self._offset_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._offset_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, self._tex_width, rows,
                        0, GL.GL_RGB, GL.GL_FLOAT, padded)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._gpu_ready = True

    # ------------------------------------------------------------------
    # Resize / paint
    # ------------------------------------------------------------------

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE:
            return
        GL.glViewport(0, 0, max(1, w), max(1, h))

    def _mvp(self) -> tuple[np.ndarray, np.ndarray]:
        w, h = max(1, self.width()), max(1, self.height())
        proj = _perspective(45.0, w / float(h), 0.1, 100.0)
        base_z = -2.6 - self._zoom
        view = (
            _translate(self._pan_x, self._pan_y, base_z)
            @ _rotate_x(self._pitch)
            @ _rotate_y(180.0 + self._yaw)
        )
        model = head_model_matrix(
            self._head_rot, self._head_pos, neck_anchor=self._neck_anchor
        )
        mv = view @ model
        mvp = np.ascontiguousarray(proj @ mv, dtype=np.float32)
        normal_mat = np.ascontiguousarray(mv[:3, :3], dtype=np.float32)
        return mvp, normal_mat

    def paintGL(self) -> None:  # noqa: N802
        if not OPENGL_AVAILABLE:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._gpu_ready:
            mvp, nmat = self._mvp()
            GL.glUseProgram(self._program)
            loc = GL.glGetUniformLocation
            GL.glUniformMatrix4fv(loc(self._program, "uMVP"), 1, GL.GL_TRUE, mvp)
            GL.glUniformMatrix3fv(loc(self._program, "uNormalMat"), 1, GL.GL_TRUE, nmat)
            GL.glUniform1fv(loc(self._program, "uWeights"), _MAX_MORPHS, self._weights)
            GL.glUniform1i(loc(self._program, "uVertexCount"), self._mesh.vertex_count)
            GL.glUniform1i(loc(self._program, "uTexWidth"), self._tex_width)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._offset_tex)
            GL.glUniform1i(loc(self._program, "uOffsets"), 0)
            GL.glBindVertexArray(self._vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
        if self._selected and self._border_program:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glLineWidth(3.0)
            GL.glUseProgram(self._border_program)
            GL.glBindVertexArray(self._border_vao)
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 4)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glEnable(GL.GL_DEPTH_TEST)


class MeshPlaceholder(QWidget):
    """Shown instead of a MeshRenderer when OpenGL is unavailable."""

    clicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lbl = QLabel(text, self)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("background:#16161a; color:#778; padding:8px;")
        lay.addWidget(self._lbl)

    def set_mesh(self, mesh) -> None:
        pass

    def set_weights(self, weights) -> None:
        pass

    def set_selected(self, selected: bool) -> None:
        self._lbl.setStyleSheet(
            "background:#16161a; color:#9ad; border:2px solid #3ab; padding:8px;"
            if selected else "background:#16161a; color:#778; padding:8px;"
        )

    def has_mesh(self) -> bool:
        return False

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)
