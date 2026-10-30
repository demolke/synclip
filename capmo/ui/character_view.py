"""
A software-rendered character playback widget (GL-free).

Holds a rig (:class:`capmo.skinning.SkinData`) and a rig-pose track, renders each
frame with :func:`capmo.animate.render_character` (pure-numpy rasterizer, no
OpenGL / MediaPipe) and blits the BGR image into a :class:`QLabel` via
``QImage``/``QPixmap`` -- the same pattern :class:`capmo.ui.main_window.CameraTile`
uses for live previews. A :class:`QTimer` advances frames for play/pause and an
optional slow yaw orbit makes the pose easier to read.

The numpy rasterizer needs tens of milliseconds per frame (more than the play
timer's interval), so rendering runs in a worker thread
(:class:`~capmo.ui.workers.LatestWorker`): the timer only advances the frame
index and requests a render; if the renderer is behind, the frames in between
are skipped (playback keeps its pace, the picture just updates less often) and
the GUI thread only blits finished images. The widget degrades gracefully when
no rig / track is loaded -- it just shows a placeholder label instead of
crashing. Swapping in a real OpenGL view later only means replacing the render
call + the blit.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .. import animate
from .workers import LatestWorker


class CharacterView(QWidget):
    """Plays a rig-pose track by software-rendering each frame to a QLabel."""

    def __init__(self, parent=None, *, width: int = 360, height: int = 480,
                 fps: float = 20.0, orbit: bool = True) -> None:
        super().__init__(parent)
        self._w = int(width)
        self._h = int(height)
        self._orbit = orbit

        self._skin = None
        self._track: list[dict] = []
        self._frame = 0
        self._yaw = 0.0
        self._cam = None                          # cached (centre, height) of the rig
        self._gen = 0                             # bumped per track: drops stale renders
        self._renderer = LatestWorker(self._render_job, "capmo-character-render", self)
        self._renderer.done.connect(self._on_rendered)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("no character loaded", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(self._w, self._h)
        self._label.setStyleSheet("background:#16161a; color:#667;")
        lay.addWidget(self._label, stretch=1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.advance)
        self._interval_ms = max(1, int(1000.0 / max(fps, 1.0)))

    # -- public API ----------------------------------------------------------

    def set_track(self, skin, track) -> None:
        """Load a rig (:class:`SkinData`) + a rig-pose track (list of dicts).

        Requests the first frame at once (so the view isn't blank before
        :meth:`play`; it appears when the renderer thread delivers it). A
        ``None`` skin or empty track shows the placeholder.
        """
        self.stop()
        self._gen += 1
        self._cam = None                          # the new rig may have another size
        self._skin = skin
        self._track = list(track) if track is not None else []
        self._frame = 0
        self._yaw = 0.0
        if self._skin is None or not self._track:
            self._label.setText("no character loaded")
            self._label.setPixmap(QPixmap())
            return
        self._request_render()

    def play(self) -> None:
        if self._skin is not None and self._track:
            self._timer.start(self._interval_ms)

    def stop(self) -> None:
        self._timer.stop()

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def advance(self) -> None:
        """Step to the next frame (looping at the end) and request its render."""
        if self._skin is None or not self._track:
            self.stop()
            return
        self._frame = (self._frame + 1) % len(self._track)
        if self._orbit:
            self._yaw += math.radians(4.0)       # gentle turntable
        self._request_render()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait for the pending render and show it (tests / screenshots)."""
        from PySide6.QtWidgets import QApplication
        ok = self._renderer.wait_idle(timeout)
        QApplication.processEvents()
        return ok

    def shutdown(self) -> None:
        """Stop playing and join the render thread (window close)."""
        self.stop()
        self._renderer.stop()

    def current_pixmap(self) -> QPixmap:
        """The QLabel's current pixmap (for tests / screenshots)."""
        pm = self._label.pixmap()
        return pm if pm is not None else QPixmap()

    # -- rendering -----------------------------------------------------------

    def _base_camera(self):
        """Cache the bind-framed camera centre + distance for the loaded rig."""
        if self._cam is None:
            bind = self._skin.verts
            lo, hi = bind.min(0), bind.max(0)
            self._cam = ((lo + hi) / 2.0, float(hi[1] - lo[1]) or 1.0)
        return self._cam

    def _request_render(self) -> None:
        center, height = self._base_camera()
        self._renderer.submit("frame", (self._gen, self._skin, self._track[self._frame],
                                        center, height, self._yaw))

    def _render_job(self, _key, job):
        """Render thread: one frame (``(gen, BGR image)``)."""
        from .. import synth
        gen, skin, pose, center, height, yaw = job
        dist = height * 1.8
        eye = center + np.array([dist * math.sin(yaw), 0.0, dist * math.cos(yaw)])
        R, t = synth.look_at(eye, center)
        f = 1.05 * max(self._w, self._h)
        K = synth.pinhole_K(f, f, self._w / 2.0, self._h / 2.0)
        return gen, animate.render_character(skin, pose, K, R, t, self._w, self._h)

    @Slot(object, object)
    def _on_rendered(self, _key, result) -> None:
        gen, img = result
        if gen == self._gen:
            self._blit(img)

    def _blit(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        bgr = np.ascontiguousarray(bgr)
        qimg = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(pix)
