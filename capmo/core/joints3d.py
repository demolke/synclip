"""
The ``<take>_joints3d.npz`` contract written by ``reconstruct_take`` (and the
pipeline), and its conversion for the in-app character playback.

File keys: ``joints`` (T,33,3) float32 metres, ``valid`` (T,33) bool, ``fps``,
``t`` (T,) seconds, ``names`` (33,) MediaPipe landmark names; optional
``t_source`` / ``frame_idx``. World frame (Blender convention): right-handed,
``+Z`` up, floor at ``z = 0``, the performer's average facing along ``-Y``
(their left is ``+X``). ``joints`` is finite everywhere; bridged samples have
``valid = False``.

The vendored glTF character (and :mod:`capmo.animate` / :mod:`capmo.retarget`)
is **+Y up, facing +Z**: :meth:`Joints3D.to_y_up` maps ``(x, y, z) -> (x, z, -y)``,
which turns ``-Y`` facing into ``+Z`` and keeps the performer's left on ``+X``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Z-up (Blender / reconstruct_take) -> Y-up (glTF / animate): x'=x, y'=z, z'=-y
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


@dataclass
class Joints3D:
    joints: np.ndarray          # (T,33,3)
    valid: np.ndarray           # (T,33) bool
    fps: float
    t: np.ndarray               # (T,) s
    names: tuple = ()
    up: str = "z"               # "z" (file frame) or "y" (after to_y_up)

    @property
    def T(self) -> int:
        return int(self.joints.shape[0])

    @property
    def t_ms(self) -> np.ndarray:
        return np.asarray(self.t, np.float64) * 1000.0

    def to_y_up(self) -> "Joints3D":
        """The same track in the Y-up character frame (no-op if already Y-up)."""
        if self.up == "y":
            return self
        J = np.asarray(self.joints, np.float64) @ Z_UP_TO_Y_UP.T
        return Joints3D(J, np.asarray(self.valid, bool), self.fps, self.t, self.names, "y")


def load_joints3d(path: str) -> Joints3D:
    d = np.load(path, allow_pickle=False)
    J = np.asarray(d["joints"], np.float64)
    fps = float(d["fps"]) if "fps" in d.files else 30.0
    t = np.asarray(d["t"], np.float64) if "t" in d.files else np.arange(len(J)) / fps
    names = tuple(str(n) for n in d["names"]) if "names" in d.files else ()
    valid = np.asarray(d["valid"], bool) if "valid" in d.files else np.isfinite(J).all(-1)
    return Joints3D(J, valid, fps, t, names)
