"""
ChArUco camera calibration for the capmo rig.

A ChArUco board (a chessboard with ArUco markers in the white squares) gives far
more robust, sub-pixel correspondences than a bare QR, which matters for
multi-camera extrinsics. The flow has two stages:

1. **Intrinsics** (per camera): detect the board across several views of one
   camera and solve for its matrix ``K`` + distortion.
2. **Extrinsics** (the rig): with intrinsics known, a *single* synchronized
   frame in which a camera sees the board gives that camera's pose **in the
   board frame** (via PnP). All cameras solved against the same board therefore
   share one world frame -- exactly what triangulation needs.

The board model and ChArUco detection (:class:`BoardConfig`, :class:`Detection`,
:func:`detect`) live in :mod:`capmo.core.board` and are re-exported here; this
module keeps the per-camera intrinsics accumulator, PnP and the rig container
(:class:`RigCalibration`, ``rig.json``) used by ``calibrate_board`` and the
reconstruction tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("capmo.calibration")

try:
    import cv2
    _ARUCO = hasattr(cv2, "aruco")
except ImportError:  # pragma: no cover
    cv2 = None
    _ARUCO = False


# The board model and the detector live in capmo.core.board (one implementation
# for the UI, extract_keypoints and calibrate_board); re-exported here.
from .core.board import BoardConfig, Detection, detect  # noqa: E402,F401


def _match_points(board, det: Detection):
    """Return matched (objPoints (N,1,3), imgPoints (N,1,2)) for a detection."""
    return board.matchImagePoints(det.corners, det.ids)


class IntrinsicCalibrator:
    """Accumulate per-camera board views, then solve for K + distortion."""

    def __init__(self, board_cfg: BoardConfig, min_corners: int = 6) -> None:
        self._cfg = board_cfg
        self._board = board_cfg.board()
        self._min = min_corners
        self._obj: list[np.ndarray] = []
        self._img: list[np.ndarray] = []
        self.image_size: tuple[int, int] | None = None   # (w, h)

    @property
    def count(self) -> int:
        return len(self._obj)

    def add(self, det: Detection, image_size: tuple[int, int]) -> bool:
        """Add one view; returns True if it had enough corners to keep."""
        if det.count < self._min:
            return False
        obj, img = _match_points(self._board, det)
        if obj is None or len(obj) < self._min:
            return False
        self._obj.append(np.asarray(obj, np.float32))
        self._img.append(np.asarray(img, np.float32))
        self.image_size = image_size
        return True

    def solve(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (K (3x3), dist, rms_reprojection_error)."""
        if self.count < 1 or self.image_size is None:
            raise ValueError("no views accumulated")
        rms, K, dist, _r, _t = cv2.calibrateCamera(
            self._obj, self._img, self.image_size, None, None)
        return K, dist.ravel(), float(rms)


def estimate_pose(det: Detection, K: np.ndarray, dist: np.ndarray,
                  board_cfg: BoardConfig, min_corners: int = 4):
    """Pose of the board in the camera frame -> (rvec(3), tvec(3)) or None."""
    if det.count < min_corners:
        return None
    obj, img = _match_points(board_cfg.board(), det)
    if obj is None or len(obj) < min_corners:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.asarray(dist, np.float64))
    if not ok:
        return None
    return rvec.ravel(), tvec.ravel()


@dataclass
class CameraCalibration:
    """One camera's intrinsics + its pose in the (board) world frame.

    The pose stored is the **board-in-camera** transform from PnP, i.e. a board
    point X_b maps to the camera frame as ``X_c = R @ X_b + t``. The projection
    matrix below uses exactly that, so a 3D world (=board) point projects to this
    camera's pixels.
    """
    cam_id: str
    image_size: tuple[int, int]
    K: np.ndarray            # (3,3)
    dist: np.ndarray         # (k,)
    rvec: np.ndarray         # (3,)
    tvec: np.ndarray         # (3,)

    def rotation_matrix(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(np.asarray(self.rvec, np.float64))
        return R

    def projection_matrix(self) -> np.ndarray:
        """3x4 ``K [R|t]`` mapping world (board) points to this camera's pixels."""
        R = self.rotation_matrix()
        Rt = np.hstack([R, np.asarray(self.tvec, np.float64).reshape(3, 1)])
        return np.asarray(self.K, np.float64) @ Rt

    def to_dict(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "image_size": list(self.image_size),
            "K": np.asarray(self.K, float).tolist(),
            "dist": np.asarray(self.dist, float).ravel().tolist(),
            "rvec": np.asarray(self.rvec, float).ravel().tolist(),
            "tvec": np.asarray(self.tvec, float).ravel().tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            cam_id=d["cam_id"],
            image_size=tuple(d["image_size"]),
            K=np.array(d["K"], float),
            dist=np.array(d["dist"], float),
            rvec=np.array(d["rvec"], float),
            tvec=np.array(d["tvec"], float),
        )


@dataclass
class RigCalibration:
    """The whole multi-camera rig: board + per-camera calibration."""
    board: BoardConfig = field(default_factory=BoardConfig)
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "board": self.board.to_dict(),
            "cameras": {k: c.to_dict() for k, c in self.cameras.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigCalibration":
        return cls(
            board=BoardConfig.from_dict(d.get("board", {})),
            cameras={k: CameraCalibration.from_dict(v)
                     for k, v in d.get("cameras", {}).items()},
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RigCalibration":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Human-readable rig description (pure: numpy only, no cv2 / no I/O).
# ---------------------------------------------------------------------------

def _euler_yaw_pitch_roll(R_c2w: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll (degrees) of a camera-to-world rotation matrix.

    Uses the common Z(yaw)-Y(pitch)-X(roll) convention applied to the columns
    of ``R_c2w`` (the camera axes expressed in world coordinates). This is only
    for eyeballing orientation, so the exact convention matters less than that
    it is consistent and stable.
    """
    r = np.asarray(R_c2w, float)
    sy = -r[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        yaw = np.arctan2(r[1, 0], r[0, 0])
        roll = np.arctan2(r[2, 1], r[2, 2])
    else:  # gimbal lock
        yaw = np.arctan2(-r[0, 1], r[1, 1])
        roll = 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def camera_center(cam: CameraCalibration) -> np.ndarray:
    """World position (camera center) of a camera: ``-R.T @ t``."""
    R = cam.rotation_matrix()
    t = np.asarray(cam.tvec, float).reshape(3)
    return (-R.T @ t)


def camera_forward(cam: CameraCalibration) -> np.ndarray:
    """Unit world-space direction the camera looks along (its +Z axis)."""
    R = cam.rotation_matrix()
    fwd = R.T @ np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(fwd)
    return fwd / n if n > 0 else fwd


def describe_rig(rig: RigCalibration) -> str:
    """Return a human-readable, multi-line description of the rig layout.

    World frame = the calibration board frame (origin at the board). For each
    camera the report gives its id, world position (camera center, metres),
    look/forward direction, yaw/pitch/roll, distance to the world/board origin,
    and the reprojection focal length; it then lists pairwise baseline distances
    between cameras. Intended for eyeballing "where each camera is and how the
    rig is arranged" from a print() to stdout.
    """
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("capmo rig calibration")
    b = rig.board
    lines.append(
        f"board: {b.squares_x}x{b.squares_y} squares, "
        f"square={b.square_len:.3f}m marker={b.marker_len:.3f}m ({b.dict_name})")
    lines.append(f"world frame = board frame (origin at board)")
    lines.append(f"cameras: {len(rig.cameras)}")
    lines.append("-" * 64)

    centers: dict[str, np.ndarray] = {}
    for cid, cam in rig.cameras.items():
        C = camera_center(cam)
        centers[cid] = C
        fwd = camera_forward(cam)
        R_c2w = cam.rotation_matrix().T
        yaw, pitch, roll = _euler_yaw_pitch_roll(R_c2w)
        dist_origin = float(np.linalg.norm(C))
        fx = float(np.asarray(cam.K, float)[0, 0])
        lines.append(f"[{cid}]")
        lines.append(
            f"    position (m):  x={C[0]:+.3f}  y={C[1]:+.3f}  z={C[2]:+.3f}")
        lines.append(
            f"    look dir:      ({fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f})")
        lines.append(
            f"    orientation:   yaw={yaw:+.1f}  pitch={pitch:+.1f}  roll={roll:+.1f} (deg)")
        lines.append(f"    dist to origin: {dist_origin:.3f} m")
        lines.append(
            f"    image_size: {tuple(cam.image_size)}  fx={fx:.1f}px")

    lines.append("-" * 64)
    ids = list(rig.cameras.keys())
    if len(ids) >= 2:
        lines.append("baselines (camera-to-camera distance, m):")
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b_id = ids[i], ids[j]
                d = float(np.linalg.norm(centers[a] - centers[b_id]))
                lines.append(f"    {a} <-> {b_id}: {d:.3f}")
    else:
        lines.append("baselines: (need >= 2 cameras)")
    lines.append("=" * 64)
    return "\n".join(lines)
