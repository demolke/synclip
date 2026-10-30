"""
The offline processing library: recording -> 2D keypoints -> rig -> 3D joints.

ONE implementation, used by the CLI tools (``capmo/tools/*``, which add the
argument parsing and console output) and by the Qt UI ("Process…" / "View"):

* :func:`extract_keypoints` -- MediaPipe pose (or ChArUco detections with
  ``calib=True``) per camera of a capmo session directory or of video files
  -> ``<out>/<take>/<cam>.npz`` + ``meta.json`` (= ``capmo.tools.extract_keypoints``).
* :func:`calibrate_board` -- board detections of a take -> metric ``rig.json``
  (= ``capmo.tools.calibrate_board``).
* :func:`reconstruct_take` -- keypoint take (+ optional rig, else body
  self-calibration) -> ``<take>_joints3d.npz`` (= ``capmo.tools.reconstruct_take``).
* :func:`process_session` -- the whole chain for one recording: pose
  extraction; if the manifest names a board (the UI's "Board check" was on)
  and no rig is given, board detection + ``calibrate_board`` (cameras without
  a board solution are self-calibrated from the body and anchored to the
  board cameras, ``fallback_selfcal``; if the board cannot place at least two
  cameras, the whole rig is self-calibrated); then the reconstruction.
  Returns a :class:`PipelineResult` whose :meth:`~PipelineResult.joints3d`
  is the 33-joint track (:mod:`capmo.core.joints3d`).
* :func:`triangulate_poses` -- in-memory variant for already calibrated
  cameras: per-camera lists of :class:`~capmo.core.pose.PoseResult` ->
  :class:`capmo.multiview.TriangulationResult` (no files, no self-calibration).

``progress(stage, detail, fraction)`` is called at every stage / camera
boundary; raising from it (e.g. :class:`Cancelled`) aborts the run there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .joints3d import Joints3D, load_joints3d

Progress = Callable[[str, str, "float | None"], None]


class PipelineError(RuntimeError):
    pass


class Cancelled(PipelineError):
    """Raised from a progress callback to stop the pipeline."""


@dataclass
class PipelineResult:
    session_dir: str
    work_dir: str
    take_dir: str
    out_dir: str
    joints3d_path: str
    rig_path: str | None
    rig_source: str                  # "given" | "board" | "selfcal"
    report: dict = field(default_factory=dict)
    board_report: dict | None = None
    warnings: list = field(default_factory=list)

    def joints3d(self) -> Joints3D:
        return load_joints3d(self.joints3d_path)


def _noop(*_a, **_k):
    pass


# ------------------------------------------------------------------ stages
def extract_keypoints(inputs, out_root: str, take: str, *, calib: bool = False,
                      estimator=None, cam_ids=None, options: dict | None = None,
                      on_camera=None) -> dict:
    """Run the extractor (pose, or board with *calib*) on a session directory
    or video files. *options*: CLI options as ``{"--model": "heavy",
    "--calib-every": 1, "--proxy": True, ...}``. Returns ``meta.json``.
    Raises :class:`PipelineError` if it fails."""
    from ..tools import extract_keypoints as ek
    argv = ["--out", out_root, "--take", take, "--quiet"]
    if calib:
        argv.append("--calib")
    if cam_ids:
        argv += ["--cam-ids", cam_ids if isinstance(cam_ids, str) else ",".join(cam_ids)]
    for k, v in (options or {}).items():
        if v is True:
            argv.append(k)
        elif v is not None and v is not False:
            argv += [k, str(v)]
    argv += [inputs] if isinstance(inputs, str) else list(inputs)
    args = ek.build_parser().parse_args(argv)
    code = ek.run(args, estimator, on_camera=on_camera)
    if code != 0:
        raise PipelineError(f"extract_keypoints{' --calib' if calib else ''} failed "
                            f"(exit {code}) on {inputs}")
    with open(os.path.join(out_root, take, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def calibrate_board(take_dir: str, out: str, **kw) -> dict:
    """``calibrate_board.run`` (raises ``BoardCalibError``); returns the report."""
    from ..tools import calibrate_board as cb
    return cb.run(take_dir, out, **kw)


def reconstruct_take(take_dir: str, out_dir: str, rig: str | None = None, **kw) -> dict:
    """``reconstruct_take.run``; returns the report."""
    from ..tools import reconstruct_take as rt
    return rt.run(take_dir, out_dir, rig, **kw)


def session_board_offsets(take_dir: str) -> dict:
    """``calibrate_board --offsets`` for a take extracted from a capmo session:
    its per-camera ``frame_idx`` start at each camera's first frame, the common
    clock is ``t`` (``calibrate_board`` applies this itself to session takes;
    see :func:`capmo.core.sync.timeline.session_take_offsets`)."""
    from .sync.timeline import session_take_offsets
    return session_take_offsets(take_dir)


# ---------------------------------------------------------------- the chain
def process_session(session_dir: str, work_dir: str | None = None, *, rig: str | None = None,
                    board: str | bool = "auto", estimator=None, model: str = "full",
                    cam_ids=None, max_height: int = 1080, calib_every: int | None = None,
                    render: bool = False, reconstruct_kw: dict | None = None,
                    progress: Progress | None = None, log: Callable = None) -> PipelineResult:
    """Session directory -> 3D joints (see the module docstring).

    *work_dir* (default ``<session>/processed``) receives ``kp/<name>/`` (the
    keypoint take), ``rig.json`` / ``rig_report.json`` (board path) and
    ``mocap/<name>_joints3d.npz`` (+ ``report.json``, ``rig.json``).
    *rig*: an existing ``rig.json`` (skips the board; session cameras missing
    from it are left out). *board*: ``"auto"`` (use the manifest's board if
    any), ``False`` (always self-calibrate). *estimator*: injected pose
    estimator (tests); else MediaPipe *model*. *calib_every*: board detection
    stride (default: ~10 per second of video).
    """
    from .session_io import is_session_dir, load_session
    progress = progress or _noop
    log = log or _noop
    if not is_session_dir(session_dir):
        raise PipelineError(f"{session_dir} is not a capmo session (no manifest.json)")
    sess = load_session(session_dir)
    if not sess.cameras:
        raise PipelineError(f"session {session_dir} has no camera with frames")
    name = sess.name
    work_dir = work_dir or os.path.join(session_dir, "processed")
    kp_root = os.path.join(work_dir, "kp")
    take_dir = os.path.join(kp_root, name)
    out_dir = os.path.join(work_dir, "mocap")
    warnings: list[str] = []

    cams = list(sess.cam_ids)
    if cam_ids:
        cams = [c for c in cams if c in set(cam_ids)]
    if rig:
        with open(rig, encoding="utf-8") as f:
            in_rig = set(json.load(f).get("cameras", {}))
        dropped = [c for c in cams if c not in in_rig]
        cams = [c for c in cams if c in in_rig]
        if dropped:
            msg = f"camera(s) {', '.join(dropped)} not in {rig}: left out"
            warnings.append(msg)
            log(msg)
    if len(cams) < 2:
        raise PipelineError(f"need >= 2 cameras, have {cams}")
    select = ",".join(f"{c}={c}" for c in cams)       # select by cam id, keep names

    def on_camera(stage):
        def hook(cam, i, n):
            progress(stage, cam, i / max(n, 1))
        return hook

    # 1) pose keypoints
    progress("extract", "pose keypoints", 0.0)
    meta = extract_keypoints(session_dir, kp_root, name, estimator=estimator, cam_ids=select,
                             options={"--model": model, "--max-height": max_height},
                             on_camera=on_camera("extract"))
    seen = {e["cam_id"]: int(e.get("n_detected", 0)) for e in meta.get("videos", [])}
    if sum(1 for c in cams if seen.get(c, 0) > 0) < 2:
        raise PipelineError("no person detected in at least two cameras (detections per "
                            "camera: " + ", ".join(f"{c} {seen.get(c, 0)}" for c in cams) + ")")

    # 2) board -> rig
    rig_path, rig_source, board_report = rig, ("given" if rig else "selfcal"), None
    if rig is None and board is not False and sess.board:
        fps = float(sess.nominal_fps or sess.cameras[0].fps_nominal or 30.0)
        every = calib_every or max(1, int(round(fps / 10.0)))
        from ..tools import calibrate_board as cb
        expected = (cb.BoardCalibError, PipelineError, OSError, ValueError) + (
            (cb.cv2.error,) if cb.cv2 is not None else ())
        progress("board", "board detections", 0.0)
        try:
            extract_keypoints(session_dir, kp_root, name, calib=True, cam_ids=select,
                              options={"--calib-every": every, "--min-corners": 8},
                              on_camera=on_camera("board"))
            progress("board", "calibrate_board", None)
            rig_path = os.path.join(work_dir, "rig.json")
            board_report = calibrate_board(        # session take: clock offsets
                take_dir, rig_path, report_path=os.path.join(work_dir, "rig_report.json"),
                fallback_selfcal=True, pose_take=take_dir, log=log)
            rig_source = "board"
            fb = (board_report.get("fallback_selfcal") or {}).get("cameras")
            if fb:
                msg = f"camera(s) {', '.join(fb)} not placed by the board: self-calibrated"
                warnings.append(msg)
                log(msg)
        except Cancelled:
            raise
        except expected as exc:
            # no usable board solution (too few views, extraction failed, degenerate
            # data, an OpenCV solver error); programming errors (AttributeError,
            # TypeError, a missing cv2.aruco, ...) propagate
            msg = f"board calibration failed ({exc}); self-calibrating from the body"
            warnings.append(msg)
            log(msg)
            rig_path, rig_source = None, "selfcal"

    # 3) reconstruction
    progress("reconstruct", rig_source, None)
    kw = dict(render=render, log=log)
    kw.update(reconstruct_kw or {})
    report = reconstruct_take(take_dir, out_dir, rig_path, **kw)
    progress("done", out_dir, 1.0)
    return PipelineResult(session_dir=session_dir, work_dir=work_dir, take_dir=take_dir,
                          out_dir=out_dir,
                          joints3d_path=os.path.join(out_dir, f"{name}_joints3d.npz"),
                          rig_path=rig_path, rig_source=rig_source, report=report,
                          board_report=board_report, warnings=warnings)


# ------------------------------------------------------- in-memory variant
def triangulate_poses(cameras, poses: dict, **kw):
    """Triangulate per-frame 2D poses of calibrated cameras.

    *cameras*: ``{cam_id: Camera | CameraCalibration}`` (or a
    ``RigCalibration``); *poses*: ``{cam_id: [PoseResult | None, ...]}``, the
    lists aligned frame by frame (same length). Returns
    :class:`capmo.multiview.TriangulationResult` (``X`` (T,33,3), ``valid``,
    ``n_views``, ``err_px`` ...; ``**kw`` go to
    :func:`capmo.multiview.reconstruct_take`).
    """
    from .. import multiview as mv
    from .pose import NUM_LANDMARKS
    if hasattr(cameras, "cameras"):
        cameras = cameras.cameras
    ids = [c for c in poses if c in cameras]
    if len(ids) < 2:
        raise PipelineError("need >= 2 calibrated cameras with poses")
    T = len(poses[ids[0]])
    kp = np.full((len(ids), T, NUM_LANDMARKS, 2), np.nan)
    vis = np.zeros((len(ids), T, NUM_LANDMARKS))
    cams = []
    for v, c in enumerate(ids):
        cam = cameras[c]
        cams.append(cam if isinstance(cam, mv.Camera) else mv.Camera.from_calibration(cam))
        if len(poses[c]) != T:
            raise PipelineError("pose lists must have the same length")
        for t, r in enumerate(poses[c]):
            if r is None:
                continue
            kp[v, t] = r.points
            vis[v, t] = r.visibility
    res = mv.reconstruct_take(cams, kp, vis, **kw)
    res.cam_ids = ids
    return res
