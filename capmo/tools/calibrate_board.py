"""
Calibrate a multi-camera rig from ChArUco board detections
(``extract_keypoints --calib`` output) -> ``rig.json``.

    python -m capmo.tools.calibrate_board KP_TAKE_DIR --out rig.json \\
        [--start S --end E] [--report report.json] [--overlay-dir DIR]

Typical use: every video of a take *starts* with the board being shown to the
cameras for a few seconds, then the performance follows. Extract the board
window with ``extract_keypoints --calib --start 0 --end E --calib-every 1``,
run this tool, then ``reconstruct_take --rig rig.json --start E`` on the pose
extraction of the same videos.

Pipeline:

1. **Intrinsics** (per camera): from all board views (>= ``--min-corners``,
   non-degenerate), a well-spread subset of at most ``--max-views`` is chosen
   (farthest-point sampling over image position, board size and tilt), then
   ``cv2.calibrateCamera`` with the ChArUco object points; views whose RMS is
   far above the rest are dropped and the solve repeated. ``--dist`` picks the
   distortion model. ``--intrinsics-only cam=K.json`` takes a camera's
   intrinsics from a file instead (only its pose is solved).
2. **Sync**: views of different cameras are *simultaneous* when their frame
   indices agree (within ``--sync-tol`` frames) after the per-camera offsets
   (``--offsets camB=+2,camC=-1``: camera X's frame ``k + off`` shows the same
   instant as the reference camera's frame ``k``). ``--estimate-offsets``
   estimates them by cross-correlating the board centre's 3D speed seen by each
   camera (view-independent, needs dense sampling, e.g. ``--calib-every 1``).
   A take extracted from a capmo session (``meta.json`` ``timebase: session``)
   gets them from the common host clock when neither option is given.
3. **Extrinsics**: per simultaneous view, PnP in every camera gives the
   relative pose of each camera pair; the pair samples are averaged robustly
   (rotation chordal mean + translation median, outliers rejected). A camera
   graph (edge = pair with enough shared views) is built and the cameras are
   chained from the reference camera along a maximum spanning tree, so a
   camera that never co-sees the board with the reference is still placed
   through the others.
4. **Bundle adjustment** of all camera poses (and, with ``--ba-intrinsics``,
   the free intrinsics) plus one board pose per simultaneous view, minimising
   the board-corner reprojection error (scipy ``least_squares``, robust loss,
   sparse Jacobian). The reference camera's pose fixes the gauge; the board
   fixes the metric scale.
5. **World frame** = the board frame at one reference view (``--world-frame
   FRAME`` = reference-camera frame index; default: the simultaneous view with
   the most cameras and corners): origin at the board's outer corner next to
   ChArUco corner 0, ``+X`` along the board's ``squares_x`` direction, ``+Y``
   along ``squares_y`` (down the printed pattern), ``+Z = X x Y`` pointing
   *into* the board (away from the cameras), metres. ``reconstruct_take``
   re-derives floor/up/heading from the performer anyway and keeps the scale.

Outputs: ``rig.json`` (:class:`capmo.calibration.RigCalibration`, loadable by
``selfcal.from_rig`` / ``reconstruct_take --rig``), a ``describe_rig`` print,
a JSON report (``--report``) with per-camera intrinsic RMS and view counts, BA
RMS, the camera graph and warnings, and optional overlay images of the
reprojected board on sample frames (``--overlay-dir``; the videos are taken
from ``meta.json`` or ``--video cam=path``).

A camera with no usable board view (or not linked to the others) is an error
naming it; ``--fallback-selfcal`` instead self-calibrates all cameras from the
body keypoints of the take and anchors that solution to the board-calibrated
cameras with a similarity (needs >= 2 board cameras and a pose extraction).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .. import multiview as mv
from .. import selfcal as sc
from ..calibration import BoardConfig, CameraCalibration, RigCalibration, describe_rig
from ..core import board

# Thresholds used for the report's warnings.
WARN_INTRINSIC_RMS_PX = 1.0
WARN_BA_RMS_PX = 2.0
# coverage / tilt thresholds and the view-quality metrics are shared with the
# live board-coverage indicator (capmo.core.board / capmo.core.board_coverage)
WARN_COVERAGE = board.GOOD_COVERAGE
WARN_MAX_TILT_DEG = board.GOOD_MAX_TILT_DEG
COVERAGE_GRID = board.COVERAGE_GRID

# distortion model -> (cv2 flags, indices of the free coefficients in k1,k2,p1,p2,k3)
DIST_MODELS = {
    "none": ("FIX_K1 FIX_K2 FIX_K3 ZERO_TANGENT_DIST", []),
    "k1": ("FIX_K2 FIX_K3 ZERO_TANGENT_DIST", [0]),
    "k1k2": ("FIX_K3 ZERO_TANGENT_DIST", [0, 1]),
    "k1k2p": ("FIX_K3", [0, 1, 2, 3]),
    "k1k2pk3": ("", [0, 1, 2, 3, 4]),
}


class BoardCalibError(RuntimeError):
    """A calibration that cannot be completed (message names the cause)."""


def _flags(model: str, fix_aspect: bool = False, fix_pp: bool = False) -> int:
    f = 0
    for name in DIST_MODELS[model][0].split():
        f |= getattr(cv2, "CALIB_" + name)
    if fix_aspect:
        f |= cv2.CALIB_FIX_ASPECT_RATIO
    if fix_pp:
        f |= cv2.CALIB_FIX_PRINCIPAL_POINT
    return f


# --------------------------------------------------------------------- data
@dataclass
class View:
    """One board detection in one camera frame."""
    frame_idx: int
    t: float
    ids: np.ndarray            # (N,) ChArUco corner ids
    img: np.ndarray            # (N,2) pixels
    obj: np.ndarray            # (N,3) board coordinates (m), z = 0
    rvec: np.ndarray | None = None     # board -> camera (PnP)
    tvec: np.ndarray | None = None
    pnp_rms: float = np.nan

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class CamData:
    cam_id: str
    size: tuple[int, int]
    fps: float
    views: list[View]
    n_raw: int = 0
    video: str | None = None
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    intr_source: str = ""
    intr: dict = field(default_factory=dict)


def _parse_map(items, cast=str) -> dict:
    """``["a=1,b=2", "c=3"]`` -> ``{"a": cast("1"), ...}``."""
    out = {}
    for it in items or []:
        for part in str(it).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise BoardCalibError(f"expected cam=value, got {part!r}")
            k, v = part.split("=", 1)
            out[k.strip()] = cast(v.strip())
    return out


def load_board_take(take_dir: str, cam_ids=None, start=None, end=None, min_corners=8,
                    log=print):
    """Read ``<cam>.calib.npz`` files -> (list[CamData], BoardConfig)."""
    meta_path = os.path.join(take_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    paths = {e["cam_id"]: e.get("path") for e in meta.get("calib_videos", [])}
    if cam_ids is None:
        cam_ids = [e["cam_id"] for e in meta.get("calib_videos", [])
                   if os.path.exists(os.path.join(take_dir, f"{e['cam_id']}.calib.npz"))]
        if not cam_ids:
            cam_ids = sorted(f[:-len(".calib.npz")] for f in os.listdir(take_dir)
                             if f.endswith(".calib.npz"))
    if not cam_ids:
        raise BoardCalibError(f"no <cam>.calib.npz in {take_dir}: run "
                              "extract_keypoints --calib on the videos first")
    board_cfg = None
    corners3d = None
    cams = []
    for cid in cam_ids:
        p = os.path.join(take_dir, f"{cid}.calib.npz")
        if not os.path.exists(p):
            raise BoardCalibError(f"camera {cid}: {p} missing")
        d = np.load(p, allow_pickle=False)
        cfg = BoardConfig.from_dict(json.loads(str(d["board_json"])))
        if board_cfg is None:
            board_cfg = cfg
            corners3d = np.asarray(cfg.board().getChessboardCorners(), np.float64)
        elif (cfg.squares_x, cfg.squares_y, cfg.square_len) != (
                board_cfg.squares_x, board_cfg.squares_y, board_cfg.square_len):
            raise BoardCalibError(f"camera {cid}: different board {cfg.to_dict()} vs "
                                  f"{board_cfg.to_dict()}")
        offs = d["offsets"]
        views = []
        n_raw = len(d["t"])
        for k in range(n_raw):
            t = float(d["t"][k])
            if start is not None and t < start - 1e-6:
                continue
            if end is not None and t > end + 1e-6:
                continue
            ids = d["ids"][offs[k]:offs[k + 1]].astype(int)
            img = d["corners"][offs[k]:offs[k + 1]].astype(np.float64)
            if len(ids) < min_corners:
                continue
            obj = corners3d[ids]
            if not _non_degenerate(obj, board_cfg.square_len):
                continue
            views.append(View(int(d["frame_idx"][k]), t, ids, img, obj))
        cams.append(CamData(cid, (int(d["width"]), int(d["height"])), float(d["fps"]),
                            views, n_raw, paths.get(cid)))
        log(f"  {cid}: {n_raw} detections, {len(views)} usable views "
            f"(>= {min_corners} corners, not collinear{', in window' if start or end else ''})")
    return cams, board_cfg


def _non_degenerate(obj, square_len) -> bool:
    """Corners must span a 2D area (not one row/column)."""
    return board.non_degenerate(obj, square_len)


# --------------------------------------------------------------- intrinsics
def _rough_tilt(view: View, size) -> tuple[float, float]:
    """(tilt angle deg, tilt direction rad) of the board vs the image plane."""
    return board.rough_tilt(view.obj, view.img, size)


def select_views(views: list[View], size, max_views: int) -> list[int]:
    """Indices of a well-spread subset (image position, size, tilt) of *views*."""
    if len(views) <= max_views:
        return list(range(len(views)))
    w, h = size
    feats = []
    for v in views:
        c = v.img.mean(0)
        area = np.ptp(v.img[:, 0]) * np.ptp(v.img[:, 1]) / (w * h)
        tilt, dirn = _rough_tilt(v, size)
        st = np.sin(np.radians(tilt))
        feats.append([c[0] / w, c[1] / h, 0.25 * np.log(max(area, 1e-4)),
                      0.8 * st * np.cos(dirn), 0.8 * st * np.sin(dirn)])
    F = np.array(feats)
    cnt = np.array([v.n for v in views], float)
    q = np.sqrt(cnt / cnt.max())
    sel = [int(np.argmax(cnt))]
    dmin = np.linalg.norm(F - F[sel[0]], axis=1)
    while len(sel) < max_views:
        score = dmin * q
        score[sel] = -1
        k = int(np.argmax(score))
        sel.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[k], axis=1))
    return sorted(sel)


def coverage(views: list[View], size, grid=COVERAGE_GRID) -> float:
    """Fraction of image grid cells containing at least one detected corner."""
    return board.coverage((v.img for v in views), size, grid)


def calibrate_intrinsics(cam: CamData, *, dist_model="k1k2p", max_views=40, min_views=10,
                         outlier_factor=3.0, outlier_px=1.0, fix_aspect=False, fix_pp=False,
                         log=print) -> dict:
    """Solve K + distortion for one camera; returns an info dict (K, dist, rms, ...)."""
    if len(cam.views) < 3:
        raise BoardCalibError(
            f"camera {cam.cam_id}: only {len(cam.views)} usable board view(s), cannot solve its "
            "intrinsics (show the board to it longer, give --intrinsics-only "
            f"{cam.cam_id}=K.json, or use the self-calibration path)")
    sel = select_views(cam.views, cam.size, max_views)
    flags = _flags(dist_model, fix_aspect, fix_pp)
    rejected = []
    for it in range(6):
        obj = [cam.views[i].obj.astype(np.float32) for i in sel]
        img = [cam.views[i].img.astype(np.float32) for i in sel]
        rms, K, dist, _r, _t, _si, _se, per_view = cv2.calibrateCameraExtended(
            obj, img, cam.size, None, None, flags=flags)
        per_view = per_view.ravel()
        med = float(np.median(per_view))
        bad = np.flatnonzero((per_view > outlier_factor * med) & (per_view > outlier_px))
        if not len(bad) or len(sel) - len(bad) < max(4, min(min_views, len(sel)) // 2):
            break
        rejected += [cam.views[sel[b]].frame_idx for b in bad]
        sel = [s for k, s in enumerate(sel) if k not in set(bad.tolist())]
    used = [cam.views[i] for i in sel]
    tilts = [_rough_tilt(v, cam.size)[0] for v in used]
    info = {"K": K, "dist": dist.ravel()[:5], "rms_px": float(rms), "n_views_used": len(sel),
            "n_views_available": len(cam.views), "rejected_view_frames": rejected,
            "per_view_rms_px": {"median": float(np.median(per_view)),
                                "max": float(per_view.max())},
            "coverage": coverage(cam.views, cam.size), "coverage_used": coverage(used, cam.size),
            "tilt_deg": {"min": float(min(tilts)), "median": float(np.median(tilts)),
                         "max": float(max(tilts))},
            "dist_model": dist_model}
    log(f"  {cam.cam_id}: intrinsics from {len(sel)}/{len(cam.views)} views: rms {rms:.3f}px, "
        f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, dist "
        + " ".join(f"{x:+.4f}" for x in info["dist"]) + f"; coverage {info['coverage']:.0%}, "
        f"tilt max {info['tilt_deg']['max']:.0f} deg"
        + (f"; dropped {len(rejected)} outlier views" if rejected else ""))
    return info


def load_intrinsics(path: str, cam_id: str):
    """(K, dist, size|None) from a ``{"K", "dist"}`` JSON or a rig.json."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "cameras" in d:
        if cam_id not in d["cameras"]:
            raise BoardCalibError(f"{path} has no camera {cam_id}")
        d = d["cameras"][cam_id]
    K = np.array(d["K"], float)
    dist = np.zeros(5)
    dd = np.array(d.get("dist", []), float).ravel()[:5]
    dist[:len(dd)] = dd
    size = tuple(d["image_size"]) if "image_size" in d else None
    return K, dist, size


# ---------------------------------------------------------------------- PnP
def solve_view_poses(cam: CamData, max_rms_px=3.0) -> int:
    """PnP for every view of *cam* with its intrinsics; returns the number kept."""
    n = 0
    for v in cam.views:
        v.rvec = v.tvec = None
        obj = v.obj.astype(np.float64)
        img = v.img.astype(np.float64)
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            ok = False
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rv, tv)
        pr, _ = cv2.projectPoints(obj, rv, tv, cam.K, cam.dist)
        v.pnp_rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - img) ** 2, 1))))
        if tv.ravel()[2] <= 0 or v.pnp_rms > max_rms_px:
            continue
        v.rvec, v.tvec = rv.ravel(), tv.ravel()
        n += 1
    return n


def _Rt(view: View):
    return mv.rodrigues(view.rvec), view.tvec


# --------------------------------------------------------------------- sync
# board_speed_signal / estimate_board_offsets live in capmo.core.sync.board_motion
# (next to the other offset estimators); re-exported for the old names.
from ..core.sync.board_motion import (  # noqa: E402,F401
    _xcorr_lag, board_speed_signal, estimate_board_offsets)


@dataclass
class Event:
    """One instant seen by several cameras: cam index -> view."""
    key: float                   # reference-camera frame index
    members: dict


def match_events(cams: list[CamData], offsets, tol=0.5) -> list[Event]:
    """Group PnP'd views of different cameras showing the same instant."""
    entries = sorted(((v.frame_idx - offsets[c], c, v)
                      for c, cam in enumerate(cams) for v in cam.views if v.rvec is not None),
                     key=lambda e: (e[0], e[1]))
    events = []
    cur = None
    for key, c, v in entries:
        if cur is not None and key - cur.key <= tol + 1e-9 and c not in cur.members:
            cur.members[c] = v
            continue
        if cur is not None and key - cur.key <= tol + 1e-9 and c in cur.members:
            # same camera twice within tol: keep the one nearer the event key
            if abs(key - cur.key) < abs(cur.members[c].frame_idx - offsets[c] - cur.key):
                cur.members[c] = v
            continue
        cur = Event(float(key), {c: v})
        events.append(cur)
    return events


# --------------------------------------------------------------- extrinsics
def _robust_pair(Rs, ts, ws):
    """Robust mean of relative pose samples -> (R, t, inlier mask, spreads)."""
    Rs, ts, ws = np.asarray(Rs), np.asarray(ts), np.asarray(ws, float)
    inl = np.ones(len(Rs), bool)
    for _ in range(3):
        R = sc.mean_rotation(Rs[inl], ws[inl])
        t = np.median(ts[inl], axis=0)
        ang = sc.rotation_angle_deg(np.einsum("ij,njk->nik", R.T, Rs))
        dist = np.linalg.norm(ts - t, axis=1)
        a_thr = max(3.0 * np.median(ang[inl]), 0.5)
        d_thr = max(3.0 * np.median(dist[inl]), 0.01 * max(np.linalg.norm(t), 1e-3))
        new = (ang <= a_thr) & (dist <= d_thr)
        if new.sum() < max(1, len(Rs) // 3) or (new == inl).all():
            break
        inl = new
    R = sc.mean_rotation(Rs[inl], ws[inl])
    t = np.average(ts[inl], axis=0, weights=ws[inl])
    ang = sc.rotation_angle_deg(np.einsum("ij,njk->nik", R.T, Rs[inl]))
    dist = np.linalg.norm(ts[inl] - t, axis=1)
    return R, t, inl, float(np.median(ang)), float(np.median(dist))


def pairwise_poses(events: list[Event], V: int, min_shared=3) -> dict:
    """(i,j) -> relative pose j->i (X_i = R X_j + t) averaged over shared events."""
    samples = {}
    for e in events:
        cs = sorted(e.members)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                i, j = cs[a], cs[b]
                Ri, ti = _Rt(e.members[i])
                Rj, tj = _Rt(e.members[j])
                R = Ri @ Rj.T
                samples.setdefault((i, j), []).append(
                    (R, ti - R @ tj, min(e.members[i].n, e.members[j].n)))
    edges = {}
    for (i, j), ss in samples.items():
        R, t, inl, sa, sd = _robust_pair([s[0] for s in ss], [s[1] for s in ss],
                                         [s[2] for s in ss])
        edges[(i, j)] = {"R": R, "t": t, "n": len(ss), "n_inliers": int(inl.sum()),
                         "spread_deg": sa, "spread_m": sd,
                         "usable": bool(inl.sum() >= min_shared)}
    return edges


def chain_cameras(edges: dict, V: int, ref: int):
    """Poses (ref frame -> camera) along a maximum spanning tree of the graph.

    Returns (poses {cam: (R, t)}, tree [(parent, child)], unreachable [cams]).
    """
    adj = {v: {} for v in range(V)}
    for (i, j), e in edges.items():
        if not e["usable"]:
            continue
        adj[i][j] = (e["n_inliers"], e["R"], e["t"])                 # j -> i
        adj[j][i] = (e["n_inliers"], e["R"].T, -e["R"].T @ e["t"])   # i -> j
    poses = {ref: (np.eye(3), np.zeros(3))}
    tree = []
    while True:
        best = None
        for p in poses:
            for c, (w, _R, _t) in adj[p].items():
                if c not in poses and (best is None or w > best[0]):
                    best = (w, p, c)
        if best is None:
            break
        _, p, c = best
        # adj[p][c] maps c -> p; we need p -> c: X_c = R_cp X_p + t_cp
        _w, R_pc, t_pc = adj[p][c]
        R_cp, t_cp = R_pc.T, -R_pc.T @ t_pc
        Rp, tp = poses[p]
        poses[c] = (R_cp @ Rp, R_cp @ tp + t_cp)
        tree.append((p, c))
    unreachable = [v for v in range(V) if v not in poses]
    return poses, tree, unreachable


# ---------------------------------------------------------- bundle adjust
def bundle_adjust_board(cams: list[CamData], poses: dict, events: list[Event], ref: int, *,
                        refine_intrinsics=False, dist_model="k1k2p", loss="huber",
                        f_scale=1.0, max_nfev=200, outlier_px=3.0, log=print) -> dict:
    """Joint BA of camera poses (+ intrinsics) and per-event board poses."""
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    V = len(cams)
    order = [c for c in range(V) if c in poses]
    others = [c for c in order if c != ref]
    col_of = {c: k for k, c in enumerate(others)}
    free_d = DIST_MODELS[dist_model][1]
    intr_cams = [c for c in order if refine_intrinsics and cams[c].intr_source == "board"]
    icol = {c: k for k, c in enumerate(intr_cams)}
    ni = 4 + len(free_d)
    events = [e for e in events if sum(c in poses for c in e.members) >= 2]
    E = len(events)
    if E == 0:
        raise BoardCalibError("no simultaneous board views between calibrated cameras")

    # observations
    oc, oe, P, uv = [], [], [], []
    for k, e in enumerate(events):
        for c, v in e.members.items():
            if c not in poses:
                continue
            oc.append(np.full(v.n, c))
            oe.append(np.full(v.n, k))
            P.append(v.obj)
            uv.append(v.img)
    oc, oe = np.concatenate(oc), np.concatenate(oe)
    P, uv = np.concatenate(P), np.concatenate(uv)
    # board poses are parametrised about the board centre (well conditioned)
    bc = P.mean(0)
    P = P - bc

    # initial event poses (board -> ref) from the member with the most corners
    ev0 = np.zeros((E, 6))
    for k, e in enumerate(events):
        c = max((c for c in e.members if c in poses), key=lambda c: e.members[c].n)
        Rb, tb = _Rt(e.members[c])
        Rc, tc = poses[c]
        ev0[k, :3] = mv.rotvec(Rc.T @ Rb)
        ev0[k, 3:] = Rc.T @ (tb - tc) + Rc.T @ Rb @ bc
    cam0 = np.concatenate([np.r_[mv.rotvec(poses[c][0]), poses[c][1]] for c in others]) \
        if others else np.zeros(0)
    intr0 = []
    for c in intr_cams:
        K, d = cams[c].K, cams[c].dist
        intr0 += [1.0, K[1, 1] / K[0, 0], K[0, 2] / K[0, 0], K[1, 2] / K[0, 0]] + [d[q] for q in free_d]
    n_cam = 6 * len(others)
    n_ev = 6 * E
    p0 = np.concatenate([cam0, ev0.ravel(), np.array(intr0, float)])

    Kall = np.stack([cams[c].K if c in poses else np.eye(3) for c in range(V)])
    Dall = np.stack([cams[c].dist if c in poses else np.zeros(5) for c in range(V)])

    def unpack(p):
        Rc = np.tile(np.eye(3), (V, 1, 1))
        tc = np.zeros((V, 3))
        for c in others:
            k = col_of[c]
            Rc[c] = mv.rodrigues(p[6 * k:6 * k + 3])
            tc[c] = p[6 * k + 3:6 * k + 6]
        ev = p[n_cam:n_cam + n_ev].reshape(E, 6)
        K = Kall.copy()
        D = Dall.copy()
        for c in intr_cams:
            q = p[n_cam + n_ev + ni * icol[c]:n_cam + n_ev + ni * (icol[c] + 1)]
            K[c, 0, 0], K[c, 1, 1], K[c, 0, 2], K[c, 1, 2] = np.asarray(q[:4]) * Kall[c, 0, 0]
            D[c, free_d] = q[4:]
        return Rc, tc, mv.rodrigues(ev[:, :3]), ev[:, 3:], K, D

    def project(p, sel=slice(None)):
        Rc, tc, Re, te, K, D = unpack(p)
        c, e = oc[sel], oe[sel]
        Xr = np.einsum("nij,nj->ni", Re[e], P[sel]) + te[e]
        Xc = np.einsum("nij,nj->ni", Rc[c], Xr) + tc[c]
        x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        k1, k2, p1, p2, k3 = D[c].T
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([K[c, 0, 0] * xd + K[c, 0, 2], K[c, 1, 1] * yd + K[c, 1, 2]], -1)

    def build_sparsity(mask):
        idx = np.flatnonzero(mask)
        M = len(idx)
        rows2 = np.concatenate([2 * np.arange(M), 2 * np.arange(M) + 1])
        c2 = np.concatenate([oc[idx], oc[idx]])
        e2 = np.concatenate([oe[idx], oe[idx]])
        ri, ci = [], []
        colmap = np.array([col_of.get(c, -1) for c in range(V)])
        cc = colmap[c2]
        m = cc >= 0
        for d in range(6):
            ri.append(rows2[m])
            ci.append(6 * cc[m] + d)
            ri.append(rows2)
            ci.append(n_cam + 6 * e2 + d)
        for c in intr_cams:
            m = c2 == c
            for q in range(ni):
                ri.append(rows2[m])
                ci.append(np.full(m.sum(), n_cam + n_ev + ni * icol[c] + q))
        ri, ci = np.concatenate(ri), np.concatenate(ci)
        return coo_matrix((np.ones(len(ri), np.int8), (ri, ci)),
                          shape=(2 * M, len(p0))).tocsr()

    def per_cam(err, mask):
        out = {}
        for c in order:
            m = mask & (oc == c)
            e = err[m]
            out[c] = {"n_obs": int(m.sum()),
                      "rms_px": float(np.sqrt(np.mean(e ** 2))) if m.any() else float("nan"),
                      "median_px": float(np.median(e)) if m.any() else float("nan")}
        return out

    mask = np.ones(len(oc), bool)
    err0 = np.linalg.norm(project(p0) - uv, axis=1)
    before = per_cam(err0, mask)
    log(f"  BA: {len(order)} cameras, {E} board poses, {len(oc)} corner observations; rms before "
        + " ".join(f"{cams[c].cam_id} {before[c]['rms_px']:.2f}" for c in order))
    p = p0
    nfev = 0
    n_out = 0
    for rnd in range(2):
        def residuals(q, idx=np.flatnonzero(mask)):
            return (project(q, idx) - uv[idx]).ravel()
        sol = least_squares(residuals, p, jac_sparsity=build_sparsity(mask), loss=loss,
                            f_scale=f_scale, x_scale="jac", method="trf", max_nfev=max_nfev,
                            tr_options={"atol": 1e-12, "btol": 1e-12})
        p = sol.x
        nfev += sol.nfev
        err = np.linalg.norm(project(p) - uv, axis=1)
        thr = max(outlier_px, 5 * float(np.median(err[mask])))
        new = err <= thr
        if rnd == 0 and (~new & mask).any():
            n_out = int((~new).sum())
            mask = new
            continue
        break
    after = per_cam(err, mask)
    Rc, tc, Re, te, K, D = unpack(p)
    te = te - np.einsum("nij,j->ni", Re, bc)      # back to the board-origin convention
    log(f"  BA done ({nfev} evals, {n_out} outlier corners dropped): rms after "
        + " ".join(f"{cams[c].cam_id} {after[c]['rms_px']:.2f}" for c in order))
    return {"R": {c: Rc[c] for c in order}, "t": {c: tc[c] for c in order},
            "K": {c: K[c] for c in order}, "dist": {c: D[c] for c in order},
            "events": events, "Re": Re, "te": te, "before": before, "after": after,
            "rms_all_px": float(np.sqrt(np.mean(err[mask] ** 2))), "n_obs": int(mask.sum()),
            "n_outliers": n_out, "nfev": nfev, "intrinsics_refined": [cams[c].cam_id for c in intr_cams]}


# --------------------------------------------------------------- overlays
def render_overlays(cams, ba, board_cfg, corners3d, out_dir, world_key, n=4, log=print):
    from .extract_keypoints import open_video
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for c, cam in enumerate(cams):
        if not cam.video or not os.path.exists(cam.video) or c not in ba["R"]:
            continue
        evs = [(k, e) for k, e in enumerate(ba["events"]) if c in e.members]
        if not evs:
            continue
        pick = [evs[i] for i in sorted(set(np.linspace(0, len(evs) - 1, n).astype(int)))]
        want = {e.members[c].frame_idx: (k, e) for k, e in pick}
        t0 = min(e.members[c].t for _k, e in pick)
        src = open_video(cam.video)
        try:
            for idx, _t, img in src.frames(max(0.0, t0 - 1.0)):   # t0: on the npz t clock, as frames()
                if idx > max(want):
                    break
                if idx not in want:
                    continue
                k, e = want[idx]
                v = e.members[c]
                R = ba["R"][c] @ ba["Re"][k]
                t = ba["R"][c] @ ba["te"][k] + ba["t"][c]
                pr, _ = cv2.projectPoints(corners3d, mv.rotvec(R), t, ba["K"][c], ba["dist"][c])
                pr = pr.reshape(-1, 2)
                for q in pr:
                    if np.isfinite(q).all():
                        cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 5, (0, 0, 255), 1,
                                   cv2.LINE_AA)
                for q in v.img:
                    x, y = int(round(q[0])), int(round(q[1]))
                    cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
                err = np.linalg.norm(pr[v.ids] - v.img, axis=1)
                cv2.putText(img, f"{cam.cam_id} frame {idx}: {v.n} corners, rms "
                            f"{np.sqrt(np.mean(err ** 2)):.2f}px"
                            + (" [world frame]" if e.key == world_key else ""),
                            (10, 28), 0, 0.7, (0, 255, 255), 2)
                fn = os.path.join(out_dir, f"board_{cam.cam_id}_f{idx:06d}.jpg")
                cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                written.append(fn)
        finally:
            src.close()
    log(f"  wrote {len(written)} overlay image(s) to {out_dir}")
    return written


# ------------------------------------------------------------- fallback
def selfcal_fallback(pose_dir, cam_ids, world_cams: dict, *, start=None, end=None, hfov=90.0,
                     log=print):
    """Body self-calibration of all cameras, anchored to the board cameras.

    *world_cams*: cam_id -> mv.Camera in the board world frame (>= 2 needed).
    Returns (cam_id -> mv.Camera in the world frame, info).
    """
    from . import reconstruct_take as rt
    if len(world_cams) < 2:
        raise BoardCalibError("--fallback-selfcal needs >= 2 board-calibrated cameras to "
                              "anchor the body self-calibration")
    take = sc.load_take(pose_dir, cam_ids)
    take = sc.trim_take(take, start, end)
    sync = sc.estimate_offsets(take)
    shift = np.where(np.abs(sync["offsets"]) >= 0.75, np.round(sync["offsets"]), 0).astype(int)
    for v in range(take.V):
        if shift[v]:
            take.kp[v], take.vis[v] = sc.shift_track(take.kp[v], take.vis[v], float(shift[v]))
    cams_s, _ = rt.calibrate(take, hfov=hfov, log=log)
    anchor = [c for c in cam_ids if c in world_cams]
    ia = [cam_ids.index(c) for c in anchor]
    sim = sc.camera_similarity([cams_s[i] for i in ia], [world_cams[c] for c in anchor])
    cams_w = sim.apply_cameras(cams_s)
    resid = sc.compare_cameras([cams_w[i] for i in ia], [world_cams[c] for c in anchor], anchor)
    log("  selfcal anchored to the board cameras: scale %.4f; residuals " % sim.s + ", ".join(
        f"{c} {d['center_diff_m'] * 100:.1f} cm / {d['rotation_diff_deg']:.2f} deg"
        for c, d in resid["cameras"].items()))
    return dict(zip(cam_ids, cams_w)), {"scale": sim.s, "anchor_cams": anchor,
                                        "anchor_residuals": resid["cameras"],
                                        "sync_shift_frames": shift.tolist()}


# ------------------------------------------------------------------ driver
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def run(take_dir, out, *, start=None, end=None, cam_ids=None, ref_cam=None, min_corners=8,
        min_views=10, max_views=40, dist_model="k1k2p", fix_aspect=False, fix_pp=False,
        intrinsics_files=None, offsets=None, estimate_offsets=False, max_lag=10, sync_tol=0.5,
        min_shared=3, max_events=400, pnp_max_px=3.0, ba_intrinsics=False, ba_loss="huber",
        ba_f_scale=1.0, max_nfev=200, world_frame=None, report_path=None, overlay_dir=None,
        videos=None, fallback_selfcal=False, pose_take=None, pose_start=None, pose_end=None,
        save_intrinsics=None, log=print) -> dict:
    """Run the whole board calibration; returns the report dict (raises BoardCalibError)."""
    report = {"take": take_dir, "window": {"start": start, "end": end}, "warnings": [],
              "errors": [], "cameras": {}}

    def warn(msg):
        report["warnings"].append(msg)
        log(f"  WARNING: {msg}")

    if not offsets and not estimate_offsets and _is_session_take(take_dir):
        # each camera of a capmo session counts its own frames from its own
        # start; the common host clock (t) gives the frame-index offsets
        from ..core.sync.timeline import session_take_offsets
        offsets = session_take_offsets(take_dir)
        if cam_ids:                       # only the loaded cameras (re-referenced in _run)
            offsets = {c: o for c, o in offsets.items() if c in cam_ids}
        report["offsets_source"] = "session clock"
        log("  session take: frame offsets from the common clock: " + ", ".join(
            f"{c} {o:+.2f}" for c, o in offsets.items()))
    try:
        _run(report, warn, take_dir, out, start, end, cam_ids, ref_cam, min_corners, min_views,
             max_views, dist_model, fix_aspect, fix_pp, intrinsics_files or {}, offsets or {},
             estimate_offsets, max_lag, sync_tol, min_shared, max_events, pnp_max_px,
             ba_intrinsics, ba_loss, ba_f_scale, max_nfev, world_frame, overlay_dir,
             videos or {}, fallback_selfcal, pose_take, pose_start, pose_end, save_intrinsics,
             log)
    except BoardCalibError as exc:
        report["errors"].append(str(exc))
        report["ok"] = False
        if report_path:
            _write_json(report_path, report)
        raise
    report["ok"] = True
    if report_path:
        _write_json(report_path, report)
        log(f"report: {report_path}")
    return report


def _is_session_take(take_dir) -> bool:
    try:
        with open(os.path.join(take_dir, "meta.json"), encoding="utf-8") as f:
            return json.load(f).get("timebase") == "session"
    except (OSError, ValueError):
        return False


def _write_json(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(d), f, indent=1)


def _run(report, warn, take_dir, out, start, end, cam_ids, ref_cam, min_corners, min_views,
         max_views, dist_model, fix_aspect, fix_pp, intr_files, offsets_in, estimate_offsets,
         max_lag, sync_tol, min_shared, max_events, pnp_max_px, ba_intrinsics, ba_loss,
         ba_f_scale, max_nfev, world_frame, overlay_dir, videos, fallback, pose_take,
         pose_start, pose_end, save_intrinsics, log):
    log(f"board calibration of {take_dir}")
    cams, board_cfg = load_board_take(take_dir, cam_ids, start, end, min_corners, log=log)
    corners3d = np.asarray(board_cfg.board().getChessboardCorners(), np.float64)
    board_center = np.array([board_cfg.squares_x * board_cfg.square_len / 2,
                             board_cfg.squares_y * board_cfg.square_len / 2, 0.0])
    ids = [c.cam_id for c in cams]
    for k in list(intr_files) + list(offsets_in) + list(videos):
        if k not in ids:
            raise BoardCalibError(f"unknown camera {k!r} (cameras: {ids})")
    for c in cams:
        if c.cam_id in videos:
            c.video = videos[c.cam_id]
    ref = ids.index(ref_cam) if ref_cam else 0
    if ref_cam and ref_cam not in ids:
        raise BoardCalibError(f"--ref-cam {ref_cam} not among {ids}")
    report.update(board=board_cfg.to_dict(), cam_ids=ids, ref_cam=ids[ref],
                  dist_model=dist_model)
    no_views = [c.cam_id for c in cams if not c.views and c.cam_id not in intr_files]

    # 1) intrinsics
    log("intrinsics:")
    failed = []
    for c in cams:
        rc = report["cameras"].setdefault(c.cam_id, {})
        rc.update(n_detections=c.n_raw, n_views_usable=len(c.views), image_size=list(c.size))
        if c.cam_id in intr_files:
            K, dist, size = load_intrinsics(intr_files[c.cam_id], c.cam_id)
            if size and tuple(size) != tuple(c.size):
                raise BoardCalibError(f"camera {c.cam_id}: intrinsics file is for {size}, "
                                      f"video is {c.size}")
            c.K, c.dist, c.intr_source = K, dist, "file"
            rc.update(intrinsics_source="file", intrinsics_file=intr_files[c.cam_id])
            log(f"  {c.cam_id}: intrinsics from {intr_files[c.cam_id]}")
            continue
        if len(c.views) < 3:
            failed.append(c.cam_id)
            continue
        info = calibrate_intrinsics(c, dist_model=dist_model, max_views=max_views,
                                    min_views=min_views, fix_aspect=fix_aspect, fix_pp=fix_pp,
                                    log=log)
        c.K, c.dist, c.intr_source, c.intr = info["K"], np.asarray(info["dist"], float), "board", info
        rc.update(intrinsics_source="board", intrinsic_rms_px=info["rms_px"],
                  n_views_intrinsics=info["n_views_used"],
                  rejected_view_frames=info["rejected_view_frames"],
                  per_view_rms_px=info["per_view_rms_px"], coverage=info["coverage"],
                  tilt_deg=info["tilt_deg"])
        if len(c.views) < min_views:
            warn(f"{c.cam_id}: only {len(c.views)} usable board views (< --min-views "
                 f"{min_views}); intrinsics may be poor")
        if info["rms_px"] > WARN_INTRINSIC_RMS_PX:
            warn(f"{c.cam_id}: high intrinsic RMS {info['rms_px']:.2f}px (> "
                 f"{WARN_INTRINSIC_RMS_PX}): blur, rolling shutter, a bent board or a wrong "
                 "board size")
        if info["coverage"] < WARN_COVERAGE:
            warn(f"{c.cam_id}: poor image coverage {info['coverage']:.0%} (< "
                 f"{WARN_COVERAGE:.0%} of the image cells saw a corner): move the board into "
                 "the image corners/edges (distortion and the principal point are poorly "
                 "determined), or use --fix-principal-point and/or --ba-intrinsics")
        if info["tilt_deg"]["max"] < WARN_MAX_TILT_DEG:
            warn(f"{c.cam_id}: little board tilt (max {info['tilt_deg']['max']:.0f} deg): tilt "
                 "the board 20-45 deg towards the camera, or the focal length is weakly "
                 "determined")
        if save_intrinsics:
            os.makedirs(save_intrinsics, exist_ok=True)
            _write_json(os.path.join(save_intrinsics, f"{c.cam_id}.K.json"),
                        {"cam_id": c.cam_id, "image_size": list(c.size), "K": c.K,
                         "dist": c.dist, "rms_px": info["rms_px"]})
    if failed and not fallback:
        what = ("never sees the board" if set(failed) <= set(no_views)
                else "has too few board views (< 3)")
        raise BoardCalibError(
            f"camera(s) {failed} {what} in the window: cannot calibrate it from the board. "
            "Show the board to every camera, or pass --fallback-selfcal (body "
            "self-calibration anchored to the board cameras), or skip --rig and let "
            "reconstruct_take self-calibrate from the body.")

    # 2) PnP, sync, events
    for c in cams:
        if c.K is not None:
            n = solve_view_poses(c, pnp_max_px)
            report["cameras"][c.cam_id]["n_views_pnp"] = n
    offsets = np.zeros(len(cams))
    if estimate_offsets:
        log("sync: board-motion cross-correlation ...")
        est = estimate_board_offsets(cams, board_center, ref, max_lag=max_lag, log=log)
        offsets = est["offsets"].astype(float)
        report["offsets_estimated"] = {
            "offsets_frames": dict(zip(ids, est["offsets"].tolist())),
            "unresolved": [ids[v] for v in est["unresolved"]],
            "pairs": {f"{ids[i]}-{ids[j]}": p for (i, j), p in est["pairs"].items()}}
        for v in est["unresolved"]:
            if cams[v].K is not None:
                warn(f"{ids[v]}: offset not resolved from the board motion (no overlap or weak "
                     "correlation); assuming 0")
    for k, o in offsets_in.items():
        offsets[ids.index(k)] = o
    if offsets_in:
        offsets = offsets - offsets[ref]
    report["offsets_frames"] = dict(zip(ids, offsets.tolist()))
    events = match_events(cams, offsets, sync_tol)
    multi = [e for e in events if len(e.members) >= 2]
    if len(multi) > max_events:
        keep = np.unique(np.linspace(0, len(multi) - 1, max_events).round().astype(int))
        multi = [multi[i] for i in keep]
    log(f"sync: {len(events)} board instants, {len(multi)} seen by >= 2 cameras"
        + (f" (offsets {dict(zip(ids, offsets.tolist()))})" if np.any(offsets) else ""))
    for c, cam in enumerate(cams):
        report["cameras"][cam.cam_id]["n_simultaneous_views"] = sum(c in e.members for e in multi)

    # 3) camera graph
    edges = pairwise_poses(multi, len(cams), min_shared)
    poses, tree, unreachable = chain_cameras(edges, len(cams), ref)
    unreachable = [v for v in unreachable if cams[v].K is not None] + \
        [ids.index(f) for f in failed]
    report["graph"] = {
        "edges": {f"{ids[i]}-{ids[j]}": {"shared_views": e["n"], "inliers": e["n_inliers"],
                                         "spread_deg": e["spread_deg"], "spread_m": e["spread_m"],
                                         "usable": e["usable"]} for (i, j), e in edges.items()},
        "spanning_tree": [f"{ids[p]}->{ids[c]}" for p, c in tree],
        "connected": [ids[v] for v in sorted(poses)],
        "not_connected": sorted(ids[v] for v in set(unreachable))}
    log("camera graph: " + ", ".join(
        f"{ids[i]}-{ids[j]} {e['n_inliers']}/{e['n']}" for (i, j), e in sorted(edges.items()))
        + "; tree " + " ".join(f"{ids[p]}->{ids[c]}" for p, c in tree))
    if unreachable and not fallback:
        bad = sorted(ids[v] for v in set(unreachable))
        raise BoardCalibError(
            f"camera(s) {bad} not linked to {ids[ref]}: they never see the board at the same "
            f"time as a linked camera (>= {min_shared} simultaneous views needed). Hold the board "
            "where they and a neighbour both see it, check --offsets/--estimate-offsets, or pass "
            "--fallback-selfcal.")
    if unreachable:
        warn(f"camera(s) {sorted(ids[v] for v in set(unreachable))} not linked by the board: "
             "using the body self-calibration fallback for them")
    for (i, j), e in edges.items():
        if e["usable"] and e["spread_deg"] > 1.0:
            warn(f"{ids[i]}-{ids[j]}: relative pose samples disagree (median {e['spread_deg']:.2f} "
                 "deg): moving board with unsynchronised cameras? Hold the board still briefly "
                 "or set --offsets")

    # 4) bundle adjustment
    log("bundle adjustment:")
    ba = bundle_adjust_board(cams, poses, multi, ref, refine_intrinsics=ba_intrinsics,
                             dist_model=dist_model, loss=ba_loss, f_scale=ba_f_scale,
                             max_nfev=max_nfev, log=log)
    report["ba"] = {"rms_px": ba["rms_all_px"], "n_obs": ba["n_obs"], "n_board_poses": len(ba["events"]),
                    "outlier_corners_dropped": ba["n_outliers"], "nfev": ba["nfev"],
                    "intrinsics_refined": ba["intrinsics_refined"],
                    "per_camera_before": {ids[c]: d for c, d in ba["before"].items()},
                    "per_camera": {ids[c]: d for c, d in ba["after"].items()}}
    for c, d in ba["after"].items():
        report["cameras"][ids[c]].update(ba_rms_px=d["rms_px"], ba_median_px=d["median_px"],
                                         ba_n_obs=d["n_obs"])
        if d["rms_px"] > WARN_BA_RMS_PX:
            warn(f"{ids[c]}: high BA RMS {d['rms_px']:.2f}px (> {WARN_BA_RMS_PX}): check sync "
                 "(offsets), motion blur, or a wrong intrinsic model")

    # 5) world frame = board at one event
    evs = ba["events"]
    if world_frame is not None:
        we = int(np.argmin([abs(e.key - world_frame) for e in evs]))
    else:
        we = max(range(len(evs)), key=lambda k: (len(evs[k].members),
                                                 sum(v.n for v in evs[k].members.values())))
    Rw, tw = ba["Re"][we], ba["te"][we]       # board(world) -> ref camera
    world = {}
    for c in ba["R"]:
        R = ba["R"][c] @ Rw
        world[ids[c]] = mv.Camera(ba["K"][c], ba["dist"][c], R, ba["R"][c] @ tw + ba["t"][c])
    report["world_frame"] = {
        "definition": ("board frame at the reference instant: origin at the board's outer "
                       "corner next to ChArUco corner 0, +X along squares_x, +Y along squares_y "
                       "(down the printed pattern), +Z into the board (away from the cameras), "
                       "metres"),
        "ref_cam_frame_idx": evs[we].key,
        "t": evs[we].members[min(evs[we].members)].t,
        "cameras_seeing_it": [ids[c] for c in sorted(evs[we].members)]}
    log(f"world frame: board at {ids[ref]} frame {evs[we].key:.0f} (seen by "
        f"{', '.join(ids[c] for c in sorted(evs[we].members))})")

    # 6) fallback for cameras the board could not place
    if unreachable:
        missing = sorted(ids[v] for v in set(unreachable))
        log(f"fallback: body self-calibration for {missing} ...")
        pose_dir = pose_take or take_dir
        try:
            allc, finfo = selfcal_fallback(pose_dir, ids, world, start=pose_start, end=pose_end,
                                           log=log)
        except (OSError, KeyError) as exc:
            raise BoardCalibError(f"--fallback-selfcal: cannot load pose keypoints from "
                                  f"{pose_dir} ({exc}); pass --pose-take") from exc
        for m in missing:
            world[m] = allc[m]
            report["cameras"][m]["intrinsics_source"] = "selfcal"
        report["fallback_selfcal"] = dict(finfo, cameras=missing)

    rig = RigCalibration(board=board_cfg)
    for cid in ids:
        if cid not in world:
            continue
        cam = world[cid]
        size = next(c.size for c in cams if c.cam_id == cid)
        rig.cameras[cid] = CameraCalibration(cid, tuple(size), cam.K, cam.dist, mv.rotvec(cam.R),
                                             cam.t)
        report["cameras"][cid].update(K=cam.K, dist=cam.dist, center_m=cam.center)
    d = rig.to_dict()
    d["board_calibration"] = {"tool": "capmo.tools.calibrate_board", "take": take_dir,
                              "window": [start, end], "ref_cam": ids[ref],
                              "world_frame": report["world_frame"],
                              "offsets_frames": report["offsets_frames"],
                              "ba_rms_px": ba["rms_all_px"],
                              "fallback_selfcal": report.get("fallback_selfcal", {}).get("cameras", [])}
    if out:
        _write_json(out, d)
        log(f"wrote {out}")
    report["rig"] = out
    log(describe_rig(rig))
    if overlay_dir:
        report["overlays"] = render_overlays(cams, ba, board_cfg, corners3d, overlay_dir,
                                             evs[we].key, log=log)
    report["summary"] = {cid: {k: report["cameras"][cid].get(k) for k in
                               ("intrinsic_rms_px", "n_views_intrinsics", "ba_rms_px",
                                "n_simultaneous_views", "intrinsics_source")} for cid in ids}
    return report


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capmo.tools.calibrate_board",
                                description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("take_dir", help="folder with <cam>.calib.npz (extract_keypoints --calib)")
    p.add_argument("--out", required=True, help="rig.json to write")
    p.add_argument("--report", default=None, help="JSON report to write")
    p.add_argument("--start", type=float, default=None, help="use detections at t >= START s")
    p.add_argument("--end", type=float, default=None, help="use detections at t <= END s")
    p.add_argument("--cams", default=None, help="comma-separated camera subset/order")
    p.add_argument("--ref-cam", default=None, help="reference camera (default: the first)")
    g = p.add_argument_group("intrinsics")
    g.add_argument("--min-corners", type=int, default=8, help="corners per usable view")
    g.add_argument("--min-views", type=int, default=10, help="warn below this many views")
    g.add_argument("--max-views", type=int, default=40, help="views per intrinsic solve")
    g.add_argument("--dist", choices=sorted(DIST_MODELS), default="k1k2p",
                   help="distortion model: none, k1, k1k2, k1k2p (k1,k2,p1,p2; default), "
                        "k1k2pk3")
    g.add_argument("--fix-aspect", action="store_true", help="fx = fy")
    g.add_argument("--fix-principal-point", action="store_true",
                   help="principal point at the image centre")
    g.add_argument("--intrinsics-only", action="append", default=[], metavar="CAM=FILE",
                   help="take CAM's intrinsics from FILE (a {K, dist} JSON or a rig.json); "
                        "only its pose is solved. Repeatable / comma-separated")
    g.add_argument("--save-intrinsics", default=None, metavar="DIR",
                   help="write DIR/<cam>.K.json for reuse with --intrinsics-only")
    g = p.add_argument_group("sync / extrinsics")
    g.add_argument("--offsets", action="append", default=[], metavar="CAM=FRAMES",
                   help="known frame offsets: CAM's frame k+FRAMES shows the reference "
                        "camera's frame k (e.g. camB=+2,camC=-1)")
    g.add_argument("--estimate-offsets", action="store_true",
                   help="estimate the offsets from the board motion (dense sampling needed)")
    g.add_argument("--max-lag", type=int, default=10, help="offset search range (frames)")
    g.add_argument("--sync-tol", type=float, default=0.5,
                   help="max frame difference for simultaneous views (after offsets)")
    g.add_argument("--min-shared", type=int, default=3,
                   help="simultaneous views needed to link two cameras")
    g.add_argument("--max-events", type=int, default=400, help="cap on board poses in the BA")
    g.add_argument("--pnp-max-px", type=float, default=3.0,
                   help="drop views whose single-view PnP RMS exceeds this")
    g.add_argument("--world-frame", type=float, default=None, metavar="FRAME",
                   help="reference-camera frame whose board pose is the world frame")
    g = p.add_argument_group("bundle adjustment")
    g.add_argument("--ba-intrinsics", action="store_true",
                   help="also refine the board-solved intrinsics in the BA")
    g.add_argument("--ba-loss", default="huber", choices=["linear", "huber", "soft_l1", "cauchy"])
    g.add_argument("--ba-f-scale", type=float, default=1.0, help="robust loss scale (px)")
    g.add_argument("--max-nfev", type=int, default=200)
    g = p.add_argument_group("output / fallback")
    g.add_argument("--overlay-dir", default=None, help="write reprojection overlay JPEGs here")
    g.add_argument("--video", action="append", default=[], metavar="CAM=PATH",
                   help="video for the overlays (default: the path in meta.json)")
    g.add_argument("--fallback-selfcal", action="store_true",
                   help="place cameras the board cannot by body self-calibration, anchored to "
                        "the board cameras")
    g.add_argument("--pose-take", default=None,
                   help="pose keypoint folder for --fallback-selfcal (default: TAKE_DIR)")
    g.add_argument("--pose-start", type=float, default=None,
                   help="--fallback-selfcal: use pose frames at t >= this (s)")
    g.add_argument("--pose-end", type=float, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        run(a.take_dir, a.out, start=a.start, end=a.end,
            cam_ids=[c.strip() for c in a.cams.split(",")] if a.cams else None,
            ref_cam=a.ref_cam, min_corners=a.min_corners, min_views=a.min_views,
            max_views=a.max_views, dist_model=a.dist, fix_aspect=a.fix_aspect,
            fix_pp=a.fix_principal_point, intrinsics_files=_parse_map(a.intrinsics_only),
            offsets=_parse_map(a.offsets, float), estimate_offsets=a.estimate_offsets,
            max_lag=a.max_lag, sync_tol=a.sync_tol, min_shared=a.min_shared,
            max_events=a.max_events, pnp_max_px=a.pnp_max_px, ba_intrinsics=a.ba_intrinsics,
            ba_loss=a.ba_loss, ba_f_scale=a.ba_f_scale, max_nfev=a.max_nfev,
            world_frame=a.world_frame, report_path=a.report, overlay_dir=a.overlay_dir,
            videos=_parse_map(a.video), fallback_selfcal=a.fallback_selfcal,
            pose_take=a.pose_take, pose_start=a.pose_start, pose_end=a.pose_end,
            save_intrinsics=a.save_intrinsics)
    except BoardCalibError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
