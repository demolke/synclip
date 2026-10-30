"""
Self-calibration of a multi-camera rig from human body keypoints alone.

No board needed: the performer *is* the calibration target. Pipeline:

1. :func:`estimate_offsets` -- per-camera time offset by scanning shifts of the
   keypoint tracks and minimising the median Sampson (epipolar) error of a
   RANSAC fundamental matrix fitted per shift (calibration independent).
2. :func:`initial_cameras` -- guessed intrinsics (focal from a FOV guess,
   principal point at the centre, no distortion); the best-covisible pair's
   essential matrix gives the first relative pose, the other cameras are
   registered by PnP-RANSAC against the growing triangulated point cloud.
3. :func:`bundle_adjust` -- scipy ``least_squares`` (soft-L1 loss, sparse
   Jacobian) over extrinsics, focal(s), radial ``k1`` and the 3D points. The
   gauge is fixed by the reference camera (pose) and its mean baseline (scale);
   the metric scale is set afterwards from a body prior.
4. :func:`world_alignment` -- metric scale from a human bone-length prior,
   floor plane fitted to the lowest foot landmarks (``z = 0``), ``+Z`` up,
   origin under the mean pelvis, and the performer's average facing along
   ``-Y`` (Blender "front" convention), right-handed.

Cameras are :class:`capmo.multiview.Camera`; :func:`to_rig` / :func:`from_rig`
convert to/from :class:`capmo.calibration.RigCalibration` (``rig.json``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .multiview import (Camera, NUM_JOINTS, quiet, robust_triangulate, rodrigues,
                        rotvec, triangulate)

# Joints that are stable enough to drive calibration (face centre, ears,
# shoulders..wrists, hips..feet). Fingers and the inner/outer eye points are
# left out (tiny, noisy).
CALIB_JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


# ------------------------------------------------------------------- loading
@dataclass
class Take:
    cam_ids: list[str]
    kp: np.ndarray          # (V,T,33,2) pixels, nan where not detected
    vis: np.ndarray         # (V,T,33), 0 where not detected
    t: np.ndarray           # (T,) seconds
    fps: float
    size: tuple[int, int]   # (w, h)
    path: str = ""
    frame_idx: np.ndarray | None = None   # (V,T) video frame index of each row
    first_idx: np.ndarray | None = None   # (V,) first decoded frame of each npz/proxy
    # per-camera timelines (see load_take): "shared" (all cameras use t, the
    # legacy behaviour) or "resampled" (each camera's own t resampled onto t)
    timeline: str = "shared"
    cam_t: list | None = None             # per-camera source times (s), when known
    sync_prior: np.ndarray | None = None  # (V,) prior offsets (frames) for estimate_offsets
    session: dict | None = None           # meta.json "session" block (capmo sessions)

    @property
    def V(self) -> int:
        return self.kp.shape[0]

    @property
    def T(self) -> int:
        return self.kp.shape[1]


def resample_track(t_src, kp, vis, det, fidx, grid, fps, max_gap_frames: float = 2.5):
    """Resample one camera's 2D track onto *grid* (seconds).

    Linear interpolation between the two neighbouring samples when both are
    detected and at most *max_gap_frames* apart (per joint: nan stays nan, vis =
    the smaller one); otherwise the nearest sample if within half a frame and
    detected; else not detected. Returns ``(kp, vis, frame_idx)`` on the grid
    (frame_idx = the nearest source frame, -1 outside the source span).
    """
    t_src = np.asarray(t_src, np.float64)
    G = len(grid)
    out_kp = np.full((G,) + kp.shape[1:], np.nan)
    out_vis = np.zeros((G,) + vis.shape[1:])
    out_fi = np.full(G, -1, np.int64)
    if len(t_src) == 0:
        return out_kp, out_vis, out_fi
    P = 1.0 / fps
    j = np.searchsorted(t_src, grid)
    i0 = np.clip(j - 1, 0, len(t_src) - 1)
    i1 = np.clip(j, 0, len(t_src) - 1)
    near = np.where(np.abs(t_src[i0] - grid) <= np.abs(t_src[i1] - grid), i0, i1)
    inside = (grid >= t_src[0] - 0.5 * P) & (grid <= t_src[-1] + 0.5 * P)
    out_fi[inside] = fidx[near[inside]]
    gap = t_src[i1] - t_src[i0]
    lin = ((j > 0) & (j < len(t_src)) & (gap > 0) & (gap <= max_gap_frames * P + 1e-9)
           & det[i0] & det[i1])
    a = np.where(lin, (grid - t_src[i0]) / np.where(gap > 0, gap, 1.0), 0.0)
    kp_l = (1 - a)[:, None, None] * kp[i0] + a[:, None, None] * kp[i1]
    vis_l = np.minimum(vis[i0], vis[i1])
    exact = np.abs(t_src[near] - grid) <= 1e-9
    nn = (~lin | exact) & inside & (np.abs(t_src[near] - grid) <= 0.5 * P + 1e-9) & det[near]
    out_kp[lin], out_vis[lin] = kp_l[lin], vis_l[lin]
    out_kp[nn], out_vis[nn] = kp[near[nn]], vis[near[nn]]
    out_vis[~np.isfinite(out_kp[..., 0])] = 0.0
    return out_kp, out_vis, out_fi


def load_take(take_dir: str, cam_ids: list[str] | None = None, timeline: str = "auto") -> Take:
    """Load ``<cam>.npz`` keypoint files written by ``extract_keypoints``.

    *timeline*:

    * ``"first"`` -- legacy: rows are frame numbers, truncated to the shortest
      camera, the first camera's ``t`` for all.
    * ``"resample"`` -- each camera's own ``t`` is resampled
      (:func:`resample_track`) onto a common grid at the nominal fps, anchored
      on the first frame of the first camera that has frames and spanning all
      cameras.
    * ``"auto"`` (default) -- ``"resample"`` for takes extracted from a capmo
      session (``meta.json`` ``timebase == "session"``: per-frame host-clock
      times) or when the cameras' ``t`` differ by more than half a frame over
      their common rows; otherwise ``"first"`` (unchanged behaviour for
      ordinary videos).

    For a session take ``sync_prior`` holds the clock-based offsets (frames)
    for :func:`estimate_offsets`: zero after resampling (the clocks already
    aligned the tracks), the start-time differences with ``"first"``.
    """
    meta = {}
    mp = os.path.join(take_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    if cam_ids is None:
        if meta.get("videos"):
            cam_ids = [v["cam_id"] for v in meta["videos"]]
        else:
            cam_ids = sorted(f[:-4] for f in os.listdir(take_dir)
                             if f.endswith(".npz") and ".calib" not in f)
    data = [np.load(os.path.join(take_dir, f"{c}.npz"), allow_pickle=False) for c in cam_ids]
    ts = [d["t"].astype(np.float64) for d in data]
    T = min(len(t) for t in ts)
    fps = float(data[0]["fps"])
    session_timed = meta.get("timebase") == "session"
    if timeline == "auto":
        differ = bool(T) and any(np.abs(t[:T] - ts[0][:T]).max() > 0.5 / fps for t in ts[1:])
        timeline = "resample" if (session_timed or differ) else "first"
    if timeline not in ("first", "resample"):
        raise ValueError(f"timeline must be auto|first|resample, got {timeline!r}")
    size = (int(data[0]["width"]), int(data[0]["height"]))
    first = np.array([int(d["window_first_frame_idx"]) if "window_first_frame_idx" in d.files
                      else 0 for d in data], np.int64)
    session = meta.get("session")

    def arrays(d, n):
        kp = d["kp"][:n].astype(np.float64)
        vis = d["vis"][:n].astype(np.float64)
        det = (d["detected"][:n].astype(bool) if "detected" in d.files
               else np.isfinite(d["kp"][:n, 0, 0]))
        kp[~det] = np.nan
        vis[~det] = 0.0
        fi = d["frame_idx"][:n].astype(np.int64) if "frame_idx" in d.files else np.arange(n)
        return kp, np.nan_to_num(vis), det, fi

    if timeline == "first":
        parts = [arrays(d, T) for d in data]
        prior = None
        if session_timed and T:
            prior = (ts[0][0] - np.array([t[0] for t in ts])) * fps
        return Take(list(cam_ids), np.stack([p[0] for p in parts]),
                    np.stack([p[1] for p in parts]), ts[0][:T].copy(), fps, size, take_dir,
                    np.stack([p[3] for p in parts]), first, "shared",
                    [t.copy() for t in ts], prior, session)
    nominal = float((session or {}).get("nominal_fps") or fps)
    nonempty = [t for t in ts if len(t)]
    if not nonempty:
        raise ValueError(f"{take_dir}: no camera has any frame ({', '.join(cam_ids)})")
    lo = min(t[0] for t in nonempty)
    hi = max(t[-1] for t in nonempty)
    t_ref = nonempty[0][0]                # the first camera with frames anchors the grid
    k0 = int(np.ceil((lo - t_ref) * nominal - 1e-6))
    k1 = int(np.floor((hi - t_ref) * nominal + 1e-6))
    grid = t_ref + np.arange(k0, k1 + 1) / nominal
    kps, viss, fis = [], [], []
    for d, t in zip(data, ts):
        kp, vis, det, fi = arrays(d, len(t))
        a, b, c = resample_track(t, kp, vis, det, fi, grid, nominal)
        kps.append(a)
        viss.append(b)
        fis.append(c)
    return Take(list(cam_ids), np.stack(kps), np.stack(viss), grid, nominal, size, take_dir,
                np.stack(fis), first, "resampled", [t.copy() for t in ts],
                np.zeros(len(cam_ids)) if session_timed else None, session)


def trim_take(take: Take, start: float | None = None, end: float | None = None) -> Take:
    """Rows of *take* whose time ``t`` (video seconds) lies in ``[start, end]``.

    Everything downstream (sync scan, calibration, BA stride, outputs) then only
    sees the kept rows. ``frame_idx`` keeps the mapping to the video frames.
    """
    if start is None and end is None:
        return take
    eps = 0.25 / take.fps if take.fps else 1e-6
    keep = np.ones(take.T, bool)
    if start is not None:
        keep &= take.t >= start - eps
    if end is not None:
        keep &= take.t <= end + eps
    rows = np.flatnonzero(keep)
    if len(rows) < 2:
        raise ValueError(f"--start/--end keep {len(rows)} frames of {take.T} "
                         f"(take covers t={take.t[0]:.2f}..{take.t[-1]:.2f} s)")
    sl = slice(rows[0], rows[-1] + 1)
    fidx = take.frame_idx if take.frame_idx is not None else np.tile(np.arange(take.T), (take.V, 1))
    return dataclasses.replace(take, cam_ids=list(take.cam_ids), kp=take.kp[:, sl].copy(),
                               vis=take.vis[:, sl].copy(), t=take.t[sl].copy(),
                               frame_idx=fidx[:, sl].copy())


def shift_track(kp: np.ndarray, vis: np.ndarray, delta: float):
    """Resample one camera's track at ``frame + delta`` (linear interpolation).

    *kp* (T,33,2), *vis* (T,33). Samples whose two neighbour frames are not both
    detected become nan / 0.
    """
    T = kp.shape[0]
    pos = np.arange(T) + delta
    i0 = np.floor(pos).astype(int)
    a = (pos - i0)[:, None]
    i1 = i0 + (a[:, 0] > 1e-9)
    ok = (i0 >= 0) & (i1 < T)
    i0c, i1c = np.clip(i0, 0, T - 1), np.clip(i1, 0, T - 1)
    out = (1 - a[..., None]) * kp[i0c] + a[..., None] * kp[i1c]
    v = np.minimum(vis[i0c], vis[i1c])
    out[~ok] = np.nan
    v[~ok] = 0
    v[~np.isfinite(out[..., 0])] = 0
    return out, v


# ---------------------------------------------------------------------- sync
def _pair_corr(kp_i, vis_i, kp_j, vis_j, joints, vis_min):
    a = kp_i[:, joints].reshape(-1, 2)
    b = kp_j[:, joints].reshape(-1, 2)
    ok = ((vis_i[:, joints] >= vis_min) & (vis_j[:, joints] >= vis_min)).ravel()
    ok &= np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return a[ok], b[ok]


def sampson_median(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Median Sampson distance (px) of a RANSAC fundamental fit to a <-> b."""
    if len(a) < 30:
        return np.nan
    cv2.setRNGSeed(seed)
    F, _m = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3.0, 0.999, 5000)
    if F is None or F.shape != (3, 3):
        return np.nan
    x1 = np.hstack([a, np.ones((len(a), 1))])
    x2 = np.hstack([b, np.ones((len(b), 1))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, 1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return float(np.median(np.sqrt(num / den)))


def estimate_offsets(take: Take, *, max_shift: int = 4, sub_step: float = 0.25,
                     vis_min: float = 0.7, min_common: int = 60,
                     max_sampson_px: float = 8.0, prior=None) -> dict:
    """Per-camera time offset (frames, relative to camera 0) from epipolar fits.

    For every camera pair with enough co-detected frames, the second camera's
    track is shifted by integer then sub-frame steps and a fundamental matrix
    is fitted per shift; the shift with the lowest median Sampson error is the
    pair offset. Pair offsets are fused into per-camera offsets by least
    squares (camera 0 fixed at 0), using only *trusted* pairs (good best fit
    that clearly beats distant shifts); a camera with no trusted pair keeps
    offset 0 and ``"unresolved"`` lists it. Returns ``{"offsets": (V,), "pairs": {...}}``
    where ``offset[c]`` means camera c's frame ``k + offset`` matches camera 0's
    frame ``k`` (apply with :func:`shift_track`).

    *prior* (V,) frames, e.g. from the recording clocks (``Take.sync_prior``):
    each pair's scan is centred on the (rounded) prior difference instead of 0,
    and a camera with no trusted pair keeps its prior instead of 0. ``None``
    (default) is the same as all zeros.
    """
    V = take.V
    prior = np.zeros(V) if prior is None else np.asarray(prior, np.float64) - float(prior[0])
    joints = CALIB_JOINTS
    pairs = {}
    for i in range(V):
        for j in range(i + 1, V):
            common = int(((take.vis[i] > 0).any(1) & (take.vis[j] > 0).any(1)).sum())
            if common < min_common:
                continue
            curve = {}
            c0 = int(round(prior[j] - prior[i]))
            for d in range(c0 - max_shift, c0 + max_shift + 1):
                kj, vj = shift_track(take.kp[j], take.vis[j], d)
                curve[float(d)] = sampson_median(*_pair_corr(take.kp[i], take.vis[i],
                                                             kj, vj, joints, vis_min))
            finite = {k: v for k, v in curve.items() if np.isfinite(v)}
            if not finite:
                continue
            d0 = min(finite, key=finite.get)
            for d in np.arange(d0 - 1 + sub_step, d0 + 1, sub_step):
                if float(d) in curve:
                    continue
                kj, vj = shift_track(take.kp[j], take.vis[j], float(d))
                curve[float(d)] = sampson_median(*_pair_corr(take.kp[i], take.vis[i],
                                                             kj, vj, joints, vis_min))
            finite = {k: v for k, v in curve.items() if np.isfinite(v)}
            best = min(finite, key=finite.get)
            # a pair is trusted only if the best shift fits well and clearly
            # beats shifts +-2 frames away (false detections ruin the fit)
            far = [finite.get(best + s) for s in (-2.0, 2.0)]
            far = [f for f in far if f is not None]
            trusted = bool(finite[best] <= max_sampson_px
                           and (not far or min(far) > 1.15 * finite[best]))
            pairs[(i, j)] = {"offset": best, "curve": dict(sorted(curve.items())),
                             "common": common, "best_px": finite[best], "trusted": trusted}
    # least squares fuse: o_j - o_i = d_ij, o_0 = 0
    rows, rhs, wts = [], [], []
    for (i, j), p in pairs.items():
        if not p["trusted"]:
            continue
        r = np.zeros(V)
        r[j], r[i] = 1, -1
        rows.append(r)
        rhs.append(p["offset"])
        wts.append(np.sqrt(p["common"]))
    r0 = np.zeros(V)
    r0[0] = 1
    rows.append(r0)
    rhs.append(0.0)
    wts.append(100.0)
    A = np.array(rows) * np.array(wts)[:, None]
    b = np.array(rhs) * np.array(wts)
    off = np.linalg.lstsq(A, b, rcond=None)[0]
    linked = {0}
    for _ in range(V):
        for (i, j), p in pairs.items():
            if p["trusted"] and (i in linked or j in linked):
                linked |= {i, j}
    unresolved = [v for v in range(V) if v not in linked]
    off[unresolved] = prior[unresolved]
    return {"offsets": off, "pairs": pairs, "unresolved": unresolved}


def refine_offsets(cams: list[Camera], take: Take, keep: np.ndarray | None = None, *,
                   deltas=np.arange(-2.0, 2.01, 0.25), vis_min: float = 0.5,
                   min_gain: float = 0.10) -> dict:
    """Calibrated sync check: per camera, the shift of its track that best
    reprojects the 3D points triangulated from the *other* cameras.

    Error is swap-invariant (min over the joint and its mirror) and robust
    (median). ``offsets`` is a sub-frame (parabolic) estimate for information;
    ``integer_offsets`` is the integer shift to apply, non-zero only when it
    lowers the median error by more than *min_gain* vs. no shift.
    """
    from .multiview import FULL_SWAP, triangulate, reproj_err
    V, T = take.V, take.T
    keep = (take.vis > 0).any(-1) if keep is None else keep
    J = np.array(CALIB_JOINTS)
    Pn = np.stack([c.Pn for c in cams])
    fpx = np.array([c.f for c in cams])
    xy = np.stack([c.normalize(take.kp[v]) for v, c in enumerate(cams)])
    offsets, curves, integer = np.zeros(V), [], np.zeros(V, int)
    counts = []
    for v in range(V):
        others = [u for u in range(V) if u != v]
        o = xy[others][:, :, J].transpose(1, 2, 0, 3).reshape(-1, V - 1, 2)
        w = take.vis[others][:, :, J].transpose(1, 2, 0).reshape(-1, V - 1)
        w = np.where((w >= vis_min) & keep[others].T.repeat(len(J), 0), w, 0)
        X = triangulate(Pn[others], o, w).reshape(T, len(J), 3)
        curve = {}
        for d in deltas:
            kv, wv = shift_track(take.kp[v], take.vis[v] * keep[v][:, None], float(d))
            xv = cams[v].normalize(kv)
            e1 = reproj_err(Pn[[v]], fpx[[v]], np.nan_to_num(X.reshape(-1, 3)),
                            xv[:, J].reshape(-1, 1, 2)).reshape(T, len(J))
            e2 = reproj_err(Pn[[v]], fpx[[v]], np.nan_to_num(X.reshape(-1, 3)),
                            xv[:, FULL_SWAP[J]].reshape(-1, 1, 2)).reshape(T, len(J))
            e = np.minimum(e1, e2)
            m = np.isfinite(X[..., 0]) & (wv[:, J] >= vis_min) & np.isfinite(e)
            curve[float(d)] = float(np.median(e[m])) if m.sum() > 50 else np.nan
            if abs(d) < 1e-9:
                counts.append(int(m.sum()))
        fin = {k: c for k, c in curve.items() if np.isfinite(c)}
        if fin:
            ks = np.array(sorted(fin))
            cs = np.array([fin[k] for k in ks])
            i = int(np.argmin(cs))
            best = ks[i]
            if 0 < i < len(ks) - 1:                # parabolic sub-step refinement
                den = cs[i - 1] - 2 * cs[i] + cs[i + 1]
                if den > 0:
                    best = ks[i] + 0.5 * (cs[i - 1] - cs[i + 1]) / den * (ks[1] - ks[0])
            offsets[v] = best
            # integer decision: only move if an integer shift is clearly better
            # than none (fractional shifts get an interpolation-smoothing bonus,
            # so they are not compared against integer ones)
            ints = {k: c for k, c in fin.items() if abs(k - round(k)) < 1e-9}
            if 0.0 in ints:
                kb = min(ints, key=ints.get)
                if ints[kb] < (1 - min_gain) * ints[0.0]:
                    integer[v] = int(round(kb))
        curves.append(curve)
    return {"offsets": offsets, "integer_offsets": integer, "curves": curves,
            "n_samples": counts}


# ------------------------------------------------------------ initialisation
def default_K(size, f: float) -> np.ndarray:
    w, h = size
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return 0.5 * width / np.tan(np.radians(hfov_deg) / 2)


def _register_pnp(cam: Camera, kp, vis, Xmap, vis_min, f0):
    obj, img = [], []
    m = (vis >= vis_min) & np.isfinite(Xmap).all(-1) & np.isfinite(kp).all(-1)
    m[:, [j for j in range(NUM_JOINTS) if j not in CALIB_JOINTS]] = False
    obj = Xmap[m]
    img = cam.normalize(kp[m])
    if len(obj) < 20:
        raise RuntimeError("not enough 2D-3D matches to register a camera")
    ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, np.eye(3), None,
                                         reprojectionError=6.0 / f0,
                                         iterationsCount=2000, confidence=0.999,
                                         flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None:
        raise RuntimeError("PnP failed")
    inl = inl.ravel()
    ok, rv, tv = cv2.solvePnP(obj[inl], img[inl], np.eye(3), None, rv, tv,
                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    cam.R = rodrigues(rv.ravel())
    cam.t = tv.ravel()
    return len(inl), len(obj)


def initial_cameras(take: Take, f0: float, *, ref: int = 0, vis_min: float = 0.7,
                    log=print) -> list[Camera]:
    """Essential matrix for the best pair with *ref*, PnP for the rest."""
    V = take.V
    cams = [Camera(default_K(take.size, f0)) for _ in range(V)]
    det = (take.vis > 0).any(-1)
    co = [(int((det[ref] & det[j]).sum()), j) for j in range(V) if j != ref]
    _, j0 = max(co)
    a, b = _pair_corr(take.kp[ref], take.vis[ref], take.kp[j0], take.vis[j0],
                      CALIB_JOINTS, vis_min)
    na, nb = cams[ref].normalize(a), cams[j0].normalize(b)
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, 2.0 / f0)
    _, R, tt, mask2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask.copy())
    cams[j0].R, cams[j0].t = R, tt.ravel()
    log(f"  init pair {take.cam_ids[ref]}-{take.cam_ids[j0]}: E inliers "
        f"{int(mask.sum())}/{len(na)}, cheirality {int((mask2 > 0).sum())}")
    registered = [ref, j0]
    while len(registered) < V:
        Xmap = _triangulate_registered(cams, take, registered)
        rest = [j for j in range(V) if j not in registered]
        cnt = [(int(((take.vis[j] >= vis_min) & np.isfinite(Xmap).all(-1)).sum()), j)
               for j in rest]
        _, j = max(cnt)
        ni, nt = _register_pnp(cams[j], take.kp[j], take.vis[j], Xmap, vis_min, f0)
        log(f"  PnP {take.cam_ids[j]}: {ni}/{nt} inliers")
        registered.append(j)
    return cams


def _triangulate_registered(cams, take, registered):
    sub = [cams[i] for i in registered]
    xy = np.stack([cams[i].normalize(take.kp[i]) for i in registered])
    vis = take.vis[registered]
    X, valid, *_ = robust_triangulate(sub, xy, vis, vis_thresh=0.6, max_err_px=15.0)
    return X


# ------------------------------------------------------------ bundle adjust
@dataclass
class BAResult:
    cams: list[Camera]
    points: np.ndarray            # (N,3)
    rms_before: np.ndarray        # (V,) px, on the BA observation set
    rms_after: np.ndarray
    med_after: np.ndarray
    rms_in_before: np.ndarray     # (V,) px, RMS over inlier observations
    rms_in_after: np.ndarray      #   (final error < inlier_px)
    med_before: np.ndarray
    n_obs: np.ndarray             # (V,)
    cost: float
    nfev: int


def _project_obs(R, t, f, k1, k2, c, X):
    Xc = np.einsum("nij,nj->ni", R, X) + t
    z = Xc[:, 2:3]
    xy = Xc[:, :2] / z
    r2 = np.sum(xy * xy, 1, keepdims=True)
    return xy * (1 + k1[:, None] * r2 + k2[:, None] * r2 * r2) * f[:, None] + c


def bundle_adjust(cams: list[Camera], kp: np.ndarray, vis: np.ndarray,
                  X0: np.ndarray, obs_mask: np.ndarray, *, ref: int = 0,
                  shared_focal: bool = False, distortion: str = "k1",
                  f_scale: float = 3.0, max_nfev: int = 60, inlier_px: float = 20.0,
                  log=print) -> BAResult:
    """Bundle-adjust extrinsics, focal(s), radial distortion and 3D points.

    *kp* (V,T,33,2) pixels (already L/R-relabelled), *vis* (V,T,33),
    *X0* (T,33,3) initial points (nan = not a BA point), *obs_mask* (V,T,33)
    which observations to use. Gauge: camera *ref* is held fixed and the mean
    distance of the other camera centres from it is softly pinned (scale).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    V = len(cams)
    pt_mask = np.isfinite(X0).all(-1)
    pt_index = -np.ones(pt_mask.shape, int)
    pt_index[pt_mask] = np.arange(pt_mask.sum())
    P0 = X0[pt_mask]
    N = len(P0)
    om = obs_mask & pt_mask[None] & np.isfinite(kp).all(-1)
    cv, ct, cj = np.nonzero(om)
    pid = pt_index[ct, cj]
    uv = kp[cv, ct, cj]
    wobs = np.sqrt(np.clip(vis[cv, ct, cj], 0.05, 1.0))
    M = len(cv)

    others = [v for v in range(V) if v != ref]
    n_ext = 6 * len(others)
    nf = 1 if shared_focal else V
    nk = {"none": 0, "k1": 1, "k1k2": 2}[distortion]
    n_int = nf + nk * V
    n_cam = n_ext + n_int
    C_ref = cams[ref].center
    d0 = np.mean([np.linalg.norm(cams[v].center - C_ref) for v in others])
    cc = np.stack([c.K[:2, 2] for c in cams])

    def pack(cs, P):
        ext = np.concatenate([np.r_[rotvec(cs[v].R), cs[v].t] for v in others]) if others else []
        fs = [np.mean([c.f for c in cs])] if shared_focal else [c.f for c in cs]
        ks = [c.dist[i] for c in cs for i in range(nk)]
        return np.concatenate([ext, np.log(fs), ks, P.ravel()])

    def unpack(p):
        rv = np.zeros((V, 3))
        tv = np.zeros((V, 3))
        rv[ref] = rotvec(cams[ref].R)
        tv[ref] = cams[ref].t
        for i, v in enumerate(others):
            rv[v] = p[6 * i:6 * i + 3]
            tv[v] = p[6 * i + 3:6 * i + 6]
        f = np.exp(p[n_ext:n_ext + nf])
        f = np.repeat(f, V) if shared_focal else f
        k = p[n_ext + nf:n_cam].reshape(V, nk) if nk else np.zeros((V, 0))
        P = p[n_cam:].reshape(N, 3)
        return rv, tv, f, k, P

    def residuals(p):
        rv, tv, f, k, P = unpack(p)
        R = rodrigues(rv)
        k1 = k[:, 0] if nk >= 1 else np.zeros(V)
        k2 = k[:, 1] if nk >= 2 else np.zeros(V)
        pr = _project_obs(R[cv], tv[cv], f[cv], k1[cv], k2[cv], cc[cv], P[pid])
        res = ((pr - uv) * wobs[:, None]).ravel()
        C = -np.einsum("vji,vj->vi", R, tv)
        dmean = np.mean(np.linalg.norm(C[others] - C[ref], axis=1))
        extra = [100.0 * (dmean / d0 - 1.0)]
        if nk:
            extra += list(5.0 * k.ravel())     # weak prior towards no distortion
        return np.concatenate([res, extra])

    nres = 2 * M + 1 + nk * V
    ri, ci = [], []
    obs_rows = np.concatenate([2 * np.arange(M), 2 * np.arange(M) + 1])
    obs_cam = np.concatenate([cv, cv])
    obs_pid = np.concatenate([pid, pid])
    for i, v in enumerate(others):
        sel = obs_rows[obs_cam == v]
        for d in range(6):
            ri.append(sel)
            ci.append(np.full(len(sel), 6 * i + d))
    for v in range(V):
        sel = obs_rows[obs_cam == v]
        ri.append(sel)
        ci.append(np.full(len(sel), n_ext + (0 if shared_focal else v)))
        for q in range(nk):
            ri.append(sel)
            ci.append(np.full(len(sel), n_ext + nf + v * nk + q))
    for d in range(3):
        ri.append(obs_rows)
        ci.append(n_cam + 3 * obs_pid + d)
    ri.append(np.full(n_ext, 2 * M))
    ci.append(np.arange(n_ext))
    ri.append(2 * M + 1 + np.arange(nk * V))
    ci.append(n_ext + nf + np.arange(nk * V))
    ri, ci = np.concatenate(ri), np.concatenate(ci)
    J = coo_matrix((np.ones(len(ri), np.int8), (ri, ci)), shape=(nres, n_cam + 3 * N))

    def per_cam_rms(r):
        e = np.linalg.norm(r[:2 * M].reshape(M, 2) / wobs[:, None], axis=1)
        rms = np.array([np.sqrt(np.mean(e[cv == v] ** 2)) if (cv == v).any() else np.nan
                        for v in range(V)])
        med = np.array([np.median(e[cv == v]) if (cv == v).any() else np.nan
                        for v in range(V)])
        return rms, med, e

    p0 = pack(cams, P0)
    r0 = residuals(p0)
    rms0, med0, e0 = per_cam_rms(r0)
    log(f"  BA: {V} cams, {N} points, {M} observations; rms before "
        + " ".join(f"{x:.2f}" for x in rms0))
    sol = least_squares(residuals, p0, jac_sparsity=J.tocsr(), loss="soft_l1",
                        f_scale=f_scale, x_scale="jac", method="trf",
                        max_nfev=max_nfev, verbose=0)
    rv, tv, f, k, P = unpack(sol.x)
    out = []
    for v in range(V):
        d = np.zeros(5)
        d[:nk] = k[v]
        out.append(Camera(np.array([[f[v], 0, cc[v, 0]], [0, f[v], cc[v, 1]], [0, 0, 1]]),
                          d, rodrigues(rv[v]), tv[v]))
    rms1, med1, e1 = per_cam_rms(sol.fun)
    inl = e1 < inlier_px
    rin0 = np.array([np.sqrt(np.mean(e0[(cv == v) & inl] ** 2)) for v in range(V)])
    rin1 = np.array([np.sqrt(np.mean(e1[(cv == v) & inl] ** 2)) for v in range(V)])
    log(f"  BA done ({sol.nfev} evals): rms after "
        + " ".join(f"{x:.2f}" for x in rms1) + "; f " + " ".join(f"{x:.0f}" for x in f)
        + ("; k1 " + " ".join(f"{x:+.3f}" for x in k[:, 0]) if nk else "")
        + "; inlier rms " + " ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(rin0, rin1)))
    Pall = np.full(X0.shape, np.nan)
    Pall[pt_mask] = P
    n_obs = np.array([(cv == v).sum() for v in range(V)])
    return BAResult(out, Pall, rms0, rms1, med1, rin0, rin1, med0, n_obs,
                    float(sol.cost), int(sol.nfev))


# ------------------------------------------------------------- world frame
# Reference segment lengths (m) between MediaPipe landmarks for an adult of
# ~1.72 m (joint-centre anthropometry; MediaPipe's own world-landmark output
# for this kind of body lands in the same range).
LIMB_PRIOR = {"upper_arm": 0.29, "forearm": 0.25, "thigh": 0.42, "shin": 0.41}
SEGMENTS = {
    "shoulder_width": [(11, 12)],
    "hip_width": [(23, 24)],
    "upper_arm": [(11, 13), (12, 14)],
    "forearm": [(13, 15), (14, 16)],
    "thigh": [(23, 25), (24, 26)],
    "shin": [(25, 27), (26, 28)],
}


def segment_medians(X: np.ndarray) -> dict:
    out = {}
    for name, prs in SEGMENTS.items():
        ls = np.concatenate([np.linalg.norm(X[:, a] - X[:, b], axis=-1) for a, b in prs])
        ls = ls[np.isfinite(ls)]
        out[name] = float(np.median(ls)) if len(ls) else np.nan
    return out


@dataclass
class Similarity:
    """World transform ``X_w = s * R @ X + T`` (proper rotation)."""
    s: float
    R: np.ndarray
    T: np.ndarray

    def apply(self, X):
        return self.s * (np.asarray(X) @ self.R.T) + self.T

    def apply_cameras(self, cams: list[Camera]) -> list[Camera]:
        out = []
        for c in cams:
            Rn = c.R @ self.R.T
            tn = self.s * c.t - Rn @ self.T
            out.append(Camera(c.K.copy(), c.dist.copy(), Rn, tn))
        return out


def _fit_plane(P, iters=500, thresh=0.03, seed=0):
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        d = np.abs((P - s[0]) @ n)
        inl = d < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (n, s[0]), inl
    Q = P[best_in]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c)
    return vt[-1], c, best_in


def world_alignment(X: np.ndarray, *, scale_mode: str = "limbs",
                    shoulder_width: float = 0.38, scale: float | None = None) -> tuple:
    """Similarity mapping a reconstruction (arbitrary frame) to the world frame.

    Scale: ``limbs`` matches the summed median upper-arm + forearm + thigh +
    shin lengths to :data:`LIMB_PRIOR`; ``shoulder`` matches the median
    shoulder width to *shoulder_width*; an explicit *scale* overrides both.
    Up: plane fitted (RANSAC) to the per-frame lowest foot landmark, oriented
    towards the head; floor at ``z = 0``. Origin: mean pelvis xy. Heading: the
    circular mean of the performer's facing (from hips + shoulders) maps to
    ``-Y``. Returns ``(Similarity, info)``.
    """
    med = segment_medians(X)
    if scale is not None:
        s = float(scale)
    elif scale_mode == "shoulder":
        s = shoulder_width / med["shoulder_width"]
    else:
        s = sum(LIMB_PRIOR.values()) / sum(med[k] for k in LIMB_PRIOR)
    Xs = X * s
    mid_sh = 0.5 * (Xs[:, 11] + Xs[:, 12])
    mid_hip = 0.5 * (Xs[:, 23] + Xs[:, 24])
    spine = mid_sh - mid_hip
    spine = spine[np.isfinite(spine).all(1)]
    up0 = np.median(spine / np.linalg.norm(spine, axis=1, keepdims=True), 0)
    up0 /= np.linalg.norm(up0)
    feet = Xs[:, [27, 28, 29, 30, 31, 32]]
    h = feet @ up0
    h = np.where(np.isfinite(h), h, np.inf)
    lo = np.argmin(h, 1)
    fr = np.isfinite(h.min(1))
    P = feet[np.flatnonzero(fr), lo[fr]]
    n, c, inl = _fit_plane(P)
    if np.dot(n, up0) < 0:
        n = -n
    z = n
    # facing: cross(left - right, up), from hips and shoulders
    fac = []
    for l, r in ((23, 24), (11, 12)):
        f = np.cross(Xs[:, l] - Xs[:, r], z)
        fac.append(f)
    with quiet():
        f = np.nanmean(np.stack(fac), 0)
    f = f - (f @ z)[:, None] * z
    f = f[np.isfinite(f).all(1)]
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    mean_f = fn.mean(0)
    resultant = float(np.linalg.norm(mean_f))
    y = -mean_f / np.linalg.norm(mean_f)          # facing -> -Y
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                        # rows: new axes in old frame
    pel = mid_hip[np.isfinite(mid_hip).all(1)]
    pel_w = pel @ R.T
    T = np.array([-pel_w[:, 0].mean(), -pel_w[:, 1].mean(), -(c @ z)])
    info = {"scale": s, "segment_medians_raw": med, "floor_inlier_frac": float(inl.mean()),
            "facing_resultant": resultant, "scale_mode": scale_mode if scale is None else "fixed"}
    return Similarity(s, R, T), info


def mean_rotation(Rs, w=None) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (N,3,3), optionally weighted."""
    Rs = np.asarray(Rs, float)
    w = np.ones(len(Rs)) if w is None else np.asarray(w, float)
    M = np.einsum("n,nij->ij", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    """Angle(s) of rotation matrix(es) (...,3,3) in degrees."""
    tr = np.trace(np.asarray(R, float), axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))


def camera_similarity(src: list[Camera], dst: list[Camera]) -> Similarity:
    """Similarity mapping the frame of cameras *src* onto that of *dst*.

    Both lists describe the same physical cameras in two frames (e.g. a body
    self-calibration with arbitrary gauge vs a metric board calibration). The
    rotation is the mean of the per-camera ``R_dst^T R_src``, the scale the
    median ratio of camera-to-camera distances, the translation the mean
    residual of the centres. Needs >= 2 cameras (>= 1 if scales are equal).
    """
    R = mean_rotation([d.R.T @ s.R for s, d in zip(src, dst)])
    Cs = np.stack([c.center for c in src])
    Cd = np.stack([c.center for c in dst])
    ratios = [np.linalg.norm(Cd[i] - Cd[j]) / np.linalg.norm(Cs[i] - Cs[j])
              for i in range(len(src)) for j in range(i + 1, len(src))
              if np.linalg.norm(Cs[i] - Cs[j]) > 1e-9]
    s = float(np.median(ratios)) if ratios else 1.0
    T = np.mean(Cd - s * Cs @ R.T, axis=0)
    return Similarity(s, R, T)


def compare_cameras(a: list[Camera], b: list[Camera], cam_ids: list[str]) -> dict:
    """Pose differences of camera set *a* vs *b* after a best similarity a -> b."""
    sim = camera_similarity(a, b)
    aw = sim.apply_cameras(a)
    out = {"scale_a_to_b": sim.s, "cameras": {}}
    for cid, ca, cb in zip(cam_ids, aw, b):
        out["cameras"][cid] = {
            "center_diff_m": float(np.linalg.norm(ca.center - cb.center)),
            "rotation_diff_deg": float(rotation_angle_deg(ca.R @ cb.R.T)),
            "focal_a_px": ca.f, "focal_b_px": cb.f,
            "focal_diff_pct": float(100 * (ca.f / cb.f - 1))}
    return out


# ------------------------------------------------------------------ rig I/O
def to_rig(cams: list[Camera], cam_ids: list[str], size, note: str = ""):
    from .calibration import BoardConfig, CameraCalibration, RigCalibration
    rig = RigCalibration(board=BoardConfig())
    for cid, c in zip(cam_ids, cams):
        rig.cameras[cid] = CameraCalibration(cid, tuple(size), c.K.copy(), c.dist.copy(),
                                             rotvec(c.R), c.t.copy())
    return rig


def save_rig(path: str, cams, cam_ids, size, extra: dict | None = None) -> None:
    d = to_rig(cams, cam_ids, size).to_dict()
    if extra:
        d["selfcal"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def from_rig(path: str, cam_ids: list[str]) -> list[Camera]:
    from .calibration import RigCalibration
    rig = RigCalibration.load(path)
    missing = [c for c in cam_ids if c not in rig.cameras]
    if missing:
        raise KeyError(f"rig.json has no calibration for {missing}")
    out = []
    for cid in cam_ids:
        cc = rig.cameras[cid]
        out.append(Camera(cc.K, cc.dist, cc.rotation_matrix(), cc.tvec))
    return out
