"""
Robust multi-view triangulation of MediaPipe body landmarks (pure numpy + cv2).

The one triangulation of capmo (``reconstruct_take``, the pipeline and the
end-to-end tests); it handles a whole take of real detections (a single
instant is just ``T = 1``):

* a lens-aware :class:`Camera` (OpenCV ``K``/``dist``/``R``/``t``) so points are
  undistorted to normalized coordinates before the linear solve;
* batched, depth-reweighted DLT over thousands of points at once;
* **left/right flip repair** -- MediaPipe often mirrors the whole body, or just
  the legs/arms, in side and back views. Per frame and per body part (face,
  arms, legs) every combination of "this view is swapped" is triangulated and
  the cheapest consistent labelling wins; the mirror ambiguity left over (all
  views swapped looks just as consistent) is resolved by anatomical chirality
  (the nose is in front of the ears/shoulders, toes in front of heels);
* leave-worst-view-out outlier rejection per joint, and a >=2 view requirement.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

NUM_JOINTS = 33


@contextmanager
def quiet():
    """Silence numpy's all-nan / empty-slice RuntimeWarnings (expected here)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

# MediaPipe Pose landmark names, in the model's output order.
NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
IDX = {n: i for i, n in enumerate(NAMES)}

# (left, right) landmark pairs, grouped into parts that MediaPipe tends to
# mirror together.
PART_PAIRS = {
    "face": [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)],
    "arms": [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)],
    "legs": [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)],
}
PART_JOINTS = {p: sorted({j for pr in prs for j in pr}) for p, prs in PART_PAIRS.items()}
PART_JOINTS["face"] = [0] + PART_JOINTS["face"]


def swap_perm(parts=("face", "arms", "legs")) -> np.ndarray:
    """Index permutation that swaps left/right labels of the given parts."""
    perm = np.arange(NUM_JOINTS)
    for p in parts:
        for a, b in PART_PAIRS[p]:
            perm[a], perm[b] = b, a
    return perm


FULL_SWAP = swap_perm()


# --------------------------------------------------------------------- camera
def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector(s) (...,3) -> rotation matrix(es) (...,3,3)."""
    r = np.asarray(rvec, np.float64)
    th = np.linalg.norm(r, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, r / np.maximum(th, 1e-12), 0.0)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    Kx = np.stack([np.stack([zero, -kz, ky], -1),
                   np.stack([kz, zero, -kx], -1),
                   np.stack([-ky, kx, zero], -1)], -2)
    th = th[..., None]
    eye = np.broadcast_to(np.eye(3), Kx.shape)
    return eye + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (via cv2)."""
    v, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return v.ravel()


@dataclass
class Camera:
    """Pinhole camera with OpenCV distortion: ``x_cam = R X + t``."""
    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.K = np.asarray(self.K, np.float64)
        d = np.zeros(5)
        dd = np.asarray(self.dist, np.float64).ravel()[:5]
        d[:len(dd)] = dd
        self.dist = d
        self.R = np.asarray(self.R, np.float64)
        self.t = np.asarray(self.t, np.float64).ravel()

    @classmethod
    def from_calibration(cls, cam) -> "Camera":
        """From a :class:`capmo.calibration.CameraCalibration` (``rig.json`` entry)."""
        return cls(K=cam.K, dist=cam.dist, R=rodrigues(np.asarray(cam.rvec, np.float64)),
                   t=np.asarray(cam.tvec, np.float64))

    @property
    def f(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def Pn(self) -> np.ndarray:
        """Normalized projection [R|t] (works on undistorted normalized coords)."""
        return np.hstack([self.R, self.t[:, None]])

    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2, k3 = self.dist
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], -1)

    def normalized_to_pixels(self, xy: np.ndarray) -> np.ndarray:
        d = self.distort_normalized(xy)
        K = self.K
        return np.stack([K[0, 0] * d[..., 0] + K[0, 1] * d[..., 1] + K[0, 2],
                         K[1, 1] * d[..., 1] + K[1, 2]], -1)

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points (...,3) -> pixels (...,2) (nan stays nan)."""
        Xc = np.asarray(X, np.float64) @ self.R.T + self.t
        z = Xc[..., 2:3]
        with np.errstate(invalid="ignore", divide="ignore"):
            xy = Xc[..., :2] / z
        return self.normalized_to_pixels(xy)

    def normalize(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (...,2) -> undistorted normalized coords (nan-safe)."""
        uv = np.asarray(uv, np.float64)
        out = np.full(uv.shape, np.nan)
        flat = uv.reshape(-1, 2)
        ok = np.isfinite(flat).all(1)
        res = out.reshape(-1, 2)
        if ok.any():
            if not np.any(self.dist):
                res[ok, 0] = (flat[ok, 0] - self.K[0, 2] - self.K[0, 1] / self.K[1, 1]
                              * (flat[ok, 1] - self.K[1, 2])) / self.K[0, 0]
                res[ok, 1] = (flat[ok, 1] - self.K[1, 2]) / self.K[1, 1]
            else:
                K = self.K
                yd = (flat[ok, 1] - K[1, 2]) / K[1, 1]
                xd = (flat[ok, 0] - K[0, 2] - K[0, 1] * yd) / K[0, 0]
                d = np.stack([xd, yd], -1)
                x = d.copy()
                for _ in range(30):              # fixed-point: d = distort(x)
                    x = x + (d - self.distort_normalized(x))
                res[ok] = x
        return res.reshape(uv.shape)


# ---------------------------------------------------------------- triangulate
def triangulate(Pn: np.ndarray, xy: np.ndarray, w: np.ndarray,
                iters: int = 2) -> np.ndarray:
    """Batched weighted DLT.

    *Pn*: (V,3,4) normalized projections; *xy*: (N,V,2) normalized observations
    (anything where ``w == 0`` is ignored, nan allowed there); *w*: (N,V)
    weights. Rows are re-weighted by 1/depth *iters* times so the linear solve
    approximates the geometric (reprojection) error. Returns (N,3); rows with
    fewer than two weighted views are nan.
    """
    xy = np.where(np.isfinite(xy), xy, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    N, V = w.shape
    out = np.full((N, 3), np.nan)
    ok = (w > 0).sum(1) >= 2
    if not ok.any():
        return out
    xy, w = xy[ok], w[ok]
    r0 = xy[..., 0:1] * Pn[None, :, 2] - Pn[None, :, 0]      # (n,V,4)
    r1 = xy[..., 1:2] * Pn[None, :, 2] - Pn[None, :, 1]
    scale = np.ones_like(w)
    X = None
    for it in range(iters + 1):
        ww = (w * scale)[..., None]
        A = np.concatenate([r0 * ww, r1 * ww], axis=1)       # (n,2V,4)
        _u, _s, vt = np.linalg.svd(A)
        Xh = vt[:, -1]
        Xh = Xh / np.where(np.abs(Xh[:, 3:4]) > 1e-12, Xh[:, 3:4], 1e-12)
        X = Xh[:, :3]
        depth = np.einsum("vk,nk->nv", Pn[:, 2], Xh)
        scale = 1.0 / np.maximum(np.abs(depth), 1e-6)
    out[ok] = X
    return out


def reproj_err(Pn: np.ndarray, fpx: np.ndarray, X: np.ndarray,
               xy: np.ndarray) -> np.ndarray:
    """Approx. pixel error (normalized error * focal) of X (N,3) in each view."""
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], -1)
    p = np.einsum("vij,nj->nvi", Pn, Xh)
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = p[..., :2] / p[..., 2:3]
        e = np.linalg.norm(proj - xy, axis=-1) * fpx[None, :]
        e = np.where(p[..., 2] > 0, e, 1e4)
    return e


@dataclass
class TriangulationResult:
    X: np.ndarray            # (T,33,3) nan where unsolved
    valid: np.ndarray        # (T,33) bool
    n_views: np.ndarray      # (T,33) views used after outlier rejection
    err_px: np.ndarray       # (T,33) mean reprojection error of used views
    used: np.ndarray         # (V,T,33) bool: view used for the joint
    swaps: np.ndarray        # (V,T,3) bool: face/arms/legs labels swapped in view
    kept: np.ndarray | None = None   # (V,T) bool: detection agreed with consensus


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def _part_score(part: str, Xp: dict, up: np.ndarray, ref_facing):
    """Chirality score (>0 = labels plausible) for one part's 3D joints."""
    def g(i):
        return Xp.get(i)

    def facing(l, r):
        if g(l) is None or g(r) is None:
            return None
        f = np.cross(g(l) - g(r), up)
        n = np.linalg.norm(f)
        return f / n if n > 1e-9 else None

    s, fac = 0.0, None
    if part == "face":
        fac = facing(7, 8) if facing(7, 8) is not None else facing(3, 6)
        if fac is not None and g(0) is not None:
            mid = (0.5 * (g(7) + g(8)) if g(7) is not None and g(8) is not None
                   else 0.5 * (g(3) + g(6)))
            s += float(np.dot(_unit(g(0) - mid), fac))
    elif part == "arms":
        fac = facing(11, 12)
        if fac is not None and ref_facing.get("nose_dir") is not None:
            s += float(np.dot(ref_facing["nose_dir"], fac))
        if fac is not None and ref_facing.get("face") is not None:
            s += float(np.dot(ref_facing["face"], fac))
    elif part == "legs":
        fac = facing(23, 24)
        if fac is not None:
            feet = [g(31) - g(29) if g(31) is not None and g(29) is not None else None,
                    g(32) - g(30) if g(32) is not None and g(30) is not None else None]
            feet = [f for f in feet if f is not None]
            if feet:
                fd = np.mean(feet, 0)
                fd = fd - np.dot(fd, up) * up
                if np.linalg.norm(fd) > 1e-9:
                    s += float(np.dot(_unit(fd), fac))
            if ref_facing.get("arms") is not None:
                s += float(np.dot(ref_facing["arms"], fac))
    return s, fac


def resolve_swaps(cams: list[Camera], xy: np.ndarray, vis: np.ndarray, *,
                  vis_thresh: float = 0.3, tau_px: float = 30.0) -> np.ndarray:
    """Per view/frame/part left-right swap flags (V,T,3), see module doc.

    *xy*: (V,T,33,2) normalized observations (nan = not detected);
    *vis*: (V,T,33).
    """
    V, T = xy.shape[:2]
    Pn = np.stack([c.Pn for c in cams])
    fpx = np.array([c.f for c in cams])
    swaps = np.zeros((V, T, 3), bool)
    parts = ["face", "arms", "legs"]
    det = np.isfinite(xy[..., 0, 0])                                   # (V,T)
    for t in range(T):
        views = np.flatnonzero(det[:, t])
        n = len(views)
        if n < 2:
            continue
        combos = np.array(list(product([0, 1], repeat=n - 1)), bool)
        combos = np.concatenate([np.zeros((len(combos), 1), bool), combos], 1)  # (C,n)
        Pv, fv = Pn[views], fpx[views]
        canon = {}
        for pi, part in enumerate(parts):
            J = np.array(PART_JOINTS[part])
            perm = swap_perm([part])
            o = xy[views, t][:, J]                    # (n,J,2)
            os_ = xy[views, t][:, perm[J]]
            wv = vis[views, t][:, J]
            ws = vis[views, t][:, perm[J]]
            sel = combos[:, :, None]                  # (C,n,1)
            obs = np.where(sel[..., None], os_[None], o[None])        # (C,n,J,2)
            w = np.where(sel, ws[None], wv[None])                     # (C,n,J)
            w = np.where(w >= vis_thresh, w, 0.0)
            obs = obs.transpose(0, 2, 1, 3).reshape(-1, n, 2)
            w = w.transpose(0, 2, 1).reshape(-1, n)
            X = triangulate(Pv, obs, w, iters=1)
            e = reproj_err(Pv, fv, np.nan_to_num(X), obs)
            cost = np.where(w > 0, np.minimum(e, tau_px) ** 2 * w, 0.0)
            cost = np.where(np.isfinite(X[:, :1]), cost, tau_px ** 2 * w)
            cost = cost.reshape(len(combos), len(J), n).sum((1, 2))
            # prefer fewer swapped views on near-ties
            cost = cost + 1e-3 * combos.sum(1)
            best = int(np.argmin(cost))
            Xb = X.reshape(len(combos), len(J), 3)[best]
            canon[part] = (combos[best], J, Xb)
        # chirality, using label-independent midpoints for the body "up"
        def mid(part, a, b):
            Jl = list(canon[part][1])
            Xb = canon[part][2]
            pa, pb = Xb[Jl.index(a)], Xb[Jl.index(b)]
            return 0.5 * (pa + pb) if np.isfinite(pa).all() and np.isfinite(pb).all() else None
        ms, mh = mid("arms", 11, 12), mid("legs", 23, 24)
        if ms is None or mh is None:
            for pi, part in enumerate(parts):
                swaps[views, t, pi] = canon[part][0]
            continue
        up = _unit(ms - mh)
        ref = {}
        nose = canon["face"][2][0]
        if np.isfinite(nose).all():
            nd = nose - ms
            nd = nd - np.dot(nd, up) * up
            ref["nose_dir"] = _unit(nd) if np.linalg.norm(nd) > 1e-9 else None
        for pi, part in enumerate(parts):
            combo, J, Xb = canon[part]
            Xp = {int(j): Xb[i] for i, j in enumerate(J) if np.isfinite(Xb[i]).all()}
            s, fac = _part_score(part, Xp, up, ref)
            if s < 0:
                combo = ~combo
                fac = None if fac is None else -fac
            ref[part] = fac
            swaps[views, t, pi] = combo
    return swaps


def apply_swaps(arr: np.ndarray, swaps: np.ndarray) -> np.ndarray:
    """Relabel per-view arrays (V,T,33,...) by swap flags (V,T,3)."""
    out = arr.copy()
    for pi, part in enumerate(["face", "arms", "legs"]):
        perm = swap_perm([part])
        J = np.array(PART_JOINTS[part])
        v, t = np.nonzero(swaps[:, :, pi])
        if len(v):
            out[v[:, None], t[:, None], J[None, :]] = arr[v[:, None], t[:, None], perm[J][None, :]]
    return out


def robust_triangulate(cams: list[Camera], xy: np.ndarray, vis: np.ndarray, *,
                       vis_thresh: float = 0.3, min_views: int = 2,
                       out_px: float = 15.0, out_ratio: float = 2.5,
                       max_err_px: float = 40.0) -> tuple:
    """Triangulate labelled observations with leave-worst-view-out rejection.

    *xy* (V,T,33,2) normalized, *vis* (V,T,33). A view is dropped from a joint
    while >=3 views remain and its error exceeds max(out_px, out_ratio*median).
    A joint with mean error > *max_err_px* after that is marked invalid.
    Returns (X, valid, n_views, err_px, used).
    """
    V, T = xy.shape[:2]
    Pn = np.stack([c.Pn for c in cams])
    fpx = np.array([c.f for c in cams])
    o = xy.transpose(1, 2, 0, 3).reshape(-1, V, 2)
    w = vis.transpose(1, 2, 0).reshape(-1, V).astype(np.float64)
    w = np.where((w >= vis_thresh) & np.isfinite(o[..., 0]), np.maximum(w, 1e-3), 0.0)
    for _ in range(V - 2):
        X = triangulate(Pn, o, w)
        e = np.where(w > 0, reproj_err(Pn, fpx, np.nan_to_num(X), o), np.nan)
        nv = (w > 0).sum(1)
        with quiet():
            med = np.nanmedian(e, 1)
            worst = np.nanargmax(np.where(np.isfinite(e), e, -1), 1)
        emax = e[np.arange(len(e)), worst]
        drop = (nv >= 3) & (emax > np.maximum(out_px, out_ratio * med))
        if not drop.any():
            break
        w[np.flatnonzero(drop), worst[drop]] = 0.0
    X = triangulate(Pn, o, w)
    e = np.where(w > 0, reproj_err(Pn, fpx, np.nan_to_num(X), o), np.nan)
    nv = (w > 0).sum(1)
    with quiet():
        err = np.nanmean(e, 1)
    valid = (nv >= min_views) & np.isfinite(X).all(1) & (err <= max_err_px)
    X[~valid] = np.nan
    used = (w > 0) & valid[:, None]
    return (X.reshape(T, NUM_JOINTS, 3), valid.reshape(T, NUM_JOINTS),
            nv.reshape(T, NUM_JOINTS), err.reshape(T, NUM_JOINTS),
            used.reshape(T, NUM_JOINTS, V).transpose(2, 0, 1))


CONSENSUS_JOINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def consensus_views(cams: list[Camera], xy: np.ndarray, vis: np.ndarray, *,
                    max_med_px: float = 20.0, vis_thresh: float = 0.3) -> np.ndarray:
    """(V,T) bool: which per-frame detections agree with the other views.

    Detectors hallucinate people in clutter; such a detection is inconsistent
    with every genuine view for *every* joint. Each camera pair is tested per
    frame: the two views are triangulated (as labelled, and with one view
    mirrored -- MediaPipe L/R swaps must not count as disagreement) and the
    median over joints of the mean reprojection error is the pair's
    disagreement. Per frame the largest subset of views that is pairwise
    consistent (< *max_med_px*) is kept (ties: lowest total disagreement); a
    frame whose detections agree with no other view keeps none (a lone
    detection cannot be triangulated anyway).
    """
    V, T = xy.shape[:2]
    Pn = np.stack([c.Pn for c in cams])
    fpx = np.array([c.f for c in cams])
    J = np.array(CONSENSUS_JOINTS)
    det = np.isfinite(xy[..., 0, 0])
    dis = np.full((V, V, T), np.inf)
    for u in range(V):
        for v in range(u + 1, V):
            frames = np.flatnonzero(det[u] & det[v])
            if not len(frames):
                continue
            best = np.full(len(frames), np.inf)
            for perm in (np.arange(NUM_JOINTS), FULL_SWAP):
                o = np.stack([xy[u][frames][:, J], xy[v][frames][:, perm[J]]], 2).reshape(-1, 2, 2)
                w = np.stack([vis[u][frames][:, J], vis[v][frames][:, perm[J]]], 2).reshape(-1, 2)
                w = np.where(w >= vis_thresh, w, 0.0)
                X = triangulate(Pn[[u, v]], o, w)
                e = reproj_err(Pn[[u, v]], fpx[[u, v]], np.nan_to_num(X), o).mean(1)
                e = np.where(np.isfinite(X[:, 0]), e, np.nan).reshape(len(frames), len(J))
                with quiet():
                    m = np.nanmedian(e, 1)
                best = np.fmin(best, np.where(np.isfinite(m), m, np.inf))
            dis[u, v, frames] = dis[v, u, frames] = best
    subsets = sorted((sub for n in range(V, 1, -1)
                      for sub in combinations(range(V), n)),
                     key=len, reverse=True)
    keep = np.zeros((V, T), bool)
    for t in range(T):
        dv = np.flatnonzero(det[:, t])
        if len(dv) < 2:
            continue
        best_sub, best_cost = None, np.inf
        for sub in subsets:
            if best_sub is not None and len(sub) < len(best_sub):
                break
            if not all(det[i, t] for i in sub):
                continue
            pd = [dis[i, j, t] for k, i in enumerate(sub) for j in sub[k + 1:]]
            if max(pd) < max_med_px and sum(pd) < best_cost:
                best_sub, best_cost = sub, sum(pd)
        if best_sub is not None:
            keep[list(best_sub), t] = True
    return keep


def plausible_2d(kp: np.ndarray) -> np.ndarray:
    """(...,) bool: the detection is anatomically plausible for an upright rig.

    Back views from high cameras make MediaPipe fold the skeleton (the "face"
    lands on the hips). With upright (unrolled) cameras the nose must be above
    the hip midpoint in the image (smaller y); the eyes/ears likewise.
    """
    hip_y = 0.5 * (kp[..., 23, 1] + kp[..., 24, 1])
    with quiet():
        head_y = np.nanmedian(kp[..., [0, 2, 5, 7, 8], 1], axis=-1)
        return ~(head_y >= hip_y)


def reconstruct_take(cams: list[Camera], kp: np.ndarray, vis: np.ndarray, *,
                     fix_swaps: bool = True, consensus: bool = True,
                     consensus_px: float = 20.0, upright_check: bool = True,
                     **kw) -> TriangulationResult:
    """Full robust reconstruction of pixel keypoints (V,T,33,2) (nan = missing)."""
    xy = np.stack([c.normalize(kp[v]) for v, c in enumerate(cams)])
    vis = np.asarray(vis, np.float64).copy()
    keep = np.isfinite(xy[..., 0, 0])
    if upright_check:
        keep &= plausible_2d(kp)
        xy[~keep] = np.nan
        vis[~keep] = 0.0
    if consensus:
        keep = consensus_views(cams, xy, vis, max_med_px=consensus_px)
        xy[~keep] = np.nan
        vis[~keep] = 0.0
    if fix_swaps:
        swaps = resolve_swaps(cams, xy, vis)
        xy = apply_swaps(xy, swaps)
        vis = apply_swaps(vis, swaps)
    else:
        swaps = np.zeros(kp.shape[:2] + (3,), bool)
    X, valid, nv, err, used = robust_triangulate(cams, xy, vis, **kw)
    return TriangulationResult(X, valid, nv, err, used, swaps, keep)
