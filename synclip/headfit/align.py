"""Align an arbitrary head mesh to the GNM frame (metres, +Y up, +Z forward).

No assumption is made about the input's orientation, units or up axis:

1. centre + RMS-normalise;
2. estimate the bilateral symmetry plane (PCA candidates scored by
   mirror-ICP) and enumerate 8 proper-rotation candidate views around it;
3. render each candidate frontally, run MediaPipe, keep the most confident;
   re-render from the landmark-derived frontal pose at 0 / +-25 deg yaw and take
   the per-landmark median of the back-projected 3D points;
4. Umeyama similarity (STABLE_IDX landmarks) to the GNM mean landmarks, which
   fixes the scale to metres;
5. rigid ICP (no scale) of the face region onto the GNM mean face region,
   regularised by the landmark pairs.

Fallback when no face is detected: similarity ICP of the GNM mean face against
every candidate orientation; the lowest residual wins (``used_fallback``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import landmarks as L

# GNM vertex groups ---------------------------------------------------------
_FACE_INCLUDE = ("skin_exterior",)
_FACE_EXCLUDE = ("eyes", "eye_sockets", "eye_interiors", "mouth_sock", "teeth", "gums", "tongue", "ears")


@dataclass
class AlignResult:
    mesh: trimesh.Trimesh  # target in the GNM frame (metres)
    transform: np.ndarray  # (4,4) similarity: original target -> GNM frame
    scale: float
    target_landmarks: dict[int, np.ndarray]  # mp_idx -> (3,) GNM frame
    gnm_landmarks: dict[int, tuple[int, np.ndarray]]  # mp_idx -> (tri, bary) on GNM mesh
    confidence: float
    used_fallback: bool
    log: list[str] = field(default_factory=list)
    # outer + inner lip contours (L.LIP_IDX, detected ones only; empty on fallback), used
    # only by the detail fit, never for alignment
    lip_landmarks: dict[int, np.ndarray] = field(default_factory=dict)
    gnm_lip_landmarks: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)


# ------------------------------------------------------------------ helpers

def _similarity(T_lin: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = T_lin
    M[:3, 3] = t
    return M


def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ M[:3, :3].T + M[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True, w: np.ndarray | None = None):
    """Least-squares ``dst ~ s R src + t`` (proper rotation, never a mirror)."""
    w = np.ones(len(src)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd * w[:, None]).T @ xs
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = float(w @ (xs**2).sum(1))
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def _icp_point(src: np.ndarray, dst: np.ndarray, iters: int = 30, trim: float = 0.8,
               with_scale: bool = False, tree: cKDTree | None = None):
    """Trimmed point-to-point ICP of ``src`` onto ``dst``. Returns (M 4x4, residual)."""
    tree = tree or cKDTree(dst)
    M = np.eye(4)
    cur = src.copy()
    res = np.inf
    for _ in range(iters):
        d, j = tree.query(cur)
        keep = d <= np.quantile(d, trim)
        s, R, t = umeyama(src[keep], dst[j[keep]], with_scale)
        M = _similarity(s * R, t)
        cur = _apply(M, src)
        new = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(res - new) < 1e-7:
            res = new
            break
        res = new
    d, _ = tree.query(cur)
    keep = d <= np.quantile(d, trim)
    return M, float(np.sqrt(np.mean(d[keep] ** 2)))


def _sample(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:  # very old trimesh
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def _symmetry_plane(pts: np.ndarray, log: list[str]) -> tuple[np.ndarray, float]:
    """Return (unit normal n, offset d) of the plane n.x = d (pts are centred)."""
    _, _, Vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    tree = cKDTree(pts)
    sub = pts[:: max(1, len(pts) // 4000)]
    best = None
    for k, n in enumerate(Vt):
        H = np.eye(3) - 2 * np.outer(n, n)
        M, res = _icp_point(sub @ H.T, pts, iters=40, trim=0.9, tree=tree)
        A = M[:3, :3] @ H  # improper isometry ~ reflection about the true plane
        w, V = np.linalg.eigh((A + A.T) / 2)
        n_ref = V[:, 0]
        d = float(n_ref @ M[:3, 3]) / 2.0
        log.append(f"symmetry candidate PCA{k}: mirror-ICP rms {res:.4f} (normalised units)")
        if best is None or res < best[0]:
            best = (res, n_ref / np.linalg.norm(n_ref), d)
    log.append(f"symmetry plane normal {np.round(best[1], 3).tolist()} (rms {best[0]:.4f})")
    return best[1], best[2]


def _candidates(pts: np.ndarray, n: np.ndarray) -> list[np.ndarray]:
    """8 proper rotations (world->view) whose X axis lies along +-n."""
    P = np.eye(3) - np.outer(n, n)
    q = (pts - pts.mean(0)) @ P
    _, _, Vt = np.linalg.svd(q, full_matrices=False)
    a, b = Vt[0], Vt[1]
    a = a - (a @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    out = []
    for f_ax, u_ax in ((a, b), (b, a)):
        for fs in (1, -1):
            for us in (1, -1):
                f, u = fs * f_ax, us * u_ax
                x = np.cross(u, f)
                R = np.stack([x, u, f])
                assert np.linalg.det(R) > 0
                out.append(R)
    return out


def _gnm_face_mask(npz_path=None) -> np.ndarray:
    d = np.load(npz_path or L.GNM_NPZ)
    names = [str(s) for s in d["vertex_group_names"]]
    g = d["vertex_groups"]
    m = np.zeros(g.shape[1], bool)
    for nm in _FACE_INCLUDE:
        m |= g[names.index(nm)] > 0.5
    for nm in _FACE_EXCLUDE:
        if nm in names:
            m &= ~(g[names.index(nm)] > 0.5)
    return m


def _rigid_icp_with_landmarks(src, dst_pts, dst_nrm, lm_src, lm_dst, iters=40,
                              max_dist=0.012, trim=0.85, lm_weight=0.5):
    """Point-to-plane rigid ICP of src -> (dst_pts, dst_nrm) plus landmark pairs.

    Returns 4x4 rigid transform applied to src.
    """
    tree = cKDTree(dst_pts)
    M = np.eye(4)
    n_lm = max(len(lm_src), 1)
    for it in range(iters):
        cur = _apply(M, src)
        lm = _apply(M, lm_src)
        d, j = tree.query(cur)
        thr = min(max_dist, float(np.quantile(d, trim)))
        k = d <= thr
        if k.sum() < 50:
            break
        p, q, nn = cur[k], dst_pts[j[k]], dst_nrm[j[k]]
        # point-to-plane rows: [p x n, n] . [w, t] = n.(q - p)
        A1 = np.hstack([np.cross(p, nn), nn])
        b1 = np.einsum("ij,ij->i", nn, q - p)
        # landmark rows (point-to-point), weighted to lm_weight of the total
        wl = np.sqrt(lm_weight * k.sum() / n_lm)
        A2, b2 = [], []
        for ax in range(3):
            e = np.zeros(3)
            e[ax] = 1.0
            A2.append(np.hstack([np.cross(lm, e), np.tile(e, (len(lm), 1))]))
            b2.append((lm_dst - lm)[:, ax])
        A = np.vstack([A1] + [wl * a for a in A2])
        b = np.concatenate([b1] + [wl * bb for bb in b2])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        Rinc = Rotation.from_rotvec(x[:3]).as_matrix()
        Minc = _similarity(Rinc, x[3:])
        M = Minc @ M
        if np.linalg.norm(x) < 1e-7:
            break
    return M


def _rmse_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))) * 1000.0


# --------------------------------------------------------------------- main

def align(target: trimesh.Trimesh, npz_path=None, size: int = 512, n_samples: int = 30000,
          face_radius: float = 0.03) -> AlignResult:
    log: list[str] = []
    target = trimesh.Trimesh(np.asarray(target.vertices, dtype=np.float64), np.asarray(target.faces), process=False)

    # (1) centre + RMS normalise ------------------------------------------
    pts0 = _sample(target, n_samples)
    c = pts0.mean(0)
    r = float(np.sqrt(np.mean(np.sum((pts0 - c) ** 2, axis=1))))
    N = _similarity(np.eye(3) / r, -c / r)  # original -> normalised
    tn = target.copy()
    tn.apply_transform(N)
    pts = _apply(N, pts0)
    log.append(f"normalised: centroid {np.round(c, 4).tolist()}, rms radius {r:.4g}")

    # (2) symmetry plane + candidate orientations --------------------------
    n_sym, _ = _symmetry_plane(pts, log)
    cands = _candidates(pts, n_sym)

    # GNM mean + its landmarks
    gnm = L.load_gnm_mean(npz_path)
    gnm_lm_all = L.gnm_mean_landmarks(npz_path)
    idx = [i for i in L.STABLE_IDX if i in gnm_lm_all]
    gnm_lm_pts = L.bary_points(gnm.vertices, gnm.faces, gnm_lm_all, idx)

    # (3) frontal detection on each candidate -----------------------------
    best = None
    for ci, R in enumerate(cands):
        p, conf, _ = L.detect_view(tn, R, size)
        if p is None:
            log.append(f"candidate {ci}: no face")
            continue
        sel = p[idx]
        ok = np.isfinite(sel).all(1)
        log.append(f"candidate {ci}: conf {conf:.3f}, {ok.sum()}/{len(idx)} landmarks on surface")
        if ok.sum() < 0.6 * len(idx):
            continue
        if best is None or conf > best[0]:
            best = (conf, ci, p)

    used_fallback = best is None or best[0] < 0.15
    if not used_fallback:
        conf, ci, p = best
        log.append(f"best candidate {ci} (conf {conf:.3f})")
        # frontal pose from landmarks (handles rolled / tilted candidates)
        sel = p[idx]
        ok = np.isfinite(sel).all(1)
        _, Rf, _ = umeyama(sel[ok], gnm_lm_pts[ok])
        med, conf2, _ = L.multiview_landmarks(tn, Rf, size=size)
        if med is None:
            log.append("refined frontal view failed; using single-view landmarks")
            med = p
        else:
            conf = max(conf, conf2)
        tl = med[idx]
        ok = np.isfinite(tl).all(1)
        if ok.sum() < 0.6 * len(idx):
            used_fallback = True
            log.append("too few refined landmarks -> fallback")

    if not used_fallback:
        # (4) Umeyama with one round of outlier rejection ------------------
        s, R, t = umeyama(tl[ok], gnm_lm_pts[ok])
        res = np.linalg.norm((s * tl[ok] @ R.T + t) - gnm_lm_pts[ok], axis=1)
        inl = np.zeros(len(idx), bool)
        inl[np.nonzero(ok)[0]] = res <= max(2.5 * np.median(res), 0.004)
        s, R, t = umeyama(tl[inl], gnm_lm_pts[inl])
        U = _similarity(s * R, t)
        lm_aligned = _apply(U, tl)
        rm_u = _rmse_mm(lm_aligned[inl], gnm_lm_pts[inl])
        log.append(f"umeyama: scale {s / r:.6g} (orig units -> m), {inl.sum()}/{len(idx)} inliers, "
                   f"landmark RMSE {rm_u:.2f} mm")

        # (5) rigid ICP on the face region ---------------------------------
        face = _gnm_face_mask(npz_path)
        tree_lm = cKDTree(gnm_lm_pts[inl])
        dg, _ = tree_lm.query(gnm.vertices)
        gmask = face & (dg < face_radius)
        tgt_pts = _apply(U, pts)
        dt, _ = cKDTree(lm_aligned[inl]).query(tgt_pts)
        src = tgt_pts[dt < face_radius]
        Mi = _rigid_icp_with_landmarks(src, gnm.vertices[gmask], gnm.vertex_normals[gmask],
                                       lm_aligned[inl], gnm_lm_pts[inl])
        lm_final = _apply(Mi, lm_aligned)
        rm_i = _rmse_mm(lm_final[inl], gnm_lm_pts[inl])
        d_icp, _ = cKDTree(gnm.vertices[gmask]).query(_apply(Mi, src))
        log.append(f"icp: {len(src)} target pts vs {gmask.sum()} GNM face verts, "
                   f"median surface dist {np.median(d_icp) * 1000:.2f} mm, landmark RMSE {rm_i:.2f} mm")
        T = Mi @ U @ N
        target_lm = {i: lm_final[k] for k, i in enumerate(idx) if inl[k]}
        gnm_lm = {i: gnm_lm_all[i] for i in target_lm}
        lip = [i for i in L.LIP_IDX if i in gnm_lm_all and np.isfinite(med[i]).all()]
        lip_pts = _apply(Mi @ U, med[lip]) if lip else np.zeros((0, 3))
        lip_lm = {i: lip_pts[k] for k, i in enumerate(lip)}
        log.append(f"lip contour landmarks: {len(lip)}/{len(L.LIP_IDX)}")
        confidence = float(conf)
    else:
        # Fallback: similarity ICP of the GNM face against each candidate ----
        log.append("no reliable face detection -> ICP fallback over candidate orientations")
        face = _gnm_face_mask(npz_path)
        gskin = gnm.vertices[face]
        gc = gnm.vertices.mean(0)
        gr = float(np.sqrt(np.mean(np.sum((gnm.vertices - gc) ** 2, axis=1))))
        gsrc = (gskin - gc) / gr
        tree = cKDTree(pts)
        bestf = None
        for ci, R in enumerate(cands):
            # place the GNM face into candidate frame R (view frame == GNM frame),
            # tops of heads matched
            src = gsrc @ R  # GNM frame -> normalised target frame (R^T)
            up = R[1]
            shift = up * (float((pts @ up).max()) - float(((gnm.vertices - gc) / gr @ np.array([0, 1, 0])).max()))
            M0 = _similarity(np.eye(3), shift)
            M, res = _icp_point(_apply(M0, src), pts, iters=50, trim=0.7, with_scale=True, tree=tree)
            log.append(f"fallback candidate {ci}: rms {res:.4f}")
            if bestf is None or res < bestf[0]:
                bestf = (res, M @ M0, R)
        res, Mg, R = bestf
        # GNM(normalised, rotated by R^T) -> target(normalised): x_t = Mg (R^T g)
        # invert to get target(normalised) -> GNM metres.
        A = Mg[:3, :3] @ R.T / gr  # maps (g - gc) -> x_t
        Ainv = np.linalg.inv(A)
        G = _similarity(Ainv, gc - Ainv @ Mg[:3, 3])
        T = G @ N
        s = float(np.cbrt(np.linalg.det(T[:3, :3])))
        tmp = target.copy()
        tmp.apply_transform(T)
        gpts = L.bary_points(gnm.vertices, gnm.faces, gnm_lm_all, idx)
        closest, _, _ = trimesh.proximity.closest_point(tmp, gpts)
        target_lm = {i: closest[k] for k, i in enumerate(idx)}
        gnm_lm = {i: gnm_lm_all[i] for i in idx}
        lip_lm = {}
        confidence = 0.0
        log.append(f"fallback: best candidate rms {res:.4f}, scale {s:.6g}, "
                   f"landmark RMSE {_rmse_mm(closest, gpts):.2f} mm")

    scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    Rcheck = T[:3, :3] / scale
    assert np.linalg.det(Rcheck) > 0 and np.allclose(Rcheck @ Rcheck.T, np.eye(3), atol=1e-6)
    out = target.copy()
    out.apply_transform(T)
    return AlignResult(mesh=out, transform=T, scale=scale, target_landmarks=target_lm,
                       gnm_landmarks=gnm_lm, confidence=confidence, used_fallback=used_fallback, log=log,
                       lip_landmarks=lip_lm, gnm_lip_landmarks={i: gnm_lm_all[i] for i in lip_lm})
