"""Non-rigid GNM fit to an (already roughly aligned) target head mesh.

1. Identity (alternating optimisation):
     * identity solve (closed form, Cholesky): forward point-to-plane ICP on the
       GNM face mask + reverse point-to-plane term (target -> GNM face) +
       optional landmark term (decaying weight) + Tikhonov prior lam*||id||^2;
     * rigid-only (no scale) re-alignment of the target to the current GNM shape.
   Coarse-to-fine over the number of active head PCs (10 -> 40 -> 170) with a
   decaying prior weight.
2. Expression (optional): the 170 head PCs and the lower-face + eye-region
   expression PCs (350; tongue/pupils fixed at 0) are solved jointly, with a
   prior on the mean squared face displacement (mm^2) the expression causes.
3. Shape recovery (after 2): identity is refitted with expression held at 0
   against the same target and frame, so shape the expression absorbed goes into
   the neutral export instead of being dropped.
4. Detail offset (optional): a per-vertex template offset D (V,3) solved as a
   sparse least-squares problem (point-to-plane data on the face mask +
   Laplacian smoothness + small damping) on top of the neutral shape.  Eyeballs,
   teeth, gums and tongue get a rigid translation following the surrounding skin.
   Lips (see ``LipFit``): the identity stages' point-to-plane data lets the GNM
   vermilion border slide along the surface and the landmark term has decayed
   away by then, so the lips come out thin.  The detail solve therefore adds the
   lips (incl. the seam row) to the data mask with heavier rows and a softer
   Laplacian, point-to-point MediaPipe lip contours (outer + inner), a reverse
   term from the target's lips onto the GNM lips and ties the paired upper/lower
   seam rows together.
   Features (see ``FeatureDetail``): curvature-adaptive Laplacian weight on the
   face and a face-wide two-sided term, so creases and ridges survive.
Ears (see ``EarFit``): robust ear correspondences are added to the identity and
detail solves; occluded or missing ears get no ear data.

Pose = 0, eye/teeth identity PCs fixed at 0.  All residuals are expressed in
millimetres internally so prior weights are in "mm^2 per unit PC^2" (PCs are
sqrt-eigenvalue scaled, prior ~ N(0,1)).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

from . import gnm_data

MM = 1000.0


@dataclasses.dataclass
class Stage:
    n_pcs: int
    lam: float          # Tikhonov weight (mm^2 per unit PC^2, per-term normalised)
    iters: int


DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage(10, 1.0, 3),
    Stage(40, 0.1, 4),
    Stage(170, 0.01, 6),
)


@dataclasses.dataclass
class FitResult:
    id_full253: np.ndarray            # (253,) GNM identity vector (eyes/teeth = 0)
    id_head170: np.ndarray            # (170,)
    verts: np.ndarray                 # (V,3) neutral export: template + D + identity (expr = 0), metres
    R: np.ndarray                     # (3,3) rigid refinement applied to target
    t: np.ndarray                     # (3,)  target_refined = target @ R.T + t
    log: list                         # per-iteration dicts
    residuals: dict                   # final forward residual stats of ``verts`` (mm)
    runtime_s: float = 0.0
    expression: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(gnm_data.N_EXPR))
    #                                   (383,) fitted expression (reference only; the export is neutral)
    id_head170_expr: np.ndarray | None = None   # identity of the joint identity+expression fit
    expr_verts: np.ndarray | None = None        # (V,3) joint identity+expression fit (no D)
    expr_stats: dict = dataclasses.field(default_factory=dict)
    template_offset: np.ndarray | None = None   # (V,3) detail offset D added to the template
    detail_stats: dict = dataclasses.field(default_factory=dict)

    @property
    def joints(self) -> np.ndarray:
        """(J,3) bind-pose joints of the neutral export (template joints moved by
        joint_regressor @ D)."""
        return gnm_data.load().joints(self.id_full253, self.template_offset)

    def transform_target(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts) @ self.R.T + self.t

    @property
    def rigid_4x4(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m


@dataclasses.dataclass(frozen=True)
class LipFit:
    """Lip terms of the detail-offset solve (weights are relative to the face data)."""
    data_weight: float = 4.0     # data-row multiplier on the lips + 1 ring
    smooth: float = 0.1          # Laplacian weight multiplier on the lips + 1 ring (2-ring blend)
    w_contour: float = 0.3       # point-to-point MediaPipe outer + inner lip contours
    w_reverse: float = 1.0       # target lip samples -> nearest GNM lip vertex (point-to-plane)
    w_seam: float = 1.0          # paired upper/lower seam-row offsets kept equal

    def scaled(self, k: float) -> "LipFit":
        """All lip terms scaled by strength k (0 = like the rest of the face, 1 = unchanged)."""
        return dataclasses.replace(self, data_weight=1 + (self.data_weight - 1) * k, smooth=self.smooth ** k,
                                   w_contour=self.w_contour * k, w_reverse=self.w_reverse * k,
                                   w_seam=self.w_seam * k)


@dataclasses.dataclass(frozen=True)
class FeatureDetail:
    """Feature-aware terms of the detail-offset solve (face and lips).

    The Laplacian weight of each face vertex is scaled by ``1 / (1 + (H / h0)^2)``
    (clipped to ``floor``), H = mean curvature (1/m) of the target under the vertex,
    so creases and ridges (nose, brows, nasolabial fold, chin) are smoothed less.  A
    face-wide two-sided term pulls the nearest interior GNM face vertex onto every
    target face sample (point-to-plane), so concave features get filled in too."""
    h0: float = 40.0             # curvature scale (1/m); 40 ~ a 25 mm radius
    floor: float = 0.1           # lowest Laplacian weight multiplier
    radius: float = 0.003        # ball radius (m) of the target mean-curvature measure
    w_two_sided: float = 1.0     # target face samples -> nearest GNM face vertex (point-to-plane)
    n_samples: int = 80_000      # target samples for the two-sided term
    max_dist: float = 0.008      # two-sided pairs further apart than this (m) are dropped


FEATURE_DETAIL = {"off": None, "moderate": FeatureDetail()}


@dataclasses.dataclass(frozen=True)
class EarFit:
    """Ear terms (GNM ``ears`` group) of the identity and detail-offset solves.

    Ear correspondences are robust to ears that are occluded (hair, caps), cropped or
    missing: a pair is kept only if it is closer than ``max_dist``, the normals agree,
    the target point's nearest GNM skin vertex is an ear vertex and it is not in the
    worst ``trim`` fraction.  A side with fewer than ``min_coverage`` of its outer ear
    vertices (> 8 mm from the rest of the skin) paired is treated as missing and gets
    no ear data at all."""
    data_weight: float = 1.0     # per-row weight of the ear data relative to a face data row
    smooth: float = 1.0          # Laplacian weight multiplier on the ears (+1 ring)
    identity_weight: float = 0.5  # per-row weight of the ear data in the identity solves (0 = off)
    identity_max_dist: float = 0.012  # m, pair distance cap in the identity solves
    max_dist: float = 0.006      # m, pair distance cap in the detail solve
    trim: float = 0.1
    min_coverage: float = 0.4


# --------------------------------------------------------------------------- utils
def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _rotvec_to_R(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class _TargetSampler:
    """Dense surface samples of the target (original frame) with face normals."""

    def __init__(self, mesh, n_samples: int, seed: int = 0):
        import trimesh
        # trimesh >= 4 ignores np.random.seed (its own Generator): pass seed explicitly
        try:
            pts, fidx = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        except TypeError:  # old trimesh: global numpy RNG
            rng_state = np.random.get_state()
            np.random.seed(seed)
            try:
                pts, fidx = trimesh.sample.sample_surface(mesh, n_samples)
            finally:
                np.random.set_state(rng_state)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
        fn = np.asarray(mesh.face_normals, dtype=np.float64)
        self.points = np.vstack([np.asarray(pts, np.float64), v])
        self.normals = np.vstack([fn[fidx], vn])
        self.tree = cKDTree(self.points)


def _trim(d: np.ndarray, ok: np.ndarray, frac: float) -> np.ndarray:
    """ok without the ``frac`` largest distances among the ok entries."""
    ok = ok.copy()
    if ok.sum() > 10 and frac > 0:
        ok &= d <= np.quantile(d[ok], 1.0 - frac)
    return ok


def _bary_points(verts, tris, tri_idx, bary):
    return np.einsum("lk,lkd->ld", bary, verts[tris[tri_idx]])


# --------------------------------------------------------------------------- fit
def fit(target_mesh,
        target_landmarks: Mapping[int, Sequence[float]] | None = None,
        gnm_landmarks: Mapping[int, tuple] | None = None,
        *,
        stages: Sequence[Stage] = DEFAULT_STAGES,
        init_id: np.ndarray | None = None,
        w_fwd: float = 1.0,
        w_p2p: float = 0.1,
        w_rev: float = 0.3,
        w_lmk: float = 1.0,
        lmk_decay: float = 0.7,
        rev_radius: float = 0.01,
        rev_trim: float = 0.10,
        max_normal_angle_deg: float = 60.0,
        dist_reject_factor: float = 3.0,
        rigid_iters: int = 2,
        refine_rigid: bool = True,
        n_samples: int = 200_000,
        fit_expression: bool = True,
        expr_prior: float = 0.1,
        expr_iters: int = 8,
        recover_iters: int = 4,
        detail: bool = True,
        detail_smooth: float = 1.0,
        detail_damping: float = 1e-4,
        detail_iters: int = 4,
        lip_landmarks: Mapping[int, Sequence[float]] | None = None,
        gnm_lip_landmarks: Mapping[int, tuple] | None = None,
        lips: LipFit | None = LipFit(),
        features: FeatureDetail | None = FeatureDetail(),
        ears: EarFit | None = EarFit(),
        stiffen_rounds: int = 3,
        stiffen_factor: float = 10.0,
        stiffen_membrane: float = 1e-3,
        pose_fn: Callable[[np.ndarray, np.ndarray | None], Sequence[np.ndarray]] | None = None,
        verbose: bool = False) -> FitResult:
    """Fit GNM (identity, optionally expression and a detail offset) to ``target_mesh``.

    expr_prior: weight on the mean squared face displacement (mm^2) of the expression.
    detail_smooth: Laplacian smoothness weight alpha of the detail offset (larger = smoother).
        Flips are first removed by local stiffening, the rest by translating patches
        rigidly (see ``_unflip``).
    lips: lip terms of the detail solve (None = treat the lips like the rest of the face).
    features: feature-aware detail terms (curvature-adaptive Laplacian weight on the face
        and lips, face-wide two-sided term; None = off).  The lip terms keep their own
        Laplacian weight where it is lower.
    ears: fit the GNM ``ears`` group too (identity and detail; robust to occluded or
        missing ears, see ``EarFit``; None = ears follow the head only through the
        identity and the smoothness term).
    lip_landmarks / gnm_lip_landmarks: MediaPipe lip contour points on the target and
        {key: (tri_idx, bary)} on GNM, like target/gnm_landmarks; used only by ``lips``.
    stiffen_rounds / stiffen_factor / stiffen_membrane: before the rigid-patch guard, the
        edges within 2 rings of triangles the offset flips get an edge-strain penalty
        w * |D_i - D_j|^2 / l0^2 (w starts at ``stiffen_membrane`` and is multiplied by
        ``stiffen_factor`` on every further round) and the offset is re-solved (warm, 2
        iterations), up to ``stiffen_rounds`` times.  Slivers (tiny l0) at the lip corners
        get stiff while the rest of the lip stays free.
    pose_fn: (id_full253, D or None) -> posed vertex arrays (e.g. jawOpen, smile; see
        ``evaluate.gnm_pose_fn``).  The flip guard then also checks every posed mesh
        against the same pose without D, not only the neutral bind pose.

    target_mesh: trimesh.Trimesh already in the GNM frame (metres, +Y up, +Z fwd),
        roughly rigidly aligned to the GNM mean.
    target_landmarks: {key: (3,)} landmark positions on the target (same frame).
    gnm_landmarks: {key: (tri_idx, bary(3,))} on GNM triangles. Only keys present
        in both dicts are used.
    """
    t0 = time.time()
    G = gnm_data.load()
    tris = G.triangles
    face = G.face_mask()
    face_idx = np.nonzero(face)[0]
    BT = _basis_T(G)                                  # (V,3,170) float32

    # interior face vertices (exclude the mask boundary ring) for reverse term
    interior = face & ~G.dilate(~face, 1)
    interior_idx = np.nonzero(interior)[0]

    tgt = _TargetSampler(target_mesh, n_samples)
    rng = np.random.default_rng(0)
    rev_sub = rng.choice(len(tgt.points), size=min(40_000, len(tgt.points)), replace=False)

    # landmarks
    lm_keys = []
    if target_landmarks and gnm_landmarks:
        lm_keys = [k for k in gnm_landmarks if k in target_landmarks]
    if lm_keys:
        lm_tri = np.array([int(gnm_landmarks[k][0]) for k in lm_keys])
        lm_bary = np.array([np.asarray(gnm_landmarks[k][1], float) for k in lm_keys])
        lm_bary /= lm_bary.sum(1, keepdims=True)
        lm_tgt0 = np.array([np.asarray(target_landmarks[k], float) for k in lm_keys])
        lm_vid = tris[lm_tri]                          # (L,3)

    n_max = max(s.n_pcs for s in stages)
    idv = np.zeros(gnm_data.N_HEAD)
    if init_id is not None:
        idv[:len(init_id)] = init_id
    R = np.eye(3)
    t = np.zeros(3)
    log: list[dict] = []
    cos_thr = np.cos(np.deg2rad(max_normal_angle_deg))
    it_global = 0

    def forward_pairs(V, N, face_idx=face_idx):
        """GNM face verts -> closest target sample. Returns idx, q, nq (GNM frame)."""
        x = V[face_idx]
        xq = (x - t) @ R                               # into original target frame
        d, j = tgt.tree.query(xq, workers=-1)
        q = tgt.points[j] @ R.T + t
        nq = tgt.normals[j] @ R.T
        ok = np.einsum("ij,ij->i", nq, N[face_idx]) > cos_thr
        med = np.median(d[ok]) if ok.any() else np.inf
        ok &= d < dist_reject_factor * max(med, 1e-4)
        return face_idx[ok], q[ok], nq[ok], d

    def reverse_pairs(V, N, interior_idx=interior_idx):
        """Target samples near the face -> nearest interior GNM face vertex."""
        p = tgt.points[rev_sub] @ R.T + t
        pn = tgt.normals[rev_sub] @ R.T
        tree = cKDTree(V[interior_idx])
        d, j = tree.query(p, distance_upper_bound=rev_radius, workers=-1)
        ok = np.isfinite(d)
        vid = interior_idx[np.where(ok, j, 0)]
        ok &= np.einsum("ij,ij->i", pn, N[vid]) > cos_thr
        vid, p, d = vid[ok], p[ok], d[ok]
        if len(d) > 10:
            keep = d <= np.quantile(d, 1.0 - rev_trim)
            vid, p = vid[keep], p[keep]
        return vid, p

    # ---------------- ears (see EarFit) ----------------
    ear_mask = G.group("ears") if ears is not None else np.zeros(G.n_verts, bool)
    ear_idx = np.nonzero(ear_mask)[0]
    ext_idx = np.nonzero(G.group("skin_exterior"))[0]
    ear_sides = [G.template[ear_idx, 0] > 0, G.template[ear_idx, 0] <= 0]
    # ear vertices that stand off the head (helix, antihelix, lobe; > 8 mm from the
    # rest of the skin on the template): only these decide whether an ear is present,
    # since a cropped or covered ear's root still finds pairs on the side of the head
    head_skin = G.group("skin_exterior") & ~ear_mask
    ear_out = (cKDTree(G.template[head_skin]).query(G.template[ear_idx])[0] > 0.008) if len(ear_idx) else ear_idx
    ear_last: dict = {}

    def ear_rows(V, N, w_row, max_dist):
        """Robust ear data rows [(per-row weight, vid, target normal, target point)]:
        forward (GNM ear vertex -> closest target sample) and reverse (target sample ->
        its nearest GNM exterior skin vertex, kept only if that is an ear vertex)."""
        if not len(ear_idx):
            return []
        ext_tree = cKDTree(V[ext_idx])
        d, j = tgt.tree.query((V[ear_idx] - t) @ R, distance_upper_bound=max_dist, workers=-1)
        ok = np.isfinite(d)
        j = np.where(ok, j, 0)
        q, nq = tgt.points[j] @ R.T + t, tgt.normals[j] @ R.T
        ok &= np.einsum("ij,ij->i", nq, N[ear_idx]) > cos_thr
        if ok.any():  # mutual: the target point must lie closest to the GNM ear
            ok[ok] = ear_mask[ext_idx[ext_tree.query(q[ok], workers=-1)[1]]]
        ok = _trim(d, ok, ears.trim)
        keep_side = np.zeros(len(ear_idx), bool)
        cover = []
        for side in ear_sides:
            cover.append(float(ok[side & ear_out].mean()))
            keep_side |= side & (cover[-1] >= ears.min_coverage)
        ok &= keep_side
        # reverse: target samples near the GNM ears
        p = tgt.points @ R.T + t
        dv, _ = cKDTree(V[ear_idx]).query(p, distance_upper_bound=max_dist, workers=-1)
        near = np.nonzero(np.isfinite(dv))[0]
        dr, k = ext_tree.query(p[near], workers=-1)
        vid = ext_idx[k]
        pn = tgt.normals[near] @ R.T
        ear_pos = np.searchsorted(ear_idx, vid).clip(0, len(ear_idx) - 1)
        okr = ear_mask[vid] & keep_side[ear_pos]
        okr &= np.einsum("ij,ij->i", pn, N[vid]) > cos_thr
        okr = _trim(dr, okr, ears.trim)
        ear_last.update(n_fwd=int(ok.sum()), n_rev=int(okr.sum()), coverage=[round(c, 3) for c in cover])
        n_r = int(okr.sum())
        w_r = w_row * (w_rev / w_fwd) * len(ear_idx) / max(n_r, len(ear_idx))
        return [(w_row, ear_idx[ok], nq[ok], q[ok]), (w_r, vid[okr], pn[okr], p[near][okr])]

    def ear_normal_eq(B, rows, n):
        """Normal equations of the ear rows for a linear model T + B x (B: (V,3,n))."""
        H = np.zeros((n, n))
        g = np.zeros(n)
        for w_, vid, nrm, p in rows:
            if len(vid):
                A_ = np.einsum("fd,fdn->fn", nrm, B[vid, :, :n].astype(np.float64)) * MM
                b_ = np.einsum("fd,fd->f", nrm, p - G.template[vid]) * MM
                H += w_ * A_.T @ A_
                g += w_ * A_.T @ b_
        return H, g

    w_ear_id = 0.0 if ears is None else ears.identity_weight * w_fwd / len(face_idx)

    for si, st in enumerate(stages):
        n = min(st.n_pcs, n_max)
        for k in range(st.iters):
            ti = time.time()
            V = G.verts(idv)
            N = vertex_normals(V, tris)

            # ---------------- identity solve ----------------
            fv, q, nq, _ = forward_pairs(V, N)
            Bf = BT[fv, :, :n].astype(np.float64)                     # (F,3,n)
            # point-to-plane rows: nq . (T + B id) = nq . q
            Af = np.einsum("fd,fdn->fn", nq, Bf) * MM
            bf = np.einsum("fd,fd->f", nq, q - G.template[fv]) * MM
            # small point-to-point rows for tangential stability
            Ap = Bf.reshape(-1, n) * MM
            bp = (q - G.template[fv]).reshape(-1) * MM
            wf = w_fwd / len(fv)
            wp = w_p2p / len(fv)
            H = wf * Af.T @ Af + wp * Ap.T @ Ap
            g = wf * Af.T @ bf + wp * Ap.T @ bp

            rv, rp = reverse_pairs(V, N)
            n_rev = len(rv)
            if n_rev:
                Br = BT[rv, :, :n].astype(np.float64)
                nr = N[rv]
                Ar = np.einsum("fd,fdn->fn", nr, Br) * MM
                br = np.einsum("fd,fd->f", nr, rp - G.template[rv]) * MM
                wr = w_rev / n_rev
                H += wr * Ar.T @ Ar
                g += wr * Ar.T @ br

            if w_ear_id > 0:
                He, ge = ear_normal_eq(BT, ear_rows(V, N, w_ear_id, ears.identity_max_dist), n)
                H += He
                g += ge

            wl = 0.0
            if lm_keys:
                wl = w_lmk * (lmk_decay ** it_global)
                lt = lm_tgt0 @ R.T + t
                Bl = np.einsum("lk,lkdn->ldn", lm_bary, BT[lm_vid, :, :n].astype(np.float64))
                Tl = np.einsum("lk,lkd->ld", lm_bary, G.template[lm_vid])
                Al = Bl.reshape(-1, n) * MM
                bl = (lt - Tl).reshape(-1) * MM
                H += (wl / len(lm_keys)) * Al.T @ Al
                g += (wl / len(lm_keys)) * Al.T @ bl

            H[np.diag_indices(n)] += st.lam
            c = scipy.linalg.cho_factor(H)
            idv = np.zeros(gnm_data.N_HEAD)
            idv[:n] = scipy.linalg.cho_solve(c, g)
            V = G.verts(idv)
            N = vertex_normals(V, tris)

            # ---------------- rigid re-solve (target) ----------------
            if refine_rigid:
                for _ in range(rigid_iters):
                    R, t = _rigid_step(V, N, face_idx, tgt, R, t, cos_thr, dist_reject_factor)

            fv2, q2, nq2, d_all = forward_pairs(V, N)
            r_pl = np.abs(np.einsum("fd,fd->f", nq2, q2 - V[fv2])) * MM
            entry = dict(stage=si, n_pcs=n, lam=st.lam, iter=k, n_fwd=int(len(fv)),
                         n_rev=int(n_rev), w_lmk=float(wl),
                         fwd_p2plane_mean_mm=float(r_pl.mean()),
                         fwd_p2p_median_mm=float(np.median(d_all) * MM),
                         id_absmax=float(np.abs(idv).max()), id_norm=float(np.linalg.norm(idv)),
                         sec=round(time.time() - ti, 3))
            if lm_keys:
                lp = np.einsum("lk,lkd->ld", lm_bary, V[lm_vid])
                entry["lmk_rmse_mm"] = float(np.sqrt(((lp - (lm_tgt0 @ R.T + t)) ** 2).sum(1).mean()) * MM)
            log.append(entry)
            if verbose:
                print(entry)
            it_global += 1

    def all_pairs(V, N, face_idx=face_idx, interior_idx=interior_idx):
        fv, q, nq, _ = forward_pairs(V, N, face_idx)
        rv, rp = reverse_pairs(V, N, interior_idx)
        return fv, q, nq, rv, rp

    def linear_icp(B, Hprior, x0, iters, rigid, stage, prior_mean=None):
        """x = argmin data(T + B x) + (x - m)' Hprior (x - m) (ICP, re-paired each iter).
        B: (V,3,n) float32. Optionally re-solves the rigid target alignment."""
        nonlocal R, t
        n = B.shape[2]
        x = x0.copy()
        m0 = np.zeros(n) if prior_mean is None else prior_mean
        for k in range(iters):
            ti = time.time()
            V = G.template + B @ x
            N = vertex_normals(V, tris)
            fv, q, nq, rv, rp = all_pairs(V, N)
            Bf = B[fv].astype(np.float64)
            Af = np.einsum("fd,fdn->fn", nq, Bf) * MM
            bf = np.einsum("fd,fd->f", nq, q - G.template[fv]) * MM
            Ap = Bf.reshape(-1, n) * MM
            bp = (q - G.template[fv]).reshape(-1) * MM
            H = (w_fwd / len(fv)) * Af.T @ Af + (w_p2p / len(fv)) * Ap.T @ Ap + Hprior
            g = (w_fwd / len(fv)) * Af.T @ bf + (w_p2p / len(fv)) * Ap.T @ bp + Hprior @ m0
            if len(rv):
                Br = B[rv].astype(np.float64)
                Ar = np.einsum("fd,fdn->fn", N[rv], Br) * MM
                br = np.einsum("fd,fd->f", N[rv], rp - G.template[rv]) * MM
                H += (w_rev / len(rv)) * Ar.T @ Ar
                g += (w_rev / len(rv)) * Ar.T @ br
            if w_ear_id > 0:
                He, ge = ear_normal_eq(B, ear_rows(V, N, w_ear_id, ears.identity_max_dist), n)
                H += He
                g += ge
            x = scipy.linalg.cho_solve(scipy.linalg.cho_factor(H), g)
            V = G.template + B @ x
            N = vertex_normals(V, tris)
            if rigid and refine_rigid:
                for _ in range(rigid_iters):
                    R, t = _rigid_step(V, N, face_idx, tgt, R, t, cos_thr, dist_reject_factor)
            fv2, q2, nq2, d_all = forward_pairs(V, N)
            r_pl = np.abs(np.einsum("fd,fd->f", nq2, q2 - V[fv2])) * MM
            entry = dict(stage=stage, n_pcs=n, iter=k, n_fwd=int(len(fv)), n_rev=int(len(rv)),
                         fwd_p2plane_mean_mm=float(r_pl.mean()),
                         fwd_p2p_median_mm=float(np.median(d_all) * MM),
                         id_absmax=float(np.abs(x[:gnm_data.N_HEAD]).max()),
                         sec=round(time.time() - ti, 3))
            log.append(entry)
            if verbose:
                print(entry)
        return x

    expr = np.zeros(gnm_data.N_EXPR)
    expr_stats: dict = {}
    id_expr = expr_verts = None
    if fit_expression:
        # ---------------- joint identity + expression ----------------
        eidx = G.expression_indices()
        BE = np.ascontiguousarray(G.expression_basis[eidx].transpose(1, 2, 0))       # (V,3,E) f32
        B_all = np.concatenate([BT, BE], axis=2)
        BEf = BE[face_idx].astype(np.float64)
        gram = np.einsum("vdn,vdm->nm", BEf, BEf) * MM * MM / len(face_idx)       # mean sq. mm / coef
        Hp = scipy.linalg.block_diag(stages[-1].lam * np.eye(gnm_data.N_HEAD),
                                     expr_prior * gram + 1e-6 * np.eye(len(eidx)))
        x = linear_icp(B_all, Hp, np.concatenate([idv, np.zeros(len(eidx))]), expr_iters, True, "expr")
        id_expr = x[:gnm_data.N_HEAD]
        ex = x[gnm_data.N_HEAD:]
        expr[eidx] = ex
        expr_verts = G.template + B_all @ x
        disp = (BE @ ex) * MM                                                      # (V,3) mm
        dn = np.linalg.norm(disp[face_idx], axis=1)
        comp = np.linalg.norm(BE * ex, axis=1).max(0) * MM                        # per-PC max disp
        top = np.argsort(-comp)[:8]
        expr_stats = dict(n_pcs=int(len(eidx)), prior=float(expr_prior),
                          coef_norm=float(np.linalg.norm(ex)), coef_absmax=float(np.abs(ex).max()),
                          face_rms_disp_mm=float(np.sqrt((dn ** 2).mean())),
                          face_max_disp_mm=float(dn.max()),
                          top_components_max_disp_mm=[(G.expression_names[eidx[k]], round(float(comp[k]), 3))
                                                      for k in top])
        del B_all, BE, BEf
        # ---------------- shape recovery: identity with expression = 0 ----------------
        idv = linear_icp(BT, stages[-1].lam * np.eye(gnm_data.N_HEAD), id_expr, recover_iters, False,
                         "recover")

    Vn = G.verts(idv)
    D = None
    detail_stats: dict = {}
    if detail:
        ti = time.time()
        pair_fn, lip_terms = all_pairs, None
        dmask = face
        if lips is not None:
            lip_terms = _lip_terms(G, lips, tgt, R, t, lip_landmarks, gnm_lip_landmarks)
            dmask = face | lip_terms["lips"]
            didx = np.nonzero(dmask)[0]
            dint = np.nonzero(dmask & ~G.dilate(~dmask, 1))[0]
            pair_fn = lambda V, N: all_pairs(V, N, didx, dint)  # noqa: E731
        # per-vertex Laplacian weights and extra point-to-plane rows of the detail solve
        alpha_w = np.ones(G.n_verts) if lip_terms is None else lip_terms["alpha_w"].copy()
        row_fns = []
        if features is not None:
            # not on the lid margins: their sliver triangles would take most of the strain
            eye_open = G.group("eyes") | G.group("eye_interiors") | G.group("eye_sockets")
            fmask = (dmask | G.group("upper_lip") | G.group("lower_lip")) & ~G.dilate(eye_open, 3)
            alpha_w = np.minimum(alpha_w, _curvature_alpha(G, target_mesh, (Vn - t) @ R, fmask, features))
            fs = rng.choice(n_samples, size=min(features.n_samples, n_samples), replace=False)
            f_pts, f_nrm = tgt.points[fs] @ R.T + t, tgt.normals[fs] @ R.T
            f_set = np.nonzero(face & ~G.dilate(~face, 1))[0]
            row_fns.append(lambda V, N: [_nearest_vertex_rows(V, N, f_pts, f_nrm, f_set, features.w_two_sided,
                                                               features.max_dist)])
        if len(ear_idx):
            ez = G.dilate(ear_mask, 1)
            alpha_w[ez] = alpha_w[ez] * ears.smooth
            w_ear = ears.data_weight * w_fwd / len(face_idx)
            row_fns.append(lambda V, N: ear_rows(V, N, w_ear, ears.max_dist))
        terms = dict(alpha_w=alpha_w, rows=row_fns)
        edges = np.array(sp.triu(G.vertex_adjacency, 1).nonzero()).T            # (E,2)
        ew = np.zeros(len(edges))                                                 # membrane weights
        solve = lambda D0, it: _follow_rigid_parts(G, _solve_detail(  # noqa: E731
            G, Vn, pair_fn, detail_smooth, detail_damping, it, w_fwd, w_p2p, w_rev, lip_terms,
            D0=D0, membrane=(edges, ew), terms=terms), Vn)
        D = solve(None, detail_iters)
        posed = None if pose_fn is None else (lambda D_: pose_fn(G.to_full(idv), D_))  # noqa: E731
        ref = posed(None) if posed is not None else []
        n_stiff = 0
        for _ in range(stiffen_rounds):
            # stiffen the edges around triangles D flips (neutral or posed) and re-solve
            fl = _flips(G, Vn, D, ref, posed)
            if not len(fl):
                break
            seed = np.zeros(G.n_verts, bool)
            seed[tris[fl].ravel()] = True
            v = G.dilate(seed, 2)
            e_in = v[edges].all(1)
            ew[e_in] = np.where(ew[e_in] > 0, ew[e_in] * stiffen_factor, stiffen_membrane)
            n_stiff += 1
            D = solve(D, 2)
        D, n_fixed = _unflip(G, Vn, D, posed=posed)
        detail_stats, flipped = _offset_quality(G, Vn, D)
        n_posed = ([len(_flipped(tris, a, b)) for a, b in zip(posed(None), posed(D))]
                   if posed is not None else [])
        detail_stats.update(alpha=float(detail_smooth), damping=float(detail_damping),
                            n_unflip_verts=n_fixed, n_stiffen_rounds=n_stiff,
                            n_stiffened_edges=int((ew > 0).sum()), sec=round(time.time() - ti, 3))
        if posed is not None:
            detail_stats.update(flipped_triangles_posed=n_posed)
        if features is not None:
            detail_stats.update(features=dataclasses.asdict(features),
                                alpha_w_face_mean=float(alpha_w[face].mean()))
        if len(ear_idx):
            detail_stats.update(ears=dict(dataclasses.asdict(ears), **ear_last))
        if lip_terms is not None:
            detail_stats.update(lips=dataclasses.asdict(lips), n_lip_contour=int(len(lip_terms["contour_pts"])),
                                n_lip_reverse_samples=int(len(lip_terms["rev_pts"])),
                                n_seam_pairs=int(len(lip_terms["seam_pairs"])))
        log.append(dict(stage="detail", **detail_stats))
        if verbose:
            print(log[-1])
        if len(flipped) or sum(n_posed):
            print(f"warning: detail offset still flips {len(flipped)} triangles"
                  + (f" (posed: {n_posed})" if n_posed else ""))

    V = Vn if D is None else Vn + D
    N = vertex_normals(V, tris)
    _, _, _, d_all = forward_pairs(V, N)
    res = dict(fwd_mean_mm=float(d_all.mean() * MM), fwd_median_mm=float(np.median(d_all) * MM),
               fwd_p95_mm=float(np.quantile(d_all, 0.95) * MM))
    return FitResult(id_full253=G.to_full(idv), id_head170=idv, verts=V, R=R, t=t,
                     log=log, residuals=res, runtime_s=time.time() - t0,
                     expression=expr, id_head170_expr=id_expr, expr_verts=expr_verts,
                     expr_stats=expr_stats, template_offset=D, detail_stats=detail_stats)


# --------------------------------------------------------------------------- detail offset
def _laplacian(G) -> sp.csr_matrix:
    """Uniform graph Laplacian I - Deg^-1 A over the whole mesh."""
    A = G.vertex_adjacency.astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    return (sp.eye(G.n_verts) - sp.diags(1.0 / np.maximum(deg, 1)) @ A).tocsr()


def _lip_regions(G) -> dict:
    """Lip vertex sets: lips (upper | lower lip groups), zone (lips + 1 ring) and seam
    (lip vertices adjacent to the mouth sock)."""
    ul, ll = G.group("upper_lip"), G.group("lower_lip")
    lips = ul | ll
    ms = G.group("mouth_sock")
    seam = G.dilate(ms, 1) & ~ms & lips
    return dict(upper=ul, lower=ll, lips=lips, seam=seam, zone=G.dilate(lips, 1), mouth_sock=ms)


def _seam_pairs(G, reg: dict) -> np.ndarray:
    """(P,2) upper/lower seam-row vertex pairs: each upper seam vertex with the nearest
    lower one on the (closed-mouth) template."""
    u = np.nonzero(reg["seam"] & reg["upper"])[0]
    lo = np.nonzero(reg["seam"] & reg["lower"])[0]
    if not len(u) or not len(lo):
        return np.zeros((0, 2), int)
    _, j = cKDTree(G.template[lo]).query(G.template[u])
    return np.stack([u, lo[j]], 1)


def _inside_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of (n,2) points against a closed (m,2) polygon."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    a, b = poly, np.roll(poly, -1, axis=0)
    cross = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return ((cross & (x < xi)).sum(1) % 2) == 1


def _lip_terms(G, lips: LipFit, tgt: "_TargetSampler", R, t, lip_landmarks, gnm_lip_landmarks) -> dict:
    """Everything the detail solve needs for the lip terms, target data in the GNM frame
    (``target @ R.T + t``)."""
    from .landmarks import OUTER_LIP_IDX
    reg = _lip_regions(G)
    zone = reg["zone"]
    vweight = np.ones(G.n_verts)
    vweight[zone] = lips.data_weight
    aw = np.ones(G.n_verts)
    aw[zone] = lips.smooth
    aw[G.dilate(zone, 2) & ~zone] = np.sqrt(lips.smooth)       # blend back over 2 rings

    lm = lip_landmarks or {}
    glm = gnm_lip_landmarks or {}
    keys = [k for k in lm if k in glm and np.isfinite(np.asarray(lm[k], float)).all()]
    c_tri = G.triangles[np.array([int(glm[k][0]) for k in keys], int)].reshape(-1, 3)
    c_bary = np.array([np.asarray(glm[k][1], float) for k in keys]).reshape(-1, 3)
    c_bary /= np.maximum(c_bary.sum(1, keepdims=True), 1e-12)
    c_raw = np.array([np.asarray(lm[k], float) for k in keys]).reshape(-1, 3)

    # target lip samples: inside the outer contour seen from the front, not far behind
    # the contour and not facing away (the target frame is the aligned, frontal one)
    rev_pts = rev_nrm = np.zeros((0, 3))
    outer = [k for k in OUTER_LIP_IDX if k in keys]
    if len(outer) >= 8:
        poly = np.array([lm[k] for k in outer], float)
        zmin = c_raw[:, 2].min() - 0.004
        p, n = tgt.points, tgt.normals
        ok = (p[:, 2] > zmin) & (n[:, 2] > -0.2)
        ok[ok] = _inside_polygon(p[ok, :2], poly[:, :2])
        rev_pts, rev_nrm = p[ok] @ R.T + t, n[ok] @ R.T
    return dict(lips=reg["lips"], vweight=vweight, alpha_w=aw, contour_tri=c_tri, contour_bary=c_bary,
                contour_pts=c_raw @ R.T + t, rev_pts=rev_pts, rev_nrm=rev_nrm,
                rev_set=np.nonzero(zone & ~reg["mouth_sock"])[0], seam_pairs=_seam_pairs(G, reg), cfg=lips)


def _solve_detail(G, Vb, pair_fn, alpha, beta, iters, w_fwd, w_p2p, w_rev, lip: dict | None = None,
                  D0: np.ndarray | None = None, membrane: tuple[np.ndarray, np.ndarray] | None = None,
                  terms: dict | None = None) -> np.ndarray:
    """Per-vertex offset D (V,3), metres, minimising (mm units, per-term normalised)
    data(Vb + D) + alpha*|L D|^2 + beta*|D|^2, re-paired each iteration.
    Data = forward point-to-plane (+ small point-to-point) and reverse point-to-plane.
    ``lip`` (from ``_lip_terms``): per-vertex data weights and Laplacian weights
    (alpha * |W^1/2 L D|^2), point-to-point lip contours, target-lip -> GNM-lip
    point-to-plane and equal offsets of paired upper/lower seam vertices.
    D0: start value; membrane (edges (E,2), weights (E,)): sum w |D_i - D_j|^2 / l0^2 (edge strain, mm units).
    terms: ``alpha_w`` (V,) Laplacian weights (replace the lip ones) and ``rows``, functions
    (V, N) -> [(per-row weight, vid, normal, target point)] of extra point-to-plane rows
    (features' two-sided term, ears), re-evaluated every iteration."""
    nV = G.n_verts
    L = _laplacian(G)
    terms = terms or {}
    aw = terms.get("alpha_w", None if lip is None else lip["alpha_w"])
    LtL = L.T @ L if aw is None else L.T @ sp.diags(aw) @ L
    reg = sp.kron(LtL.tocsr(), sp.eye(3)) * (alpha / nV * 1e3) + sp.eye(3 * nV) * beta
    if membrane is not None and np.any(membrane[1] > 0):
        e, w = membrane[0][membrane[1] > 0], membrane[1][membrane[1] > 0]
        l0 = np.linalg.norm(Vb[e[:, 0]] - Vb[e[:, 1]], axis=1) * MM
        A = sp.csr_matrix((np.tile([1.0, -1.0], len(e)), (np.repeat(np.arange(len(e)), 2), e.ravel())),
                          shape=(len(e), nV))
        reg = reg + sp.kron((A.T @ sp.diags(w / np.maximum(l0, 1e-3) ** 2) @ A).tocsr(), sp.eye(3))
    tris = G.triangles
    D = np.zeros((nV, 3)) if D0 is None else D0.copy()
    sw = np.ones(nV) if lip is None else np.sqrt(lip["vweight"])
    H_lip = sp.csr_matrix((3 * nV, 3 * nV))
    g_lip = np.zeros(3 * nV)
    if lip is not None:
        cfg = lip["cfg"]
        cols3 = lambda vid: (3 * np.asarray(vid)[..., None] + np.arange(3))  # noqa: E731
        pr = lip["seam_pairs"]
        if len(pr) and cfg.w_seam > 0:
            P = len(pr)
            Ms = sp.csr_matrix((np.tile([1.0, -1.0], 3 * P),
                                (np.repeat(np.arange(3 * P), 2),
                                 np.stack([cols3(pr[:, 0]), cols3(pr[:, 1])], -1).reshape(-1))),
                               shape=(3 * P, 3 * nV))
            H_lip = H_lip + (cfg.w_seam / P) * (Ms.T @ Ms)
        c_tri, c_bary = lip["contour_tri"], lip["contour_bary"]
        if len(c_tri) and cfg.w_contour > 0:
            n_c = len(c_tri)
            # row (l, d) = sum_k bary[l,k] * D[tri[l,k], d]
            Mc = sp.csr_matrix((np.repeat(c_bary[:, None, :], 3, axis=1).reshape(-1),
                                (np.repeat(np.arange(3 * n_c), 3),
                                 cols3(c_tri).transpose(0, 2, 1).reshape(-1))),
                               shape=(3 * n_c, 3 * nV))
            bc = (lip["contour_pts"] - np.einsum("lk,lkd->ld", c_bary, Vb[c_tri])).ravel() * MM
            H_lip = H_lip + (cfg.w_contour / n_c) * (Mc.T @ Mc)
            g_lip = g_lip + (cfg.w_contour / n_c) * (Mc.T @ bc)
        rev_tree_set = lip["rev_set"]

    def plane_rows(vid, nrm, tp, w=None):
        m = len(vid)
        s_ = np.ones(m) if w is None else w
        M = sp.csr_matrix(((nrm * s_[:, None]).ravel(),
                           (np.repeat(np.arange(m), 3), (3 * vid[:, None] + np.arange(3)).ravel())),
                          shape=(m, 3 * nV))
        return M, np.einsum("ij,ij->i", nrm, tp - Vb[vid]) * MM * s_

    for _ in range(iters):
        V = Vb + D
        N = vertex_normals(V, tris)
        fv, q, nq, rv, rp = pair_fn(V, N)
        Mf, bf = plane_rows(fv, nq, q, sw[fv])
        m = len(fv)
        s3 = np.repeat(sw[fv], 3)
        Mp = sp.csr_matrix((s3, (np.arange(3 * m), (3 * fv[:, None] + np.arange(3)).ravel())),
                           shape=(3 * m, 3 * nV))
        bp = (q - Vb[fv]).ravel() * MM * s3
        H = (w_fwd / m) * (Mf.T @ Mf) + (w_p2p / m) * (Mp.T @ Mp) + reg + H_lip
        g = (w_fwd / m) * (Mf.T @ bf) + (w_p2p / m) * (Mp.T @ bp) + g_lip
        if len(rv):
            Mr, br = plane_rows(rv, N[rv], rp, sw[rv])
            H = H + (w_rev / len(rv)) * (Mr.T @ Mr)
            g = g + (w_rev / len(rv)) * (Mr.T @ br)
        if lip is not None and cfg.w_reverse > 0 and len(lip["rev_pts"]):
            # two-sided lips: every target lip sample pulls its nearest GNM lip vertex
            p, pn = lip["rev_pts"], lip["rev_nrm"]
            d, j = cKDTree(V[rev_tree_set]).query(p, workers=-1)
            vid = rev_tree_set[j]
            ok = (np.einsum("ij,ij->i", pn, N[vid]) > 0.3) & (d < 0.006)
            if ok.any():
                Ml, bl = plane_rows(vid[ok], pn[ok], p[ok])
                H = H + (cfg.w_reverse / ok.sum()) * (Ml.T @ Ml)
                g = g + (cfg.w_reverse / ok.sum()) * (Ml.T @ bl)
        for fn in terms.get("rows", ()):
            for w_, vid, nrm, tp in fn(V, N):
                if len(vid):
                    Mx, bx = plane_rows(vid, nrm, tp)
                    H = H + w_ * (Mx.T @ Mx)
                    g = g + w_ * (Mx.T @ bx)
        D = spl.spsolve(H.tocsc(), g).reshape(nV, 3) / MM
    return D


def _nearest_vertex_rows(V, N, pts, nrm, vset, w, max_dist, min_cos=0.3):
    """Two-sided rows: every target sample (pts, nrm) pulls its nearest GNM vertex of
    ``vset`` onto its tangent plane; pairs further than ``max_dist`` or with disagreeing
    normals are dropped.  Returns (per-row weight w / n, vid, nrm, pts)."""
    d, j = cKDTree(V[vset]).query(pts, workers=-1)
    vid = vset[j]
    ok = (np.einsum("ij,ij->i", nrm, N[vid]) > min_cos) & (d < max_dist)
    return w / max(int(ok.sum()), 1), vid[ok], nrm[ok], pts[ok]


def _curvature_alpha(G, target_mesh, Vt: np.ndarray, mask: np.ndarray, cfg: FeatureDetail) -> np.ndarray:
    """(V,) Laplacian weight multipliers 1 / (1 + (H / h0)^2) on ``mask`` (1 elsewhere),
    H = mean curvature (1/m) of the (welded) target at the point closest to each vertex
    (``Vt``: GNM vertices in the target frame), then eroded softly over 2 rings so
    low-weight zones keep their extent."""
    import trimesh
    tw = trimesh.Trimesh(np.asarray(target_mesh.vertices), np.asarray(target_mesh.faces), process=False)
    tw.merge_vertices(merge_tex=True, merge_norm=True)
    tw = trimesh.Trimesh(tw.vertices, tw.faces, process=True)
    idx = np.nonzero(mask)[0]
    cp, _, _ = trimesh.proximity.closest_point(tw, Vt[idx])
    H = trimesh.curvature.discrete_mean_curvature_measure(tw, cp, cfg.radius) / (np.pi * cfg.radius ** 2)
    aw = np.ones(G.n_verts)
    aw[idx] = np.clip(1.0 / (1.0 + (np.abs(H) / cfg.h0) ** 2), cfg.floor, 1.0)
    A = G.vertex_adjacency.astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(2):
        aw = np.minimum(aw, (A @ aw + aw) / (deg + 1))
    return aw


def _follow_rigid_parts(G, D: np.ndarray, Vb: np.ndarray, eye_ring_radius: float = 0.022) -> np.ndarray:
    """Eyeballs translate with the mean D of the skin within ``eye_ring_radius`` of their
    centre (lids/socket); teeth, gums and tongue with the mean D of the mouth sock."""
    D = D.copy()
    eyes = G.group("eyes")
    skin = G.group("skin")
    for side in ("left_eye", "right_eye"):
        eb = G.group(side) & eyes
        c = Vb[eb].mean(0)
        ring = skin & ~eyes & (np.linalg.norm(Vb - c, axis=1) < eye_ring_radius)
        D[eb] = D[ring].mean(0)
    mouth = G.group("teeth") | G.group("gums") | G.group("tongue")
    D[mouth] = D[G.group("mouth_sock")].mean(0)
    return D


def _flipped(tris: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Indices of triangles whose normal in B points against the one in A."""
    na = np.cross(A[tris[:, 1]] - A[tris[:, 0]], A[tris[:, 2]] - A[tris[:, 0]])
    nb = np.cross(B[tris[:, 1]] - B[tris[:, 0]], B[tris[:, 2]] - B[tris[:, 0]])
    return np.nonzero(np.einsum("ij,ij->i", na, nb) < 0)[0]


def _flips(G, Vb: np.ndarray, D: np.ndarray, ref: Sequence[np.ndarray] = (), posed=None) -> np.ndarray:
    """Triangles D flips in the bind pose or in any posed mesh (vs ``ref`` = posed(None))."""
    fl = _flipped(G.triangles, Vb, Vb + D)
    if posed is not None:
        for a, b in zip(ref, posed(D)):
            fl = np.union1d(fl, _flipped(G.triangles, a, b))
    return fl


def _unflip(G, Vb: np.ndarray, D: np.ndarray, max_rounds: int = 4,
            posed: Callable[[np.ndarray | None], Sequence[np.ndarray]] | None = None) -> tuple[np.ndarray, int]:
    """Guard against flipped triangles (typically slivers at the lip corners): give the
    vertices of each flipped triangle and their 1-ring (growing each round) the mean
    offset of that patch, i.e. translate the patch rigidly so it cannot flip.
    ``posed(D)`` (optional) returns posed meshes; a triangle that flips in any of them
    relative to ``posed(None)`` is treated the same way.
    Returns (D, number of vertices changed)."""
    D = D.copy()
    changed = np.zeros(G.n_verts, bool)
    ref = posed(None) if posed is not None else []
    for k in range(max_rounds):
        flipped = _flips(G, Vb, D, ref, posed)
        if not len(flipped):
            break
        seed = np.zeros(G.n_verts, bool)
        seed[G.triangles[flipped].ravel()] = True
        patch = G.dilate(seed, k + 1)
        # one shared offset per connected patch
        A = G.vertex_adjacency[patch][:, patch]
        n_cc, lab = csg.connected_components(A, directed=False)
        pidx = np.nonzero(patch)[0]
        for c in range(n_cc):
            v = pidx[lab == c]
            D[v] = D[v].mean(0)
        changed |= patch
    return D, int(changed.sum())


def _offset_quality(G, Vb: np.ndarray, D: np.ndarray) -> tuple[dict, np.ndarray]:
    """Magnitude of D, edge strain |l1/l0 - 1| and flipped triangles of Vb + D vs Vb.
    Returns (stats, indices of flipped triangles)."""
    tris = G.triangles
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    V1 = Vb + D
    l0 = np.linalg.norm(Vb[e[:, 0]] - Vb[e[:, 1]], axis=1)
    l1 = np.linalg.norm(V1[e[:, 0]] - V1[e[:, 1]], axis=1)
    s = np.abs(l1 / np.maximum(l0, 1e-12) - 1)
    face = G.face_mask()
    dn = np.linalg.norm(D, axis=1) * MM
    flipped = _flipped(tris, Vb, V1)
    return dict(max_mm=float(dn.max()), face_rms_mm=float(np.sqrt((dn[face] ** 2).mean())),
                edge_strain_p99_pct=float(np.quantile(s, 0.99) * 100),
                edge_strain_max_pct=float(s.max() * 100),
                flipped_triangles=int(len(flipped))), flipped


def _rigid_step(V, N, face_idx, tgt, R, t, cos_thr, dist_fac):
    """One linearised point-to-plane rigid update of the target (no scale)."""
    x = V[face_idx]
    nx = N[face_idx]
    d, j = tgt.tree.query((x - t) @ R, workers=-1)
    p = tgt.points[j] @ R.T + t
    pn = tgt.normals[j] @ R.T
    ok = np.einsum("ij,ij->i", pn, nx) > cos_thr
    if ok.sum() < 20:
        return R, t
    ok &= d < dist_fac * np.median(d[ok])
    x, nx, p = x[ok], nx[ok], p[ok]
    c = p.mean(0)
    pc = p - c
    # moving target point p -> c + (I + [w]x)(p - c) + dt: residual n.(p + w x pc + dt - x)
    A = np.hstack([np.cross(pc, nx), nx])
    b = np.einsum("ij,ij->i", nx, x - p)
    H = A.T @ A + 1e-9 * np.eye(6)
    sol = np.linalg.solve(H, A.T @ b)
    dR = _rotvec_to_R(sol[:3])
    R_new = dR @ R
    t_new = dR @ (t - c) + c + sol[3:]
    return R_new, t_new


_BT_CACHE: dict = {}


def _basis_T(G) -> np.ndarray:
    key = id(G)
    if key not in _BT_CACHE:
        _BT_CACHE[key] = np.ascontiguousarray(G.head_basis.transpose(1, 2, 0)).astype(np.float32)
    return _BT_CACHE[key]


def gnm_landmarks_from_vertex_weights(vid: np.ndarray, w: np.ndarray) -> dict:
    """Convert (L,3) vertex triples + weights to {i: (tri_idx, bary)} on GNM triangles."""
    G = gnm_data.load()
    tris = G.triangles
    out = {}
    key = {tuple(sorted(tr)): i for i, tr in enumerate(tris.tolist())}
    for i, (vv, ww) in enumerate(zip(vid, w)):
        ti = key.get(tuple(sorted(vv.tolist())))
        if ti is None:
            continue
        order = [list(vv).index(v) for v in tris[ti]]
        out[i] = (ti, np.asarray(ww)[order] / np.sum(ww))
    return out
