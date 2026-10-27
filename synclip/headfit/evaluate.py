"""Evaluation, renders and export for GNM head fits.

Metrics are computed on the GNM face mask, in millimetres.  Renders use a
small private numpy z-buffer rasteriser (orthographic, Lambert shading) so no
GL / embree is needed; images are written with Pillow.
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import time
from typing import Mapping, Sequence

import numpy as np
import trimesh

from . import gnm_data

MM = 1000.0


# --------------------------------------------------------------------------- metrics
def _stats(d_mm: np.ndarray, prefix: str) -> dict:
    if len(d_mm) == 0:
        return {f"{prefix}_n": 0}
    return {
        f"{prefix}_n": int(len(d_mm)),
        f"{prefix}_mean_mm": float(d_mm.mean()),
        f"{prefix}_median_mm": float(np.median(d_mm)),
        f"{prefix}_p95_mm": float(np.quantile(d_mm, 0.95)),
        f"{prefix}_within_2mm_pct": float((d_mm <= 2.0).mean() * 100),
        f"{prefix}_within_5mm_pct": float((d_mm <= 5.0).mean() * 100),
    }


def gnm_to_target_distances(verts: np.ndarray, target_mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex (all V) exact distance (mm) from GNM vertices to the target surface;
    only face-mask entries are computed, others are NaN."""
    G = gnm_data.load()
    face = G.face_mask()
    out = np.full(len(verts), np.nan)
    _, d, _ = trimesh.proximity.closest_point(target_mesh, verts[face])
    out[face] = d * MM
    return out


def target_to_gnm_distances(verts: np.ndarray, target_mesh: trimesh.Trimesh,
                            n_samples: int = 30_000, seed: int = 0) -> np.ndarray:
    """Distances (mm) from target surface samples to the fitted GNM surface, kept only
    where the closest GNM point lies on an interior face-mask triangle."""
    G = gnm_data.load()
    face = G.face_mask()
    interior = face & ~G.dilate(~face, 2)
    tri_ok = interior[G.triangles].all(1)
    gm = trimesh.Trimesh(verts, G.triangles, process=False)
    # restrict to target samples inside the (padded) face bounding box
    fb = verts[face]
    lo, hi = fb.min(0) - 0.01, fb.max(0) + 0.01
    try:  # trimesh >= 4 ignores np.random.seed: pass seed explicitly
        pts, _ = trimesh.sample.sample_surface_even(target_mesh, n_samples, seed=seed)
    except TypeError:  # old trimesh: global numpy RNG
        rs = np.random.get_state()
        np.random.seed(seed)
        try:
            pts, _ = trimesh.sample.sample_surface_even(target_mesh, n_samples)
        finally:
            np.random.set_state(rs)
    pts = pts[np.all((pts > lo) & (pts < hi), axis=1)]
    _, d, tri = trimesh.proximity.closest_point(gm, pts)
    return d[tri_ok[tri]] * MM


def landmark_rmse(verts, target_landmarks, gnm_landmarks) -> float | None:
    if not target_landmarks or not gnm_landmarks:
        return None
    G = gnm_data.load()
    keys = [k for k in gnm_landmarks if k in target_landmarks]
    if not keys:
        return None
    tri = G.triangles[[int(gnm_landmarks[k][0]) for k in keys]]
    bary = np.array([np.asarray(gnm_landmarks[k][1], float) for k in keys])
    bary /= bary.sum(1, keepdims=True)
    p = np.einsum("lk,lkd->ld", bary, verts[tri])
    q = np.array([np.asarray(target_landmarks[k], float) for k in keys])
    return float(np.sqrt(((p - q) ** 2).sum(1).mean()) * MM)


def metrics(verts: np.ndarray, target_mesh: trimesh.Trimesh,
            target_landmarks: Mapping | None = None,
            gnm_landmarks: Mapping | None = None) -> dict:
    """Face-mask metrics (mm). ``target_mesh``/landmarks must be in the frame of ``verts``
    (i.e. already transformed by the fit's rigid refinement)."""
    d_fwd = gnm_to_target_distances(verts, target_mesh)
    d_fwd = d_fwd[np.isfinite(d_fwd)]
    d_rev = target_to_gnm_distances(verts, target_mesh)
    m = {}
    m.update(_stats(d_fwd, "gnm2tgt"))
    m.update(_stats(d_rev, "tgt2gnm"))
    lr = landmark_rmse(verts, target_landmarks, gnm_landmarks)
    if lr is not None:
        m["landmark_rmse_mm"] = lr
    return m


# --------------------------------------------------------------------------- render
def _look_rotation(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """World->camera rotation; camera looks down -Z. yaw rotates the subject."""
    a, b = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def rasterize(verts: np.ndarray, faces: np.ndarray, R: np.ndarray, center: np.ndarray,
              extent: float, size: int = 512, vcolors: np.ndarray | None = None,
              base_color=(0.86, 0.70, 0.60), bg=(38, 38, 42)) -> np.ndarray:
    """Orthographic z-buffer rasteriser. Returns (size,size,3) uint8.

    vcolors: optional (V,3) floats in [0,1] (NaN rows -> base_color).
    """
    v = (np.asarray(verts, float) - center) @ R.T
    px = size / extent
    xs = (v[:, 0] * px) + size / 2.0
    ys = size / 2.0 - v[:, 1] * px
    zs = v[:, 2]
    faces = np.asarray(faces)
    vn = _vnormals(v, faces)
    col = np.tile(np.asarray(base_color, float), (len(v), 1))
    if vcolors is not None:
        ok = np.all(np.isfinite(vcolors), axis=1)
        col[ok] = vcolors[ok]
    # cull back faces
    e1 = np.stack([xs[faces[:, 1]] - xs[faces[:, 0]], ys[faces[:, 1]] - ys[faces[:, 0]]], 1)
    e2 = np.stack([xs[faces[:, 2]] - xs[faces[:, 0]], ys[faces[:, 2]] - ys[faces[:, 0]]], 1)
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    faces = faces[np.abs(area) > 1e-9]
    fx, fy = xs[faces], ys[faces]
    x0 = np.clip(np.floor(fx.min(1)).astype(int), 0, size - 1)
    x1 = np.clip(np.ceil(fx.max(1)).astype(int), 0, size - 1)
    y0 = np.clip(np.floor(fy.min(1)).astype(int), 0, size - 1)
    y1 = np.clip(np.ceil(fy.max(1)).astype(int), 0, size - 1)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pix_all, z_all, f_all, b_all = [], [], [], []
    # bucket triangles by bounding-box size so candidates are generated vectorised
    bs = np.maximum(w, h)
    for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)):
        sel = np.nonzero((bs > lo) & (bs <= hi))[0]
        if len(sel) == 0:
            continue
        k = int(bs[sel].max())
        for chunk in np.array_split(sel, max(1, len(sel) * k * k // 2_000_000 + 1)):
            oy, ox = np.mgrid[0:k, 0:k]
            cx = x0[chunk, None] + ox.ravel()[None]
            cy = y0[chunk, None] + oy.ravel()[None]
            valid = (cx <= x1[chunk, None]) & (cy <= y1[chunk, None])
            pxc, pyc = cx + 0.5, cy + 0.5
            ax, ay = fx[chunk, 0, None], fy[chunk, 0, None]
            bx, by = fx[chunk, 1, None], fy[chunk, 1, None]
            qx, qy = fx[chunk, 2, None], fy[chunk, 2, None]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            l0 = ((by - qy) * (pxc - qx) + (qx - bx) * (pyc - qy)) / den
            l1 = ((qy - ay) * (pxc - qx) + (ax - qx) * (pyc - qy)) / den
            l2 = 1 - l0 - l1
            inside = valid & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            ii, jj = np.nonzero(inside)
            fi = chunk[ii]
            bary = np.stack([l0[ii, jj], l1[ii, jj], l2[ii, jj]], 1)
            z = (bary * zs[faces[fi]]).sum(1)
            pix_all.append(cy[ii, jj] * size + cx[ii, jj])
            z_all.append(z)
            f_all.append(fi)
            b_all.append(bary)
    img = np.empty((size * size, 3), np.uint8)
    img[:] = bg
    if not pix_all:
        return img.reshape(size, size, 3)
    pix = np.concatenate(pix_all)
    z = np.concatenate(z_all)
    fi = np.concatenate(f_all)
    bary = np.concatenate(b_all)
    order = np.lexsort((-z, pix))            # nearest (largest z) first per pixel
    pix, fi, bary = pix[order], fi[order], bary[order]
    first = np.ones(len(pix), bool)
    first[1:] = pix[1:] != pix[:-1]
    pix, fi, bary = pix[first], fi[first], bary[first]
    tv = faces[fi]
    n = (bary[:, :, None] * vn[tv]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = (bary[:, :, None] * col[tv]).sum(1)
    light = np.array([0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light)                  # two-sided
    shade = 0.25 + 0.75 * lam
    img[pix] = np.clip(c * shade[:, None] * 255, 0, 255).astype(np.uint8)
    return img.reshape(size, size, 3)


def _vnormals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def error_colormap(err_mm: np.ndarray, vmax: float = 5.0) -> np.ndarray:
    """Blue (0) -> green -> yellow -> red (>= vmax). NaN stays NaN."""
    t = np.clip(np.nan_to_num(np.asarray(err_mm, float) / vmax, nan=0.0), 0, 1)
    stops = np.array([[0.15, 0.30, 0.85], [0.20, 0.80, 0.30], [0.95, 0.90, 0.20], [0.90, 0.15, 0.10]])
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    c = stops[i] * (1 - f) + stops[i + 1] * f
    c[~np.isfinite(err_mm)] = np.nan
    return c


def _face_bounds_view(verts):
    G = gnm_data.load()
    fb = verts[G.face_mask()]
    center = 0.5 * (fb.min(0) + fb.max(0))
    extent = float((fb.max(0) - fb.min(0))[:2].max()) * 1.35
    return center, extent


def render_views(target_mesh: trimesh.Trimesh, verts: np.ndarray, out_dir, prefix: str = "fit",
                 err_mm: np.ndarray | None = None, size: int = 512, vmax: float = 5.0) -> dict:
    """Front and 3/4 renders of target and fit, plus an error heatmap on the fit.

    Returns {name: path}. Also writes a composite ``{prefix}_compare.png``
    (rows: target, fit, error; cols: front, 3/4).
    """
    from PIL import Image, ImageDraw

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = gnm_data.load()
    center, extent = _face_bounds_view(verts)
    if err_mm is None:
        err_mm = gnm_to_target_distances(verts, target_mesh)
    ecol = error_colormap(err_mm, vmax)
    grey = np.full((len(verts), 3), 0.55)
    ecol = np.where(np.isfinite(ecol), ecol, grey)
    views = {"front": _look_rotation(0), "3q": _look_rotation(-35)}
    tv, tf = np.asarray(target_mesh.vertices), np.asarray(target_mesh.faces)
    paths, tiles = {}, []
    for row, (name, V, F, col) in enumerate((("target", tv, tf, None),
                                             ("fit", verts, G.triangles, None),
                                             ("error", verts, G.triangles, ecol))):
        rimgs = []
        for vname, R in views.items():
            img = rasterize(V, F, R, center, extent, size, vcolors=col)
            p = out_dir / f"{prefix}_{name}_{vname}.png"
            Image.fromarray(img).save(p)
            paths[f"{name}_{vname}"] = str(p)
            rimgs.append(img)
        tiles.append(np.concatenate(rimgs, 1))
    comp = Image.fromarray(np.concatenate(tiles, 0))
    d = ImageDraw.Draw(comp)
    for r, lbl in enumerate(("target", "fit", f"GNM->target err (0..{vmax:g} mm)")):
        d.text((8, r * size + 6), lbl, fill=(255, 255, 255))
    p = out_dir / f"{prefix}_compare.png"
    comp.save(p)
    paths["compare"] = str(p)
    return paths


# --------------------------------------------------------------------------- posing
# mouth poses the template-offset flip guard checks (see fit.fit(pose_fn=...)): name -> [(FACS, weight)]
CHECK_POSES: dict = {
    "jawOpen": [("jawOpen", 1.0)],
    "smile": [("mouthSmileLeft", 1.0), ("mouthSmileRight", 1.0)],
    "jawOpen+smile": [("jawOpen", 0.7), ("mouthSmileLeft", 0.7), ("mouthSmileRight", 0.7)],
}


def all_target_poses() -> dict:
    """CHECK_POSES plus every exported morph target (gnm_model.ALL_FACS at weight 1)."""
    with contextlib.redirect_stdout(io.StringIO()):
        import gnm_model as model
    return {**CHECK_POSES, **{n: [(n, 1.0)] for n in model.ALL_FACS}}


def gnm_pose_fn(poses: Mapping = CHECK_POSES):
    """(id_full253, D or None) -> [posed verts per pose], posed through gnm_model exactly as
    the exported morph targets are (D on the template, joints moved by joint_regressor @ D)."""
    with contextlib.redirect_stdout(io.StringIO()):
        import gnm_model as model
    params = []
    for items in poses.values():
        e = np.zeros(model.expression_dim, np.float32)
        r = np.zeros((model.num_joints, 3), np.float32)
        t = np.zeros(3, np.float32)
        for n, w in items:
            de, dr, dt = model.facs_to_full_params(n, w)
            e, r, t = e + de, r + dr, t + dt
        params.append((e, r, t))

    def pose(id_full253, D=None):
        ident = np.asarray(id_full253, np.float32)
        Df = None if D is None else np.asarray(D, np.float32)
        return [np.asarray(model.gen_verts(ident, e, r, t, Df), np.float64) for e, r, t in params]
    return pose


# --------------------------------------------------------------------------- export
def export(fit_result, out_dir, name: str = "gnm_fit", extra_meta: Mapping | None = None) -> dict:
    """Write the fitted head as a Blender-ready .glb through the GNM viewer's exporter
    (``gnm_model.build_gltf_with_quads_and_facs``: quad side channel, UVs, ARKit + FACS +
    semantic morph targets), with the template offset D in effect for the base (expression 0,
    pose 0) and every target, plus the params/metrics (.json)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = gnm_data.load()
    D = fit_result.template_offset
    with contextlib.redirect_stdout(io.StringIO()):
        import gnm_model as model
        ident = np.asarray(fit_result.id_full253, np.float32)
        rots = np.zeros((model.num_joints, 3), np.float32)
        trans = np.zeros(3, np.float32)
        base_v = model.gen_verts(ident, np.zeros(model.expression_dim, np.float32), rots, trans, D)
        facs_full = [model.facs_to_full_params(n, 1.0) for n in model.ALL_FACS]
        glb = out_dir / f"{name}.glb"
        model.build_gltf_with_quads_and_facs(
            base_verts=base_v, base_tris=model.triangles, base_quads=model.quads,
            quad_uvs_arr=model.quad_uvs, tri_uvs_arr=model.tri_uvs,
            facs_names=model.ALL_FACS, facs_full_params=facs_full,
            identity=ident, rotations=rots, translation=trans, out_path=str(glb),
            template_offset=D)
    joints = G.joints(fit_result.id_full253, D)
    # identity the fitted expression was solved with (the neutral ``id`` was refitted after it)
    id_expr = (G.to_full(fit_result.id_head170_expr) if fit_result.id_head170_expr is not None
               else np.asarray(fit_result.id_full253))
    expr = np.asarray(fit_result.expression, float)
    meta = {
        "model": "GNM v3_0 head",
        "frame": "metres, +Y up, +Z forward (GNM bind pose)",
        "identity_names_order": "vertex_identity_basis order (170 head_*, 3 eyes_*, 80 teeth_*)",
        "id": [float(x) for x in fit_result.id_full253],
        "mesh": "neutral: template + template_offset + identity, expression = 0, pose = 0",
        "morph_targets": f"{len(model.ALL_FACS)} (gnm_model.ALL_FACS: ARKit 52 + FACS AUs + semantic), "
                         "deltas from the fitted base with the template offset in effect",
        "expression": [float(x) for x in expr],
        "id_with_expression": [float(x) for x in id_expr],
        "expression_note": ("fitted expression (383, expression_basis order), for reference only: "
                            "the exported mesh and rig are neutral; it pairs with id_with_expression "
                            "and the plain template (no template_offset)" if np.any(expr) else
                            "zeros (expression fitting disabled)"),
        "expression_stats": dict(fit_result.expr_stats),
        "detail_offset": dict(fit_result.detail_stats) if fit_result.template_offset is not None
        else "disabled",
        "joint_positions": joints.tolist(),
        "pose": "zeros",
        "glb": glb.name,
        "target_rigid_refinement_4x4": fit_result.rigid_4x4.tolist(),
        "residuals": fit_result.residuals,
        "runtime_s": fit_result.runtime_s,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps(meta, indent=1))
    return {"glb": str(glb), "json": str(js)}
