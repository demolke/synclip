"""Fit the GNM head (identity, expression, detail offset) to an arbitrary head mesh.

    python fit_head.py MESH [-o OUT_DIR] [--metric] [--no-renders]
                            [--landmark-weight W] [--prior L]
                            [--no-fit-expression] [--expr-prior L]
                            [--no-detail] [--detail-smooth A]
                            [--no-lips] [--lip-weight W] [--guard-all-targets]
                            [--feature-detail {off,moderate}] [--no-ears]

Pipeline: load geometry (trimesh, force='mesh'; morph targets are ignored) ->
headfit.align (orientation + similarity to the GNM frame, MediaPipe landmarks)
-> headfit.fit (170 head identity PCs + rigid refinement; joint identity +
expression fit; identity refit with expression = 0; per-vertex template
offset) -> headfit.evaluate (metrics, renders, export of .glb/.json).
The .glb comes from the GNM viewer's exporter (gnm_model): neutral base
(expression = 0) with the offset baked in, plus its morph targets generated
from that fitted base.
It also writes <stem>_viewer.json (+ <stem>_offset.npy) for gnm_head_viewer.py.

--metric: the input is already in metres. Only the rigid part of the alignment
is applied (the Umeyama scale is undone about the landmark centroid), so the
target keeps its true size and GNM's identity PCs (mainly PC0) absorb head
size. Without it the target is scaled to the GNM mean size.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# Same layout gnm_model.py expects (it does sys.path.append('gnm') relative to cwd);
# make it work from any cwd.
for p in (HERE, HERE / "gnm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

LOW_CONFIDENCE = 0.5


def _die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _check_gnm():
    from headfit import gnm_data
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            import gnm_model
        ok = bool(getattr(gnm_model, "GNM_OK", False))
    except Exception as e:  # pragma: no cover - import failure
        ok = False
        buf.write(f"\n{type(e).__name__}: {e}")
    if not ok:
        tail = "\n".join(buf.getvalue().strip().splitlines()[-5:])
        _die("GNM model not available (gnm_model.GNM_OK is False). Initialise the submodule:\n"
             "  git submodule update --init synclip/gnm\n" + tail)
    for f in (gnm_data.NPZ_PATH,):
        if not pathlib.Path(f).exists():
            _die(f"GNM data file missing: {f}")


def _apply(M, p):
    p = np.asarray(p, float)
    return p @ M[:3, :3].T + M[:3, 3]


def _metric_rows(m: dict, label: str) -> list:
    rows = [
        (f"[{label}] GNM->target mean / median / p95 (mm)",
         f"{m['gnm2tgt_mean_mm']:.2f} / {m['gnm2tgt_median_mm']:.2f} / {m['gnm2tgt_p95_mm']:.2f}"),
        (f"[{label}] GNM->target within 2 / 5 mm (%)",
         f"{m['gnm2tgt_within_2mm_pct']:.1f} / {m['gnm2tgt_within_5mm_pct']:.1f}"),
    ]
    if m.get("tgt2gnm_n"):
        rows.append((f"[{label}] target->GNM mean / median / p95 (mm)",
                     f"{m['tgt2gnm_mean_mm']:.2f} / {m['tgt2gnm_median_mm']:.2f} / {m['tgt2gnm_p95_mm']:.2f}"))
    if "landmark_rmse_mm" in m:
        rows.append((f"[{label}] landmark RMSE (mm)", f"{m['landmark_rmse_mm']:.2f}"))
    return rows


def write_viewer_params(out_dir: pathlib.Path, stem: str, fr) -> dict:
    """Write ``<stem>_viewer.json`` in gnm_head_viewer's Save/Load format (``gnm_params``:
    identity 253, expression 383, rotations 4x3, translation 3) with the fitted identity,
    expression 0 and neutral pose. When the fit has a template offset D it goes to
    ``<stem>_offset.npy`` (float32 (V,3)) and the JSON references it by a path relative
    to the JSON (``template_offset``); viewers without offset support ignore the key."""
    out_dir = pathlib.Path(out_dir)
    data = {
        "format": "gnm_params",
        "identity": [float(x) for x in np.asarray(fr.id_full253, np.float32)],
        "expression": [0.0] * len(fr.expression),
        "rotations": [[0.0, 0.0, 0.0] for _ in range(4)],
        "translation": [0.0, 0.0, 0.0],
    }
    paths = {}
    if fr.template_offset is not None:
        npy = out_dir / f"{stem}_offset.npy"
        np.save(npy, np.asarray(fr.template_offset, np.float32))
        data["template_offset"] = npy.name
        paths["offset"] = str(npy)
    js = out_dir / f"{stem}_viewer.json"
    js.write_text(json.dumps(data, indent=2))
    paths["viewer_json"] = str(js)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit GNM head identity (+ expression, detail offset) to a head mesh.")
    ap.add_argument("mesh", type=pathlib.Path)
    ap.add_argument("-o", "--out-dir", type=pathlib.Path, default=None,
                    help="output directory (default: fit_out/<mesh stem>/)")
    ap.add_argument("--metric", action="store_true",
                    help="input is in metres: keep its true size (rigid alignment only)")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--landmark-weight", type=float, default=1.0, metavar="W",
                    help="initial landmark term weight in the fit (default 1.0, 0 disables)")
    ap.add_argument("--prior", type=float, default=1.0, metavar="L",
                    help="multiplier on the identity prior (Tikhonov) weight of every stage (default 1.0)")
    ap.add_argument("--no-fit-expression", dest="fit_expression", action="store_false",
                    help="do not fit expression jointly with identity (identity-only fit)")
    ap.add_argument("--expr-prior", type=float, default=0.1, metavar="L",
                    help="expression prior weight on the mean squared face displacement in mm^2 (default 0.1)")
    ap.add_argument("--no-detail", dest="detail", action="store_false",
                    help="do not solve the per-vertex template offset")
    ap.add_argument("--detail-smooth", type=float, default=1.0, metavar="A",
                    help="Laplacian smoothness weight of the template offset (default 1.0; larger = smoother)")
    ap.add_argument("--no-lips", dest="lips", action="store_false",
                    help="no lip-specific terms in the template offset (lips fitted like the rest of the face)")
    ap.add_argument("--lip-weight", type=float, default=1.0, metavar="W",
                    help="multiplier on the lip terms of the template offset (default 1.0)")
    ap.add_argument("--guard-all-targets", action="store_true",
                    help="flip guard of the template offset checks every exported morph target, "
                         "not only jawOpen / smile / jawOpen+smile (slower, slightly smoother lips)")
    ap.add_argument("--feature-detail", choices=("off", "moderate"), default="moderate",
                    help="feature-aware template offset: curvature-adaptive smoothness and a face-wide "
                         "two-sided term (default moderate)")
    ap.add_argument("--no-ears", dest="ears", action="store_false",
                    help="do not fit the ears (they follow the head through identity and smoothness only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not a.mesh.exists():
        _die(f"mesh not found: {a.mesh}")
    _check_gnm()

    from headfit import align as A, evaluate as E, fit as F

    out = a.out_dir or pathlib.Path("fit_out") / a.mesh.stem
    t0 = time.time()

    # ---- load (geometry only) -------------------------------------------
    target = trimesh.load(str(a.mesh), force="mesh")
    if not isinstance(target, trimesh.Trimesh) or len(target.faces) == 0:
        _die(f"could not load a triangle mesh from {a.mesh}")
    target = trimesh.Trimesh(np.asarray(target.vertices, float), np.asarray(target.faces), process=False)
    print(f"loaded {a.mesh}: {len(target.vertices)} verts, {len(target.faces)} faces, "
          f"extent {np.round(target.extents, 4).tolist()}")

    # ---- align ----------------------------------------------------------
    al = A.align(target)
    if a.verbose:
        print("\n".join("  " + s for s in al.log))
    T = al.transform.copy()
    lm = {k: np.asarray(v, float) for k, v in al.target_landmarks.items()}
    lip_lm = {k: np.asarray(v, float) for k, v in al.lip_landmarks.items()}
    if a.metric:
        # undo the similarity scale about the landmark centroid: x' = c + (x - c)/s
        c = np.mean(list(lm.values()), axis=0)
        S = np.eye(4)
        S[:3, :3] /= al.scale
        S[:3, 3] = c - c / al.scale
        T = S @ T
        lm = {k: _apply(S, v) for k, v in lm.items()}
        lip_lm = {k: _apply(S, v) for k, v in lip_lm.items()}
    tgt = target.copy()
    tgt.apply_transform(T)
    applied_scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
    print(f"align: umeyama scale {al.scale:.6g}, applied scale {applied_scale:.6g}"
          f"{' (metric)' if a.metric else ''}, confidence {al.confidence:.3f}, "
          f"fallback {al.used_fallback}, {len(lm)} landmarks")
    if al.used_fallback:
        print("warning: no reliable face detection; used the ICP fallback alignment "
              "(landmarks are projected from the GNM mean, not detected). Check the renders.", file=sys.stderr)
    elif al.confidence < LOW_CONFIDENCE:
        print(f"warning: low landmark confidence {al.confidence:.2f} (< {LOW_CONFIDENCE}); "
              "check the renders.", file=sys.stderr)

    # ---- fit ------------------------------------------------------------
    stages = [F.Stage(s.n_pcs, s.lam * a.prior, s.iters) for s in F.DEFAULT_STAGES]
    use_lm = a.landmark_weight > 0 and not al.used_fallback
    lips = F.LipFit().scaled(a.lip_weight) if a.lips else None
    fr = F.fit(tgt, lm if use_lm else None, al.gnm_landmarks if use_lm else None,
               stages=stages, w_lmk=a.landmark_weight, fit_expression=a.fit_expression,
               expr_prior=a.expr_prior, detail=a.detail, detail_smooth=a.detail_smooth,
               lip_landmarks=lip_lm, gnm_lip_landmarks=al.gnm_lip_landmarks, lips=lips,
               features=F.FEATURE_DETAIL[a.feature_detail], ears=F.EarFit() if a.ears else None,
               pose_fn=E.gnm_pose_fn(E.all_target_poses() if a.guard_all_targets else E.CHECK_POSES),
               verbose=a.verbose)

    # ---- evaluate (target AND landmarks get the fit's rigid refinement) --
    tgt_fit = tgt.copy()
    tgt_fit.apply_transform(fr.rigid_4x4)
    # fallback landmarks are projections of the GNM mean, not detections: no RMSE
    lm_fit = {} if al.used_fallback else {k: fr.transform_target(v) for k, v in lm.items()}
    # neutral export (expr = 0, D baked in) vs target; and the joint identity+expression fit
    m = E.metrics(fr.verts, tgt_fit, lm_fit, al.gnm_landmarks)
    m_expr = (E.metrics(fr.expr_verts, tgt_fit, lm_fit, al.gnm_landmarks)
              if fr.expr_verts is not None else None)

    total = fr.rigid_4x4 @ T  # original input -> fitted GNM frame
    name = f"{a.mesh.stem}_gnmfit"
    meta = {
        "input": str(a.mesh),
        "metric_mode": bool(a.metric),
        "id_head170": [float(x) for x in fr.id_head170],
        "alignment": {
            "transform_4x4": al.transform.tolist(),
            "umeyama_scale": al.scale,
            "applied_transform_4x4": T.tolist(),
            "applied_scale": applied_scale,
            "confidence": al.confidence,
            "used_fallback": bool(al.used_fallback),
            "n_landmarks": len(lm),
        },
        "input_to_gnm_4x4": total.tolist(),
        "fit_options": {"landmark_weight": a.landmark_weight, "prior": a.prior,
                        "landmarks_used": bool(use_lm), "fit_expression": bool(a.fit_expression),
                        "expr_prior": a.expr_prior, "detail": bool(a.detail),
                        "detail_smooth": a.detail_smooth, "lips": bool(a.lips), "lip_weight": a.lip_weight,
                        "n_lip_landmarks": len(lip_lm), "guard_all_targets": bool(a.guard_all_targets),
                        "feature_detail": a.feature_detail, "ears": bool(a.ears)},
        "metrics": m,
        "metrics_note": "metrics = neutral export (expression 0, template offset) vs target; "
                        "metrics_with_expression = joint identity+expression fit (no offset) vs target",
        "metrics_with_expression": m_expr,
    }
    if fr.template_offset is not None:
        # D is stored as float32 for the viewer; export with the same rounded D so the
        # viewer reproduces this .glb exactly.
        fr = dataclasses.replace(fr, template_offset=np.asarray(fr.template_offset, np.float32))
    paths = E.export(fr, out, name, meta)
    paths.update(write_viewer_params(out, a.mesh.stem, fr))
    if not a.no_renders:
        paths.update(E.render_views(tgt_fit, fr.verts, out, name))
        if fr.expr_verts is not None:
            paths["compare_expr"] = E.render_views(tgt_fit, fr.expr_verts, out, name + "_expr")["compare"]

    # ---- report ---------------------------------------------------------
    rows = _metric_rows(m, "neutral export")
    if m_expr is not None:
        rows += _metric_rows(m_expr, "with-expression fit")
        es = fr.expr_stats
        rows.append(("expression face disp rms / max (mm)",
                     f"{es['face_rms_disp_mm']:.2f} / {es['face_max_disp_mm']:.2f}  (|e| {es['coef_norm']:.2f})"))
    if fr.template_offset is not None:
        ds = fr.detail_stats
        rows.append(("offset max / face rms (mm), alpha",
                     f"{ds['max_mm']:.2f} / {ds['face_rms_mm']:.3f}, {ds['alpha']:g}"))
        rows.append(("offset edge strain p99 / max (%), flips",
                     f"{ds['edge_strain_p99_pct']:.2f} / {ds['edge_strain_max_pct']:.1f}, {ds['flipped_triangles']}"
                     + (f" (posed {'/'.join(map(str, ds['flipped_triangles_posed']))})"
                        if "flipped_triangles_posed" in ds else "")))
    rows += [("id PC0 / |id|max", f"{fr.id_head170[0]:+.2f} / {np.abs(fr.id_head170).max():.2f}"),
             ("runtime (s)", f"{time.time() - t0:.1f}")]
    w = max(len(r[0]) for r in rows)
    print("-" * (w + 24))
    for k, v in rows:
        print(f"{k:<{w}}  {v}")
    print("-" * (w + 24))
    print("outputs:")
    for k in ("glb", "json", "viewer_json", "offset", "compare", "compare_expr"):
        if k in paths:
            print(f"  {k:12s} {paths[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
