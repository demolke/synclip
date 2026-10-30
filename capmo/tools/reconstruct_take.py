"""
Reconstruct a 3D landmark track from a keypoint take (``extract_keypoints``
output), self-calibrating the cameras from the body when no rig is given.

    python -m capmo.tools.reconstruct_take KP_TAKE_DIR --out OUT_DIR [--rig rig.json]

Writes to OUT_DIR:

* ``<take>_joints3d.npz`` -- ``joints`` (T,33,3) float32 metres, ``valid`` (T,33)
  bool, ``fps``, ``t`` (T,), ``names`` (33,) MediaPipe landmark names. World
  frame (Blender convention): right-handed, ``+Z`` up, floor at ``z = 0``,
  origin under the mean pelvis, the performer's average facing along ``-Y``
  (so their left is ``+X``). ``joints`` is finite everywhere; samples that were
  bridged over a long gap (or never triangulated) have ``valid = False``.
* ``<take>_joints3d_raw.npz`` -- same keys, robust triangulation only (no
  smoothing / bone-length enforcement; nan where unsolved).
* ``rig.json`` -- capmo :class:`~capmo.calibration.RigCalibration` in that same
  world frame (``board`` is a placeholder; a ``selfcal`` block records how it
  was obtained), so it can be passed back with ``--rig``.
* ``report.json`` + ``validation/*.jpg|png`` -- metrics and images.

With ``--rig`` (e.g. a ChArUco calibration) self-calibration is skipped; the
rig's metric scale is kept and only the world frame (floor/up/heading) is
re-derived from the take. ``--compare-selfcal`` additionally self-calibrates
and reports how far those cameras are from the rig (after a best similarity).

Per-camera timelines (``--timeline``): a take extracted from a capmo session
(or whose cameras' ``t`` differ) is resampled onto one grid at the nominal fps
-- each camera's 2D keypoints are interpolated at the grid times (linear
between detected neighbours, else nearest within half a frame) -- and the
clock-based sync is passed to the epipolar offset scan as its prior, which is
then refined as usual. Ordinary videos keep the legacy behaviour.

``--start``/``--end`` (seconds of video time, the clock of the npz ``t``) trim
the take before anything else runs, e.g. to drop a calibration-board segment at
the start: sync, calibration, BA and all outputs only see the kept frames. In
the output npz ``t`` always starts at 0 (the first output row, trimmed or not);
``t_source`` keeps the take's time (video time, or session time for a capmo
session take) and ``frame_idx`` the reference (first) camera's video frame
index of every row (both are always written).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from capmo import motion, multiview as mv, selfcal as sc

BONE_REPORT = {
    "upper_arm_L": (11, 13), "upper_arm_R": (12, 14),
    "forearm_L": (13, 15), "forearm_R": (14, 16),
    "thigh_L": (23, 25), "thigh_R": (24, 26),
    "shin_L": (25, 27), "shin_R": (26, 28),
    "shoulder_width": (11, 12), "hip_width": (23, 24),
}
DRAW_BONES = motion.BONES[:24] + [(7, 8), (0, 7), (0, 8)]


# ------------------------------------------------------------------ metrics
def reproj_stats(cams, kp, vis, mask, inlier_px=20.0):
    """Per-camera reprojection stats of a robust triangulation with *cams*."""
    k = np.where(mask[..., None], kp, np.nan)
    w = np.where(mask, vis, 0.0)
    xy = np.stack([c.normalize(k[v]) for v, c in enumerate(cams)])
    X, valid, *_ = mv.robust_triangulate(cams, xy, w, max_err_px=1e9, out_px=1e9)
    out = []
    for v, c in enumerate(cams):
        m = mask[v] & valid & np.isfinite(kp[v][..., 0])
        e = np.linalg.norm(c.project(X[m]) - kp[v][m], axis=-1)
        out.append({"n": int(m.sum()), "rms": float(np.sqrt(np.mean(e ** 2))),
                    "rms_inlier": float(np.sqrt(np.mean(e[e < inlier_px] ** 2))),
                    "median": float(np.median(e)),
                    "inlier_frac": float(np.mean(e < inlier_px))})
    return out


def final_reproj(cams, joints, valid, kp, mask):
    out = []
    for v, c in enumerate(cams):
        m = mask[v] & valid & np.isfinite(kp[v][..., 0])
        e = np.linalg.norm(c.project(joints[m]) - kp[v][m], axis=-1)
        out.append({"n": int(m.sum()), "median": float(np.median(e)),
                    "rms_inlier": float(np.sqrt(np.mean(e[e < 20] ** 2)))})
    return out


def bone_stats(X, valid=None):
    out = {}
    for name, (a, b) in BONE_REPORT.items():
        L = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
        if valid is not None:
            L = np.where(valid[:, a] & valid[:, b], L, np.nan)
        L = L[np.isfinite(L)]
        out[name] = {"mean": float(L.mean()), "std": float(L.std()),
                     "cv_pct": float(100 * L.std() / L.mean()), "n": int(len(L))}
    return out


def jitter(X, valid, fps):
    """Mean |acceleration| (m/s^2) over joints with 3 consecutive valid frames."""
    m = valid[:-2] & valid[1:-1] & valid[2:]
    a = (X[2:] - 2 * X[1:-1] + X[:-2]) * fps * fps
    n = np.linalg.norm(a, axis=-1)
    core = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    pm = m[:, 23] & m[:, 24]
    pel = np.linalg.norm(0.5 * (a[:, 23] + a[:, 24]), axis=-1)
    return {"pelvis_median": float(np.median(pel[pm])),
            "mean_all": float(n[m].mean()),
            "mean_core": float(n[:, core][m[:, core]].mean()),
            "median_core": float(np.median(n[:, core][m[:, core]]))}


def floor_stats(X, valid):
    feet = [27, 28, 29, 30, 31, 32]
    ank = np.concatenate([X[valid[:, j], j, 2] for j in (27, 28)])
    zf = np.where(valid[:, feet], X[:, feet, 2], np.nan)
    with mv.quiet():
        lo = np.nanmin(zf, 1)
    lo = lo[np.isfinite(lo)]
    pct = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95)}
    return {"ankle_z": pct(ank), "lowest_foot_point_z": pct(lo),
            "frac_lowest_foot_within_5cm": float(np.mean(np.abs(lo) < 0.05)),
            "frac_lowest_foot_below_-5cm": float(np.mean(lo < -0.05))}


def lr_stats(X, valid):
    """Left/right symmetry and chirality of the final track."""
    out = {}
    for b in ("upper_arm", "forearm", "thigh", "shin"):
        (la, lb), (ra, rb) = BONE_REPORT[b + "_L"], BONE_REPORT[b + "_R"]
        L = np.linalg.norm(X[:, la] - X[:, lb], axis=-1)
        R = np.linalg.norm(X[:, ra] - X[:, rb], axis=-1)
        m = valid[:, la] & valid[:, lb] & valid[:, ra] & valid[:, rb]
        out[f"{b}_L/R_median_ratio"] = float(np.median(L[m] / R[m]))
    up = np.array([0, 0, 1.0])
    fac = np.cross(X[:, 11] - X[:, 12], up)
    nose = X[:, 0] - 0.5 * (X[:, 11] + X[:, 12])
    m = valid[:, [0, 11, 12]].all(1)
    out["frac_nose_in_front_of_shoulders"] = float(np.mean(np.sum(fac[m] * nose[m], 1) > 0))
    fl = np.cross(X[:, 23] - X[:, 24], up)
    toes = 0.5 * (X[:, 31] + X[:, 32] - X[:, 29] - X[:, 30])
    m = valid[:, [23, 24, 29, 30, 31, 32]].all(1)
    out["frac_toes_in_front_of_hips"] = float(np.mean(np.sum(fl[m] * toes[m], 1) > 0))
    m = valid[:, [11, 12, 23, 24]].all(1)
    out["frac_shoulder_hip_same_side"] = float(np.mean(
        np.sum((X[m, 11] - X[m, 12]) * (X[m, 23] - X[m, 24]), 1) > 0))
    return out


# ------------------------------------------------------------------ renders
def _read_frames(path, wanted):
    import av
    wanted = set(int(w) for w in wanted)
    out = {}
    with av.open(path) as cont:
        for i, fr in enumerate(cont.decode(video=0)):
            if i in wanted:
                out[i] = fr.to_ndarray(format="bgr24")
            if i >= max(wanted):
                break
    return out


def _side_color(a, b):
    na, nb = mv.NAMES[a], mv.NAMES[b]
    if na.startswith("left") and nb.startswith("left"):
        return (255, 140, 0)      # BGR blue-ish: left
    if na.startswith("right") and nb.startswith("right"):
        return (0, 0, 255)        # red: right
    return (0, 220, 255)


def _proxy_index(take, v, rows):
    """Proxy-video frame numbers of take rows (proxy starts at the npz window)."""
    rows = np.asarray(rows, int)
    if take.frame_idx is None:
        return rows
    first = 0 if take.first_idx is None else int(take.first_idx[v])
    return take.frame_idx[v, rows] - first


def render_overlays(take, cams, joints, valid, raw_X, keep, out_dir, n=4, log=print):
    import cv2
    paths = []
    for v, cid in enumerate(take.cam_ids):
        proxy = os.path.join(take.path, f"{cid}.proxy.mp4")
        if not os.path.exists(proxy):
            continue
        good = np.flatnonzero(keep[v] & valid[:, [11, 12, 23, 24]].all(1))
        miss = np.flatnonzero(~keep[v] & valid[:, [11, 12, 23, 24]].all(1))
        frames = list(good[np.linspace(0, len(good) - 1, n - (1 if len(miss) else 0)).astype(int)]) if len(good) else []
        if len(miss):
            frames.append(int(miss[len(miss) // 2]))
        if not frames:
            continue
        pidx = dict(zip(frames, _proxy_index(take, v, frames).tolist()))
        imgs = _read_frames(proxy, list(pidx.values()))
        tiles = []
        for fi in frames:
            im = imgs.get(pidx[fi])
            if im is None:
                continue
            h, w = im.shape[:2]
            s = np.array([w / take.size[0], h / take.size[1]])
            if np.isfinite(take.kp[v, fi]).all():
                for p in take.kp[v, fi] * s:
                    cv2.circle(im, (int(p[0]), int(p[1])), 3, (200, 200, 200), 1, cv2.LINE_AA)
            uv = cams[v].project(joints[fi]) * s
            for a, b in DRAW_BONES:
                if valid[fi, a] and valid[fi, b]:
                    cv2.line(im, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]),
                             _side_color(a, b), 2, cv2.LINE_AA)
            for j in range(33):
                if valid[fi, j]:
                    cv2.circle(im, tuple(int(x) for x in uv[j]), 2, (0, 255, 0), -1)
            # crop around the body for legibility
            pts = uv[valid[fi]]
            x0, y0 = np.clip(pts.min(0) - 60, 0, [w, h]).astype(int)
            x1, y1 = np.clip(pts.max(0) + 60, 0, [w, h]).astype(int)
            full = cv2.resize(im, (w // 2, h // 2))
            crop = im[y0:y1, x0:x1]
            if crop.size:
                ch = h // 2
                cw = max(1, int(crop.shape[1] * ch / crop.shape[0]))
                crop = cv2.resize(crop, (min(cw, w // 2), ch))
            tag = "detected" if keep[v, fi] else "NOT detected/kept"
            vf = int(take.frame_idx[v, fi]) if take.frame_idx is not None else fi
            cv2.putText(full, f"{cid} f{fi} (video {vf}) {tag}", (8, 22), 0, 0.6, (0, 0, 255), 2)
            tile = np.zeros((h // 2, w // 2 + w // 2, 3), np.uint8)
            tile[:, :w // 2] = full
            tile[:, w // 2:w // 2 + crop.shape[1]] = crop
            tiles.append(tile)
        if tiles:
            p = os.path.join(out_dir, f"reproj_{cid}.jpg")
            cv2.imwrite(p, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(p)
    log(f"  wrote {len(paths)} reprojection overlays")
    return paths


def render_3d_if_possible(*args, log=print, **kwargs):
    """:func:`render_3d`, or a note and no images when matplotlib is missing
    (it is an optional dependency; the reconstruction itself does not need it)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        log("matplotlib not installed - skipping the 3D validation plot "
            "(pip install matplotlib; --no-render silences this)")
        return []
    return render_3d(*args, **kwargs)


def render_3d(joints, valid, cams, cam_ids, fps, out_dir, raw=None, report=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(joints)
    paths = []
    frames = np.linspace(30, T - 30, 8).astype(int)
    fig = plt.figure(figsize=(20, 10))
    for i, fi in enumerate(frames):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        X = joints[fi]
        c = X[[23, 24]].mean(0)
        neck = np.stack([0.5 * (X[11] + X[12]), 0.5 * (X[7] + X[8])])
        ax.plot(*neck.T, color="0.3", lw=1.5)
        for a, b in DRAW_BONES:
            col = _side_color(a, b)
            col = (col[2] / 255, col[1] / 255, col[0] / 255)
            ax.plot(*X[[a, b]].T, color=col, lw=2 if valid[fi, a] and valid[fi, b] else 0.7,
                    ls="-" if valid[fi, a] and valid[fi, b] else ":")
        if raw is not None:
            ax.scatter(*raw[fi].T, s=4, c="k", alpha=0.4)
        gx, gy = np.meshgrid(np.linspace(c[0] - 1, c[0] + 1, 5), np.linspace(c[1] - 1, c[1] + 1, 5))
        ax.plot_wireframe(gx, gy, np.zeros_like(gx), color="0.7", lw=0.5)
        ax.quiver(c[0], c[1], 0, 0, -0.5, 0, color="g")   # -Y = nominal front
        ax.set_xlim(c[0] - 1, c[0] + 1); ax.set_ylim(c[1] - 1, c[1] + 1); ax.set_zlim(0, 2)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-90 + 35 * (i % 2))
        ax.set_title(f"frame {fi} (t={fi / fps:.1f}s)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("final skeleton (blue=left, red=right; dotted=bridged/invalid; "
                 "black dots=raw triangulation; green arrow=-Y)")
    plt.tight_layout()
    p = os.path.join(out_dir, "skeleton3d_frames.png")
    fig.savefig(p, dpi=70)
    plt.close(fig)
    paths.append(p)

    # layout: cameras + pelvis trajectory (top view and side view)
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    pel = joints[:, [23, 24]].mean(1)
    for ax, (i, j, lab) in zip(axs, [(0, 1, "top (x,y)"), (0, 2, "side (x,z)")]):
        ax.plot(pel[:, i], pel[:, j], "-", lw=0.8, color="0.4", label="pelvis")
        for cid, cam in zip(cam_ids, cams):
            C = cam.center
            fwd = cam.R[2]
            ax.plot(C[i], C[j], "o", ms=9)
            ax.annotate(cid, (C[i], C[j]), textcoords="offset points", xytext=(5, 5))
            ax.arrow(C[i], C[j], fwd[i], fwd[j], head_width=0.1, color="C3")
        if j == 2:
            ax.axhline(0, color="k", lw=0.5)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(lab)
    fig.suptitle("camera layout (world: Z up, floor z=0, performer faces -Y on average)")
    p = os.path.join(out_dir, "layout.png")
    fig.savefig(p, dpi=70)
    plt.close(fig)
    paths.append(p)

    # time series: bone lengths, foot height, views
    fig, axs = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    t = np.arange(T) / fps
    for k, (name, (a, b)) in enumerate(list(BONE_REPORT.items())[:8]):
        if raw is not None:
            axs[0].plot(t, np.linalg.norm(raw[:, a] - raw[:, b], axis=-1), ".", ms=1.5, color=f"C{k}", alpha=0.4)
        L = np.linalg.norm(joints[:, a] - joints[:, b], axis=-1)
        axs[0].plot(t, np.where(valid[:, a] & valid[:, b], L, np.nan), "-", color=f"C{k}", lw=1, label=name)
    axs[0].legend(ncol=8, fontsize=8); axs[0].set_ylabel("bone length (m)")
    axs[0].set_title("bone lengths: dots = raw triangulation, lines = final")
    for j, col in ((27, "C0"), (28, "C3"), (31, "C2"), (32, "C1")):
        axs[1].plot(t, np.where(valid[:, j], joints[:, j, 2], np.nan), color=col, lw=1, label=mv.NAMES[j])
    axs[1].axhline(0, color="k", lw=0.5); axs[1].set_ylabel("z (m)"); axs[1].legend(fontsize=8)
    axs[1].plot(t, np.where(valid[:, 0], joints[:, 0, 2], np.nan), color="0.5", lw=1, label="nose")
    axs[1].set_title("foot / nose height")
    if report is not None:
        axs[2].plot(t, report["_n_views"], ".", ms=2)
        axs[2].set_ylabel("detections agreeing (views)")
    axs[2].set_xlabel("s")
    plt.tight_layout()
    p = os.path.join(out_dir, "timeseries.png")
    fig.savefig(p, dpi=70)
    plt.close(fig)
    paths.append(p)
    return paths


# ------------------------------------------------------------------- driver
def _save_npz(path, joints, valid, fps, t, t_source=None, frame_idx=None):
    extra = {}
    if t_source is not None:
        extra["t_source"] = np.asarray(t_source, np.float64)
    if frame_idx is not None:
        extra["frame_idx"] = np.asarray(frame_idx, np.int64)
    np.savez_compressed(path, joints=np.asarray(joints, np.float32), valid=np.asarray(valid, bool),
                        fps=np.float64(fps), t=np.asarray(t, np.float64),
                        names=np.array(mv.NAMES), **extra)


def calibrate(take, *, hfov=90.0, shared_focal=True, distortion="none", rounds=2,
              stride=3, max_nfev=200, log=print):
    """Self-calibrate (arbitrary gauge). Returns (cams, info)."""
    f0 = sc.focal_from_hfov(take.size[0], hfov)
    log(f"self-calibration: f0={f0:.0f}px (HFOV {hfov} deg), shared focal={shared_focal}, "
        f"distortion={distortion}")
    cams0 = sc.initial_cameras(take, f0, log=log)
    cams = cams0
    bas = []
    for r in range(rounds):
        res = mv.reconstruct_take(cams, take.kp, take.vis)
        kp_l, vis_l = _labelled(take, res)
        X0 = res.X.copy()
        m = np.zeros(X0.shape[:2], bool)
        m[np.ix_(np.arange(0, take.T, stride), sc.CALIB_JOINTS)] = True
        X0[~m] = np.nan
        ba = sc.bundle_adjust(cams, kp_l, vis_l, X0, res.used, shared_focal=shared_focal,
                              distortion=distortion, max_nfev=max_nfev, log=log)
        cams = ba.cams
        bas.append(ba)
    return cams, {"cams0": cams0, "ba": bas, "f0": f0}


def _labelled(take, res):
    kp = np.where(res.kept[..., None, None], take.kp, np.nan)
    vis = np.where(res.kept[..., None], take.vis, 0.0)
    return mv.apply_swaps(kp, res.swaps), mv.apply_swaps(vis, res.swaps)


def run(take_dir, out_dir, rig=None, *, hfov=90.0, shared_focal=True, distortion="none",
        scale_mode="limbs", shoulder_width=0.38, cutoff_hz=7.0, core_hz=4.0, mid_hz=6.0,
        rounds=2, stride=3,
        max_nfev=200, apply_sync=True, render=True, prefix=None, start=None, end=None,
        compare_selfcal=False, timeline="auto", log=print):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vdir = os.path.join(out_dir, "validation")
    os.makedirs(vdir, exist_ok=True)
    take = sc.load_take(take_dir, timeline=timeline)
    if take.timeline == "resampled":
        log(f"timeline: per-camera times resampled onto a common {take.fps:g} fps grid"
            + (" (capmo session: host-clock synced)" if take.session else ""))
    T_full = take.T
    trimmed = start is not None or end is not None
    if trimmed:
        take = sc.trim_take(take, start, end)
        log(f"trim: keeping t={take.t[0]:.3f}..{take.t[-1]:.3f} s ({take.T} of {T_full} frames, "
            f"video frames {int(take.frame_idx[0, 0])}..{int(take.frame_idx[0, -1])} of "
            f"{take.cam_ids[0]})")
    prefix = prefix or os.path.basename(os.path.normpath(take_dir))
    report = {"take": take_dir, "cams": take.cam_ids, "T": take.T, "fps": take.fps,
              "size": take.size}
    report["timeline"] = {"mode": take.timeline,
                          "sync_prior_frames": (dict(zip(take.cam_ids, take.sync_prior.round(3).tolist()))
                                                if take.sync_prior is not None else None)}
    if take.session:
        report["timeline"]["session"] = {k: take.session.get(k) for k in
                                         ("session", "t0_ms", "nominal_fps", "sync")}
    report["trim"] = {"start": start, "end": end, "T_full": T_full,
                      "t_first_source": float(take.t[0]), "t_last_source": float(take.t[-1]),
                      "frame_idx_first": int(take.frame_idx[0, 0]),
                      "frame_idx_last": int(take.frame_idx[0, -1]),
                      "t_rebased": True}
    det = (take.vis > 0).any(-1)
    report["detection_rate"] = dict(zip(take.cam_ids, det.mean(1).round(3).tolist()))
    log(f"take {take_dir}: {take.V} cams, {take.T} frames @ {take.fps} fps; detection "
        + ", ".join(f"{c} {r:.0%}" for c, r in zip(take.cam_ids, det.mean(1))))

    # --- sync (calibration-free epipolar scan)
    log("sync: epipolar scan ...")
    sync = sc.estimate_offsets(take, prior=take.sync_prior)
    off = sync["offsets"]
    report["sync_epipolar"] = {
        "offsets_frames": dict(zip(take.cam_ids, off.round(3).tolist())),
        "pairs": {f"{take.cam_ids[i]}-{take.cam_ids[j]}": {
            "offset": p["offset"], "common_frames": p["common"], "trusted": p["trusted"],
            "median_sampson_px": {str(k): round(v, 3) for k, v in p["curve"].items()}}
            for (i, j), p in sync["pairs"].items()}}
    report["sync_epipolar"]["unresolved"] = [take.cam_ids[v] for v in sync["unresolved"]]
    log("  epipolar offsets (frames): " + ", ".join(f"{c} {o:+.2f}" for c, o in zip(take.cam_ids, off))
        + "; untrusted pairs: " + (", ".join(f"{take.cam_ids[i]}-{take.cam_ids[j]}"
                                            for (i, j), p in sync["pairs"].items() if not p["trusted"]) or "none"))
    shift = np.zeros(take.V, int)

    def apply_shift(sh):
        for v in range(take.V):
            if sh[v]:
                take.kp[v], take.vis[v] = sc.shift_track(take.kp[v], take.vis[v], float(sh[v]))

    if apply_sync:
        shift = np.where(np.abs(off) >= 0.75, np.round(off), 0).astype(int)
        if np.any(shift):
            log(f"  applying integer shifts {shift.tolist()}")
            apply_shift(shift)

    # --- calibration (+ calibrated sync check, which can trigger one redo)
    if rig:
        cams = sc.from_rig(rig, take.cam_ids)
        cal_info = None
        log(f"using rig {rig}")
    else:
        cams, cal_info = calibrate(take, hfov=hfov, shared_focal=shared_focal,
                                   distortion=distortion, rounds=rounds, stride=stride,
                                   max_nfev=max_nfev, log=log)
    if apply_sync:
        res0 = mv.reconstruct_take(cams, take.kp, take.vis)
        rs0 = sc.refine_offsets(cams, take, res0.kept)
        # re-reference to camera 0 (as estimate_offsets): its rows label the output
        # (t, frame_idx), so camera 0 itself must not move
        extra = np.asarray(rs0["integer_offsets"], int)
        extra = extra - extra[0]
        log("  calibrated sync check (frames): " + ", ".join(
            f"{c} {o:+.2f}" for c, o in zip(take.cam_ids, rs0["offsets"])))
        if np.any(extra):
            log(f"  applying extra integer shifts {extra.tolist()} and recalibrating")
            apply_shift(extra)
            shift = shift + extra
            if not rig:
                cams, cal_info = calibrate(take, hfov=hfov, shared_focal=shared_focal,
                                           distortion=distortion, rounds=rounds, stride=stride,
                                           max_nfev=max_nfev, log=log)
    report["sync_applied_shift_frames"] = dict(zip(take.cam_ids, shift.tolist()))
    if rig and compare_selfcal:
        log("compare: self-calibrating from the body to compare with the rig ...")
        try:
            cams_sc, _ = calibrate(take, hfov=hfov, shared_focal=shared_focal,
                                   distortion=distortion, rounds=rounds, stride=stride,
                                   max_nfev=max_nfev, log=log)
            cmp = sc.compare_cameras(cams_sc, cams, take.cam_ids)
            report["compare_selfcal"] = cmp
            log("  selfcal vs rig (after similarity, scale %.3f): " % cmp["scale_a_to_b"] + ", ".join(
                f"{c} {d['center_diff_m'] * 100:.1f} cm / {d['rotation_diff_deg']:.2f} deg / "
                f"f {d['focal_diff_pct']:+.1f}%" for c, d in cmp["cameras"].items()))
        except Exception as exc:          # a sanity check must not kill the run
            report["compare_selfcal"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"  selfcal comparison failed: {exc}")

    # --- world frame
    res = mv.reconstruct_take(cams, take.kp, take.vis)
    sim, winfo = sc.world_alignment(res.X, scale_mode=scale_mode, shoulder_width=shoulder_width,
                                    scale=1.0 if rig else None)
    cams_w = sim.apply_cameras(cams)
    log(f"world: scale {sim.s:.4f} ({winfo['scale_mode']}), floor inliers "
        f"{winfo['floor_inlier_frac']:.0%}, facing resultant {winfo['facing_resultant']:.2f}")
    report["world"] = winfo
    if cal_info is not None:
        cam_heights = {c: float(k.center[2]) for c, k in zip(take.cam_ids, cams_w)}
        report["calibration"] = {
            "method": "self-calibration from body keypoints (E + PnP init, sparse BA)",
            "f0_px": cal_info["f0"], "shared_focal": shared_focal, "distortion": distortion,
            "focal_px": [c.f for c in cams_w], "k1": [float(c.dist[0]) for c in cams_w],
            "hfov_deg": [float(np.degrees(2 * np.arctan(take.size[0] / 2 / c.f))) for c in cams_w],
            "camera_height_m": cam_heights,
            "camera_centers_m": {c: k.center.round(3).tolist() for c, k in zip(take.cam_ids, cams_w)},
            "ba_rounds": [{"rms_all_before": b.rms_before.tolist(), "rms_all_after": b.rms_after.tolist(),
                           "rms_inlier_before": b.rms_in_before.tolist(),
                           "rms_inlier_after": b.rms_in_after.tolist(),
                           "median_before": b.med_before.tolist(), "median_after": b.med_after.tolist(),
                           "n_obs": b.n_obs.tolist(), "nfev": b.nfev} for b in cal_info["ba"]],
        }
    sc.save_rig(os.path.join(out_dir, "rig.json"), cams_w, take.cam_ids, take.size, extra={
        "note": ("world frame: metres, +Z up, floor z=0, origin under mean pelvis, performer "
                 "faces -Y on average (Blender front). board is a placeholder."),
        "source": "rig.json" if rig else "capmo.selfcal (body keypoints)",
        "scale_mode": winfo["scale_mode"], "scale": sim.s})

    # --- final reconstruction in the world frame
    res = mv.reconstruct_take(cams_w, take.kp, take.vis)
    kp_l, vis_l = _labelled(take, res)
    if cal_info is not None:
        before = reproj_stats(cal_info["cams0"], kp_l, vis_l, res.used)
        after = reproj_stats(cams_w, kp_l, vis_l, res.used)
        report["reprojection_initial_vs_ba"] = {
            c: {"initial": b, "after_ba": a} for c, b, a in zip(take.cam_ids, before, after)}
        log("  reprojection rms_inlier (px) initial -> BA: " + ", ".join(
            f"{c} {b['rms_inlier']:.2f}->{a['rms_inlier']:.2f}" for c, b, a in zip(take.cam_ids, before, after)))
    rs = sc.refine_offsets(cams_w, take, res.kept)
    report["sync_reprojection"] = {
        "offsets_frames_subframe_estimate": dict(zip(take.cam_ids, rs["offsets"].round(3).tolist())),
        "integer_offsets_frames": dict(zip(take.cam_ids, rs["integer_offsets"].tolist())),
        "n_samples": dict(zip(take.cam_ids, rs["n_samples"])),
        "curves_median_px": [{str(k): round(v, 3) for k, v in c.items()} for c in rs["curves"]]}
    log("  reprojection sync check (frames): " + ", ".join(
        f"{c} {o:+.2f}" for c, o in zip(take.cam_ids, rs["offsets"])))

    raw_X, raw_valid = res.X, res.valid
    t_out = take.t - take.t[0]                # always starts at 0; t_source keeps take.t
    fidx_out = take.frame_idx[0]
    _save_npz(os.path.join(out_dir, f"{prefix}_joints3d_raw.npz"), raw_X, raw_valid, take.fps, t_out,
              take.t, fidx_out)
    cl = motion.clean_track(raw_X, raw_valid, take.fps, cutoff_hz=cutoff_hz, core_hz=core_hz,
                            mid_hz=mid_hz, log=log)
    joints, valid = cl["joints"], cl["valid"]
    _save_npz(os.path.join(out_dir, f"{prefix}_joints3d.npz"), joints, valid, take.fps, t_out,
              take.t, fidx_out)

    # --- metrics
    nk = res.kept.sum(0)
    core = [11, 12, 23, 24]
    report["views"] = {
        "frames_ge2_detections": float(np.mean(det.sum(0) >= 2)),
        "frames_ge3_detections": float(np.mean(det.sum(0) >= 3)),
        "frames_ge2_agreeing": float(np.mean(nk >= 2)),
        "frames_ge3_agreeing": float(np.mean(nk >= 3)),
        "frames_core_triangulated": float(np.mean(raw_valid[:, core].all(1))),
        "joint_samples_valid_raw": float(raw_valid.mean()),
        "joint_samples_valid_final": float(valid.mean()),
        "per_joint_valid_final": dict(zip(mv.NAMES, valid.mean(0).round(3).tolist())),
        "detections_rejected_by_consensus": dict(zip(take.cam_ids, (det & ~res.kept).sum(1).tolist())),
        "swap_rate_per_view_face_arms_legs": {c: (res.swaps[v][det[v]].mean(0).round(3).tolist())
                                              for v, c in enumerate(take.cam_ids)},
        "views_used_median": float(np.median(res.n_views[raw_valid])),
    }
    report["_n_views"] = nk
    report["bones_raw"] = bone_stats(raw_X, raw_valid)
    report["bones_smoothed_only"] = bone_stats(cl["smoothed"], valid)
    report["jitter_smoothed_only"] = jitter(cl["smoothed"], valid, take.fps)
    d = np.linalg.norm(joints - cl["smoothed"], axis=-1)[valid]
    report["bone_enforcement_shift_m"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95))}
    d = np.linalg.norm(joints - raw_X, axis=-1)[valid & raw_valid]
    report["final_vs_raw_m"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95))}
    report["bones_final"] = bone_stats(joints, valid)
    report["jitter_mps2"] = {"raw": jitter(raw_X, raw_valid, take.fps),
                             "final": jitter(joints, valid, take.fps)}
    report["floor"] = floor_stats(joints, valid)
    report["left_right"] = lr_stats(joints, valid)
    report["left_right"]["n_part_relabels_temporal"] = cl["n_lr_fixed"]
    report["final_reprojection_px"] = dict(zip(take.cam_ids, final_reproj(cams_w, joints, valid, kp_l, res.used)))
    report["cleanup"] = {"cutoff_hz": {"core": core_hz, "elbow_knee": mid_hz, "extremities": cutoff_hz},
                         "outlier_samples": cl["n_outliers"],
                         "bad_bone_samples": cl["n_bad_bone"],
                         "gap_samples_filled": int(cl["filled"].sum())}
    facing = np.cross(joints[:, 11] - joints[:, 12] + joints[:, 23] - joints[:, 24], [0, 0, 1.0])[:, :2]
    facing = facing / np.linalg.norm(facing, axis=1, keepdims=True)
    report["mean_facing_xy"] = facing[valid[:, [11, 12, 23, 24]].all(1)].mean(0).round(3).tolist()
    if render:
        log("rendering validation images ...")
        report["images"] = render_overlays(take, cams_w, joints, valid, raw_X, res.kept, vdir, log=log)
        report["images"] += render_3d_if_possible(joints, valid, cams_w, take.cam_ids, take.fps,
                                                  vdir, raw=raw_X, report=report, log=log)
    report.pop("_n_views")
    report["elapsed_s"] = time.time() - t_start
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    _print_summary(report, log)
    return report


def _print_summary(r, log):
    log("=" * 70)
    log("bone            raw mean/cv%      final mean/cv%")
    for k in BONE_REPORT:
        a, b = r["bones_raw"][k], r["bones_final"][k]
        log(f"  {k:15s} {a['mean']:.3f} {a['cv_pct']:5.1f}%     {b['mean']:.3f} {b['cv_pct']:5.2f}%")
    v = r["views"]
    log(f"frames >=2/>=3 agreeing views: {v['frames_ge2_agreeing']:.1%} / {v['frames_ge3_agreeing']:.1%}; "
        f"valid joint samples raw {v['joint_samples_valid_raw']:.1%}, final {v['joint_samples_valid_final']:.1%}")
    j = r["jitter_mps2"]
    log(f"jitter mean|acc| core joints: raw {j['raw']['mean_core']:.2f} -> final {j['final']['mean_core']:.2f} m/s^2")
    fl = r["floor"]
    log(f"lowest foot z p5/p50/p95: {fl['lowest_foot_point_z']['p5']:+.3f} {fl['lowest_foot_point_z']['p50']:+.3f} "
        f"{fl['lowest_foot_point_z']['p95']:+.3f}; ankle z p50 {fl['ankle_z']['p50']:.3f}")
    log(f"left/right: {r['left_right']}")
    log("=" * 70)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default=None, help="rig.json (skips self-calibration)")
    ap.add_argument("--hfov", type=float, default=90.0, help="initial horizontal FOV guess (deg)")
    ap.add_argument("--per-camera-focal", action="store_true")
    ap.add_argument("--distortion", choices=["none", "k1", "k1k2"], default="none")
    ap.add_argument("--scale", choices=["limbs", "shoulder"], default="limbs")
    ap.add_argument("--shoulder-width", type=float, default=0.38)
    ap.add_argument("--cutoff", type=float, default=7.0, help="low-pass cutoff, extremities (Hz)")
    ap.add_argument("--core-cutoff", type=float, default=4.0, help="low-pass cutoff, head/torso (Hz)")
    ap.add_argument("--mid-cutoff", type=float, default=6.0, help="low-pass cutoff, elbows/knees (Hz)")
    ap.add_argument("--rounds", type=int, default=2, help="BA rounds")
    ap.add_argument("--stride", type=int, default=3, help="BA frame subsampling")
    ap.add_argument("--no-sync", action="store_true", help="don't apply estimated integer offsets")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--prefix", default=None, help="output file prefix (default: take dir name)")
    ap.add_argument("--start", type=float, default=None,
                    help="drop frames before this video time (s), e.g. the board segment")
    ap.add_argument("--end", type=float, default=None, help="drop frames after this video time (s)")
    ap.add_argument("--compare-selfcal", action="store_true",
                    help="with --rig: also self-calibrate and report camera differences")
    ap.add_argument("--timeline", choices=["auto", "first", "resample"], default="auto",
                    help="per-camera timestamps: resample onto a common fps grid (auto: for "
                         "capmo-session takes or when the cameras' t differ), or use the "
                         "first camera's t for all (legacy)")
    a = ap.parse_args(argv)
    run(a.take_dir, a.out, a.rig, hfov=a.hfov, shared_focal=not a.per_camera_focal,
        distortion=a.distortion, scale_mode=a.scale, shoulder_width=a.shoulder_width,
        cutoff_hz=a.cutoff, core_hz=a.core_cutoff, mid_hz=a.mid_cutoff, rounds=a.rounds, stride=a.stride, apply_sync=not a.no_sync,
        render=not a.no_render, prefix=a.prefix, start=a.start, end=a.end,
        compare_selfcal=a.compare_selfcal, timeline=a.timeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
