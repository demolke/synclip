"""
Self-calibration + robust multi-view reconstruction on a synthetic take.

Four ceiling cameras watch a synthetic 33-landmark walker; its projections get
pixel noise, MediaPipe-style left/right swaps (whole body and legs only),
dropped detections and a "ghost" person in clutter. The robust triangulation
must undo all of that with known cameras; the self-calibration must recover
the rig up to a similarity from the keypoints alone; and the CLI tool must
produce the Blender interchange file in the documented world frame.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
cv2 = pytest.importorskip("cv2")

from .. import motion, multiview as mv, selfcal as sc  # noqa: E402
from .. import synth  # noqa: E402

W, H, F, FPS = 1280, 720, 750.0, 30.0

# Template in a body frame: x = performer's right, y = forward, z = up (m).
# Left landmarks sit at -x. Limb lengths match selfcal.LIMB_PRIOR.
_T = {
    0: (0, .10, 1.62), 1: (-.02, .09, 1.66), 2: (-.035, .085, 1.66), 3: (-.05, .08, 1.66),
    4: (.02, .09, 1.66), 5: (.035, .085, 1.66), 6: (.05, .08, 1.66), 7: (-.075, 0, 1.62),
    8: (.075, 0, 1.62), 9: (-.025, .09, 1.57), 10: (.025, .09, 1.57),
    11: (-.19, 0, 1.42), 12: (.19, 0, 1.42), 13: (-.19, 0, 1.13), 14: (.19, 0, 1.13),
    15: (-.19, 0, .88), 16: (.19, 0, .88), 17: (-.2, .02, .80), 18: (.2, .02, .80),
    19: (-.19, .04, .79), 20: (.19, .04, .79), 21: (-.17, .04, .83), 22: (.17, .04, .83),
    23: (-.10, 0, .95), 24: (.10, 0, .95), 25: (-.10, 0, .53), 26: (.10, 0, .53),
    27: (-.10, 0, .12), 28: (.10, 0, .12), 29: (-.10, -.05, .05), 30: (.10, -.05, .05),
    31: (-.10, .16, .03), 32: (.10, .16, .03),
}
TEMPLATE = np.array([_T[i] for i in range(33)], float)
LEG = {"left": [25, 27, 29, 31], "right": [26, 28, 30, 32]}
ARM = {"left": [13, 15, 17, 19, 21], "right": [14, 16, 18, 20, 22]}


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def walker(T=150, seed=0):
    """(T,33,3) world joints (Z up, floor z=0); mostly facing -Y."""
    rng = np.random.default_rng(seed)
    out = np.zeros((T, 33, 3))
    ph = rng.uniform(0, 2 * np.pi)
    for t in range(T):
        P = TEMPLATE.copy()
        sw = 0.45 * np.sin(2 * np.pi * 0.9 * t / FPS + ph)
        for side, sgn in (("left", 1), ("right", -1)):
            hip = P[23 if side == "left" else 24].copy()
            P[LEG[side]] = (P[LEG[side]] - hip) @ _rot_x(sgn * sw).T + hip
            sh = P[11 if side == "left" else 12].copy()
            P[ARM[side]] = (P[ARM[side]] - sh) @ _rot_x(-sgn * 0.7 * sw).T + sh
        P[:, 2] -= P[[29, 30, 31, 32], 2].min() - 0.02          # lowest foot on floor
        # heading: around -Y (+-50 deg), walking an arc in the middle of the room
        yaw = np.pi + 0.9 * np.sin(2 * np.pi * t / T)            # body +y -> world dir
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        pos = np.array([1.2 * np.sin(2 * np.pi * t / T), -1.0 + 2.0 * t / T, 0.0])
        out[t] = P @ Rz.T + pos
    return out


def rig(seed=0):
    """Four ceiling-corner cameras (OpenCV convention) looking at the middle."""
    rng = np.random.default_rng(seed)
    cams = []
    for i, (x, y) in enumerate([(-3.2, -3.4), (-2.6, 3.1), (3.4, 2.4), (3.0, -3.9)]):
        eye = np.array([x, y, 2.8]) + rng.normal(0, 0.1, 3)
        R, t = synth.look_at(eye, target=(0, 0, 0.6) + rng.normal(0, 0.2, 3), up=(0, 0, 1))
        K = sc.default_K((W, H), F * (1 + 0.0 * i))
        cams.append(mv.Camera(K, np.zeros(5), R, t))
    return cams


def observe(cams, X, *, noise=1.5, swap_rate=0.08, legs_rate=0.05, drop_rate=0.1,
            ghost_rate=0.05, seed=1):
    """Project + corrupt like MediaPipe: (kp, vis, truth swap flags)."""
    rng = np.random.default_rng(seed)
    V, T = len(cams), len(X)
    kp = np.stack([c.project(X) for c in cams]) + rng.normal(0, noise, (V, T, 33, 2))
    vis = np.clip(rng.normal(0.9, 0.08, (V, T, 33)), 0.3, 1.0)
    swapped = np.zeros((V, T), int)
    for v in range(V):
        for t in range(T):
            r = rng.random()
            if r < swap_rate:
                kp[v, t] = kp[v, t, mv.FULL_SWAP]
                swapped[v, t] = 1
            elif r < swap_rate + legs_rate:
                kp[v, t] = kp[v, t, mv.swap_perm(["legs"])]
                swapped[v, t] = 2
    ghost = rng.random((V, T)) < ghost_rate
    off = np.array([300.0, -150.0])
    kp[ghost] = kp[ghost] + off + rng.normal(0, 30, (ghost.sum(), 1, 2))
    drop = rng.random((V, T)) < drop_rate
    drop[0, :] &= False                       # keep >=2 views almost always
    drop[1, :] &= False
    out = (kp[..., 0] < 0) | (kp[..., 0] >= W) | (kp[..., 1] < 0) | (kp[..., 1] >= H)
    drop |= out.any(-1)
    kp[drop] = np.nan
    vis[drop] = 0.0
    return kp, vis, swapped, ghost & ~drop


def test_projection_round_trip_with_distortion():
    cam = rig()[0]
    cam.dist = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    X = walker(5)[0]
    uv = cam.project(X)
    xy = cam.normalize(uv)
    Xc = X @ cam.R.T + cam.t
    assert np.allclose(xy, Xc[:, :2] / Xc[:, 2:], atol=1e-7)


def test_robust_reconstruction_known_cameras():
    cams = rig()
    X = walker(120)
    kp, vis, swapped, ghost = observe(cams, X)
    res = mv.reconstruct_take(cams, kp, vis)
    # ghosts rejected, genuine detections mostly kept
    det = np.isfinite(kp[..., 0, 0])
    assert not (res.kept & ghost).any()
    assert (res.kept & det & ~ghost).sum() / (det & ~ghost).sum() > 0.95
    body = [0, 7, 8] + list(range(11, 17)) + list(range(23, 33))
    err = np.linalg.norm(res.X - X, axis=-1)[:, body]
    ok = res.valid[:, body]
    assert ok.mean() > 0.9
    assert np.median(err[ok]) < 0.012          # metres
    assert np.percentile(err[ok], 99) < 0.05   # no residual L/R mix-ups
    # cleanup: bone lengths constant, error stays small, output finite
    cl = motion.clean_track(res.X, res.valid, FPS, log=lambda *a: None)
    J = cl["joints"]
    assert np.isfinite(J).all()
    L = motion.bone_lengths(J)[:, :24]
    cv = np.nanstd(L, 0) / np.nanmean(L, 0)
    assert cv.max() < 0.01
    e2 = np.linalg.norm(J - X, axis=-1)[:, body][cl["valid"][:, body]]
    assert np.median(e2) < 0.015


def _similarity(A, B):
    """Umeyama: s, R, t with B ~= s R A + t."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum()
    return s, R, mb - s * R @ ma


def _take(cams, X, **kw):
    kp, vis, *_ = observe(cams, X, **kw)
    return sc.Take([f"cam{c}" for c in "ABCD"], kp, vis, np.arange(len(X)) / FPS, FPS, (W, H))


@pytest.mark.slow
def test_selfcal_recovers_rig_up_to_similarity():
    gt = rig()
    X = walker(150)
    take = _take(gt, X, noise=1.0, ghost_rate=0.02)
    cams = sc.initial_cameras(take, sc.focal_from_hfov(W, 90.0), log=lambda *a: None)
    for _ in range(2):
        res = mv.reconstruct_take(cams, take.kp, take.vis)
        kp = mv.apply_swaps(np.where(res.kept[..., None, None], take.kp, np.nan), res.swaps)
        vis = mv.apply_swaps(np.where(res.kept[..., None], take.vis, 0), res.swaps)
        X0 = res.X.copy()
        m = np.zeros(X0.shape[:2], bool)
        m[np.ix_(np.arange(0, take.T, 2), sc.CALIB_JOINTS)] = True
        X0[~m] = np.nan
        ba = sc.bundle_adjust(cams, kp, vis, X0, res.used, shared_focal=True,
                              max_nfev=100, log=lambda *a: None)
        cams = ba.cams
    assert abs(cams[0].f / F - 1) < 0.03
    assert (ba.rms_in_after < 2.0).all()
    Cg = np.stack([c.center for c in gt])
    Ce = np.stack([c.center for c in cams])
    s, R, t = _similarity(Ce, Cg)
    assert np.abs(s * Ce @ R.T + t - Cg).max() < 0.10           # metres over a ~9 m rig
    for c_gt, c_est in zip(gt, cams):
        dR = c_gt.R @ (c_est.R @ R.T).T
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        assert ang < 1.0
    # world alignment: metric scale from the limb prior, Z up, floor at 0, faces -Y
    res = mv.reconstruct_take(cams, take.kp, take.vis)
    sim, info = sc.world_alignment(res.X)
    Xw = sim.apply(res.X)
    camw = sim.apply_cameras(cams)
    # the world frame is fixed by the body, so compare against the ground truth
    # world up to a rotation about Z + xy translation
    s2, R2, t2 = _similarity(np.stack([c.center for c in camw]), Cg)
    assert abs(s2 - 1) < 0.04
    assert np.degrees(np.arccos(np.clip(R2[2, 2], -1, 1))) < 2.0  # Z axes agree
    assert abs(t2[2]) < 0.05                                        # floor height agrees
    fac = np.cross(Xw[:, 11] - Xw[:, 12], [0, 0, 1])[:, :2]
    fac = fac[np.isfinite(fac).all(1)]
    fac /= np.linalg.norm(fac, axis=1, keepdims=True)
    assert fac.mean(0)[1] < -0.5                                    # facing -Y


def test_sync_offset_detected():
    gt = rig()
    X = walker(160)
    kp, vis, *_ = observe(gt, X, swap_rate=0.03, legs_rate=0.0, ghost_rate=0.0, drop_rate=0.0)
    # camC lags by 2 frames: its frame k shows the pose of frame k-2
    kp[2, 2:] = kp[2, :-2].copy()
    vis[2, 2:] = vis[2, :-2].copy()
    take = sc.Take(list("ABCD"), kp, vis, np.arange(len(X)) / FPS, FPS, (W, H))
    r = sc.estimate_offsets(take, max_shift=3)
    assert round(r["offsets"][2]) == 2
    assert np.abs(np.round(r["offsets"][[0, 1, 3]])).max() == 0
    rr = sc.refine_offsets(gt, take)
    assert rr["integer_offsets"][2] == 2 and abs(rr["offsets"][2] - 2) < 0.4


@pytest.mark.slow
def test_reconstruct_take_tool(tmp_path):
    from ..tools import reconstruct_take as tool
    gt = rig()
    X = walker(150)
    kp, vis, *_ = observe(gt, X, noise=1.0, ghost_rate=0.02)
    take_dir = tmp_path / "take1"
    take_dir.mkdir()
    for v, c in enumerate("ABCD"):
        det = np.isfinite(kp[v, :, 0, 0])
        np.savez_compressed(take_dir / f"cam{c}.npz", kp=np.nan_to_num(kp[v]).astype(np.float32),
                            vis=vis[v].astype(np.float32), detected=det, t=np.arange(150) / FPS,
                            width=W, height=H, fps=FPS)
    out = tmp_path / "out"
    rep = tool.run(str(take_dir), str(out), render=False, max_nfev=80, log=lambda *a: None)
    z = np.load(out / "take1_joints3d.npz")
    assert z["joints"].shape == (150, 33, 3) and z["joints"].dtype == np.float32
    assert z["valid"].shape == (150, 33) and z["valid"].dtype == bool
    assert float(z["fps"]) == FPS and z["t"].shape == (150,)
    assert list(z["names"]) == list(mv.NAMES) and z["names"][32] == "right_foot_index"
    assert np.isfinite(z["joints"]).all()
    assert os.path.exists(out / "take1_joints3d_raw.npz")
    # world frame: the recovered body matches the truth up to a yaw + xy shift
    J = z["joints"].astype(float)
    v = z["valid"]
    s, R, t = _similarity(J[v], X[v])
    assert abs(s - 1) < 0.05 and R[2, 2] > np.cos(np.radians(3))
    assert abs(t[2]) < 0.05
    assert rep["bones_final"]["thigh_L"]["cv_pct"] < 1.0
    assert rep["mean_facing_xy"][1] < -0.5
    # the saved rig reproduces the output frame and loads back
    cams = sc.from_rig(str(out / "rig.json"), ["camA", "camB", "camC", "camD"])
    uv = cams[0].project(J[10])
    assert np.nanmedian(np.linalg.norm(uv - kp[0, 10], axis=-1)) < 15
    with open(out / "report.json") as f:
        assert "reprojection_initial_vs_ba" in json.load(f)
    # --rig path skips self-calibration and keeps the metric frame
    rep2 = tool.run(str(take_dir), str(tmp_path / "out2"), str(out / "rig.json"),
                    render=False, apply_sync=False, log=lambda *a: None)
    assert "calibration" not in rep2
    z2 = np.load(tmp_path / "out2" / "take1_joints3d.npz")
    assert np.nanmedian(np.linalg.norm(z2["joints"] - z["joints"], axis=-1)) < 0.03


@pytest.mark.slow
def test_reconstruct_take_trims_leading_board_segment(tmp_path):
    """A take whose first second has no performer (board segment): --start drops it."""
    from ..tools import reconstruct_take as tool
    gt = rig()
    X = walker(150)
    kp, vis, *_ = observe(gt, X, noise=1.0, ghost_rate=0.02)
    n0 = 30                                   # 1 s of "board" frames: nobody detected
    take_dir = tmp_path / "take1"
    take_dir.mkdir()
    for v, c in enumerate("ABCD"):
        k = np.concatenate([np.full((n0, 33, 2), np.nan), kp[v]])
        w = np.concatenate([np.zeros((n0, 33)), vis[v]])
        det = np.isfinite(k[:, 0, 0])
        np.savez_compressed(take_dir / f"cam{c}.npz", kp=np.nan_to_num(k).astype(np.float32),
                            vis=w.astype(np.float32), detected=det,
                            t=np.arange(150 + n0) / FPS, frame_idx=np.arange(150 + n0),
                            width=W, height=H, fps=FPS)
    gt_ids = [f"cam{c}" for c in "ABCD"]
    sc.save_rig(str(tmp_path / "gt_rig.json"), gt, gt_ids, (W, H))
    # the trimmed take is what every stage sees
    tk = sc.trim_take(sc.load_take(str(take_dir), gt_ids), 1.0)
    assert tk.T == 150 and tk.frame_idx[0, 0] == n0 and abs(tk.t[0] - 1.0) < 1e-9
    out = tmp_path / "out"
    rep = tool.run(str(take_dir), str(out), str(tmp_path / "gt_rig.json"), start=1.0,
                   render=False, compare_selfcal=True, max_nfev=80, log=lambda *a: None)
    z = np.load(out / "take1_joints3d.npz")
    assert z["joints"].shape == (150, 33, 3)
    assert list(z["frame_idx"]) == list(range(n0, n0 + 150))
    assert abs(z["t"][0]) < 1e-9 and abs(z["t_source"][0] - 1.0) < 1e-9
    assert np.allclose(np.diff(z["t"]), 1 / FPS)
    assert rep["T"] == 150 and rep["trim"]["T_full"] == 180 and rep["trim"]["frame_idx_first"] == n0
    assert rep["sync_applied_shift_frames"] == {c: 0 for c in gt_ids}
    J = z["joints"].astype(float)
    v = z["valid"]
    assert v[:, [11, 12, 23, 24]].all(1).mean() > 0.95      # no empty leading frames left
    s, R, t = _similarity(J[v], X[v])
    assert abs(s - 1) < 0.02 and R[2, 2] > np.cos(np.radians(3))
    # the self-calibration sanity check agrees with the (true) rig
    cmp = rep["compare_selfcal"]["cameras"]
    assert max(d["rotation_diff_deg"] for d in cmp.values()) < 1.5
    assert max(d["center_diff_m"] for d in cmp.values()) < 0.15
    # default behaviour unchanged: no trimming, t kept
    zr = np.load(out / "take1_joints3d_raw.npz")
    assert list(zr["frame_idx"]) == list(range(n0, n0 + 150))
    with pytest.raises(ValueError, match="keep"):
        sc.trim_take(sc.load_take(str(take_dir), gt_ids), 100.0)


def _write_take(take_dir, kp, vis, t):
    take_dir.mkdir()
    for v, c in enumerate("ABCD"):
        det = np.isfinite(kp[v, :, 0, 0])
        np.savez_compressed(take_dir / f"cam{c}.npz", kp=np.nan_to_num(kp[v]).astype(np.float32),
                            vis=vis[v].astype(np.float32), detected=det, t=t,
                            frame_idx=np.arange(len(t)), width=W, height=H, fps=FPS)


@pytest.mark.slow
def test_reconstruct_reference_camera_never_shifts(tmp_path, monkeypatch):
    """refine_offsets may report a shift for camera 0: it is re-referenced (the
    others move instead), so camera 0's rows keep labelling t / frame_idx."""
    from ..tools import reconstruct_take as tool
    gt = rig()
    X = walker(150)
    kp, vis, *_ = observe(gt, X, noise=1.0, ghost_rate=0.0, swap_rate=0.0, legs_rate=0.0)
    _write_take(tmp_path / "take1", kp, vis, np.arange(150) / FPS)
    ids = [f"cam{c}" for c in "ABCD"]
    sc.save_rig(str(tmp_path / "rig.json"), gt, ids, (W, H))
    real = sc.refine_offsets
    calls = []

    def fake(*a, **k):
        r = real(*a, **k)
        if not calls:
            r["integer_offsets"] = np.array([1, 0, 0, 0])
        calls.append(1)
        return r
    monkeypatch.setattr(sc, "refine_offsets", fake)
    rep = tool.run(str(tmp_path / "take1"), str(tmp_path / "out"), str(tmp_path / "rig.json"),
                   render=False, log=lambda *a: None)
    assert rep["sync_applied_shift_frames"] == {"camA": 0, "camB": -1, "camC": -1, "camD": -1}


@pytest.mark.slow
def test_reconstruct_output_t_always_starts_at_zero(tmp_path):
    """Untrimmed take whose t starts at 2 s: t starts at 0, t_source keeps 2 s."""
    from ..tools import reconstruct_take as tool
    gt = rig()
    X = walker(150)
    kp, vis, *_ = observe(gt, X, noise=1.0, ghost_rate=0.0)
    _write_take(tmp_path / "take1", kp, vis, 2.0 + np.arange(150) / FPS)
    ids = [f"cam{c}" for c in "ABCD"]
    sc.save_rig(str(tmp_path / "rig.json"), gt, ids, (W, H))
    rep = tool.run(str(tmp_path / "take1"), str(tmp_path / "out"), str(tmp_path / "rig.json"),
                   render=False, apply_sync=False, log=lambda *a: None)
    z = np.load(tmp_path / "out" / "take1_joints3d.npz")
    assert z["t"][0] == 0.0 and z["t_source"][0] == pytest.approx(2.0)
    assert np.allclose(z["t"], z["t_source"] - 2.0) and rep["trim"]["t_rebased"]


def test_render_3d_is_skipped_with_a_message_without_matplotlib(monkeypatch):
    import sys
    from ..tools import reconstruct_take as rt
    monkeypatch.setitem(sys.modules, "matplotlib", None)       # import -> ImportError
    msgs = []
    out = rt.render_3d_if_possible(None, None, None, None, 30.0, "/nonexistent", log=msgs.append)
    assert out == [] and "matplotlib not installed" in msgs[0]
