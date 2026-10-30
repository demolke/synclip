"""
ChArUco multi-camera calibration (``capmo.tools.calibrate_board``) on synthetic data.

Virtual cameras with known K / distortion / poses watch a ChArUco board that is
moved and tilted in front of them. Either the board is *rendered* into short
videos (board in the first seconds, then empty frames as the "performance")
and run through ``extract_keypoints --calib --start/--end`` like real footage,
or (faster, for the graph / sync / fallback cases) the board corners are
projected directly into ``.calib.npz`` files.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "aruco"):
    pytest.skip("cv2.aruco missing", allow_module_level=True)

from .. import multiview as mv, selfcal as sc, synth  # noqa: E402
from ..calibration import BoardConfig  # noqa: E402
from ..tools import calibrate_board as cb  # noqa: E402

W, H, F, FPS = 640, 480, 600.0, 10.0
DIST = np.array([-0.12, 0.03, 0.0008, -0.0005, 0.0])
BOARD = BoardConfig(squares_x=8, squares_y=5, square_len=0.06, marker_len=0.045)
CENTER = np.array([0.0, 0.0, 1.2])            # where the board is waved (world, Z up)


def make_rig(angles=(-32, -11, 11, 32), dist_m=1.1, seed=0):
    """Cameras on an arc in front of (-Y side of) the board area, looking at it."""
    rng = np.random.default_rng(seed)
    cams = []
    for i, a in enumerate(angles):
        a = np.radians(a)
        eye = CENTER + dist_m * np.array([np.sin(a), -np.cos(a), 0.0]) \
            + np.array([0, 0, 0.25 * (i % 2) - 0.1]) + rng.normal(0, 0.02, 3)
        R, t = synth.look_at(eye, CENTER + rng.normal(0, 0.03, 3), up=(0, 0, 1))
        fx = F * (1 + 0.03 * (i - 1.5) / 1.5)
        K = np.array([[fx, 0, W / 2 + 6 * (i - 1.5)], [0, fx, H / 2 - 4 * (i - 1.5)], [0, 0, 1]])
        cams.append(mv.Camera(K, DIST * (1 + 0.1 * i), R, t))
    return cams


def board_pose(k, n=40):
    """Board -> world (R, t) at (possibly fractional / negative) frame k."""
    u = 2 * np.pi * k / n
    # facing the cameras (board +Z = world +Y), x along world X, y down
    R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])
    ax, az, ay = 0.5 * np.sin(1.3 * u), 0.45 * np.sin(u + 0.7), 0.35 * np.sin(2.1 * u + 1.0)
    Rx = cv2.Rodrigues(np.array([ax, 0, 0.0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, az]))[0]
    Ry = cv2.Rodrigues(np.array([0, ay, 0.0]))[0]
    R = Rz @ Rx @ Ry @ R0
    c = CENTER + np.array([0.28 * np.sin(1.7 * u), 0.15 * np.sin(0.9 * u + 2.0),
                           0.18 * np.sin(2.3 * u + 0.4)])
    bc = np.array([BOARD.squares_x * BOARD.square_len / 2, BOARD.squares_y * BOARD.square_len / 2, 0])
    return R, c - R @ bc


# -------------------------------------------------------- direct projection
def write_detections(out_dir, cams, frames_per_cam, *, noise=0.15, lags=None, seed=0,
                     drop=None, ids=None, board=BOARD, size=(W, H), pose=None, fps=FPS):
    """Project board corners into ``<cam>.calib.npz`` (extract_keypoints format).

    *frames_per_cam*: board frames (reference timeline) shown; camera c's frame
    index is ``k + lags[c]``. *drop(c, k)* -> True removes that detection.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    pose = pose or board_pose
    w, h = size
    corners3d = np.asarray(board.board().getChessboardCorners(), float)
    ids = ids or [f"cam{chr(65 + i)}" for i in range(len(cams))]
    lags = lags or [0] * len(cams)
    for c, (cid, cam) in enumerate(zip(ids, cams)):
        t_l, f_l, nm, corners, idl, offs = [], [], [], [], [], [0]
        for k in frames_per_cam:
            if drop is not None and drop(c, k):
                continue
            Rb, tb = pose(k)
            R = cam.R @ Rb
            t = cam.R @ tb + cam.t
            n_cam = R[:, 2]                       # board normal in camera coords
            Xc = corners3d @ R.T + t
            cosang = (Xc @ n_cam) / np.linalg.norm(Xc, axis=1)
            if np.median(cosang) < np.cos(np.radians(70)):
                continue                          # too oblique / back side
            uv = cam.project(corners3d @ Rb.T + tb) + rng.normal(0, noise, (len(corners3d), 2))
            ok = (uv[:, 0] > 3) & (uv[:, 0] < w - 3) & (uv[:, 1] > 3) & (uv[:, 1] < h - 3)
            if ok.sum() < 6:
                continue
            fi = k + lags[c]
            t_l.append(fi / fps)
            f_l.append(fi)
            nm.append(int(ok.sum()) // 2)
            corners.append(uv[ok].astype(np.float32))
            idl.append(np.flatnonzero(ok).astype(np.int32))
            offs.append(offs[-1] + int(ok.sum()))
        np.savez_compressed(
            os.path.join(out_dir, f"{cid}.calib.npz"), t=np.array(t_l), frame_idx=np.array(f_l),
            n_markers=np.array(nm, np.int32),
            corners=np.concatenate(corners) if corners else np.zeros((0, 2), np.float32),
            ids=np.concatenate(idl) if idl else np.zeros(0, np.int32), offsets=np.array(offs),
            board_json=json.dumps(board.to_dict()), width=w, height=h, fps=fps)
    return ids


def rel_errors(est: dict, gt, ids, ref=0):
    """Max rotation (deg) / translation (m) error of camera poses relative to *ref*."""
    def rel(Rc, tc, Rr, tr):
        R = Rc @ Rr.T
        return R, tc - R @ tr
    rot, tr = [], []
    e_ref, g_ref = est[ids[ref]], gt[ref]
    for i, cid in enumerate(ids):
        if i == ref:
            continue
        Re, te = rel(est[cid].R, est[cid].t, e_ref.R, e_ref.t)
        Rg, tg = rel(gt[i].R, gt[i].t, g_ref.R, g_ref.t)
        rot.append(float(sc.rotation_angle_deg(Re @ Rg.T)))
        tr.append(float(np.linalg.norm(te - tg)))
    return max(rot), max(tr)


def load_rig(path, ids):
    return dict(zip(ids, sc.from_rig(path, ids)))


# --------------------------------------------------------------- rendering
def _board_texture(px=60, margin=20):
    img = BOARD.board().generateImage(
        (BOARD.squares_x * px + 2 * margin, BOARD.squares_y * px + 2 * margin), marginSize=margin)
    return cv2.GaussianBlur(img, (0, 0), 0.8), px, margin


def render_videos(out_dir, cams, n_board=40, n_after=12, lags=None, seed=0):
    """Write ``camX.avi`` per camera: the board for *n_board* frames, then empty."""
    rng = np.random.default_rng(seed)
    tex, px, margin = _board_texture()
    lags = lags or [0] * len(cams)
    paths = []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pix = np.stack([xx.ravel() + 0.0, yy.ravel()], 1).reshape(-1, 1, 2)
    for c, cam in enumerate(cams):
        und = cam.normalize(pix.reshape(-1, 2))          # exact inverse of the distortion
        rays = np.concatenate([und, np.ones((H * W, 1))], 1)
        p = os.path.join(out_dir, f"cam{chr(65 + c)}.avi")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        if not vw.isOpened():
            pytest.skip("cv2.VideoWriter cannot write MJPG here")
        bg = (110 + rng.normal(0, 6, (H, W))).astype(np.float32)
        for fi in range(n_board + n_after + lags[c]):
            k = fi - lags[c]
            img = bg.copy()
            if k < n_board:
                Rb, tb = board_pose(k)
                R = cam.R @ Rb
                t = cam.R @ tb + cam.t
                Hn = np.column_stack([R[:, 0], R[:, 1], t])
                q = rays @ np.linalg.inv(Hn).T
                ok = q[:, 2] > 1e-9
                xb = np.where(ok, q[:, 0] / np.where(ok, q[:, 2], 1), -1e3)
                yb = np.where(ok, q[:, 1] / np.where(ok, q[:, 2], 1), -1e3)
                mx = (margin + xb / BOARD.square_len * px - 0.5).reshape(H, W).astype(np.float32)
                my = (margin + yb / BOARD.square_len * px - 0.5).reshape(H, W).astype(np.float32)
                warped = cv2.remap(tex.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
                inside = warped >= 0
                img[inside] = warped[inside]
            img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
            vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        vw.release()
        paths.append(p)
    return paths


IDS = ["camA", "camB", "camC", "camD"]


def _quiet(*_a, **_k):
    pass


# ------------------------------------------------------------------- tests
@pytest.mark.slow
def test_board_segment_videos_to_rig(tmp_path):
    """Rendered videos: board for 4 s then 'performance'; extract window -> rig."""
    pytest.importorskip("av")
    from ..tools import extract_keypoints as ek
    gt = make_rig()
    vids = render_videos(str(tmp_path), gt, n_board=40, n_after=12)
    kp = tmp_path / "kp"
    board_args = ["--squares-x", "8", "--squares-y", "5", "--square-len", "0.06",
                  "--marker-len", "0.045"]
    assert ek.main(["--calib", "--start", "0", "--end", "3.95", "--calib-every", "1",
                    "--min-corners", "6", "--out", str(kp), "--take", "take1", "--quiet",
                    "--cam-ids", ",".join(IDS)] + board_args + vids) == 0
    d = np.load(kp / "take1" / "camA.calib.npz")
    assert d["frame_idx"].max() <= 39 and d["t"].max() <= 3.95 + 1e-6
    assert len(d["t"]) >= 36                          # nearly every board frame detected
    # a window starting mid-video seeks and keeps the absolute frame numbering
    assert ek.main(["--calib", "--start", "2.0", "--end", "2.55", "--calib-every", "1",
                    "--out", str(kp), "--take", "win", "--quiet", "--cam-ids", "camB"]
                   + board_args + [vids[1]]) == 0
    w = np.load(kp / "win" / "camB.calib.npz")
    assert list(w["frame_idx"]) == [20, 21, 22, 23, 24, 25]
    assert np.allclose(w["t"], w["frame_idx"] / FPS, atol=1e-3)

    rig_path, rep_path, ov = tmp_path / "rig.json", tmp_path / "report.json", tmp_path / "ov"
    from ..tools import calibrate_board as tool
    assert tool.main([str(kp / "take1"), "--out", str(rig_path), "--report", str(rep_path),
                      "--overlay-dir", str(ov), "--start", "0", "--end", "4"]) == 0
    rep = json.loads(rep_path.read_text())
    est = load_rig(str(rig_path), IDS)
    for i, cid in enumerate(IDS):
        assert abs(est[cid].K[0, 0] / gt[i].K[0, 0] - 1) < 0.015, cid
        assert rep["cameras"][cid]["intrinsic_rms_px"] < 0.5
        assert rep["cameras"][cid]["ba_rms_px"] < 0.5
    rot, tr = rel_errors(est, gt, IDS)
    assert rot < 1.0 and tr < 0.02, (rot, tr)
    assert rep["ok"] and not rep["errors"] and rep["ba"]["rms_px"] < 0.5
    assert rep["graph"]["not_connected"] == [] and len(rep["graph"]["spanning_tree"]) == 3
    # world frame = board pose at the reference instant: that board lies at z=0
    k = int(rep["world_frame"]["ref_cam_frame_idx"])
    Rb, tb = board_pose(k)
    cam = est["camA"]
    corners = np.asarray(BOARD.board().getChessboardCorners(), float)
    uv_est = cam.project(corners)
    uv_gt = gt[0].project(corners @ Rb.T + tb)
    assert np.median(np.linalg.norm(uv_est - uv_gt, axis=1)) < 2.0
    assert len(list(ov.glob("board_*.jpg"))) >= 4
    # the rig loads into the reconstruction tools
    rig_d = json.loads(rig_path.read_text())
    assert "board_calibration" in rig_d and rig_d["board"]["square_len"] == 0.06


def test_camera_chained_through_others(tmp_path):
    """camD never sees the board together with camA: placed via camB/camC."""
    gt = make_rig()
    write_detections(str(tmp_path), gt, range(60),
                     drop=lambda c, k: (c == 3 and k % 2 == 0) or (c == 0 and k % 2 == 1))
    rep = cb.run(str(tmp_path), str(tmp_path / "rig.json"), log=_quiet)
    edges = rep["graph"]["edges"]
    assert "camA-camD" not in edges and edges["camC-camD"]["usable"]
    tree = rep["graph"]["spanning_tree"]
    assert not any(e.endswith("->camD") and e.startswith("camA") for e in tree)
    assert any(e.endswith("->camD") for e in tree)
    est = load_rig(str(tmp_path / "rig.json"), IDS)
    rot, tr = rel_errors(est, gt, IDS)
    assert rot < 1.0 and tr < 0.02, (rot, tr)
    assert rep["ba"]["rms_px"] < 0.5


@pytest.mark.slow
def test_offsets_from_board_motion_and_known_offsets(tmp_path):
    """camC lags 2 frames, camD leads 1: estimated from the board centre speed."""
    gt = make_rig()
    write_detections(str(tmp_path), gt, range(-5, 80), lags=[0, 0, 2, -1])
    rep = cb.run(str(tmp_path), str(tmp_path / "rig.json"), estimate_offsets=True, log=_quiet)
    assert rep["offsets_frames"] == {"camA": 0.0, "camB": 0.0, "camC": 2.0, "camD": -1.0}
    rot, tr = rel_errors(load_rig(str(tmp_path / "rig.json"), IDS), gt, IDS)
    assert rot < 1.0 and tr < 0.02 and rep["ba"]["rms_px"] < 0.5
    rep2 = cb.run(str(tmp_path), str(tmp_path / "rig2.json"),
                  offsets={"camC": 2, "camD": -1}, log=_quiet)
    assert rep2["ba"]["rms_px"] < 0.5
    # ignoring the offsets is caught by the report
    rep3 = cb.run(str(tmp_path), str(tmp_path / "rig3.json"), log=_quiet)
    assert rep3["ba"]["rms_px"] > 2.0
    assert any("disagree" in w for w in rep3["warnings"])
    assert any("high BA RMS" in w for w in rep3["warnings"])


def test_camera_without_board_is_an_error(tmp_path):
    gt = make_rig()
    write_detections(str(tmp_path), gt, range(40), drop=lambda c, k: c == 2)
    rep_path = tmp_path / "rep.json"
    with pytest.raises(cb.BoardCalibError, match="camC.*never sees the board"):
        cb.run(str(tmp_path), str(tmp_path / "rig.json"), report_path=str(rep_path), log=_quiet)
    rep = json.loads(rep_path.read_text())
    assert not rep["ok"] and "self-calibrat" in rep["errors"][0]
    assert cb.main([str(tmp_path), "--out", str(tmp_path / "rig.json")]) == 1
    assert not (tmp_path / "rig.json").exists()
    # known intrinsics for camC are not enough without board views: still unlinked
    with pytest.raises(cb.BoardCalibError, match="camC"):
        cb.run(str(tmp_path), None, intrinsics_files={"camC": _k_json(tmp_path, gt[2])},
               log=_quiet)


def _k_json(tmp_path, cam):
    p = tmp_path / "K.json"
    p.write_text(json.dumps({"K": cam.K.tolist(), "dist": cam.dist.tolist(),
                             "image_size": [W, H]}))
    return str(p)


def test_known_intrinsics_and_saved_intrinsics(tmp_path):
    gt = make_rig()
    write_detections(str(tmp_path), gt, range(40))
    rep = cb.run(str(tmp_path), str(tmp_path / "rig.json"),
                 intrinsics_files={"camB": _k_json(tmp_path, gt[1])},
                 save_intrinsics=str(tmp_path / "K"), log=_quiet)
    assert rep["cameras"]["camB"]["intrinsics_source"] == "file"
    est = load_rig(str(tmp_path / "rig.json"), IDS)
    assert np.allclose(est["camB"].K, gt[1].K)
    saved = json.loads((tmp_path / "K" / "camA.K.json").read_text())
    assert abs(saved["K"][0][0] / gt[0].K[0, 0] - 1) < 0.015


@pytest.mark.slow
def test_fallback_selfcal_places_a_camera_without_board(tmp_path):
    """Walker rig: camD never sees the board; body self-calibration places it."""
    from . import test_selfcal as ts
    gt = ts.rig()
    big = BoardConfig(squares_x=8, squares_y=5, square_len=0.25, marker_len=0.18)

    def floor_pose(k, n=60):
        u = 2 * np.pi * k / n
        R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # lying face up
        R = cv2.Rodrigues(np.array([0.35 * np.sin(u), 0.35 * np.sin(1.7 * u + 1), 0.4 * u]))[0] @ R0
        c = np.array([0.8 * np.sin(u), 0.8 * np.cos(1.3 * u), 0.9 + 0.2 * np.sin(2 * u)])
        bc = np.array([1.0, 0.625, 0])
        return R, c - R @ bc

    write_detections(str(tmp_path), gt, range(60), board=big, size=(ts.W, ts.H),
                     pose=floor_pose, fps=ts.FPS, drop=lambda c, k: c == 3, noise=0.3)
    X = ts.walker(150)
    kp, vis, *_ = ts.observe(gt, X, noise=1.0, ghost_rate=0.02)
    for v, cid in enumerate(IDS):
        np.savez_compressed(tmp_path / f"{cid}.npz", kp=np.nan_to_num(kp[v]).astype(np.float32),
                            vis=vis[v].astype(np.float32), detected=np.isfinite(kp[v, :, 0, 0]),
                            t=np.arange(150) / ts.FPS, width=ts.W, height=ts.H, fps=ts.FPS)
    with pytest.raises(cb.BoardCalibError, match="camD"):
        cb.run(str(tmp_path), None, log=_quiet)
    # a 2 m board seen from ~4.5 m covers little of the image: the report says so,
    # and the documented remedy (centred principal point, k1 only) is used
    rep = cb.run(str(tmp_path), str(tmp_path / "rig.json"), fallback_selfcal=True,
                 fix_pp=True, dist_model="k1", log=_quiet)
    assert any("poor image coverage" in w for w in rep["warnings"])
    assert rep["fallback_selfcal"]["cameras"] == ["camD"]
    assert rep["cameras"]["camD"]["intrinsics_source"] == "selfcal"
    est = load_rig(str(tmp_path / "rig.json"), IDS)
    rot, tr = rel_errors({k: est[k] for k in IDS[:3]}, gt[:3], IDS[:3])
    assert rot < 0.5 and tr < 0.05                     # board cameras (6-7 m baselines)
    rot, tr = rel_errors(est, gt, IDS)
    assert rot < 2.0 and tr < 0.15, (rot, tr)          # camD: body-calibration accuracy
