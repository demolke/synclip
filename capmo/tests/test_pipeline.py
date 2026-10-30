"""
The shared offline pipeline (:mod:`capmo.core.pipeline`) end to end on a
fabricated capmo session -- the successor of the old ``process_session`` tests
(test_process.py): a known moving skeleton is "filmed" by calibrated cameras
that start at different times, run on different clocks and drop frames; the
real recorder writes the session, the real extractor decodes it (an injected
pose estimator projects the true skeleton at each frame's true capture time),
and the real reconstruction must recover the skeleton.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("scipy")
cv2 = pytest.importorskip("cv2")

from .. import multiview as mv, selfcal as sc  # noqa: E402
from ..core import pipeline as pl  # noqa: E402
from ..core.pose import NUM_LANDMARKS, PoseResult  # noqa: E402
from ..core.session_io import load_session  # noqa: E402
from .session_fixtures import CamSpec, code_of, fabricate_session  # noqa: E402
from .test_selfcal import walker  # noqa: E402

SPECS = [CamSpec("10.0.0.21", offset_ms=1200.0, start_ms=0.0),
         CamSpec("10.0.0.22", offset_ms=-2500.0, start_ms=14.0, drop=(23,)),
         CamSpec("10.0.0.23", offset_ms=55.0, start_ms=29.0, config="midstream")]
N, FPS, GOP = 60, 30, 10
TRACK = walker(T=400, seed=2)                 # (400,33,3) Z-up, sampled at 30 fps
SLOW = 0.2                                    # play it at 1/5 speed (smooth in 33 ms)


def truth_at(sec: float) -> np.ndarray:
    """The walker at a continuous time (s since the fabrication base)."""
    x = np.clip(SLOW * sec * FPS + 40.0, 0, len(TRACK) - 1.001)
    i = int(x)
    a = x - i
    return (1 - a) * TRACK[i] + a * TRACK[i + 1]


def cameras():
    """Three of test_selfcal's ceiling cameras, scaled to the 160x96 frames."""
    from .test_selfcal import rig
    cams = rig()[:3]
    S = np.diag([160 / 1280, 160 / 1280, 1.0])
    for c in cams:
        c.K = S @ c.K
        c.K[1, 2] = 48.0
    return cams


@pytest.fixture(scope="module")
def fab(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    return fabricate_session(str(root), SPECS, n=N, fps=FPS, gop=GOP, seed=5)


class ProjectingEstimator:
    """Pose stub: decodes the frame number from the grey code block, looks up
    that frame's true capture time and projects the true skeleton with the
    current camera (set from the pipeline's progress callback)."""

    def __init__(self, fab, cams_by_id, phone_of, drop=()):
        self.fab, self.cams, self.phone_of = fab, cams_by_id, phone_of
        self.cam = None
        self.drop = set(drop)                 # (cam_id, k) -> no person
        self.calls = 0

    def progress(self, stage, detail, fraction):
        if stage == "extract":
            self.cam = detail

    def estimate(self, img):
        self.calls += 1
        k = int(round((code_of(img) - 30.0) / 3.0))
        if (self.cam, k) in self.drop:
            return None
        ph = self.phone_of[self.cam]
        host = self.fab.truth[ph][k]
        X = truth_at((host - self.fab.base_ms) / 1000.0)
        pts = self.cams[self.cam].project(X)
        return PoseResult(pts, np.full(NUM_LANDMARKS, 0.95), np.zeros((NUM_LANDMARKS, 3)))


def _setup(fab, tmp_path, cam_subset=None, drop=()):
    sess = load_session(fab.dir)
    ids = sess.cam_ids
    phone_of = {c.cam_id: c.phone_id for c in sess.cameras}
    cams = dict(zip(ids, cameras()))
    keep = cam_subset or ids
    rig = str(tmp_path / "rig.json")
    sc.save_rig(rig, [cams[c] for c in keep], keep, (160, 96))
    return sess, cams, ProjectingEstimator(fab, cams, phone_of, drop), rig


def _raw(res):
    d = np.load(res.joints3d_path.replace("_joints3d.npz", "_joints3d_raw.npz"))
    return d["joints"].astype(np.float64), d["valid"], d["t"]


def _truth_on_grid(fab, sess, t):
    t0 = sess.t0_ms
    return np.stack([truth_at((t0 + 1000.0 * ti - fab.base_ms) / 1000.0) for ti in t])


def _rigid_rms(A, B):
    """RMS after the best rigid motion of A onto B (rows = points)."""
    cA, cB = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - cA).T @ (B - cB))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return float(np.sqrt(np.mean(np.sum(((A - cA) @ R.T + cB - B) ** 2, 1))))


def test_session_to_joints_with_rig(fab, tmp_path):
    sess, cams, est, rig = _setup(fab, tmp_path)
    steps = []

    def progress(stage, detail, frac):
        steps.append(stage)
        est.progress(stage, detail, frac)

    res = pl.process_session(fab.dir, str(tmp_path / "work"), rig=rig, estimator=est,
                             progress=progress)
    assert res.rig_source == "given" and res.rig_path == rig and not res.warnings
    assert os.path.exists(res.joints3d_path)
    assert steps[0] == "extract" and steps[-1] == "done" and "reconstruct" in steps
    assert est.calls > 2 * N
    j = res.joints3d()
    assert j.joints.shape[1:] == (33, 3) and j.fps == FPS and np.isfinite(j.joints).all()
    assert res.report["timeline"]["mode"] == "resampled"
    # the raw triangulation is the true skeleton (rig kept metric, world re-derived)
    X, valid, t = _raw(res)
    gt = _truth_on_grid(fab, sess, t)
    rows = valid.all(1)
    assert rows.mean() > 0.8
    rows &= t > 0.1          # the fixture's grey code is unreliable on a stream's first frame
    err = _rigid_rms(X[rows].reshape(-1, 3), gt[rows].reshape(-1, 3))
    assert err < 0.01, f"raw joints off by {err * 100:.2f} cm RMS"
    # the Y-up adapter for the character keeps the geometry
    jy = j.to_y_up()
    assert np.allclose(np.linalg.norm(jy.joints[:, 11] - jy.joints[:, 12], axis=-1),
                       np.linalg.norm(j.joints[:, 11] - j.joints[:, 12], axis=-1))
    assert np.median(jy.joints[:, 0, 1] - jy.joints[:, 27, 1]) > 1.2      # head above ankle


def test_camera_missing_from_the_rig_is_left_out(fab, tmp_path):
    sess, cams, est, rig = _setup(fab, tmp_path, cam_subset=None)
    ids = sess.cam_ids
    sc.save_rig(rig, [cams[c] for c in ids[:2]], ids[:2], (160, 96))
    res = pl.process_session(fab.dir, str(tmp_path / "work"), rig=rig, estimator=est,
                             progress=est.progress)
    assert res.report["cams"] == ids[:2]
    assert any(ids[2] in w for w in res.warnings)
    assert not os.path.exists(os.path.join(res.take_dir, f"{ids[2]}.npz"))
    X, valid, t = _raw(res)
    assert valid.all(1).mean() > 0.8


def test_missing_detection_drops_only_that_view(fab, tmp_path):
    sess = load_session(fab.dir)
    c = sess.cam_ids[2]
    first = fab.first_k[sess.camera(c).phone_id] + sess.camera(c).first_decodable
    drop = {(c, k) for k in range(first + 15, first + 35)}           # 20 frames, no person
    sess, cams, est, rig = _setup(fab, tmp_path, drop=drop)
    res = pl.process_session(fab.dir, str(tmp_path / "work"), rig=rig, estimator=est,
                             progress=est.progress, reconstruct_kw={"apply_sync": False})
    X, valid, t = _raw(res)
    d = np.load(os.path.join(res.take_dir, f"{c}.npz"))
    assert (~d["detected"]).sum() == 20
    gone = (t >= d["t"][~d["detected"]].min() + 0.05) & (t <= d["t"][~d["detected"]].max() - 0.05)
    assert gone.sum() >= 12
    assert valid[gone].all()                    # the other two views still triangulate
    gt = _truth_on_grid(fab, sess, t)
    assert _rigid_rms(X[gone].reshape(-1, 3), gt[gone].reshape(-1, 3)) < 0.01


class _Nobody:
    def estimate(self, img):
        return None


def test_errors_and_cancel(fab, tmp_path):
    with pytest.raises(pl.PipelineError, match="not a capmo session"):
        pl.process_session(str(tmp_path))
    with pytest.raises(pl.PipelineError, match="no person detected in at least two cameras"):
        pl.process_session(fab.dir, str(tmp_path / "w1"), estimator=_Nobody())
    sess, cams, est, rig = _setup(fab, tmp_path)

    def cancel(stage, detail, frac):
        raise pl.Cancelled("stop")

    with pytest.raises(pl.Cancelled):
        pl.process_session(fab.dir, str(tmp_path / "w2"), rig=rig, estimator=est, progress=cancel)
    assert est.calls == 0


def test_session_board_offsets_from_the_common_clock(tmp_path):
    from ..core.sync.timeline import frame_index_offsets
    k = np.arange(50)
    tb = (k + 8) / 30.0
    tb[40:] += 1 / 30.0                       # camera b dropped a frame
    fi = {"a": k + 3, "b": k}                 # each camera counts its own frames
    tt = {"a": k / 30.0, "b": tb}
    off = frame_index_offsets(fi, tt, 30.0)
    assert off["a"] == 0 and off["b"] == pytest.approx(-11.0)
    take = tmp_path / "take"
    take.mkdir()
    entries = []
    for c in ("a", "b"):
        np.savez(take / f"{c}.calib.npz", frame_idx=fi[c], t=tt[c], fps=30.0)
        entries.append({"cam_id": c, "npz": f"{c}.calib.npz"})
    (take / "meta.json").write_text(json.dumps({"calib_videos": entries}))
    assert pl.session_board_offsets(str(take)) == pytest.approx({"a": 0.0, "b": -11.0})


def test_triangulate_poses_with_calibrated_cameras():
    """The in-memory variant: per-camera PoseResult lists -> 3D (T,33,3)."""
    cams = dict(zip(["a", "b", "c"], cameras()))
    X = TRACK[100:103]
    poses = {c: [PoseResult(cam.project(x), np.ones(33)) for x in X] for c, cam in cams.items()}
    poses["c"][1] = None
    res = pl.triangulate_poses(cams, poses)
    assert res.cam_ids == ["a", "b", "c"] and res.X.shape == (3, 33, 3)
    assert res.valid.all() and np.abs(res.X - X).max() < 1e-6
    assert (res.n_views[1] == 2).all() and (res.n_views[0] == 3).all()
    with pytest.raises(pl.PipelineError):
        pl.triangulate_poses(cams, {"a": poses["a"]})
    assert isinstance(cams["a"], mv.Camera)


def test_board_rig_from_a_session_style_take(tmp_path):
    """The pipeline's board step on session-style detections: each camera counts
    its own frames (cameras started at different times), ``t`` is the common
    clock -> the frame-index offsets come from ``t`` and the rig is exact."""
    from . import test_calibrate_board as tcb
    gt = tcb.make_rig()
    lags = [0, 7, -4, 12]                              # camera c's frame k+lag = ref frame k
    ids = tcb.write_detections(str(tmp_path), gt, range(60), lags=lags)
    entries = []
    for c, lag in zip(ids, lags):
        p = tmp_path / f"{c}.calib.npz"
        d = dict(np.load(p))
        d["t"] = (d["frame_idx"] - lag) / tcb.FPS       # host clock: same instant, same t
        np.savez_compressed(p, **d)
        entries.append({"cam_id": c, "npz": f"{c}.calib.npz"})
    (tmp_path / "meta.json").write_text(json.dumps({"calib_videos": entries,
                                                    "timebase": "session"}))
    off = pl.session_board_offsets(str(tmp_path))
    assert off == pytest.approx(dict(zip(ids, map(float, lags))))
    # calibrate_board applies them itself to a session take (no --offsets given)
    rep = pl.calibrate_board(str(tmp_path), str(tmp_path / "rig.json"), log=lambda *a: None)
    assert rep["ok"] and rep["ba"]["rms_px"] < 0.5 and rep["offsets_source"] == "session clock"
    assert rep["offsets_frames"] == pytest.approx(dict(zip(ids, map(float, lags))))
    rot, tr = tcb.rel_errors(tcb.load_rig(str(tmp_path / "rig.json"), ids), gt, ids)
    assert rot < 1.0 and tr < 0.02


# ----------------------------------------------------------- review regressions
def test_rerun_with_fewer_cameras_drops_the_stale_one(fab, tmp_path):
    sess, cams, est, rig = _setup(fab, tmp_path)
    ids = sess.cam_ids
    work = str(tmp_path / "work")
    pl.process_session(fab.dir, work, rig=rig, estimator=est, progress=est.progress,
                       reconstruct_kw={"apply_sync": False})
    res = pl.process_session(fab.dir, work, rig=rig, estimator=est, progress=est.progress,
                             cam_ids=ids[:2], reconstruct_kw={"apply_sync": False})
    assert res.report["cams"] == ids[:2]
    meta = json.load(open(os.path.join(res.take_dir, "meta.json")))
    assert [e["cam_id"] for e in meta["videos"]] == ids[:2] and meta["timebase"] == "session"
    assert not os.path.exists(os.path.join(res.take_dir, f"{ids[2]}.npz"))


def _board_session(fab, tmp_path):
    import shutil
    d = tmp_path / "sess"
    shutil.copytree(fab.dir, d)
    man = json.load(open(d / "manifest.json"))
    man["board"] = {"squares_x": 8, "squares_y": 5, "square_len": 0.04, "marker_len": 0.02,
                    "dict_name": "DICT_6X6_250", "legacy": False}
    json.dump(man, open(d / "manifest.json", "w"))
    return str(d)


def test_board_stage_errors(fab, tmp_path, monkeypatch):
    """Only the expected board failures fall back to self-calibration; programming
    errors propagate; the per-camera fallback (fallback_selfcal) is enabled."""
    from ..tools.calibrate_board import BoardCalibError
    sdir = _board_session(fab, tmp_path)
    sess, cams, est, _rig = _setup(fab, tmp_path)
    real_extract = pl.extract_keypoints
    monkeypatch.setattr(pl, "extract_keypoints", lambda *a, calib=False, **k: (
        {} if calib else real_extract(*a, calib=calib, **k)))
    monkeypatch.setattr(pl, "reconstruct_take", lambda take_dir, out_dir, rig=None, **kw:
                        {"rig": rig})
    seen = {}

    def board(exc):
        def f(take_dir, out, **kw):
            seen.update(kw)
            if exc is not None:
                raise exc
            return {"ok": True, "fallback_selfcal": {"cameras": ["x"]}}
        return f

    monkeypatch.setattr(pl, "calibrate_board", board(AttributeError("bug")))
    with pytest.raises(AttributeError, match="bug"):
        pl.process_session(sdir, str(tmp_path / "w1"), estimator=est, progress=est.progress)
    assert seen["fallback_selfcal"] is True and seen["pose_take"].endswith(sess.name)
    monkeypatch.setattr(pl, "calibrate_board", board(BoardCalibError("too few views")))
    res = pl.process_session(sdir, str(tmp_path / "w2"), estimator=est, progress=est.progress)
    assert res.rig_source == "selfcal" and res.rig_path is None
    assert any("too few views" in w for w in res.warnings)
    monkeypatch.setattr(pl, "calibrate_board", board(None))
    res = pl.process_session(sdir, str(tmp_path / "w3"), estimator=est, progress=est.progress)
    assert res.rig_source == "board" and any("x" in w for w in res.warnings)


def test_board_rig_session_take_with_a_camera_subset(tmp_path):
    """--cam-ids a,b on a session take: the clock offsets of the other cameras
    must not be rejected as unknown cameras."""
    from . import test_calibrate_board as tcb
    gt = tcb.make_rig()
    lags = [0, 7, -4, 12]
    ids = tcb.write_detections(str(tmp_path), gt, range(60), lags=lags)
    entries = []
    for c, lag in zip(ids, lags):
        p = tmp_path / f"{c}.calib.npz"
        d = dict(np.load(p))
        d["t"] = (d["frame_idx"] - lag) / tcb.FPS
        np.savez_compressed(p, **d)
        entries.append({"cam_id": c, "npz": f"{c}.calib.npz"})
    (tmp_path / "meta.json").write_text(json.dumps({"calib_videos": entries,
                                                    "timebase": "session"}))
    sub = ids[1:3]
    rep = pl.calibrate_board(str(tmp_path), str(tmp_path / "rig.json"), cam_ids=sub,
                             log=lambda *a: None)
    assert rep["ok"] and rep["cam_ids"] == sub
    assert rep["offsets_frames"] == pytest.approx({sub[0]: 0.0, sub[1]: lags[2] - lags[1]})
