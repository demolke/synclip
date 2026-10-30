"""Synthetic end-to-end: capmo session -> session_to_take -> extract_keypoints
-> reconstruct_take's loader with per-camera timelines."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")

from ..core.session_io import load_session
from ..tools import session_to_take as st
from .session_fixtures import CamSpec, code_for, fabricate_session

SPECS = [CamSpec("10.0.0.21", offset_ms=310.0, start_ms=0.0, config="separate"),
         CamSpec("10.0.0.22", offset_ms=-2500.0, start_ms=14.0, config="inband", drop=(23,)),
         CamSpec("10.0.0.23", offset_ms=55.0, start_ms=29.0, config="midstream")]
N, FPS, GOP = 40, 30, 10


@pytest.fixture(scope="module")
def fab(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    return fabricate_session(str(root), SPECS, n=N, fps=FPS, gop=GOP, seed=3)


@pytest.mark.parametrize("fmt", ["mp4", "mkv"])
def test_session_to_take_pts_are_host_times(fab, tmp_path, fmt):
    import av
    out = tmp_path / "videos"
    assert st.main([fab.dir, "--out", str(out), "--format", fmt,
                    "--cam-ids", "camA,camB,camC"]) == 0
    meta = json.load(open(out / "take.json"))
    sess = load_session(fab.dir, "camA,camB,camC")
    t0 = sess.t0_ms
    assert meta["t0_ms"] == pytest.approx(t0)
    tol = 0.6 if fmt == "mkv" else 0.02                  # mkv stores whole ms
    for v in meta["videos"]:
        cam = sess.camera(v["cam_id"])
        exp = (cam.decodable_t_ms - t0) / 1000.0
        with av.open(str(out / v["file"])) as c:
            got = np.array([f.time for f in c.decode(video=0)])
        assert len(got) == len(exp) == v["frames"]
        assert np.abs(got - exp).max() * 1000 < tol
        if v.get("ffprobe_max_err_ms") is not None:
            assert v["ffprobe_max_err_ms"] < tol
        # and those are the true capture times (common host clock) to ~ms
        phone = {s.phone_id: s for s in SPECS}[cam.phone_id]
        truth = fab.truth[phone.phone_id][fab.first_k[phone.phone_id] + cam.first_decodable:]
        assert np.abs(got - (truth - t0) / 1000.0).max() * 1000 < 4.0 + tol
    assert {v["cam_id"]: v["leading_undecodable_frames"] for v in meta["videos"]}["camC"] == 7
    assert meta["sync"]["pair_uncertainty_ms"] < 10
    assert set(meta["sync"]["clock_offsets_ms"]) == {"camA", "camB", "camC"}


def test_session_to_take_bad_dir(tmp_path):
    assert st.main([str(tmp_path), "--out", str(tmp_path / "o")]) == 1


class CodeEstimator:
    """Stub pose: every landmark at (code, 40) where code = the frame's grey code."""

    def estimate(self, img):
        from ..pose import NUM_LANDMARKS, PoseResult
        from .session_fixtures import code_of
        c = code_of(img)
        pts = np.tile([c, 40.0], (NUM_LANDMARKS, 1))
        return PoseResult(pts, np.full(NUM_LANDMARKS, 0.9), np.zeros((NUM_LANDMARKS, 3)))


def _extract(fab, out, *extra):
    from ..tools import extract_keypoints as ek
    return ek.main(["--out", str(out), "--take", "take1", "--quiet", "--max-height", "0",
                    *extra, fab.dir], estimator=CodeEstimator())


def test_extract_keypoints_on_session_dir(fab, tmp_path):
    out = tmp_path / "kp"
    assert _extract(fab, out, "--cam-ids", "camA,camB,camC") == 0
    meta = json.load(open(out / "take1" / "meta.json"))
    assert meta["timebase"] == "session" and meta["session"]["t0_ms"] > 0
    sess = load_session(fab.dir, "camA,camB,camC")
    for v in meta["videos"]:
        cam = sess.camera(v["cam_id"])
        d = np.load(out / "take1" / v["npz"])
        assert (int(d["width"]), int(d["height"])) == (160, 96)
        assert np.allclose(d["t"], sess.frame_times_s(cam))
        assert list(d["frame_idx"]) == list(range(len(d["t"])))
        assert float(d["session_t0_ms"]) == pytest.approx(sess.t0_ms)
        k0 = fab.first_k[cam.phone_id] + cam.first_decodable
        exp = np.array([code_for(k0 + i) for i in range(len(d["t"]))])
        diff = d["kp"][:, 0, 0] - exp
        assert np.abs(diff - np.median(diff)).max() < 1.5   # frame <-> time mapping exact
        assert v["time_method"] == "meta_anchors" and v["clock_offset_ms"] is not None
    # the same numbers as going through session_to_take + plain video files
    vids = tmp_path / "videos"
    st.convert(fab.dir, str(vids), "camA,camB,camC", log=lambda *a: None)
    from ..tools import extract_keypoints as ek
    files = [str(vids / f"{c}.mp4") for c in ("camA", "camB", "camC")]
    assert ek.main(["--out", str(tmp_path / "kp2"), "--take", "t", "--quiet", *files],
                   estimator=CodeEstimator()) == 0
    for c in ("camA", "camB", "camC"):
        a = np.load(out / "take1" / f"{c}.npz")
        b = np.load(tmp_path / "kp2" / "t" / f"{c}.npz")
        assert np.abs(a["t"] - b["t"]).max() < 2e-5
        assert np.allclose(a["kp"], b["kp"])


def test_extract_session_window_and_calib(fab, tmp_path):
    out = tmp_path / "kp"
    assert _extract(fab, out, "--start", "0.5", "--end", "1.0") == 0
    sess = load_session(fab.dir)
    for c in sess.cameras:
        d = np.load(out / "take1" / f"{c.cam_id}.npz")
        ts = sess.frame_times_s(c)
        inwin = ts[(ts >= 0.5 - 0.25 / 30) & (ts <= 1.0 + 0.25 / 30)]
        assert np.allclose(d["t"], inwin)
        assert d["frame_idx"][0] == int(np.searchsorted(ts, 0.5 - 0.25 / 30))
    # calibration mode on the session (no board in these frames -> no detections)
    from ..tools import extract_keypoints as ek
    assert ek.main(["--out", str(out), "--take", "take1", "--quiet", "--calib", "--start", "0",
                    "--end", "0.6", "--calib-every", "1", fab.dir]) == 0
    meta = json.load(open(out / "take1" / "meta.json"))
    assert {e["cam_id"] for e in meta["calib_videos"]} == set(sess.cam_ids)
    assert all(e["frames_tried"] > 0 for e in meta["calib_videos"])
    assert ek.open_video(sess.cameras[0].h264_path).__class__.__name__ == "SessionVideoSource"


def test_extract_rejects_mixed_inputs(fab, tmp_path):
    from ..tools import extract_keypoints as ek
    assert ek.main(["--out", str(tmp_path), "--quiet", fab.dir, fab.dir],
                   estimator=CodeEstimator()) == 2
    assert ek.main(["--out", str(tmp_path), "--quiet", str(tmp_path)],
                   estimator=CodeEstimator()) == 2


# ------------------------------------------------- per-camera timelines (step 5)
def test_load_take_resamples_session_take(fab, tmp_path):
    from .. import selfcal as sc
    out = tmp_path / "kp"
    assert _extract(fab, out) == 0
    take = sc.load_take(str(out / "take1"))
    assert take.timeline == "resampled" and take.fps == 30
    assert np.allclose(take.sync_prior, 0) and take.session["t0_ms"] > 0
    assert np.allclose(np.diff(take.t), 1 / 30)
    sess = load_session(fab.dir)
    t0 = sess.t0_ms
    for v, cid in enumerate(take.cam_ids):
        cam = sess.camera(cid)
        k0 = fab.first_k[cam.phone_id] + cam.first_decodable
        true_t = (fab.truth[cam.phone_id][k0:] - t0) / 1000.0     # true capture times
        ks = np.arange(k0, k0 + len(true_t))
        ok = take.vis[v, :, 0] > 0
        assert ok.sum() >= len(true_t) - 3
        # where the camera is resampled, the code is the one of the true frame
        # time (linear in k): this checks the cross-camera time alignment
        exp = np.interp(take.t[ok], true_t, 30 + 3.0 * ks)
        d = take.kp[v, ok, 0, 0] - exp
        assert np.abs(d - np.median(d)).max() < 1.5, cid       # (codec noise ~1 level)
        assert np.median(np.abs(d - np.median(d))) < 0.2       # 0.2 level = 2 ms
        assert not ok[take.t < true_t[0] - 0.5 / 30].any()
    legacy = sc.load_take(str(out / "take1"), timeline="first")
    assert legacy.timeline == "shared"
    starts = np.array([t[0] for t in legacy.cam_t])
    assert np.allclose(legacy.sync_prior, (starts[0] - starts) * 30)


def test_reconstruct_run_uses_timeline_and_prior(fab, tmp_path, monkeypatch):
    """reconstruct_take.run loads per-camera timelines and passes the prior."""
    from .. import selfcal as sc
    from ..tools import reconstruct_take as rt
    out = tmp_path / "kp"
    assert _extract(fab, out) == 0
    seen = {}

    def fake_offsets(take, prior=None, **kw):
        seen["timeline"], seen["prior"] = take.timeline, prior
        raise RuntimeError("stop here")
    monkeypatch.setattr(sc, "estimate_offsets", fake_offsets)
    with pytest.raises(RuntimeError, match="stop here"):
        rt.run(str(out / "take1"), str(tmp_path / "rec"), render=False, log=lambda *a: None)
    assert seen["timeline"] == "resampled" and np.allclose(seen["prior"], 0)


def test_resample_track_vfr_and_gaps():
    from ..selfcal import resample_track
    t = np.array([0.0, 0.034, 0.066, 0.140, 0.166])        # drop between .066 and .140
    kp = np.zeros((5, 33, 2))
    kp[..., 0] = (t * 1000)[:, None]                         # x = time in ms
    vis = np.ones((5, 33))
    det = np.array([True, True, True, True, False])
    grid = np.arange(-1, 7) / 30.0
    k, v, fi = resample_track(t, kp, vis, det, np.arange(5), grid, 30.0)
    x = k[:, 0, 0]
    assert np.isnan(x[0]) and v[0, 0] == 0                  # before the span
    assert x[1] == 0.0 and fi[1] == 0
    assert np.allclose(x[2:4], grid[2:4] * 1000)            # linear interpolation
    assert np.allclose(x[4], grid[4] * 1000)                # across a 2.2-frame gap: allowed
    assert np.isnan(x[6]) and fi[6] == 4                    # nearest is undetected
    assert np.isnan(x[7])                                    # after the span


def test_estimate_offsets_prior_for_unresolved():
    from ..selfcal import Take, estimate_offsets
    V, T = 3, 10
    take = Take(["a", "b", "c"], np.full((V, T, 33, 2), np.nan), np.zeros((V, T, 33)),
                np.arange(T) / 30.0, 30.0, (100, 100))
    res = estimate_offsets(take, prior=np.array([1.0, 3.5, -2.0]))
    assert res["unresolved"] == [1, 2]
    assert np.allclose(res["offsets"], [0.0, 2.5, -3.0])
    assert np.allclose(estimate_offsets(take)["offsets"], 0)


def test_session_keyframes_in_window(fab, tmp_path):
    """--keyframes with --start on a session: thumbnails are frames of the window
    (the session source's frame index base, not round(start * fps))."""
    out = tmp_path / "kp"
    assert _extract(fab, out, "--start", "0.5", "--end", "1.0", "--keyframes", "3") == 0
    meta = json.load(open(out / "take1" / "meta.json"))
    for v in meta["videos"]:
        d = np.load(out / "take1" / v["npz"])
        kf = sorted(int(f[-10:-4]) for f in v.get("keyframes", []))
        assert len(kf) == 3 and set(kf) <= set(d["frame_idx"].tolist()), v["cam_id"]


def test_load_take_first_camera_without_frames(tmp_path):
    """A camera with no rows in the window must not break the resampled grid."""
    from .. import selfcal as sc

    def npz(c, n, t0):
        np.savez(tmp_path / f"{c}.npz", kp=np.zeros((n, 33, 2), np.float32),
                 vis=np.ones((n, 33), np.float32), t=t0 + np.arange(n) / 30, fps=30.0,
                 width=640, height=480, detected=np.ones(n, bool), frame_idx=np.arange(n))
    npz("a", 0, 0.0)
    npz("b", 30, 0.5)
    npz("c", 30, 0.6)
    json.dump({"videos": [{"cam_id": c} for c in "abc"], "timebase": "session"},
              open(tmp_path / "meta.json", "w"))
    take = sc.load_take(str(tmp_path))
    assert take.timeline == "resampled" and take.t[0] == pytest.approx(0.5)
    assert take.vis[0].sum() == 0 and take.vis[1].sum() > 0
    npz("b", 0, 0.0)
    npz("c", 0, 0.0)
    with pytest.raises(ValueError, match="no camera has any frame"):
        sc.load_take(str(tmp_path))
