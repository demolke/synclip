"""Offline keypoint/ChArUco extractor with a stubbed pose estimator."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("av")

from ..pose import PoseResult, NUM_LANDMARKS
from ..tools import extract_keypoints as ek

W, H, N, FPS = 160, 96, 8, 10


def _make_video(path, w=W, h=H, n=N, fps=FPS):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v here")
    for i in range(n):
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (10 + 10 * i, 40), 6, (255, 255, 255), -1)
        vw.write(f)
    vw.release()


class StubEstimator:
    """Returns a landmark grid at normalized (0.5, 0.25); drops frame 3."""

    def __init__(self):
        self.shapes = []

    def estimate(self, img):
        self.shapes.append(img.shape[:2])
        if len(self.shapes) == 4:
            return None
        pts = np.tile(np.array([0.5, 0.25]), (NUM_LANDMARKS, 1)) * [img.shape[1], img.shape[0]]
        return PoseResult(pts, np.full(NUM_LANDMARKS, 0.9), np.zeros((NUM_LANDMARKS, 3)))


def test_pose_extraction_shapes_and_times(tmp_path):
    vid = tmp_path / "camA.mp4"
    _make_video(vid)
    out = tmp_path / "out"
    est = StubEstimator()
    rc = ek.main(["--out", str(out), "--take", "t1", "--quiet", "--max-height", "48",
                  "--keyframes", "2", "--proxy", str(vid)], estimator=est)
    assert rc == 0
    d = np.load(out / "t1" / "camA.npz")
    assert d["kp"].shape == (N, NUM_LANDMARKS, 2) and d["kp"].dtype == np.float32
    assert d["vis"].shape == (N, NUM_LANDMARKS)
    assert d["world"].shape == (N, NUM_LANDMARKS, 3)
    assert int(d["width"]) == W and int(d["height"]) == H
    assert np.allclose(d["t"], np.arange(N) / FPS, atol=1e-3)
    assert list(d["frame_idx"]) == list(range(N))
    assert est.shapes[0] == (48, 80)                   # downscaled for the model...
    assert np.allclose(d["kp"][0], [[W * 0.5, H * 0.25]], atol=1.0)   # ...pixels at full res
    assert np.isnan(d["kp"][3]).all() and d["vis"][3].sum() == 0 and not d["detected"][3]
    meta = json.load(open(out / "t1" / "meta.json"))
    assert meta["videos"][0]["n_detected"] == N - 1 and meta["videos"][0]["width"] == W
    assert len(list((out / "t1" / "keyframes").glob("*.jpg"))) == 2
    assert (out / "t1" / "camA.proxy.mp4").stat().st_size > 0


def test_every_and_cam_ids(tmp_path):
    vid = tmp_path / "x.mp4"
    _make_video(vid)
    out = tmp_path / "out"
    assert ek.main(["--out", str(out), "--cam-ids", "left", "--every", "3", "--quiet",
                    str(vid)], estimator=StubEstimator()) == 0
    d = np.load(out / "take" / "left.npz")
    assert list(d["frame_idx"]) == [0, 3, 6]


def test_rotation_metadata_makes_upright_frames(tmp_path, monkeypatch):
    vid = tmp_path / "r.mp4"
    _make_video(vid)
    monkeypatch.setattr(ek, "_stream_rotation", lambda s: 90.0)
    out = tmp_path / "out"
    assert ek.main(["--out", str(out), "--quiet", str(vid)], estimator=StubEstimator()) == 0
    d = np.load(out / "take" / "r.npz")
    assert (int(d["width"]), int(d["height"])) == (H, W)


def test_bad_video_reports_failure(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert ek.main(["--out", str(tmp_path / "o"), "--quiet", str(bad)],
                   estimator=StubEstimator()) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_calib_detects_charuco_and_autodetects_layout(tmp_path, legacy):
    from ..calibration import BoardConfig
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco missing")
    cfg = BoardConfig(squares_x=5, squares_y=8, legacy=legacy)   # size where layouts differ
    board = cfg.board().generateImage((400, 640), marginSize=20)
    vid = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (400, 640))
    if not vw.isOpened():
        pytest.skip("no mp4v writer")
    for _ in range(6):
        vw.write(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR))
    vw.release()
    out = tmp_path / "out"
    # Deliberately start with the opposite layout; autodetect must flip.
    argv = ["--calib", "--out", str(out), "--quiet", "--calib-samples", "6",
            "--squares-x", "5", "--squares-y", "8", str(vid)]
    if not legacy:
        argv.insert(1, "--legacy")
    assert ek.main(argv) == 0
    d = np.load(out / "calib" / "c.calib.npz")
    assert len(d["t"]) >= 1
    assert d["corners"].shape[1] == 2 and len(d["ids"]) == d["offsets"][-1]
    assert json.loads(str(d["board_json"]))["legacy"] == legacy


def test_pose_window_start_end(tmp_path):
    vid = tmp_path / "camA.mp4"
    _make_video(vid, n=20)
    out = tmp_path / "out"
    est = StubEstimator()
    assert ek.main(["--out", str(out), "--take", "w", "--quiet", "--start", "0.8",
                    "--end", "1.25", "--proxy", str(vid)], estimator=est) == 0
    d = np.load(out / "w" / "camA.npz")
    assert list(d["frame_idx"]) == [8, 9, 10, 11, 12]
    assert np.allclose(d["t"], d["frame_idx"] / FPS, atol=1e-3)
    assert int(d["window_first_frame_idx"]) == 8
    meta = json.load(open(out / "w" / "meta.json"))
    assert meta["videos"][0]["start"] == 0.8 and meta["videos"][0]["t_first"] == pytest.approx(0.8)
    # --every keeps absolute frame numbers inside a window
    assert ek.main(["--out", str(out), "--take", "e", "--quiet", "--start", "0.5",
                    "--every", "4", str(vid)], estimator=StubEstimator()) == 0
    assert list(np.load(out / "e" / "camA.npz")["frame_idx"]) == [8, 12, 16]
    assert ek.main(["--out", str(out), "--quiet", "--start", "2", "--end", "1",
                    str(vid)], estimator=StubEstimator()) == 2


def test_calib_every_and_min_corners(tmp_path):
    from ..calibration import BoardConfig
    if not hasattr(cv2, "aruco"):
        pytest.skip("cv2.aruco missing")
    cfg = BoardConfig()
    board = cfg.board().generateImage((640, 400), marginSize=20)
    vid = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 400))
    if not vw.isOpened():
        pytest.skip("no mp4v writer")
    for i in range(12):
        f = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR) if i < 8 else np.full((400, 640, 3), 90, np.uint8)
        vw.write(f)
    vw.release()
    out = tmp_path / "out"
    assert ek.main(["--calib", "--out", str(out), "--quiet", "--calib-every", "2",
                    "--start", "0.2", "--end", "1.1", str(vid)]) == 0
    d = np.load(out / "calib" / "c.calib.npz")
    assert list(d["frame_idx"]) == [2, 4, 6]            # board gone from frame 8
    assert int(d["sample_step"]) == 2
    assert ek.main(["--calib", "--out", str(out), "--take", "m", "--quiet", "--calib-every", "1",
                    "--min-corners", "100", str(vid)]) == 0
    assert len(np.load(out / "m" / "c.calib.npz")["t"]) == 0


# ------------------------------------------------- meta.json / time base (review fixes)
def _make_offset_video(path, n=20, fps=FPS, offset=2.0):
    """A video whose container start_time is *offset* s (MKV, PTS from offset*fps)."""
    import av
    from fractions import Fraction
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.codec_context.time_base = Fraction(1, fps)
        for i in range(n):
            img = np.full((H, W, 3), 40, np.uint8)
            cv2.circle(img, (10 + 5 * i, 40), 6, (255, 255, 255), -1)
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts, fr.time_base = i + int(round(offset * fps)), Fraction(1, fps)
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_meta_replaces_stale_cameras_and_writes_timebase(tmp_path):
    from .. import selfcal as sc
    vids = []
    for c in "abc":
        _make_video(tmp_path / f"{c}.mp4")
        vids.append(str(tmp_path / f"{c}.mp4"))
    out = tmp_path / "out"
    take = out / "take"
    assert ek.main(["--out", str(out), "--quiet", *vids], estimator=StubEstimator()) == 0
    meta = json.load(open(take / "meta.json"))
    assert [e["cam_id"] for e in meta["videos"]] == ["a", "b", "c"]
    assert meta["timebase"] == "video"
    # fewer cameras into the same take: c is gone (entry and npz), the loader agrees
    assert ek.main(["--out", str(out), "--quiet", *vids[:2]], estimator=StubEstimator()) == 0
    meta = json.load(open(take / "meta.json"))
    assert [e["cam_id"] for e in meta["videos"]] == ["a", "b"]
    assert not (take / "c.npz").exists()
    assert sc.load_take(str(take)).cam_ids == ["a", "b"]
    # a take that held a session extraction (other clock): plain videos replace all of it
    meta.update(timebase="session", session={"nominal_fps": 30, "t0_ms": 1.0},
                calib_videos=[{"cam_id": "a", "npz": "a.calib.npz"}])
    json.dump(meta, open(take / "meta.json", "w"))
    np.savez(take / "a.calib.npz", t=np.zeros(1))
    assert ek.main(["--out", str(out), "--quiet", vids[0]], estimator=StubEstimator()) == 0
    meta = json.load(open(take / "meta.json"))
    assert meta["timebase"] == "video" and "session" not in meta
    assert "calib_videos" not in meta and not (take / "a.calib.npz").exists()
    assert [e["cam_id"] for e in meta["videos"]] == ["a"]
    tk = sc.load_take(str(take))
    assert tk.cam_ids == ["a"] and tk.timeline == "shared" and tk.session is None


def test_nonzero_container_start_time_one_clock(tmp_path):
    """A video whose container starts at t = 2 s: npz ``t`` is the presentation time
    and --start/--end are on that clock in extract and trim_take (and calibrate_board,
    which compares the npz ``t`` like trim_take)."""
    from .. import selfcal as sc
    vid = tmp_path / "camA.mkv"
    _make_offset_video(vid, n=20, offset=2.0)
    src = ek.VideoSource(str(vid))
    assert src.start_time == pytest.approx(2.0)
    assert [round(t, 3) for _i, t, _img in src.frames()][:3] == [2.0, 2.1, 2.2]
    src.close()
    out = tmp_path / "out"
    assert ek.main(["--out", str(out), "--take", "w", "--quiet", "--start", "2.8",
                    "--end", "3.25", "--keyframes", "2", str(vid)],
                   estimator=StubEstimator()) == 0
    d = np.load(out / "w" / "camA.npz")
    assert list(d["frame_idx"]) == [8, 9, 10, 11, 12]
    assert np.allclose(d["t"], 2.0 + d["frame_idx"] / FPS, atol=1e-3)
    meta = json.load(open(out / "w" / "meta.json"))
    v = meta["videos"][0]
    assert v["start_time"] == pytest.approx(2.0) and v["t_first"] == pytest.approx(2.8)
    # keyframes come from the window (frame_idx counts from the video's first frame)
    kf = sorted(int(f[-10:-4]) for f in v["keyframes"])
    assert len(kf) == 2 and set(kf) <= set(d["frame_idx"].tolist())
    # the whole video trimmed at the same seconds keeps the same frames
    assert ek.main(["--out", str(out), "--take", "all", "--quiet", str(vid)],
                   estimator=StubEstimator()) == 0
    tk = sc.trim_take(sc.load_take(str(out / "all")), 2.8, 3.25)
    assert list(tk.frame_idx[0]) == [8, 9, 10, 11, 12]
    # window size / first frame index (progress bar, keyframe picks) on the same clock
    src = ek.VideoSource(str(vid))
    assert src.first_index(2.8) == 8 and src.window_frames(2.8, 3.25) in (5, 6)   # estimate
    src.close()


def test_estimator_closed_when_cancelled(tmp_path, monkeypatch):
    from ..core import pose
    _make_video(tmp_path / "a.mp4")
    closed = []

    class Est(StubEstimator):
        def close(self):
            closed.append(True)
    monkeypatch.setattr(pose, "load_estimator", lambda model: Est())

    def cancel(cam, i, n):
        raise KeyboardInterrupt("cancel")
    args = ek.build_parser().parse_args(["--out", str(tmp_path / "o"), "--quiet",
                                         str(tmp_path / "a.mp4")])
    with pytest.raises(KeyboardInterrupt):
        ek.run(args, on_camera=cancel)
    assert closed == [True]
