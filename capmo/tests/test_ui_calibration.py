"""
Headless Qt smoke tests for the board check + processing UI
(QT_QPA_PLATFORM=offscreen).

Constructs the real :class:`CapmoMainWindow`, switches the live **Board check**
on (there is no live solve any more: no intrinsics / extrinsics, no rig.json),
feeds frames through the preview hub's stamped latest-frame cache and drives
the :class:`~capmo.core.board_coverage.LiveBoardTracker` with synthetic board
detections (projected real ChArUco geometry) -- tiles go amber (scanning) ->
green (good coverage), the status bar shows the simultaneous views per pair,
the board config + verdict land in the session manifest, and the offline
pipeline runs in a QThread worker (injected fake) and feeds View. No phones,
no real video.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ..core.board_coverage import GOOD, IDLE, SCANNING  # noqa: E402
from ..ui.main_window import CapmoMainWindow  # noqa: E402
from .test_board_coverage import CFG, GRID9, det_at  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def _make_window(tmp_path, **kw):
    # port=0 -> OS picks a free port, so the test never clashes with anything.
    return CapmoMainWindow(host="127.0.0.1", port=0, record_dir=str(tmp_path),
                           board_cfg=CFG, **kw)


class _Conn:
    def __init__(self, pid):
        self.phone_id = pid
        self.label = pid
        self.fps = 30.0
        self.frames = 0

        class _Clock:
            offset_ms = 0.0
            rtt_ms = 0.0
            ready = True

        self.clock = _Clock()


class _Feed:
    """Fake preview: per phone a frame + host time + sequence number, and a
    fake detector returning the detection queued for the phone whose frame it
    is handed. The detector depends on the image: it identifies the phone by
    the frame object it receives and records every call that sees a frame that
    differs from the pristine one (``tainted``) or runs on the GUI thread
    (``threads``), so overlay-burned-into-the-cache and detection-on-the-GUI
    bugs are caught."""

    def __init__(self, win, monkeypatch, pids):
        self.win = win
        self.frames = {p: np.full((480, 640, 3), 100 + 10 * i, np.uint8)
                       for i, p in enumerate(pids)}
        self.pristine = {p: f.copy() for p, f in self.frames.items()}
        self.state = {p: [None, 0] for p in pids}     # pid -> [t_ms, seq]
        self.dets = {}
        self.tainted = []                             # phones whose frame was modified
        self.threads = set()                          # thread ids that ran the detector
        self.detect_delay = 0.0
        conns = [_Conn(p) for p in pids]
        monkeypatch.setattr(win._server, "connections", lambda: conns)
        monkeypatch.setattr(win._preview, "latest_stamped", self.latest_stamped)
        # the frames are flat grey (no edges): switch the blur check off
        win._tracker.th.min_sharpness = 0.0
        # the tracker's detector: return the detection queued for that phone
        monkeypatch.setattr(type(win._tracker), "detector",
                            property(lambda tr: self.detect))

    @property
    def frame(self):                                  # the first phone's frame
        return next(iter(self.frames.values()))

    def latest_stamped(self, pid):
        t, seq = self.state[pid]
        return self.frames[pid], t, seq

    def detect(self, img):
        self.threads.add(threading.get_ident())
        pid = next(p for p, f in self.frames.items() if img is f)
        if not np.array_equal(img, self.pristine[pid]):
            self.tainted.append(pid)
        if self.detect_delay:
            time.sleep(self.detect_delay)
        d = self.dets.get(pid)
        if isinstance(d, Exception):
            raise d
        return d if d is not None else det_at(320, 240, ids=range(3))

    def drain(self):
        """Wait for the scan thread and deliver its results to the GUI."""
        assert self.win._scanner.wait_idle(10.0)
        QApplication.processEvents()

    def show(self, t_s, **dets):
        """Next tick: these phones show these detections at host time *t_s*."""
        for pid, det in dets.items():
            self.state[pid][0] = t_s * 1000.0
            self.state[pid][1] += 1
            self.dets[pid] = det
        self.win._scan_ts.clear()          # bypass the per-phone detect throttle
        self.win._refresh()
        self.drain()


def _views(n=18):
    out = []
    for i in range(n):
        cx, cy = GRID9[i % 9]
        out.append(det_at(cx + 25 * (i // 9), cy, (-28, 28)[i % 2],
                          (0, 1, 0) if (i // 2) % 2 else (1, 0, 0)))
    return out


def test_board_check_button_and_no_live_solve(app, tmp_path):
    win = _make_window(tmp_path)
    try:
        assert win._board_btn.text() == "Board check"
        for gone in ("_calib_btn", "_try_solve_intrinsics", "_try_finalize_rig",
                     "_intrinsics", "_intrinsics_K", "rig"):
            assert not hasattr(win, gone), gone
        win._board_btn.setChecked(True)
        assert win._board_check and win._board_btn.text() == "Stop board check"
        win._board_btn.setChecked(False)
        assert not win._board_check
        assert not os.path.exists(os.path.join(str(tmp_path), "rig.json"))
    finally:
        win.close()


def test_refresh_without_board_check_passes_frame_through(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        feed.show(1.0, camA=det_at(320, 240))
        tile = win._tiles["camA"]
        assert tile._video.pixmap() is not None and not tile._video.pixmap().isNull()
        assert tile.board_state == "off" and tile.board_text() == ""
        assert win._tracker.statuses()["camA"].detections == 0    # nothing detected
    finally:
        win.close()


def test_board_check_tile_goes_amber_then_green(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._refresh()
        win._board_btn.setChecked(True)
        tile = win._tiles["camA"]
        assert tile.board_state == IDLE and "no board" in tile.board_text()
        views = _views()
        for k, det in enumerate(views[:5]):
            feed.show(1.0 + k, camA=det)
        assert tile.board_state == SCANNING
        assert "scanning" in tile.board_text() and "5/15 views" in tile.board_text()
        assert "CameraTile" in tile.styleSheet() and "#caa64a" in tile.styleSheet()
        # the same frame again (same seq) is not ingested twice
        win._scan_ts.clear()
        win._refresh()
        feed.drain()
        assert win._tracker.status("camA").detections == 5
        for k, det in enumerate(views[5:]):
            feed.show(10.0 + k, camA=det)
        assert tile.board_state == GOOD
        assert "good coverage" in tile.board_text() and "#46c846" in tile.styleSheet()
        assert "1/1 cameras good" in win._status.text()
    finally:
        win.close()


def test_two_cameras_pairs_and_partner_hint(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA", "camB"])
        win._refresh()
        win._board_btn.setChecked(True)
        views = _views()
        for k, det in enumerate(views):            # only camA sees the board
            feed.show(1.0 + k, camA=det)
        st = win._tracker.statuses()
        assert st["camB"].state == IDLE
        assert st["camA"].hints[-1] == "show to camA together with camB"
        assert win._tiles["camA"].board_state == SCANNING      # not linked to camB yet
        for k in range(3):                          # now both, at the same instant
            feed.show(30.0 + k, camA=det_at(200 + 100 * k, 240), camB=det_at(400 - 80 * k, 240))
        assert win._tracker.pair_counts() == {"camA-camB": 3}
        assert "camA-camB:3" in win._status.text()
        assert win._tiles["camA"].board_state == GOOD
        assert "camA:3" in win._tiles["camB"].board_text()
    finally:
        win.close()


def test_detection_failure_is_swallowed(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._board_btn.setChecked(True)
        feed.show(1.0, camA=RuntimeError("detect blew up"))      # must not raise
        tile = win._tiles["camA"]
        assert tile._video.pixmap() is not None and not tile._video.pixmap().isNull()
        assert tile.board_state == IDLE
    finally:
        win.close()


def test_board_check_off_clears_tile_state(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._board_btn.setChecked(True)
        feed.show(1.0, camA=det_at(320, 240))
        tile = win._tiles["camA"]
        assert tile.board_state == SCANNING
        win._board_btn.setChecked(False)
        assert tile.board_state == "off" and tile._board.isHidden()
        assert tile.styleSheet() == ""
    finally:
        win.close()


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def test_recording_with_board_check_stores_board_in_manifest(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._board_btn.setChecked(True)
        win._rec_btn.setChecked(True)
        feed.show(1.0, camA=det_at(320, 240))
        win._rec_btn.setChecked(False)
        man = _manifest(win._recorder.last_session_dir)
        assert man["board"] == CFG.to_dict()
        assert man["board_check"]["cameras"]["camA"]["views"] == 1
        assert man["board_check"]["board"] == CFG.to_dict()
        # board check switched on only while recording -> still stored
        win._board_btn.setChecked(False)
        win._rec_btn.setChecked(True)
        win._board_btn.setChecked(True)
        win._rec_btn.setChecked(False)
        assert _manifest(win._recorder.last_session_dir)["board"] == CFG.to_dict()
        # without board check: no board -> the pipeline self-calibrates
        win._board_btn.setChecked(False)
        win._rec_btn.setChecked(True)
        win._rec_btn.setChecked(False)
        man = _manifest(win._recorder.last_session_dir)
        assert man["board"] is None and "board_check" not in man
    finally:
        win.close()


# ------------------------------------------------------- Process… / View
def _wait(app, cond, timeout=10.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    return cond()


class _FakeResult:
    def __init__(self, session_dir, path):
        self.session_dir = session_dir
        self.joints3d_path = path
        self.rig_source = "board"

    def joints3d(self):
        from ..core.joints3d import load_joints3d
        return load_joints3d(self.joints3d_path)


def _fake_npz(tmp_path):
    """A 3-frame standing skeleton in the Z-up joints3d contract."""
    from ..retarget import default_skeleton  # noqa: F401  (import check)
    J = np.zeros((3, 33, 3))
    base = {11: (0.2, 0, 1.45), 12: (-0.2, 0, 1.45), 13: (0.45, 0, 1.45), 14: (-0.45, 0, 1.45),
            15: (0.7, 0, 1.45), 16: (-0.7, 0, 1.45), 23: (0.1, 0, 0.95), 24: (-0.1, 0, 0.95),
            25: (0.1, 0, 0.5), 26: (-0.1, 0, 0.5), 27: (0.1, 0, 0.08), 28: (-0.1, 0, 0.08),
            0: (0, -0.05, 1.7)}
    for k, v in base.items():
        J[:, k] = v
    J[:, 1:11] = J[:, [0]]
    J[:, 17:23] = J[:, [15, 16, 15, 16, 15, 16]]
    J[:, 29:33] = J[:, [27, 28, 27, 28]]
    J[1, 15] = (0.45, 0, 1.7)                      # raise the left forearm
    p = str(tmp_path / "x_joints3d.npz")
    np.savez(p, joints=J.astype(np.float32), valid=np.ones((3, 33), bool), fps=30.0,
             t=np.arange(3) / 30.0, names=np.array([str(i) for i in range(33)]))
    return p


def test_process_runs_in_a_worker_thread_then_view_plays(app, tmp_path):
    import threading
    calls = {}
    npz = _fake_npz(tmp_path)

    def fake_pipeline(session_dir, progress):
        calls["thread"] = threading.current_thread()
        progress("extract", "camA", 0.5)
        progress("reconstruct", "board", None)
        return _FakeResult(session_dir, npz)

    win = _make_window(tmp_path, pipeline_fn=fake_pipeline)
    try:
        # nothing recorded yet: graceful
        assert not win.start_processing()
        assert "no recorded session" in win._status.text()
        win._rec_btn.setChecked(True)
        win._rec_btn.setChecked(False)
        sess = win._recorder.last_session_dir
        win._process_btn.click()
        assert win.processing and not win._process_btn.isEnabled()
        assert _wait(app, lambda: not win.processing)
        assert calls["thread"] is not threading.main_thread()
        assert win.last_result.session_dir == sess
        assert "Processed (board rig)" in win._status.text()
        assert win._process_btn.isEnabled()
        # View plays the processed track (Z-up file -> Y-up character)
        win._view_btn.setChecked(True)
        assert win._view_btn.text() == "Stop view"
        assert win._viewing and not win._character.isHidden()
        assert len(win._character._track) == 3
        assert "Viewing 3 frame(s)" in win._status.text()
        win._view_btn.setChecked(False)
        assert win._view_btn.text() == "View"
    finally:
        win.close()


def test_view_processes_first_and_failures_are_reported(app, tmp_path):
    npz = _fake_npz(tmp_path)
    state = {"fail": True}

    def fake_pipeline(session_dir, progress):
        if state["fail"]:
            raise RuntimeError("no person found")
        return _FakeResult(session_dir, npz)

    win = _make_window(tmp_path, pipeline_fn=fake_pipeline)
    try:
        win._rec_btn.setChecked(True)
        win._rec_btn.setChecked(False)
        win._view_btn.setChecked(True)                   # -> processes first
        assert _wait(app, lambda: not win.processing)
        assert "Process failed: RuntimeError: no person found" in win._status.text()
        assert not win._view_btn.isChecked()
        state["fail"] = False
        win._view_btn.setChecked(True)
        assert _wait(app, lambda: win._view_btn.text() == "Stop view")
        assert len(win._character._track) == 3
    finally:
        win.close()


# ------------------------------------------------- phase-2 regressions
def test_refresh_never_mutates_the_cached_preview_frame(app, tmp_path, monkeypatch):
    """The overlay / corner drawing must not be burned into the hub's frame
    (the detector reads the same array on the next tick)."""
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA", "camB"])
        win._refresh()
        win._board_btn.setChecked(True)
        from ..core.board import Detection
        feed.show(0.5, camA=Detection(None, None, 0), camB=Detection(None, None, 0))
        win._refresh()                              # nothing detected: overlay only
        assert feed.tainted == []
        for k, det in enumerate(_views(6)):
            feed.show(1.0 + k, camA=det, camB=det_at(300, 200 + 5 * k))
            win._refresh()                          # a second tick draws the held result
            for pid in feed.frames:
                assert np.array_equal(feed.frames[pid], feed.pristine[pid]), (pid, k)
        assert feed.tainted == []
        assert win._tiles["camA"]._video.pixmap() is not None
    finally:
        win.close()


@pytest.mark.parametrize("tile_size", [(300, 200), (1200, 900)])
def test_set_frame_does_not_modify_its_input(app, tile_size):
    from ..ui.main_window import CameraTile
    tile = CameraTile("a")
    tile._video.setFixedSize(*tile_size)            # smaller and larger than the frame
    frame = np.full((48, 64, 3), 90, np.uint8)
    keep = frame.copy()
    tile.set_frame(frame, overlay_text="3 markers", captured=True)
    assert np.array_equal(frame, keep)
    assert not tile._video.pixmap().isNull()


def test_detection_runs_off_the_gui_thread_and_tile_state_updates(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._refresh()
        win._board_btn.setChecked(True)
        for k, det in enumerate(_views(5)):
            feed.show(1.0 + k, camA=det)
        assert feed.threads and threading.get_ident() not in feed.threads
        assert win._scanner.thread_ident in feed.threads
        assert win._tiles["camA"].board_state == SCANNING
        assert win._tracker.status("camA").detections == 5
    finally:
        win.close()


def test_slow_detection_does_not_block_the_gui_tick(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA", "camB", "camC", "camD"])
        feed.detect_delay = 0.25
        win._refresh()
        win._board_btn.setChecked(True)
        for pid in feed.state:
            feed.state[pid][:] = [1000.0, 1]
        t0 = time.perf_counter()
        win._refresh()
        assert time.perf_counter() - t0 < 0.2        # 4 x 0.25 s of detection did not run here
        feed.drain()
        assert win._tracker.status("camD").last_corners == 3   # camD was scanned too
    finally:
        win.close()


def test_tile_is_not_redrawn_when_nothing_changed(app, tmp_path, monkeypatch):
    from ..ui.main_window import CameraTile
    win = _make_window(tmp_path)
    try:
        feed = _Feed(win, monkeypatch, ["camA"])
        win._refresh()
        calls = []
        orig = CameraTile.set_frame
        monkeypatch.setattr(CameraTile, "set_frame",
                            lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1])
        feed.show(1.0, camA=det_at(320, 240))
        win._refresh()
        win._refresh()                              # same seq: no new work
        assert len(calls) == 1
        feed.show(2.0, camA=det_at(320, 240))       # new frame: drawn again
        assert len(calls) == 2
    finally:
        win.close()


def test_view_pauses_the_refresh_timer(app, tmp_path):
    npz = _fake_npz(tmp_path)
    win = _make_window(tmp_path, pipeline_fn=lambda session_dir, progress:
                       _FakeResult(session_dir, npz))
    try:
        win._rec_btn.setChecked(True)
        win._rec_btn.setChecked(False)
        assert win._timer.isActive()
        win._view_btn.setChecked(True)
        assert _wait(app, lambda: win._view_btn.text() == "Stop view")
        assert not win._timer.isActive()
        win._view_btn.setChecked(False)
        assert win._timer.isActive()
    finally:
        win.close()


def test_quality_dropdown_is_disabled_while_recording(app, tmp_path):
    win = _make_window(tmp_path)
    try:
        assert win._quality.isEnabled()
        win._rec_btn.setChecked(True)
        assert not win._quality.isEnabled()
        win._rec_btn.setChecked(False)
        assert win._quality.isEnabled()
    finally:
        win.close()


def test_failed_record_start_resets_the_button(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    try:
        def boom(**kw):
            raise OSError("disk full")
        monkeypatch.setattr(win._recorder, "start", boom)
        win._rec_btn.setChecked(True)
        assert not win._rec_btn.isChecked() and win._rec_btn.text() == "● Record"
        assert "Could not start recording" in win._status.text() and "disk full" in win._status.text()
        assert win._quality.isEnabled()
    finally:
        win.close()


def test_closing_mid_recording_keeps_the_board_check_verdict(app, tmp_path, monkeypatch):
    win = _make_window(tmp_path)
    feed = _Feed(win, monkeypatch, ["camA"])
    win._board_btn.setChecked(True)
    win._rec_btn.setChecked(True)
    feed.show(1.0, camA=det_at(320, 240))
    sess = win._recorder.session_dir
    assert sess
    win.close()
    man = _manifest(sess)
    assert man["board"] == CFG.to_dict()
    assert man["board_check"]["cameras"]["camA"]["views"] == 1


def test_phone_connect_sends_the_selected_stream_settings(app, tmp_path):
    from .. import protocol

    class Conn:
        phone_id = "p"
        sent = []

        def send(self, msg):
            self.sent.append(msg)

    win = _make_window(tmp_path, width=1920, height=1080, fps=30)
    try:
        assert win._stream == (1920, 1080, 30)
        win._on_phone_connect(Conn())
        assert Conn.sent == [protocol.cmd_start(1920, 1080), protocol.cmd_framerate(30)]
        win._quality.setCurrentIndex(2)             # 720p60
        assert win._stream == (1280, 720, 60)
    finally:
        win.close()


def test_close_during_process_returns_at_once_and_joins_the_worker(app, tmp_path):
    exited = threading.Event()
    started = threading.Event()

    def slow_pipeline(session_dir, progress):
        started.set()
        try:
            while True:                              # only a cancel gets us out
                time.sleep(0.05)
                progress("extract", "camA", None)
        finally:
            exited.set()

    win = _make_window(tmp_path, pipeline_fn=slow_pipeline)
    closed = []
    win.closed.connect(lambda: closed.append(1))
    win._rec_btn.setChecked(True)
    win._rec_btn.setChecked(False)
    win.show()
    win._process_btn.click()
    assert _wait(app, started.is_set)
    t0 = time.perf_counter()
    win.close()
    assert time.perf_counter() - t0 < 0.5            # the GUI is not blocked on the worker
    assert win.isHidden() and not closed             # hidden; the real close is deferred
    assert _wait(app, lambda: bool(closed), timeout=10)
    assert exited.is_set() and win._thread is None and win._worker is None
    assert win._closed
    win.close()                                      # idempotent


def test_close_without_a_worker_is_immediate(app, tmp_path):
    win = _make_window(tmp_path)
    closed = []
    win.closed.connect(lambda: closed.append(1))
    win.close()
    assert closed == [1] and win._closed
