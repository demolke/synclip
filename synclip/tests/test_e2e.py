"""
T-14: end-to-end smoke of a scripted PROCESS_VIDEO run.

MediaPipe can't init on a headless CI box (no GL), so instead of a real camera
pass we drive the MainWindow's PROCESS pipeline directly: put it in
PROCESS_VIDEO mode against a fake file source, feed synthetic frame_ready
payloads (as the worker would), then signal process_finished. We assert:
  - a connected mock client received a monotonic MODE_LIVE stream, and
  - the saved take has the expected frame count and lands the app in REVIEW.

Runs headless (Qt offscreen + dummy audio). Skipped if heavy deps are absent.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from synclip.app_state import Mode, VideoSource  # noqa: E402
from synclip.ipc_server import (  # noqa: E402
    HELLO_MAGIC, MODE_LIVE, _FRAME_STRUCT, _MSG_HEADER,
)
from synclip import data as data_mod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class _Collector:
    """Background thread: connect, read HELLO, collect frame magics + positions."""

    def __init__(self, host: str, port: int) -> None:
        self.magics: list[int] = []
        self.positions: list[float] = []
        self._stop = False
        self._sock = socket.create_connection((host, port), timeout=3.0)
        self._sock.settimeout(0.5)
        self._buf = b""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                if self._stop:
                    raise ConnectionError
                continue
            if not chunk:
                raise ConnectionError
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _run(self) -> None:
        try:
            magic, length = _MSG_HEADER.unpack(self._recv_exact(_MSG_HEADER.size))
            assert magic == HELLO_MAGIC
            self._recv_exact(length)  # consume name list
            while not self._stop:
                fields = _FRAME_STRUCT.unpack(self._recv_exact(_FRAME_STRUCT.size))
                self.magics.append(fields[0])
                self.positions.append(fields[1])
        except (ConnectionError, OSError):
            pass

    def stop(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass


def test_scripted_process_run(qapp, tmp_path):
    from synclip.ui.main_window import MainWindow

    # A fake audio sidecar path so append_take can persist the take.
    audio_path = str(tmp_path / "clip.wav")
    open(audio_path, "wb").close()

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        collector = _Collector(w._ipc.host, w._ipc.port)
        # Wait until the server has accepted + handshaked the client (Windows can
        # be slow to schedule the accept thread, so polling beats a fixed sleep).
        deadline = time.monotonic() + 5.0
        while w._ipc.client_count == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert w._ipc.client_count > 0, "IPC client never connected to server"

        # Pretend a video file was loaded (skip real extraction).
        w._video_path = "clip.mp4"
        w._video_source = "clip.mp4"
        w._current_audio_path = audio_path
        w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))

        # Enter PROCESS_VIDEO via the bridge.
        plan = w._sm.start_process_video()
        assert plan.mode == Mode.PROCESS_VIDEO
        w._recorded_frames = []
        w._record_frame_index = 0
        w._process_frac = 0.0
        w._apply(plan)
        assert w._broadcast_enabled is True
        assert w._broadcast_mode == MODE_LIVE

        # Feed synthetic frames as the worker would (progress then frame).
        n_frames = 8
        for i in range(n_frames):
            w._on_process_progress((i + 1) / n_frames, 120.0)
            bs = [i / 100.0] * 52
            w._on_frame_ready(_fake_bgr(), None, bs, {"rot": [0.0] * 3, "pos": [0.0] * 3})

        assert len(w._recorded_frames) == n_frames

        # Finish: saves the take, transitions to REVIEW.
        w._on_process_finished([])
        assert w._sm.mode == Mode.REVIEW

        # The take was persisted with the right frame count.
        doc = data_mod.load_synclip(audio_path)
        assert doc is not None
        assert len(doc["takes"]) == 1
        assert len(doc["takes"][0]["streams"]["mediapipe"]) == n_frames

        # The client saw a MODE_LIVE stream with monotonic non-decreasing pos.
        # Wait up to 1s for frames to arrive (Windows IPC can be slower).
        deadline = time.monotonic() + 1.0
        while len(collector.magics) < n_frames - 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        collector.stop()
        assert len(collector.magics) >= n_frames - 1  # allow last in-flight slack
        assert all(m == MODE_LIVE for m in collector.magics)
        assert collector.positions == sorted(collector.positions)
    finally:
        w._worker.stop()


def test_recorded_take_is_raw_not_filtered(qapp, tmp_path):
    """The values we SAVE into a take must be the raw MediaPipe values --
    smoothing/gain are display/broadcast post-processing and must NOT be baked
    into the recorded data."""
    from synclip.ui.main_window import MainWindow

    audio_path = str(tmp_path / "clip2.wav")
    open(audio_path, "wb").close()

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Crank smoothing + a non-identity gain curve on the output view's
        # modifier stack so any leakage into the recorded take would show.
        from synclip.modifiers import ModifierConfig
        from synclip.ui.main_window import OUTPUT_VIEW
        w._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("curves", params={
                "gain_curve": [[0.0, 0.0], [1.0, 0.0]],
                "width_curve": [[0.0, 0.0], [1.0, 1.0]]}),
            ModifierConfig("smooth", influence=0.9),
        ]
        w._pipelines[OUTPUT_VIEW].apply_config(w._views[OUTPUT_VIEW])

        w._video_path = "clip2.mp4"
        w._video_source = "clip2.mp4"
        w._current_audio_path = audio_path
        w._sm.set_video_source(VideoSource(kind="file", path="clip2.mp4"))
        w._apply(w._sm.start_process_video())
        w._recorded_frames = []
        w._record_frame_index = 0

        raw = [0.37] * 52
        for i in range(4):
            w._on_process_progress((i + 1) / 4, 100.0)
            w._on_frame_ready(_fake_bgr(), None, list(raw),
                              {"rot": [0.0] * 3, "pos": [0.0] * 3})

        # Every recorded frame must equal the RAW input exactly.
        for fr in w._recorded_frames:
            assert fr["blendshapes"] == raw
    finally:
        w._worker.stop()


def _fake_bgr():
    import numpy as np
    return np.zeros((48, 64, 3), dtype="uint8")
