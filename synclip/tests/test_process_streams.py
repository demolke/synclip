"""
Regression tests for the blendshape streams produced during a PROCESS pass.

  - The AI stream must be set in-memory directly from the frames the generator
    emitted, so it survives even when on-disk persistence fails -- the data must
    not depend on a save/reload round-trip to appear in REVIEW. (This was the
    "AI ran but the stream is empty" bug.)
  - When the rhubarb binary is available, a 'rhubarb' viseme track is added.

Headless (Qt offscreen + dummy audio); skipped when heavy deps are absent.
"""

from __future__ import annotations

import os
import shutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from synclip.app_state import VideoSource  # noqa: E402
from synclip import data as data_mod  # noqa: E402
from synclip import rhubarb_lipsync  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _fake_bgr():
    import numpy as np
    return np.zeros((48, 64, 3), dtype="uint8")


def _enter_process(w, audio_path: str) -> None:
    """Drive the window into PROCESS_VIDEO with one synthetic recorded frame."""
    w._video_path = "clip.mp4"
    w._video_source = "clip.mp4"
    w._current_audio_path = audio_path
    w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))
    w._apply(w._sm.start_process_video())
    w._recorded_frames = []
    w._record_frame_index = 0
    w._process_frac = 0.0
    w._on_frame_ready(_fake_bgr(), None, [0.1] * 52,
                      {"rot": [0.0] * 3, "pos": [0.0] * 3})


def _pcm_wav() -> str | None:
    """A decodable PCM .wav shipped beside the rhubarb binary, if any."""
    binary = rhubarb_lipsync.rhubarb_bin()
    if not binary:
        return None
    root = os.path.dirname(os.path.realpath(shutil.which(binary) or binary))
    res = os.path.join(root, "tests", "resources")
    if not os.path.isdir(res):
        return None
    for name in sorted(os.listdir(res)):
        if name.lower().endswith(".wav") and "int16" in name and "flac" not in name:
            return os.path.join(res, name)
    return None


def test_ai_stream_set_from_emitted_frames_even_if_save_fails(
        qapp, tmp_path, monkeypatch):
    from synclip.ui.main_window import MainWindow
    audio_path = str(tmp_path / "clip.wav")
    open(audio_path, "wb").close()

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        _enter_process(w, audio_path)

        # Let the take be created (save #1), then make the stream-persistence
        # save (#2, inside _save_current_take) fail. The in-memory stream must
        # still hold the data -- it must not depend on a save/reload round-trip.
        real_save = data_mod.save_synclip
        calls = {"n": 0}
        def _boom(path, data):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise IOError("disk full")
            return real_save(path, data)
        monkeypatch.setattr(data_mod, "save_synclip", _boom)

        ai_frames = [
            {"audio_position_ms": 0.0, "blendshapes": [0.9] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [0.9] * 52},
        ]
        w._on_process_finished(ai_frames)

        assert w._streams.has("ai"), "AI stream lost when disk save failed"
        vals = w._streams.sample("ai", 0.0)
        assert vals is not None
        assert abs(vals[0] - 0.9) < 1e-6
    finally:
        w._worker.stop()


@pytest.mark.skipif(not rhubarb_lipsync.is_available(),
                    reason="rhubarb binary not on PATH")
def test_rhubarb_track_added_during_process(qapp, tmp_path):
    from synclip.ui.main_window import MainWindow
    src = _pcm_wav()
    if src is None:
        pytest.skip("no rhubarb sample wav available")
    audio_path = str(tmp_path / "clip.wav")
    shutil.copy(src, audio_path)

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        _enter_process(w, audio_path)
        w._on_process_finished([])  # no AI frames; rhubarb is independent
        assert w._streams.has("rhubarb"), "rhubarb track not added during process"
        frames = w._streams.frames("rhubarb")
        assert frames and all(len(f["blendshapes"]) == 52 for f in frames)
    finally:
        w._worker.stop()


def test_rhubarb_gets_extracted_audio_not_video(qapp, tmp_path, monkeypatch):
    """For a video source, rhubarb must be fed the extracted OGG, not the video
    container (rhubarb only reads WAV/OGG)."""
    from synclip.ui.main_window import MainWindow
    video = str(tmp_path / "clip.mp4")
    audio_ogg = str(tmp_path / "clip.ogg")
    open(video, "wb").close()
    open(audio_ogg, "wb").close()

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # _current_audio_path is the video container; the played/processed audio
        # is the extracted OGG (what _load_video_file sets up).
        _enter_process(w, video)
        w._video_audio_path = audio_ogg

        captured: dict = {}
        monkeypatch.setattr(w, "_generate_rhubarb_stream",
                            lambda path, dur: captured.update(path=path))
        w._on_process_finished([])

        assert captured.get("path") == audio_ogg, (
            f"rhubarb got {captured.get('path')!r}, expected the extracted OGG")
    finally:
        w._worker.stop()
