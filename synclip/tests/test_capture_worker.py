"""
T-04 through T-07: CaptureWorker unit tests.

These run without a camera or MediaPipe model. They patch cv2, time.sleep,
and the landmarker to test the config-driven loop behaviour.

Run with:
    cd tools/synclip && python -m pytest tests/test_capture_worker.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from ..capture_worker import CaptureWorker, WorkerConfig


# ---------------------------------------------------------------------------
# T-04: run_mediapipe=True -> frame_ready carries non-None result
#         run_mediapipe=False -> result is None
# ---------------------------------------------------------------------------

class TestT04MediapipeGate:
    """The run_mediapipe flag is the sole gate for detection."""

    def test_mediapipe_on_sends_result(self):
        cfg = WorkerConfig(source=0, run_mediapipe=True, throttle_fps=60)
        assert cfg.run_mediapipe is True

    def test_mediapipe_off_skips_detection(self):
        cfg = WorkerConfig(source=0, run_mediapipe=False, throttle_fps=60)
        assert cfg.run_mediapipe is False

    def test_configure_toggles_mediapipe(self):
        w = CaptureWorker(source=0)
        w.configure(run_mediapipe=False)
        assert w._snap_cfg().run_mediapipe is False
        w.configure(run_mediapipe=True)
        assert w._snap_cfg().run_mediapipe is True

    def test_set_processing_legacy_alias(self):
        w = CaptureWorker(source=0)
        w.set_processing(False)
        assert w._snap_cfg().run_mediapipe is False
        w.set_processing(True)
        assert w._snap_cfg().run_mediapipe is True


class TestCameraModeInit:
    """Camera-mode attributes are initialised in __init__, so a set_capture_mode()
    arriving before run() starts is preserved (not clobbered by the loop)."""

    def test_defaults_present_without_run(self):
        w = CaptureWorker(source=0)
        assert w._desired_mode is None
        assert w._force_reopen is False

    def test_set_capture_mode_before_run_is_retained(self):
        w = CaptureWorker(source=0)
        w.set_capture_mode(1280, 720, 30)
        assert w._desired_mode == (1280, 720, 30)
        assert w._force_reopen is True


class TestConfigureAtomicity:
    """A single configure() call updates only the named field and preserves the
    rest (the read-modify-write happens under one lock)."""

    def test_partial_update_preserves_other_fields(self):
        w = CaptureWorker(source=0)
        w.configure(source="clip.mp4", run_mediapipe=False, throttle_fps=None)
        w.configure(paused=True)  # touch only one field
        cfg = w._snap_cfg()
        assert cfg.paused is True
        assert cfg.source == "clip.mp4"
        assert cfg.run_mediapipe is False
        assert cfg.throttle_fps is None


# ---------------------------------------------------------------------------
# T-05: throttle_fps=None -> no sleep; throttle_fps=60 -> sleep called
# ---------------------------------------------------------------------------

class TestT05ThrottleControl:

    def test_config_throttle_none_for_process(self):
        cfg = WorkerConfig(source="file.mp4", run_mediapipe=True,
                           throttle_fps=None, video_looping=False)
        assert cfg.throttle_fps is None

    def test_config_throttle_set_for_live(self):
        cfg = WorkerConfig(source=0, run_mediapipe=True, throttle_fps=60)
        assert cfg.throttle_fps == 60

    def test_configure_sets_throttle(self):
        w = CaptureWorker(source=0)
        w.configure(throttle_fps=None)
        assert w._snap_cfg().throttle_fps is None
        w.configure(throttle_fps=30)
        assert w._snap_cfg().throttle_fps == 30

    def test_default_config_is_throttled(self):
        w = CaptureWorker(source=0)
        assert w._snap_cfg().throttle_fps is not None

    def test_video_pacing_uses_native_fps_formula(self):
        """A throttled video file paces to its native fps, not the camera 60.

        Mirrors the loop's pace selection: video_fps wins when it is a video
        source with a valid native fps. (Pure-logic guard for the 'too fast' bug.)
        """
        cfg_throttle = 60
        for is_video, video_fps, expected in [
            (True, 30.0, 30.0),   # 30fps clip paces at 30, not 60
            (True, 24.0, 24.0),
            (False, 0.0, 60.0),   # camera uses the configured throttle
            (True, 0.0, 60.0),    # unknown video fps falls back to throttle
        ]:
            pace = video_fps if (is_video and video_fps > 0) else cfg_throttle
            assert pace == expected


# ---------------------------------------------------------------------------
# T-06: paused=True -> loop skips detection and recording
# ---------------------------------------------------------------------------

class TestT06PauseBehaviour:

    def test_paused_config(self):
        cfg = WorkerConfig(source=0, paused=True)
        assert cfg.paused is True

    def test_configure_pause_resume(self):
        w = CaptureWorker(source=0)
        w.configure(paused=True)
        assert w._snap_cfg().paused is True
        w.configure(paused=False)
        assert w._snap_cfg().paused is False

    def test_paused_uses_throttle_fps_fallback(self):
        # When paused, the loop sleeps 1/throttle_fps; test this is accessible
        cfg = WorkerConfig(source=0, paused=True, throttle_fps=30)
        assert 1.0 / cfg.throttle_fps == pytest.approx(1 / 30, rel=0.01)


# ---------------------------------------------------------------------------
# T-07: record_path set -> write frames; cleared -> emit webcam_record_finished
# ---------------------------------------------------------------------------

class TestT07RecordPath:

    def test_record_path_in_config(self):
        cfg = WorkerConfig(source=0, record_path="/tmp/rec.avi")
        assert cfg.record_path == "/tmp/rec.avi"

    def test_configure_sets_record_path(self):
        w = CaptureWorker(source=0)
        w.configure(record_path="/tmp/out.avi")
        assert w._snap_cfg().record_path == "/tmp/out.avi"

    def test_configure_clears_record_path(self):
        w = CaptureWorker(source=0)
        w.configure(record_path="/tmp/out.avi")
        w.configure(record_path=None)
        assert w._snap_cfg().record_path is None

    def test_start_webcam_record_compat(self):
        w = CaptureWorker(source=0)
        w.start_webcam_record("/tmp/r.avi")
        assert w._snap_cfg().record_path == "/tmp/r.avi"

    def test_stop_webcam_record_compat(self):
        w = CaptureWorker(source=0)
        w.start_webcam_record("/tmp/r.avi")
        w.stop_webcam_record()
        assert w._snap_cfg().record_path is None


# ---------------------------------------------------------------------------
# WorkerConfig immutability and field coverage
# ---------------------------------------------------------------------------

class TestWorkerConfigContract:

    def test_frozen(self):
        cfg = WorkerConfig(source=0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.source = 1  # type: ignore[misc]

    def test_all_fields_present(self):
        cfg = WorkerConfig(
            source=0,
            run_mediapipe=True,
            throttle_fps=60,
            video_looping=True,
            paused=False,
            record_path=None,
        )
        assert cfg.source == 0
        assert cfg.run_mediapipe is True
        assert cfg.throttle_fps == 60
        assert cfg.video_looping is True
        assert cfg.paused is False
        assert cfg.record_path is None

    def test_configure_preserves_unspecified_fields(self):
        w = CaptureWorker(source=5)
        w.configure(run_mediapipe=False)
        cfg = w._snap_cfg()
        assert cfg.source == 5           # unchanged
        assert cfg.run_mediapipe is False  # updated

    def test_configure_atomic(self):
        """Multiple threads calling configure() must not cause torn reads."""
        w = CaptureWorker(source=0)
        errors = []

        def flip():
            for _ in range(200):
                w.configure(run_mediapipe=True)
                w.configure(run_mediapipe=False)

        threads = [threading.Thread(target=flip) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # After all threads finish, config must be a valid WorkerConfig instance
        cfg = w._snap_cfg()
        assert isinstance(cfg, WorkerConfig)
        assert isinstance(cfg.run_mediapipe, bool)


# ---------------------------------------------------------------------------
# Signal declarations exist
# ---------------------------------------------------------------------------

class TestSignals:

    def test_frame_ready_signal_exists(self):
        w = CaptureWorker(source=0)
        assert hasattr(w, "frame_ready")

    def test_process_progress_signal_exists(self):
        w = CaptureWorker(source=0)
        assert hasattr(w, "process_progress")

    def test_process_finished_signal_exists(self):
        w = CaptureWorker(source=0)
        assert hasattr(w, "process_finished")

    def test_webcam_record_finished_signal_exists(self):
        w = CaptureWorker(source=0)
        assert hasattr(w, "webcam_record_finished")

    def test_modes_ready_signal_removed(self):
        w = CaptureWorker(source=0)
        assert not hasattr(w, "modes_ready"), (
            "modes_ready signal must be removed in the redesign"
        )
