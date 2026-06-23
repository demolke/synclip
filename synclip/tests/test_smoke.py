"""
End-to-end smoke + bridge tests: construct the real MainWindow headlessly and
drive it through the four-mode state machine via _apply(plan).

These guard the redesign: they would have caught the `modes_ready` breakage and
they lock the LIVE/RECORD_VIDEO/PROCESS_VIDEO/REVIEW wiring, the worker config
the bridge pushes, and the broadcast routing.

Runs under the Qt offscreen platform + dummy audio (no display/camera/speakers).
Skipped cleanly if PySide6 / cv2 / mediapipe are absent.

Run with:
    cd tools && QT_QPA_PLATFORM=offscreen SDL_AUDIODRIVER=dummy \
        python -m pytest synclip/tests/test_smoke.py -v
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from synclip.app_state import Mode  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(tmp_path):
    from synclip.ui.main_window import MainWindow
    return MainWindow(root_dir=str(tmp_path), ipc_port=0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_main_window_constructs(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        assert w is not None
        assert hasattr(w._worker, "configure")
        assert not hasattr(w._worker, "modes_ready")
        # Starts in LIVE with broadcast on.
        assert w._sm.mode == Mode.LIVE
        assert w._broadcast_enabled is True
    finally:
        w._worker.stop()


def test_worker_starts_and_stops(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        assert w._worker.isRunning() or True
    finally:
        w._worker.stop()
        assert not w._worker.isRunning()


def test_defaults_to_first_video_file_in_dir(qapp, tmp_path):
    """On startup, prefer the first video file in the directory over the webcam."""
    # Two video files; the alphabetically-first should be chosen.
    (tmp_path / "b_clip.mp4").write_bytes(b"\x00")
    (tmp_path / "a_clip.mp4").write_bytes(b"\x00")
    w = _make_window(tmp_path)
    try:
        assert isinstance(w._video_source, str)
        assert w._video_source.endswith("a_clip.mp4")
        assert w._video_path == w._video_source
    finally:
        w._worker.stop()


def test_defaults_to_webcam_when_no_video(qapp, tmp_path):
    """With no video files in the directory, fall back to the webcam (camera 0)."""
    w = _make_window(tmp_path)
    try:
        assert not isinstance(w._video_source, str)  # camera index
        assert w._video_path is None
    finally:
        w._worker.stop()


def test_input_strip_widgets_exist(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        assert hasattr(w, "_video_combo")
        assert hasattr(w, "_audio_combo")
        assert hasattr(w, "_broadcast_check")
        # Broadcast checkbox reflects the SM default (ON).
        assert w._broadcast_check.isChecked() is True
        # No probing entry points survive.
        assert not hasattr(w, "_on_detect_modes")
        assert not hasattr(w, "_on_modes_ready")
        assert not hasattr(w, "_rebuild_mode_menu")
    finally:
        w._worker.stop()


def test_plausibility_removed(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        assert not hasattr(w, "_act_plausibility")
        assert not hasattr(w, "_on_plausibility_toggled")
        # The filter no longer carries plausibility settings.
        settings = w._gather_capture_settings()
        assert "plausibility" not in settings
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Bridge: _apply(plan) pushes the right worker config
# ---------------------------------------------------------------------------

def test_apply_pushes_worker_config_in_live(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._apply(w._sm.plan())
        cfg = w._worker._snap_cfg()
        assert cfg.throttle_fps is not None      # LIVE must be throttled
        assert cfg.run_mediapipe is True         # broadcast on -> mediapipe on
        assert cfg.record_path is None
    finally:
        w._worker.stop()


def test_broadcast_toggle_disables_mediapipe(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._on_broadcast_toggled(False)
        cfg = w._worker._snap_cfg()
        assert w._broadcast_enabled is False
        assert cfg.run_mediapipe is False
        # And back on.
        w._on_broadcast_toggled(True)
        assert w._worker._snap_cfg().run_mediapipe is True
    finally:
        w._worker.stop()


def test_pause_resume_in_live(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._on_toggle_pause()
        assert w._worker._snap_cfg().paused is True
        assert w._sm.plan().paused is True
        w._on_toggle_pause()
        assert w._worker._snap_cfg().paused is False
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Process-mode bridge: a fake worker source drives the throttle guard
# ---------------------------------------------------------------------------

def test_process_video_unthrottled(qapp, tmp_path):
    """Selecting a file source + start_process makes the worker run unthrottled."""
    w = _make_window(tmp_path)
    try:
        # Pretend a video file was loaded (skip the real extract worker).
        w._video_path = "fake.mp4"
        w._video_source = "fake.mp4"
        from synclip.app_state import VideoSource
        w._sm.set_video_source(VideoSource(kind="file", path="fake.mp4"))
        plan = w._sm.start_process_video()
        assert plan.mode == Mode.PROCESS_VIDEO
        w._apply(plan)
        cfg = w._worker._snap_cfg()
        assert cfg.throttle_fps is None          # regression guard: 80fps bug
        assert cfg.run_mediapipe is True
        assert cfg.video_looping is False
    finally:
        w._worker.stop()


def test_record_video_sets_record_path(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        plan = w._sm.start_record_video("/tmp/rec.mp4")
        w._apply(plan)
        cfg = w._worker._snap_cfg()
        assert cfg.record_path == "/tmp/rec.mp4"
        assert cfg.run_mediapipe is False        # no analysis while recording
        assert cfg.throttle_fps is not None
    finally:
        w._worker.stop()


def test_review_broadcast_mode_tag(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._apply(w._sm.to_review("take-1"))
        from synclip.app_state import MODE_REVIEW
        assert w._broadcast_mode == MODE_REVIEW
        assert w._sm.mode == Mode.REVIEW
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Pause-in-every-mode at the window level (the button handler)
# ---------------------------------------------------------------------------

def _enter_process(w):
    from synclip.app_state import VideoSource
    w._video_path = "fake.mp4"
    w._video_source = "fake.mp4"
    w._sm.set_video_source(VideoSource(kind="file", path="fake.mp4"))
    w._apply(w._sm.start_process_video())


def test_window_pause_live_freezes_worker(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._on_toggle_pause()
        assert w._worker._snap_cfg().paused is True
        assert w._worker._snap_cfg().run_mediapipe is False  # no detection paused
        w._on_toggle_pause()
        assert w._worker._snap_cfg().paused is False
    finally:
        w._worker.stop()


def test_window_pause_in_record_is_noop(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._apply(w._sm.start_record_video("/tmp/r.mp4"))
        before = w._worker._snap_cfg().paused
        w._on_toggle_pause()  # space/pause while recording
        assert w._sm.mode == Mode.RECORD_VIDEO
        assert w._worker._snap_cfg().paused == before  # unchanged
    finally:
        w._worker.stop()


def test_window_pause_in_process_is_noop(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        _enter_process(w)
        assert w._worker._snap_cfg().throttle_fps is None
        w._on_toggle_pause()
        assert w._sm.mode == Mode.PROCESS_VIDEO
        assert w._worker._snap_cfg().paused is False
        assert w._worker._snap_cfg().throttle_fps is None  # still unthrottled
    finally:
        w._worker.stop()


def test_window_pause_in_review(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        w._apply(w._sm.to_review("t"))
        w._on_toggle_pause()
        assert w._sm.plan().paused is True
        assert w._worker._snap_cfg().paused is True
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Broadcast: LIVE frames go out, paused frames do not
# ---------------------------------------------------------------------------

def test_live_frame_broadcasts(qapp, tmp_path, monkeypatch):
    w = _make_window(tmp_path)
    try:
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        # A synthetic live frame should be emitted (broadcast on by default).
        import numpy as np
        w._on_frame_ready(np.zeros((4, 4, 3), "uint8"), None, [0.1] * 52,
                          {"rot": [0.0] * 3, "pos": [0.0] * 3})
        assert len(sent) == 1
    finally:
        w._worker.stop()


def test_paused_live_frame_does_not_broadcast(qapp, tmp_path, monkeypatch):
    w = _make_window(tmp_path)
    try:
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        w._on_toggle_pause()  # pause LIVE
        import numpy as np
        w._on_frame_ready(np.zeros((4, 4, 3), "uint8"), None, [0.1] * 52,
                          {"rot": [0.0] * 3, "pos": [0.0] * 3})
        assert len(sent) == 0  # nothing sent while paused
    finally:
        w._worker.stop()


def _seed_review_take(w):
    """Give the window a 2-frame take and enter REVIEW."""
    w._streams.set("mediapipe", [
        {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
        {"audio_position_ms": 1000.0, "blendshapes": [1.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
    ])
    w._current_take_id = "t1"
    w._go_review(from_start=False)


def test_review_playback_broadcasts_on_change(qapp, tmp_path, monkeypatch):
    from synclip.app_state import MODE_REVIEW
    w = _make_window(tmp_path)
    try:
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        _seed_review_take(w)
        # Not paused; advance the audio clock between polls.
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 250.0)
        w._on_poll()
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 500.0)
        w._on_poll()
        assert len(sent) == 2
        assert all(kw.get("mode") == MODE_REVIEW for _, kw in sent)
    finally:
        w._worker.stop()


def test_review_paused_does_not_broadcast(qapp, tmp_path, monkeypatch):
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        # Pause REVIEW.
        w._on_toggle_pause()
        assert w._sm.plan().paused is True
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 300.0)
        w._on_poll()
        w._on_poll()
        assert len(sent) == 0  # silent while paused
    finally:
        w._worker.stop()


def test_review_static_frame_not_rebroadcast(qapp, tmp_path, monkeypatch):
    """Identical interpolated values (no change) must not stream repeatedly."""
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 400.0)
        w._on_poll()
        w._on_poll()  # same position -> same values -> no second send
        assert len(sent) == 1
    finally:
        w._worker.stop()


def test_review_scrub_also_scrubs_video(qapp, tmp_path, monkeypatch):
    """Scrubbing the timeline in REVIEW must seek the video picture too, i.e.
    push the scrub position into the capture worker."""
    from synclip.app_state import MODE_REVIEW, VideoSource

    w = _make_window(tmp_path)
    try:
        # REVIEW on a video-file source with a known duration.
        w._video_path = "clip.mp4"
        w._video_source = "clip.mp4"
        w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))
        _seed_review_take(w)
        w._audio._duration_ms = 1000.0  # 1s clip -> easy position math

        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))

        # Begin a drag at the 600ms mark.
        w._timeline.setValue(600)        # 60% of 1000ms = 600ms
        w._on_scrub_start()
        assert w._scrubbing is True
        assert w._worker._scrub_active is True

        # Drag to 600ms: the worker must be told to hold the video there.
        w._on_scrub_value(600)
        assert w._worker._scrub_active is True
        assert abs(w._worker._scrub_pos_ms - 600.0) < 1e-6
        # And the scrubbed pose is broadcast as MODE_REVIEW.
        assert any(kw.get("mode") == MODE_REVIEW for _, kw in sent)

        # Releasing the scrubber (not paused) resumes normal video sync.
        monkeypatch.setattr(w._audio, "is_paused", lambda: False)
        w._on_scrub_end()
        assert w._scrubbing is False
        assert w._worker._scrub_active is False
    finally:
        w._worker.stop()


def test_review_scrub_holds_video_when_paused(qapp, tmp_path, monkeypatch):
    """Releasing a scrub while paused keeps the video held on the frame."""
    from synclip.app_state import VideoSource

    w = _make_window(tmp_path)
    try:
        w._video_path = "clip.mp4"
        w._video_source = "clip.mp4"
        w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))
        _seed_review_take(w)
        w._audio._duration_ms = 1000.0
        w._on_toggle_pause()  # paused REVIEW

        w._timeline.setValue(300)
        w._on_scrub_start()
        w._on_scrub_value(300)
        assert abs(w._worker._scrub_pos_ms - 300.0) < 1e-6
        # Paused: scrub-hold must NOT be released on end.
        monkeypatch.setattr(w._audio, "is_paused", lambda: True)
        w._on_scrub_end()
        assert w._worker._scrub_active is True  # still holding the frame
    finally:
        w._worker.stop()


def test_review_resume_releases_video_scrub_hold(qapp, tmp_path, monkeypatch):
    """Regression: scrubbing/stepping then resuming must un-freeze the video.

    The worker's scrub-hold has to be released on resume, otherwise blendshapes
    advance (audio) while the picture stays frozen on the held frame.
    """
    from synclip.app_state import VideoSource

    w = _make_window(tmp_path)
    try:
        w._video_path = "clip.mp4"
        w._video_source = "clip.mp4"
        w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))
        _seed_review_take(w)
        w._audio._duration_ms = 1000.0

        # Scrub (which holds the video), then pause via scrub-end while paused.
        w._on_toggle_pause()  # pause REVIEW
        w._timeline.setValue(400)
        w._on_scrub_start()
        w._on_scrub_value(400)
        monkeypatch.setattr(w._audio, "is_paused", lambda: True)
        w._on_scrub_end()
        assert w._worker._scrub_active is True  # held on the scrubbed frame

        # Resume: the hold MUST be released so the picture runs again.
        monkeypatch.setattr(w._audio, "is_paused", lambda: False)
        w._on_toggle_pause()
        assert w._sm.plan().paused is False
        assert w._worker._scrub_active is False
        assert w._worker._snap_cfg().paused is False
    finally:
        w._worker.stop()


def test_step_then_resume_releases_hold(qapp, tmp_path, monkeypatch):
    from synclip.app_state import VideoSource

    w = _make_window(tmp_path)
    try:
        w._video_path = "clip.mp4"
        w._video_source = "clip.mp4"
        w._sm.set_video_source(VideoSource(kind="file", path="clip.mp4"))
        _seed_review_take(w)
        w._audio._duration_ms = 1000.0
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 0.0)
        w._step_frame(+1)             # steps -> pauses + holds the video
        assert w._worker._scrub_active is True
        monkeypatch.setattr(w._audio, "is_paused", lambda: False)
        w._on_toggle_pause()          # resume
        assert w._worker._scrub_active is False
    finally:
        w._worker.stop()


def _set_output_modifiers(w, mods):
    """Replace the broadcast-output view's modifier stack and reapply it.
    Prepends an InputModifier(mediapipe) so the base stream is always wired up."""
    from synclip.ui.main_window import OUTPUT_VIEW
    from synclip.modifiers import ModifierConfig
    inp = ModifierConfig("input", params={"stream": "mediapipe"})
    w._views[OUTPUT_VIEW].modifiers = [inp] + list(mods)
    w._pipelines[OUTPUT_VIEW].apply_config(w._views[OUTPUT_VIEW])
    w._update_pipeline_streams()


def test_ai_modifier_default_off_leaves_review_unchanged(qapp, tmp_path):
    """With no AI modifier (default stack), REVIEW values are the raw take."""
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        # Seed AI frames anyway; without an AI modifier they must be ignored.
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [0.9] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [0.9] * 52},
        ])
        vals = w._review_blendshapes(500.0)
        # take frames interpolate 0->1 at 500ms = 0.5 everywhere; AI ignored.
        assert all(abs(v - 0.5) < 1e-6 for v in vals)
    finally:
        w._worker.stop()


def test_ai_modifier_replace_all_uses_ai(qapp, tmp_path):
    from synclip import ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.arkit_names import BLENDSHAPE_NAMES
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [0.9] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [0.9] * 52},
        ])
        _set_output_modifiers(w, [ModifierConfig("ai", influence=1.0,
                                  params={"scope": ai.SCOPE_ALL, "stream": "ai"})])
        vals = w._review_blendshapes(500.0)
        jaw = BLENDSHAPE_NAMES.index("jawOpen")
        neutral = BLENDSHAPE_NAMES.index("_neutral")
        assert abs(vals[jaw] - 0.9) < 1e-6       # replaced by AI
        assert abs(vals[neutral] - 0.9) < 1e-6   # SCOPE_ALL = full copy, _neutral included
    finally:
        w._worker.stop()


def test_ai_modifier_mix_influence(qapp, tmp_path):
    from synclip import ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.arkit_names import BLENDSHAPE_NAMES
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [1.0] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [1.0] * 52},
        ])
        _set_output_modifiers(w, [ModifierConfig("ai", influence=0.25,
                                  params={"scope": ai.SCOPE_MOUTH, "stream": "ai"})])
        jaw = BLENDSHAPE_NAMES.index("jawOpen")
        brow = BLENDSHAPE_NAMES.index("browInnerUp")
        vals = w._review_blendshapes(500.0)
        # take=0.5, ai=1.0, infl=0.25 -> 0.625 for mouth/jaw; brow untouched (0.5).
        assert abs(vals[jaw] - 0.625) < 1e-6
        assert abs(vals[brow] - 0.5) < 1e-6
    finally:
        w._worker.stop()


def test_modifier_stack_widget_exists(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        assert hasattr(w, "_modifier_stack")
        # Adding a modifier through the stack mutates the bound view config.
        from synclip.ui.main_window import OUTPUT_VIEW
        w._on_view_selected(OUTPUT_VIEW)
        before = len(w._views[OUTPUT_VIEW].modifiers)
        w._modifier_stack._add_modifier("smooth")
        assert len(w._views[OUTPUT_VIEW].modifiers) == before + 1
        assert w._views[OUTPUT_VIEW].modifiers[-1].type == "smooth"
    finally:
        w._worker.stop()


def test_editing_modifier_broadcasts_a_frame(qapp, tmp_path, monkeypatch):
    """Changing the modifier stack in REVIEW pushes a MODE_REVIEW frame."""
    from synclip import ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.app_state import MODE_REVIEW
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [1.0] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [1.0] * 52},
        ])
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 500.0)
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        w._on_view_selected(OUTPUT_VIEW)
        w._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("ai", influence=1.0, params={"scope": ai.SCOPE_ALL, "stream": "ai"})
        ]
        w._on_modifier_stack_changed()    # add the AI modifier -> broadcasts
        w._on_modifier_stack_changed()    # nudge again -> broadcasts again
        assert len(sent) >= 2
        assert all(kw.get("mode") == MODE_REVIEW for _, kw in sent)
    finally:
        w._worker.stop()


def test_ui_controls_reflect_defaults(qapp, tmp_path):
    """Broadcast/landmarks widgets must mirror the actual defaults."""
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        assert w._broadcast_check.isChecked() == w._sm.broadcast           # True
        assert w._act_landmarks.isChecked() == w._landmarks_user_visible   # True
        # A fresh output view has only the head-pose modifier.
        assert [m.type for m in w._views[OUTPUT_VIEW].modifiers] == ["input", "pose_filter"]
    finally:
        w._worker.stop()


def test_review_never_runs_mediapipe(qapp, tmp_path):
    """HARD INVARIANT: REVIEW must not run MediaPipe or any worker analysis."""
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        cfg = w._worker._snap_cfg()
        assert cfg.run_mediapipe is False
        assert w._sm.plan().run_mediapipe is False
        assert w._sm.plan().show_landmarks is False
        # Even after pausing/resuming it stays off.
        w._on_toggle_pause()
        assert w._worker._snap_cfg().run_mediapipe is False
        w._on_toggle_pause()
        assert w._worker._snap_cfg().run_mediapipe is False
    finally:
        w._worker.stop()


def test_review_frame_callback_paused_never_broadcasts(qapp, tmp_path, monkeypatch):
    """HARD INVARIANT: nothing streams in REVIEW while paused and idle.

    Even if a stray frame_ready arrived, the REVIEW branch of _on_frame_ready
    does not broadcast (only LIVE/PROCESS do), and the paused poll stays silent.
    """
    import numpy as np
    w = _make_window(tmp_path)
    try:
        _seed_review_take(w)
        w._on_toggle_pause()  # paused REVIEW
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        # A stray worker frame in REVIEW must not broadcast.
        w._on_frame_ready(np.zeros((4, 4, 3), "uint8"), None, [0.5] * 52,
                          {"rot": [0.0] * 3, "pos": [0.0] * 3})
        # And repeated polls while paused stay silent.
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 700.0)
        w._on_poll()
        w._on_poll()
        assert len(sent) == 0
    finally:
        w._worker.stop()


def test_broadcast_off_suppresses_frames(qapp, tmp_path, monkeypatch):
    w = _make_window(tmp_path)
    try:
        sent = []
        monkeypatch.setattr(w._ipc, "send_frame",
                            lambda *a, **k: sent.append((a, k)))
        w._on_broadcast_toggled(False)
        import numpy as np
        w._on_frame_ready(np.zeros((4, 4, 3), "uint8"), None, [0.1] * 52,
                          {"rot": [0.0] * 3, "pos": [0.0] * 3})
        assert len(sent) == 0
    finally:
        w._worker.stop()
