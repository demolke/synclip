"""
T-01 through T-03b: AppStateMachine unit tests.

Pure Python, no Qt, no camera, no network. Run with:
    cd tools/synclip && python -m pytest tests/test_app_state.py -v
"""

from __future__ import annotations

import pytest

from ..app_state import (
    AppStateMachine,
    AudioKind,
    AudioSource,
    Mode,
    Plan,
    VideoSource,
    MODE_LIVE,
    MODE_REVIEW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _machine_with_file_video() -> AppStateMachine:
    sm = AppStateMachine()
    sm.set_video_source(VideoSource(kind="file", path="test.mp4"))
    return sm


def _machine_with_camera() -> AppStateMachine:
    sm = AppStateMachine()
    sm.set_video_source(VideoSource(kind="camera", camera_index=0))
    return sm


def _machine_with_file_audio(sm: AppStateMachine) -> AppStateMachine:
    sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="audio.ogg"))
    return sm


# ---------------------------------------------------------------------------
# T-01: Transition table - each transition yields the documented Plan fields
# ---------------------------------------------------------------------------

class TestT01TransitionTable:

    def test_initial_state_is_live(self):
        sm = AppStateMachine()
        assert sm.mode == Mode.LIVE

    def test_live_plan_defaults(self):
        sm = AppStateMachine()
        p = sm.plan()
        assert p.mode == Mode.LIVE
        assert p.throttle_fps is not None          # must be throttled in LIVE
        assert p.video_looping is True
        assert p.broadcast_mode == MODE_LIVE

    def test_live_broadcast_on_enables_mediapipe(self):
        sm = AppStateMachine()
        sm.set_broadcast(True)
        p = sm.plan()
        assert p.run_mediapipe is True
        assert p.show_landmarks is True
        assert p.broadcast is True

    def test_live_broadcast_off_disables_mediapipe(self):
        sm = AppStateMachine()
        sm.set_broadcast(False)
        p = sm.plan()
        assert p.run_mediapipe is False
        assert p.show_landmarks is False
        assert p.broadcast is False

    def test_record_video_plan(self):
        sm = _machine_with_camera()
        p = sm.start_record_video("/tmp/rec.avi")
        assert p.mode == Mode.RECORD_VIDEO
        assert p.run_mediapipe is False
        assert p.broadcast is False
        assert p.throttle_fps is not None
        assert p.record_path == "/tmp/rec.avi"
        assert p.show_landmarks is False
        assert p.video_looping is False

    def test_stop_record_video_returns_to_live(self):
        sm = _machine_with_camera()
        sm.start_record_video("/tmp/rec.avi")
        p = sm.stop_record_video()
        assert p.mode == Mode.LIVE
        assert p.record_path is None

    def test_process_video_plan(self):
        sm = _machine_with_file_video()
        p = sm.start_process_video()
        assert p.mode == Mode.PROCESS_VIDEO
        assert p.run_mediapipe is True
        assert p.throttle_fps is None              # T-03b: no throttle in PROCESS
        assert p.audio_playing is False
        assert p.broadcast is True
        assert p.broadcast_mode == MODE_LIVE
        assert p.video_looping is False

    def test_finish_process_video_moves_to_review(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        p = sm.finish_process_video("take-001")
        assert p.mode == Mode.REVIEW
        assert p.take_id == "take-001"
        assert p.run_mediapipe is False
        assert p.broadcast_mode == MODE_REVIEW

    def test_review_plan(self):
        sm = _machine_with_file_video()
        _machine_with_file_audio(sm)
        sm.start_process_video()
        p = sm.finish_process_video("take-001")
        assert p.mode == Mode.REVIEW
        assert p.video_looping is True
        assert p.throttle_fps is not None
        assert p.audio_looping is True

    def test_to_live_from_review(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        sm.finish_process_video("take-001")
        p = sm.to_live()
        assert p.mode == Mode.LIVE
        assert p.take_id is None

    def test_to_review_directly(self):
        sm = AppStateMachine()
        p = sm.to_review("take-xyz")
        assert p.mode == Mode.REVIEW
        assert p.take_id == "take-xyz"

    def test_start_process_from_review(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        sm.finish_process_video("take-001")
        p = sm.start_process_video()
        assert p.mode == Mode.PROCESS_VIDEO

    def test_broadcast_mode_tag_in_live(self):
        sm = AppStateMachine()
        assert sm.plan().broadcast_mode == MODE_LIVE

    def test_broadcast_mode_tag_in_review(self):
        sm = AppStateMachine()
        sm.to_review("t")
        assert sm.plan().broadcast_mode == MODE_REVIEW

    def test_broadcast_mode_tag_in_process(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        assert sm.plan().broadcast_mode == MODE_LIVE


# ---------------------------------------------------------------------------
# T-02: Illegal transitions are no-ops / safe
# ---------------------------------------------------------------------------

class TestT02IllegalTransitions:

    def test_pause_in_record_video_is_noop(self):
        sm = _machine_with_camera()
        sm.start_record_video("/tmp/r.avi")
        p = sm.pause()
        assert p.mode == Mode.RECORD_VIDEO
        assert p.paused is False

    def test_stop_record_from_live_is_noop(self):
        sm = AppStateMachine()
        p = sm.stop_record_video()
        assert p.mode == Mode.LIVE

    def test_process_from_record_is_noop(self):
        sm = _machine_with_camera()
        sm.start_record_video("/tmp/r.avi")
        p = sm.start_process_video()
        # video source is camera, not file -> noop
        assert p.mode == Mode.RECORD_VIDEO

    def test_process_from_camera_source_is_noop(self):
        sm = _machine_with_camera()
        p = sm.start_process_video()
        assert p.mode == Mode.LIVE  # didn't move

    def test_finish_process_from_live_is_noop(self):
        sm = AppStateMachine()
        p = sm.finish_process_video("x")
        assert p.mode == Mode.LIVE

    def test_begin_scrub_outside_review_is_noop(self):
        sm = AppStateMachine()
        p = sm.begin_scrub()
        assert p.mode == Mode.LIVE

    def test_resume_when_not_paused_is_noop(self):
        sm = AppStateMachine()
        p = sm.resume()
        assert p.paused is False
        assert p.mode == Mode.LIVE

    def test_pause_already_paused_is_noop(self):
        sm = AppStateMachine()
        sm.pause()
        p = sm.pause()
        assert p.paused is True

    def test_record_from_process_is_noop(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        p = sm.start_record_video("/tmp/r.avi")
        assert p.mode == Mode.PROCESS_VIDEO


# ---------------------------------------------------------------------------
# T-03: Audio plan correctness
# ---------------------------------------------------------------------------

class TestT03AudioPlan:

    def test_mic_in_live_no_playback(self):
        sm = _machine_with_camera()
        sm.set_audio_source(AudioSource(kind=AudioKind.MIC))
        p = sm.plan()
        assert p.audio_playing is False

    def test_none_audio_in_live_no_playback(self):
        sm = AppStateMachine()
        sm.set_audio_source(AudioSource(kind=AudioKind.NONE))
        p = sm.plan()
        assert p.audio_playing is False

    def test_file_audio_in_live_plays_and_loops(self):
        sm = AppStateMachine()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        p = sm.plan()
        assert p.audio_playing is True
        assert p.audio_looping is True

    def test_file_audio_paused_in_live_not_playing(self):
        sm = AppStateMachine()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.pause()
        p = sm.plan()
        assert p.audio_playing is False

    def test_file_audio_in_review_plays_and_loops(self):
        sm = _machine_with_file_video()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.start_process_video()
        sm.finish_process_video("t")
        p = sm.plan()
        assert p.audio_playing is True
        assert p.audio_looping is True

    def test_review_paused_audio_not_playing(self):
        sm = _machine_with_file_video()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.start_process_video()
        sm.finish_process_video("t")
        sm.pause()
        p = sm.plan()
        assert p.audio_playing is False

    def test_process_no_audio_playback(self):
        sm = _machine_with_file_video()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        p = sm.start_process_video()
        assert p.audio_playing is False

    def test_record_with_file_audio_plays_for_monitoring(self):
        sm = _machine_with_camera()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        p = sm.start_record_video("/tmp/r.avi")
        assert p.audio_playing is True

    def test_record_with_mic_audio_no_playback(self):
        sm = _machine_with_camera()
        sm.set_audio_source(AudioSource(kind=AudioKind.MIC))
        p = sm.start_record_video("/tmp/r.avi")
        assert p.audio_playing is False


# ---------------------------------------------------------------------------
# T-03b: Throttle regression guard (80 fps bug prevention)
# ---------------------------------------------------------------------------

class TestT03bThrottleGuard:

    def test_process_has_no_throttle(self):
        sm = _machine_with_file_video()
        p = sm.start_process_video()
        assert p.throttle_fps is None, (
            "PROCESS_VIDEO must not throttle - throttle_fps must be None"
        )

    def test_live_is_throttled(self):
        sm = AppStateMachine()
        p = sm.plan()
        assert p.throttle_fps is not None, (
            "LIVE must be throttled to prevent 80 fps spin (regression guard)"
        )

    def test_record_is_throttled(self):
        sm = _machine_with_camera()
        p = sm.start_record_video("/tmp/r.avi")
        assert p.throttle_fps is not None, (
            "RECORD_VIDEO must be throttled"
        )

    def test_review_is_throttled(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        p = sm.finish_process_video("t")
        assert p.throttle_fps is not None, (
            "REVIEW must be throttled"
        )

    def test_live_throttle_is_reasonable(self):
        sm = AppStateMachine()
        p = sm.plan()
        assert 1 <= p.throttle_fps <= 240, (
            f"LIVE throttle_fps={p.throttle_fps} is outside sane range [1, 240]"
        )


# ---------------------------------------------------------------------------
# T-03c: Broadcast default is ON
# ---------------------------------------------------------------------------

class TestBroadcastDefault:

    def test_broadcast_default_on(self):
        sm = AppStateMachine()
        assert sm.plan().broadcast is True

    def test_broadcast_default_enables_mediapipe(self):
        sm = AppStateMachine()
        assert sm.plan().run_mediapipe is True

    def test_broadcast_default_shows_landmarks(self):
        sm = AppStateMachine()
        assert sm.plan().show_landmarks is True


# ---------------------------------------------------------------------------
# T-03d: scrub in REVIEW stops audio
# ---------------------------------------------------------------------------

class TestScrubBehaviour:

    def _review_sm_with_audio(self) -> AppStateMachine:
        sm = _machine_with_file_video()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.start_process_video()
        sm.finish_process_video("t")
        return sm

    def test_scrub_stops_audio(self):
        sm = self._review_sm_with_audio()
        p = sm.begin_scrub()
        assert p.audio_playing is False

    def test_end_scrub_resumes_audio(self):
        sm = self._review_sm_with_audio()
        sm.begin_scrub()
        p = sm.end_scrub()
        assert p.audio_playing is True

    def test_scrub_outside_review_noop(self):
        sm = AppStateMachine()
        p = sm.begin_scrub()
        assert p.mode == Mode.LIVE

    def test_end_scrub_outside_review_noop(self):
        sm = AppStateMachine()
        p = sm.end_scrub()
        assert p.mode == Mode.LIVE


# ---------------------------------------------------------------------------
# Pause matrix: pressing pause in EVERY mode vs. documented expectations
# ---------------------------------------------------------------------------

class TestPauseMatrix:
    """For each mode, pause() must do exactly what the design says:
        LIVE          -> pausable (freezes preview + audio, mediapipe off)
        RECORD_VIDEO  -> NOT pausable (no-op; recording must not stall)
        PROCESS_VIDEO -> NOT pausable (no-op; analysis runs to completion)
        REVIEW        -> pausable (freezes take playback + audio)
    """

    def test_pause_in_live(self):
        sm = AppStateMachine()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        p = sm.pause()
        assert p.mode == Mode.LIVE
        assert p.paused is True
        assert p.run_mediapipe is False      # paused -> no detection
        assert p.audio_playing is False      # audio frozen

    def test_pause_in_record_video_noop(self):
        sm = _machine_with_camera()
        sm.start_record_video("/tmp/r.avi")
        p = sm.pause()
        assert p.mode == Mode.RECORD_VIDEO
        assert p.paused is False             # ignored

    def test_pause_in_process_video_noop(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        p = sm.pause()
        assert p.mode == Mode.PROCESS_VIDEO
        assert p.paused is False             # ignored
        assert p.throttle_fps is None        # still unthrottled

    def test_pause_in_review(self):
        sm = _machine_with_file_video()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.start_process_video()
        sm.finish_process_video("t")
        p = sm.pause()
        assert p.mode == Mode.REVIEW
        assert p.paused is True
        assert p.audio_playing is False

    def test_resume_restores_each_pausable_mode(self):
        # LIVE
        sm = AppStateMachine()
        sm.set_audio_source(AudioSource(kind=AudioKind.FILE, path="a.ogg"))
        sm.pause()
        p = sm.resume()
        assert p.paused is False
        assert p.run_mediapipe is True
        assert p.audio_playing is True


# ---------------------------------------------------------------------------
# Legal-transitions query
# ---------------------------------------------------------------------------

class TestLegalTransitions:

    def test_live_legal(self):
        sm = AppStateMachine()
        legal = sm.legal_transitions()
        assert "start_record_video" in legal
        assert "start_process_video" in legal
        assert "pause" in legal

    def test_record_legal(self):
        sm = _machine_with_camera()
        sm.start_record_video("/tmp/r.avi")
        assert sm.legal_transitions() == {"stop_record_video"}

    def test_process_legal(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        assert sm.legal_transitions() == {"finish_process_video"}

    def test_review_legal(self):
        sm = _machine_with_file_video()
        sm.start_process_video()
        sm.finish_process_video("t")
        legal = sm.legal_transitions()
        assert "to_live" in legal
        assert "pause" in legal
        assert "begin_scrub" in legal
