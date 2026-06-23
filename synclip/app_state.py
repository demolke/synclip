"""
State machine for the synclip capture app.

Owns all mutable app state and returns an immutable Plan describing what every
subsystem should be doing. Nothing here imports Qt, OpenCV, or MediaPipe.

Typical usage (two-line pattern in MainWindow):
    plan = self._state.pause()
    self._apply(plan)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

# IPC mode constants (must match ipc_server.py / godot)
MODE_LIVE = 0xAF0002
MODE_REVIEW = 0xAF0003
MODE_EDIT = 0xAF0004

# Default FPS cap for non-processing modes
_DEFAULT_THROTTLE_FPS = 60


class Mode(Enum):
    LIVE = auto()
    RECORD_VIDEO = auto()
    PROCESS_VIDEO = auto()
    REVIEW = auto()


class AudioKind(Enum):
    NONE = auto()   # silence (e.g. webcam mic selected while in LIVE)
    FILE = auto()   # audio file or video's extracted track
    MIC = auto()    # live webcam microphone (recorded in RECORD, silent in LIVE)


@dataclass
class VideoSource:
    kind: Literal["camera", "file"]
    camera_index: int | None = None
    path: str | None = None


@dataclass
class AudioSource:
    kind: AudioKind = AudioKind.NONE
    path: str | None = None  # for FILE


@dataclass(frozen=True)
class Plan:
    """Immutable description of what every subsystem should be doing.

    MainWindow._apply() is the single place that reads this and pushes
    configuration into the worker, audio player, IPC, and UI widgets.
    """
    mode: Mode
    run_mediapipe: bool
    broadcast: bool
    broadcast_mode: int          # one of MODE_LIVE / MODE_REVIEW / MODE_EDIT
    audio_playing: bool
    audio_looping: bool
    video_looping: bool
    throttle_fps: int | None     # None => no sleep (PROCESS_VIDEO only)
    paused: bool
    show_landmarks: bool
    # Record: write frames to a file path (set only in RECORD_VIDEO)
    record_path: str | None = None
    # Take used for REVIEW playback
    take_id: str | None = None


class AppStateMachine:
    """Single owner of all app state.

    Every public method returns a Plan. Callers must pass that Plan to
    MainWindow._apply(); the machine itself never touches Qt or subsystems.
    """

    def __init__(self) -> None:
        self._mode: Mode = Mode.LIVE
        self._video: VideoSource = VideoSource(kind="camera", camera_index=0)
        self._audio: AudioSource = AudioSource(kind=AudioKind.NONE)
        self._broadcast: bool = True   # default ON per spec
        self._paused: bool = False
        self._scrubbing: bool = False
        self._take_id: str | None = None
        self._record_path: str | None = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def video_source(self) -> VideoSource:
        return self._video

    @property
    def audio_source(self) -> AudioSource:
        return self._audio

    @property
    def broadcast(self) -> bool:
        return self._broadcast

    def plan(self) -> Plan:
        """Return the Plan for the current state."""
        return self._build_plan()

    def legal_transitions(self) -> set[str]:
        """Names of transitions that are currently permitted."""
        m = self._mode
        base: set[str] = set()
        if m == Mode.LIVE:
            base = {"start_record_video", "start_process_video",
                    "pause", "resume", "set_video_source", "set_audio_source",
                    "set_broadcast"}
        elif m == Mode.RECORD_VIDEO:
            base = {"stop_record_video"}
        elif m == Mode.PROCESS_VIDEO:
            base = {"finish_process_video"}
        elif m == Mode.REVIEW:
            base = {"to_live", "start_process_video",
                    "pause", "resume", "begin_scrub", "end_scrub",
                    "set_video_source", "set_audio_source", "set_broadcast"}
        return base

    # ------------------------------------------------------------------
    # Source selection (legal in LIVE and REVIEW)
    # ------------------------------------------------------------------

    def set_video_source(self, vs: VideoSource) -> Plan:
        self._video = vs
        # Switching source implies leaving REVIEW if we're there
        if self._mode == Mode.REVIEW:
            self._mode = Mode.LIVE
            self._take_id = None
            self._paused = False
        return self._build_plan()

    def set_audio_source(self, a: AudioSource) -> Plan:
        self._audio = a
        return self._build_plan()

    def set_broadcast(self, on: bool) -> Plan:
        self._broadcast = on
        return self._build_plan()

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def to_live(self) -> Plan:
        """Return to LIVE from any mode."""
        self._mode = Mode.LIVE
        self._paused = False
        self._scrubbing = False
        self._record_path = None
        return self._build_plan()

    def start_record_video(self, record_path: str) -> Plan:
        """Begin recording webcam frames (LIVE -> RECORD_VIDEO)."""
        if self._mode != Mode.LIVE:
            return self._build_plan()
        self._mode = Mode.RECORD_VIDEO
        self._paused = False
        self._record_path = record_path
        return self._build_plan()

    def stop_record_video(self) -> Plan:
        """Stop recording (RECORD_VIDEO -> LIVE)."""
        if self._mode != Mode.RECORD_VIDEO:
            return self._build_plan()
        self._mode = Mode.LIVE
        self._record_path = None
        return self._build_plan()

    def start_process_video(self) -> Plan:
        """Begin offline analysis (LIVE or REVIEW -> PROCESS_VIDEO).

        Only legal when the video source is a file.
        """
        if self._mode not in (Mode.LIVE, Mode.REVIEW):
            return self._build_plan()
        if self._video.kind != "file":
            return self._build_plan()
        self._mode = Mode.PROCESS_VIDEO
        self._paused = False
        self._take_id = None
        return self._build_plan()

    def finish_process_video(self, take_id: str) -> Plan:
        """Processing done - store take and move to REVIEW."""
        if self._mode != Mode.PROCESS_VIDEO:
            return self._build_plan()
        self._mode = Mode.REVIEW
        self._take_id = take_id
        self._paused = False
        return self._build_plan()

    def to_review(self, take_id: str) -> Plan:
        """Jump directly to REVIEW with an existing take."""
        self._mode = Mode.REVIEW
        self._take_id = take_id
        self._paused = False
        self._scrubbing = False
        return self._build_plan()

    # ------------------------------------------------------------------
    # Pause / resume / scrub (self-transitions)
    # ------------------------------------------------------------------

    def pause(self) -> Plan:
        """Pause playback. Legal in LIVE and REVIEW."""
        if self._mode not in (Mode.LIVE, Mode.REVIEW):
            return self._build_plan()
        if self._paused:
            return self._build_plan()
        self._paused = True
        return self._build_plan()

    def resume(self) -> Plan:
        """Resume playback. Legal in LIVE and REVIEW."""
        if self._mode not in (Mode.LIVE, Mode.REVIEW):
            return self._build_plan()
        if not self._paused:
            return self._build_plan()
        self._paused = False
        return self._build_plan()

    def begin_scrub(self) -> Plan:
        """User started scrubbing. Legal in REVIEW."""
        if self._mode != Mode.REVIEW:
            return self._build_plan()
        self._scrubbing = True
        return self._build_plan()

    def end_scrub(self) -> Plan:
        """User released scrubber. Legal in REVIEW."""
        if self._mode != Mode.REVIEW:
            return self._build_plan()
        self._scrubbing = False
        return self._build_plan()

    # ------------------------------------------------------------------
    # Internal Plan builder
    # ------------------------------------------------------------------

    def _build_plan(self) -> Plan:
        m = self._mode
        p = self._paused
        audio_kind = self._audio.kind

        if m == Mode.LIVE:
            # MediaPipe runs only when broadcast is on
            run_mp = self._broadcast and not p
            # Audio plays if we have a FILE source and not paused
            # MIC in LIVE -> no playback (would cause feedback)
            audio_playing = (audio_kind == AudioKind.FILE) and not p
            return Plan(
                mode=m,
                run_mediapipe=run_mp,
                broadcast=self._broadcast,
                broadcast_mode=MODE_LIVE,
                audio_playing=audio_playing,
                audio_looping=True,
                video_looping=True,
                throttle_fps=_DEFAULT_THROTTLE_FPS,
                paused=p,
                show_landmarks=run_mp,
            )

        if m == Mode.RECORD_VIDEO:
            # No MediaPipe (we want full capture rate)
            # Audio: FILE plays for monitoring (not recorded by this path; muxed after)
            # MIC: silent here (captured separately by audio backend)
            audio_playing = audio_kind == AudioKind.FILE
            return Plan(
                mode=m,
                run_mediapipe=False,
                broadcast=False,
                broadcast_mode=MODE_LIVE,
                audio_playing=audio_playing,
                audio_looping=True,
                video_looping=False,
                throttle_fps=_DEFAULT_THROTTLE_FPS,
                paused=False,
                show_landmarks=False,
                record_path=self._record_path,
            )

        if m == Mode.PROCESS_VIDEO:
            # No throttle - run as fast as possible
            # No audio playback
            return Plan(
                mode=m,
                run_mediapipe=True,
                broadcast=True,    # stream to viewer as we process
                broadcast_mode=MODE_LIVE,
                audio_playing=False,
                audio_looping=False,
                video_looping=False,
                throttle_fps=None,  # no sleep
                paused=False,
                show_landmarks=True,
            )

        if m == Mode.REVIEW:
            # Take-driven; audio plays looping (same FILE source as live)
            audio_playing = (audio_kind == AudioKind.FILE) and not p and not self._scrubbing
            return Plan(
                mode=m,
                run_mediapipe=False,
                broadcast=True,
                broadcast_mode=MODE_REVIEW,
                audio_playing=audio_playing,
                audio_looping=True,
                video_looping=True,
                throttle_fps=_DEFAULT_THROTTLE_FPS,
                paused=p,
                show_landmarks=False,
                take_id=self._take_id,
            )

        # Unreachable - satisfy type checker
        raise RuntimeError(f"Unknown mode: {m}")
