"""
pygame.mixer wrapper as a QObject with signals.

Does NOT call pygame.init() - only pygame.mixer is initialised.
The caller is responsible for creating a QTimer that calls poll() every 50 ms.
"""

from __future__ import annotations

try:
    import pygame
    import pygame.mixer
except ImportError as exc:
    raise RuntimeError(
        "pygame is required for audio playback.\n"
        "Install it with:  pip install pygame>=2.5"
    ) from exc

from PySide6.QtCore import QObject, Signal


class AudioPlayer(QObject):
    """Thin pygame.mixer.music wrapper with Qt signals."""

    # Emitted once when a play_once() finishes.
    playback_finished = Signal()

    # Emitted every poll() with current playback position in milliseconds.
    position_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        except pygame.error as exc:
            raise RuntimeError(
                "Could not initialise the audio device. Audio playback is "
                f"unavailable (is a sound device present?).\n{exc}"
            ) from exc

        self._loaded_path: str | None = None
        self._duration_ms: float | None = None
        self._loop_mode: bool = False
        self._playing_once: bool = False
        self._paused: bool = False
        # pygame's get_pos() counts from the last play() call and ignores the
        # start offset, so we track it ourselves to report seeked positions.
        self._start_offset_ms: float = 0.0
        # pygame's get_pos() keeps advancing (wall-clock) even while paused, so
        # we freeze the position during a pause and subtract the paused span on
        # resume to keep the reported position aligned with the actual audio.
        self._pause_position_ms: float = 0.0
        self._pause_raw_ms: float = 0.0
        self._pause_comp_ms: float = 0.0

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load *path* into pygame.mixer.music and probe its duration."""
        pygame.mixer.music.load(path)
        self._loaded_path = path
        self._loop_mode = False
        self._playing_once = False
        self._duration_ms = self._probe_duration(path)

    @staticmethod
    def _probe_duration(path: str) -> float | None:
        """Return audio duration in milliseconds, or None on failure."""
        try:
            import soundfile  # type: ignore[import]
            info = soundfile.info(path)
            return info.duration * 1000.0
        except Exception:
            pass
        return None

    @property
    def duration_ms(self) -> float | None:
        """Duration of the loaded file in milliseconds, or None if unknown."""
        return self._duration_ms

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play_loop(self) -> None:
        """Start looping playback (idle / rehearsal mode)."""
        if self._loaded_path is None:
            return
        self._loop_mode = True
        self._playing_once = False
        self._paused = False
        self._start_offset_ms = 0.0
        self._pause_comp_ms = 0.0
        pygame.mixer.music.play(loops=-1)

    def play_once(self) -> None:
        """Play once; ``playback_finished`` is emitted when done."""
        if self._loaded_path is None:
            return
        self._loop_mode = False
        self._playing_once = True
        self._paused = False
        self._start_offset_ms = 0.0
        self._pause_comp_ms = 0.0
        pygame.mixer.music.play(loops=0)

    def stop(self) -> None:
        """Stop playback immediately."""
        self._loop_mode = False
        self._playing_once = False
        self._paused = False
        self._start_offset_ms = 0.0
        self._pause_comp_ms = 0.0
        pygame.mixer.music.stop()

    def pause(self) -> None:
        """Pause playback, freezing the reported position."""
        if not self._paused and pygame.mixer.music.get_busy():
            # Freeze the current position and remember the raw clock so we can
            # subtract the paused span on resume.
            self._pause_position_ms = self.get_position_ms()
            self._pause_raw_ms = max(0.0, float(pygame.mixer.music.get_pos()))
            pygame.mixer.music.pause()
            self._paused = True

    def unpause(self) -> None:
        """Resume playback after a pause."""
        if self._paused:
            # get_pos() advanced (wall-clock) during the pause - subtract that
            # span so the position continues from where it was frozen.
            raw_now = max(0.0, float(pygame.mixer.music.get_pos()))
            self._pause_comp_ms += max(0.0, raw_now - self._pause_raw_ms)
            self._paused = False
            pygame.mixer.music.unpause()

    def is_paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> None:
        if self._paused:
            self.unpause()
        else:
            self.pause()

    def seek(self, pos_ms: float) -> None:
        """Jump to *pos_ms* in the loaded file (works while paused)."""
        if self._loaded_path is None:
            return
        pos_ms = max(0.0, pos_ms)
        loops = -1 if self._loop_mode else 0
        # play(start=...) restarts the stream at the requested offset.
        pygame.mixer.music.play(loops=loops, start=pos_ms / 1000.0)
        self._start_offset_ms = pos_ms
        self._pause_comp_ms = 0.0  # fresh play() - raw clock restarts at 0
        if self._paused:
            # Stay paused on the seeked frame.
            self._pause_position_ms = pos_ms
            self._pause_raw_ms = 0.0
            pygame.mixer.music.pause()

    def get_position_ms(self) -> float:
        """Return current playback position in milliseconds (0.0 if not playing).

        pygame's ``get_pos()`` counts from the call to ``play()`` and keeps
        growing across loops, so for looping playback we wrap it back into
        [0, duration) to get the position within the current loop.
        """
        # While paused the position is frozen (get_pos keeps ticking otherwise).
        if self._paused:
            pos = self._pause_position_ms
        else:
            if not pygame.mixer.music.get_busy():
                return 0.0
            raw = float(pygame.mixer.music.get_pos())
            if raw < 0.0:
                raw = 0.0
            pos = self._start_offset_ms + raw - self._pause_comp_ms
        # Clamp before the modulo: pause compensation can briefly push pos a
        # hair below zero, and Python's % would then wrap it to ~duration,
        # snapping the slider spuriously to the end of the clip.
        if pos < 0.0:
            pos = 0.0
        if self._loop_mode and self._duration_ms and self._duration_ms > 0:
            pos = pos % self._duration_ms
        return pos

    def is_playing(self) -> bool:
        """True while audio is actively playing (not paused / stopped)."""
        return bool(pygame.mixer.music.get_busy()) and not self._paused

    # ------------------------------------------------------------------
    # Polling (call from a QTimer every ~50 ms)
    # ------------------------------------------------------------------

    def poll(self) -> None:
        """Check playback state and emit signals.  Call this from a QTimer."""
        busy = pygame.mixer.music.get_busy()

        if busy:
            # Emit the compensated position (seek/pause/loop aware), not pygame's
            # raw get_pos(), so any consumer of this signal sees the same clock
            # the rest of the app uses.
            self.position_changed.emit(self.get_position_ms())
        else:
            # Track finished.
            if self._playing_once:
                self._playing_once = False
                self.playback_finished.emit()
