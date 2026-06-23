"""
Extract the audio track from a video file into a temporary OGG.

Used when the capture source is a pre-recorded video.

Prefers the ffmpeg binary bundled with ``imageio-ffmpeg`` (pip-installable,
cross-platform); falls back to a system ``ffmpeg`` on PATH.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

from PySide6.QtCore import QThread, Signal


class AudioExtractWorker(QThread):
    """Run extract_audio() in a background thread to avoid blocking the UI."""

    # Emits the extracted audio path on success, or empty string on failure.
    finished = Signal(str)

    def __init__(self, video_path: str) -> None:
        super().__init__()
        self._video_path = video_path

    def run(self) -> None:
        result = extract_audio(self._video_path)
        self.finished.emit(result or "")


# Explicit ffmpeg locations checked first (before imageio-ffmpeg / PATH).
# Override at runtime with the SYNCLIP_FFMPEG environment variable.
_HARDCODED_FFMPEG = [r"C:\Program Files\mpv\ffmpeg.exe"]


def ffmpeg_exe() -> str | None:
    """Return a path to an ffmpeg executable, or None if none is available."""
    # 1. Environment override.
    env = os.environ.get("SYNCLIP_FFMPEG")
    if env and os.path.isfile(env):
        print(f"[video_audio] using ffmpeg from SYNCLIP_FFMPEG: {env}")
        return env

    # 2. Hardcoded known locations.
    for path in _HARDCODED_FFMPEG:
        if os.path.isfile(path):
            print(f"[video_audio] using hardcoded ffmpeg: {path}")
            return path

    # 3. imageio-ffmpeg bundled binary.
    try:
        import imageio_ffmpeg  # type: ignore[import]
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[video_audio] using imageio-ffmpeg binary: {exe}")
        return exe
    except Exception:
        pass

    # 4. ffmpeg on PATH.
    exe = shutil.which("ffmpeg")
    if exe:
        print(f"[video_audio] using ffmpeg from PATH: {exe}")
    else:
        print(
            "[video_audio] no ffmpeg found (checked SYNCLIP_FFMPEG, "
            f"{_HARDCODED_FFMPEG}, imageio-ffmpeg, PATH)"
        )
    return exe


def _video_audio_path(video_path: str) -> str:
    """Return the deterministic OGG path for *video_path* based on its SHA-256."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    name = f"{stem}_audio_{h.hexdigest()[:8]}.ogg"
    return os.path.join(os.path.dirname(os.path.abspath(video_path)), name)


def extract_audio(video_path: str) -> str | None:
    """Extract *video_path*'s audio track into a 44.1 kHz stereo OGG next to the video.

    The output filename encodes a SHA-256 hash of the video file so the same
    clip always maps to the same OGG.  If the file already exists it is returned
    immediately without re-running ffmpeg.

    OGG/Vorbis is used (not WAV) because pygame's seek (play(start=...)) only
    works for OGG/MP3 - a WAV would break timeline scrubbing and pause-resume.
    Returns the path, or None if ffmpeg is unavailable or the video has no audio.
    """
    out_path = _video_audio_path(video_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 256:
        print(f"[video_audio] reusing cached audio: {out_path}")
        return out_path

    exe = ffmpeg_exe()
    if exe is None:
        return None

    cmd = [
        exe, "-y",
        "-i", video_path,
        "-vn",                      # drop video
        "-acodec", "libvorbis",     # seekable in pygame.mixer.music
        "-q:a", "5",
        "-ar", "44100",
        "-ac", "2",
        out_path,
    ]
    print(f"[video_audio] extracting audio: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,  # don't let a malformed input hang the worker forever
        )
    except subprocess.TimeoutExpired:
        print("[video_audio] ffmpeg timed out after 120s - aborting extraction")
        _safe_remove(out_path)
        return None
    except Exception as exc:
        print(f"[video_audio] failed to launch ffmpeg: {exc}")
        _safe_remove(out_path)
        return None

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        # Show the tail of ffmpeg's output, which holds the actual error.
        tail = "\n".join(stderr.splitlines()[-8:])
        print(f"[video_audio] ffmpeg exited with code {proc.returncode}:\n{tail}")
        _safe_remove(out_path)
        return None

    # An empty/headers-only OGG is tiny; a few hundred bytes means real audio.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 256:
        print(f"[video_audio] extracted {os.path.getsize(out_path)} bytes -> {out_path}")
        return out_path

    print("[video_audio] ffmpeg produced no audio (video may have no audio track)")
    _safe_remove(out_path)
    return None


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class WebcamMuxWorker(QThread):
    """Re-time a raw webcam recording and mux in the looping audio (background)."""

    # Emits the finalised video path on success, or empty string on failure.
    finished = Signal(str)

    def __init__(self, video_path: str, actual_fps: float, declared_fps: float,
                 audio_path: str | None, audio_start_ms: float,
                 out_path: str) -> None:
        super().__init__()
        self._video_path = video_path
        self._actual_fps = actual_fps
        self._declared_fps = declared_fps
        self._audio_path = audio_path
        self._audio_start_ms = audio_start_ms
        self._out_path = out_path

    def run(self) -> None:
        result = mux_webcam_recording(
            self._video_path, self._actual_fps, self._declared_fps,
            self._audio_path, self._audio_start_ms, self._out_path,
        )
        self.finished.emit(result or "")


def mux_webcam_recording(
    video_path: str,
    actual_fps: float,
    declared_fps: float,
    audio_path: str | None,
    audio_start_ms: float,
    out_path: str,
) -> str | None:
    """Re-time *video_path* to its real duration and mux in *audio_path*.

    The raw recording was written by OpenCV's VideoWriter at a fixed
    *declared_fps*, but frames actually arrived at *actual_fps*. We correct the
    playback speed with a setpts filter (ratio declared/actual) so the clip's
    duration matches real wall-clock time, then overlay the audio that was
    playing, seeked to *audio_start_ms* (where playback was when recording
    began) and looped so it covers the whole clip.

    Returns the output path, or None if ffmpeg is unavailable / fails.
    """
    exe = ffmpeg_exe()
    if exe is None:
        return None

    ratio = (declared_fps / actual_fps) if actual_fps > 0 else 1.0

    cmd = [exe, "-y", "-i", video_path]
    if audio_path:
        # Loop the audio and seek to the offset that was playing at record
        # start, so the muxed track lines up with the first recorded frame.
        cmd += [
            "-stream_loop", "-1",
            "-ss", f"{max(0.0, audio_start_ms) / 1000.0:.3f}",
            "-i", audio_path,
        ]

    cmd += ["-filter:v", f"setpts=PTS*{ratio:.6f}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio_path:
        cmd += ["-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd += [out_path]

    print(f"[video_audio] finalising webcam recording (actual {actual_fps:.1f} fps, "
          f"setpts x{ratio:.3f}): {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("[video_audio] ffmpeg timed out finalising webcam recording")
        return None
    except Exception as exc:
        print(f"[video_audio] failed to launch ffmpeg: {exc}")
        return None

    if proc.returncode != 0:
        tail = "\n".join(
            proc.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
        )
        print(f"[video_audio] ffmpeg mux exited with code {proc.returncode}:\n{tail}")
        return None

    if os.path.exists(out_path) and os.path.getsize(out_path) > 256:
        return out_path
    return None
