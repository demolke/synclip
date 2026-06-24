"""
Rhubarb Lip Sync source: phoneme/viseme audio -> ARKit-52 blendshapes.

Rhubarb Lip Sync (https://github.com/DanielSWolf/rhubarb-lip-sync, MIT) is an
offline, CPU-only, deterministic command-line tool that turns a voice recording
into a sequence of mouth shapes (Preston-Blair style: A-H, plus X for rest).

This module runs the binary and maps each mouth shape to a set of ARKit-52
blendshape weights, so the result drops into the same stream + modifier pipeline
as the MediaPipe and AI sources (it becomes the ``rhubarb`` stream).

Import-safe and dependency-light: when the ``rhubarb`` binary isn't found,
``is_available()`` returns False and ``generate_from_audio()`` returns ``[]``.
The pure mapping/parsing helpers have no external dependency and are unit-tested
directly. Override the binary location with the ``RHUBARB`` env var.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from .arkit_names import BLENDSHAPE_NAMES

_ARKIT_IDX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}

# Rhubarb's mouth shapes mapped to ARKit-52 weights. Left/right channels are
# driven symmetrically. These are deliberately conservative, mouth/jaw-only
# poses (eyes/brows are left to the recorded take); downstream Curves/Smooth
# modifiers can refine them. See the Rhubarb docs for each shape's phonemes.
#
#   A  closed mouth for P/B/M            X  idle / rest (silence)
#   B  slightly open, clenched, "EE"     G  F/V (upper teeth on lower lip)
#   C  open, "EH"/"AE"                   H  long "L" (tongue raised)
#   D  wide open, "AA"
#   E  slightly rounded, "AO"/"ER"
#   F  puckered, "UW"/"OW"/"W"

_VISEME_POSES: dict[str, dict[str, float]] = {
    "X": {},  # rest -> neutral
    "A": {"mouthClose": 0.20, "mouthPressLeft": 0.20, "mouthPressRight": 0.20},
    "B": {"jawOpen": 0.12, "mouthStretchLeft": 0.30, "mouthStretchRight": 0.30,
          "mouthSmileLeft": 0.10, "mouthSmileRight": 0.10},
    "C": {"jawOpen": 0.32, "mouthStretchLeft": 0.18, "mouthStretchRight": 0.18},
    "D": {"jawOpen": 0.58, "mouthLowerDownLeft": 0.20, "mouthLowerDownRight": 0.20},
    "E": {"jawOpen": 0.28, "mouthFunnel": 0.30, "mouthPucker": 0.15},
    "F": {"jawOpen": 0.10, "mouthPucker": 0.60, "mouthFunnel": 0.30},
    "G": {"jawOpen": 0.08, "mouthRollLower": 0.35, "mouthShrugLower": 0.20,
          "mouthLowerDownLeft": 0.10, "mouthLowerDownRight": 0.10},
    "H": {"jawOpen": 0.30, "mouthShrugUpper": 0.20},
}

# All shapes Rhubarb may emit; the extended ones (G, H, X) are opt-in on the CLI.
VISEME_SHAPES = tuple("ABCDEFGHX")


def rhubarb_bin() -> str | None:
    """Path to the rhubarb binary (``RHUBARB`` env var first, then PATH)."""
    env = os.environ.get("RHUBARB")
    if env and (os.path.isfile(env) or shutil.which(env)):
        return env
    return shutil.which("rhubarb")


def is_available() -> bool:
    """True iff the rhubarb binary can be found."""
    return rhubarb_bin() is not None


def shape_to_blendshapes(shape: str) -> list[float]:
    """Return the 52-channel ARKit weight vector for a Rhubarb mouth shape."""
    out = [0.0] * 52
    for name, weight in _VISEME_POSES.get(shape, {}).items():
        idx = _ARKIT_IDX.get(name)
        if idx is not None:
            out[idx] = weight
    return out


def parse_mouth_cues(doc: dict) -> list[tuple[float, str]]:
    """Extract ``(start_seconds, shape)`` pairs from Rhubarb's JSON output."""
    cues = doc.get("mouthCues") or []
    out: list[tuple[float, str]] = []
    for cue in cues:
        try:
            out.append((float(cue["start"]), str(cue["value"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def cues_to_frames(
    cues: list[tuple[float, str]],
    duration_ms: float | None = None,
) -> list[dict]:
    """Turn ``(start_seconds, shape)`` cues into timed ARKit-52 frames.

    One keyframe is emitted at each cue's start (the pipeline interpolates
    linearly between successive shapes, giving smooth viseme morphing). A rest
    frame is prepended at t=0 when the first cue starts later, and appended at
    *duration_ms* so the mouth settles to neutral at the end.
    """
    frames: list[dict] = []
    if not cues:
        return frames

    if cues[0][0] > 0.0:
        frames.append({"audio_position_ms": 0.0,
                       "blendshapes": shape_to_blendshapes("X")})

    for start_s, shape in cues:
        frames.append({"audio_position_ms": start_s * 1000.0,
                       "blendshapes": shape_to_blendshapes(shape)})

    if duration_ms and duration_ms > frames[-1]["audio_position_ms"]:
        frames.append({"audio_position_ms": float(duration_ms),
                       "blendshapes": shape_to_blendshapes("X")})
    return frames


def generate_from_audio(
    audio_path: str,
    duration_ms: float | None = None,
    extended_shapes: str = "GHX",
    recognizer: str = "pocketSphinx",
    timeout: float = 600.0,
) -> list[dict]:
    """Run Rhubarb on *audio_path* and return timed ARKit-52 frames.

    Each entry is ``{"audio_position_ms": float, "blendshapes": [52 floats]}``,
    matching the other blendshape sources. Returns ``[]`` (never raises) when the
    binary is missing or the run fails, so the caller can treat the rhubarb track
    as optional.

    *recognizer* is ``pocketSphinx`` (English, phoneme-accurate) or ``phonetic``
    (language-independent). *extended_shapes* selects which of G/H/X are used.
    """
    binary = rhubarb_bin()
    if binary is None:
        return []
    cmd = [binary, "-f", "json", "-r", recognizer,
           "--extendedShapes", extended_shapes, audio_path]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[rhubarb] run failed ({exc}); skipping rhubarb track")
        return []
    if proc.returncode != 0:
        print(f"[rhubarb] exited {proc.returncode}: {proc.stderr[-300:]}")
        return []
    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[rhubarb] could not parse output ({exc}); skipping rhubarb track")
        return []
    frames = cues_to_frames(parse_mouth_cues(doc), duration_ms)
    print(f"[rhubarb] generated {len(frames)} viseme keyframes for {audio_path}")
    return frames
