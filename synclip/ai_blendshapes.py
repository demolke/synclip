"""
Optional second blendshape source: an audio-driven ML model (NeuroSync-style).

This module is import-safe even when the heavy model dependencies (torch and the
model's ``utils`` package + weights) are NOT installed -- ``is_available()`` just
returns False and the rest of the app falls back to the recorded take only.

When the deps ARE importable, ``generate_from_audio(path)`` runs the model and
returns a list of per-frame ARKit-52 blendshapes (mapped by name from the model's
own channel order). Head- and eye-rotation channels the model emits are dropped:
head pose always comes from the recorded take, never from this source.

The pure ``mix_blendshapes()`` helper (take vs. AI) has no model dependency and
is unit-tested directly.
"""

from __future__ import annotations

import os

from .arkit_names import BLENDSHAPE_NAMES

# Where to find the model weights (override with SYNCLIP_AI_MODEL).
MODEL_PATH = os.environ.get("SYNCLIP_AI_MODEL", "synclip/utils/model/model.pth")

# The model's output channel order (from the reference script). Names that match
# our ARKit-52 set are mapped; the rest (tongueOut, head*/eye* rotations) drop.
AI_SHAPE_NAMES: list[str] = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawRight", "jawLeft", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight", "mouthLeft",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
    "headYaw", "headPitch", "headRoll",
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]

# Default frame rate the model emits at (used to time the frames).
AI_FPS = 60.0

# ---- Mixer enums (plain strings so the UI/state can serialize them) ----
SCOPE_ALL = "all"
SCOPE_MOUTH = "mouth"
SCOPE_NONE = "none"

MODE_MIX = "mix"
MODE_REPLACE = "replace"

# Name -> ARKit index, and the model-index -> ARKit-index map (by name).
_ARKIT_IDX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
_AI_TO_ARKIT: dict[int, int] = {
    ai_i: _ARKIT_IDX[name]
    for ai_i, name in enumerate(AI_SHAPE_NAMES)
    if name in _ARKIT_IDX
}

# "Mouth only" scope: jaw/mouth articulation plus cheekPuff (tongueOut is not in
# our ARKit-52 set). Eyes, brows and nose are excluded.
_MOUTH_PREFIXES = ("mouth", "jaw")
_MOUTH_EXTRA = {"cheekPuff"}


def _scope_indices(scope: str) -> set[int]:
    if scope == SCOPE_ALL:
        # Everything except _neutral (a basis channel, not an expression).
        return {i for i, n in enumerate(BLENDSHAPE_NAMES) if n != "_neutral"}
    if scope == SCOPE_MOUTH:
        return {
            i for i, n in enumerate(BLENDSHAPE_NAMES)
            if n.startswith(_MOUTH_PREFIXES) or n in _MOUTH_EXTRA
        }
    return set()


def mix_blendshapes(
    take: list[float],
    ai: list[float] | None,
    scope: str,
    mode: str,
    influence: float,
) -> list[float]:
    """Blend recorded *take* values with the *ai* source per the mixer settings.

    - scope SCOPE_NONE (or no ai) -> the take is returned unchanged.
    - scope selects which channels the AI affects (all expression channels, or
      mouth/jaw only).
    - mode MODE_REPLACE swaps those channels for the AI values; MODE_MIX blends
      them by *influence* in [0, 1].
    Head pose is never part of this (it is carried separately from the take).
    """
    out = list(take)
    if ai is None or scope == SCOPE_NONE:
        return out
    w = 0.0 if influence < 0.0 else 1.0 if influence > 1.0 else influence
    for i in _scope_indices(scope):
        if i >= len(ai):
            continue
        if mode == MODE_REPLACE:
            out[i] = ai[i]
        else:
            out[i] = (1.0 - w) * take[i] + w * ai[i]
    return out


def _map_to_arkit(model_frame) -> list[float]:
    """Map one model output frame (AI_SHAPE_NAMES order) to ARKit-52 by name."""
    out = [0.0] * 52
    n = len(model_frame)
    for ai_i, ark_i in _AI_TO_ARKIT.items():
        if ai_i < n:
            out[ark_i] = float(model_frame[ai_i])
    return out


# ---------------------------------------------------------------------------
# Model availability + inference
# ---------------------------------------------------------------------------
#
# The feature is ALWAYS available: gated on ``from utils.config import config``.
# When that import (the real model) is present we run it; when it isn't, we emit
# dummy *neutral* frames (all-zero blendshapes) so the entire mixer/REVIEW path
# can be exercised and tested without the model.

_model_available: bool | None = None
_model = None
_device = None


def model_available() -> bool:
    """True iff the real model package (``utils.config``) is importable.

    Cached; never raises. When False, ``generate_from_audio`` falls back to
    dummy neutral frames.
    """
    global _model_available
    if _model_available is None:
        try:
            from utils.config import config  # noqa: F401
            _model_available = True
        except Exception:
            _model_available = False
    return _model_available


def is_available() -> bool:
    """The AI source is always usable (real model, or dummy-neutral fallback)."""
    return True


def _ensure_model():
    global _model, _device
    if _model is None:
        import torch
        from utils.config import config
        from utils.model.model import load_model
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = load_model(MODEL_PATH, config, _device)
    return _model, _device


def _timed(frame_blendshapes: list[list[float]], duration_ms: float | None,
           fps: float) -> list[dict]:
    """Attach an audio_position_ms to each mapped frame.

    The model may emit at a different rate than we assume, and a different number
    of frames than the take. So when the audio *duration* is known we space the
    frames evenly across it (n frames -> 0..duration), which keeps the AI
    timeline aligned to the audio regardless of the model's true fps. Only when
    the duration is unknown do we fall back to the nominal *fps*. Either way the
    caller interpolates by position, so lengths/fps need not match the take.
    """
    n = len(frame_blendshapes)
    if n == 0:
        return []
    if duration_ms and duration_ms > 0 and n > 1:
        step_ms = duration_ms / (n - 1)
    else:
        step_ms = 1000.0 / fps if fps > 0 else 1000.0 / AI_FPS
    return [{"audio_position_ms": i * step_ms, "blendshapes": bs}
            for i, bs in enumerate(frame_blendshapes)]


def _dummy_neutral_frames(duration_ms: float | None, fps: float) -> list[dict]:
    """Neutral (all-zero) frames spanning *duration_ms* at *fps*."""
    step_ms = 1000.0 / fps if fps > 0 else 1000.0 / AI_FPS
    total = duration_ms if (duration_ms and duration_ms > 0) else step_ms
    n = max(2, int(total / step_ms) + 1)
    return _timed([[0.0] * 52 for _ in range(n)], duration_ms, fps)


def generate_from_audio(
    audio_path: str,
    duration_ms: float | None = None,
    fps: float = AI_FPS,
) -> list[dict]:
    """Return timed ARKit-52 frames for *audio_path*.

    Each entry is {"audio_position_ms": float, "blendshapes": [52 floats]}.
    Uses the in-process model when ``model_available()``; otherwise emits dummy
    neutral frames. Never raises for a missing model.
    """
    if not model_available():
        return _dummy_neutral_frames(duration_ms, fps)

    try:
        import numpy as np
        from utils.config import config
        from utils.generate_face_shapes import generate_facial_data_from_bytes

        model, device = _ensure_model()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        data = generate_facial_data_from_bytes(audio_bytes, model, device, config)
        frames = data.tolist() if isinstance(data, np.ndarray) else data

        # Re-time across the known audio duration so an unknown model fps / frame
        # count still lines up with the take's audio clock.
        mapped = [_map_to_arkit(nums) for nums in frames]
        return _timed(mapped, duration_ms, fps)
    except Exception as exc:  # model present but inference failed -> neutral
        print(f"[ai_blendshapes] generation failed ({exc}); using neutral frames")
        return _dummy_neutral_frames(duration_ms, fps)
