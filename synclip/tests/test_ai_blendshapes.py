"""
AI second-source mixer tests (ai_blendshapes module).

The real model deps are absent here, so the module runs its dummy-neutral path
-- which is exactly what lets us exercise the whole mixer end-to-end.
"""

from __future__ import annotations

from .. import ai_blendshapes as ai
from ..arkit_names import BLENDSHAPE_NAMES

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


# ---------------------------------------------------------------------------
# Availability + dummy generation
# ---------------------------------------------------------------------------

def test_module_imports_without_model():
    # is_available is always True (real model OR dummy neutral fallback).
    assert ai.is_available() is True
    # The real model package is not present in this environment.
    assert ai.model_available() is False


def test_generate_returns_neutral_frames_without_model(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    frames = ai.generate_from_audio(str(wav), duration_ms=1000.0, fps=60.0)
    assert len(frames) >= 2
    assert all(len(f["blendshapes"]) == 52 for f in frames)
    assert all(v == 0.0 for f in frames for v in f["blendshapes"])  # neutral


def test_frames_retimed_across_duration_regardless_of_fps(tmp_path):
    """Frames are spaced across the known duration, so an unknown model fps /
    frame count still lines up with the audio clock."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"\x00")
    frames = ai.generate_from_audio(str(wav), duration_ms=2000.0, fps=30.0)
    assert frames[0]["audio_position_ms"] == 0.0
    assert abs(frames[-1]["audio_position_ms"] - 2000.0) < 1e-6


def test_map_to_arkit_drops_head_and_tongue():
    # Build a model frame (AI order) with a 1.0 in head/tongue channels and a
    # known mouth channel; only the mouth one should survive into ARKit-52.
    n = len(ai.AI_SHAPE_NAMES)
    frame = [0.0] * n
    frame[ai.AI_SHAPE_NAMES.index("jawOpen")] = 0.7
    frame[ai.AI_SHAPE_NAMES.index("tongueOut")] = 1.0
    frame[ai.AI_SHAPE_NAMES.index("headYaw")] = 1.0
    frame[ai.AI_SHAPE_NAMES.index("leftEyeYaw")] = 1.0
    out = ai._map_to_arkit(frame)
    assert len(out) == 52
    assert out[_IDX["jawOpen"]] == 0.7
    # tongueOut / head* / eye-rotation are not in ARKit-52 -> dropped (no error).
    assert "tongueOut" not in _IDX


# ---------------------------------------------------------------------------
# mix_blendshapes
# ---------------------------------------------------------------------------

def _take():
    return [0.2] * 52


def _ai():
    return [0.8] * 52


def test_scope_none_returns_take_unchanged():
    out = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_NONE, ai.MODE_MIX, 1.0)
    assert out == _take()


def test_replace_all_swaps_expression_channels():
    out = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_ALL, ai.MODE_REPLACE, 1.0)
    # _neutral is excluded from "all"; everything else is replaced.
    assert out[_IDX["_neutral"]] == 0.2
    assert out[_IDX["jawOpen"]] == 0.8
    assert out[_IDX["browInnerUp"]] == 0.8


def test_mix_all_blends_by_influence():
    out = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_ALL, ai.MODE_MIX, 0.5)
    # 0.5*0.2 + 0.5*0.8 = 0.5
    assert abs(out[_IDX["jawOpen"]] - 0.5) < 1e-9
    assert out[_IDX["_neutral"]] == 0.2  # untouched


def test_mouth_scope_only_affects_mouth_and_jaw():
    out = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_MOUTH, ai.MODE_REPLACE, 1.0)
    assert out[_IDX["jawOpen"]] == 0.8        # jaw -> affected
    assert out[_IDX["mouthClose"]] == 0.8     # mouth -> affected
    assert out[_IDX["cheekPuff"]] == 0.8      # cheekPuff included
    assert out[_IDX["browInnerUp"]] == 0.2    # brow -> untouched
    assert out[_IDX["eyeBlinkLeft"]] == 0.2   # eye -> untouched
    assert out[_IDX["noseSneerLeft"]] == 0.2  # nose -> untouched


def test_influence_clamped():
    hi = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_ALL, ai.MODE_MIX, 5.0)
    assert abs(hi[_IDX["jawOpen"]] - 0.8) < 1e-9  # clamped to 1.0
    lo = ai.mix_blendshapes(_take(), _ai(), ai.SCOPE_ALL, ai.MODE_MIX, -5.0)
    assert abs(lo[_IDX["jawOpen"]] - 0.2) < 1e-9  # clamped to 0.0


def test_none_ai_returns_take():
    out = ai.mix_blendshapes(_take(), None, ai.SCOPE_ALL, ai.MODE_MIX, 1.0)
    assert out == _take()
