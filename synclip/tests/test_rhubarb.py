"""
Tests for the Rhubarb viseme source (rhubarb_lipsync).

The mapping/parsing helpers are pure and tested without the binary. A single
end-to-end test runs the real `rhubarb` binary and is skipped when it isn't on
PATH (mirroring the Godot/Blender harness tests).
"""

from __future__ import annotations

import os
import shutil

import pytest

from .. import rhubarb_lipsync as rb
from ..arkit_names import BLENDSHAPE_NAMES

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def test_every_shape_maps_to_52_channels():
    for shape in rb.VISEME_SHAPES:
        vec = rb.shape_to_blendshapes(shape)
        assert len(vec) == 52
        assert all(0.0 <= v <= 1.0 for v in vec)


def test_rest_shape_is_neutral():
    assert rb.shape_to_blendshapes("X") == [0.0] * 52


def test_unknown_shape_is_neutral():
    # Forward-compatible: an unmapped letter must not raise, just go neutral.
    assert rb.shape_to_blendshapes("Z") == [0.0] * 52


def test_wide_open_shape_drives_jaw_more_than_medium():
    d = rb.shape_to_blendshapes("D")
    c = rb.shape_to_blendshapes("C")
    assert d[_IDX["jawOpen"]] > c[_IDX["jawOpen"]] > 0.0


def test_pucker_shape_drives_pucker_not_jaw():
    f = rb.shape_to_blendshapes("F")
    assert f[_IDX["mouthPucker"]] > 0.4
    assert f[_IDX["jawOpen"]] < 0.2


def test_closed_shape_does_not_open_jaw():
    a = rb.shape_to_blendshapes("A")
    assert a[_IDX["jawOpen"]] == 0.0
    assert a[_IDX["mouthClose"]] > 0.0


def test_parse_mouth_cues_extracts_pairs():
    doc = {"mouthCues": [
        {"start": 0.0, "end": 0.3, "value": "X"},
        {"start": 0.3, "end": 0.5, "value": "D"},
    ]}
    assert rb.parse_mouth_cues(doc) == [(0.0, "X"), (0.3, "D")]


def test_parse_mouth_cues_skips_malformed():
    doc = {"mouthCues": [{"start": 0.0, "value": "X"}, {"bad": 1}, {"end": 0.5}]}
    assert rb.parse_mouth_cues(doc) == [(0.0, "X")]


def test_cues_to_frames_one_keyframe_per_cue_in_ms():
    cues = [(0.0, "X"), (0.33, "C"), (0.5, "D")]
    frames = rb.cues_to_frames(cues)
    assert [f["audio_position_ms"] for f in frames] == [0.0, 330.0, 500.0]
    assert frames[2]["blendshapes"][_IDX["jawOpen"]] > 0.5  # the "D" frame


def test_cues_to_frames_prepends_rest_when_first_cue_is_late():
    frames = rb.cues_to_frames([(0.2, "C")])
    assert frames[0]["audio_position_ms"] == 0.0
    assert frames[0]["blendshapes"] == [0.0] * 52  # rest
    assert frames[1]["audio_position_ms"] == 200.0


def test_cues_to_frames_appends_rest_at_duration():
    frames = rb.cues_to_frames([(0.0, "D")], duration_ms=1000.0)
    assert frames[-1]["audio_position_ms"] == 1000.0
    assert frames[-1]["blendshapes"] == [0.0] * 52


def test_cues_to_frames_empty_is_empty():
    assert rb.cues_to_frames([]) == []


def test_generate_returns_empty_without_binary(monkeypatch):
    monkeypatch.setattr(rb, "rhubarb_bin", lambda: None)
    assert rb.generate_from_audio("/no/such.wav") == []
    assert rb.is_available() is False


# ---------------------------------------------------------------------------
# End-to-end: run the real binary (skipped when rhubarb isn't installed)
# ---------------------------------------------------------------------------

def _rhubarb_test_wav() -> str | None:
    """A WAV shipped with the rhubarb distribution next to the binary, if any."""
    binary = rb.rhubarb_bin()
    if not binary:
        return None
    root = os.path.dirname(os.path.realpath(shutil.which(binary) or binary))
    res = os.path.join(root, "tests", "resources")
    if os.path.isdir(res):
        # Rhubarb only decodes uncompressed PCM/IEEE-float; some samples are
        # FLAC/ADPCM with a .wav extension, so prefer a plain int16 PCM file.
        names = [n for n in sorted(os.listdir(res)) if n.lower().endswith(".wav")]
        pcm = [n for n in names if "int16" in n and "flac" not in n.lower()]
        for name in (pcm + names):
            if "flac" not in name.lower():
                return os.path.join(res, name)
    return None


@pytest.mark.skipif(not rb.is_available(), reason="rhubarb binary not on PATH")
def test_generate_from_audio_end_to_end():
    wav = _rhubarb_test_wav()
    if wav is None:
        pytest.skip("no rhubarb sample wav available")
    frames = rb.generate_from_audio(wav, duration_ms=1000.0)
    assert isinstance(frames, list)
    assert frames, "rhubarb produced no frames"
    for f in frames:
        assert len(f["blendshapes"]) == 52
        assert "audio_position_ms" in f
    positions = [f["audio_position_ms"] for f in frames]
    assert positions == sorted(positions)
