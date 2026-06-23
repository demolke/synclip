"""
Tests for Stream, interp_blendshapes, and interp_head_pose in data.
"""

from __future__ import annotations

import pytest
from ..data import Stream, StreamStore, interp_blendshapes, interp_head_pose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(pos_ms: float, value: float = 0.0, with_pose: bool = False) -> dict:
    bs = [value] * 52
    f: dict = {"audio_position_ms": pos_ms, "blendshapes": bs}
    if with_pose:
        f["head_pose"] = {"rot": [value] * 3, "pos": [value] * 3}
    return f


# ---------------------------------------------------------------------------
# interp_blendshapes
# ---------------------------------------------------------------------------

def test_interp_empty_returns_none():
    assert interp_blendshapes([], 0.0) is None


def test_interp_exact_first():
    frames = [_frame(0.0, 0.5), _frame(100.0, 1.0)]
    result = interp_blendshapes(frames, 0.0)
    assert abs(result[0] - 0.5) < 1e-9


def test_interp_exact_last():
    frames = [_frame(0.0, 0.0), _frame(100.0, 1.0)]
    result = interp_blendshapes(frames, 100.0)
    assert abs(result[0] - 1.0) < 1e-9


def test_interp_before_start_clamps():
    frames = [_frame(50.0, 0.3), _frame(100.0, 0.9)]
    result = interp_blendshapes(frames, 0.0)
    assert abs(result[0] - 0.3) < 1e-9


def test_interp_after_end_clamps():
    frames = [_frame(0.0, 0.2), _frame(100.0, 0.8)]
    result = interp_blendshapes(frames, 200.0)
    assert abs(result[0] - 0.8) < 1e-9


def test_interp_midpoint():
    frames = [_frame(0.0, 0.0), _frame(100.0, 1.0)]
    result = interp_blendshapes(frames, 50.0)
    assert abs(result[0] - 0.5) < 1e-9


def test_interp_single_frame():
    frames = [_frame(0.0, 0.7)]
    assert abs(interp_blendshapes(frames, 999.0)[0] - 0.7) < 1e-9


def test_interp_uses_cached_positions():
    frames = [_frame(0.0, 0.0), _frame(100.0, 1.0)]
    positions = [0.0, 100.0]
    result = interp_blendshapes(frames, 25.0, positions)
    assert abs(result[0] - 0.25) < 1e-9


def test_interp_duplicate_times_returns_first():
    frames = [_frame(50.0, 0.1), _frame(50.0, 0.9)]
    result = interp_blendshapes(frames, 50.0)
    # Should not divide by zero and should return either boundary value cleanly.
    assert result is not None
    assert 0.0 <= result[0] <= 1.0


# ---------------------------------------------------------------------------
# interp_head_pose
# ---------------------------------------------------------------------------

def test_head_pose_interp_none_if_no_key():
    frames = [_frame(0.0), _frame(100.0)]
    assert interp_head_pose(frames, 50.0) is None


def test_head_pose_interp_midpoint():
    frames = [_frame(0.0, 0.0, with_pose=True), _frame(100.0, 1.0, with_pose=True)]
    result = interp_head_pose(frames, 50.0)
    assert result is not None
    assert abs(result["rot"][0] - 0.5) < 1e-9
    assert abs(result["pos"][1] - 0.5) < 1e-9


def test_head_pose_clamps_before_start():
    frames = [_frame(0.0, 0.4, with_pose=True), _frame(100.0, 0.9, with_pose=True)]
    result = interp_head_pose(frames, -10.0)
    assert abs(result["rot"][0] - 0.4) < 1e-9


def test_head_pose_clamps_after_end():
    frames = [_frame(0.0, 0.1, with_pose=True), _frame(100.0, 0.8, with_pose=True)]
    result = interp_head_pose(frames, 200.0)
    assert abs(result["rot"][0] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Stream dataclass
# ---------------------------------------------------------------------------

def test_stream_from_frames_extracts_positions():
    frames = [_frame(10.0), _frame(20.0), _frame(30.0)]
    s = Stream.from_frames(frames)
    assert s.positions == [10.0, 20.0, 30.0]
    assert len(s.frames) == 3


def test_stream_values_at_empty_returns_none():
    s = Stream()
    assert s.values_at(0.0) is None


def test_stream_values_at_interpolates():
    frames = [_frame(0.0, 0.0), _frame(100.0, 1.0)]
    s = Stream.from_frames(frames)
    result = s.values_at(50.0)
    assert result is not None
    assert abs(result[0] - 0.5) < 1e-9


def test_stream_head_pose_at_returns_none_if_no_pose():
    frames = [_frame(0.0), _frame(100.0)]
    s = Stream.from_frames(frames)
    assert s.head_pose_at(50.0) is None


def test_stream_head_pose_at_interpolates():
    frames = [_frame(0.0, 0.0, with_pose=True), _frame(100.0, 1.0, with_pose=True)]
    s = Stream.from_frames(frames)
    result = s.head_pose_at(50.0)
    assert result is not None
    assert abs(result["rot"][0] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# retarget._MOUTH_INDICES consistency with ai_blendshapes
# ---------------------------------------------------------------------------

def test_retarget_mouth_indices_match_ai_blendshapes_scope():
    from ..retarget import _MOUTH_INDICES
    from .. import ai_blendshapes
    expected = sorted(ai_blendshapes._scope_indices(ai_blendshapes.SCOPE_MOUTH))
    assert _MOUTH_INDICES == expected


def test_retarget_mouth_indices_exclude_eye_and_brow():
    from ..retarget import _MOUTH_INDICES
    from ..arkit_names import BLENDSHAPE_NAMES
    for idx in _MOUTH_INDICES:
        name = BLENDSHAPE_NAMES[idx]
        assert not name.startswith("eye"), f"eye channel {name!r} in _MOUTH_INDICES"
        assert not name.startswith("brow"), f"brow channel {name!r} in _MOUTH_INDICES"


def test_retarget_mouth_indices_include_jaw_and_mouth():
    from ..retarget import _MOUTH_INDICES
    from ..arkit_names import BLENDSHAPE_NAMES
    names = {BLENDSHAPE_NAMES[i] for i in _MOUTH_INDICES}
    assert any(n.startswith("jaw") for n in names)
    assert any(n.startswith("mouth") for n in names)


# ---------------------------------------------------------------------------
# StreamStore
# ---------------------------------------------------------------------------

def test_stream_store_set_and_has():
    store = StreamStore()
    assert not store.has("ai")
    store.set("ai", [_frame(0.0, 0.5)])
    assert store.has("ai")


def test_stream_store_clear():
    store = StreamStore()
    store.set("ai", [_frame(0.0, 0.5)])
    store.clear("ai")
    assert not store.has("ai")


def test_stream_store_set_empty_clears():
    store = StreamStore()
    store.set("ai", [_frame(0.0, 0.5)])
    store.set("ai", [])
    assert not store.has("ai")


def test_stream_store_sample_returns_none_for_missing():
    store = StreamStore()
    assert store.sample("ai", 0.0) is None


def test_stream_store_sample_interpolates():
    store = StreamStore()
    store.set("ai", [_frame(0.0, 0.0), _frame(100.0, 1.0)])
    vals = store.sample("ai", 50.0)
    assert vals is not None
    assert abs(vals[1] - 0.5) < 1e-9


def test_stream_store_frames_returns_list():
    store = StreamStore()
    frames = [_frame(0.0, 0.3)]
    store.set("ai", frames)
    got = store.frames("ai")
    assert len(got) == 1
    assert got[0]["blendshapes"][0] == pytest.approx(0.3)


def test_stream_store_frames_returns_empty_for_missing():
    store = StreamStore()
    assert store.frames("missing") == []


def test_stream_store_positions_cached():
    store = StreamStore()
    store.set("ai", [_frame(10.0), _frame(200.0)])
    assert store.positions("ai") == pytest.approx([10.0, 200.0])


def test_stream_store_prepare_all_shape():
    store = StreamStore()
    store.set("mediapipe", [_frame(0.0)])
    store.set("ai", [_frame(0.0), _frame(100.0)])
    result = store.prepare_all()
    assert set(result.keys()) == {"mediapipe", "ai"}
    frames_mp, pos_mp = result["mediapipe"]
    assert len(frames_mp) == 1
    assert pos_mp == [0.0]


def test_stream_store_frames_mutation_visible():
    store = StreamStore()
    store.set("ai", [_frame(0.0, 0.0)])
    store.frames("ai")[0]["blendshapes"][5] = 0.99
    assert store.frames("ai")[0]["blendshapes"][5] == pytest.approx(0.99)


def test_stream_store_clear_all():
    store = StreamStore()
    store.set("ai", [_frame(0.0)])
    store.set("retarget", [_frame(0.0)])
    store.clear_all()
    assert not store.has("ai")
    assert not store.has("retarget")
