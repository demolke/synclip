"""
Tests for rapid mouth-closure detection and post-mix enforcement.

These codify the "post-mix track" behaviour: detect quick 1-3 frame mouth
closures (p/b/m) in a source animation, and re-assert them on the mixed output
so smoothing / AI-mixing / interpolation can't wash them away.
"""

from __future__ import annotations

from synclip.mouth_closure import (
    JAW_OPEN,
    MOUTH_CLOSE,
    ClosureEvent,
    detect_closures,
    enforce_closure,
    merge_events,
)


def _frames(jaw_values: list[float], fps: float = 30.0) -> list[dict]:
    """Build frames with the given jawOpen values, evenly spaced in ms."""
    step = 1000.0 / fps
    out = []
    for i, j in enumerate(jaw_values):
        bs = [0.0] * 52
        bs[JAW_OPEN] = j
        out.append({"audio_position_ms": i * step, "blendshapes": bs})
    return out


def _positions(frames):
    return [f["audio_position_ms"] for f in frames]


# ---- Detection ------------------------------------------------------------

def test_detects_single_frame_closure():
    # open ... open, CLOSE (1 frame), open ... open
    frames = _frames([0.6, 0.6, 0.6, 0.0, 0.6, 0.6])
    events = detect_closures(frames)
    assert len(events) == 1
    assert events[0].strength > 0.5


def test_detects_two_frame_closure():
    frames = _frames([0.6, 0.6, 0.05, 0.05, 0.6, 0.6])
    events = detect_closures(frames)
    assert len(events) == 1


def test_detects_three_frame_closure():
    frames = _frames([0.6, 0.0, 0.0, 0.0, 0.6])
    events = detect_closures(frames)
    assert len(events) == 1


def test_steady_open_has_no_closure():
    frames = _frames([0.6, 0.6, 0.6, 0.6, 0.6])
    assert detect_closures(frames) == []


def test_close_and_stay_shut_is_not_a_closure():
    # mouth opens then closes and STAYS closed -> only one open shoulder -> ignore
    frames = _frames([0.6, 0.6, 0.0, 0.0, 0.0, 0.0])
    assert detect_closures(frames) == []


def test_shallow_dip_below_threshold_ignored():
    # dip of only 0.05, below default drop=0.10
    frames = _frames([0.6, 0.6, 0.55, 0.6, 0.6])
    assert detect_closures(frames, drop=0.10) == []


def test_dip_between_closed_shoulders_ignored():
    # shoulders below open_min -> not a speech closure
    frames = _frames([0.1, 0.1, 0.0, 0.1, 0.1])
    assert detect_closures(frames, open_min=0.15) == []


def test_wide_closure_beyond_max_width_not_collapsed_to_one():
    # 4-frame closure exceeds max_width=3; the shoulders for any 1-3 window are
    # themselves closed, so nothing fires.
    frames = _frames([0.6, 0.0, 0.0, 0.0, 0.0, 0.6])
    assert detect_closures(frames, max_width=3) == []


# ---- Merge ----------------------------------------------------------------

def test_merge_overlapping_events_keeps_deeper():
    a = ClosureEvent(0.0, 10.0, 20.0, strength=0.3)
    b = ClosureEvent(15.0, 25.0, 30.0, strength=0.7)
    merged = merge_events([a, b])
    assert len(merged) == 1
    assert merged[0].start_ms == 0.0
    assert merged[0].end_ms == 30.0
    assert merged[0].center_ms == 25.0      # deeper event's centre
    assert merged[0].strength == 0.7


def test_merge_keeps_disjoint_events():
    a = ClosureEvent(0.0, 10.0, 20.0, 0.3)
    b = ClosureEvent(40.0, 50.0, 60.0, 0.5)
    assert len(merge_events([a, b])) == 2


# ---- Enforcement ----------------------------------------------------------

def test_enforce_closes_mouth_at_center():
    events = [ClosureEvent(0.0, 50.0, 100.0, 0.5)]
    vals = [0.0] * 52
    vals[JAW_OPEN] = 0.8
    out = enforce_closure(vals, 50.0, events, amount=0.9)
    # At the centre the envelope is 1.0, so jawOpen is scaled by (1 - 0.9).
    assert out[JAW_OPEN] == 0.8 * (1.0 - 0.9)
    assert out[MOUTH_CLOSE] == 0.9


def test_enforce_outside_window_unchanged():
    events = [ClosureEvent(0.0, 50.0, 100.0, 0.5)]
    vals = [0.0] * 52
    vals[JAW_OPEN] = 0.8
    out = enforce_closure(vals, 200.0, events, amount=0.9)
    assert out[JAW_OPEN] == 0.8
    assert out[MOUTH_CLOSE] == 0.0


def test_enforce_envelope_zero_at_shoulders():
    events = [ClosureEvent(0.0, 50.0, 100.0, 0.5)]
    vals = [0.0] * 52
    vals[JAW_OPEN] = 0.8
    # Exactly at the shoulder the envelope is 0 -> no change.
    out = enforce_closure(vals, 0.0, events, amount=0.9)
    assert out[JAW_OPEN] == 0.8


def test_enforce_partial_between_shoulder_and_center():
    events = [ClosureEvent(0.0, 50.0, 100.0, 0.5)]
    vals = [0.0] * 52
    vals[JAW_OPEN] = 1.0
    out = enforce_closure(vals, 25.0, events, amount=1.0)
    # Half-way up the rising cosine: envelope = 0.5, so jawOpen ~ 0.5.
    assert 0.4 < out[JAW_OPEN] < 0.6


def test_enforce_noop_when_amount_zero():
    events = [ClosureEvent(0.0, 50.0, 100.0, 0.5)]
    vals = [0.0] * 52
    vals[JAW_OPEN] = 0.8
    out = enforce_closure(vals, 50.0, events, amount=0.0)
    assert out[JAW_OPEN] == 0.8
