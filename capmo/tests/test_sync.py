"""Nearest-timestamp frame matching across cameras (capmo.core.sync.frames;
formerly capmo.sync). Pure timestamp logic."""

from __future__ import annotations

from ..core.sync.frames import nearest as _nearest, synchronized_frames


def test_nearest_picks_closest():
    ts = [0.0, 10.0, 20.0, 30.0]
    assert _nearest(ts, 0.0) == 0
    assert _nearest(ts, 12.0) == 1
    assert _nearest(ts, 16.0) == 2
    assert _nearest(ts, 100.0) == 3
    assert _nearest(ts, -5.0) == 0


def test_two_streams_match_within_tolerance():
    a = [0.0, 33.0, 66.0, 100.0]
    b = [5.0, 40.0, 70.0, 105.0]      # offset but within 20 ms
    sets = synchronized_frames({"a": a, "b": b}, tol_ms=20.0)
    assert len(sets) == 4
    for fs in sets:
        assert set(fs.frames) == {"a", "b"}
        assert fs.spread_ms <= 7.0


def test_master_is_highest_rate_stream():
    fast = [i * 10.0 for i in range(10)]    # 10 frames
    slow = [0.0, 50.0]                       # 2 frames
    sets = synchronized_frames({"slow": slow, "fast": fast}, tol_ms=6.0)
    # Anchored on the fast stream, only the fast frames near a slow one match.
    assert all("fast" in s.frames for s in sets)
    matched_slow = {s.frames["slow"] for s in sets if "slow" in s.frames}
    assert matched_slow == {0, 1}


def test_frames_outside_tolerance_are_dropped():
    a = [0.0, 100.0]
    b = [3.0, 500.0]      # second b frame is far from any a frame
    sets = synchronized_frames({"a": a, "b": b}, tol_ms=20.0, min_cams=2)
    # only the (0,3) pair is within tolerance for both cameras
    assert len(sets) == 1
    assert sets[0].frames == {"a": 0, "b": 0}


def test_min_cams_requires_enough_views():
    a = [0.0, 10.0]
    b = [0.0, 10.0]
    c = [1000.0]                                  # never near a/b
    sets = synchronized_frames({"a": a, "b": b, "c": c}, tol_ms=5.0, min_cams=3)
    assert sets == []                             # c can't join -> no 3-cam set
    sets2 = synchronized_frames({"a": a, "b": b, "c": c}, tol_ms=5.0, min_cams=2)
    assert len(sets2) == 2                        # a+b pairs are fine


def test_empty_and_single_stream():
    assert synchronized_frames({}, tol_ms=20.0) == []
    assert synchronized_frames({"a": [1.0, 2.0]}, tol_ms=20.0) == []
