"""Tests for the generalized CurveEditor widget (response-curve + animation)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _editor(**kw):
    from synclip.ui.curve_editor import CurveEditor
    ed = CurveEditor(**kw)
    ed.resize(220, 140)   # margin 10 -> 200x120 plotting area, deterministic
    return ed


# ---------------------------------------------------------------------------
# Response-curve (lut) mode keeps its existing behaviour
# ---------------------------------------------------------------------------

def test_lut_mode_defaults_to_identity_and_emits_lut(qapp):
    from synclip.curve_lut import LUT_SIZE
    ed = _editor()
    assert ed.points() == [[0.0, 0.0], [1.0, 1.0]]
    seen = []
    ed.curve_changed.connect(seen.append)
    ed.set_points([(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)])
    assert len(seen) == 1 and len(seen[0]) == LUT_SIZE   # baked LUT payload


# ---------------------------------------------------------------------------
# Animation mode
# ---------------------------------------------------------------------------

def test_animation_mode_emits_raw_points(qapp):
    ed = _editor(eval_mode="animation", y_range=(-1.0, 1.0), reference=0.0,
                 default_points=[(0.0, 0.0), (1.0, 0.0)])
    assert ed.points() == [[0.0, 0.0], [1.0, 0.0]]
    seen = []
    ed.curve_changed.connect(seen.append)
    ed.set_points([(0.0, 0.5), (1.0, -0.5)])
    assert seen[-1] == [[0.0, 0.5], [1.0, -0.5]]   # raw points, not a LUT


def test_signed_y_range_round_trips_through_pixels(qapp):
    ed = _editor(eval_mode="animation", y_range=(-1.0, 1.0))
    # y = 0.5 in [-1, 1] -> normalised 0.75 -> 30px from top (of 120).
    px = ed._to_px(0.3, 0.5)
    assert abs(px.y() - 40.0) < 1e-6   # margin 10 + (1-0.75)*120
    x, y = ed._to_curve(px.x(), px.y())
    assert abs(x - 0.3) < 1e-6 and abs(y - 0.5) < 1e-6


def test_to_curve_clamps_y_into_range(qapp):
    ed = _editor(eval_mode="animation", y_range=(-1.0, 1.0))
    # Way above the top edge -> clamps to ymax; below bottom -> ymin.
    _, y_hi = ed._to_curve(50.0, -500.0)
    _, y_lo = ed._to_curve(50.0, 5000.0)
    assert y_hi == 1.0 and y_lo == -1.0


def test_reset_restores_configured_default(qapp):
    ed = _editor(eval_mode="animation", y_range=(-1.0, 1.0),
                 default_points=[(0.0, 0.0), (1.0, 0.0)])
    ed.set_points([(0.0, 0.7), (0.5, -0.2), (1.0, 0.3)])
    ed.reset_to_default()
    assert ed.points() == [[0.0, 0.0], [1.0, 0.0]]


def test_set_playhead_is_safe(qapp):
    ed = _editor(eval_mode="animation", y_range=(-1.0, 1.0))
    ed.set_playhead(0.42)
    ed.set_playhead(None)   # clear
