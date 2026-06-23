"""
Tests for head-pose application in the 3D preview (MeshRenderer).

Covers the regression where MediaPipe's large camera-space translation threw
the preview head far off screen, and the requirement that disabling the head
pose (zeroed axes) returns the head exactly to the original base view.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")

from synclip.ui.mesh_renderer import (  # noqa: E402
    head_model_matrix,
    _HEAD_POS_SCALE,
)


def test_zero_pose_is_identity():
    """A zero pose must produce the identity model matrix, so disabling the
    head pose leaves the head at the original base view."""
    m = head_model_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.allclose(m, np.eye(4), atol=1e-6)


def test_translation_is_scaled_down():
    """A large MediaPipe translation must be scaled into mesh units (not applied
    raw), so the head doesn't fly off screen."""
    pos = [10.0, -20.0, 30.0]
    m = head_model_matrix([0.0, 0.0, 0.0], pos)
    # With no rotation the model matrix is a pure (scaled) translation.
    assert m[0, 3] == pytest.approx(10.0 * _HEAD_POS_SCALE, abs=1e-5)
    assert m[1, 3] == pytest.approx(-20.0 * _HEAD_POS_SCALE, abs=1e-5)
    assert m[2, 3] == pytest.approx(30.0 * _HEAD_POS_SCALE, abs=1e-5)
    # And the scaled magnitude is far smaller than the raw translation.
    assert abs(m[2, 3]) < 30.0


def test_rotation_applied():
    """A non-zero rotation must rotate the upper-left 3x3 (not stay identity)."""
    m = head_model_matrix([0.0, 90.0, 0.0], [0.0, 0.0, 0.0])
    assert not np.allclose(m[:3, :3], np.eye(3), atol=1e-3)


def test_rotation_in_place_not_orbiting():
    """Rotation must be applied about the head's own centre: with zero
    translation, a pure rotation leaves the translation column at the origin
    (the bug applied R @ T, which orbited the head by the large translation)."""
    m = head_model_matrix([0.0, 45.0, 0.0], [0.0, 0.0, 0.0])
    assert np.allclose(m[:3, 3], [0.0, 0.0, 0.0], atol=1e-6)


def test_neck_anchor_zero_matches_no_anchor():
    """neck_anchor=0 must equal rotating about the head centre (default)."""
    rot, pos = [0.0, 30.0, 0.0], [0.0, 0.0, 0.0]
    a = head_model_matrix(rot, pos, neck_anchor=0.0)
    b = head_model_matrix(rot, pos)
    assert np.allclose(a, b, atol=1e-6)


def test_neck_anchor_keeps_neck_point_fixed():
    """With neck_anchor=1, the neck pivot point must stay fixed under rotation
    (the head swings about it) while the head centre moves."""
    from synclip.ui.mesh_renderer import _NECK_OFFSET
    # Pitch (rot_x) so the off-axis neck point and head centre actually move.
    m = head_model_matrix([40.0, 0.0, 0.0], [0.0, 0.0, 0.0], neck_anchor=1.0)
    neck = np.array([0.0, -_NECK_OFFSET, 0.0, 1.0])
    moved = m @ neck
    assert np.allclose(moved[:3], neck[:3], atol=1e-5), \
        "neck pivot point must stay in place"
    # The head centre (origin) should NOT stay fixed when pivoting about the neck.
    head_centre = m @ np.array([0.0, 0.0, 0.0, 1.0])
    assert not np.allclose(head_centre[:3], [0.0, 0.0, 0.0], atol=1e-2)


def test_neck_anchor_zero_pose_still_identity():
    """A zero pose stays identity regardless of neck anchor."""
    m = head_model_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], neck_anchor=1.0)
    assert np.allclose(m, np.eye(4), atol=1e-6)


def _make_renderer():
    from synclip.ui.mesh_renderer import MeshRenderer
    return MeshRenderer(label="test")


def test_disabling_pose_restores_base_view(qapp_module):
    """set_head_pose(None) / zeroed pose must give the same MVP as never having
    applied a pose -- i.e. disabling head pose returns to the previous look."""
    r = _make_renderer()
    base_mvp, _ = r._mvp()

    r.set_head_pose({"rot": [15.0, 25.0, 5.0], "pos": [8.0, -4.0, 12.0]})
    posed_mvp, _ = r._mvp()
    assert not np.allclose(base_mvp, posed_mvp, atol=1e-4), \
        "Applying a pose should change the MVP"

    r.set_head_pose(None)
    disabled_mvp, _ = r._mvp()
    assert np.allclose(base_mvp, disabled_mvp, atol=1e-6), \
        "Disabling the head pose must restore the original base view"


@pytest.fixture(scope="module")
def qapp_module():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
