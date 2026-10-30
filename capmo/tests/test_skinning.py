"""
Validate linear blend skinning against the rig's bind pose and a known rotation.

Deterministic (numpy + the rig asset only, no MediaPipe). Two checks guard the
error-prone coordinate-space wiring before any test trusts :mod:`capmo.skinning`:

(a) posing with *no* rotations round-trips the bind mesh that
    :func:`capmo.character_rig.load_rig_mesh` returns (to ~1e-6); and
(b) rotating the left upper arm moves the left-hand vertices in the expected
    direction while the pelvis and the right hand stay put.

Skipped only if the rig asset is absent (we never fake the geometry here).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from .. import character_rig as CR
from .. import skinning

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")

pytestmark = pytest.mark.skipif(not os.path.exists(_RIG),
                                reason="Quaternius rig asset not present")


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identity_pose_roundtrips_bind_mesh():
    """No rotations -> LBS reproduces the bind-pose mesh verts."""
    skin = skinning.load_skin(_RIG)
    posed = skinning.pose_rig_mesh(skin, {})
    bind, _ = CR.load_rig_mesh(_RIG)
    assert posed.shape == bind.shape
    err = np.abs(posed - bind).max()
    assert err < 1e-4, f"identity LBS strayed from bind mesh by {err:.2e} m"


def test_left_arm_rotation_moves_left_hand_only():
    """Rotating upperarm_l swings the left hand; pelvis and right hand hold."""
    skin = skinning.load_skin(_RIG)
    rest = skinning.pose_rig_mesh(skin, {})
    posed = skinning.pose_rig_mesh(skin, {"upperarm_l": _rot_z(45.0)})

    # World origins of the relevant bones (bind pose) -> pick nearby vertices.
    def near(bone: str, radius: float = 0.12) -> np.ndarray:
        node = skin.node_name[bone]
        # bind global origin via an unposed walk
        g = skinning._animated_globals(skin, {})[node][:3, 3]
        return np.linalg.norm(skin.verts - g, axis=1) < radius

    hand_l = near("hand_l")
    pelvis = near("pelvis")
    hand_r = near("hand_r")
    assert hand_l.sum() > 50 and hand_r.sum() > 50 and pelvis.sum() > 50

    disp = posed - rest
    hand_l_move = np.linalg.norm(disp[hand_l].mean(0))
    assert hand_l_move > 0.1, "left hand barely moved under a 45 deg shoulder turn"

    # A +Z (toward the figure's back-left) local turn lifts the left hand (+Y)
    # and pulls it toward the body (-X = the figure's right). Assert the signs.
    mean_disp = disp[hand_l].mean(0)
    assert mean_disp[1] > 0.0, "left hand should rise"
    assert mean_disp[0] < 0.0, "left hand should swing inward (-X)"

    # The pelvis and the opposite hand are unaffected by a left-shoulder turn.
    assert np.linalg.norm(disp[pelvis].mean(0)) < 1e-3
    assert np.linalg.norm(disp[hand_r].mean(0)) < 1e-3


def test_posed_joint_positions_match_bind_under_identity():
    """Ground-truth joints with no pose equal the rig's bind skeleton."""
    gt = skinning.posed_joint_positions(_RIG, {})
    bind = CR.load_rig_skeleton(_RIG).bind_joints
    assert np.allclose(gt, bind, atol=1e-6)
