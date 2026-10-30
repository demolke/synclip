"""
Binding the retarget armature to a character rig, across different proportions.

The headless tests prove the proportion invariant on synthetic rigs: the same
captured motion retargeted onto a big rig and a small rig yields **identical bone
directions** (rotations are proportion-free) while the root translation scales
with rig size. A second group loads the real vendored Quaternius glTF when it is
present and checks the bind skeleton comes out sane and drivable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ..retarget import (
    default_skeleton, retarget_pose, retarget_track, forward_kinematics,
    landmarks_to_joints, quat_to_matrix, quat_from_matrix, JOINTS, JOINT_INDEX,
)
from .. import character_rig as CR
from .test_retarget import _bind_landmarks, _random_landmarks, _bone_dirs, _frame


# --------------------------------------------------------------------------- #
# quat_to_matrix
# --------------------------------------------------------------------------- #

def test_quat_to_matrix_matches_from_matrix():
    R = quat_from_matrix(np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]))  # 90 about y
    M = quat_to_matrix(R)
    assert np.allclose(M @ [1, 0, 0], [0, 0, -1], atol=1e-9)
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-9)        # orthonormal


# --------------------------------------------------------------------------- #
# Proportion independence (synthetic rigs)
# --------------------------------------------------------------------------- #

_DEFAULT = default_skeleton()


def _scaled_rig(scale: float):
    """A default skeleton whose every bind joint is scaled about the origin."""
    bind = {name: scale * _DEFAULT.bind_joints[JOINT_INDEX[name]]
            for name, _ in JOINTS}
    return default_skeleton(bind)


def test_rotations_are_proportion_free():
    """Same capture onto a 2x rig and a 0.5x rig -> identical bone directions."""
    pts = _random_landmarks(0)
    valid = np.ones(33, bool)
    big = _scaled_rig(2.0)
    small = _scaled_rig(0.5)
    pose_big = retarget_pose(big, pts, valid)
    pose_small = retarget_pose(small, pts, valid)

    dirs_big = _bone_dirs(big, forward_kinematics(big, pose_big))
    dirs_small = _bone_dirs(small, forward_kinematics(small, pose_small))
    assert np.allclose(dirs_big, dirs_small, atol=1e-9)
    # ...but the rigs keep their own lengths (2.0 vs 0.5 scale -> 4x).
    assert np.allclose(big.rest_len, 4.0 * small.rest_len, atol=1e-9)


def test_root_scale_rescales_translation_by_torso_ratio():
    pts = _bind_landmarks()                      # captured torso == default (0.5 m)
    valid = np.ones(33, bool)
    rig = _scaled_rig(3.0)                        # rig torso == 1.5 m
    frames = [_frame(pts, valid, 0.0)]

    ratio = CR.root_scale_for(rig, frames)
    assert ratio == pytest.approx(3.0, rel=1e-6)

    raw = retarget_track(frames, rig, root_scale=1.0)
    scaled = retarget_track(frames, rig, root_scale=ratio)
    assert np.allclose(scaled[0].root_pos, ratio * raw[0].root_pos, atol=1e-9)


def test_root_scale_defaults_to_one_without_torso():
    pts = _random_landmarks(1)
    valid = np.ones(33, bool)
    valid[11] = valid[12] = False                # no chest -> no torso measurement
    assert CR.root_scale_for(default_skeleton(), [_frame(pts, valid, 0.0)]) == 1.0


# --------------------------------------------------------------------------- #
# Real Quaternius rig (skipped when the asset is absent)
# --------------------------------------------------------------------------- #

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")
_HAS_RIG = os.path.exists(_RIG)


@pytest.mark.skipif(not _HAS_RIG, reason="Quaternius rig asset not present")
def test_load_real_quaternius_rig():
    skel = CR.load_rig_skeleton(_RIG)
    assert skel.num_bones == 14
    assert np.all(skel.rest_len > 0)
    # Roughly human, Y-up, ~1.8 m tall: chest sits above pelvis, head above chest.
    head = skel.bind_joints[JOINT_INDEX["head"]]
    pelvis = skel.bind_joints[JOINT_INDEX["pelvis"]]
    chest = skel.bind_joints[JOINT_INDEX["chest"]]
    assert chest[1] > pelvis[1] and head[1] > chest[1]
    assert 1.2 < head[1] - skel.bind_joints[JOINT_INDEX["l_ankle"]][1] < 2.2


@pytest.mark.skipif(not _HAS_RIG, reason="Quaternius rig asset not present")
def test_retarget_drives_real_rig_with_its_own_proportions():
    skel = CR.load_rig_skeleton(_RIG)
    pts = _random_landmarks(2)
    valid = np.ones(33, bool)
    pose = retarget_pose(skel, pts, valid)
    assert pose.bone_valid.all() and pose.root_valid

    # FK on the rig reproduces the captured directions on the rig's lengths.
    jp_fk = forward_kinematics(skel, pose)
    jp_meas, _ = landmarks_to_joints(pts, valid)
    assert np.allclose(_bone_dirs(skel, jp_fk), _bone_dirs(skel, jp_meas), atol=1e-9)
