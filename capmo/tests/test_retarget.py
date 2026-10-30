"""
Retargeting a 3D pose track onto a humanoid character.

The core invariant is a forward-kinematics round-trip: retarget a known set of
3D joint positions into bone rotations, run FK, and every bone must point in the
measured direction. Root facing, gap filling, smoothing and serialization are
exercised too. Pure numpy -- no cameras, no video.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .. import retarget as R
from ..retarget import (
    default_skeleton, retarget_pose, forward_kinematics, retarget_track, retarget_joints,
    landmarks_to_joints, clip_to_dict, quat_rotate, quat_mul, quat_conj,
    quat_from_two_vectors, quat_from_matrix, quat_slerp, IDENTITY_QUAT, JOINT_INDEX,
)


def _bind_landmarks() -> np.ndarray:
    """A (33, 3) landmark frame whose joints match the skeleton's bind pose."""
    skel = default_skeleton()
    pts = np.zeros((33, 3))
    # Each joint sources one or two landmarks; set every source landmark to the
    # bind joint position (averaging two equal points is a no-op).
    for ji, (_, src) in enumerate(R.JOINTS):
        for s in src:
            pts[s] = skel.bind_joints[ji]
    return pts


def _bone_dirs(skel, jp):
    """Measured unit direction of every bone from joint positions *jp*."""
    out = []
    for b in range(skel.num_bones):
        d = jp[skel.tail_joint[b]] - jp[skel.head_joint[b]]
        out.append(d / np.linalg.norm(d))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Quaternion helpers
# --------------------------------------------------------------------------- #

def test_quat_from_two_vectors_aligns():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    q = quat_from_two_vectors(a, b)
    assert np.allclose(quat_rotate(q, a), b, atol=1e-9)


def test_quat_from_two_vectors_handles_opposite():
    a = np.array([0.0, 0.0, 1.0])
    q = quat_from_two_vectors(a, -a)
    assert np.allclose(quat_rotate(q, a), -a, atol=1e-9)


def test_quat_from_matrix_roundtrip():
    # 90 deg about y: x -> -z
    R90 = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    q = quat_from_matrix(R90)
    assert np.allclose(quat_rotate(q, [1, 0, 0]), [0, 0, -1], atol=1e-9)
    assert np.allclose(quat_rotate(q, [0, 1, 0]), [0, 1, 0], atol=1e-9)


def test_quat_mul_conj_inverse():
    q = quat_from_two_vectors([1, 0, 0], [0.3, 0.8, -0.2])
    assert np.allclose(quat_mul(q, quat_conj(q)), IDENTITY_QUAT, atol=1e-9)


def test_quat_slerp_endpoints_and_midpoint():
    a = IDENTITY_QUAT
    b = quat_from_matrix(np.array([[0, -1.0, 0], [1, 0, 0], [0, 0, 1]]))  # 90 about z
    assert np.allclose(quat_slerp(a, b, 0.0), a, atol=1e-9)
    assert np.allclose(quat_rotate(quat_slerp(a, b, 1.0), [1, 0, 0]),
                       quat_rotate(b, [1, 0, 0]), atol=1e-9)
    mid = quat_slerp(a, b, 0.5)                       # 45 deg about z
    assert np.allclose(quat_rotate(mid, [1, 0, 0]),
                       [np.cos(np.pi / 4), np.sin(np.pi / 4), 0], atol=1e-9)


# --------------------------------------------------------------------------- #
# Skeleton structure
# --------------------------------------------------------------------------- #

def test_skeleton_is_a_valid_tree():
    skel = default_skeleton()
    for b in range(skel.num_bones):
        par = skel.parent[b]
        assert par < b                                    # parents precede children
        if par >= 0:                                      # child head == parent tail
            assert skel.head_joint[b] == skel.tail_joint[par]
    assert np.all(skel.rest_len > 0)
    assert np.allclose(np.linalg.norm(skel.rest_dir, axis=1), 1.0)


def test_bind_pose_has_identity_root_orientation():
    skel = default_skeleton()
    pts = _bind_landmarks()
    pose = retarget_pose(skel, pts, np.ones(33, bool))
    assert pose.root_valid
    assert np.allclose(pose.root_rot, IDENTITY_QUAT, atol=1e-9)
    # bind pose -> every bone already at rest -> identity local rotations
    assert np.allclose(pose.local_rot, IDENTITY_QUAT, atol=1e-9)


# --------------------------------------------------------------------------- #
# Retarget / FK round-trip
# --------------------------------------------------------------------------- #

def _random_landmarks(seed):
    """A plausible, fully-visible 33-landmark frame perturbed from the bind pose."""
    rng = np.random.default_rng(seed)
    pts = _bind_landmarks()
    return pts + rng.uniform(-0.15, 0.15, size=pts.shape)


def test_fk_reproduces_measured_bone_directions():
    skel = default_skeleton()
    pts = _random_landmarks(0)
    valid = np.ones(33, bool)
    pose = retarget_pose(skel, pts, valid)

    jp_meas, _ = landmarks_to_joints(pts, valid)
    jp_fk = forward_kinematics(skel, pose)
    # Character keeps its own proportions, but every bone must point the same way.
    assert np.allclose(_bone_dirs(skel, jp_fk), _bone_dirs(skel, jp_meas), atol=1e-9)


def test_fk_reproduces_positions_when_proportions_match():
    """If the character's bind pose *is* the captured frame, FK restores it exactly."""
    pts = _random_landmarks(1)
    valid = np.ones(33, bool)
    jp, _ = landmarks_to_joints(pts, valid)
    bind = {name: jp[JOINT_INDEX[name]] for name, _ in R.JOINTS}
    skel = default_skeleton(bind)
    pose = retarget_pose(skel, pts, valid)
    jp_fk = forward_kinematics(skel, pose)
    assert np.allclose(jp_fk, jp, atol=1e-9)


def test_root_captures_facing():
    skel = default_skeleton()
    pts = _bind_landmarks()
    # Yaw the whole body 90 deg about the vertical (y) axis.
    Ry = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    rot = pts @ Ry.T
    pose = retarget_pose(skel, rot, np.ones(33, bool))
    # Root orientation should encode that yaw: forward (+z) maps like Ry @ +z.
    assert np.allclose(quat_rotate(pose.root_rot, [0, 0, 1]), Ry @ [0, 0, 1], atol=1e-9)
    assert np.allclose(quat_rotate(pose.root_rot, [1, 0, 0]), Ry @ [1, 0, 0], atol=1e-9)


# --------------------------------------------------------------------------- #
# Visibility / gaps
# --------------------------------------------------------------------------- #

def test_invalid_joint_leaves_bone_at_rest():
    skel = default_skeleton()
    pts = _random_landmarks(2)
    valid = np.ones(33, bool)
    valid[15] = False                                     # left wrist unseen
    pose = retarget_pose(skel, pts, valid)
    fa = skel.bone_names.index("l_forearm")
    assert not pose.bone_valid[fa]
    assert np.allclose(pose.local_rot[fa], IDENTITY_QUAT, atol=1e-9)
    assert pose.bone_valid[skel.bone_names.index("l_upperarm")]  # neighbour solved


def test_missing_root_landmarks_flag_root_invalid():
    skel = default_skeleton()
    pts = _random_landmarks(3)
    valid = np.ones(33, bool)
    valid[23] = valid[24] = False                         # both hips gone -> no pelvis
    pose = retarget_pose(skel, pts, valid)
    assert not pose.root_valid
    assert np.allclose(pose.root_rot, IDENTITY_QUAT, atol=1e-9)
    assert not np.all(np.isfinite(pose.root_pos))


# --------------------------------------------------------------------------- #
# Track-level: gap fill + smoothing + serialization
# --------------------------------------------------------------------------- #

def _frame(points, valid, t_ms):
    """A per-frame object for the frame-based wrappers (``t_ms`` + ``skeleton``)."""
    return SimpleNamespace(t_ms=t_ms, skeleton=SimpleNamespace(points=points, valid=valid))


def test_track_fills_gaps_by_holding_last_pose():
    skel = default_skeleton()
    pts = _random_landmarks(4)
    full = np.ones(33, bool)
    drop = full.copy()
    drop[16] = False                                      # right wrist gone in frame 1
    frames = [_frame(pts, full, 0.0), _frame(pts, drop, 33.0)]
    poses = retarget_track(frames, skel, fill_gaps=True)
    # the joints3d-array entry point is the same computation
    arr = retarget_joints(np.stack([pts, pts]), np.stack([full, drop]), [0.0, 33.0], skel)
    for a, b in zip(arr, poses):
        assert a.t_ms == b.t_ms and np.allclose(a.local_rot, b.local_rot)
        assert np.array_equal(a.bone_valid, b.bone_valid)
    fa = skel.bone_names.index("r_forearm")
    # The held rotation equals what the previous (full) frame solved.
    assert np.allclose(poses[1].local_rot[fa], poses[0].local_rot[fa], atol=1e-9)
    assert not poses[1].bone_valid[fa]


def test_track_smoothing_moves_toward_previous():
    skel = default_skeleton()
    p0 = _bind_landmarks()
    p1 = _random_landmarks(5)
    frames = [_frame(p0, np.ones(33, bool), 0.0), _frame(p1, np.ones(33, bool), 33.0)]
    raw = retarget_track(frames, skel, smooth=0.0)
    sm = retarget_track(frames, skel, smooth=0.5)
    b = skel.bone_names.index("l_upperarm")
    # Smoothed frame-1 rotation sits between the bind pose and the raw target.
    raw_ang = 2 * np.arccos(np.clip(abs(raw[1].local_rot[b][0]), 0, 1))
    sm_ang = 2 * np.arccos(np.clip(abs(sm[1].local_rot[b][0]), 0, 1))
    assert 0 < sm_ang < raw_ang + 1e-9
    assert sm_ang < raw_ang


def test_clip_serialization_roundtrips_through_json():
    import json
    skel = default_skeleton()
    pts = _random_landmarks(6)
    valid = np.ones(33, bool)
    valid[28] = False                                     # leave one bone invalid
    poses = retarget_track([_frame(pts, valid, 0.0)], skel)
    d = json.loads(json.dumps(clip_to_dict(skel, poses)))
    assert d["format"] == "capmo.character_clip/1"
    assert d["skeleton"]["bones"] == skel.bone_names
    assert len(d["frames"]) == 1
    fr = d["frames"][0]
    assert len(fr["local_rot"]) == skel.num_bones
    assert len(fr["bone_valid"]) == skel.num_bones
    assert False in fr["bone_valid"]
