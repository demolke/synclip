"""
The capmo bridge + software character render (no GL, no MediaPipe).

Part 1 -- ``rig_pose_from_joints`` is validated by a deterministic round-trip:
apply known SWING-ONLY bone rotations to the rig, read OUR 17 ground-truth joints
via :func:`capmo.skinning.posed_joint_positions`, recover the per-node local
rotations, and assert the recovered pose re-skins to the *same* mesh as the known
pose. Pure swings carry no twist, so this is exact (~1e-5 m). The identity pose
must reproduce the bind mesh.

Part 2 -- the software renderer (:func:`capmo.animate.render_character` /
``render_track``) must produce distinct, plausibly-framed, deterministic images
for distinct poses. Pure numpy + the synth rasterizer; runs headless.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .. import animate, skinning

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")


def _axis_angle(axis, deg: float) -> np.ndarray:
    a = math.radians(deg)
    x, y, z = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# Swing-only pose: hinge about each bone's local +X, like the posed-E2E test.
_KNOWN = {
    "lowerarm_l": _axis_angle((1, 0, 0), 70.0),
    "lowerarm_r": _axis_angle((1, 0, 0), 70.0),
    "upperarm_l": _axis_angle((1, 0, 0), 20.0),
    "upperarm_r": _axis_angle((1, 0, 0), 20.0),
    "calf_l": _axis_angle((1, 0, 0), 25.0),
    "calf_r": _axis_angle((1, 0, 0), 25.0),
}


def test_round_trip_recovers_swing_only_pose():
    skin = skinning.load_skin(_RIG)
    jp = skinning.posed_joint_positions(skin, _KNOWN)
    jv = np.ones(len(jp), bool)

    recovered = animate.rig_pose_from_joints(skin, jp, jv)
    # Every driven bone we posed must come back.
    assert set(_KNOWN).issubset(recovered)

    v_known = skinning.pose_rig_mesh(skin, _KNOWN)
    v_rec = skinning.pose_rig_mesh(skin, recovered)
    err = np.linalg.norm(v_known - v_rec, axis=1)
    # Pure swings are exactly recoverable: tight tolerance, well under 1e-3 m.
    assert err.max() < 1e-3, f"round-trip mesh error {err.max():.3e} m too big"


def test_identity_pose_reproduces_bind_mesh():
    skin = skinning.load_skin(_RIG)
    jp = skinning.posed_joint_positions(skin, {})       # bind ground-truth joints
    jv = np.ones(len(jp), bool)
    recovered = animate.rig_pose_from_joints(skin, jp, jv)

    v_bind = skinning.pose_rig_mesh(skin, {})
    v_rec = skinning.pose_rig_mesh(skin, recovered)
    err = np.linalg.norm(v_bind - v_rec, axis=1)
    assert err.max() < 1e-3, f"bind reproduce error {err.max():.3e} m too big"


def test_invalid_joints_leave_bones_at_bind():
    skin = skinning.load_skin(_RIG)
    jp = skinning.posed_joint_positions(skin, _KNOWN)
    jv = np.ones(len(jp), bool)
    jv[:] = False                                       # nothing solved
    recovered = animate.rig_pose_from_joints(skin, jp, jv)
    assert recovered == {}                              # no swings -> all at bind


def test_skeleton_wrapper_matches_joint_path():
    skin = skinning.load_skin(_RIG)
    # 33 landmarks where each of our joints' source landmark sits on the rig node.
    jp = skinning.posed_joint_positions(skin, _KNOWN)
    points = np.zeros((33, 3))
    valid = np.zeros(33, bool)
    from ..retarget import JOINTS, JOINT_INDEX
    for name, src in JOINTS:
        for s in src:
            points[s] = jp[JOINT_INDEX[name]]
            valid[s] = True
    from_skel = animate.rig_pose_from_skeleton(skin, points, valid)
    from_joints = animate.rig_pose_from_joints(skin, *_landmarks(skin, points, valid))
    assert set(from_skel) == set(from_joints)
    for k in from_skel:
        assert np.allclose(from_skel[k], from_joints[k])


def _landmarks(skin, points, valid):
    from ..retarget import landmarks_to_joints
    return landmarks_to_joints(points, valid)


class _FakeFrame:
    def __init__(self, skel):
        self.skeleton = skel


class _FakeSkel:
    def __init__(self, points, valid):
        self.points = points
        self.valid = valid


def test_animate_track_holds_last_for_dropouts():
    skin = skinning.load_skin(_RIG)
    jp = skinning.posed_joint_positions(skin, _KNOWN)
    from ..retarget import JOINTS, JOINT_INDEX

    def mk(valid_mask):
        pts = np.zeros((33, 3))
        val = np.zeros(33, bool)
        for name, src in JOINTS:
            for s in src:
                pts[s] = jp[JOINT_INDEX[name]]
                val[s] = valid_mask
        return _FakeFrame(_FakeSkel(pts, val))

    frames = [mk(True), mk(False), mk(True)]            # middle frame drops out
    track = animate.animate_track(skin, frames, alpha=0.0)
    assert len(track) == 3
    # The dropout frame holds the previous solved pose rather than snapping to bind.
    assert set(track[1]) == set(track[0])
    for k in track[0]:
        assert np.allclose(track[1][k], track[0][k])
    # the joints3d-array entry point (what View uses) gives the same track
    arr = animate.animate_joints(skin, np.stack([f.skeleton.points for f in frames]),
                                 np.stack([f.skeleton.valid for f in frames]), alpha=0.0)
    assert [sorted(p) for p in arr] == [sorted(p) for p in track]
    for a, b in zip(arr, track):
        for k in a:
            assert np.allclose(a[k], b[k])


def test_render_distinct_poses_differ_and_are_framed():
    skin = skinning.load_skin(_RIG)
    pose_a: dict = {}
    pose_b = {"upperarm_l": _axis_angle((1, 0, 0), 45.0),
              "lowerarm_l": _axis_angle((1, 0, 0), 60.0)}
    imgs = animate.render_track(skin, [pose_a, pose_b], W=320, H=440)
    assert len(imgs) == 2
    for img in imgs:
        assert img.shape == (440, 320, 3)
        frac = float(np.any(img != 128, axis=2).mean())
        assert 0.01 < frac < 0.9, f"implausible figure fraction {frac:.3f}"
    assert not np.array_equal(imgs[0], imgs[1])         # poses render differently


def test_render_is_deterministic():
    skin = skinning.load_skin(_RIG)
    pose = {"calf_r": _axis_angle((1, 0, 0), 30.0)}
    a = animate.render_character(skin, pose,
                                 *animate._default_camera(skin, 256, 320), 256, 320)
    b = animate.render_character(skin, pose,
                                 *animate._default_camera(skin, 256, 320), 256, 320)
    assert np.array_equal(a, b)


def test_alpha_is_the_weight_of_the_new_frame():
    """alpha=1 follows the raw track exactly (no smoothing); a smaller alpha
    lags behind it (more smoothing); 0 switches smoothing off."""
    skin = skinning.load_skin(_RIG)
    from ..retarget import JOINTS, JOINT_INDEX

    def mk(pose):
        jp = skinning.posed_joint_positions(skin, pose)
        pts, val = np.zeros((33, 3)), np.zeros(33, bool)
        for name, src in JOINTS:
            for s in src:
                pts[s], val[s] = jp[JOINT_INDEX[name]], True
        return pts, val

    (p0, v0), (p1, v1) = mk({}), mk(_KNOWN)
    J, V = np.stack([p0, p1]), np.stack([v0, v1])
    raw = animate.animate_joints(skin, J, V, alpha=0.0)
    one = animate.animate_joints(skin, J, V, alpha=1.0)
    half = animate.animate_joints(skin, J, V, alpha=0.5)
    node = "lowerarm_l"
    assert np.allclose(one[1][node], raw[1][node], atol=1e-9)
    d_raw = np.abs(raw[1][node] - raw[0][node]).sum()
    assert d_raw > 0.1
    assert 0 < np.abs(half[1][node] - raw[0][node]).sum() < d_raw
