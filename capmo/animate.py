"""
Drive the skinned character rig from a reconstructed 3D pose track, and render it.

This is the last phase of the capmo loop. Earlier phases turn a multi-camera
capture into a 3D landmark track (:mod:`capmo.core.pipeline` /
``reconstruct_take`` -> the ``joints3d`` contract of :mod:`capmo.core.joints3d`,
33 MediaPipe landmarks per frame). :mod:`capmo.skinning` can *pose* the vendored glTF character by a
set of extra **local** bone rotations and linear-blend-skin the mesh. The missing
bridge -- built here -- is going the other way: from the captured joint
*positions* to the per-rig-node local rotations that make the character strike
that pose.

Three parts:

* **The bridge** (:func:`rig_pose_from_joints`): given OUR 17 joint world
  positions, aim each driven rig bone at its measured direction by the minimal
  *swing* that carries the bone's bind world direction onto the measured one,
  then divide out the parent to recover the extra *local* rotation skinning wants.
  Processed parent-first over the whole skin hierarchy so forward kinematics
  reproduces each bone's target global. Pure swings (no twist) are *exactly*
  recoverable -- the round-trip test proves it to ~1e-9 m.
* **Software rendering** (:func:`render_character` / :func:`render_track`):
  ``pose_rig_mesh`` -> per-triangle Lambert shade -> :func:`capmo.synth.rasterize`.
  Deliberately GL-free (a pure-numpy rasterizer) so it runs headless in tests and
  on any machine; an OpenGL view is left as a future swap.
* **A track helper** (:func:`animate_joints`): per-frame poses over a joints3d
  track, with optional temporal slerp smoothing and hold-last gap filling.

Twist about a bone's own axis is not observable from joint positions, so only the
swing is recovered (as in :mod:`capmo.retarget`). The spine is approximated by a
single driven segment (``spine_01`` pelvis->chest); ``spine_02``/``spine_03`` and
the clavicles/hands/feet/fingers stay at bind.
"""

from __future__ import annotations

import numpy as np

from . import skinning, synth
from .character_rig import QUATERNIUS_JOINTS
from .retarget import (
    JOINT_INDEX, landmarks_to_joints, quat_from_matrix, quat_from_two_vectors,
    quat_slerp, quat_to_matrix,
)

# Driven rig node -> (head our-joint, tail our-joint, rig child node used for the
# bind direction). The bind-pose world bone direction is taken between the rig
# nodes the head/tail joints map to (the tail's node is the listed child); we
# swing that onto the measured head->tail. Anchoring dbind on the SAME pair of
# joints that dmeas uses makes a pure swing exactly recoverable -- for the limbs
# the node's own origin already equals its head-joint node, and for the spine /
# neck (whose head joint sits below the bone's own node) it is what keeps the
# round-trip tight (see :func:`rig_pose_from_joints`).
_DRIVEN: dict[str, tuple[str, str, str]] = {
    "upperarm_l": ("l_shoulder", "l_elbow", "lowerarm_l"),
    "lowerarm_l": ("l_elbow", "l_wrist", "hand_l"),
    "upperarm_r": ("r_shoulder", "r_elbow", "lowerarm_r"),
    "lowerarm_r": ("r_elbow", "r_wrist", "hand_r"),
    "thigh_l": ("l_hip", "l_knee", "calf_l"),
    "calf_l": ("l_knee", "l_ankle", "foot_l"),
    "thigh_r": ("r_hip", "r_knee", "calf_r"),
    "calf_r": ("r_knee", "r_ankle", "foot_r"),
    "spine_01": ("pelvis", "chest", "spine_03"),
    "neck_01": ("chest", "head", "Head"),
}

_MIN_DIR = 1e-9


# --------------------------------------------------------------------------- #
# Part 1 -- the bridge: reconstructed skeleton -> rig-node local rotations
# --------------------------------------------------------------------------- #

def _topo_order(skin: skinning.SkinData) -> list[int]:
    """All nodes that matter (skin joints + ancestors), ordered parents-first."""
    needed: set[int] = set()
    for j in skin.skin_joints:
        n = j
        while n is not None and n not in needed:
            needed.add(n)
            n = skin.parent.get(n)
    # Stable parent-first order: repeatedly emit nodes whose parent is already out.
    order: list[int] = []
    emitted: set[int] = set()
    remaining = set(needed)
    while remaining:
        progressed = False
        for n in sorted(remaining):
            p = skin.parent.get(n)
            if p is None or p in emitted or p not in needed:
                order.append(n)
                emitted.add(n)
                remaining.discard(n)
                progressed = True
        if not progressed:                       # cycle / detached -- emit the rest
            order.extend(sorted(remaining))
            break
    return order


def _bind_frames(skin: skinning.SkinData):
    """Precompute, once per rig, the bind globals + per-node bind/local rotations.

    Returns ``(Rbind, Lbind, dbind)`` dicts keyed by glTF node index:

    * ``Rbind[n]`` -- rotation part (3x3) of the node's bind global transform,
    * ``Lbind[n]`` -- rotation part (3x3) of the node's bind local TRS,
    * ``dbind[n]`` -- bind world bone direction (child origin - node origin),
      only for driven nodes whose child exists.
    """
    g = skinning._animated_globals(skin, {})
    Rbind = {n: g[n][:3, :3].copy() for n in g}
    Lbind = {n: skin.local[n][:3, :3].copy() for n in g}
    dbind: dict[int, np.ndarray] = {}
    for name, (head, tail, _child) in _DRIVEN.items():
        n = skin.node_name.get(name)
        hn = skin.node_name.get(QUATERNIUS_JOINTS[head])
        tn = skin.node_name.get(QUATERNIUS_JOINTS[tail])
        if n is None or hn is None or tn is None:
            continue
        d = g[tn][:3, 3] - g[hn][:3, 3]          # bind dir between the joint nodes
        nd = np.linalg.norm(d)
        if nd >= _MIN_DIR:
            dbind[n] = d / nd
    return Rbind, Lbind, dbind


def rig_pose_from_joints(skin: skinning.SkinData, jp: np.ndarray,
                         jv: np.ndarray) -> dict[str, np.ndarray]:
    """Solve per-rig-node extra **local** rotations from OUR 17 joint positions.

    *jp* is ``(17, 3)`` world joint positions, *jv* ``(17,)`` validity (the order
    of :data:`capmo.retarget.JOINTS`). Each driven bone is aimed at its measured
    head->tail direction by the minimal swing from its bind world direction; the
    swing is then expressed as a rotation **local** to the bone (dividing out the
    parent's animated global) so that skinning's ``G = G_parent @ local @ extra``
    reproduces the desired global. The returned dict maps each driven rig *node
    name* (only those that actually got a swing this frame) to a 3x3 rotation,
    suitable as ``bone_rotations`` for :func:`capmo.skinning.pose_rig_mesh`.

    Pure swings are exactly recoverable, so for a swing-only ground-truth pose
    this is an exact inverse of :func:`capmo.skinning.pose_rig_mesh` (see tests).
    """
    jp = np.asarray(jp, np.float64)
    jv = np.asarray(jv, bool)
    Rbind, Lbind, dbind = _bind_frames(skin)

    # name lookups for driven nodes
    node_of = {name: skin.node_name[name] for name in _DRIVEN
               if name in skin.node_name}
    driven_by_idx = {node_of[name]: name for name in node_of}

    Ranim: dict[int, np.ndarray] = {}            # animated global rotation per node
    extra_out: dict[str, np.ndarray] = {}

    for n in _topo_order(skin):
        p = skin.parent.get(n)
        Rap = Ranim[p] if p is not None and p in Ranim else np.eye(3)

        swung = False
        if n in driven_by_idx and n in dbind:
            head, tail, _ = _DRIVEN[driven_by_idx[n]]
            hi, ti = JOINT_INDEX[head], JOINT_INDEX[tail]
            if jv[hi] and jv[ti]:
                m = jp[ti] - jp[hi]
                nm = np.linalg.norm(m)
                if nm >= _MIN_DIR:
                    dmeas = m / nm
                    D = quat_to_matrix(quat_from_two_vectors(dbind[n], dmeas))
                    target_global = D @ Rbind[n]
                    swung = True

        if not swung:                            # hold bind: extra = identity
            target_global = Rap @ Lbind[n]

        # extra = Lbind^T @ Rap^T @ target_global  (so G_parent@local@extra == target)
        extra = Lbind[n].T @ Rap.T @ target_global
        Ranim[n] = target_global
        if swung:
            extra_out[driven_by_idx[n]] = extra
    return extra_out


def rig_pose_from_skeleton(skin: skinning.SkinData, points: np.ndarray,
                           valid: np.ndarray) -> dict[str, np.ndarray]:
    """33-landmark convenience wrapper for :func:`rig_pose_from_joints`."""
    jp, jv = landmarks_to_joints(points, valid)
    return rig_pose_from_joints(skin, jp, jv)


def animate_joints(skin: skinning.SkinData, joints: np.ndarray, valid: np.ndarray, *,
                   alpha: float = 0.0) -> list[dict[str, np.ndarray]]:
    """Per-frame rig poses for a 3D landmark track (the ``joints3d`` contract:
    *joints* (T,33,3), *valid* (T,33), in the rig's +Y-up frame -- use
    :meth:`capmo.core.joints3d.Joints3D.to_y_up` on a ``reconstruct_take`` file).

    Each output is a ``{rig node name: 3x3}`` pose for
    :func:`capmo.skinning.pose_rig_mesh`. A driven node that drops out (invalid /
    degenerate this frame) **holds its last** solved rotation. With *alpha* in
    ``(0, 1]`` a causal slerp low-pass damps jitter on each node's rotation:
    *alpha* is the weight of the new frame (1 = no smoothing, smaller = smoother,
    0 = off).
    """
    joints = np.asarray(joints, np.float64)
    valid = np.asarray(valid, bool)
    raw = [rig_pose_from_skeleton(skin, joints[i], valid[i]) for i in range(len(joints))]
    return _hold_and_smooth(raw, alpha)


def animate_track(skin: skinning.SkinData, frames, *,
                  alpha: float = 0.0) -> list[dict[str, np.ndarray]]:
    """:func:`animate_joints` over per-frame objects exposing
    ``.skeleton.points`` (33, 3) and ``.skeleton.valid`` (33,)."""
    raw = [rig_pose_from_skeleton(skin, f.skeleton.points, f.skeleton.valid)
           for f in frames]
    return _hold_and_smooth(raw, alpha)


def _hold_and_smooth(raw, alpha):
    # Hold-last for nodes that drop out (so the figure doesn't snap to bind).
    held: list[dict[str, np.ndarray]] = []
    last: dict[str, np.ndarray] = {}
    for pose in raw:
        merged = dict(last)
        merged.update(pose)
        held.append({k: v.copy() for k, v in merged.items()})
        last = merged

    if alpha and held:
        a = float(np.clip(alpha, 0.0, 1.0))
        prev = held[0]
        for pose in held[1:]:
            for name, R in pose.items():
                if name in prev:
                    q = quat_slerp(quat_from_matrix(prev[name]),
                                   quat_from_matrix(R), a)
                    pose[name] = quat_to_matrix(q)
            prev = pose
    return held


# --------------------------------------------------------------------------- #
# Part 2 -- software character rendering (GL-free)
# --------------------------------------------------------------------------- #

def _lambert_shade(verts: np.ndarray, tris: np.ndarray,
                   base=(150, 180, 200), light=(0.3, 0.5, 1.0),
                   ambient: float = 0.35) -> np.ndarray:
    """Per-triangle Lambert intensity over *base* (BGR), as the E2E render uses."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    n = n / norm
    L = np.asarray(light, np.float64)
    L = L / np.linalg.norm(L)
    inten = ambient + (1.0 - ambient) * np.clip(n @ L, 0.0, 1.0)
    col = np.asarray(base, np.float64)[None, :] * inten[:, None]
    return np.clip(col, 0, 255).astype(np.uint8)


def render_character(skin: skinning.SkinData, bone_rotations: dict[str, np.ndarray],
                     K: np.ndarray, R: np.ndarray, t: np.ndarray,
                     W: int, H: int, *, base_color=(150, 180, 200),
                     bg: int = 128) -> np.ndarray:
    """Pose, Lambert-shade and rasterize the character -> HxWx3 uint8 BGR image."""
    verts = skinning.pose_rig_mesh(skin, bone_rotations or {})
    mesh = synth.Mesh(verts, skin.tris, _lambert_shade(verts, skin.tris, base=base_color))
    return synth.rasterize(mesh, K, R, t, W, H, bg=bg)


def _default_camera(skin: skinning.SkinData, W: int, H: int, *,
                    cam_eye=None, cam_target=None):
    """A front-facing pinhole framed on the rig's bind bounding box."""
    bind = skin.verts
    lo, hi = bind.min(0), bind.max(0)
    center = np.asarray(cam_target, np.float64) if cam_target is not None \
        else (lo + hi) / 2.0
    height = float(hi[1] - lo[1]) or 1.0
    if cam_eye is not None:
        eye = np.asarray(cam_eye, np.float64)
    else:
        eye = center + np.array([0.0, 0.0, height * 1.8])
    R, t = synth.look_at(eye, center)
    f = 1.05 * max(W, H)                          # frame the figure with a margin
    K = synth.pinhole_K(f, f, W / 2.0, H / 2.0)
    return K, R, t


def render_track(skin: skinning.SkinData, rig_pose_track, *,
                 cam_eye=None, cam_target=None, W: int = 480, H: int = 640,
                 base_color=(150, 180, 200), bg: int = 128) -> list[np.ndarray]:
    """Render a whole rig-pose track to a list of BGR images (fixed camera)."""
    K, R, t = _default_camera(skin, W, H, cam_eye=cam_eye, cam_target=cam_target)
    return [render_character(skin, pose, K, R, t, W, H,
                             base_color=base_color, bg=bg)
            for pose in rig_pose_track]
