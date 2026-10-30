"""
Retarget a triangulated 3D pose track onto a humanoid character.

The reconstruction (``reconstruct_take`` / :mod:`capmo.core.pipeline`) gives us,
per instant, a set of 3D joint *positions* (33 MediaPipe landmarks, the
``joints3d`` contract of :mod:`capmo.core.joints3d`).
A character armature is not driven by positions but by **bone rotations** about a
moving root, so this module converts the position track into a pose track an
engine (Godot/Blender, the next phase) can apply directly:

    joints3d (T, 33 landmark points)  ->  CharacterPose (root + per-bone rotations)

How it works
------------
We define a fixed humanoid armature (:func:`default_skeleton`) from a canonical
T-pose: an ordered list of bones, each with a parent, a head/tail joint, and the
bone's *rest direction* and *rest length* in the bind pose. The character keeps
its own proportions (rest lengths); only the joint *angles* are copied from the
captured person.

For each frame:

* the **root** (pelvis) gets a full 3-DOF orientation built from the body's
  hip/spine basis, plus a world position -- so the character's facing is kept;
* every **bone** is aimed at its measured direction by the minimal rotation that
  carries its rest direction onto ``normalize(tail - head)`` (a *swing*; bone
  roll/twist is not observable from joint positions alone), then stored as a
  rotation **local** to its parent.

Because each local rotation is derived by dividing out its parent's global
rotation, forward kinematics (:func:`forward_kinematics`) reproduces every bone's
measured direction exactly -- the round-trip the tests check. Joints the cameras
did not solve leave their bone at the rest pose (and flagged invalid); the track
helpers fill such gaps and optionally smooth the result.

Pure numpy + stdlib, fully unit-tested. The quaternion helpers are generic enough
to lift into a shared layer in the eventual synclip merge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Quaternion helpers (w, x, y, z). Generic -- candidates for a shared layer.
# --------------------------------------------------------------------------- #

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return IDENTITY_QUAT.copy()
    q = q / n
    return q if q[0] >= 0 else -q          # keep a canonical (w >= 0) sign


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector *v* by unit quaternion *q*."""
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit-ish *a* onto unit-ish *b* (a swing)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return IDENTITY_QUAT.copy()
    a = a / na
    b = b / nb
    d = float(np.dot(a, b))
    if d >= 1.0 - 1e-9:
        return IDENTITY_QUAT.copy()
    if d <= -1.0 + 1e-9:                    # opposite: 180 deg about any perp axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return quat_normalize(np.array([0.0, *axis]))
    axis = np.cross(a, b)
    w = 1.0 + d
    return quat_normalize(np.array([w, *axis]))


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Quaternion from a 3x3 rotation matrix (columns = rotated basis)."""
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from unit quaternion *q* (w, x, y, z)."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation from *a* to *b* by fraction *t* in [0, 1]."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:                              # take the shorter arc
        b = -b
        d = -d
    if d > 1.0 - 1e-7:                        # nearly aligned -> lerp
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(d)
    s = np.sin(theta)
    return quat_normalize((np.sin((1 - t) * theta) / s) * a + (np.sin(t * theta) / s) * b)


# --------------------------------------------------------------------------- #
# Humanoid skeleton definition
# --------------------------------------------------------------------------- #

# Joints we drive, and which MediaPipe Pose landmark(s) give each one's position
# (a tuple is averaged; a joint is valid only if every source landmark is).
JOINTS: list[tuple[str, tuple[int, ...]]] = [
    ("pelvis", (23, 24)),       # 0  root
    ("chest", (11, 12)),        # 1
    ("head", (7, 8)),           # 2  (ear midpoint)
    ("l_shoulder", (11,)),      # 3
    ("l_elbow", (13,)),         # 4
    ("l_wrist", (15,)),         # 5
    ("r_shoulder", (12,)),      # 6
    ("r_elbow", (14,)),         # 7
    ("r_wrist", (16,)),         # 8
    ("l_hip", (23,)),           # 9
    ("l_knee", (25,)),          # 10
    ("l_ankle", (27,)),         # 11
    ("l_foot", (31,)),          # 12
    ("r_hip", (24,)),           # 13
    ("r_knee", (26,)),          # 14
    ("r_ankle", (28,)),         # 15
    ("r_foot", (32,)),          # 16
]
JOINT_INDEX = {name: i for i, (name, _) in enumerate(JOINTS)}

# Bones as (name, head joint, tail joint, parent bone | None). A child bone's
# head must equal its parent's tail so positions chain; root-attached bones
# (parent None) hang off the pelvis by a fixed rest offset.
_BONES = [
    ("spine",      "pelvis",     "chest",      None),
    ("neck",       "chest",      "head",       "spine"),
    ("l_clavicle", "chest",      "l_shoulder", "spine"),
    ("l_upperarm", "l_shoulder", "l_elbow",    "l_clavicle"),
    ("l_forearm",  "l_elbow",    "l_wrist",    "l_upperarm"),
    ("r_clavicle", "chest",      "r_shoulder", "spine"),
    ("r_upperarm", "r_shoulder", "r_elbow",    "r_clavicle"),
    ("r_forearm",  "r_elbow",    "r_wrist",    "r_upperarm"),
    ("l_thigh",    "l_hip",      "l_knee",     None),
    ("l_shin",     "l_knee",     "l_ankle",    "l_thigh"),
    ("l_foot",     "l_ankle",    "l_foot",     "l_shin"),
    ("r_thigh",    "r_hip",      "r_knee",     None),
    ("r_shin",     "r_knee",     "r_ankle",    "r_thigh"),
    ("r_foot",     "r_ankle",    "r_foot",     "r_shin"),
]

# Canonical T-pose joint positions (metres, +x = character's right, +y = up,
# +z = forward). Built so the body basis equals the world axes, i.e. the bind
# pose has an identity root orientation.
_BIND = {
    "pelvis": (0.0, 0.0, 0.0),
    "chest": (0.0, 0.50, 0.0),
    "head": (0.0, 0.75, 0.0),
    "l_shoulder": (-0.18, 0.50, 0.0), "l_elbow": (-0.45, 0.50, 0.0),
    "l_wrist": (-0.70, 0.50, 0.0),
    "r_shoulder": (0.18, 0.50, 0.0), "r_elbow": (0.45, 0.50, 0.0),
    "r_wrist": (0.70, 0.50, 0.0),
    "l_hip": (-0.10, 0.0, 0.0), "l_knee": (-0.10, -0.45, 0.0),
    "l_ankle": (-0.10, -0.90, 0.0), "l_foot": (-0.10, -0.95, 0.16),
    "r_hip": (0.10, 0.0, 0.0), "r_knee": (0.10, -0.45, 0.0),
    "r_ankle": (0.10, -0.90, 0.0), "r_foot": (0.10, -0.95, 0.16),
}


@dataclass
class Skeleton:
    """A humanoid armature: ordered bones with bind-pose geometry.

    All per-bone arrays are indexed by bone order (parents precede children).
    """
    bone_names: list[str]
    head_joint: np.ndarray      # (B,) joint index a bone starts at
    tail_joint: np.ndarray      # (B,) joint index a bone ends at
    parent: np.ndarray          # (B,) parent bone index, -1 if root-attached
    rest_dir: np.ndarray        # (B, 3) unit bind-pose direction (head -> tail)
    rest_len: np.ndarray        # (B,) bind-pose bone length
    root_offset: np.ndarray     # (B, 3) head offset from pelvis (root-attached only)
    bind_joints: np.ndarray     # (J, 3) bind-pose joint positions

    @property
    def num_bones(self) -> int:
        return len(self.bone_names)


def default_skeleton(bind: dict | None = None) -> Skeleton:
    """Build the humanoid :class:`Skeleton` from a bind pose (T-pose default)."""
    bind = bind or _BIND
    bind_joints = np.array([bind[name] for name, _ in JOINTS], np.float64)
    pelvis = bind_joints[JOINT_INDEX["pelvis"]]
    # Root-attached head offsets live in the root's local frame, so a non-axis-
    # aligned bind pose (its body basis != world) still round-trips through FK.
    bind_valid = np.ones(len(bind_joints), bool)
    bind_basis = _body_basis(bind_joints, bind_valid)
    q_root_bind = quat_from_matrix(bind_basis) if bind_basis is not None else IDENTITY_QUAT

    names, head, tail, parent = [], [], [], []
    rest_dir, rest_len, root_off = [], [], []
    name_to_idx: dict[str, int] = {}
    for name, h, t, par in _BONES:
        hi, ti = JOINT_INDEX[h], JOINT_INDEX[t]
        d = bind_joints[ti] - bind_joints[hi]
        length = float(np.linalg.norm(d))
        names.append(name)
        head.append(hi)
        tail.append(ti)
        parent.append(-1 if par is None else name_to_idx[par])
        rest_dir.append(d / length if length > 1e-9 else np.array([0.0, 1.0, 0.0]))
        rest_len.append(length)
        offset = (quat_rotate(quat_conj(q_root_bind), bind_joints[hi] - pelvis)
                  if par is None else np.zeros(3))
        root_off.append(offset)
        name_to_idx[name] = len(names) - 1

    return Skeleton(
        bone_names=names,
        head_joint=np.array(head, int), tail_joint=np.array(tail, int),
        parent=np.array(parent, int),
        rest_dir=np.array(rest_dir, np.float64), rest_len=np.array(rest_len, np.float64),
        root_offset=np.array(root_off, np.float64), bind_joints=bind_joints,
    )


# --------------------------------------------------------------------------- #
# Joint extraction + the per-frame pose
# --------------------------------------------------------------------------- #

@dataclass
class CharacterPose:
    """One frame of character animation."""
    t_ms: float
    root_pos: np.ndarray        # (3,) pelvis world position (nan if unsolved)
    root_rot: np.ndarray        # (4,) pelvis orientation quaternion (w,x,y,z)
    local_rot: np.ndarray       # (B, 4) per-bone rotation local to its parent
    bone_valid: np.ndarray      # (B,) bool: bone was solved this frame
    root_valid: bool            # the body basis was solvable this frame


def landmarks_to_joints(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a (33, 3) landmark set + validity to (J, 3) joint positions + validity."""
    points = np.asarray(points, np.float64)
    valid = np.asarray(valid, bool)
    J = len(JOINTS)
    jp = np.full((J, 3), np.nan)
    jv = np.zeros(J, bool)
    for i, (_, src) in enumerate(JOINTS):
        if all(valid[s] for s in src):
            jp[i] = np.mean(points[list(src)], axis=0)
            jv[i] = True
    return jp, jv


def _body_basis(jp: np.ndarray, jv: np.ndarray) -> np.ndarray | None:
    """Right/up/forward basis (columns) from the hip+spine, or None if unsolved."""
    need = ("pelvis", "chest", "l_hip", "r_hip")
    if not all(jv[JOINT_INDEX[n]] for n in need):
        return None
    pelvis = jp[JOINT_INDEX["pelvis"]]
    up = jp[JOINT_INDEX["chest"]] - pelvis
    lateral = jp[JOINT_INDEX["r_hip"]] - jp[JOINT_INDEX["l_hip"]]
    nu = np.linalg.norm(up)
    if nu < 1e-9:
        return None
    up = up / nu
    fwd = np.cross(lateral, up)
    nf = np.linalg.norm(fwd)
    if nf < 1e-9:
        return None
    fwd = fwd / nf
    right = np.cross(up, fwd)
    return np.column_stack([right, up, fwd])


def retarget_pose(skel: Skeleton, points: np.ndarray, valid: np.ndarray,
                  t_ms: float = 0.0, min_dir_len: float = 1e-6) -> CharacterPose:
    """Retarget one frame of 3D landmarks onto the character armature."""
    jp, jv = landmarks_to_joints(points, valid)

    basis = _body_basis(jp, jv)
    root_valid = basis is not None
    root_rot = quat_from_matrix(basis) if root_valid else IDENTITY_QUAT.copy()
    pelvis_i = JOINT_INDEX["pelvis"]
    root_pos = jp[pelvis_i].copy() if jv[pelvis_i] else np.full(3, np.nan)

    B = skel.num_bones
    local = np.tile(IDENTITY_QUAT, (B, 1)).astype(np.float64)
    bone_valid = np.zeros(B, bool)
    glob = [None] * B                        # FK-consistent global rotation per bone
    for b in range(B):
        par = skel.parent[b]
        parent_global = root_rot if par < 0 else glob[par]
        h, t = skel.head_joint[b], skel.tail_joint[b]
        if jv[h] and jv[t]:
            m = jp[t] - jp[h]
            if np.linalg.norm(m) >= min_dir_len:
                g = quat_from_two_vectors(skel.rest_dir[b], m)
                local[b] = quat_mul(quat_conj(parent_global), g)
                bone_valid[b] = True
                glob[b] = g
        if glob[b] is None:                  # unsolved -> rest pose, FK stays consistent
            glob[b] = parent_global
    return CharacterPose(t_ms=t_ms, root_pos=root_pos, root_rot=root_rot,
                         local_rot=local, bone_valid=bone_valid, root_valid=root_valid)


def forward_kinematics(skel: Skeleton, pose: CharacterPose) -> np.ndarray:
    """Joint world positions (J, 3) produced by applying *pose* to *skel*.

    The inverse of :func:`retarget_pose`: each bone is aimed by its global
    rotation (parent global * local) and stepped by the bind-pose bone length, so
    the result carries the captured *angles* with the character's proportions.
    """
    B = skel.num_bones
    root_pos = pose.root_pos if np.all(np.isfinite(pose.root_pos)) else np.zeros(3)
    jp = np.full((len(skel.bind_joints), 3), np.nan)
    jp[JOINT_INDEX["pelvis"]] = root_pos
    glob = [None] * B
    head = np.zeros((B, 3))
    for b in range(B):
        par = skel.parent[b]
        if par < 0:
            glob[b] = quat_mul(pose.root_rot, pose.local_rot[b])
            head[b] = root_pos + quat_rotate(pose.root_rot, skel.root_offset[b])
        else:
            glob[b] = quat_mul(glob[par], pose.local_rot[b])
            head[b] = jp[skel.tail_joint[par]]
        tail = head[b] + skel.rest_len[b] * quat_rotate(glob[b], skel.rest_dir[b])
        jp[skel.head_joint[b]] = head[b]
        jp[skel.tail_joint[b]] = tail
    return jp


# --------------------------------------------------------------------------- #
# Track-level retarget (gap fill + smoothing)
# --------------------------------------------------------------------------- #

def retarget_joints(joints: np.ndarray, valid: np.ndarray, t_ms=None,
                    skel: Skeleton | None = None, *, fill_gaps: bool = True,
                    smooth: float = 0.0, root_scale: float = 1.0) -> list[CharacterPose]:
    """Retarget a whole 3D landmark track (the ``joints3d`` contract, see
    :mod:`capmo.core.joints3d`: *joints* (T,33,3), *valid* (T,33); *t_ms* (T,)
    or None for frame numbers). The track must be in the armature's +Y-up
    frame (``Joints3D.to_y_up()`` for a ``reconstruct_take`` file).

    With *fill_gaps* an unsolved bone holds its last solved rotation (and the
    root holds its last position/facing); *smooth* in (0, 1] applies a causal
    slerp/lerp low-pass to damp jitter.

    Bone rotations are inherently proportion-free -- they ride on *skel*'s own
    rest lengths -- so retargeting a tall capture onto a short rig just needs
    *skel* to be the target rig (see :func:`capmo.character_rig.load_rig_skeleton`).
    The one positional output, the root translation, carries the capture's scale,
    so *root_scale* rescales it into the rig's units (e.g. target/source torso
    length, :func:`capmo.character_rig.root_scale_for`).
    """
    skel = skel or default_skeleton()
    joints = np.asarray(joints, np.float64)
    valid = np.asarray(valid, bool)
    tt = np.arange(len(joints), dtype=np.float64) if t_ms is None else np.asarray(t_ms, np.float64)
    poses = [retarget_pose(skel, joints[i], valid[i], t_ms=float(tt[i]))
             for i in range(len(joints))]
    return _finish_track(poses, skel, fill_gaps, smooth, root_scale)


def retarget_track(frames, skel: Skeleton | None = None, *,
                   fill_gaps: bool = True, smooth: float = 0.0,
                   root_scale: float = 1.0) -> list[CharacterPose]:
    """:func:`retarget_joints` over per-frame objects exposing ``t_ms`` and
    ``skeleton.points`` (33,3) / ``skeleton.valid`` (33,)."""
    frames = list(frames)
    if not frames:
        return []
    J = np.stack([np.asarray(f.skeleton.points, np.float64) for f in frames])
    V = np.stack([np.asarray(f.skeleton.valid, bool) for f in frames])
    return retarget_joints(J, V, [f.t_ms for f in frames], skel, fill_gaps=fill_gaps,
                           smooth=smooth, root_scale=root_scale)


def _finish_track(poses, skel, fill_gaps, smooth, root_scale):
    if fill_gaps:
        last_local = None
        last_root_pos = None
        last_root_rot = None
        for p in poses:
            if last_local is not None:
                miss = ~p.bone_valid
                p.local_rot[miss] = last_local[miss]
                if not p.root_valid:
                    p.root_rot = last_root_rot.copy()
                if not np.all(np.isfinite(p.root_pos)) and last_root_pos is not None:
                    p.root_pos = last_root_pos.copy()
            last_local = p.local_rot.copy()
            if p.root_valid:
                last_root_rot = p.root_rot.copy()
            if np.all(np.isfinite(p.root_pos)):
                last_root_pos = p.root_pos.copy()

    if smooth and poses:
        a = float(np.clip(smooth, 0.0, 1.0))
        prev = poses[0]
        for p in poses[1:]:
            for b in range(skel.num_bones):
                p.local_rot[b] = quat_slerp(prev.local_rot[b], p.local_rot[b], a)
            p.root_rot = quat_slerp(prev.root_rot, p.root_rot, a)
            if np.all(np.isfinite(p.root_pos)) and np.all(np.isfinite(prev.root_pos)):
                p.root_pos = prev.root_pos + a * (p.root_pos - prev.root_pos)
            prev = p

    if root_scale != 1.0:
        for p in poses:
            p.root_pos = p.root_pos * root_scale
    return poses


# --------------------------------------------------------------------------- #
# Serialization -- an engine-ready animation clip
# --------------------------------------------------------------------------- #

def clip_to_dict(skel: Skeleton, poses: list[CharacterPose]) -> dict:
    """Serialise the armature + pose track to a plain dict (JSON-ready)."""
    return {
        "format": "capmo.character_clip/1",
        "joints": [name for name, _ in JOINTS],
        "skeleton": {
            "bones": skel.bone_names,
            "head_joint": skel.head_joint.tolist(),
            "tail_joint": skel.tail_joint.tolist(),
            "parent": skel.parent.tolist(),
            "rest_dir": skel.rest_dir.tolist(),
            "rest_len": skel.rest_len.tolist(),
            "root_offset": skel.root_offset.tolist(),
            "bind_joints": skel.bind_joints.tolist(),
        },
        "frames": [
            {
                "t_ms": p.t_ms,
                "root_pos": [None if not np.isfinite(v) else float(v) for v in p.root_pos],
                "root_rot": p.root_rot.tolist(),
                "root_valid": bool(p.root_valid),
                "local_rot": p.local_rot.tolist(),
                "bone_valid": p.bone_valid.astype(bool).tolist(),
            }
            for p in poses
        ],
    }


def save_clip(path: str, skel: Skeleton, poses: list[CharacterPose]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clip_to_dict(skel, poses), f)
