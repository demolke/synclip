"""MediaPipe-33 joint-position mocap -> Blender Rigify FK retargeting.

Works three ways:

* Blender add-on: install / run this file, an N-panel "Mocap Retarget" appears in
  the 3D viewport sidebar (Retarget / Re-bake button, live scrub mode, all the
  tweakable parameters).  Settings are stored per rig object (Object.mocap_retarget),
  so several characters in one scene are tweaked / re-baked independently.
  embed_text_module() / embed_npz() (CLI: --embed) store this module and the mocap in
  the .blend, so the panel and Re-bake work in a self-contained file.
* Python API (headless or inside Blender)::

      import retarget as mr
      prm = mr.Params(smooth_sigma=2.0)
      info = mr.retarget(rig_obj, "/path/take_joints3d.npz", prm)

* CLI::

      /opt/bpyenv/bin/python retarget.py --blend rig.blend --npz take.npz \
          --out retargeted.blend [--rig rig] [--embed] [--set smooth_sigma=2 foot_lock=0 ...]

Method (see README.md): every body segment gets an orthonormal frame built
from the joint positions (primary axis = segment direction, secondary axis =
a stable anatomical twist reference).  The *same* construction applied to
landmarks measured on the target rig's rest pose gives the rest frame, so the
world-space delta rotation D = F_pose * F_rest^T maps rest -> pose with the
correct twist and no rest-pose bias.  Deltas are made local to the parent,
split into swing/twist, hinge-projected (elbow, knee), clamped to anatomical
limits and finally converted to the local quaternion of the Rigify FK
control.  Only rotations (+ the torso location) are keyed, the bone lengths
of the target rig are never changed -> zero stretch by construction.

Params.control_mode = LEGS_IK_ARMS_FK / BOTH additionally keys Rigify's IK controls
(hand_ik / foot_ik / toe_ik / pole targets), computed analytically from the same FK solution
(ik_bases()), so the IK_FK sliders switch between two identical poses.
"""

bl_info = {
    "name": "Mocap Retarget (MediaPipe -> Rigify)",
    "author": "synclip",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Mocap Retarget",
    "description": "Retarget 33-joint position mocap onto Rigify FK (and optionally IK) controls",
    "category": "Animation",
}

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict, fields

import numpy as np

try:
    import bpy
    from bpy.app.handlers import persistent
except ImportError:  # pure numpy use (tests)
    bpy = None

    def persistent(f):
        return f


# =============================================================== constants
MP = {n: i for i, n in enumerate([
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"])}
SIDES = (("L", "left"), ("R", "right"))

# control -> (logical parent control, segment whose world delta it carries)
CONTROLS = [
    ("torso", None, "mid"),
    ("spine_fk.001", "torso", "lowmid"),
    ("spine_fk", "spine_fk.001", "pelvis"),
    ("spine_fk.002", "torso", "upmid"),
    ("spine_fk.003", "spine_fk.002", "chest"),
    ("neck", "spine_fk.003", "neck"),
    ("head", "neck", "head"),
]
for _s, _ in SIDES:
    CONTROLS += [
        ("shoulder." + _s, "spine_fk.003", "clav_" + _s),
        ("upper_arm_fk." + _s, "shoulder." + _s, "uarm_" + _s),
        ("forearm_fk." + _s, "upper_arm_fk." + _s, "farm_" + _s),
        ("hand_fk." + _s, "forearm_fk." + _s, "hand_" + _s),
        ("thigh_fk." + _s, "spine_fk", "thigh_" + _s),
        ("shin_fk." + _s, "thigh_fk." + _s, "shin_" + _s),
        ("foot_fk." + _s, "shin_fk." + _s, "foot_" + _s),
        ("toe_fk." + _s, "foot_fk." + _s, "toe_" + _s),
    ]
CTRL_PARENT = {c: p for c, p, _ in CONTROLS}
CTRL_SEG = {c: s for c, _, s in CONTROLS}

# per-limb offset groups -> segment they rotate (in the control's rest frame)
OFFSET_GROUPS = {
    "spine": "chest", "head": "head",
    "arm_L": "uarm_L", "forearm_L": "farm_L", "hand_L": "hand_L",
    "leg_L": "thigh_L", "shin_L": "shin_L", "foot_L": "foot_L",
    "arm_R": "uarm_R", "forearm_R": "farm_R", "hand_R": "hand_R",
    "leg_R": "thigh_R", "shin_R": "shin_R", "foot_R": "foot_R",
}
SEG_CTRL = {s: c for c, _, s in CONTROLS}

# Rigify IK limb controls (suffix .L / .R).  The IK keys are computed from the FK solution:
#   end   : IK control rigidly attached to the FK end control (hand_ik <- hand_fk, foot_ik <- foot_fk)
#   pole  : pole target (pole_vector = 1) rigidly attached to the upper FK control
#   toe   : toe_ik rotation = toe_fk's rotation relative to the foot
#   rest  : controls kept at identity (the IK chain base, foot roll controls)
#   chain : bones whose bone-level ik_stretch is zeroed (ik_zero_bone_stretch)
IK_LIMBS = {
    "arm": dict(prop="upper_arm_parent", end="hand_ik", end_fk="hand_fk", pole="upper_arm_ik_target",
                upper_fk="upper_arm_fk", toe=None, toe_fk=None, foot_fk=None,
                rest=("upper_arm_ik",), chain=("upper_arm_ik", "MCH-forearm_ik")),
    "leg": dict(prop="thigh_parent", end="foot_ik", end_fk="foot_fk", pole="thigh_ik_target",
                upper_fk="thigh_fk", toe="toe_ik", toe_fk="toe_fk", foot_fk="foot_fk",
                rest=("thigh_ik", "foot_heel_ik", "foot_spin_ik"), chain=("thigh_ik", "MCH-shin_ik")),
}
CONTROL_MODES = ("FK", "LEGS_IK_ARMS_FK", "BOTH")
# IK_FK value per mode and limb kind (1 = FK, 0 = IK)
IK_FK_SWITCH = {"FK": {"arm": 1.0, "leg": 1.0},
                "LEGS_IK_ARMS_FK": {"arm": 1.0, "leg": 0.0},
                "BOTH": {"arm": 1.0, "leg": 1.0}}


# =============================================================== parameters
@dataclass
class Params:
    # input / timing
    frame_start: int = 0            # first mocap frame used
    frame_end: int = -1             # last mocap frame (inclusive), -1 = all
    step: int = 1                   # subsampling: key every n-th mocap frame
    scene_start: int = 1            # scene frame of the first key
    mocap_yaw: float = 0.0          # deg, rotate input about Z
    mocap_space: str = "WORLD"      # WORLD: the mocap's coordinates are scene world coordinates (moving the rig
                                    # object does not move the performance); OBJECT: the mocap's world frame is
                                    # attached to the rig object (several characters side by side: just move
                                    # each rig object)
    # filtering
    reject_outliers: float = 0.45   # drop joints whose bone length deviates > this ratio from the median (0 = off)
    despike: int = 5                # median window (frames, odd, 0/1 = off)
    smooth_sigma: float = 1.5       # gaussian sigma on joint positions (frames)
    rot_smooth: float = 1.0         # gaussian sigma on solved rotations (frames)
    fix_lr_swaps: bool = True        # repair frames whose left / right labels are swapped
    gap_erode: int = 2               # frames dropped next to tracking gaps (>= 5 frames)
    infer_knees: bool = True         # missing knee with hip + ankle observed: place it by 2-bone IK
    gap_fill_rotations: bool = True  # unobserved frames: slerp solved rotations instead of solving interpolated joints
    max_ang_vel: float = 30.0       # deg/frame cap on any control's local rotation speed (0 = off)
    rot_despike: float = 35.0       # deg: local-rotation jumps that revert within 6 frames are removed (0 = off)
    # trunk / head
    pelvis_follow: float = 0.6      # 0 = pelvis stays upright, 1 = follows the trunk
    trunk_lateral: float = 0.5     # lateral trunk lean: 0 = keep the shoulder line, 1 = match the trunk
    head_face_weight: float = 0.5   # head pitch: 1 = from nose/eyes vs ears, 0 = from the neck line
    neck_share: float = 0.4         # part of the head rotation done by the neck
    head_limit: float = 75.0        # deg, head vs chest swing / twist
    clavicle_share: float = 0.25    # part of the arm elevation above horizontal given to the clavicle
    # limbs
    twist_mix: float = 0.8          # forearm twist (pronation) taken from the hand plane
    forearm_twist_limit: float = 90.0
    arm_total_twist_limit: float = 150.0  # upper arm + forearm twist (Rigify's twist segment flips near 180)
    upperarm_twist_limit: float = 95.0
    thigh_twist_limit: float = 60.0
    shin_twist_limit: float = 20.0
    leg_twist_mix: float = 0.5      # shin (tibial) twist taken from the foot direction
    twist_rate_limit: float = 20.0  # deg/frame max change of any limb/head twist
    continuity: float = 0.25        # deg of fit error traded per deg of twist change between frames
    hinge_strict: float = 1.0       # 1 = elbow/knee bend only around the hinge axis
    hyperextend: float = 5.0        # deg of allowed elbow / knee hyper-extension
    elbow_max: float = 150.0
    knee_max: float = 155.0
    bend_lo: float = 3.0            # deg: below -> limb twist from the fallback reference
    bend_hi: float = 15.0           # deg: above -> limb twist fully from the bend plane
    hand_mix: float = 0.8           # 0 = hands keep forearm orientation
    hand_limit: float = 75.0
    foot_limit: float = 60.0
    # root
    root_auto_scale: bool = True    # scale translation by the leg-length ratio
    root_scale: float = 1.0         # extra multiplier
    root_height_offset: float = 0.0
    root_motion: str = "FULL"       # FULL | IN_PLACE (XY removed) | FROM_FEET (XY integrated from the planted foot,
                                    # for hip-centred input such as single-camera MediaPipe world landmarks)
    ground_mode: str = "AUTO"       # AUTO (CLAMP for FULL root motion, else PLANT) | PLANT (lowest foot on the
                                    # floor every frame) | CLAMP (only no penetration) | NONE
    ground_smooth: float = 2.0
    # foot lock
    foot_lock: bool = True
    foot_lock_height: float = 0.035  # m above floor for a contact
    foot_lock_speed: float = 0.6    # m/s horizontal ankle speed for a contact
    foot_lock_blend: int = 4         # frames of blend in/out
    foot_lock_radius: float = 0.06   # m the free ankle may drift from a planted foot before it creeps
    foot_creep: float = 0.005        # m/frame max creep of a planted foot
    swing_clearance: float = 0.015   # m, minimum height of a foot that is not planted
    foot_lock_hip_drop: bool = True  # lower the hips when a planted foot is out of reach
    foot_pivot: float = 25.0         # deg the leg may turn before a planted foot re-anchors
    foot_heading_from_leg: float = 0.7  # planted feet: heading from the leg (1) or the foot points (0)
    foot_pole: float = 0.7           # planted feet: IK knee direction pulled towards the foot heading
    foot_flatten: float = 1.0        # planted feet: 1 = flat on the floor (yaw kept), 0 = mocap orientation
    heel_rise_max: float = 35.0      # deg: a planted foot out of reach lifts its heel (rolls about the ball, toes
                                     # flat) up to this before the hips are lowered (0 = off: hips drop at once)
    foot_scale_ref_leg: float = 0.9  # m: foot-lock height / speed / radius / creep are given for a rig with this
                                     # leg length (hip->ankle) and scaled by rig_leg / this (0 = absolute metres)
    # output
    set_fk: bool = True             # set the Rigify IK_FK switches (per control_mode); off = leave them alone
    control_mode: str = "FK"        # FK: FK controls only (IK_FK = 1) | LEGS_IK_ARMS_FK: FK keys + matching IK
                                    # keys on all limbs, legs switched to IK, arms to FK | BOTH: the same keys,
                                    # every limb switched to FK (flip any IK_FK slider without a pop)
    ik_zero_bone_stretch: bool = True  # IK modes: set the bone-level ik_stretch of Rigify's IK chain bones to 0
                                       # (Rigify ships 0.1, which stretches limbs up to ~4% even with IK_Stretch = 0)
    action_name: str = "mocap_retarget"
    offsets: dict = field(default_factory=dict)  # group -> (x,y,z) deg

    def copy(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Params(**d)

    @classmethod
    def from_dict(cls, d):
        """Params from a stored dict (e.g. an action's ``mocap_params``): keys that are not
        fields of this version (renamed / removed parameters) are dropped with a warning,
        missing ones keep their defaults."""
        known = {f.name for f in fields(cls)}
        dropped = sorted(k for k in d if k not in known)
        if dropped:
            warnings.warn("mocap_retarget: ignoring unknown stored parameter(s) %s" % ", ".join(dropped))
        return cls(**{k: v for k, v in d.items() if k in known})


# Joint limits (deg) that are not Params: used by the per-frame solve and re-applied
# after the temporal processing (_limits), so both use the same values.
HAND_TWIST_LIMIT = 5.0      # hand vs forearm twist (the solve gives the hand no twist of its own:
                            # pronation is the forearm's; this is the slack for smoothing / offsets)
FOOT_TWIST_LIMIT = 25.0     # foot vs shin twist
THIGH_SWING_LIMIT = 140.0   # thigh vs pelvis swing
FOOT_LOCK_TWIST_LIMIT = 60.0  # planted feet (looser: they must not follow the noisy shin, only stay
FOOT_LOCK_SWING_EXTRA = 20.0  # clear of Rigify's shin/foot twist-blend flip); swing = foot_limit + this


# =============================================================== small math
def nrm(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


def perp(v, axis):
    return v - axis * np.dot(v, axis)


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / max(e1 - e0, 1e-9), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_axis_angle(axis, ang):
    axis = nrm(axis)
    return np.concatenate([[math.cos(ang / 2)], axis * math.sin(ang / 2)])


def q_to_m(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m_to_q(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def q_slerp(a, b, t):
    d = np.dot(a, b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(min(d, 1.0))
    s = math.sin(th)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s


def m_slerp(A, B, t):
    return q_to_m(q_slerp(m_to_q(A), m_to_q(B), t))


def rot_arc(a, b):
    """Shortest-arc rotation matrix taking unit a to unit b."""
    a, b = nrm(a), nrm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        ax = nrm(np.cross(a, [1.0, 0, 0]))
        if np.linalg.norm(ax) < 1e-6:
            ax = nrm(np.cross(a, [0, 1.0, 0]))
        return q_to_m(q_axis_angle(ax, math.pi))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def rot_axis(axis, ang):
    return q_to_m(q_axis_angle(axis, ang))


def swing_twist(R, axis):
    """R = Swing @ Twist(axis).  Returns (Swing matrix, twist angle rad)."""
    q = m_to_q(R)
    p = np.dot(q[1:], axis)
    tw = np.array([q[0], *(axis * p)])
    n = np.linalg.norm(tw)
    if n < 1e-9:
        ang = math.pi
        tw = np.array([0.0, *axis])
    else:
        tw = tw / n
        ang = 2 * math.atan2(np.dot(tw[1:], axis), tw[0])
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
    sw = q_mul(q, q_conj(tw))
    return q_to_m(sw), ang


def slerp_dir(a, b, t):
    a, b = nrm(a), nrm(b)
    c = float(np.clip(np.dot(a, b), -1, 1))
    th = math.acos(c)
    if th < 1e-6:
        return b
    if th > math.pi - 1e-4:
        b = nrm(b + 1e-3 * nrm(np.cross(a, [0, 0, 1.0]) + [1e-3, 0, 0]))
        c = float(np.clip(np.dot(a, b), -1, 1))
        th = math.acos(c)
    return nrm((math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th))


def frame_x(x, yhint, fallback=None):
    """Frame with X = x, Y = yhint orthogonalised, Z = X x Y. Columns [X Y Z]."""
    x = nrm(x)
    y = perp(yhint, x)
    if np.linalg.norm(y) < 1e-4 * max(np.linalg.norm(yhint), 1e-9) and fallback is not None:
        y = perp(fallback, x)
    y = nrm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def frame_yz(y, zhint, fallback=None):
    """Frame with Y = y, Z = zhint orthogonalised, X = Y x Z."""
    y = nrm(y)
    z = perp(zhint, y)
    if np.linalg.norm(z) < 1e-3 and fallback is not None:
        z = perp(fallback, y)
    z = nrm(z)
    return np.stack([np.cross(y, z), y, z], axis=1)


def euler_xyz(deg):
    x, y, z = np.radians(deg)
    return rot_axis([0, 0, 1.0], z) @ rot_axis([0, 1.0, 0], y) @ rot_axis([1.0, 0, 0], x)


# =============================================================== input
EMBED_PREFIX = "text:"


def load_mocap(path):
    """Contract npz (joints (T,33,3), valid (T,33), fps, t).  ``text:NAME`` reads an npz
    embedded in the open .blend as a base64 text datablock (see embed_npz)."""
    if path.startswith(EMBED_PREFIX):
        import base64
        import io
        if bpy is None or path[len(EMBED_PREFIX):] not in bpy.data.texts:
            raise FileNotFoundError("embedded mocap %r not found in this .blend" % path)
        raw = base64.b64decode(bpy.data.texts[path[len(EMBED_PREFIX):]].as_string())
        d = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        d = np.load(path, allow_pickle=False)   # the contract has no object arrays (names: <U16)
    j = d["joints"].astype(np.float64)
    valid = d["valid"].astype(bool) if "valid" in d.files else np.isfinite(j).all(-1)
    valid &= np.isfinite(j).all(-1)
    fps = float(d["fps"]) if "fps" in d.files else 30.0
    t = d["t"] if "t" in d.files else np.arange(len(j)) / fps
    return {"joints": j, "valid": valid, "fps": fps, "t": np.asarray(t), "path": path}


def _gauss1d(x, sigma):
    if sigma <= 0.01:
        return x
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x)
    flat = pad.reshape(pad.shape[0], -1)
    of = out.reshape(out.shape[0], -1)
    for c in range(flat.shape[1]):
        of[:, c] = np.convolve(flat[:, c], k, mode="valid")
    return out


# (proximal, distal) bone pairs used for the length-consistency outlier test
_LEN_SEGS = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
             (27, 31), (28, 32), (27, 29), (28, 30), (11, 12), (23, 24)]


def _length_outliers(j, v, tol):
    """(T,33) mask of joints whose bone length is inconsistent with the take's median."""
    bad = np.zeros(v.shape, bool)
    for a, b in _LEN_SEGS:
        ok = v[:, a] & v[:, b]
        if ok.sum() < 10:
            continue
        L = np.linalg.norm(j[:, a] - j[:, b], axis=1)
        med = np.median(L[ok])
        r = np.abs(L / max(med, 1e-6) - 1.0)
        o = ok & (r > tol)
        bad[o, b] = True
        if (a, b) in ((11, 12), (23, 24)):
            bad[o, a] = True
    # the trunk (hip centre -> shoulder centre) inconsistent -> whole frame
    ok = v[:, [11, 12, 23, 24]].all(1)
    tr = np.linalg.norm((j[:, 11] + j[:, 12] - j[:, 23] - j[:, 24]) / 2, axis=1)
    if ok.sum() >= 10:
        med = np.median(tr[ok])
        bad[ok & (np.abs(tr / med - 1) > tol)] = True
    return bad


# joints whose loss makes a frame unsolvable (trunk + limbs down to wrists / ankles)
CORE = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# max deviation (m, for a ~1.7 m person) from the hip-relative temporal median
_TEMPORAL_TOL = {11: .15, 12: .15, 23: .12, 24: .12, 13: .3, 14: .3, 25: .3, 26: .3,
                 15: .45, 16: .45, 27: .45, 28: .45}


def _temporal_outliers(j, v, scale=1.0, win=9):
    """Joints that jump away from their hip-relative running median (data glitches)."""
    T = len(j)
    bad = np.zeros(v.shape, bool)
    hc = (j[:, 23] + j[:, 24]) / 2
    okh = v[:, 23] & v[:, 24]
    r = win // 2
    for k, tol in _TEMPORAL_TOL.items():
        ok = v[:, k] & okh
        rel = np.where(ok[:, None], j[:, k] - hc, np.nan)
        pad = np.pad(rel, [(r, r), (0, 0)], mode="edge")
        stack = np.stack([pad[i:i + T] for i in range(win)], 0)
        cnt = np.isfinite(stack[..., 0]).sum(0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        dev = np.linalg.norm(rel - med, axis=1)
        bad[:, k] = ok & (cnt >= 3) & (dev > tol * scale)
    return bad


_LR = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(a, a + 1) for a in range(11, 33, 2)]
_SWAP = np.arange(33)
for _a, _b in _LR:
    _SWAP[_a], _SWAP[_b] = _b, _a


def _fix_lr_swaps(j, v, margin=0.03):
    """Multi-view triangulation sometimes swaps the left / right labels for a few frames,
    which would spin the rig by 180 deg.  The nose is unpaired, so it is a label-independent
    witness: the facing direction from the shoulders (left x up) and from the hips must
    point to the nose side.  When both point away (by > margin m), relabel the frame.
    Works in place; returns the number of fixed frames."""
    up = np.array([0, 0, 1.0])
    fixed, seeds = 0, []

    def facing(t, a, b):
        f = np.cross(j[t, a] - j[t, b], up)[:2]
        n = (j[t, 0] - (j[t, a] + j[t, b]) / 2)[:2]
        return float(np.dot(nrm(f), n))

    def ambiguous(t):   # nose test inconclusive (missing, or head turned sideways)
        if not v[t, [0, 11, 12]].all():
            return True
        return max(facing(t, 11, 12), facing(t, 23, 24) if v[t, [23, 24]].all() else -1) <= margin

    for t in range(len(j)):
        if not v[t, [0, 11, 12]].all():
            continue
        fs = facing(t, 11, 12)
        fh = facing(t, 23, 24) if v[t, [23, 24]].all() else fs
        if fs < -margin and fh < -margin:
            j[t] = j[t, _SWAP]
            v[t] = v[t, _SWAP]
            fixed += 1
            seeds.append(t)
    # neighbours of a confirmed swap whose nose test was ambiguous (head turned): relabel
    # them too when the swapped labelling is much closer to the corrected neighbour
    body = list(range(11, 33))
    done = set(seeds)
    for t0 in seeds:
        for step in (1, -1):
            t = t0 + step
            while 0 <= t < len(j) and t not in done and abs(t - t0) <= 5 and ambiguous(t):
                ref = t - step
                m, ms = v[t] & v[ref], v[t, _SWAP] & v[ref]
                if m[body].sum() < 6 or ms[body].sum() < 6:
                    break
                c0 = np.mean(np.linalg.norm(j[t, body][m[body]] - j[ref, body][m[body]], axis=1))
                js = j[t, _SWAP]
                c1 = np.mean(np.linalg.norm(js[body][ms[body]] - j[ref, body][ms[body]], axis=1))
                if not c1 < 0.6 * c0:
                    break
                j[t], v[t] = js, v[t, _SWAP]
                done.add(t)
                fixed += 1
                t += step
    return fixed


def _gap_margins(v, r, min_gap=5):
    """Frames within r of a gap (>= min_gap invalid frames) of the same joint: detections
    degrade right before tracking is lost / right after it is regained."""
    T = len(v)
    bad = np.zeros(v.shape, bool)
    for k in range(v.shape[1]):
        col = v[:, k]
        t = 0
        while t < T:
            if not col[t]:
                e = t
                while e < T and not col[e]:
                    e += 1
                if e - t >= min_gap:
                    bad[max(0, t - r):t, k] = True
                    bad[e:min(T, e + r), k] = True
                t = e
            else:
                t += 1
    return bad


def _short_islands(v, min_len):
    """Valid runs shorter than min_len frames between invalid frames (flickering tracking)."""
    bad = np.zeros(v.shape, bool)
    T = len(v)
    for k in range(v.shape[1]):
        t = 0
        while t < T:
            if v[t, k]:
                e = t
                while e < T and v[e, k]:
                    e += 1
                if e - t < min_len and t > 0 and e < T:
                    bad[t:e, k] = True
                t = e
            else:
                t += 1
    return bad


def _infer_knees(j, v):
    """Knee missing but hip + ankle observed (e.g. occluded in a crouch): place the knee by
    2-bone IK with the take's median thigh / shin lengths, bending forward (pelvis facing).
    Marks the inferred knees valid in v (in place)."""
    up = np.array([0, 0, 1.0])
    for hp, kn, an in ((23, 25, 27), (24, 26, 28)):
        okl = v[:, hp] & v[:, kn] & v[:, an]
        if okl.sum() < 10:
            continue
        l1 = float(np.median(np.linalg.norm(j[okl, kn] - j[okl, hp], axis=1)))
        l2 = float(np.median(np.linalg.norm(j[okl, an] - j[okl, kn], axis=1)))
        need = v[:, hp] & v[:, an] & ~v[:, kn] & v[:, 23] & v[:, 24]
        for t in np.where(need)[0]:
            d = j[t, an] - j[t, hp]
            dl = min(max(np.linalg.norm(d), abs(l1 - l2) + 1e-3), (l1 + l2) * 0.999)
            u = nrm(d)
            fwd = np.cross(nrm(j[t, 23] - j[t, 24]), up)   # left x up = forward
            pole = perp(fwd, u)
            if np.linalg.norm(pole) < 1e-6:
                continue
            pole = nrm(pole)
            ca = (l1 * l1 + dl * dl - l2 * l2) / (2 * l1 * dl)
            j[t, kn] = j[t, hp] + l1 * (ca * u + math.sqrt(max(0.0, 1 - ca * ca)) * pole)
            v[t, kn] = True


def preprocess(mocap, prm, return_mask=False):
    """Outlier rejection, gap fill (linear), median despike, gaussian smoothing, yaw.
    -> (T,33,3) [, (T,33) mask of joints that were valid after rejection]"""
    j = mocap["joints"].copy()
    v = mocap["valid"].copy()
    T = len(j)
    if prm.fix_lr_swaps:
        _fix_lr_swaps(j, v)
    if prm.reject_outliers > 0:
        v &= ~_length_outliers(j, v, prm.reject_outliers)
        ok = v[:, [11, 12, 23, 24]].all(1)
        trunk = np.linalg.norm((j[ok, 11] + j[ok, 12] - j[ok, 23] - j[ok, 24]) / 2, axis=1)
        sc = float(np.median(trunk)) / 0.5 if ok.sum() > 10 else 1.0
        v &= ~_temporal_outliers(j, v, sc)
    if prm.gap_erode > 0:
        v &= ~_gap_margins(v, prm.gap_erode)
    if prm.infer_knees:
        _infer_knees(j, v)
    idx = np.arange(T)
    never = [k for k in range(j.shape[1]) if not v[:, k].any()]
    if never:
        names = {i: n for n, i in MP.items()}
        raise ValueError("mocap joint(s) never observed (no valid frame after outlier rejection): %s; "
                         "the segments they define cannot be solved. Check the take (valid flags) or "
                         "relax reject_outliers / gap_erode" % ", ".join(names.get(k, str(k)) for k in never))
    for k in range(j.shape[1]):
        ok = v[:, k]
        for c in range(3):
            j[:, k, c] = np.interp(idx, idx[ok], j[ok, k, c])
    w = int(prm.despike)
    if w >= 3:
        w |= 1
        r = w // 2
        pad = np.pad(j, [(r, r), (0, 0), (0, 0)], mode="edge")
        win = np.stack([pad[i:i + T] for i in range(w)], 0)
        j = np.median(win, axis=0)
    j = _gauss1d(j, prm.smooth_sigma)
    if prm.mocap_yaw:
        j = j @ rot_axis([0, 0, 1.0], math.radians(prm.mocap_yaw)).T
    if return_mask:
        # flickering 1-2 frame detections still inform the joint positions (e.g. the hip
        # height in a crouch), but do not count as "observed" for rotation gap filling
        if prm.gap_erode > 0:
            v = v & ~_short_islands(v, 3)
        return j, v
    return j


# =============================================================== rig model
class RigModel:
    """Pure-numpy description of the target rig's rest pose (armature space)."""

    def __init__(self):
        self.R = {}      # control -> 3x3 rest rotation
        self.H = {}      # control -> rest head
        self.lm = {}     # landmark name -> rest position
        self.points = {}  # extra named rest points attached to controls
        self.leg_len = 1.0
        self.hip_center = np.zeros(3)
        self.world = np.eye(4)
        self.ik_rest = {}  # Rigify IK control -> 4x4 rest matrix (armature space); empty = no IK controls
        self.ik_locframe = {}  # IK control -> 3x3 axes its pose location is expressed in (bone or parent rest)

    @property
    def has_ik(self):
        return bool(self.ik_rest)

    # -- landmark access with MediaPipe names
    def rest_landmarks(self):
        P = np.zeros((33, 3))
        for n, i in MP.items():
            if n in self.lm:
                P[i] = self.lm[n]
        return P


def _parametric_head(lm, h0, h1):
    L = np.linalg.norm(h1 - h0)
    up = nrm(h1 - h0)
    fwd = nrm(perp(np.array([0, -1.0, 0]), up))
    side = nrm(np.cross(fwd, up))  # +X for a -Y facing rig
    lm.setdefault("left_ear", h0 + up * 0.40 * L + side * 0.42 * L)
    lm.setdefault("right_ear", h0 + up * 0.40 * L - side * 0.42 * L)
    lm.setdefault("left_eye", h0 + up * 0.56 * L + fwd * 0.45 * L + side * 0.18 * L)
    lm.setdefault("right_eye", h0 + up * 0.56 * L + fwd * 0.45 * L - side * 0.18 * L)
    lm.setdefault("nose", h0 + up * 0.36 * L + fwd * 0.62 * L)


def build_model(rig):
    """Extract a RigModel from a generated Rigify rig object."""
    m = RigModel()
    bones = rig.data.bones
    for c, _, _ in CONTROLS:
        if c not in bones:
            raise KeyError("control bone %r missing - is this a Rigify human rig?" % c)
        b = bones[c]
        m.R[c] = np.array(b.matrix_local.to_3x3().normalized())
        m.H[c] = np.array(b.head_local)

    def H(n):
        return np.array(bones[n].head_local)

    def Tl(n):
        return np.array(bones[n].tail_local)

    lm = {}
    stored = json.loads(rig.get("mocap_landmarks", "{}")) if "mocap_landmarks" in rig.keys() else {}
    for s, mp in SIDES:
        lm[mp + "_shoulder"] = H("ORG-upper_arm." + s)
        lm[mp + "_elbow"] = H("ORG-forearm." + s)
        lm[mp + "_wrist"] = H("ORG-hand." + s)
        lm[mp + "_hip"] = H("ORG-thigh." + s)
        lm[mp + "_knee"] = H("ORG-shin." + s)
        lm[mp + "_ankle"] = H("ORG-foot." + s)
        lm[mp + "_foot_index"] = Tl("ORG-toe." + s)
        for k, bn in (("index", "ORG-f_index.01."), ("pinky", "ORG-f_pinky.01."), ("thumb", "ORG-thumb.02.")):
            if (bn + s) in bones:
                lm[mp + "_" + k] = Tl(bn + s)
        if ("ORG-heel.02." + s) in bones:
            lm[mp + "_heel"] = (H("ORG-heel.02." + s) + Tl("ORG-heel.02." + s)) / 2
    for k, v in stored.items():
        lm[k] = np.array(v)
    _parametric_head(lm, H("ORG-spine.006"), Tl("ORG-spine.006"))
    for s, mp in SIDES:  # fallbacks for minimal rigs
        hand_dir = nrm(Tl("ORG-hand." + s) - H("ORG-hand." + s))
        lm.setdefault(mp + "_index", Tl("ORG-hand." + s) + np.array([0, -0.03, 0]))
        lm.setdefault(mp + "_pinky", Tl("ORG-hand." + s) + np.array([0, 0.03, 0]))
        lm.setdefault(mp + "_thumb", H("ORG-hand." + s) + hand_dir * 0.03 + np.array([0, -0.04, 0]))
        lm.setdefault(mp + "_heel", H("ORG-foot." + s) * np.array([1, 1, 0]) + np.array([0, 0.05, 0]))
    m.lm = lm
    m.leg_len = 0.5 * sum(np.linalg.norm(lm[p + "_knee"] - lm[p + "_hip"]) +
                          np.linalg.norm(lm[p + "_ankle"] - lm[p + "_knee"]) for _, p in SIDES)
    m.hip_center = (lm["left_hip"] + lm["right_hip"]) / 2
    # foot contact points (attached to foot_fk; toe_fk carries the same delta)
    for s, mp in SIDES:
        m.points["heel_" + s] = ("foot_fk." + s, lm[mp + "_heel"])
        m.points["ball_" + s] = ("foot_fk." + s, H("ORG-toe." + s) - np.array([0, 0, 0.0]))
        m.points["tip_" + s] = ("toe_fk." + s, Tl("ORG-toe." + s))   # == foot_fk while the toe is not bent
        m.points["ankle_" + s] = ("foot_fk." + s, H("ORG-foot." + s))
        m.points["hip_" + s] = ("spine_fk", lm[mp + "_hip"])
        m.points["knee_" + s] = ("thigh_fk." + s, lm[mp + "_knee"])
    m.points["hipc"] = ("spine_fk", m.hip_center)
    m.points["shc"] = ("spine_fk.003", (lm["left_shoulder"] + lm["right_shoulder"]) / 2)
    # the rest floor height of each foot (lowest contact point at rest)
    m.floor_rest = min(m.points[k][1][2] for k in m.points if k.split("_")[0] in ("heel", "ball", "tip"))
    m.world = np.array(rig.matrix_world)
    names = []
    for s, _ in SIDES:
        for L in IK_LIMBS.values():
            names += [n + "." + s for n in (L["end"], L["pole"], L["toe"], L["prop"]) + L["rest"] + L["chain"] if n]
    if all(n in bones for n in names):
        for n in names:
            M = np.array(bones[n].matrix_local)
            M[:3, :3] = np.array(bones[n].matrix_local.to_3x3().normalized())
            m.ik_rest[n] = M
            b = bones[n]
            # Bone.use_local_location off (Rigify's IK controls): location is in the parent's rest axes
            m.ik_locframe[n] = M[:3, :3] if b.use_local_location else (
                np.array(b.parent.matrix_local.to_3x3().normalized()) if b.parent else np.eye(3))
    return m


# =============================================================== frame solver
def _trunk_frames(P, prm):
    hl, hr, sl, sr = P[MP["left_hip"]], P[MP["right_hip"]], P[MP["left_shoulder"]], P[MP["right_shoulder"]]
    mh, ms = (hl + hr) / 2, (sl + sr) / 2
    trunk = ms - mh
    up = np.array([0, 0, 1.0])
    pel_up = slerp_dir(up, trunk, prm.pelvis_follow) if np.linalg.norm(trunk) > 1e-6 else up
    Fp = frame_x(hl - hr, pel_up, fallback=up)
    Fc = frame_x(sl - sr, trunk, fallback=up)
    return Fp, Fc


def _head_frame(P, face_weight=0.5):
    """X = ear line.  Pitch from a blend of the face (nose + eyes ahead of the ears; noisy
    in depth for monocular capture) and the neck (shoulder centre -> ear centre)."""
    el, er = P[MP["left_ear"]], P[MP["right_ear"]]
    me = (el + er) / 2
    x = nrm(el - er)
    fwd = (P[MP["nose"]] - me) + ((P[MP["left_eye"]] + P[MP["right_eye"]]) / 2 - me)
    z = nrm(perp(fwd, x))
    Ff = np.stack([x, np.cross(z, x), z], axis=1)
    y = nrm(perp(me - (P[MP["left_shoulder"]] + P[MP["right_shoulder"]]) / 2, x))
    Fn = np.stack([x, y, np.cross(x, y)], axis=1)
    if face_weight >= 1.0:
        return Ff
    if face_weight <= 0.0:
        return Fn
    return m_slerp(Fn, Ff, face_weight)


def _hand_frame(P, mp):
    w, i, p = P[MP[mp + "_wrist"]], P[MP[mp + "_index"]], P[MP[mp + "_pinky"]]
    return frame_yz((i + p) / 2 - w, i - p)


def _foot_frame(P, mp):
    # Y = ankle -> toe.  The roll reference is the shin axis (knee - ankle): MediaPipe
    # heel/toe points are too noisy to give foot inversion, and the heel-ankle-toe
    # triangle degenerates; the shin gives a stable "up" for the foot.
    a, k, t = P[MP[mp + "_ankle"]], P[MP[mp + "_knee"]], P[MP[mp + "_foot_index"]]
    return frame_yz(t - a, k - a, fallback=np.array([0, 0, 1.0]))


class RestFrames:
    """Rest frames of the target, built by the very same functions as the pose frames."""

    def __init__(self, model, prm):
        P = model.rest_landmarks()
        self.P = P
        self.F = {}
        self.F["pelvis"], self.F["chest"] = _trunk_frames(P, prm)
        self.F["head"] = _head_frame(P, prm.head_face_weight)
        self.hinge_bend0 = {}
        for s, mp in SIDES:
            sh, el, wr = P[MP[mp + "_shoulder"]], P[MP[mp + "_elbow"]], P[MP[mp + "_wrist"]]
            hp, kn, an = P[MP[mp + "_hip"]], P[MP[mp + "_knee"]], P[MP[mp + "_ankle"]]
            heel, toe = P[MP[mp + "_heel"]], P[MP[mp + "_foot_index"]]
            u, f = nrm(el - sh), nrm(wr - el)
            # upper arm: front = side of the (small) rest elbow bend
            if np.degrees(math.acos(np.clip(np.dot(u, f), -1, 1))) < 2.0:
                raise ValueError("rest elbow is straight: cannot infer hinge axis")
            self.F["uarm_" + s] = frame_yz(u, f)
            self.F["farm_" + s] = frame_yz(f, P[MP[mp + "_index"]] - P[MP[mp + "_pinky"]])
            self.F["hand_" + s] = _hand_frame(P, mp)
            t, sh_ = nrm(kn - hp), nrm(an - kn)
            ffwd = toe - heel
            self.F["thigh_" + s] = frame_yz(t, -sh_ if np.linalg.norm(perp(sh_, t)) > 0.02 else ffwd,
                                            fallback=ffwd)
            # make the thigh "front" agree with the foot when the rest knee is ~straight
            if np.dot(self.F["thigh_" + s][:, 2], ffwd) < 0:
                self.F["thigh_" + s] = frame_yz(t, ffwd)
            self.F["shin_" + s] = frame_yz(sh_, ffwd)
            self.F["foot_" + s] = _foot_frame(P, mp)
            self.hinge_bend0["elbow_" + s] = math.acos(np.clip(np.dot(u, f), -1, 1))


def _limb_front(u, child, fallback_front, bend_sign, prm):
    """Twist reference ("front", unit, perpendicular to u) for upper arm / thigh.

    Bend-plane front: bend_sign * (child direction perpendicular to u)
    (+ for the elbow: the forearm bends forward; - for the knee).  Blended
    with the fallback (minimal twist / foot direction) when the limb is
    nearly straight and the bend plane is ill-defined.
    """
    cp = perp(child, u)
    bend = math.degrees(math.atan2(np.linalg.norm(cp), np.dot(child, u)))
    fb = nrm(perp(fallback_front, u))
    if np.linalg.norm(cp) < 1e-6:
        return fb, bend
    fbend = nrm(cp) * bend_sign
    w = smoothstep(prm.bend_lo, prm.bend_hi, bend)
    ang = math.atan2(np.dot(np.cross(fb, fbend), u), np.dot(fb, fbend))
    return rot_axis(u, ang * w) @ fb, bend


def _solve_limb(Dpar, u, child, Fup, Flow, fallback, bend_sign, up_twist_lim, up_swing_lim,
                hinge, low_hint, low_twist_mix, low_twist_lim, prm, st, key, off_up=None, off_low=None,
                total_twist=None):
    """Two-segment limb (upper arm + forearm or thigh + shin).

    The bend plane only fixes the upper segment's twist up to a sign (a noisy,
    slightly hyper-extended limb looks exactly like a flexed limb with a 180 deg
    twisted upper segment).  Both signs are solved with all limits applied and
    the one whose constrained lower segment best matches the data wins
    (small bias to the previous frame's twist and to normal flexion).
    """
    best = None
    prev_tw = st.get("tw_" + key + "_up")
    for sg in (1.0, -1.0):
        front, _ = _limb_front(u, child, fallback, bend_sign * sg, prm)
        Du = _local_fix(Dpar, frame_yz(u, front) @ Fup.T, Fup[:, 1], 1.0, up_twist_lim, up_swing_lim,
                        offset=off_up)
        _, tw = swing_twist(Dpar.T @ Du, Fup[:, 1])
        Dl = _local_fix(Du, frame_yz(child, low_hint, fallback=Du @ Flow[:, 2]) @ Flow.T, Flow[:, 1],
                        hinge=hinge)
        err = math.degrees(math.acos(np.clip(np.dot(Dl @ Flow[:, 1], child), -1, 1)))
        score = err + (0.0 if sg > 0 else 2.0)
        if prev_tw is not None:
            d = abs((tw - prev_tw + math.pi) % (2 * math.pi) - math.pi)
            score += prm.continuity * math.degrees(d)
        if best is None or score < best[0]:
            best = (score, front)
    front = best[1]
    Du = _local_fix(Dpar, frame_yz(u, front) @ Fup.T, Fup[:, 1], 1.0, up_twist_lim, up_swing_lim,
                    offset=off_up, st=st, key=key + "_up")
    rng = None
    if total_twist is not None:
        # Rigify's mid twist segment blends the (anti-twist) limb root with the lower
        # segment: upper + lower twist must stay well below 180 deg or it flips
        _, tw_u = swing_twist(Dpar.T @ Du, Fup[:, 1])
        T = math.radians(total_twist)
        rng = (-T - tw_u, T - tw_u)
    Dl = _local_fix(Du, frame_yz(child, low_hint, fallback=Du @ Flow[:, 2]) @ Flow.T, Flow[:, 1],
                    low_twist_mix, low_twist_lim, None, hinge, offset=off_low, st=st, key=key + "_low",
                    twist_range=rng)
    return Du, Dl


def _local_fix(Dp, Dc, a, twist_mix=1.0, twist_lim=180.0, swing_lim=None,
               hinge=None, offset=None, st=None, key=None, twist_range=None):
    """Constrain the parent-relative rotation of a segment.

    Dp, Dc: world deltas of parent / child.  a: child rest axis (rest world).
    hinge = (h, ap, sign, lo_deg, hi_deg, strict): hinge axis (rest world), parent
    rest axis, flexion sign, flexion limits.  Returns the new child delta.
    """
    L = Dp.T @ Dc
    S, tw = swing_twist(L, a)
    lim = math.radians(twist_lim)
    if st is not None and key is not None:
        # temporal continuity: unwrap the measured twist against the previous frame
        # (so a reading near +-180 cannot hop between the two limits) ...
        prev_raw = st.get("raw_" + key)
        if prev_raw is not None:
            tw += 2 * math.pi * round((prev_raw - tw) / (2 * math.pi))
            if abs(tw) > math.pi and abs((tw + math.pi) % (2 * math.pi) - math.pi) * twist_mix < 0.8 * lim:
                tw = (tw + math.pi) % (2 * math.pi) - math.pi   # wound up but legal again: unwind
        st["raw_" + key] = tw
    lo, hi = (-lim, lim) if twist_range is None else (max(-lim, twist_range[0]), min(lim, twist_range[1]))
    tw = float(np.clip(tw * twist_mix, lo, hi))
    if st is not None and key is not None:
        # ... and rate-limit the applied twist (a reference that spins becomes a roll, not a pop)
        prev = st.get("tw_" + key)
        rate = math.radians(st.get("_twist_rate", 180.0))
        if prev is not None:
            tw = prev + float(np.clip(tw - prev, -rate, rate))
        st["tw_" + key] = tw
    b = S @ a
    if hinge is not None:
        h, ap, sign, lo, hi, strict = hinge
        bp = perp(b, h)
        bp = nrm(bp) if np.linalg.norm(bp) > 1e-6 else ap
        b = slerp_dir(b, bp, strict) if strict < 1.0 else bp
        flex = sign * math.atan2(np.dot(np.cross(ap, b), h), np.dot(ap, b))
        fl = min(max(flex, -math.radians(lo)), math.radians(hi))
        if fl != flex:
            b = rot_axis(h, sign * (fl - flex)) @ b
    if swing_lim is not None:
        c = math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))
        if c > swing_lim:
            b = slerp_dir(a, b, swing_lim / c)
    L2 = rot_arc(a, b) @ rot_axis(a, tw)
    if offset is not None:
        L2 = L2 @ offset
    return Dp @ L2


def _fix_trunk(D, model, P, iters=3, lateral=0.5):
    """Bend the spine so the rig's hip-centre -> shoulder-centre line matches the mocap
    trunk in the sagittal plane.  The pelvis follows the trunk only partly
    (pelvis_follow), so the spine must take the rest of a forward / backward lean.
    Only rotation about the chest's left-right axis is used, which keeps the shoulder
    line exact; the correction is spread over the spine (chest takes all of it)."""
    tgt = (P[11] + P[12]) / 2 - (P[23] + P[24]) / 2
    w = {"mid": 0.25, "upmid": 0.6, "chest": 1.0}
    x = nrm(P[11] - P[12])
    for _ in range(iters):
        pos = fk_heads(D, model, np.zeros(3))
        v = fk_point(D, model, pos, "shc") - fk_point(D, model, pos, "hipc")
        vp, tp = perp(v, x), perp(tgt, x)
        if np.linalg.norm(vp) < 1e-6 or np.linalg.norm(tp) < 1e-6:
            break
        ang = math.atan2(np.dot(np.cross(vp, tp), x), np.dot(vp, tp))
        if abs(ang) < 1e-4:
            break
        for seg, k in w.items():
            D[seg] = rot_axis(x, ang * k) @ D[seg]
        D["lowmid"] = m_slerp(D["pelvis"], D["mid"], 0.5)
    if lateral > 0:
        # remaining (lateral) mismatch: the mocap trunk is not perpendicular to its own
        # shoulder line (noise / shrug); split it between trunk and shoulder line
        pos = fk_heads(D, model, np.zeros(3))
        v = fk_point(D, model, pos, "shc") - fk_point(D, model, pos, "hipc")
        q = m_to_q(rot_arc(v, tgt))
        for seg, k in w.items():
            D[seg] = q_to_m(q_slerp(np.array([1.0, 0, 0, 0]), q, k * lateral)) @ D[seg]
        D["lowmid"] = m_slerp(D["pelvis"], D["mid"], 0.5)


def solve_pose(P, model, rest, prm, state=None):
    """Solve one frame. P: (33,3) joints (armature space).  -> {segment: 3x3 world delta}."""
    F = rest.F
    st = state if state is not None else {}
    st["_twist_rate"] = prm.twist_rate_limit
    D = {}
    off = {}
    for g, seg in OFFSET_GROUPS.items():
        o = prm.offsets.get(g)
        if o is not None and any(abs(x) > 1e-6 for x in o):
            R = model.R[SEG_CTRL[seg]]
            off[seg] = R @ euler_xyz(o) @ R.T     # bone-local euler -> rest world
    Fp, Fc = _trunk_frames(P, prm)
    D["pelvis"] = Fp @ F["pelvis"].T
    Dc = Fc @ F["chest"].T
    if "chest" in off:
        Dc = Dc @ off["chest"]
    D["chest"] = Dc
    D["mid"] = m_slerp(D["pelvis"], D["chest"], 0.5)
    D["lowmid"] = m_slerp(D["pelvis"], D["chest"], 0.25)
    D["upmid"] = m_slerp(D["pelvis"], D["chest"], 0.75)
    _fix_trunk(D, model, P, lateral=prm.trunk_lateral)
    # head / neck
    Dh = _head_frame(P, prm.head_face_weight) @ F["head"].T
    head_axis = rest.F["head"][:, 1]
    Dh = _local_fix(D["chest"], Dh, head_axis, 1.0, prm.head_limit, prm.head_limit, offset=off.get("head"),
                    st=st, key="head")
    D["neck"] = D["chest"] @ m_slerp(np.eye(3), D["chest"].T @ Dh, prm.neck_share)
    D["head"] = Dh
    for s, mp in SIDES:
        j = lambda n: P[MP[mp + "_" + n]]
        # ---------------- arm
        sh, el, wr = j("shoulder"), j("elbow"), j("wrist")
        u, f = nrm(el - sh), nrm(wr - el)
        # clavicle: share of the elevation above horizontal (in the chest frame)
        Dcl = D["chest"]
        if prm.clavicle_share > 0:
            up_c = D["chest"] @ rest.F["chest"][:, 1]
            elev = math.asin(np.clip(np.dot(u, up_c), -1, 1))
            if elev > 0:
                # chest frame Z = forward; +rotation about forward raises the left side
                fwd_c = D["chest"] @ rest.F["chest"][:, 2]
                sign = 1.0 if s == "L" else -1.0
                Dcl = rot_axis(fwd_c, sign * prm.clavicle_share * elev) @ D["chest"]
        D["clav_" + s] = Dcl
        Fu0, Ff0 = F["uarm_" + s], F["farm_" + s]
        a_u = Fu0[:, 1]
        # minimal-twist fallback: rest front carried by the clavicle, swung onto u
        fmin = rot_arc(Dcl @ a_u, u) @ (Dcl @ Fu0[:, 2])
        hinge = (Fu0[:, 0], a_u, +1.0, prm.hyperextend, prm.elbow_max, prm.hinge_strict)
        # forearm twist reference = radial direction of the hand (index - pinky)
        Du, Df = _solve_limb(Dcl, u, f, Fu0, Ff0, fmin, +1.0, prm.upperarm_twist_limit, None, hinge,
                             j("index") - j("pinky"), prm.twist_mix, prm.forearm_twist_limit, prm, st,
                             "arm_" + s, off.get("uarm_" + s), off.get("farm_" + s), prm.arm_total_twist_limit)
        D["uarm_" + s] = Du
        D["farm_" + s] = Df
        Dhd = _hand_frame(P, mp) @ F["hand_" + s].T
        Dhd = _local_fix(Df, Dhd, F["hand_" + s][:, 1], 0.0, HAND_TWIST_LIMIT, prm.hand_limit, st=st,
                         key="hand_" + s)
        Dhd = Df @ m_slerp(np.eye(3), Df.T @ Dhd, prm.hand_mix)
        if "hand_" + s in off:
            Dhd = Dhd @ off["hand_" + s]
        D["hand_" + s] = Dhd
        # ---------------- leg
        hp, kn, an = j("hip"), j("knee"), j("ankle")
        heel, toe = j("heel"), j("foot_index")
        t, sv = nrm(kn - hp), nrm(an - kn)
        Ft0 = F["thigh_" + s]
        a_t = Ft0[:, 1]
        ffwd = toe - heel
        Fs0 = F["shin_" + s]
        fmin = rot_arc(D["pelvis"] @ a_t, t) @ (D["pelvis"] @ Ft0[:, 2])
        # straight leg: knee direction from the foot (heel -> toe), else minimal twist
        fb = ffwd if np.linalg.norm(perp(nrm(ffwd), t)) > 0.3 else fmin
        hinge = (Ft0[:, 0], a_t, -1.0, prm.hyperextend, prm.knee_max, prm.hinge_strict)
        Dt, Ds = _solve_limb(D["pelvis"], t, sv, Ft0, Fs0, fb, -1.0, prm.thigh_twist_limit, THIGH_SWING_LIMIT, hinge,
                             ffwd, prm.leg_twist_mix, prm.shin_twist_limit, prm, st, "leg_" + s,
                             off.get("thigh_" + s), off.get("shin_" + s))
        D["thigh_" + s] = Dt
        D["shin_" + s] = Ds
        Dft = _foot_frame(P, mp) @ F["foot_" + s].T
        Dft = _local_fix(Ds, Dft, F["foot_" + s][:, 1], 1.0, FOOT_TWIST_LIMIT, prm.foot_limit,
                         offset=off.get("foot_" + s),
                         st=st, key="foot_" + s)
        D["foot_" + s] = Dft
        D["toe_" + s] = Dft
    return D


# =============================================================== kinematics on deltas
def fk_heads(D, model, t):
    """World (armature) head positions of all controls; torso head moved by t."""
    pos = {}
    for c, p, seg in CONTROLS:
        if p is None:
            pos[c] = model.H[c] + t
        elif p in pos and CTRL_SEG[p] in D:   # partial D (during the solve) is allowed
            pos[c] = pos[p] + D[CTRL_SEG[p]] @ (model.H[c] - model.H[p])
    return pos


def fk_point(D, model, pos, name):
    c, p0 = model.points[name]
    return pos[c] + D[CTRL_SEG[c]] @ (p0 - model.H[c])


def two_bone_ik(D, model, rest, pos, s, target, foot_pole=0.0):
    """Re-aim thigh/shin so the ankle reaches target (bone lengths fixed)."""
    hip = fk_point(D, model, pos, "hip_" + s)
    knee = fk_point(D, model, pos, "knee_" + s)
    l1 = np.linalg.norm(model.points["knee_" + s][1] - model.points["hip_" + s][1])
    l2 = np.linalg.norm(model.points["ankle_" + s][1] - model.points["knee_" + s][1])
    d = target - hip
    dl = np.linalg.norm(d)
    dl_c = min(max(dl, abs(l1 - l2) + 1e-4), (l1 + l2) * 0.9995)
    u = d / max(dl, 1e-9)
    Ft0 = rest.F["thigh_" + s]
    front = perp(D["thigh_" + s] @ Ft0[:, 2], u)
    front = nrm(front) if np.linalg.norm(front) > 1e-6 else nrm(perp(knee - hip, u))
    if foot_pole > 0:
        # the knee points roughly along the foot: heading of the foot projected on the IK plane normal.
        # The toe's orientation is used: it equals the foot's except during a heel lift, where the
        # pitched-down foot axis comes close to the leg axis and its projection is unstable
        fh = perp(D["toe_" + s] @ rest.F["foot_" + s][:, 1], u)
        if np.linalg.norm(fh) > 1e-3 and np.dot(nrm(fh), front) > -0.5:
            front = slerp_dir(front, nrm(fh), min(foot_pole, 1.0))
    cosa = (l1 * l1 + dl_c * dl_c - l2 * l2) / (2 * l1 * dl_c)
    sina = math.sqrt(max(0.0, 1 - cosa * cosa))
    knee2 = hip + l1 * (cosa * u + sina * front)
    ank2 = hip + u * dl_c
    tdir = nrm(knee2 - hip)
    Dt = frame_yz(tdir, front) @ Ft0.T
    Dshin_old_local = D["thigh_" + s].T @ D["shin_" + s]
    # keep the shin twist, re-aim within the (exact) hinge plane
    Fs0 = rest.F["shin_" + s]
    a_s = Fs0[:, 1]
    _, tw = swing_twist(Dshin_old_local, a_s)
    b_loc = Dt.T @ nrm(ank2 - knee2)
    Ds = Dt @ (rot_arc(a_s, b_loc) @ rot_axis(a_s, tw))
    # keep foot world orientation
    D["thigh_" + s] = Dt
    D["shin_" + s] = Ds
    return np.linalg.norm(ank2 - target)


# =============================================================== sequence solve
@dataclass
class Solution:
    frames: np.ndarray            # mocap frame indices
    scene_frames: np.ndarray      # scene frame numbers
    D: list                       # per frame dict segment -> 3x3 world delta
    t: np.ndarray                 # (N,3) torso translation (armature space)
    contacts: dict                # side -> (N,) bool
    joints: np.ndarray            # (N,33,3) the processed target joints (armature space)
    scale: float
    info: dict


def _contacts(h, sp, prm, fps):
    return _contacts_clean((h < prm.foot_lock_height) & (sp < prm.foot_lock_speed / fps))


def _contacts_clean(c):
    c = np.asarray(c, bool)
    # close small gaps, remove blips
    n = len(c)
    c2 = c.copy()
    for i in range(n):
        if not c[i]:
            l = i - 1
            r = i + 1
            while r < n and not c[r] and r - i < 3:
                r += 1
            if l >= 0 and r < n and c[l] and c[r] and r - l <= 3:
                c2[i] = True
    out = c2.copy()
    i = 0
    while i < n:
        if c2[i]:
            k = i
            while k < n and c2[k]:
                k += 1
            if k - i < 3:
                out[i:k] = False
            i = k
        else:
            i += 1
    return out


def _limits(rest, prm):
    """seg -> kwargs for _local_fix: the joint limits re-applied after any temporal
    processing (smoothing / gap filling) so no filtered pose leaves the legal range."""
    L = {"head": dict(twist_lim=prm.head_limit, swing_lim=prm.head_limit)}
    for s in ("L", "R"):
        Fu0, Ft0 = rest.F["uarm_" + s], rest.F["thigh_" + s]
        L["uarm_" + s] = dict(twist_lim=prm.upperarm_twist_limit)
        L["farm_" + s] = dict(twist_lim=prm.forearm_twist_limit, total=("uarm_" + s, prm.arm_total_twist_limit),
                              hinge=(Fu0[:, 0], Fu0[:, 1], +1.0, prm.hyperextend, prm.elbow_max, prm.hinge_strict))
        L["hand_" + s] = dict(twist_lim=HAND_TWIST_LIMIT, swing_lim=prm.hand_limit)
        L["thigh_" + s] = dict(twist_lim=prm.thigh_twist_limit, swing_lim=THIGH_SWING_LIMIT)
        L["shin_" + s] = dict(twist_lim=prm.shin_twist_limit,
                              hinge=(Ft0[:, 0], Ft0[:, 1], -1.0, prm.hyperextend, prm.knee_max, prm.hinge_strict))
        L["foot_" + s] = dict(twist_lim=FOOT_TWIST_LIMIT, swing_lim=prm.foot_limit)
    return L


def _enforce(d, seg, Dp, Dn, rest, lim, st=None):
    kw = dict(lim)
    tot = kw.pop("total", None)
    rng = None
    if tot is not None:
        up, T = tot
        par = CTRL_SEG[CTRL_PARENT[SEG_CTRL[up]]]
        _, tw_u = swing_twist(d[par].T @ d[up], rest.F[up][:, 1])
        rng = (-math.radians(T) - tw_u, math.radians(T) - tw_u)
    return _local_fix(Dp, Dn, rest.F[seg][:, 1], twist_range=rng, st=st, key="enf_" + seg, **kw)


def _qang(a, b):
    return 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


def _despike_quats(Q, thr, max_len=6):
    """Replace short excursions (jump > thr deg that comes back within max_len frames)
    by a slerp between the frames around it.  Removes single-frame data glitches."""
    Q = Q.copy()
    n = len(Q)
    k = 1
    while k < n:
        if _qang(Q[k], Q[k - 1]) > thr:
            for j in range(k + 1, min(n, k + max_len + 1)):
                if _qang(Q[j], Q[k - 1]) < thr * 0.5:
                    for i in range(k, j):
                        Q[i] = q_slerp(Q[k - 1], Q[j], (i - k + 1) / (j - k + 1))
                    k = j
                    break
        k += 1
    for k in range(1, n):
        if np.dot(Q[k], Q[k - 1]) < 0:
            Q[k] = -Q[k]
    return Q


def _limit_ang_vel(Q, max_deg):
    """Forward-backward angular speed limiter (deg/frame) on a quaternion track."""
    def one(Q):
        Q = Q.copy()
        for k in range(1, len(Q)):
            a = _qang(Q[k - 1], Q[k])
            if a > max_deg:
                Q[k] = q_slerp(Q[k - 1], Q[k], max_deg / a)
        return Q
    F = one(Q)
    B = one(F[::-1])[::-1]
    out = np.array([q_slerp(f, b, 0.5) for f, b in zip(F, B)])
    for k in range(1, len(out)):
        if np.dot(out[k], out[k - 1]) < 0:
            out[k] = -out[k]
    return out


def smooth_rotations(Ds, rest, prm):
    """Gaussian smoothing of the parent-local rotations (hemisphere aligned), then
    re-composition parent -> child with the elbow / knee hinge re-enforced."""
    lims = _limits(rest, prm)
    Lq = {}
    for c, p, seg in CONTROLS:
        Q = np.array([m_to_q((d[CTRL_SEG[p]].T if p else np.eye(3)) @ d[seg]) for d in Ds])
        for k in range(1, len(Q)):
            if np.dot(Q[k], Q[k - 1]) < 0:
                Q[k] = -Q[k]
        if prm.rot_despike > 0:
            Q = _despike_quats(Q, prm.rot_despike)
        if prm.max_ang_vel > 0:
            Q = _limit_ang_vel(Q, prm.max_ang_vel)
        Q = _gauss1d(Q, prm.rot_smooth)
        Lq[c] = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    st = {"_twist_rate": prm.twist_rate_limit}   # continuity (unwrap + rate limit) as in the solve
    for k, d in enumerate(Ds):
        for c, p, seg in CONTROLS:
            Dp = d[CTRL_SEG[p]] if p else np.eye(3)
            Dn = Dp @ q_to_m(Lq[c][k])
            if seg in lims:
                Dn = _enforce(d, seg, Dp, Dn, rest, lims[seg], st)
            d[seg] = Dn


def _root_from_feet(Ds, model, tr):
    """XY root trajectory that keeps the lower (planted) foot still: integrate minus the
    weighted horizontal velocity of both feet, the lower foot dominating."""
    n = len(Ds)
    A = {s: np.zeros((n, 3)) for s in "LR"}
    low = {s: np.zeros(n) for s in "LR"}
    for k, d in enumerate(Ds):
        pos = fk_heads(d, model, tr[k])
        for s in "LR":
            pts = [fk_point(d, model, pos, nm + "_" + s) for nm in ("heel", "ball", "tip")]
            A[s][k] = np.mean(pts, axis=0)
            low[s][k] = min(p[2] for p in pts)
    wl = 1.0 / (1.0 + np.exp(np.clip((low["L"] - low["R"]) / 0.012, -50, 50)))
    wl = _gauss1d(wl, 1.0)
    v = np.zeros((n, 2))
    v[1:] = -(wl[1:, None] * np.diff(A["L"][:, :2], axis=0) + (1 - wl[1:, None]) * np.diff(A["R"][:, :2], axis=0))
    return np.cumsum(v, axis=0)


# joints a control's solved rotation depends on (gap filling is per control)
def _ctrl_joints(c):
    side = 0 if c.endswith(".L") else 1
    def jj(l, r):
        return l if side == 0 else r
    base = c.split(".")[0]
    return {"upper_arm_fk": [jj(11, 12), jj(13, 14)], "forearm_fk": [jj(13, 14), jj(15, 16)],
            "hand_fk": [jj(15, 16), jj(17, 18), jj(19, 20)], "thigh_fk": [23, 24, jj(25, 26)],
            "shin_fk": [jj(25, 26), jj(27, 28)], "foot_fk": [jj(25, 26), jj(27, 28), jj(31, 32)],
            "toe_fk": [jj(27, 28), jj(31, 32)], "neck": [0, 7, 8, 11, 12], "head": [0, 7, 8, 11, 12],
            "spine_fk": [23, 24, 11, 12], "shoulder": [11, 12],
            }.get(base, [11, 12, 23, 24])


def _fill_rotation_gaps(Ds, vm):
    """Per control: a segment whose joints were observed keeps its solved *world* rotation;
    one that was not observed gets its parent-local rotation slerped between the nearest
    frames where it was observed, under its (possibly observed) parent.  Nothing is solved
    from linearly interpolated joint positions."""
    n = len(Ds)
    W0 = [{k: v.copy() for k, v in d.items()} for d in Ds]
    locs = {}
    for c, p, seg in CONTROLS:
        locs[c] = [m_to_q((d[CTRL_SEG[p]].T if p else np.eye(3)) @ d[seg]) for d in W0]
    oks = {c: vm[:, _ctrl_joints(c)].all(1) for c, _, _ in CONTROLS}
    for k in range(n):
        for c, p, seg in CONTROLS:
            ok = oks[c]
            if ok[k] or not ok.any():
                Ds[k][seg] = W0[k][seg]
                continue
            good = np.where(ok)[0]
            i = np.searchsorted(good, k)
            a_ = good[i - 1] if i > 0 else None
            b_ = good[i] if i < len(good) else None
            if a_ is None:
                q = locs[c][b_]
            elif b_ is None:
                q = locs[c][a_]
            else:
                q = q_slerp(locs[c][a_], locs[c][b_], (k - a_) / (b_ - a_))
            Dp = Ds[k][CTRL_SEG[p]] if p else np.eye(3)
            Ds[k][seg] = Dp @ q_to_m(q)


def _intervals(c):
    ivs, k = [], 0
    while k < len(c):
        if c[k]:
            e = k
            while e < len(c) and c[e]:
                e += 1
            ivs.append((k, e))
            k = e
        else:
            k += 1
    return ivs


def _flat_foot(D, a_rest):
    """Yaw-only orientation with the same heading as D (the rest foot is flat)."""
    h0 = np.array([a_rest[0], a_rest[1], 0.0])
    a1 = D @ a_rest
    h1 = np.array([a1[0], a1[1], 0.0])
    if np.linalg.norm(h0) < 1e-6 or np.linalg.norm(h1) < 1e-6:
        return D
    yaw = math.atan2(np.cross(h0, h1)[2], np.dot(h0, h1))
    return rot_axis([0, 0, 1.0], yaw)


def _foot_lock(Ds, model, rest, prm, tr, fps, mocap_feet=None):
    """Contact detection, planted-foot anchoring (position + flat orientation), hip drop
    so locked feet stay reachable, 2-bone IK (fixed bone lengths) and penetration fix."""
    n = len(Ds)
    sides = ("L", "R")
    contacts, W, TGT, LOCKLIM = {}, {}, {}, {}
    B = max(1, prm.foot_lock_blend)

    def ankle_low(d, k, s):
        pos = fk_heads(d, model, tr[k])
        low = min(fk_point(d, model, pos, nm + "_" + s)[2] for nm in ("heel", "ball", "tip")) - model.floor_rest
        return fk_point(d, model, pos, "ankle_" + s), low

    def wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi

    for s in sides:
        A = np.zeros((n, 3))
        low = np.zeros(n)
        for k, d in enumerate(Ds):
            A[k], low[k] = ankle_low(d, k, s)
        sp = np.zeros(n)
        sp[1:] = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1)
        sp[0] = sp[1] if n > 1 else 0
        c = _contacts(low, _gauss1d(sp, 1.0), prm, fps)
        if mocap_feet is not None:
            # world-space input: the performer's own foot (scaled) says when it is planted;
            # the rig-based detection is kept for frames whose foot joints were not observed
            mlow, msp, mok = mocap_feet[s]
            cm = _contacts(mlow, _gauss1d(msp, 1.0), prm, fps)
            c = np.where(mok, cm, c)
            c = _contacts_clean(c)
        contacts[s] = c
        w = np.zeros(n)
        wsum = np.zeros(n)
        axy = np.zeros((n, 2))
        qacc = [None] * n
        a_rest = rest.F["foot_" + s][:, 1]
        # leg heading (shin-implied foot direction) per frame, de-noised
        hd = np.array([Ds[k]["shin_" + s] @ rest.F["shin_" + s][:, 2] for k in range(n)])
        yaw = _gauss1d(np.unwrap(np.arctan2(hd[:, 1], hd[:, 0])), 4.0)
        sf = rest.F["shin_" + s][:, 2]
        yaw_off = math.atan2(sf[1], sf[0]) - math.atan2(a_rest[1], a_rest[0])
        for (a, e) in _intervals(c):
            m = min(e - a, 10)   # the planting moment defines the initial position / orientation
            # position ratchet: the anchor creeps (<= foot_creep per frame) towards the free
            # ankle only while that is further than foot_lock_radius away, so mocap foot drift
            # never over-stretches the leg and never turns into a visible step or slide
            xy = A[a:a + m, :2].mean(axis=0)
            xys = []
            for k in range(a, e):
                dv = A[k, :2] - xy
                dd = np.linalg.norm(dv)
                if dd > prm.foot_lock_radius:
                    xy = xy + dv / dd * min(dd - prm.foot_lock_radius, prm.foot_creep)
                xys.append(xy.copy())
            Qf = np.array([m_to_q(Ds[k]["foot_" + s]) for k in range(a, a + m)])
            Qf[Qf @ Qf[0] < 0] *= -1
            qa = Qf.mean(axis=0)
            qa /= np.linalg.norm(qa)
            # a planted foot is flat on the floor (also hides depth-foreshortened toes)
            qa = q_slerp(qa, m_to_q(_flat_foot(q_to_m(qa), a_rest)), prm.foot_flatten)
            v = q_to_m(qa) @ a_rest
            h_foot = math.atan2(v[1], v[0])
            # heading: mostly from the leg (the shin's implied foot direction); the foot
            # points alone are the noisiest part of the capture
            h_leg = math.atan2(np.mean(np.sin(yaw[a:a + m])), np.mean(np.cos(yaw[a:a + m]))) - yaw_off
            h0 = h_foot + prm.foot_heading_from_leg * wrap(h_leg - h_foot)
            qa = m_to_q(rot_axis([0, 0, 1.0], h0 - h_foot) @ q_to_m(qa))
            # pivot ratchet: the planted foot turns only when the leg has turned more than
            # foot_pivot deg away from it (rotation about the ankle, at <= 4 deg/frame)
            piv = math.radians(prm.foot_pivot)
            hk, qks = h0, []
            for k in range(a, e):
                dl = wrap(yaw[k] - yaw_off - hk)
                if abs(dl) > piv:
                    hk += float(np.clip(dl - math.copysign(piv, dl), -math.radians(4), math.radians(4)))
                qks.append(m_to_q(rot_axis([0, 0, 1.0], hk - h0) @ q_to_m(qa)))
            for k in range(max(0, a - B), min(n, e + B)):
                wk = 1.0 if a <= k < e else (1.0 - (a - k) / (B + 1) if k < a else 1.0 - (k - e + 1) / (B + 1))
                wk = wk * wk * (3 - 2 * wk)
                q0 = qks[min(max(k - a, 0), e - a - 1)]
                xy = xys[min(max(k - a, 0), e - a - 1)]
                q_ = -q0 if (qacc[k] is not None and np.dot(qacc[k], q0) < 0) else q0
                wsum[k] += wk
                axy[k] += wk * xy
                qacc[k] = wk * q_ if qacc[k] is None else qacc[k] + wk * q_
                w[k] = max(w[k], wk)
        # looser than the free-foot limits: the planted foot must not follow the noisy shin,
        # only stay clear of Rigify's shin/foot twist-blend flip
        lock_lim = dict(twist_lim=FOOT_LOCK_TWIST_LIMIT, swing_lim=prm.foot_limit + FOOT_LOCK_SWING_EXTRA)
        tgt = np.full((n, 3), np.nan)
        for k, d in enumerate(Ds):
            if w[k] > 0:
                qan = qacc[k] / np.linalg.norm(qacc[k])
                d["foot_" + s] = d["toe_" + s] = q_to_m(q_slerp(m_to_q(d["foot_" + s]), qan, w[k]))
        if prm.max_ang_vel > 0:
            # re-orienting the planted foot can be abrupt at the lock blends: cap its speed
            Qf = np.array([m_to_q(d["foot_" + s]) for d in Ds])
            for k in range(1, n):
                if np.dot(Qf[k], Qf[k - 1]) < 0:
                    Qf[k] = -Qf[k]
            Qf = _limit_ang_vel(Qf, prm.max_ang_vel)
            for k, d in enumerate(Ds):
                d["foot_" + s] = d["toe_" + s] = q_to_m(Qf[k])
        for k, d in enumerate(Ds):
            if w[k] <= 0:
                continue
            A[k], low[k] = ankle_low(d, k, s)
            t = A[k].copy()
            t[:2] = (1 - w[k]) * A[k, :2] + w[k] * axy[k] / wsum[k]
            if c[k] or low[k] < 0:
                t[2] = A[k, 2] - low[k]
            tgt[k] = t
        W[s], TGT[s], LOCKLIM[s] = w, tgt, lock_lim
    # a locked foot out of reach: first lift its heel (the foot rolls about the ball, the
    # toes stay flat on the floor, as in a push-off) up to heel_rise_max, then lower the
    # hips for the rest (no stretch!).  Without the heel lift nearly every stance frame near
    # full extension dropped the hips and bent both knees ~15 deg more than the performer's
    # (worse for short legs with big feet).  Pivoting about the heel in front of the hips
    # (toes up) was tried: it switches pivots mid-stance and flips the foot.
    drop = np.zeros(n)
    heel_max = math.radians(max(0.0, prm.heel_rise_max))
    for s in sides:
        l1 = np.linalg.norm(model.points["knee_" + s][1] - model.points["hip_" + s][1])
        l2 = np.linalg.norm(model.points["ankle_" + s][1] - model.points["knee_" + s][1])
        reach = (l1 + l2) * 0.99
        ball_off = model.H["toe_fk." + s] - model.points["ankle_" + s][1]
        for k, d in enumerate(Ds):
            if W[s][k] <= 0:
                continue
            hip = fk_point(d, model, fk_heads(d, model, tr[k]), "hip_" + s)
            dv = TGT[s][k] - hip
            if heel_max > 0 and np.linalg.norm(dv) > reach:
                Df = d["foot_" + s]
                ax = perp(Df @ rest.F["foot_" + s][:, 0], np.array([0, 0, 1.0]))
                if np.linalg.norm(ax) > 1e-6:
                    ax = nrm(ax)
                    ball = TGT[s][k] + Df @ ball_off
                    r0 = TGT[s][k] - ball
                    sg = 1.0 if (rot_axis(ax, 0.1) @ r0)[2] > r0[2] else -1.0

                    def ank(ph):
                        return ball + rot_axis(ax, sg * ph) @ r0
                    if np.linalg.norm(ank(heel_max) - hip) > reach:
                        ph = heel_max
                    else:
                        lo, hi = 0.0, heel_max
                        for _ in range(14):
                            mid_ = 0.5 * (lo + hi)
                            lo, hi = (mid_, hi) if np.linalg.norm(ank(mid_) - hip) > reach else (lo, mid_)
                        ph = hi
                    TGT[s][k] = ank(ph)
                    d["foot_" + s] = rot_axis(ax, sg * ph) @ Df   # the toe keeps its flat orientation
                    dv = TGT[s][k] - hip
            hz = np.linalg.norm(dv[:2])
            if np.linalg.norm(dv) > reach and hz < reach:
                drop[k] = max(drop[k], (hip[2] - TGT[s][k][2]) - math.sqrt(reach ** 2 - hz ** 2))
    if drop.any() and prm.foot_lock_hip_drop:
        r = B
        pad = np.pad(drop, r, mode="edge")
        drop = np.max(np.stack([pad[i:i + n] for i in range(2 * r + 1)]), axis=0)
        drop = _gauss1d(drop, B / 2.0)
        tr[:, 2] -= drop
    # IK: locked feet to their targets, any other foot out of the floor
    for s in sides:
        for k, d in enumerate(Ds):
            pos = fk_heads(d, model, tr[k])
            if W[s][k] > 0:
                # pole: knee towards the planted foot's heading (blended by the lock weight)
                two_bone_ik(d, model, rest, pos, s, TGT[s][k], foot_pole=prm.foot_pole * W[s][k])
                # loose ankle limits only (the planted foot must not follow the noisy shin);
                # a toe bent by the heel lift keeps its own (flat) orientation
                rigid_toe = np.allclose(d["foot_" + s], d["toe_" + s])
                d["foot_" + s] = _enforce(d, "foot_" + s, d["shin_" + s], d["foot_" + s], rest, LOCKLIM[s])
                if rigid_toe:
                    d["toe_" + s] = d["foot_" + s]
        # final grounding: nothing below the floor after IK / ankle limits, and a foot that
        # is not planted (i.e. moving) clears the floor instead of skimming over it.
        for k, d in enumerate(Ds):
            A_, low_ = ankle_low(d, k, s)
            goal = max(0.0, prm.swing_clearance * (1.0 - W[s][k]))
            need = goal - low_
            if need > 1e-4:
                two_bone_ik(d, model, rest, fk_heads(d, model, tr[k]), s, A_ + np.array([0, 0, need]))
    return contacts


def mocap_to_rig(world, prm):
    """4x4 mocap -> armature space.  WORLD: the inverse of the rig's world matrix (the
    performance stays where it was captured); OBJECT: identity (it moves with the rig)."""
    if getattr(prm, "mocap_space", "WORLD") == "OBJECT":
        return np.eye(4)
    return np.linalg.inv(np.asarray(world, float))


def mocap_frames(T, prm):
    """The mocap frames used (prm.frame_start .. frame_end, -1 = last) of a T-frame take."""
    f0 = prm.frame_start
    f1 = T - 1 if prm.frame_end < 0 else prm.frame_end
    if not 0 <= f0 < T:
        raise ValueError("frame_start %d is outside the mocap's frames 0..%d" % (f0, T - 1))
    if f1 >= T:
        raise ValueError("frame_end %d is outside the mocap's frames 0..%d (-1 = last)" % (f1, T - 1))
    if f1 < f0:
        raise ValueError("frame_end %d is before frame_start %d" % (f1, f0))
    return np.arange(f0, f1 + 1)


def root_reference(Jm, vmask, frames):
    """(mocap leg length, mocap stance-ankle height) over the used *frames*, from the
    preprocessed mocap *Jm* in its own (world) frame and the observed-joint mask."""
    leg_m = np.linalg.norm(Jm[frames][:, [25, 26]] - Jm[frames][:, [23, 24]], axis=-1) + \
        np.linalg.norm(Jm[frames][:, [27, 28]] - Jm[frames][:, [25, 26]], axis=-1)
    ank = np.minimum(Jm[frames, 27, 2], Jm[frames, 28, 2])
    ok_a = vmask[frames][:, [27, 28]].all(1)
    mocap_ankle = float(np.percentile(ank[ok_a], 20)) if ok_a.sum() > 10 else 0.0
    return float(np.median(leg_m)), mocap_ankle


def root_scale(model, mocap_leg, prm):
    """Translation scale: the leg-length ratio (root_auto_scale) times root_scale."""
    return (model.leg_len / mocap_leg if prm.root_auto_scale else 1.0) * prm.root_scale


def root_target(hipc_m, scale, mocap_ankle, model, prm, Wi):
    """Target of the hip centre (N,3), armature space, for mocap hip centres *hipc_m* (N,3)
    in the mocap's own frame.  Shared by the bake and live mode.

    The scaling happens in the mocap frame (around the mocap world origin, which for a
    calibrated capture lies under the mean pelvis) and only then is the result mapped into
    the armature space with *Wi* (mocap_to_rig): with mocap_space WORLD the scaled
    performance therefore stays put in the world whatever the rig object's location /
    heading; with OBJECT (Wi = identity) it is attached to the rig object.  The hip height
    is scaled above the stance-ankle level, not the floor: the target's foot (ankle height)
    does not scale with its legs, and a mismatch would bend / over-extend every stance leg.
    IN_PLACE / FROM_FEET keep the rig's own hip XY (FROM_FEET then integrates the feet in
    the bake)."""
    Mi = np.linalg.inv(Wi)                        # armature -> mocap frame
    rig_ankle = 0.5 * (model.lm["left_ankle"][2] + model.lm["right_ankle"][2])
    rig_ankle_m = float(Mi[2, :3] @ np.array([0.0, 0.0, rig_ankle]) + Mi[2, 3])
    tm = np.empty((len(hipc_m), 3))
    tm[:, :2] = scale * hipc_m[:, :2]
    tm[:, 2] = rig_ankle_m + scale * (hipc_m[:, 2] - mocap_ankle) + prm.root_height_offset
    target = tm @ Wi[:3, :3].T + Wi[:3, 3]
    if prm.root_motion in ("IN_PLACE", "FROM_FEET"):
        target[:, :2] = model.hip_center[:2]
    return target


def solve_sequence(model, mocap, prm, progress=None):
    Jm, vmask = preprocess(mocap, prm, return_mask=True)
    # mocap world -> armature space of the rig
    Wi = mocap_to_rig(model.world, prm)
    J = Jm @ Wi[:3, :3].T + Wi[:3, 3]
    frames = mocap_frames(len(J), prm)
    f0 = frames[0]
    rest = RestFrames(model, prm)
    # translation scale from leg lengths
    mocap_leg, mocap_ankle = root_reference(Jm, vmask, frames)
    scale = root_scale(model, mocap_leg, prm)
    st = {}
    Ds = []
    for k, i in enumerate(frames):
        Ds.append(solve_pose(J[i], model, rest, prm, st))
        if progress and k % 50 == 0:
            progress(k / len(frames))
    # frames whose core skeleton was not observed: interpolate the solved rotations
    # (slerp of parent-local rotations) instead of trusting interpolated positions
    if prm.gap_fill_rotations:
        _fill_rotation_gaps(Ds, vmask[frames])
    if prm.rot_smooth > 0.01 or prm.rot_despike > 0 or prm.max_ang_vel > 0:
        smooth_rotations(Ds, rest, prm)
    # root translation: hip centre follows the scaled mocap hip centre
    target = root_target((Jm[frames, 23] + Jm[frames, 24]) / 2, scale, mocap_ankle, model, prm, Wi)
    rig_ankle = 0.5 * (model.lm["left_ankle"][2] + model.lm["right_ankle"][2])
    tr = np.zeros((len(frames), 3))
    for k, d in enumerate(Ds):
        pos0 = fk_heads(d, model, np.zeros(3))
        tr[k] = target[k] - fk_point(d, model, pos0, "hipc")
    # ground
    def lowest(k, s=None):
        pos = fk_heads(Ds[k], model, tr[k])
        sides = [s] if s else ["L", "R"]
        return min(fk_point(Ds[k], model, pos, n + "_" + x)[2] for x in sides for n in ("heel", "ball", "tip"))
    gmode = prm.ground_mode if prm.ground_mode != "AUTO" else ("CLAMP" if prm.root_motion == "FULL" else "PLANT")
    if gmode in ("PLANT", "CLAMP"):
        low = np.array([lowest(k) for k in range(len(frames))]) - model.floor_rest
        corr = -low if gmode == "PLANT" else np.maximum(0.0, -low)
        if gmode == "CLAMP" and prm.foot_lock:
            # a leg whose joints were not observed (gap-filled, e.g. an occluded crouch) must
            # not lift the observed hips: its penetration is removed later by bending it (IK)
            vmf = vmask[frames]
            obs = {"L": vmf[:, [23, 25, 27]].all(1), "R": vmf[:, [24, 26, 28]].all(1)}
            for k in range(len(frames)):
                sides_ok = [x for x in ("L", "R") if obs[x][k]]
                if len(sides_ok) < 2:
                    lk = lowest(k, sides_ok[0]) - model.floor_rest if sides_ok else 0.0
                    corr[k] = max(0.0, -lk)
            low = -corr   # the smoothing guard below then protects only the observed legs
        corr = _gauss1d(corr, prm.ground_smooth)
        corr = np.maximum(corr, -low)  # never let smoothing re-introduce penetration
        tr[:, 2] += corr
    if prm.root_motion == "FROM_FEET":
        tr[:, :2] += _root_from_feet(Ds, model, tr)
    mfeet = None
    if prm.root_motion == "FULL":
        mfeet = {}
        fz = J[frames][:, 27:33, 2][vmask[frames][:, 27:33]]
        floor_m = float(np.percentile(fz, 2)) * scale if fz.size > 20 else 0.0   # mocap floor height
        for s_, (ia, ih, it) in (("L", (27, 29, 31)), ("R", (28, 30, 32))):
            P = J[frames][:, [ia, ih, it]] * scale
            msp = np.zeros(len(frames))
            msp[1:] = np.linalg.norm(np.diff(P[:, 0, :2], axis=0), axis=1)
            mok = vmask[frames][:, [ia, ih, it]].all(1)
            mok[1:] &= mok[:-1]
            mfeet[s_] = (P[:, :, 2].min(axis=1) - floor_m, msp, mok)
    # foot-lock lengths are tuned for an adult-size rig: scale them with the target's legs
    # (a 1.2 m character's lifted foot is only half as high as an adult's)
    fprm = prm
    if prm.foot_scale_ref_leg > 0:
        u = model.leg_len / prm.foot_scale_ref_leg
        fprm = prm.copy(foot_lock_height=prm.foot_lock_height * u, foot_lock_speed=prm.foot_lock_speed * u,
                        foot_lock_radius=prm.foot_lock_radius * u, foot_creep=prm.foot_creep * u)
    contacts = _foot_lock(Ds, model, rest, fprm, tr, mocap["fps"], mfeet) if prm.foot_lock else {}
    sframes = prm.scene_start + (frames - f0)
    info = {"mocap_leg": float(mocap_leg), "rig_leg": float(model.leg_len), "scale": float(scale),
            "mocap_ankle_height": mocap_ankle, "rig_ankle_height": float(rig_ankle),
            "n_frames": int(len(frames)), "fps": float(mocap["fps"])}
    return Solution(frames, sframes, Ds, tr, contacts, J[frames], scale, info)


def control_bases(sol, model):
    """Local pose-bone quaternions per control (N,4) with hemisphere continuity + torso location."""
    out = {}
    for c, p, seg in CONTROLS:
        Rc = model.R[c]
        Q = np.zeros((len(sol.D), 4))
        for k, d in enumerate(sol.D):
            Dp = d[CTRL_SEG[p]] if p else np.eye(3)
            Q[k] = m_to_q(Rc.T @ (Dp.T @ d[seg]) @ Rc)
            if k and np.dot(Q[k], Q[k - 1]) < 0:
                Q[k] = -Q[k]
        out[c] = Q
    Rt = model.R["torso"]
    loc = sol.t @ Rt  # == (Rt.T @ t.T).T
    return out, loc


def ik_pose(D, model, t):
    """Local (basis) transforms of the Rigify IK controls that reproduce one solved FK frame.

    Every keyed IK control is rigidly attached to an FK control: its posed armature matrix is
    A @ X_rest, with A the FK control's rest->pose motion (x -> pos_c + D_seg (x - H_c)).  The
    IK controls are parented to Root (IK_parent = pole_parent = 1, Root not animated), so their
    basis rotation is X_rest^-1 A X_rest and their location the head offset in the bone's
    location axes (the parent's rest axes when Bone.use_local_location is off, as in Rigify).
    toe_ik's parent follows foot_ik, so only the toe's rotation relative to the foot is keyed.  Returns {bone: (loc (3,), quat (4,))}."""
    pos = fk_heads(D, model, t)
    out = {}
    for s, _ in SIDES:
        for L in IK_LIMBS.values():
            c = L["end_fk"] + "." + s
            Dc = D[CTRL_SEG[c]]
            X = model.ik_rest[L["end"] + "." + s]
            Rx, hx = X[:3, :3], X[:3, 3]
            head = pos[c] + Dc @ (hx - model.H[c])
            out[L["end"] + "." + s] = (model.ik_locframe[L["end"] + "." + s].T @ (head - hx), m_to_q(Rx.T @ Dc @ Rx))
            u = L["upper_fk"] + "." + s
            X = model.ik_rest[L["pole"] + "." + s]
            Rx, hx = X[:3, :3], X[:3, 3]
            head = pos[u] + D[CTRL_SEG[u]] @ (hx - model.H[u])
            out[L["pole"] + "." + s] = (model.ik_locframe[L["pole"] + "." + s].T @ (head - hx), np.array([1.0, 0, 0, 0]))
            if L["toe"]:
                Rx = model.ik_rest[L["toe"] + "." + s][:3, :3]
                Dr = D[CTRL_SEG[L["foot_fk"] + "." + s]].T @ D[CTRL_SEG[L["toe_fk"] + "." + s]]
                out[L["toe"] + "." + s] = (np.zeros(3), m_to_q(Rx.T @ Dr @ Rx))
    return out


def ik_bases(sol, model):
    """ik_pose() for every solved frame: {bone: (loc (N,3), quat (N,4))}, quaternion
    hemisphere continuity along the take."""
    per = [ik_pose(d, model, t) for d, t in zip(sol.D, sol.t)]
    out = {}
    for n in per[0]:
        loc = np.array([p[n][0] for p in per])
        Q = np.array([p[n][1] for p in per])
        for k in range(1, len(Q)):
            if np.dot(Q[k], Q[k - 1]) < 0:
                Q[k] = -Q[k]
        out[n] = (loc, Q)
    return out


def ik_active(model, prm):
    """True when the IK controls are keyed (control_mode != FK and the rig has them)."""
    mode = getattr(prm, "control_mode", "FK")
    if mode not in CONTROL_MODES:
        raise ValueError("control_mode must be one of %s, not %r" % (CONTROL_MODES, mode))
    if mode == "FK":
        return False
    if not model.has_ik:
        warnings.warn("control_mode %s: the rig has no standard Rigify IK limb controls; FK only" % mode)
        return False
    return True


# =============================================================== Blender side
def _rig_setup(rig, prm, ik=False):
    """Rigify switches for plain FK inheritance; with ik=True also the IK controls' setup
    (IK_Stretch 0, pole on, Root parents, bone ik_stretch 0, controls at identity) and the
    IK_FK switch per control_mode."""
    for pb in rig.pose.bones:
        if prm.set_fk and "IK_FK" in pb.keys():
            pb["IK_FK"] = 1.0
        if "FK_limb_follow" in pb.keys():
            pb["FK_limb_follow"] = 0.0
    if ik:
        pbs = rig.pose.bones
        for s, _ in SIDES:
            for kind, L in IK_LIMBS.items():
                pb = pbs[L["prop"] + "." + s]
                if prm.set_fk:
                    pb["IK_FK"] = IK_FK_SWITCH[prm.control_mode][kind]
                for k, v in (("IK_Stretch", 0.0), ("pole_vector", True), ("IK_parent", 1), ("pole_parent", 1)):
                    if k in pb.keys():
                        pb[k] = type(pb[k])(v)
                if prm.ik_zero_bone_stretch:
                    for n in L["chain"]:
                        pbs[n + "." + s].ik_stretch = 0.0
                for n in (L["end"], L["pole"], L["toe"]) + L["rest"]:
                    if n:
                        b = pbs[n + "." + s]
                        if n in (L["end"], L["toe"]):
                            b.rotation_mode = "QUATERNION"
                        b.matrix_basis = b.matrix_basis.Identity(4)
    tb = rig.pose.bones.get("torso")
    if tb is not None:
        # the solver assumes plain FK inheritance for the neck/head
        # (Rigify: constraint influence = 1 - follow, so follow = 1 -> pure FK)
        for k in ("neck_follow", "head_follow"):
            if k in tb.keys():
                tb[k] = 1.0
    for c, _, _ in CONTROLS + [("hips", 0, 0), ("chest", 0, 0)]:
        pb = rig.pose.bones.get(c)
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    rig.update_tag()   # custom-property switches drive constraints: re-evaluate the drivers


def bake(rig, sol, model, prm, scene=None):
    """Write an Action (quaternion keys + torso location) onto the rig and set the frame
    range / fps of *scene* (default: the rig's scene, else the context scene)."""
    from bpy_extras import anim_utils
    ik = ik_active(model, prm)
    if prm.step > 1 and sol.contacts:
        warnings.warn("step %d: only every %d-th frame is keyed (linear in between), so the foot "
                      "lock / planted feet between the keys are lost (feet may slide)" % (prm.step, prm.step))
    _rig_setup(rig, prm, ik)
    Q, loc = control_bases(sol, model)
    name = prm.action_name or "mocap_retarget"
    old = bpy.data.actions.get(name)
    if old is not None:
        others = [o for o in bpy.data.objects if o is not rig and o.animation_data and o.animation_data.action == old]
        if others:   # another character uses an action of that name: never delete its animation
            name = "%s_%s" % (name, rig.name)
            old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = act
    slot = rig.animation_data.action_slot or act.slots.new(id_type="OBJECT", name=rig.name)
    rig.animation_data.action_slot = slot
    cb = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    fr = sol.scene_frames.astype(np.float64)
    sel = np.arange(0, len(fr), max(1, prm.step))
    if sel[-1] != len(fr) - 1:
        sel = np.append(sel, len(fr) - 1)

    def put(path, idx, vals, group):
        fc = cb.fcurves.new(path, index=idx, group_name=group)
        fc.keyframe_points.add(len(sel))
        co = np.empty(2 * len(sel))
        co[0::2] = fr[sel]
        co[1::2] = vals[sel]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * len(sel))  # LINEAR
        fc.update()

    for c, q in Q.items():
        for i in range(4):
            put('pose.bones["%s"].rotation_quaternion' % c, i, q[:, i], c)
    for i in range(3):
        put('pose.bones["torso"].location', i, loc[:, i], "torso")
    if ik:   # IK controls matching the FK keys (the IK_FK sliders switch without a pop)
        for n, (lc, q) in ik_bases(sol, model).items():
            for i in range(3):
                put('pose.bones["%s"].location' % n, i, lc[:, i], n)
            if not n.startswith(tuple(L["pole"] for L in IK_LIMBS.values())):
                for i in range(4):
                    put('pose.bones["%s"].rotation_quaternion' % n, i, q[:, i], n)
    sol.info["control_mode"] = prm.control_mode if ik else "FK"
    act["mocap_params"] = json.dumps(asdict(prm))
    act["mocap_info"] = json.dumps(sol.info)
    sc = scene or (rig.users_scene[0] if rig.users_scene else bpy.context.scene)
    set_scene_fps(sc, sol.info["fps"])
    sc.frame_start = int(fr[0])
    sc.frame_end = int(fr[-1])
    return act


def set_scene_fps(sc, fps):
    """render.fps (int) and fps_base so that fps / fps_base == *fps* (29.97 -> 30 / 1.001)."""
    n = max(1, int(round(fps)))
    sc.render.fps = n
    sc.render.fps_base = n / float(fps)


def retarget(rig, npz_path, prm=None, progress=None, scene=None):
    """Headless entry point: solve + bake.  Returns info dict."""
    prm = prm or Params()
    mocap = load_mocap(npz_path)
    model = build_model(rig)
    sol = solve_sequence(model, mocap, prm, progress)
    act = bake(rig, sol, model, prm, scene)
    act["mocap_npz"] = npz_path if npz_path.startswith(EMBED_PREFIX) else os.path.abspath(npz_path)
    info = dict(sol.info)
    info["action"] = act.name
    info["contact_frames"] = {s: int(c.sum()) for s, c in sol.contacts.items()}
    return info


def apply_pose(rig, D, model, t, ik=False):
    """Set the pose bones directly from one solved frame (live mode)."""
    for c, p, seg in CONTROLS:
        Rc = model.R[c]
        Dp = D[CTRL_SEG[p]] if p else np.eye(3)
        q = m_to_q(Rc.T @ (Dp.T @ D[seg]) @ Rc)
        pb = rig.pose.bones[c]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = q
    rig.pose.bones["torso"].location = model.R["torso"].T @ t
    if ik:
        for n, (lc, q) in ik_pose(D, model, t).items():
            pb = rig.pose.bones[n]
            pb.location = lc
            if not n.startswith(tuple(L["pole"] for L in IK_LIMBS.values())):
                pb.rotation_quaternion = q


# ---------------------------------------------------------------- add-on UI
# Settings live on each rig object (Object.mocap_retarget), so several characters in one
# scene are tweaked / re-baked independently.  The panel edits the active rig (or the
# scene's "Rigify rig" pick when the active object is not a Rigify armature).
# Scene.mocap_retarget is kept for files made by version 1.0 (scene-level settings).
_LIVE = {}   # _live_key(rig) -> cache


def is_rigify_rig(ob):
    return ob is not None and ob.type == "ARMATURE" and "torso" in ob.data.bones and "thigh_fk.L" in ob.data.bones


def settings_owner(s):
    """The rig a settings group belongs to (object settings) or points to (scene settings)."""
    if bpy is not None and isinstance(s.id_data, bpy.types.Object):
        return s.id_data
    return getattr(s, "rig", None)


def panel_rig(context):
    ob = context.active_object
    if is_rigify_rig(ob):
        return ob
    s = getattr(context.scene, "mocap_retarget", None)
    return s.rig if s is not None else None


def params_from_settings(s):
    prm = Params()
    for f in fields(Params):
        if f.name == "offsets":
            continue
        if hasattr(s, f.name):
            setattr(prm, f.name, getattr(s, f.name))
    prm.offsets = {g: tuple(math.degrees(x) for x in getattr(s, "off_" + g)) for g in OFFSET_GROUPS}
    return prm


def params_from_scene(scene):
    """Version-1.0 API: parameters of the scene-level settings."""
    return params_from_settings(scene.mocap_retarget)


def settings_from_params(s, prm, npz_path=None):
    """Mirror Params (+ the npz path) into a settings group (object or scene)."""
    if npz_path is not None:
        s.npz_path = npz_path
    for f in fields(Params):
        if f.name != "offsets" and hasattr(s, f.name):
            try:
                setattr(s, f.name, getattr(prm, f.name))
            except Exception:
                pass
    for g in OFFSET_GROUPS:
        v = prm.offsets.get(g, (0, 0, 0))
        setattr(s, "off_" + g, tuple(math.radians(x) for x in v))


def _live_key(rig):
    """_LIVE cache key: the rig object's session uid (stable while the file is open, unlike
    its name, which can be reused by another object after a rename)."""
    return getattr(rig, "session_uid", None) or rig.as_pointer()


def live_solve_frame(scene, rig=None):
    """Re-solve the current frame for one rig (default: the scene settings' rig).

    Same per-frame solve and root target (root_target, over the same frame range and
    observed-joint mask) as the bake, so the live pose equals the baked one where the bake
    adds nothing temporal.  Not in live mode (they need the whole sequence): the foot lock /
    contacts, rotation smoothing / despike / speed cap and rotation gap filling, ground
    correction smoothing, and FROM_FEET's integration of the planted foot (live keeps the
    hips in place horizontally, like IN_PLACE)."""
    if rig is None:
        rig = scene.mocap_retarget.rig
    if rig is None:
        return
    s = rig.mocap_retarget
    if not s.npz_path:
        return
    prm = params_from_settings(s)
    npz = s.npz_path if s.npz_path.startswith(EMBED_PREFIX) else bpy.path.abspath(s.npz_path)
    key = (npz, prm.despike, prm.smooth_sigma, prm.mocap_yaw, prm.mocap_space, prm.reject_outliers,
           prm.fix_lr_swaps, prm.gap_erode, prm.infer_knees, prm.frame_start, prm.frame_end,
           tuple(np.round(np.array(rig.matrix_world), 6).ravel()))
    L = _LIVE.setdefault(_live_key(rig), {"key": None, "state": {}, "prev": None})
    if L["key"] != key:
        L["mocap"] = load_mocap(npz)
        L["model"] = build_model(rig)
        Jm, vmask = preprocess(L["mocap"], prm, return_mask=True)
        Wi = mocap_to_rig(L["model"].world, prm)
        L["Jm"], L["Wi"] = Jm, Wi
        L["J"] = Jm @ Wi[:3, :3].T + Wi[:3, 3]
        L["frames"] = mocap_frames(len(Jm), prm)
        L["mocap_leg"], L["mocap_ankle"] = root_reference(Jm, vmask, L["frames"])
        L["key"] = key
        L["state"] = {}
    model = L["model"]
    rest = RestFrames(model, prm)
    f0, f1 = int(L["frames"][0]), int(L["frames"][-1])
    i = int(min(max(prm.frame_start + (scene.frame_current - prm.scene_start), f0), f1))
    if L["prev"] is None or abs(L["prev"] - i) > 1:
        L["state"] = {}
    L["prev"] = i
    D = solve_pose(L["J"][i], model, rest, prm, L["state"])
    scale = root_scale(model, L["mocap_leg"], prm)
    hipc_m = (L["Jm"][i, 23] + L["Jm"][i, 24])[None] / 2
    tgt = root_target(hipc_m, scale, L["mocap_ankle"], model, prm, L["Wi"])[0]
    pos0 = fk_heads(D, model, np.zeros(3))
    t = tgt - fk_point(D, model, pos0, "hipc")
    gmode = prm.ground_mode if prm.ground_mode != "AUTO" else ("CLAMP" if prm.root_motion == "FULL" else "PLANT")
    if gmode != "NONE":
        pos = fk_heads(D, model, t)
        low = min(fk_point(D, model, pos, n + "_" + x)[2] for x in "LR" for n in ("heel", "ball", "tip")) - model.floor_rest
        t[2] += -low if gmode == "PLANT" else max(0.0, -low)
    apply_pose(rig, D, model, t, ik=ik_active(model, prm))


def _live_rigs(scene):
    rigs = [o for o in scene.objects if o.type == "ARMATURE" and getattr(o, "mocap_retarget", None) is not None
            and o.mocap_retarget.live]
    s = getattr(scene, "mocap_retarget", None)   # version-1.0 files: scene-level live toggle
    if s is not None and s.live and s.rig is not None and s.rig not in rigs:
        if not s.rig.mocap_retarget.npz_path:
            settings_from_params(s.rig.mocap_retarget, params_from_settings(s), s.npz_path)
        rigs.append(s.rig)
    return rigs


@persistent
def _live_handler(scene, depsgraph=None):
    for rig in _live_rigs(scene):
        try:
            live_solve_frame(scene, rig)
        except Exception as e:  # never break playback
            print("mocap live (%s):" % rig.name, e)


def _ensure_live_handler(on):
    # frame_change_pre: the pose is set before the depsgraph evaluates the new frame, so
    # every evaluation of the frame (viewport, F12 / animation renders) includes it; with
    # _post it relied on Blender re-evaluating what post handlers changed
    for hs in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        for h in list(hs):
            if getattr(h, "__name__", "") == "_live_handler":
                hs.remove(h)
    if on:
        bpy.app.handlers.frame_change_pre.append(_live_handler)


def _on_live(self, context):
    rig = settings_owner(self)
    if self.live and rig is not None:
        ad = rig.animation_data
        if ad and ad.action is not None:
            # live pose would fight the baked action: unassign it, remember it for Live off
            rig["mocap_live_prev_action"] = ad.action.name
            rig["mocap_live_prev_slot"] = ad.action_slot.identifier if ad.action_slot else ""
            ad.action = None
        prm = params_from_settings(self)
        _rig_setup(rig, prm, prm.control_mode != "FK" and build_model(rig).has_ik)
    elif rig is not None and "mocap_live_prev_action" in rig.keys():
        act = bpy.data.actions.get(rig["mocap_live_prev_action"])
        slot_id = rig.get("mocap_live_prev_slot", "")
        del rig["mocap_live_prev_action"]
        if "mocap_live_prev_slot" in rig.keys():
            del rig["mocap_live_prev_slot"]
        if act is not None and (rig.animation_data is None or rig.animation_data.action is None):
            rig.animation_data_create()
            rig.animation_data.action = act
            slot = act.slots.get(slot_id) if slot_id else None
            if slot is not None:
                rig.animation_data.action_slot = slot
        _LIVE.pop(_live_key(rig), None)
    _ensure_live_handler(bool(_live_rigs(context.scene)))
    if self.live:
        _live_handler(context.scene)


def _on_param(self, context):
    if self.live:
        _live_handler(context.scene)


def _on_cache_param(self, context):
    rig = settings_owner(self)
    if rig is not None:
        _LIVE.pop(_live_key(rig), None)
    _on_param(self, context)


if bpy is not None:
    from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                           IntProperty, PointerProperty, StringProperty)

    def _fp(default, mn, mx, desc, upd=_on_param, **kw):
        return FloatProperty(default=default, min=mn, max=mx, description=desc, update=upd, **kw)

    _ANN = {
        "npz_path": StringProperty(name="Mocap .npz", subtype="FILE_PATH", update=_on_cache_param,
                                   description="Path of the joints npz, or text:NAME for one embedded in this .blend"),
        "rig": PointerProperty(name="Rigify rig", type=bpy.types.Object,
                               poll=lambda self, o: o.type == "ARMATURE", update=_on_cache_param,
                               description="Rig the panel edits when the active object is not a Rigify rig"),
        "live": BoolProperty(name="Live (scrub)", default=False, update=_on_live,
                             description="Re-solve the current frame on frame change instead of using the baked action"),
        "frame_start": IntProperty(name="Mocap start", default=0, min=0, update=_on_param),
        "frame_end": IntProperty(name="Mocap end", default=-1, min=-1, update=_on_param),
        "step": IntProperty(name="Key every", default=1, min=1, max=10),
        "scene_start": IntProperty(name="Scene start", default=1, update=_on_param),
        "mocap_yaw": _fp(0.0, -180, 180, "Rotate mocap about Z (deg)", _on_cache_param),
        "mocap_space": EnumProperty(name="Mocap space", items=[
            ("WORLD", "World", "Mocap coordinates are world coordinates (moving the rig object does not move the motion)"),
            ("OBJECT", "Rig object", "The mocap's world frame is attached to the rig object (move the rig to place the performance)")],
            default="WORLD", update=_on_cache_param),
        "reject_outliers": _fp(0.45, 0, 2, "Bone-length outlier rejection (0 = off)", _on_cache_param),
        "despike": IntProperty(name="Despike", default=5, min=0, max=15, update=_on_cache_param),
        "smooth_sigma": _fp(1.5, 0, 10, "Gaussian smoothing of the joints (frames)", _on_cache_param, name="Smoothing"),
        "rot_smooth": _fp(1.0, 0, 10, "Gaussian smoothing of solved rotations (bake only)", name="Rotation smoothing"),
        "fix_lr_swaps": BoolProperty(name="Fix L/R swaps", default=True, update=_on_cache_param),
        "gap_erode": IntProperty(name="Gap erode", default=2, min=0, max=10, update=_on_cache_param),
        "infer_knees": BoolProperty(name="Infer missing knees", default=True, update=_on_cache_param),
        "gap_fill_rotations": BoolProperty(name="Gap fill in rotation space", default=True),
        "max_ang_vel": _fp(30, 0, 180, "Max local rotation speed (deg/frame, bake only)"),
        "rot_despike": _fp(35, 0, 180, "Remove rotation glitches larger than this (deg, bake only)"),
        "pelvis_follow": _fp(0.6, 0, 1, "Pelvis tilt follows the trunk"),
        "trunk_lateral": _fp(0.5, 0, 1, "Lateral lean: 0 = shoulder line exact, 1 = trunk exact"),
        "neck_share": _fp(0.4, 0, 1, "Share of head rotation done by the neck"),
        "head_face_weight": _fp(0.5, 0, 1, "Head pitch from the face (1) or the neck line (0)"),
        "head_limit": _fp(75, 0, 180, "Head vs chest limit (deg)"),
        "clavicle_share": _fp(0.25, 0, 1, "Clavicle share of arm elevation"),
        "twist_mix": _fp(0.8, 0, 1, "Forearm twist taken from the hand plane", name="Twist mix"),
        "forearm_twist_limit": _fp(90, 0, 180, "deg"),
        "arm_total_twist_limit": _fp(150, 0, 175, "Upper arm + forearm twist limit (deg)"),
        "upperarm_twist_limit": _fp(95, 0, 180, "deg"),
        "thigh_twist_limit": _fp(60, 0, 180, "deg"),
        "shin_twist_limit": _fp(20, 0, 180, "deg"),
        "leg_twist_mix": _fp(0.5, 0, 1, "Shin twist taken from the foot direction"),
        "twist_rate_limit": _fp(20, 1, 180, "Max twist change per frame (deg)"),
        "continuity": _fp(0.25, 0, 2, "Prefer temporally consistent limb twist (deg error per deg twist change)"),
        "hinge_strict": _fp(1.0, 0, 1, "Elbow/knee bend restricted to the hinge axis", name="Hinge strictness"),
        "hyperextend": _fp(5, 0, 30, "Allowed hyper-extension (deg)"),
        "elbow_max": _fp(150, 0, 180, "deg"),
        "knee_max": _fp(155, 0, 180, "deg"),
        "bend_lo": _fp(3, 0, 90, "Below: limb twist from fallback reference"),
        "bend_hi": _fp(15, 0, 90, "Above: limb twist from bend plane"),
        "hand_mix": _fp(0.8, 0, 1, "Hand rotation amount"),
        "hand_limit": _fp(75, 0, 180, "deg"),
        "foot_limit": _fp(60, 0, 180, "deg"),
        "root_auto_scale": BoolProperty(name="Auto scale (leg ratio)", default=True, update=_on_param),
        "root_scale": _fp(1.0, 0.01, 10, "Root translation multiplier", name="Root scale"),
        "root_height_offset": _fp(0.0, -2, 2, "m", name="Root height offset", unit="LENGTH"),
        "root_motion": EnumProperty(name="Root motion", items=[("FULL", "Full", "Scaled mocap hip translation"),
                                                               ("IN_PLACE", "In place", "No XY translation"),
                                                               ("FROM_FEET", "From feet", "XY from the planted foot (hip-centred input)")],
                                    default="FULL", update=_on_param),
        "ground_mode": EnumProperty(name="Ground", items=[("AUTO", "Auto", "Clamp for full root motion, else plant"),
                                                          ("PLANT", "Plant", "Lowest foot on the floor"),
                                                          ("CLAMP", "Clamp", "Only prevent penetration"),
                                                          ("NONE", "None", "")], default="AUTO", update=_on_param),
        "ground_smooth": _fp(2.0, 0, 10, "frames"),
        "foot_lock": BoolProperty(name="Foot lock", default=True),
        "foot_lock_height": _fp(0.035, 0, 0.5, "m", name="Lock height", unit="LENGTH"),
        "foot_lock_speed": _fp(0.6, 0, 5, "m/s", name="Lock speed"),
        "foot_lock_blend": IntProperty(name="Lock blend", default=4, min=0, max=30),
        "foot_flatten": _fp(1.0, 0, 1, "Planted feet flat on the floor"),
        "heel_rise_max": _fp(35, 0, 80, "Planted foot out of reach: max heel lift (deg) before the hips drop"),
        "foot_scale_ref_leg": _fp(0.9, 0, 3, "Lock height/speed/radius/creep are for a rig with this leg length; "
                                  "scaled by rig leg / this (0 = absolute)", name="Foot ref. leg", unit="LENGTH"),
        "foot_pole": _fp(0.7, 0, 1, "Planted feet: knee follows the foot heading"),
        "foot_lock_radius": _fp(0.06, 0.0, 1, "Free-ankle drift before a planted foot creeps (m)", unit="LENGTH"),
        "foot_creep": _fp(0.005, 0, 0.1, "Max creep of a planted foot per frame (m)", unit="LENGTH"),
        "swing_clearance": _fp(0.015, 0, 0.2, "Min height of a non-planted foot (m)", unit="LENGTH"),
        "foot_lock_hip_drop": BoolProperty(name="Hip drop to reach", default=True),
        "foot_pivot": _fp(25, 1, 180, "Leg turn (deg) before a planted foot re-anchors"),
        "foot_heading_from_leg": _fp(0.7, 0, 1, "Planted feet: heading from the leg (1) or foot points (0)"),
        "set_fk": BoolProperty(name="Set IK/FK switches", default=True,
                               description="Set the Rigify IK_FK switches as the control mode says (off: leave them)"),
        "control_mode": EnumProperty(name="Controls", items=[
            ("FK", "FK", "Key the FK controls only; every limb in FK (IK_FK = 1)"),
            ("LEGS_IK_ARMS_FK", "Legs IK, arms FK",
             "Key the FK controls and matching IK controls (hand_ik, foot_ik, toe_ik, poles); legs switched to IK, "
             "arms to FK. Planted feet become editable foot_ik keys; the IK_FK sliders switch without a pop. "
             "Caveat: Rigify's IK chain has no forearm / shin twist, so a limb in IK moves the mocap's forearm "
             "twist (up to ~90 deg) into the wrist - keep the arms in FK"),
            ("BOTH", "FK + IK keys",
             "Key the FK controls and matching IK controls on all limbs; every limb left in FK (switch any limb "
             "to IK with its IK_FK slider without a pop). Same arm-twist caveat as Legs IK")],
            default="FK", update=_on_param,
            description="Which Rigify controls are keyed and which IK_FK switch state is set"),
        "ik_zero_bone_stretch": BoolProperty(
            name="Zero IK bone stretch", default=True,
            description="IK modes: set the bone-level IK stretch of Rigify's IK chain bones to 0. Rigify ships 0.1, "
                        "which lets the IK solver stretch the limbs by up to ~4% even with the IK Stretch slider at 0"),
        "action_name": StringProperty(name="Action", default="mocap_retarget"),
    }
    for _g in OFFSET_GROUPS:
        _ANN["off_" + _g] = FloatVectorProperty(name=_g, subtype="EULER", size=3, update=_on_param)

    MocapRetargetSettings = type("MocapRetargetSettings", (bpy.types.PropertyGroup,), {
        "__annotations__": _ANN, "__module__": __name__,
        "mocap_retarget_version": tuple(bl_info["version"])})

    class MOCAPRT_OT_retarget(bpy.types.Operator):
        """Solve the whole take and bake it onto this rig's FK (and, per Controls, IK) controls (its own action)"""
        bl_idname = "mocap_retarget.bake"
        bl_label = "Retarget / Re-bake"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            rig = panel_rig(context)
            if rig is None:
                self.report({"ERROR"}, "Select a Rigify rig (or pick one in the panel)")
                return {"CANCELLED"}
            s = rig.mocap_retarget
            if not s.npz_path:
                self.report({"ERROR"}, "Set the mocap .npz path of %s" % rig.name)
                return {"CANCELLED"}
            if s.live:
                s.live = False
            npz = s.npz_path if s.npz_path.startswith(EMBED_PREFIX) else bpy.path.abspath(s.npz_path)
            wm = context.window_manager
            wm.progress_begin(0, 100)
            try:
                info = retarget(rig, npz, params_from_settings(s),
                                progress=lambda f: wm.progress_update(int(f * 100)))
            finally:
                wm.progress_end()
            self.report({"INFO"}, "%s: baked %d frames into %s, scale %.3f" % (
                rig.name, info["n_frames"], info["action"], info["scale"]))
            return {"FINISHED"}

    class MOCAPRT_OT_reset_offsets(bpy.types.Operator):
        bl_idname = "mocap_retarget.reset_offsets"
        bl_label = "Reset offsets"

        def execute(self, context):
            rig = panel_rig(context)
            if rig is None:
                return {"CANCELLED"}
            for g in OFFSET_GROUPS:
                setattr(rig.mocap_retarget, "off_" + g, (0, 0, 0))
            return {"FINISHED"}

    class MOCAPRT_OT_from_action(bpy.types.Operator):
        """Load the parameters stored on the rig's baked action (or the scene settings of an older file)"""
        bl_idname = "mocap_retarget.settings_from_action"
        bl_label = "Load settings from baked action"

        def execute(self, context):
            rig = panel_rig(context)
            if rig is None:
                return {"CANCELLED"}
            act = rig.animation_data.action if rig.animation_data else None
            if act is not None and "mocap_params" in act.keys():
                prm = Params.from_dict(json.loads(act["mocap_params"]))
                settings_from_params(rig.mocap_retarget, prm, act.get("mocap_npz"))
            elif context.scene.mocap_retarget.npz_path:
                ss = context.scene.mocap_retarget
                settings_from_params(rig.mocap_retarget, params_from_settings(ss), ss.npz_path)
            else:
                self.report({"WARNING"}, "No stored settings found")
                return {"CANCELLED"}
            return {"FINISHED"}

    class VIEW3D_PT_mocap_retarget(bpy.types.Panel):
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Mocap Retarget"
        bl_label = "Mocap Retarget"

        def draw(self, context):
            col = self.layout.column()
            rig = panel_rig(context)
            if not is_rigify_rig(context.active_object):
                col.prop(context.scene.mocap_retarget, "rig")
            if rig is None:
                col.label(text="Select a generated Rigify rig", icon="INFO")
                return
            s = rig.mocap_retarget
            col.label(text="Settings of: %s" % rig.name, icon="ARMATURE_DATA")
            col.prop(s, "npz_path")
            row = col.row(align=True)
            row.operator("mocap_retarget.bake", icon="ACTION")
            row.prop(s, "live", toggle=True, icon="PLAY")
            act = rig.animation_data.action if rig.animation_data else None
            col.label(text="Action: %s" % (act.name if act else "-"))
            if not s.npz_path:
                col.operator("mocap_retarget.settings_from_action", icon="IMPORT")
            for title, names in (
                ("Frames", ("frame_start", "frame_end", "step", "scene_start", "mocap_yaw", "mocap_space")),
                ("Filtering", ("reject_outliers", "despike", "smooth_sigma", "rot_smooth", "rot_despike", "max_ang_vel", "gap_fill_rotations", "infer_knees", "gap_erode", "fix_lr_swaps")),
                ("Trunk / head", ("pelvis_follow", "trunk_lateral", "neck_share", "head_face_weight", "head_limit", "clavicle_share")),
                ("Twist / hinge", ("twist_mix", "forearm_twist_limit", "arm_total_twist_limit", "upperarm_twist_limit", "thigh_twist_limit",
                                   "shin_twist_limit", "leg_twist_mix", "twist_rate_limit", "continuity", "hinge_strict", "hyperextend", "elbow_max", "knee_max",
                                   "bend_lo", "bend_hi", "hand_mix", "hand_limit", "foot_limit")),
                ("Root / ground", ("root_auto_scale", "root_scale", "root_height_offset", "root_motion",
                                   "ground_mode", "ground_smooth")),
                ("Foot lock", ("foot_lock", "foot_lock_height", "foot_lock_speed", "foot_lock_blend", "foot_flatten", "foot_pole", "foot_heading_from_leg", "foot_pivot", "foot_lock_hip_drop", "swing_clearance", "foot_lock_radius", "foot_creep", "heel_rise_max", "foot_scale_ref_leg")),
                ("Output", ("control_mode", "set_fk", "ik_zero_bone_stretch", "action_name")),
            ):
                box = col.box()
                box.label(text=title)
                for n in names:
                    box.prop(s, n)
            box = col.box()
            box.label(text="Per-limb rotation offsets (bone local)")
            for g in OFFSET_GROUPS:
                box.prop(s, "off_" + g)
            box.operator("mocap_retarget.reset_offsets")

    _CLASSES = (MocapRetargetSettings, MOCAPRT_OT_retarget, MOCAPRT_OT_reset_offsets, MOCAPRT_OT_from_action,
                VIEW3D_PT_mocap_retarget)


def _registered_settings_class():
    """The settings class of the copy of this add-on currently registered (the type of
    Object.mocap_retarget; only panels are reachable through bpy.types), or None."""
    prop = bpy.types.Object.bl_rna.properties.get("mocap_retarget")
    return type(prop.fixed_type) if prop is not None else None


def _registered_version():
    """bl_info version of the copy of this add-on currently registered (None = none)."""
    old = _registered_settings_class()
    if old is None:
        return None
    v = getattr(old, "mocap_retarget_version", None)
    if v is None:   # copies before 1.1 do not tag the class: ask their module
        v = getattr(sys.modules.get(old.__module__), "bl_info", {}).get("version", (0,))
    return tuple(v)


def register():
    """Register the panel / settings.  Idempotent: a copy registered earlier (add-on, text
    module, another import) is replaced -- unless it comes from another module and has an
    equal or newer bl_info version (e.g. an installed add-on vs the older copy embedded in
    a .blend), which is then kept.  Returns False when it skipped."""
    old = _registered_settings_class()
    mine = tuple(bl_info["version"])
    if old is not None and old is not MocapRetargetSettings and old.__module__ != __name__:
        have = _registered_version()
        if have is not None and have >= mine:
            print("mocap_retarget %s (%s): version %s is already registered (%s); keeping it" % (
                ".".join(map(str, mine)), __name__, ".".join(map(str, have)), old.__module__))
            return False
    for c in _CLASSES:
        old = getattr(bpy.types, c.__name__, None)
        if old is not None and old is not c:
            try:
                bpy.utils.unregister_class(old)
            except Exception:
                pass
    for c in _CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:   # already registered (same class object)
            pass
    bpy.types.Scene.mocap_retarget = bpy.props.PointerProperty(type=MocapRetargetSettings)
    bpy.types.Object.mocap_retarget = bpy.props.PointerProperty(type=MocapRetargetSettings)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_on_load":
            bpy.app.handlers.load_post.remove(h)
    bpy.app.handlers.load_post.append(_on_load)
    return True


@persistent
def _on_load(*_):
    # files saved with a live rig: re-install the frame-change handler
    _LIVE.clear()
    for sc in bpy.data.scenes:
        if _live_rigs(sc):
            _ensure_live_handler(True)
            break


def unregister():
    _ensure_live_handler(False)
    for h in list(bpy.app.handlers.load_post):
        if getattr(h, "__name__", "") == "_on_load":
            bpy.app.handlers.load_post.remove(h)
    for t in (bpy.types.Scene, bpy.types.Object):
        if hasattr(t, "mocap_retarget"):
            del t.mocap_retarget
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


# ---------------------------------------------------------------- self-contained .blend
TEXT_MODULE = "mocap_retarget.py"
_CLI_GUARD = """if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
"""
_TEXT_FOOTER = """
# ---- embedded copy (text datablock, Register enabled): registers the panel when the
# .blend is opened with Python auto-run allowed (or run this text once: Text > Run Script)
if bpy is not None:
    register()
"""


def embed_text_module(name=TEXT_MODULE, source=None):
    """Store this module in the open .blend as a text datablock with "Register" (use_module)
    on, so the Mocap Retarget panel is available when the file is opened in Blender
    (Python auto-run must be allowed, or run the text once via Text > Run Script)."""
    if source is None:
        with open(os.path.abspath(__file__)) as fh:
            source = fh.read()
    if source.endswith(_TEXT_FOOTER):        # re-embedding from an embedded copy
        source = source[:-len(_TEXT_FOOTER)]
    if source.rstrip().endswith(_CLI_GUARD.rstrip()):  # running the text must not start the CLI
        source = source.rstrip()[:-len(_CLI_GUARD.rstrip())]
    txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    txt.clear()
    txt.write(source + _TEXT_FOOTER)
    txt.use_module = True
    return txt


def embed_npz(path, name=None):
    """Store an npz in the open .blend (base64 text datablock); returns the ``text:NAME``
    path that load_mocap / the panel accept, so the file can be re-baked anywhere."""
    import base64
    name = name or os.path.basename(path)
    txt = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    txt.clear()
    with open(path, "rb") as fh:
        txt.write(base64.b64encode(fh.read()).decode("ascii"))
    txt["mocap_source_path"] = os.path.abspath(path)
    return EMBED_PREFIX + name


# ---------------------------------------------------------------- CLI
def _parse_set(items, prm):
    for it in items or []:
        k, v = it.split("=", 1)
        if k.startswith("off_"):
            prm.offsets[k[4:]] = tuple(float(x) for x in v.split(","))
            continue
        cur = getattr(prm, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        setattr(prm, k, v)
    return prm


def main(argv):
    import argparse
    import time
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--blend", default="/home/user/data/mocap/target_rig.blend")
    p.add_argument("--npz", default="/home/user/data/mocap/take1_joints3d.npz")
    p.add_argument("--out", default="/home/user/data/mocap/retargeted.blend")
    p.add_argument("--rig", default="rig")
    p.add_argument("--set", nargs="*", help="param=value overrides (off_arm_L=x,y,z for offsets)")
    p.add_argument("--embed", action="store_true",
                   help="store this module (panel, text datablock with Register on) and the npz in the saved .blend")
    a = p.parse_args(argv)
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    register()
    rig = bpy.data.objects[a.rig]
    prm = _parse_set(a.set, Params())
    t0 = time.time()
    npz = embed_npz(a.npz) if a.embed else a.npz
    info = retarget(rig, npz, prm)
    # mirror into the panel settings (per rig, plus the scene pick) so the saved file opens
    # ready for tweaking
    npz_ui = npz if a.embed else os.path.abspath(a.npz)
    settings_from_params(rig.mocap_retarget, prm, npz_ui)
    settings_from_params(bpy.context.scene.mocap_retarget, prm, npz_ui)
    bpy.context.scene.mocap_retarget.rig = rig
    if a.embed:
        embed_text_module()
    print("retarget: %.1fs %s" % (time.time() - t0, json.dumps(info)))
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
