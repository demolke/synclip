"""
Full-stack E2E against a KNOWN articulated pose: skin -> render -> MediaPipe ->
reconstruct -> measure recovered-vs-ground-truth.

Where ``test_e2e_render_pose`` closes the loop on the static bind A-pose and only
asserts plausibility, this test *poses* the rig into a known articulated stance
(elbows bent, arms forward, knees slightly bent), renders it from the same three
calibrated synthetic cameras, runs the **real** MediaPipe PoseLandmarker on each
render, reconstructs the 3D skeleton, and quantifies the recovered pose against
the ground truth we applied via :func:`capmo.skinning.posed_joint_positions`.

Unlike the static test this one is **un-gated**: cv2, mediapipe and the rig asset
are guaranteed in this environment, and MediaPipe is forced onto its CPU delegate
(no GL anywhere -- the rasterizer is pure numpy). Anything wrong here is a real
failure, not a skip.

Ground truth caveat: MediaPipe's body landmarks are anatomical surface points
(e.g. the acromion for a "shoulder"), offset from the rig's *bone* joints. So the
position metrics use a rigid Procrustes alignment, and the upper-arm direction /
elbow-flexion tolerances are loosened to absorb that fixed shoulder-landmark
offset (documented per assertion). The forearm/thigh/shin directions, which are
not anchored at the offset shoulder, recover tightly.

All measured numbers are printed (run with ``-s`` to see them).
"""

from __future__ import annotations

import math
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

from .. import skinning, synth
from ..calibration import CameraCalibration, RigCalibration
from ..core.pipeline import triangulate_poses
from ..core.pose import PoseEstimator, ensure_model
from ..retarget import JOINT_INDEX, retarget_pose

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")

# Render config mirrors the static E2E test: portrait 720p, full-height figure
# ~3 m back, front + -/+30 deg cameras built from the exact render extrinsics.
_W, _H = 720, 1280
_K = synth.pinhole_K(1100.0, 1100.0, _W / 2.0, _H / 2.0)
_CENTER = np.array([0.0, 0.9, 0.0])
_DIST = 3.0
_CAM_ANGLES = {"front": 0.0, "left": 30.0, "right": -30.0}

# Major joints we score: our joint name -> MediaPipe landmark index.
_MAJOR = {
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}
# Limbs scored for direction: name -> (proximal MP idx, distal MP idx) and the
# matching ground-truth joint pair.
_LIMBS = {
    "l_upperarm": ((11, 13), ("l_shoulder", "l_elbow")),
    "r_upperarm": ((12, 14), ("r_shoulder", "r_elbow")),
    "l_forearm": ((13, 15), ("l_elbow", "l_wrist")),
    "r_forearm": ((14, 16), ("r_elbow", "r_wrist")),
    "l_thigh": ((23, 25), ("l_hip", "l_knee")),
    "r_thigh": ((24, 26), ("r_hip", "r_knee")),
    "l_shin": ((25, 27), ("l_knee", "l_ankle")),
    "r_shin": ((26, 28), ("r_knee", "r_ankle")),
}


def _axis_angle(axis, deg: float) -> np.ndarray:
    """3x3 rotation of *deg* degrees about (unit-ised) *axis* (Rodrigues)."""
    a = math.radians(deg)
    x, y, z = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# The chosen pose, as local rotations on the rig's bone nodes. The bones'
# local +X is the elbow/knee hinge (verified empirically): +75 deg on each
# forearm bends the elbow to ~100 deg flexion and lifts the hands forward (out
# of self-occlusion); +20 deg on the upper arms brings them down/forward from
# the A-pose; +25 deg on the calves is a slight knee bend. Modest angles keep
# MediaPipe detection solid from all three views (heavier arm rotation only
# worsened the shoulder-landmark offset error, see module note).
_POSE = {
    "lowerarm_l": _axis_angle((1, 0, 0), 75.0),
    "lowerarm_r": _axis_angle((1, 0, 0), 75.0),
    "upperarm_l": _axis_angle((1, 0, 0), 20.0),
    "upperarm_r": _axis_angle((1, 0, 0), 20.0),
    "calf_l": _axis_angle((1, 0, 0), 25.0),
    "calf_r": _axis_angle((1, 0, 0), 25.0),
}


def _lambert_shade(verts, tris, base=(150, 180, 200),
                   light=(0.3, 0.5, 1.0), ambient: float = 0.35) -> np.ndarray:
    """Per-triangle Lambert intensity over *base* (BGR) -- form MediaPipe needs."""
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


def _cpu_estimator() -> PoseEstimator:
    """A MediaPipe estimator pinned to the CPU delegate (no GL dependency)."""
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(
            model_asset_path=ensure_model("lite"),
            delegate=mp_base.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.5)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def backend(image_bgr: np.ndarray):
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        return np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float64)

    return PoseEstimator(backend=backend)


def _render_views():
    """Skin the mesh into _POSE and render a BGR image + calibration per camera."""
    skin = skinning.load_skin(_RIG)
    verts = skinning.pose_rig_mesh(skin, _POSE)
    mesh = synth.Mesh(verts, skin.tris, _lambert_shade(verts, skin.tris))
    cams: dict[str, CameraCalibration] = {}
    images: dict[str, np.ndarray] = {}
    for cid, deg in _CAM_ANGLES.items():
        ang = math.radians(deg)
        eye = _CENTER + np.array([_DIST * math.sin(ang), 0.0, _DIST * math.cos(ang)])
        R, t = synth.look_at(eye, _CENTER)
        images[cid] = synth.rasterize(mesh, _K, R, t, _W, _H, bg=128)
        rvec, _ = cv2.Rodrigues(R)
        cams[cid] = CameraCalibration(cid, (_W, _H), _K, np.zeros(5), rvec.ravel(), t)
    return RigCalibration(cameras=cams), images


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) between two 3-vectors."""
    cosv = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(np.clip(cosv, -1.0, 1.0)))


def _flexion(p: np.ndarray, a: int, b: int, c: int) -> float:
    """Interior angle (degrees) at joint *b* between segments b->a and b->c."""
    return _angle(p[a] - p[b], p[c] - p[b])


def _rigid_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rigid (rotation+translation, no scale) Procrustes fit of A onto B."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                 # reflect -> keep a proper rotation
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return (R @ (A - cA).T).T + cB


def test_posed_reconstruction_matches_ground_truth():
    estimator = _cpu_estimator()
    rig, images = _render_views()

    # Detection must be solid in every view (un-gated: failure, not skip).
    torso = list(_MAJOR.values())
    for cid, img in images.items():
        res = estimator.estimate(img)
        assert res is not None, f"MediaPipe found no person in the {cid} render"
        assert res.visibility[torso].min() > 0.5, f"low torso visibility in {cid}"

    res = triangulate_poses(rig, {cid: [estimator.estimate(images[cid])]
                                  for cid in _CAM_ANGLES})
    P, valid = res.X[0], res.valid[0]
    assert sorted(res.cam_ids) == ["front", "left", "right"] and res.kept.all()

    gt = skinning.posed_joint_positions(_RIG, _POSE)

    # --- 1. Reprojection residual: the three views agree on each joint --------
    reproj = float(np.nanmax(res.err_px[0, torso]))
    print(f"\nreprojection residual (max over major joints): {reproj:.2f} px")
    # Observed ~9.8 px; calibration is exact, so this is pure detection scatter.
    assert reproj < 20.0, f"views disagree: {reproj:.1f} px"

    # --- 2. Per-joint 3D position error (raw + Procrustes-aligned) ------------
    names = list(_MAJOR)
    rec = np.array([P[_MAJOR[n]] for n in names])
    gtm = np.array([gt[JOINT_INDEX[n]] for n in names])
    for n in names:
        assert valid[_MAJOR[n]], f"joint {n} did not triangulate"

    raw = np.linalg.norm(rec - gtm, axis=1)
    aligned = np.linalg.norm(_rigid_align(rec, gtm) - gtm, axis=1)
    print(f"raw 3D pos err   (m): mean={raw.mean():.3f} "
          f"median={np.median(raw):.3f} max={raw.max():.3f}")
    print(f"aligned pos err  (m): mean={aligned.mean():.3f} "
          f"median={np.median(aligned):.3f} max={aligned.max():.3f}")
    # Observed aligned max ~0.07 m; MediaPipe landmarks sit a few cm off the bone
    # joints, so a human ~10 cm tolerance after rigid alignment is the right bar.
    assert aligned.max() < 0.10, f"aligned joint error {aligned.max():.3f} m too big"
    assert raw.max() < 0.12, f"raw joint error {raw.max():.3f} m too big"

    # --- 3. Limb-direction angular error (offset-invariant) -------------------
    ang_err = {}
    for limb, ((pa, pb), (ga, gb)) in _LIMBS.items():
        rv = P[pb] - P[pa]
        gv = gt[JOINT_INDEX[gb]] - gt[JOINT_INDEX[ga]]
        ang_err[limb] = _angle(rv, gv)
    print("limb direction errors (deg): "
          + ", ".join(f"{k}={v:.1f}" for k, v in ang_err.items()))
    # Forearm/thigh/shin are anchored away from the offset shoulder and recover
    # tightly (observed <=8 deg) -> 15 deg bar. The upper arm starts at the
    # acromion landmark, which is offset from the rig shoulder node, tilting the
    # measured upper-arm vector by a fixed ~25 deg -> looser 30 deg bar.
    for limb, err in ang_err.items():
        bound = 30.0 if "upperarm" in limb else 15.0
        assert err < bound, f"{limb} direction off by {err:.1f} deg (> {bound})"

    # --- 4. Joint-flexion recovery: did we recover the bend? ------------------
    for side in ("l", "r"):
        sh, el, wr = _MAJOR[f"{side}_shoulder"], _MAJOR[f"{side}_elbow"], _MAJOR[f"{side}_wrist"]
        hp, kn, an = _MAJOR[f"{side}_hip"], _MAJOR[f"{side}_knee"], _MAJOR[f"{side}_ankle"]
        rec_elbow = _flexion(P, sh, el, wr)
        gt_elbow = _flexion(gt, JOINT_INDEX[f"{side}_shoulder"],
                            JOINT_INDEX[f"{side}_elbow"], JOINT_INDEX[f"{side}_wrist"])
        rec_knee = _flexion(P, hp, kn, an)
        gt_knee = _flexion(gt, JOINT_INDEX[f"{side}_hip"],
                           JOINT_INDEX[f"{side}_knee"], JOINT_INDEX[f"{side}_ankle"])
        print(f"{side} elbow flexion: recovered={rec_elbow:.1f} deg gt={gt_elbow:.1f} deg "
              f"| knee: recovered={rec_knee:.1f} deg gt={gt_knee:.1f} deg")
        # The applied bend is clearly recovered. Elbow is systematically shallow
        # by ~18 deg (the same shoulder-landmark offset that tilts the upper
        # arm), so allow +/-25 deg; the knee tracks within ~6 deg -> +/-20 deg.
        assert abs(rec_elbow - gt_elbow) < 25.0, f"{side} elbow flexion off"
        assert abs(rec_knee - gt_knee) < 20.0, f"{side} knee flexion off"

    # --- 5. The recovered skeleton retargets to a finite, unit-quat pose ------
    target = skinning.CR.load_rig_skeleton(_RIG)
    pose = retarget_pose(target, P, valid)
    assert pose.root_valid and pose.bone_valid.any()
    for q in pose.local_rot[pose.bone_valid]:
        assert np.all(np.isfinite(q))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6
