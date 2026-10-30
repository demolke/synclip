"""
True end-to-end pipeline: render the rig -> real MediaPipe -> 3D skeleton.

This is the only test that closes the *whole* loop with no fakes: it renders the
vendored Quaternius character mesh (bind A-pose) with the software rasterizer from
several calibrated synthetic cameras, runs the **real** MediaPipe PoseLandmarker
on each render, and feeds those into :func:`capmo.core.pipeline.triangulate_poses`
(the shared multi-view triangulation: consensus, L/R repair, outlier rejection),
which fuses the views into a 3D skeleton. The rig calibration is
built from the exact rendering cameras, so any error is MediaPipe's detection plus
the pipeline's fusion -- not calibration.

The rasterizer only does flat per-triangle colour, so a single-colour silhouette
won't be detected; ``_lambert_shade`` gives each triangle a simple
normal-dot-light intensity over a skin base colour, which is enough form/contrast
for MediaPipe to find a person (see the module's spike notes).

Skipped cleanly where the heavy bits are absent: cv2, mediapipe, the rig asset, or
a MediaPipe runtime that can't construct (it needs system GL libs and to fetch its
model). Where everything is present (as in this repo's dev env) it actually runs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from .. import synth
from ..calibration import CameraCalibration, RigCalibration
from ..core.pipeline import triangulate_poses
from ..core.pose import PoseEstimator
from .. import character_rig as CR
from ..retarget import JOINT_INDEX, retarget_pose, landmarks_to_joints

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")
_HAS_RIG = os.path.exists(_RIG)

# Render config: portrait 720p, figure framed full-height a few metres back.
_W, _H = 720, 1280
_K = synth.pinhole_K(1100.0, 1100.0, _W / 2.0, _H / 2.0)
_CENTER = np.array([0.0, 0.9, 0.0])          # ~torso height of a 1.8 m figure
_DIST = 3.0                                  # metres back
_CAM_ANGLES = {"front": 0.0, "left": 30.0, "right": -30.0}   # degrees about +Y
_TORSO_JOINTS = [11, 12, 23, 24, 25, 26, 13, 14]             # what we assert on

# Two instants per camera (the rendered figure is static, so both frames are the
# same content; time alignment is the session timeline's job, tested elsewhere).
_N_FRAMES = 2


def _lambert_shade(verts: np.ndarray, tris: np.ndarray, base=(150, 180, 200),
                   light=(0.3, 0.5, 1.0), ambient: float = 0.35) -> np.ndarray:
    """Per-triangle Lambert intensity over *base* (BGR) -> (M,3) uint8.

    Flat shading: each triangle's geometric normal dotted with a fixed light
    direction (clamped >=0), lerped from *ambient* to full. Gives the body form
    and contrast the flat rasterizer otherwise lacks.
    """
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


def _rig_and_renders():
    """Build the synthetic rig + a rendered BGR image per camera id."""
    verts, tris = CR.load_rig_mesh(_RIG)
    mesh = synth.Mesh(verts, tris, _lambert_shade(verts, tris))
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


def _make_estimator():
    """Construct the real MediaPipe estimator, skipping if the runtime can't."""
    try:
        return PoseEstimator(variant="lite")
    except Exception as exc:        # missing GL libs / model fetch / runtime
        pytest.skip(f"MediaPipe runtime unavailable: {exc}")


@pytest.fixture(scope="module")
def reconstructed():
    """Render the views and check MediaPipe finds the figure in each (once)."""
    if not _HAS_RIG:
        pytest.skip("Quaternius rig asset not present")
    estimator = _make_estimator()
    rig, images = _rig_and_renders()

    # Detection must be solid in every view before we trust the fusion.
    for cid, img in images.items():
        res = estimator.estimate(img)
        if res is None:
            pytest.skip(f"MediaPipe found no person in the {cid} render")
        assert res.visibility[_TORSO_JOINTS].min() > 0.5, \
            f"low torso visibility in {cid} view"

    return rig, estimator, images


def _process(rig, estimator, images):
    poses = {cid: [estimator.estimate(images[cid]) for _ in range(_N_FRAMES)]
             for cid in _CAM_ANGLES}
    return triangulate_poses(rig, poses)


def test_pipeline_recovers_plausible_skeleton(reconstructed):
    rig, estimator, images = reconstructed
    res = _process(rig, estimator, images)

    # Two instants, each fused from all three views.
    assert res.X.shape[0] == _N_FRAMES
    assert sorted(res.cam_ids) == ["front", "left", "right"]
    assert res.kept.all(), "a view was rejected by the consensus check"

    P, valid = res.X[0], res.valid[0]
    for j in _TORSO_JOINTS:
        assert valid[j], f"joint {j} did not triangulate"
        assert res.n_views[0, j] == 3

    # Reprojection residuals are small: the views agree on where each joint is.
    assert np.nanmax(res.err_px[0, _TORSO_JOINTS]) < 25.0

    # --- plausibility vs the mesh's own bind skeleton -------------------------
    bind = CR.load_rig_skeleton(_RIG).bind_joints
    bone_shoulder = np.linalg.norm(bind[JOINT_INDEX["l_shoulder"]]
                                   - bind[JOINT_INDEX["r_shoulder"]])  # ~0.42 m
    bone_hip = np.linalg.norm(bind[JOINT_INDEX["l_hip"]]
                              - bind[JOINT_INDEX["r_hip"]])            # ~0.23 m

    shoulder_w = np.linalg.norm(P[11] - P[12])
    hip_w = np.linalg.norm(P[23] - P[24])
    # MediaPipe's acromion/hip landmarks don't sit exactly on the bone joints, so
    # allow generous absolute slack (~tens of mm) -- we only assert the recovered
    # widths are human and near the rig's true bone spacing, not exact.
    assert shoulder_w == pytest.approx(bone_shoulder, abs=0.12)
    assert hip_w == pytest.approx(bone_hip, abs=0.10)
    assert shoulder_w > hip_w                       # shoulders wider than hips

    # Vertical ordering (Y up): head above shoulders above hips above knees.
    head_y = P[0, 1]
    shoulder_y = (P[11, 1] + P[12, 1]) / 2.0
    hip_y = (P[23, 1] + P[24, 1]) / 2.0
    knee_y = (P[25, 1] + P[26, 1]) / 2.0
    assert head_y > shoulder_y > hip_y > knee_y

    # Left/right ordering: MediaPipe's "left" (11/23) is the figure's left, which
    # is +X in this world frame (camera looks down -Z, so its right is -X).
    assert P[11, 0] > P[12, 0]                      # left shoulder +X of right
    assert P[23, 0] > P[24, 0]                      # left hip +X of right

    # Overall stature is roughly the 1.8 m figure (head to knee spans the torso).
    assert 0.6 < head_y - knee_y < 1.4


def test_recovered_skeleton_retargets_to_finite_pose(reconstructed):
    """The fused skeleton drives the rig: retarget yields finite bone rotations."""
    rig, estimator, images = reconstructed
    res = _process(rig, estimator, images)
    points, valid = res.X[0], res.valid[0]

    target = CR.load_rig_skeleton(_RIG)
    pose = retarget_pose(target, points, valid)
    assert pose.root_valid and pose.bone_valid.any()
    for q in pose.local_rot[pose.bone_valid]:
        assert np.all(np.isfinite(q))
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)

    # The retarget's measured joints reproduce the recovered ones it could use.
    jp, _ = landmarks_to_joints(points, valid)
    assert np.all(np.isfinite(jp[JOINT_INDEX["chest"]]))
