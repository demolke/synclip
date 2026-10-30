"""
Triangulating per-view 2D poses of calibrated cameras into a 3D skeleton --
now the one multi-view triangulation (:mod:`capmo.multiview`, also behind
``reconstruct_take`` and :func:`capmo.core.pipeline.triangulate_poses`); the
single-instant ``capmo.reconstruct`` module it replaced is gone.

A known skeleton is projected into several synthetic cameras and reconstructed;
the recovered joints must match the originals. Visibility gating, the
reprojection residual and the ``CameraCalibration`` (rig.json) path are
exercised too. The skeleton is a random point cloud, so the body-specific
stages (upright check, ghost consensus, L/R swap repair) are switched off here;
they are covered in test_selfcal.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from .. import multiview as mv, synth

RAW = dict(upright_check=False, consensus=False, fix_swaps=False)


def _camera(eye):
    K = synth.pinhole_K(800.0, 800.0, 320.0, 240.0)
    R, t = synth.look_at(eye)
    return mv.Camera(K, np.zeros(5), R, t)


def _scene(seed=0, k=33):
    rng = np.random.default_rng(seed)
    skeleton = rng.uniform([-0.4, -0.4, -0.3], [0.4, 0.4, 0.3], size=(k, 3))
    cams = [_camera((0, 0, 2.0)), _camera((-1.1, 0.3, 1.6)), _camera((1.1, 0.3, 1.6))]
    return skeleton, cams


def _obs(skeleton, cams, vis=None):
    kp = np.stack([c.project(skeleton)[None] for c in cams])           # (V,1,33,2)
    v = np.ones(kp.shape[:3]) if vis is None else np.asarray(vis, float)[:, None, :]
    return kp, v


def test_reconstructs_known_skeleton():
    skeleton, cams = _scene()
    res = mv.reconstruct_take(cams, *_obs(skeleton, cams), **RAW)
    assert res.valid.all()
    assert np.max(np.linalg.norm(res.X[0] - skeleton, axis=1)) < 1e-6
    assert np.nanmax(res.err_px) < 1e-6
    assert (res.n_views == 3).all()


def test_visibility_gates_and_weights():
    skeleton, cams = _scene(seed=1)
    vis = np.ones((3, 33))
    vis[1, 0] = vis[2, 0] = 0.0          # joint 0 seen by one camera only
    vis[2, 1] = 0.0                      # joint 1 by exactly two
    res = mv.reconstruct_take(cams, *_obs(skeleton, cams, vis), vis_thresh=0.5, min_views=2, **RAW)
    assert not res.valid[0, 0] and res.n_views[0, 0] == 1
    assert np.isnan(res.X[0, 0]).all()
    assert res.valid[0, 1] and res.n_views[0, 1] == 2
    assert np.linalg.norm(res.X[0, 1] - skeleton[1]) < 1e-6
    assert res.valid[0, 2:].all()


def test_reprojection_residual_flags_disagreement():
    skeleton, cams = _scene(seed=2)
    kp, vis = _obs(skeleton, cams)
    kp[0, 0, 5] += np.array([10.0, -10.0])        # one view off by 14 px on joint 5
    res = mv.reconstruct_take(cams, kp, vis, **RAW)
    assert res.valid[0, 5]
    assert res.err_px[0, 5] > 1.0
    assert res.err_px[0, 6] < 1e-6                # an untouched joint stays clean
    # far off: a joint whose views cannot agree (mean error > max_err_px) is
    # rejected rather than returned wrong
    kp[0, 0, 5] += np.array([200.0, -200.0])
    res = mv.reconstruct_take(cams, kp, vis, **RAW)
    assert not res.valid[0, 5] and np.isnan(res.X[0, 5]).all()
    assert res.err_px[0, 5] > 40.0 and res.valid[0, 6]


def test_camera_calibration_projection_path():
    """rig.json cameras (CameraCalibration) triangulate through Camera.from_calibration."""
    cv2 = pytest.importorskip("cv2")
    from ..calibration import CameraCalibration
    from ..core.pipeline import triangulate_poses
    from ..core.pose import PoseResult

    skeleton, _ = _scene(seed=3)
    K = synth.pinhole_K(800.0, 800.0, 320.0, 240.0)
    cams, poses = {}, {}
    for cid, eye in [("a", (0, 0, 2.0)), ("b", (1.0, 0.0, 1.7))]:
        R, t = synth.look_at(eye)
        rvec, _ = cv2.Rodrigues(R)
        cam = CameraCalibration(cid, (640, 480), K, np.zeros(5), rvec.ravel(), t)
        P = cam.projection_matrix()
        Xh = np.c_[skeleton, np.ones(33)] @ P.T
        cams[cid] = cam
        poses[cid] = [PoseResult(Xh[:, :2] / Xh[:, 2:], np.ones(33))]
        assert np.allclose(mv.Camera.from_calibration(cam).project(skeleton), Xh[:, :2] / Xh[:, 2:])
    res = triangulate_poses(cams, poses, **RAW)
    assert res.valid.all()
    assert np.max(np.linalg.norm(res.X[0] - skeleton, axis=1)) < 1e-6
