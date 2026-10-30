"""
Deterministic unit test for ``calibration.describe_rig()``.

Builds a synthetic 3-camera ``RigCalibration`` from known rotations/translations
(rotation matrices constructed with pure numpy, so no cv2 is needed beyond the
``rotation_matrix()`` Rodrigues call, which we gate). Asserts the rendered
description names every camera and reports the correct world positions
(``-R.T @ t``) and pairwise baselines.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ..calibration import (
    BoardConfig, CameraCalibration, RigCalibration, camera_center, describe_rig,
)


def _K():
    return np.array([[700.0, 0, 320.0], [0, 700.0, 240.0], [0, 0, 1.0]])


def _cam_looking_at_origin(cid, eye):
    """A camera placed at world ``eye`` looking at the board origin.

    Returns a CameraCalibration whose camera center equals ``eye`` exactly.
    """
    eye = np.asarray(eye, float)
    fwd = -eye / np.linalg.norm(eye)            # look toward origin
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    # World->camera rotation: rows are the camera axes in world coords.
    R = np.vstack([right, true_up, fwd])
    t = -R @ eye                                # so that -R.T @ t == eye
    rvec, _ = cv2.Rodrigues(R)
    return CameraCalibration(cid, (640, 480), _K(), np.zeros(5),
                             rvec.ravel(), t)


def test_describe_rig_positions_and_baselines():
    eyes = {
        "camL": np.array([-0.30, 0.05, 0.50]),
        "camC": np.array([0.0, 0.0, 0.55]),
        "camR": np.array([0.30, 0.05, 0.50]),
    }
    cams = {cid: _cam_looking_at_origin(cid, e) for cid, e in eyes.items()}
    rig = RigCalibration(board=BoardConfig(), cameras=cams)

    text = describe_rig(rig)

    # Every camera id appears.
    for cid in eyes:
        assert cid in text

    # Reported camera centers match the known eye positions.
    for cid, eye in eyes.items():
        C = camera_center(cams[cid])
        assert np.allclose(C, eye, atol=1e-6), f"{cid}: {C} != {eye}"

    # Position lines contain the correct rounded coordinates.
    for cid, eye in eyes.items():
        assert f"x={eye[0]:+.3f}" in text
        assert f"y={eye[1]:+.3f}" in text
        assert f"z={eye[2]:+.3f}" in text

    # Pairwise baselines are present and numerically correct.
    assert "baselines" in text
    base_LR = float(np.linalg.norm(eyes["camL"] - eyes["camR"]))
    assert f"{base_LR:.3f}" in text          # 0.600
    base_LC = float(np.linalg.norm(eyes["camL"] - eyes["camC"]))
    assert f"{base_LC:.3f}" in text


def test_describe_rig_single_camera_no_baseline_crash():
    cam = _cam_looking_at_origin("solo", np.array([0.0, 0.0, 0.5]))
    rig = RigCalibration(cameras={"solo": cam})
    text = describe_rig(rig)
    assert "solo" in text
    assert "need >= 2 cameras" in text
