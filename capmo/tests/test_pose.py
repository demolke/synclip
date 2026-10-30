"""Pose wrapper logic (with an injected backend; no MediaPipe needed)."""

from __future__ import annotations

import numpy as np
import pytest

from ..pose import PoseEstimator, PoseResult, NUM_LANDMARKS


def _fake_raw(seed=0):
    rng = np.random.default_rng(seed)
    raw = np.zeros((NUM_LANDMARKS, 4))
    raw[:, 0] = rng.uniform(0, 1, NUM_LANDMARKS)     # x normalized
    raw[:, 1] = rng.uniform(0, 1, NUM_LANDMARKS)     # y normalized
    raw[:, 3] = rng.uniform(0, 1, NUM_LANDMARKS)     # visibility
    return raw


def test_normalized_landmarks_become_pixels():
    raw = _fake_raw()
    est = PoseEstimator(backend=lambda img: raw)
    img = np.zeros((480, 640, 3), np.uint8)
    res = est.estimate(img)
    assert isinstance(res, PoseResult)
    assert res.points.shape == (NUM_LANDMARKS, 2)
    assert np.allclose(res.points[:, 0], raw[:, 0] * 640)
    assert np.allclose(res.points[:, 1], raw[:, 1] * 480)
    assert np.allclose(res.visibility, raw[:, 3])
    assert res.points[:, 0].max() <= 640 and res.points[:, 1].max() <= 480


def test_no_person_returns_none():
    est = PoseEstimator(backend=lambda img: None)
    assert est.estimate(np.zeros((10, 10, 3), np.uint8)) is None


def test_wrong_landmark_count_raises():
    est = PoseEstimator(backend=lambda img: np.zeros((10, 4)))
    with pytest.raises(ValueError):
        est.estimate(np.zeros((10, 10, 3), np.uint8))


def test_visibility_is_clipped():
    raw = _fake_raw()
    raw[0, 3] = 2.0      # out-of-range confidence
    raw[1, 3] = -1.0
    est = PoseEstimator(backend=lambda img: raw)
    res = est.estimate(np.zeros((100, 100, 3), np.uint8))
    assert res.visibility[0] == 1.0 and res.visibility[1] == 0.0


def test_real_mediapipe_backend_on_blank_image():
    """The actual MediaPipe path constructs and finds no person in a blank frame."""
    pytest.importorskip("mediapipe")
    try:
        est = PoseEstimator(variant="lite")     # builds the real backend
    except Exception as exc:                    # offline: model can't be fetched
        pytest.skip(f"could not build MediaPipe PoseLandmarker: {exc}")
    assert est.estimate(np.zeros((480, 640, 3), np.uint8)) is None
