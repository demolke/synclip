"""Moved to :mod:`capmo.core.pose` (the one MediaPipe pose wrapper + model
handling, shared by ``extract_keypoints`` and the pipeline); this old import
path stays as an alias."""
from .core.pose import (  # noqa: F401
    AVAILABLE, MODEL_DIR, NUM_LANDMARKS, PoseEstimator, PoseResult, _ensure_model,
    _mediapipe_backend, ensure_model, estimate_full_res, load_estimator, model_path,
    model_url, mp, resize_to_height)
