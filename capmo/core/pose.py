"""
2D body-pose estimation per camera view -- the ONE MediaPipe wrapper of capmo
(``extract_keypoints``, the offline pipeline and the tests all use it; the old
``capmo.pose`` path is an alias).

Wraps MediaPipe Pose (Tasks API ``PoseLandmarker``), which returns 33 body
landmarks for a single person. We keep what triangulation needs: each
landmark's pixel location and a visibility score (used both to gate which views
vote for a joint and to weight the triangulation), plus the metric
hip-centred world landmarks when the backend gives them.

Model handling: :func:`model_path` / :func:`model_url` / :func:`ensure_model`
(the ``pose_landmarker_{lite,full,heavy}.task`` files live in the ``capmo/``
package directory and are downloaded on first use); :func:`load_estimator`
prints where the model comes from (the extractor's console output).
:func:`estimate_full_res` runs a frame through the model at a capped height and
maps the landmarks back to full-resolution pixels.

MediaPipe is optional (``AVAILABLE``) and, like the decoder, the backend is
injectable so the conversion logic is testable without the model: a *backend* is
any callable ``image_bgr -> ndarray (33, 4)`` of normalized ``[x, y, z,
visibility]`` rows, or ``None`` when no person is found.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import mediapipe as mp  # type: ignore
    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where mediapipe is absent
    mp = None
    AVAILABLE = False

NUM_LANDMARKS = 33   # MediaPipe Pose topology

# the .task files live next to the capmo package (capmo/pose_landmarker_*.task)
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = MODEL_DIR
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


def model_path(variant: str = "full") -> str:
    """Where the *variant* ('lite' | 'full' | 'heavy') model file lives/will live."""
    return os.path.join(_MODEL_DIR, f"pose_landmarker_{variant}.task")


def model_url(variant: str = "full") -> str:
    return _MODEL_URL.format(variant=variant)


def ensure_model(variant: str = "full") -> str:
    """Path to the PoseLandmarker .task model, downloading it on first use."""
    path = model_path(variant)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = _MODEL_URL.format(variant=variant)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


_ensure_model = ensure_model          # old private name


@dataclass
class PoseResult:
    """One view's 2D pose."""
    points: np.ndarray        # (33, 2) pixel coordinates
    visibility: np.ndarray    # (33,) in [0, 1]
    world: np.ndarray | None = None   # (33, 3) metric hip-centred landmarks, if the backend gives them


def _mediapipe_backend(variant: str, min_confidence: float) -> Callable:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=ensure_model(variant)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def run(image_bgr: np.ndarray):
        # MediaPipe wants RGB; slicing avoids pulling cv2 into this module.
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        raw = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float64)
        wl = getattr(res, "pose_world_landmarks", None)
        if wl:
            world = np.array([[p.x, p.y, p.z] for p in wl[0]], np.float64)
            return raw, world
        return raw

    run.close = landmarker.close      # lets callers release the model deterministically
    return run


class PoseEstimator:
    """Runs pose on a BGR frame and returns pixel landmarks + visibility."""

    def __init__(self, backend: Callable | None = None, variant: str = "full",
                 min_confidence: float = 0.5) -> None:
        if backend is None:
            if not AVAILABLE:
                raise RuntimeError("mediapipe is not installed")
            backend = _mediapipe_backend(variant, min_confidence)
        self._backend = backend

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if close:
            close()

    def estimate(self, image_bgr: np.ndarray) -> PoseResult | None:
        """Return the view's :class:`PoseResult`, or ``None`` if no person."""
        raw = self._backend(image_bgr)
        if raw is None:
            return None
        world = None
        if isinstance(raw, tuple):          # backend may return (raw, world(33,3))
            raw, world = raw
            world = None if world is None else np.asarray(world, np.float64)
        raw = np.asarray(raw, np.float64)
        if raw.shape[0] != NUM_LANDMARKS:
            raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {raw.shape[0]}")
        h, w = image_bgr.shape[:2]
        points = raw[:, :2] * np.array([w, h], np.float64)   # normalized -> pixels
        visibility = np.clip(raw[:, 3], 0.0, 1.0)
        return PoseResult(points=points, visibility=visibility, world=world)


# ----------------------------------------------------------------- helpers
def load_estimator(variant: str = "full", log=print, **kw) -> PoseEstimator:
    """Build the MediaPipe estimator for *variant*, saying whether the model is
    cached or being downloaded (and from where)."""
    path = model_path(variant)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        log(f"MediaPipe model: pose_landmarker_{variant}.task (cached at {path})")
    else:
        log(f"MediaPipe model: pose_landmarker_{variant}.task -- downloading from "
            f"{model_url(variant)}\n  to {path}")
    est = PoseEstimator(variant=variant, **kw)
    log("Model ready.")
    return est


def resize_to_height(img: np.ndarray, h_max: int):
    """Downscale *img* (keeping the aspect) if it is taller than *h_max* (0 = never)."""
    import cv2
    h, w = img.shape[:2]
    if h_max and h > h_max:
        nw = max(1, int(round(w * h_max / h)))
        return cv2.resize(img, (nw, h_max), interpolation=cv2.INTER_AREA)
    return img


def estimate_full_res(estimator, img: np.ndarray, max_height: int = 1080,
                      size: tuple[int, int] | None = None) -> PoseResult | None:
    """Pose of a full-resolution frame, run on a copy at most *max_height* tall;
    the landmarks are mapped back to *size* ``(W, H)`` pixels (default: *img*'s)."""
    small = resize_to_height(img, max_height)
    sh, sw = small.shape[:2]
    res = estimator.estimate(small)
    if res is None:
        return None
    W, H = size if size is not None else (img.shape[1], img.shape[0])
    pts = np.asarray(res.points, np.float64) * np.array([W / sw, H / sh])
    return PoseResult(points=pts, visibility=res.visibility, world=getattr(res, "world", None))
