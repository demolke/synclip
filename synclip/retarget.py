"""
Analysis-by-synthesis expression retargeting.

After a PROCESS pass finishes, the raw MediaPipe blendshapes are already in the
take frames. This module re-renders each frame through the headless software
rasterizer (offscreen_renderer.SoftwareRenderer - no Qt/GL), runs MediaPipe
IMAGE-mode on the render, compares the read-back blendshapes to the target, and
nudges the stored weights until the rendered face matches what MediaPipe would
measure on the original video.

Algorithm (damped fixed-point iteration per frame):
    w = target.copy()
    for _ in range(max_iter):
        read = mediapipe(render(w))
        err  = target - read
        w    = clamp(w + gain * err, 0, 1)
        if max|err| < tol:
            break
    store w back into the frame
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import shutil
import urllib.request
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from . import ai_blendshapes

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILENAME = "face_landmarker.task"
_MODEL_PATH = os.path.join(_MODEL_DIR, _MODEL_FILENAME)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# Derived from ai_blendshapes so the mouth-scope definition stays in one place.
_MOUTH_INDICES = sorted(ai_blendshapes._scope_indices(ai_blendshapes.SCOPE_MOUTH))


@dataclass
class RetargetConfig:
    """User-editable retargeting parameters."""
    max_iter: int = 8
    tolerance: float = 0.02
    gain: float = 0.6
    scope: str = "mouth"          # "mouth" | "all"

    def to_dict(self) -> dict:
        return {
            "retarget_max_iter": self.max_iter,
            "retarget_tolerance": self.tolerance,
            "retarget_gain": self.gain,
            "retarget_scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetargetConfig":
        c = cls()
        c.max_iter = int(d.get("retarget_max_iter", c.max_iter))
        c.tolerance = float(d.get("retarget_tolerance", c.tolerance))
        c.gain = float(d.get("retarget_gain", c.gain))
        c.scope = str(d.get("retarget_scope", c.scope))
        return c


def _ensure_model() -> str:
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 0:
        return _MODEL_PATH
    tmp = _MODEL_PATH + ".part"
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=30) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, _MODEL_PATH)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return _MODEL_PATH


def _run_mediapipe(landmarker, rgb: np.ndarray) -> list[float] | None:
    """Run MediaPipe FaceLandmarker (IMAGE mode) on an HxWx3 uint8 RGB array.
    Returns 52 blendshape floats or None if no face detected."""
    try:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return None
        return [c.score for c in result.face_blendshapes[0]]
    except Exception as exc:
        # A genuine MediaPipe failure is not the same as "no face"; surface it
        # so a crash isn't silently treated as a frame to leave unchanged.
        print(f"[retarget] MediaPipe detect failed: {exc}")
        return None


def _build_landmarker():
    """Create a MediaPipe FaceLandmarker in IMAGE (single-frame) mode."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        model_path = _ensure_model()
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
        )
        return mp_vision.FaceLandmarker.create_from_options(opts)
    except Exception as exc:
        raise RuntimeError(f"Cannot build FaceLandmarker: {exc}") from exc


def _retarget_core(
    weights: list[float],
    grab_rgb,
    landmarker,
    cfg: RetargetConfig,
    mask: list[int],
) -> tuple[list[float], int, bool]:
    """Damped fixed-point retargeting of one frame.

    *grab_rgb* is a callable ``w -> HxWx3 uint8 RGB | None`` that renders the
    given weights. This is the only rendering hook, so the same iteration drives
    both the headless software renderer and the Qt OpenGL widget.
    """
    w = list(weights)
    converged = False
    iters = 0
    for i in range(cfg.max_iter):
        rgb = grab_rgb(w)
        if rgb is None:
            break
        read = _run_mediapipe(landmarker, rgb)
        if read is None:
            break
        iters = i + 1
        max_err = 0.0
        for ch in mask:
            if ch >= len(w) or ch >= len(read):
                continue
            err = weights[ch] - read[ch]
            w[ch] = max(0.0, min(1.0, w[ch] + cfg.gain * err))
            max_err = max(max_err, abs(err))
        if max_err < cfg.tolerance:
            converged = True
            break
    return w, iters, converged


# ---------------------------------------------------------------------------
# Multiprocessing worker helpers (module-level so they are picklable)
# ---------------------------------------------------------------------------

# Per-worker process state (populated by _mp_init, used by _mp_frame).
_mp_renderer = None
_mp_landmarker = None
_mp_cfg = None
_mp_mask = None


def _mp_init(mesh, width: int, height: int, cfg_dict: dict, mask: list[int]) -> None:
    """Initializer called once per worker process."""
    global _mp_renderer, _mp_landmarker, _mp_cfg, _mp_mask
    from .offscreen_renderer import SoftwareRenderer
    _mp_renderer = SoftwareRenderer(mesh, width=width, height=height)
    try:
        _mp_landmarker = _build_landmarker()
    except Exception:
        _mp_landmarker = None
    _mp_cfg = RetargetConfig.from_dict(cfg_dict)
    _mp_mask = mask


def _mp_frame(frame: dict) -> tuple[dict, bool]:
    """Retarget one frame in a worker process."""
    bs = frame.get("blendshapes")
    if not bs or _mp_landmarker is None:
        return frame, False
    new_w, iters, _ = _retarget_core(
        bs, _mp_renderer.render, _mp_landmarker, _mp_cfg, _mp_mask
    )
    new_frame = dict(frame)
    new_frame["blendshapes"] = new_w
    return new_frame, iters > 0


def retarget_stream(
    take_stream,
    mesh,
    cfg: RetargetConfig,
    landmarker=None,
    width: int = 384,
    height: int = 384,
    progress_cb=None,
):
    """Headless analysis-by-synthesis retarget of a whole Stream.

    Renders every frame through the software rasterizer, nudges its stored
    weights until the render matches the target.

    Parameters
    ----------
    take_stream:  source Stream (mediapipe take frames).
    mesh:         a loaded HeadMesh (head_mesh.load_head_mesh).
    cfg:          RetargetConfig.
    landmarker:   optional pre-built FaceLandmarker; one is created (and closed)
                  here when omitted.
    width/height: render resolution handed to MediaPipe.
    progress_cb:  optional ``(done, total) -> None`` for UI progress.

    Returns
    -------
    (Stream, any_detected) - the retargeted stream and whether MediaPipe ever
    saw a face (when False, the frames are unchanged copies of the input).
    """
    from .data import Stream
    from .offscreen_renderer import SoftwareRenderer

    mask = _MOUTH_INDICES if cfg.scope == "mouth" else list(range(52))
    frames = take_stream.frames
    total = len(frames)
    any_detected = False

    # Use frame-level multiprocessing when the caller did not provide a
    # pre-built landmarker (pool workers build their own) and there is more
    # than one frame to process.  Each worker initialises its own renderer
    # and landmarker once, then processes frames sequentially within the worker.
    n_workers = min(4, os.cpu_count() or 1, max(1, total))
    use_pool = landmarker is None and n_workers > 1 and total > 1

    if use_pool:
        cfg_dict = cfg.to_dict()
        ctx = _mp.get_context("spawn")
        updated = []
        with ctx.Pool(
            processes=n_workers,
            initializer=_mp_init,
            initargs=(mesh, width, height, cfg_dict, mask),
        ) as pool:
            for i, (new_frame, detected) in enumerate(pool.imap(_mp_frame, frames)):
                updated.append(new_frame)
                if detected:
                    any_detected = True
                if progress_cb is not None:
                    progress_cb(i + 1, total)

        return Stream.from_frames(updated), any_detected

    # Sequential fallback: caller supplied a landmarker, or only 1 frame/worker.
    renderer = SoftwareRenderer(mesh, width=width, height=height)
    own_landmarker = landmarker is None
    if own_landmarker:
        landmarker = _build_landmarker()

    updated = []
    try:
        for i, frame in enumerate(frames):
            if progress_cb is not None:
                progress_cb(i + 1, total)
            bs = frame.get("blendshapes")
            if not bs:
                updated.append(frame)
                continue
            new_w, iters, _conv = _retarget_core(
                bs, renderer.render, landmarker, cfg, mask
            )
            if iters > 0:
                any_detected = True
            new_frame = dict(frame)
            new_frame["blendshapes"] = new_w
            updated.append(new_frame)
    finally:
        if own_landmarker:
            try:
                landmarker.close()
            except Exception:
                pass

    return Stream.from_frames(updated), any_detected


