"""
Headless retarget tests.

The analysis-by-synthesis loop is exercised without MediaPipe (which can't load
in CI) by monkeypatching ``_run_mediapipe`` - the rendering hook and the
fixed-point math are what we're verifying, not the face detector itself.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from .. import retarget
from ..retarget import RetargetConfig, _retarget_core, retarget_stream
from ..data import Stream
from ..arkit_names import BLENDSHAPE_NAMES

_GLB = os.path.join(
    os.path.dirname(__file__), "..", "..", "godot", "data", "head.glb"
)
_HAS_GLB = os.path.isfile(_GLB)


# ---------------------------------------------------------------------------
# _retarget_core: pure iteration, no renderer/mesh needed
# ---------------------------------------------------------------------------

def _dummy_rgb(_w):
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_core_converges_when_detector_matches(monkeypatch):
    """If the detector reports the target, the frame converges in one step."""
    target = [0.0] * 52
    target[25] = 0.5
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: list(target))
    cfg = RetargetConfig(max_iter=8, tolerance=0.02, gain=0.6)
    w, iters, converged = _retarget_core(target, _dummy_rgb, object(), cfg, list(range(52)))
    assert iters == 1
    assert converged
    assert w == pytest.approx(target)


def test_core_adjusts_toward_target(monkeypatch):
    """A detector that under-reports drives the masked weight upward."""
    target = [0.0] * 52
    target[25] = 0.8
    # Detector always sees 0 -> positive error -> weight climbs.
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: [0.0] * 52)
    cfg = RetargetConfig(max_iter=5, tolerance=0.001, gain=0.6)
    w, iters, converged = _retarget_core(target, _dummy_rgb, object(), cfg, [25])
    assert iters == 5            # never converges (detector ignores our renders)
    assert not converged
    assert w[25] > target[25]    # pushed up toward a brighter render


def test_core_mask_limits_channels(monkeypatch):
    target = [0.3] * 52
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: [0.0] * 52)
    cfg = RetargetConfig(max_iter=3, tolerance=0.001, gain=0.5)
    w, _iters, _conv = _retarget_core(target, _dummy_rgb, object(), cfg, [10])
    assert w[10] != target[10]            # masked channel moved
    for ch in range(52):
        if ch != 10:
            assert w[ch] == pytest.approx(target[ch])  # others untouched


def test_core_no_face_breaks_immediately(monkeypatch):
    target = [0.4] * 52
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: None)
    cfg = RetargetConfig()
    w, iters, converged = _retarget_core(target, _dummy_rgb, object(), cfg, list(range(52)))
    assert iters == 0
    assert not converged
    assert w == pytest.approx(target)   # unchanged


# ---------------------------------------------------------------------------
# retarget_stream: end-to-end over a Stream (needs the mesh for rendering)
# ---------------------------------------------------------------------------

requires_glb = pytest.mark.skipif(not _HAS_GLB, reason="head.glb not available")


@pytest.fixture(scope="module")
def mesh():
    from .. import head_mesh
    return head_mesh.load_head_mesh(_GLB)


def _take(n=3):
    bs = [0.0] * 52
    bs[25] = 0.4
    return Stream.from_frames([
        {"audio_position_ms": i * 100.0, "blendshapes": list(bs)} for i in range(n)
    ])


@requires_glb
def test_stream_preserves_count_and_positions(monkeypatch, mesh):
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: [0.0] * 52)
    take = _take(3)
    cfg = RetargetConfig(max_iter=1, scope="all")
    result, detected = retarget_stream(
        take, mesh, cfg, landmarker=object(), width=48, height=48
    )
    assert isinstance(result, Stream)
    assert len(result.frames) == 3
    assert result.positions == pytest.approx([0.0, 100.0, 200.0])
    assert detected is True


@requires_glb
def test_stream_no_detection_returns_copies(monkeypatch, mesh):
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: None)
    take = _take(2)
    cfg = RetargetConfig(max_iter=2, scope="all")
    result, detected = retarget_stream(
        take, mesh, cfg, landmarker=object(), width=48, height=48
    )
    assert detected is False
    for src, out in zip(take.frames, result.frames):
        assert out["blendshapes"] == pytest.approx(src["blendshapes"])


@requires_glb
def test_stream_reports_progress(monkeypatch, mesh):
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: [0.0] * 52)
    seen = []
    retarget_stream(
        _take(4), mesh, RetargetConfig(max_iter=1, scope="all"),
        landmarker=object(), width=48, height=48,
        progress_cb=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


@requires_glb
def test_stream_uses_mouth_scope_by_default(monkeypatch, mesh):
    """Default scope='mouth' must leave non-mouth channels untouched."""
    monkeypatch.setattr(retarget, "_run_mediapipe", lambda lm, rgb: [0.0] * 52)
    take = _take(1)
    # Seed a brow channel that is outside the mouth scope.
    brow = BLENDSHAPE_NAMES.index("browInnerUp")
    take.frames[0]["blendshapes"][brow] = 0.5
    result, _ = retarget_stream(
        take, mesh, RetargetConfig(max_iter=3), landmarker=object(),
        width=48, height=48,
    )
    assert result.frames[0]["blendshapes"][brow] == pytest.approx(0.5)
