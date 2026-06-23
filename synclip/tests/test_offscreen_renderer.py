"""
Headless software rasterizer tests.

Renders the real head.glb without any Qt/GL context and checks that the image
is well-formed, that morph weights actually deform the silhouette, and that the
output is deterministic.

The final group validates the whole point of the renderer: that MediaPipe can
actually find a face in what we draw.  Those tests need MediaPipe's native
libraries at runtime, so they skip cleanly where it can't load (e.g. CI without
the GL shared objects).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from .. import head_mesh
from ..arkit_names import BLENDSHAPE_NAMES
from ..offscreen_renderer import SoftwareRenderer, _BG

_GLB = os.path.join(os.path.dirname(__file__), "..", "..", "godot", "data", "head.glb")
_HAS_GLB = os.path.isfile(_GLB)
_HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None

# Background red channel as it lands in the uint8 image - derived from the real
# constant so the tests don't drift when the background colour is tuned.
_BG_RED = int(_BG[0] * 255)

pytestmark = pytest.mark.skipif(not _HAS_GLB, reason="head.glb not available")


@pytest.fixture(scope="module")
def mesh():
    return head_mesh.load_head_mesh(_GLB)


def _neutral():
    return [0.0] * 52


def _expression(**names) -> list[float]:
    """Build a weight vector from {blendshape_name: weight} pairs."""
    w = [0.0] * 52
    for name, value in names.items():
        w[BLENDSHAPE_NAMES.index(name)] = value
    return w


# ---------------------------------------------------------------------------
# Image well-formedness
# ---------------------------------------------------------------------------

def test_render_shape_and_dtype(mesh):
    img = SoftwareRenderer(mesh, 96, 96).render(_neutral())
    assert img.shape == (96, 96, 3)
    assert img.dtype == np.uint8


def test_render_draws_a_face(mesh):
    """The neutral render must put a non-background blob on screen."""
    img = SoftwareRenderer(mesh, 128, 128).render(_neutral())
    drawn = int((img[:, :, 0] != _BG_RED).sum())
    assert drawn > 500, f"expected a rendered face, got {drawn} non-bg pixels"


def test_render_is_deterministic(mesh):
    r = SoftwareRenderer(mesh, 80, 80)
    assert np.array_equal(r.render(_neutral()), r.render(_neutral()))


def test_morph_changes_image(mesh):
    """Opening the jaw must change the rendered pixels."""
    r = SoftwareRenderer(mesh, 128, 128)
    neutral = r.render(_neutral())
    opened = r.render(_expression(jawOpen=1.0))
    assert not np.array_equal(neutral, opened), "jawOpen did not alter the render"


def test_size_property(mesh):
    assert SoftwareRenderer(mesh, 64, 48).size == (64, 48)


def test_empty_mesh_renders_background():
    """A mesh with no triangles renders pure background (no crash)."""
    class _Empty:
        base = np.zeros((3, 3), dtype=np.float32)
        indices = np.zeros(0, dtype=np.uint32)

        def blended(self, weights):
            return self.base

    img = SoftwareRenderer(_Empty(), 32, 32).render(_neutral())
    assert img.shape == (32, 32, 3)
    assert int((img[:, :, 0] != _BG_RED).sum()) == 0


# ---------------------------------------------------------------------------
# Shading: the render must have a real light gradient, not a flat blob.
# ---------------------------------------------------------------------------

def test_render_has_shading_gradient(mesh):
    """Lit face pixels must span a wide brightness range (depth, not a flat fill)."""
    img = SoftwareRenderer(mesh, 256, 256).render(_neutral())
    face = img[img[:, :, 0] != _BG_RED]
    assert face.size, "nothing was drawn"
    spread = int(face.max()) - int(face.min())
    assert spread > 80, f"shading too flat: brightness spread only {spread}"


# ---------------------------------------------------------------------------
# MediaPipe: the renderer exists so MediaPipe can re-read the face. Validate it.
#
# MediaPipe's native libraries are run in a *subprocess* (_mp_detect_worker), not
# imported here: loading them into the pytest interpreter inflates the process
# image - which slows the fork()-heavy IPC tests elsewhere in the suite - and has
# been seen to deadlock on shutdown.  The worker renders the expression set, runs
# detection, and reports per-expression results as JSON, so this stays a fast,
# self-contained integration check.
# ---------------------------------------------------------------------------

requires_mediapipe = pytest.mark.skipif(
    not _HAS_MEDIAPIPE, reason="mediapipe not installed"
)


def _run_detection_worker() -> dict[str, bool]:
    """Run the detection worker in a clean subprocess; skip if MP can't load."""
    import json
    import subprocess
    import sys

    tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proc = subprocess.run(
        [sys.executable, "-m", "synclip.tests._mp_detect_worker"],
        cwd=tools_dir, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 3:
        pytest.skip(f"MediaPipe runtime unavailable: {proc.stdout.strip()}")
    assert proc.returncode == 0, f"detection worker failed:\n{proc.stderr[-800:]}"
    json_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert json_lines, f"worker produced no JSON:\n{proc.stdout[-800:]}"
    return json.loads(json_lines[-1])


@requires_mediapipe
def test_mediapipe_detects_face_across_expressions():
    """MediaPipe must locate a face across the expression range we retarget,
    including a low-resolution render - validated out-of-process."""
    results = _run_detection_worker()
    assert results, "worker returned no expressions"
    for name, detected in results.items():
        assert detected, f"MediaPipe found no face in the '{name}' render"
