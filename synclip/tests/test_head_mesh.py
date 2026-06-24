"""
GLB head-mesh loader tests.

Exercises the dependency-light glTF parser against the real head.glb shipped
with the Godot viewer: sparse morph accessors, the ARKit name mapping (reused
from the connection-negotiation rules) and the blend math.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from .. import head_mesh
from ..arkit_names import BLENDSHAPE_NAMES

_GLB = os.path.join(
    os.path.dirname(__file__), "..", "..", "godot", "data", "head.glb"
)
_HAS_GLB = os.path.isfile(_GLB)


# ---------------------------------------------------------------------------
# Name mapping (shared with the IPC negotiation rules)
# ---------------------------------------------------------------------------

def test_view_array_reads_interleaved_stride():
    """The strided (interleaved) buffer-view path must read the same values as a
    tightly-packed one, including for the final element (no trailing padding)."""
    import struct
    # Two VEC3 float32 vertices interleaved with 4 bytes of padding per 12-byte
    # item (stride=16); the last item carries no trailing padding.
    v0 = (1.0, 2.0, 3.0)
    v1 = (4.0, 5.0, 6.0)
    buf = (struct.pack("<3f", *v0) + b"\x00\x00\x00\x00"
           + struct.pack("<3f", *v1))
    gltf = {"bufferViews": [{"buffer": 0, "byteOffset": 0, "byteStride": 16}]}
    arr = head_mesh._view_array(gltf, buf, 0, 0, np.float32, 3, 2)
    assert arr.shape == (2, 3)
    np.testing.assert_allclose(arr[0], v0)
    np.testing.assert_allclose(arr[1], v1)


def test_normalize_folds_l_r_suffix():
    assert head_mesh.normalize_shape_name("browDown_L") == "browDownLeft"
    assert head_mesh.normalize_shape_name("mouthStretch_R") == "mouthStretchRight"
    assert head_mesh.normalize_shape_name("jawOpen") == "jawOpen"


def test_arkit_index_for_matches_canonical():
    assert head_mesh.arkit_index_for("browDown_L") == BLENDSHAPE_NAMES.index("browDownLeft")
    assert head_mesh.arkit_index_for("jawOpen") == BLENDSHAPE_NAMES.index("jawOpen")
    # tongueOut is not in ARKit-52 -> no match.
    assert head_mesh.arkit_index_for("tongueOut") is None


# ---------------------------------------------------------------------------
# Real GLB load
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_GLB, reason="head.glb not present")
def test_loads_real_head_glb():
    m = head_mesh.load_head_mesh(_GLB)
    assert m.vertex_count > 1000
    assert m.indices.size > 0 and m.indices.size % 3 == 0
    assert m.arkit_offsets.shape == (52, m.vertex_count, 3)
    # This rig maps every ARKit channel except those with no morph (tongueOut is
    # the rig's only non-ARKit shape), so we expect a high mapped count.
    assert m.mapped_count >= 50
    assert "jawOpen" in m.mapping


@pytest.mark.skipif(not _HAS_GLB, reason="head.glb not present")
def test_blended_moves_for_jaw_open():
    m = head_mesh.load_head_mesh(_GLB)
    w = np.zeros(52, dtype=np.float32)
    w[BLENDSHAPE_NAMES.index("jawOpen")] = 1.0
    out = m.blended(w)
    assert out.shape == m.base.shape
    # jawOpen must actually deform the mesh.
    assert float(np.abs(out - m.base).max()) > 1e-4


@pytest.mark.skipif(not _HAS_GLB, reason="head.glb not present")
def test_zero_weights_returns_base():
    m = head_mesh.load_head_mesh(_GLB)
    out = m.blended(np.zeros(52, dtype=np.float32))
    assert np.allclose(out, m.base)


@pytest.mark.skipif(not _HAS_GLB, reason="head.glb not present")
def test_base_is_centred_and_unit_scaled():
    m = head_mesh.load_head_mesh(_GLB)
    assert float(np.abs(m.base.mean(axis=0)).max()) < 0.2
    assert float(np.linalg.norm(m.base, axis=1).max()) <= 1.0 + 1e-5


def test_blended_rejects_wrong_length():
    m = head_mesh.HeadMesh(
        base=np.zeros((3, 3), dtype=np.float32),
        indices=np.zeros(0, dtype=np.uint32),
        arkit_offsets=np.zeros((52, 3, 3), dtype=np.float32),
        name="t", source_names=[], mapped_count=0, mapping={}, path="",
    )
    with pytest.raises(ValueError):
        m.blended(np.zeros(10, dtype=np.float32))
