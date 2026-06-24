"""
Minimal, dependency-light glTF/GLB loader for the 3D preview.

We only need three things out of a head model: the base vertex positions, the
triangle indices, and the 52 ARKit morph-target offset arrays (re-indexed into
our canonical BLENDSHAPE_NAMES order). That is a small, well-defined slice of
glTF, so rather than pull in a heavy dependency we parse the container here with
nothing but the stdlib + numpy.

The morph-target *names* in a rig rarely match ARKit's canonical spelling
(``browDown_L`` vs ``browDownLeft``), so we apply the same normalisation the
Godot viewer uses: strip known prefixes, fold ``_L/_R`` to ``Left/Right`` and
match case-insensitively. Unmapped ARKit channels get zero offsets (the mesh
just won't move for those), which is exactly what we want for a preview.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .arkit_names import BLENDSHAPE_NAMES

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

# glTF componentType -> numpy dtype.
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
# glTF accessor type -> component count.
_TYPE_COUNT = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}

# Rig prefixes to strip before matching (mirrors the Godot viewer).
_NAME_PREFIXES = ("blendShape.", "blendShape1.", "shapeKey.", "bs_")

_ARKIT_LOWER = {n.lower(): i for i, n in enumerate(BLENDSHAPE_NAMES)}


def normalize_shape_name(raw: str) -> str:
    """Normalise a rig morph-target name to an ARKit canonical candidate."""
    name = raw
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith("_L"):
        name = name[:-2] + "Left"
    elif name.endswith("_R"):
        name = name[:-2] + "Right"
    return name


def arkit_index_for(raw: str) -> int | None:
    """ARKit-52 index for a rig morph-target name, or None if it doesn't map."""
    return _ARKIT_LOWER.get(normalize_shape_name(raw).lower())


@dataclass
class HeadMesh:
    """A loaded head: base geometry + ARKit-ordered morph offsets."""

    base: np.ndarray            # (N, 3) float32, centred + unit-scaled
    indices: np.ndarray         # (M,) uint32 triangle indices (may be empty)
    arkit_offsets: np.ndarray   # (52, N, 3) float32, reindexed to ARKit order
    name: str
    source_names: list[str]     # the rig's own morph-target names, in file order
    mapped_count: int           # how many ARKit channels found a morph target
    mapping: dict               # arkit canonical name -> rig morph name
    path: str

    @property
    def vertex_count(self) -> int:
        return int(self.base.shape[0])

    def blended(self, weights) -> np.ndarray:
        """base + sum(weight[i] * offset[i]) for the 52 ARKit weights."""
        w = np.asarray(weights, dtype=np.float32)
        if w.shape[0] != 52:
            raise ValueError(f"expected 52 weights, got {w.shape[0]}")
        # (52,1,1) * (52,N,3) -> sum over morphs -> (N,3)
        return self.base + (w[:, None, None] * self.arkit_offsets).sum(axis=0)


def _read_glb(path: str) -> tuple[dict, bytes]:
    """Return (gltf_json, binary_chunk) for a .glb, or (json, b'') for .gltf."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == _GLB_MAGIC:
        _magic, _ver, length = struct.unpack_from("<III", data, 0)
        off = 12
        gltf_json: dict | None = None
        bin_chunk = b""
        while off < length:
            clen, ctype = struct.unpack_from("<II", data, off)
            body = data[off + 8: off + 8 + clen]
            if ctype == _CHUNK_JSON:
                gltf_json = json.loads(body.decode("utf-8"))
            elif ctype == _CHUNK_BIN:
                bin_chunk = body
            off += 8 + clen
        if gltf_json is None:
            raise ValueError("GLB has no JSON chunk")
        return gltf_json, bin_chunk
    # Plain .gltf: JSON with external/base64 buffers is not supported here.
    gltf_json = json.loads(data.decode("utf-8"))
    return gltf_json, b""


def _view_array(gltf: dict, buf: bytes, view_idx: int, byte_offset: int,
                dtype, ncomp: int, count: int) -> np.ndarray:
    """Read *count* x *ncomp* elements of *dtype* from a buffer view."""
    view = gltf["bufferViews"][view_idx]
    base_off = view.get("byteOffset", 0) + byte_offset
    item_bytes = np.dtype(dtype).itemsize * ncomp
    stride = view.get("byteStride") or item_bytes
    if stride == item_bytes:
        flat = np.frombuffer(buf, dtype=dtype, count=count * ncomp, offset=base_off)
        return flat.reshape(count, ncomp)
    # Interleaved (strided) buffer view: take the item_bytes of each stride slot
    # in one vectorised pass. n_bytes stops exactly at the last element so we
    # never read past the view (the final slot may omit trailing stride padding).
    n_bytes = stride * (count - 1) + item_bytes
    region = np.frombuffer(buf, dtype=np.uint8, count=n_bytes, offset=base_off)
    rows = np.lib.stride_tricks.as_strided(
        region, shape=(count, item_bytes), strides=(stride, 1)
    )
    return np.ascontiguousarray(rows).view(dtype).reshape(count, ncomp)


def _accessor_array(gltf: dict, buf: bytes, accessor_idx: int) -> np.ndarray:
    """Read an accessor into a (count, ncomp) array, honouring sparse storage.

    Morph-target accessors are commonly *sparse*: a dense base (often omitted,
    meaning all-zero) plus a short list of (index, value) overrides. glTF allows
    such accessors to have no ``bufferView`` at all.
    """
    acc = gltf["accessors"][accessor_idx]
    dtype = _COMPONENT_DTYPE[acc["componentType"]]
    ncomp = _TYPE_COUNT[acc["type"]]
    count = acc["count"]

    if "bufferView" in acc:
        arr = _view_array(gltf, buf, acc["bufferView"], acc.get("byteOffset", 0),
                          dtype, ncomp, count).astype(dtype)
    else:
        arr = np.zeros((count, ncomp), dtype=dtype)  # sparse base defaults to 0

    sparse = acc.get("sparse")
    if sparse:
        sn = sparse["count"]
        idx_info = sparse["indices"]
        val_info = sparse["values"]
        idx_dtype = _COMPONENT_DTYPE[idx_info["componentType"]]
        sparse_idx = _view_array(
            gltf, buf, idx_info["bufferView"], idx_info.get("byteOffset", 0),
            idx_dtype, 1, sn,
        ).reshape(-1)
        sparse_vals = _view_array(
            gltf, buf, val_info["bufferView"], val_info.get("byteOffset", 0),
            dtype, ncomp, sn,
        )
        arr = arr.copy()
        arr[sparse_idx] = sparse_vals

    return arr.astype(np.float32) if dtype == np.float32 else arr


def load_head_mesh(path: str) -> HeadMesh:
    """Load *path* (.glb) into a HeadMesh with ARKit-ordered morph offsets."""
    gltf, buf = _read_glb(path)
    meshes = gltf.get("meshes") or []
    if not meshes:
        raise ValueError(f"{path}: no meshes")

    # Pick the mesh that has morph targets (fall back to the first).
    mesh = next(
        (m for m in meshes if any(p.get("targets") for p in m["primitives"])),
        meshes[0],
    )
    prim = next(
        (p for p in mesh["primitives"] if p.get("targets")),
        mesh["primitives"][0],
    )

    base = _accessor_array(gltf, buf, prim["attributes"]["POSITION"]).astype(np.float32)
    n = base.shape[0]

    indices = np.empty(0, dtype=np.uint32)
    if "indices" in prim:
        indices = _accessor_array(gltf, buf, prim["indices"]).reshape(-1).astype(np.uint32)

    # Centre + unit-scale so any rig fits the preview camera regardless of its
    # authored units; offsets are deltas so they only need the same scale factor.
    centre = base.mean(axis=0)
    base = base - centre
    radius = float(np.linalg.norm(base, axis=1).max()) or 1.0
    scale = 1.0 / radius
    base *= scale

    source_names = list(mesh.get("extras", {}).get("targetNames") or [])
    targets = prim.get("targets") or []

    arkit_offsets = np.zeros((52, n, 3), dtype=np.float32)
    mapping: dict[str, str] = {}   # arkit canonical name -> rig morph name
    unmapped: list[str] = []
    for ti, target in enumerate(targets):
        raw_name = source_names[ti] if ti < len(source_names) else f"target{ti}"
        ark_i = arkit_index_for(raw_name) if "POSITION" in target else None
        if ark_i is None:
            unmapped.append(raw_name)
            continue
        off = _accessor_array(gltf, buf, target["POSITION"]).astype(np.float32) * scale
        arkit_offsets[ark_i] = off
        mapping[BLENDSHAPE_NAMES[ark_i]] = raw_name
    mapped = len(mapping)

    # Report the negotiated mapping exactly as the IPC clients do on connect, so
    # a rig with mis-named shape keys is obvious from the console.
    print(f"[head_mesh] {path}: '{mesh.get('name')}' "
          f"{mapped}/52 ARKit blendshapes mapped")
    for ark_name in BLENDSHAPE_NAMES:
        if ark_name == "_neutral":
            continue
        rig = mapping.get(ark_name)
        print(f"    {ark_name:<22} <- {rig if rig else '(unmapped)'}")
    if unmapped:
        print(f"    rig morph targets with no ARKit match: {unmapped}")

    return HeadMesh(
        base=base,
        indices=indices,
        arkit_offsets=arkit_offsets,
        name=str(mesh.get("name") or "Head"),
        source_names=source_names,
        mapped_count=mapped,
        mapping=mapping,
        path=path,
    )
