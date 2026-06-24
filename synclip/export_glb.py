"""
Export a take's blendshape animation into an existing GLB mesh file.

Reads the source GLB, strips any existing animations, reduces keyframes
(removing frames whose values lie within *tolerance* of the linear
interpolation between their neighbours), injects a morph-weights animation,
and writes a new GLB file.

The morph target order in the GLB may differ from the BLENDSHAPE_NAMES order
used internally, so we build an index map from targetNames -> our index.
"""

from __future__ import annotations

import json
import struct
import os
from typing import Sequence

import numpy as np

from .arkit_names import BLENDSHAPE_NAMES

_OUR_IDX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN  = 0x004E4942
_COMPONENT_FLOAT = 5126


def _read_glb(path: str) -> tuple[dict, bytes]:
    """Parse GLB -> (json_dict, binary_chunk_bytes). binary may be b''."""
    data = open(path, "rb").read()
    magic, _ver, _total = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"Not a GLB file: {path}")
    offset = 12
    j_dict = {}
    bin_data = b""
    while offset < len(data):
        clen, ctype = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + clen]
        if ctype == _CHUNK_JSON:
            j_dict = json.loads(chunk.rstrip(b"\x20"))
        elif ctype == _CHUNK_BIN:
            bin_data = chunk
        offset += 8 + clen
    return j_dict, bin_data


def _write_glb(j_dict: dict, bin_data: bytes) -> bytes:
    """Serialise (json_dict, binary) -> GLB bytes."""
    json_bytes = json.dumps(j_dict, separators=(",", ":")).encode()
    # JSON chunk must be 4-byte aligned (padded with spaces)
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b"\x20" * pad_j
    # Binary chunk must be 4-byte aligned (padded with zeros)
    pad_b = (4 - len(bin_data) % 4) % 4 if bin_data else 0
    bin_data_padded = bin_data + b"\x00" * pad_b

    chunks = bytearray()
    chunks += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    if bin_data_padded:
        chunks += struct.pack("<II", len(bin_data_padded), _CHUNK_BIN) + bin_data_padded

    header = struct.pack("<III", _GLB_MAGIC, 2, 12 + len(chunks))
    return bytes(header) + bytes(chunks)


def _reduce_keyframes(
    times: list[float],
    weights_rows: list[list[float]],
    tolerance: float = 0.001,
) -> tuple[list[float], list[list[float]]]:
    """Remove keyframes that are redundant under linear interpolation.

    A frame at index i is redundant if, for every channel, its value lies
    within *tolerance* of the linear interpolation between frames i-1 and i+1.
    We run a single forward pass (sufficient for typical motion-capture data).
    """
    if len(times) <= 2:
        return times, weights_rows

    keep = [True] * len(times)
    i = 1
    while i < len(times) - 1:
        if not keep[i - 1]:
            i += 1
            continue
        # Find nearest kept predecessor
        prev = i - 1
        while prev > 0 and not keep[prev]:
            prev -= 1
        t0, t1 = times[prev], times[i + 1]
        if t1 == t0:
            i += 1
            continue
        alpha = (times[i] - t0) / (t1 - t0)
        redundant = True
        w0, w2 = weights_rows[prev], weights_rows[i + 1]
        wi = weights_rows[i]
        for ch in range(len(wi)):
            interp = w0[ch] + alpha * (w2[ch] - w0[ch])
            if abs(wi[ch] - interp) > tolerance:
                redundant = False
                break
        if redundant:
            keep[i] = False
        i += 1

    out_times = [t for t, k in zip(times, keep) if k]
    out_weights = [w for w, k in zip(weights_rows, keep) if k]
    return out_times, out_weights


def export_glb(
    source_glb: str,
    frames: list[dict],
    output_path: str,
    tolerance: float = 0.001,
) -> int:
    """Write *frames* as a morph-weight animation into a copy of *source_glb*.

    Parameters
    ----------
    source_glb:   path to the mesh GLB (must have 52 morph targets)
    frames:       list of frame dicts with keys ``audio_position_ms`` and
                  ``blendshapes`` (52-element list, BLENDSHAPE_NAMES order)
    output_path:  destination .glb path
    tolerance:    max per-channel error for keyframe removal (default 0.001)

    Returns
    -------
    Number of keyframes written after reduction.
    """
    if not frames:
        raise ValueError("export_glb: no frames to export")

    j, bin_data = _read_glb(source_glb)

    # -- find mesh node and morph target names --------------------------------
    # Pick the mesh that actually carries morph targets (the same choice
    # head_mesh.load_head_mesh makes) instead of assuming mesh 0, so a multi-mesh
    # GLB animates the right node.
    meshes = j.get("meshes") or []
    if not meshes:
        raise ValueError("GLB has no meshes")
    mesh_idx, mesh = next(
        ((i, m) for i, m in enumerate(meshes)
         if any(p.get("targets") for p in m.get("primitives", []))),
        (0, meshes[0]),
    )
    target_names: list[str] = mesh.get("extras", {}).get("targetNames", [])
    if not target_names:
        raise ValueError("GLB mesh has no targetNames in extras")

    # Build reorder map: glb_slot -> our_index (or -1 if unknown)
    glb_to_ours = [_OUR_IDX.get(name, -1) for name in target_names]

    # Find the node that references this mesh
    mesh_node_idx = next(
        (i for i, n in enumerate(j.get("nodes", [])) if n.get("mesh") == mesh_idx), 0
    )

    # -- strip existing animations -------------------------------------------
    j.pop("animations", None)

    # -- build time + weight arrays ------------------------------------------
    times_s = [f["audio_position_ms"] / 1000.0 for f in frames]
    n_targets = len(target_names)

    weights_rows: list[list[float]] = []
    for f in frames:
        bs = f.get("blendshapes", [])
        row = []
        for glb_slot, our_idx in enumerate(glb_to_ours):
            if our_idx >= 0 and our_idx < len(bs):
                row.append(float(bs[our_idx]))
            else:
                row.append(0.0)
        weights_rows.append(row)

    # -- reduce keyframes -----------------------------------------------------
    times_s, weights_rows = _reduce_keyframes(times_s, weights_rows, tolerance)
    n_keys = len(times_s)

    # -- append to binary buffer ---------------------------------------------
    existing_views = j.get("bufferViews", [])
    existing_accessors = j.get("accessors", [])
    existing_buffers = j.get("buffers", [{}])

    # The animation accessors are float32; glTF requires each accessor's data to
    # start on a 4-byte boundary. Pad the original binary up to alignment before
    # we append, so every appended offset is aligned regardless of the source.
    if len(bin_data) % 4:
        bin_data = bin_data + b"\x00" * (4 - len(bin_data) % 4)
    orig_bin_len = len(bin_data)
    extra = bytearray()

    def _add_accessor(data_np, type_str, count, minmax=None):
        off = orig_bin_len + len(extra)
        raw = data_np.astype(np.float32).tobytes()
        extra.extend(raw)
        bv_idx = len(existing_views)
        existing_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw)})
        acc_idx = len(existing_accessors)
        entry = {"bufferView": bv_idx, "componentType": _COMPONENT_FLOAT,
                 "count": count, "type": type_str}
        # min/max is only meaningful (and required by glTF) for the sampler
        # input - the keyframe times. The flattened weights array must not carry
        # the time range as its bounds.
        if minmax is not None:
            entry["min"], entry["max"] = minmax
        existing_accessors.append(entry)
        return acc_idx

    times_np = np.array(times_s, dtype=np.float32)
    weights_flat = np.array([v for row in weights_rows for v in row], dtype=np.float32)

    time_acc = _add_accessor(times_np, "SCALAR", n_keys,
                             minmax=([float(min(times_s))], [float(max(times_s))]))
    weights_acc = _add_accessor(weights_flat, "SCALAR", n_keys * n_targets)

    # -- update buffer total length ------------------------------------------
    existing_buffers[0]["byteLength"] = orig_bin_len + len(extra)

    # -- build animation JSON ------------------------------------------------
    anim = {
        "name": "synclip",
        "channels": [{"sampler": 0, "target": {"node": mesh_node_idx, "path": "weights"}}],
        "samplers": [{"input": time_acc, "output": weights_acc, "interpolation": "LINEAR"}],
    }
    j["animations"] = [anim]
    j["bufferViews"] = existing_views
    j["accessors"] = existing_accessors
    j["buffers"] = existing_buffers

    new_glb = _write_glb(j, bin_data + bytes(extra))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(new_glb)

    return n_keys
