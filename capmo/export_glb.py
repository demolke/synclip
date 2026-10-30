"""
Export a capmo character animation (+ the calibrated camera rig) to glTF/GLB.

The retarget phase (:mod:`capmo.retarget`) produces a :class:`Skeleton` plus a
track of :class:`CharacterPose` (root translation + per-bone *local* rotations).
This module bakes that into a self-contained ``.glb`` that any glTF 2.0 viewer
can open, so the capture can be eyeballed against the rig:

* a **node-per-bone skeleton** parented to match ``skel.parent`` (root = pelvis
  at ``root_pos``), animated by TRS channels that reproduce
  :func:`capmo.retarget.forward_kinematics` exactly -- a root ``translation``
  channel and a per-bone ``rotation`` channel carrying each bone's rotation
  *local to its node parent* (the same ``local_rot`` FK consumes);
* optionally a thin box of bone geometry per bone so the figure is visible;
* one **camera node per calibrated camera** placed at the camera's world centre
  ``C = -R.T @ t`` with a glTF ``perspective`` camera attached, plus an optional
  axis-tripod / frustum gizmo so the rig is visible even in viewers that don't
  draw camera icons.

Coordinate conventions handled here
-----------------------------------
* quaternions are stored ``(w, x, y, z)`` internally (see
  :mod:`capmo.retarget`); glTF wants ``(x, y, z, w)`` -- we reorder on write;
* OpenCV cameras look down **+Z** with **+Y down**; a glTF camera looks down
  **-Z** with **+Y up**. So a camera node's rotation is the camera-to-world
  rotation ``R_c2w = R.T`` composed with a 180 deg rotation about the camera's
  own X axis (which flips both Z and Y) -- ``R_node = R_c2w @ Rx(180)``.

Everything lives in the one calibration/world frame the skeleton is
reconstructed in, so cameras and the animated figure are spatially consistent.

Pure numpy + stdlib; the GLB writing mirrors ``synclip.export_glb._write_glb``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Sequence

import numpy as np

from .retarget import (
    JOINT_INDEX,
    CharacterPose,
    Skeleton,
    quat_from_matrix,
)

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_COMPONENT_FLOAT = 5126
_COMPONENT_USHORT = 5123


# --------------------------------------------------------------------------- #
# GLB container (mirrors synclip/export_glb.py)
# --------------------------------------------------------------------------- #

def _write_glb(j_dict: dict, bin_data: bytes) -> bytes:
    """Serialise (json_dict, binary) -> GLB bytes (12-byte header, 2 chunks)."""
    json_bytes = json.dumps(j_dict, separators=(",", ":")).encode()
    pad_j = (4 - len(json_bytes) % 4) % 4
    json_bytes += b"\x20" * pad_j
    pad_b = (4 - len(bin_data) % 4) % 4 if bin_data else 0
    bin_padded = bin_data + b"\x00" * pad_b

    chunks = bytearray()
    chunks += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    if bin_padded:
        chunks += struct.pack("<II", len(bin_padded), _CHUNK_BIN) + bin_padded

    header = struct.pack("<III", _GLB_MAGIC, 2, 12 + len(chunks))
    return bytes(header) + bytes(chunks)


def _read_glb(path: str) -> tuple[dict, bytes]:
    """Parse GLB -> (json_dict, binary_chunk_bytes). Mirrors synclip's reader."""
    data = open(path, "rb").read()
    magic, _ver, _total = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"Not a GLB file: {path}")
    offset = 12
    j_dict: dict = {}
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


# --------------------------------------------------------------------------- #
# Buffer / accessor builder
# --------------------------------------------------------------------------- #

class _BufferBuilder:
    """Accumulate float/ushort arrays into one binary buffer + accessor table."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []

    def _align(self, n: int = 4) -> None:
        pad = (n - len(self.blob) % n) % n
        if pad:
            self.blob.extend(b"\x00" * pad)

    def add_float(self, data: np.ndarray, type_str: str, with_minmax: bool = False) -> int:
        """Append a float32 accessor; ``data`` is (count,) or (count, ncomp)."""
        arr = np.asarray(data, np.float32)
        count = arr.shape[0] if arr.ndim >= 1 else 1
        self._align(4)
        off = len(self.blob)
        raw = np.ascontiguousarray(arr).tobytes()
        self.blob.extend(raw)
        bv = len(self.buffer_views)
        self.buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw)})
        acc = {"bufferView": bv, "componentType": _COMPONENT_FLOAT,
               "count": int(count), "type": type_str}
        if with_minmax:
            flat = arr.reshape(count, -1)
            acc["min"] = [float(v) for v in flat.min(axis=0)]
            acc["max"] = [float(v) for v in flat.max(axis=0)]
        idx = len(self.accessors)
        self.accessors.append(acc)
        return idx

    def add_ushort_indices(self, data: np.ndarray) -> int:
        """Append a uint16 SCALAR accessor (mesh element indices)."""
        arr = np.asarray(data, np.uint16).ravel()
        self._align(4)
        off = len(self.blob)
        raw = arr.tobytes()
        self.blob.extend(raw)
        bv = len(self.buffer_views)
        self.buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw)})
        idx = len(self.accessors)
        self.accessors.append({"bufferView": bv, "componentType": _COMPONENT_USHORT,
                               "count": int(arr.size), "type": "SCALAR"})
        return idx


# --------------------------------------------------------------------------- #
# Quaternion reorder helpers
# --------------------------------------------------------------------------- #

def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (x, y, z, w) for glTF."""
    q = np.asarray(q, np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


# 180 deg about local X: flips Y and Z (OpenCV +Z/-Y look -> glTF -Z/+Y look).
_RX_180 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


# --------------------------------------------------------------------------- #
# Bone-box geometry (a thin rectangular prism from head to tail along +Y local)
# --------------------------------------------------------------------------- #

def _unit_bone_box(length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """A thin box from y=0 to y=length, ``width`` thick on x/z. (verts, tris)."""
    h = width * 0.5
    v = np.array([
        [-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h],         # base
        [-h, length, -h], [h, length, -h], [h, length, h], [-h, length, h],  # top
    ], np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3],          # base
        [4, 6, 5], [4, 7, 6],          # top
        [0, 4, 5], [0, 5, 1],          # sides
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], np.uint16)
    return v, f


def _axis_tripod(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small camera gizmo: three short bars along local +X/+Y/+Z (verts, tris).

    Returns (verts, tris, colors-unused-placeholder). The bars sit at the node
    origin so, attached to a camera node, they show the camera's orientation.
    """
    t = scale * 0.06          # bar half-thickness
    L = scale                 # bar length
    verts = []
    tris = []

    def bar(axis: int):
        # a thin box from origin extending +axis by L
        base = len(verts)
        dirs = [np.zeros(3) for _ in range(8)]
        # build a box L long on `axis`, t thick on the other two
        for i, (sa, sb, sc) in enumerate([
            (0, -t, -t), (0, t, -t), (0, t, t), (0, -t, t),
            (L, -t, -t), (L, t, -t), (L, t, t), (L, -t, t),
        ]):
            p = np.zeros(3)
            p[axis] = sa
            others = [k for k in range(3) if k != axis]
            p[others[0]] = sb
            p[others[1]] = sc
            verts.append(p.astype(np.float32))
        box_f = [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ]
        for f in box_f:
            tris.append([base + f[0], base + f[1], base + f[2]])

    bar(0)
    bar(1)
    bar(2)
    return (np.array(verts, np.float32), np.array(tris, np.uint16), np.zeros(0))


# --------------------------------------------------------------------------- #
# Camera intrinsics -> glTF perspective
# --------------------------------------------------------------------------- #

def _perspective_from_K(K: np.ndarray, image_size: Sequence[int],
                        znear: float = 0.05, zfar: float = 1000.0) -> dict:
    """Build a glTF ``perspective`` camera from intrinsics K + image (w, h)."""
    K = np.asarray(K, np.float64)
    w, h = float(image_size[0]), float(image_size[1])
    fy = float(K[1, 1]) if K[1, 1] else h
    yfov = 2.0 * float(np.arctan2(h * 0.5, fy))
    aspect = (w / h) if h else 1.0
    return {
        "type": "perspective",
        "perspective": {
            "aspectRatio": aspect,
            "yfov": yfov,
            "znear": znear,
            "zfar": zfar,
        },
    }


# --------------------------------------------------------------------------- #
# Main exporter
# --------------------------------------------------------------------------- #

def export_animation_glb(
    path: str,
    skeleton: Skeleton,
    poses: list[CharacterPose],
    rig=None,
    *,
    include_camera_geometry: bool = True,
    include_bone_geometry: bool = True,
    fps_hint: float | None = None,
) -> dict:
    """Write *skeleton* + *poses* (+ optional camera *rig*) to a GLB at *path*.

    Parameters
    ----------
    path:                    output ``.glb`` path.
    skeleton:                a :class:`capmo.retarget.Skeleton`.
    poses:                   a track of :class:`capmo.retarget.CharacterPose`.
    rig:                     optional :class:`capmo.calibration.RigCalibration`;
                             cameras are omitted when ``None``.
    include_camera_geometry: add a small axis tripod gizmo per camera node.
    include_bone_geometry:   add a thin box per bone so the figure is visible.
    fps_hint:                unused for the sampler (times come from ``t_ms``);
                             accepted for API symmetry / future use.

    Returns the glTF JSON dict that was written (handy for tests).
    """
    if not poses:
        raise ValueError("export_animation_glb: empty pose track")

    skel = skeleton
    B = skel.num_bones
    bb = _BufferBuilder()

    nodes: list[dict] = []
    scene_roots: list[int] = []
    cameras: list[dict] = []
    meshes: list[dict] = []

    # ---- bind-pose node transforms ----------------------------------------- #
    # The skeleton is a chain of bone nodes. Node 0 is the pelvis *root* node
    # (carries the world root translation + root_rot via the animation). Each
    # bone is a node whose parent node is its FK parent (root for parent<0).
    # A bone node's rest TRANSLATION places its head at the parent's tail in the
    # bind pose, expressed in the parent's local frame; its rotation channel
    # carries that bone's `local_rot`, reproducing forward_kinematics exactly.
    #
    # We model the *node* local rotation so that, in the bind pose, the bone box
    # (authored along +Y) points along skel.rest_dir[b]. Concretely the node's
    # rest rotation R0_b satisfies R0_b @ (+Y) == rest_dir[b]; the animation's
    # rotation channel then multiplies in the captured local swing.
    #
    # To match FK precisely we instead store, per node, the *global* convention
    # baked through the parent chain. Simplest exact match: keep node rotations
    # equal to the pose `local_rot` (relative to parent) and place geometry so
    # that identity local_rot reproduces the bind pose.

    pelvis_bind = skel.bind_joints[JOINT_INDEX["pelvis"]]

    # Root (pelvis) node: index 0.
    root_node_idx = 0
    nodes.append({
        "name": "pelvis_root",
        "translation": [float(v) for v in pelvis_bind],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "children": [],
    })
    scene_roots.append(root_node_idx)

    # Map bone index -> node index.
    bone_node: list[int] = [-1] * B

    # Bind-pose head world positions per bone, to derive rest translations.
    bind_head = np.zeros((B, 3))
    for b in range(B):
        bind_head[b] = skel.bind_joints[skel.head_joint[b]]

    # Rest node rotation for a bone: identity. We instead encode geometry so the
    # box points along rest_dir in the bone's own frame, and the rotation channel
    # carries local_rot directly. The node's rest translation is the bone head
    # offset from its parent, expressed in the parent's *bind* local frame
    # (parents have identity bind rotation in our T-pose, so it's just the world
    # delta). This reproduces FK because retarget builds local_rot exactly as the
    # rotation relative to the parent's global rotation.

    for b in range(B):
        par = skel.parent[b]
        if par < 0:
            parent_node = root_node_idx
            # head offset from pelvis, in the root's *local* frame (matches
            # forward_kinematics' root_offset, which is exactly this).
            rest_t = skel.root_offset[b]
        else:
            parent_node = bone_node[par]
            # The parent node's origin sits at the parent bone's HEAD; this
            # child's head is the parent's TAIL. In the parent's local (bind)
            # frame that displacement is the parent's bind bone vector, so the
            # child node origin lands on the parent tail under FK rotations.
            rest_t = (skel.bind_joints[skel.tail_joint[par]]
                      - skel.bind_joints[skel.head_joint[par]])
        node = {
            "name": skel.bone_names[b],
            "translation": [float(v) for v in rest_t],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "children": [],
        }
        idx = len(nodes)
        nodes.append(node)
        bone_node[b] = idx
        nodes[parent_node]["children"].append(idx)

    # ---- bone box geometry -------------------------------------------------- #
    if include_bone_geometry:
        for b in range(B):
            length = float(skel.rest_len[b])
            verts, tris = _unit_bone_box(length, max(length * 0.06, 1e-3))
            # The box is authored along +Y; orient it onto rest_dir via a child
            # node so the bone's own rotation channel stays clean.
            rest_dir = np.asarray(skel.rest_dir[b], np.float64)
            q_box = quat_from_matrix(_basis_from_y(rest_dir))
            pos_acc = bb.add_float(verts, "VEC3", with_minmax=True)
            idx_acc = bb.add_ushort_indices(tris)
            mesh_idx = len(meshes)
            meshes.append({
                "name": f"{skel.bone_names[b]}_box",
                "primitives": [{
                    "attributes": {"POSITION": pos_acc},
                    "indices": idx_acc,
                    "mode": 4,
                }],
            })
            geom_node = {
                "name": f"{skel.bone_names[b]}_geom",
                "rotation": [float(v) for v in _wxyz_to_xyzw(q_box)],
                "mesh": mesh_idx,
            }
            gidx = len(nodes)
            nodes.append(geom_node)
            nodes[bone_node[b]].setdefault("children", []).append(gidx)

    # ---- animation: build samplers/channels -------------------------------- #
    times_s = np.array([float(p.t_ms) / 1000.0 for p in poses], np.float64)
    # glTF requires strictly defined input min/max; ensure monotonic times.
    time_acc = bb.add_float(times_s, "SCALAR", with_minmax=True)
    n_keys = len(poses)

    samplers: list[dict] = []
    channels: list[dict] = []

    # Root translation channel (pelvis node).
    root_pos = np.array([
        p.root_pos if np.all(np.isfinite(p.root_pos)) else pelvis_bind
        for p in poses
    ], np.float64)
    root_t_acc = bb.add_float(root_pos, "VEC3")
    samplers.append({"input": time_acc, "output": root_t_acc, "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1,
                     "target": {"node": root_node_idx, "path": "translation"}})

    # Root rotation channel (pelvis node).
    root_rot = np.array([_wxyz_to_xyzw(p.root_rot) for p in poses], np.float64)
    root_r_acc = bb.add_float(root_rot, "VEC4")
    samplers.append({"input": time_acc, "output": root_r_acc, "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1,
                     "target": {"node": root_node_idx, "path": "rotation"}})

    # Per-bone rotation channels (local_rot, reordered to xyzw).
    for b in range(B):
        rots = np.array([_wxyz_to_xyzw(p.local_rot[b]) for p in poses], np.float64)
        acc = bb.add_float(rots, "VEC4")
        samplers.append({"input": time_acc, "output": acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": bone_node[b], "path": "rotation"}})

    animations = [{"name": "capmo", "samplers": samplers, "channels": channels}]

    # ---- cameras ------------------------------------------------------------ #
    if rig is not None:
        for cam_id, cam in rig.cameras.items():
            R = np.asarray(cam.rotation_matrix(), np.float64)
            t = np.asarray(cam.tvec, np.float64).reshape(3)
            C = -R.T @ t                       # camera centre in world
            R_c2w = R.T                         # camera-to-world rotation
            R_node = R_c2w @ _RX_180            # OpenCV +Z look -> glTF -Z look
            q_node = quat_from_matrix(R_node)

            cam_obj_idx = len(cameras)
            cameras.append(_perspective_from_K(cam.K, cam.image_size))

            cam_node = {
                "name": str(cam_id),
                "translation": [float(v) for v in C],
                "rotation": [float(v) for v in _wxyz_to_xyzw(q_node)],
                "camera": cam_obj_idx,
                "children": [],
            }
            cam_node_idx = len(nodes)
            nodes.append(cam_node)
            scene_roots.append(cam_node_idx)

            if include_camera_geometry:
                gscale = 0.15
                verts, tris, _ = _axis_tripod(gscale)
                pos_acc = bb.add_float(verts, "VEC3", with_minmax=True)
                idx_acc = bb.add_ushort_indices(tris)
                mesh_idx = len(meshes)
                meshes.append({
                    "name": f"{cam_id}_gizmo",
                    "primitives": [{
                        "attributes": {"POSITION": pos_acc},
                        "indices": idx_acc,
                        "mode": 4,
                    }],
                })
                gnode = {"name": f"{cam_id}_gizmo", "mesh": mesh_idx}
                gidx = len(nodes)
                nodes.append(gnode)
                cam_node["children"].append(gidx)

    # ---- assemble glTF ------------------------------------------------------ #
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "capmo.export_glb"},
        "scene": 0,
        "scenes": [{"name": "capmo_capture", "nodes": scene_roots}],
        "nodes": nodes,
        "animations": animations,
        "buffers": [{"byteLength": len(bb.blob)}],
        "bufferViews": bb.buffer_views,
        "accessors": bb.accessors,
    }
    if meshes:
        gltf["meshes"] = meshes
    if cameras:
        gltf["cameras"] = cameras

    # Strip empty "children" lists (cosmetic, keeps JSON valid + tidy).
    for n in nodes:
        if "children" in n and not n["children"]:
            del n["children"]

    glb = _write_glb(gltf, bytes(bb.blob))
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(glb)
    return gltf


def _basis_from_y(y_dir: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns) whose +Y axis maps to unit ``y_dir``."""
    y = np.asarray(y_dir, np.float64)
    n = np.linalg.norm(y)
    if n < 1e-12:
        return np.eye(3)
    y = y / n
    # pick a reference not parallel to y
    ref = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(ref, y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])
