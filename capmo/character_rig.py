"""
Bind the capture's retarget armature to a vendored character rig (glTF).

:mod:`capmo.retarget` produces a pose for our 14-bone humanoid built from a
canonical T-pose. To drive a *real* character mesh we retarget onto **that mesh's
own skeleton** instead: we read the rig's bind pose out of its glTF, express our
17 joints as the world positions of the rig's matching bones, and hand those to
:func:`capmo.retarget.default_skeleton`. The result is our bone topology carrying
the **rig's proportions and rest directions** -- so the per-bone *swing*
rotations land correctly on a character whose limbs are nothing like the captured
person's. (Bone rotations never use the capture's lengths; only the root
*translation* does, which :func:`root_scale_for` rescales -- see ``retarget_track``.)

Shipped rig: Quaternius "Universal Base Characters" (CC0), a UE-mannequin-style
skeleton -- ``pelvis / spine_01..03 / neck_01 / Head``, ``clavicle/upperarm/
lowerarm/hand`` and ``thigh/calf/foot/ball`` per side. :data:`QUATERNIUS_JOINTS`
maps our joints onto it. The loader is generic glTF, so another rig only needs a
new joint map.

Pure numpy + stdlib glTF JSON parsing (mirrors ``synclip.head_mesh``).
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

from .retarget import JOINTS, Skeleton, default_skeleton, quat_to_matrix

# glTF componentType -> numpy dtype; accessor type -> component count.
_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                    5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

# Our joint name -> the rig bone whose node origin sits at that joint. Quaternius
# names a wrist ``hand_*``, a forearm ``lowerarm_*``, a shin ``calf_*`` and a toe
# ``ball_*``; our "chest" (where neck + clavicles branch) is its ``spine_03``.
QUATERNIUS_JOINTS: dict[str, str] = {
    "pelvis": "pelvis", "chest": "spine_03", "head": "Head",
    "l_shoulder": "upperarm_l", "l_elbow": "lowerarm_l", "l_wrist": "hand_l",
    "r_shoulder": "upperarm_r", "r_elbow": "lowerarm_r", "r_wrist": "hand_r",
    "l_hip": "thigh_l", "l_knee": "calf_l", "l_ankle": "foot_l", "l_foot": "ball_l",
    "r_hip": "thigh_r", "r_knee": "calf_r", "r_ankle": "foot_r", "r_foot": "ball_r",
}

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A


def _load_gltf_json(path: str) -> dict:
    """Read the JSON document from a ``.gltf`` or ``.glb`` file."""
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if struct.unpack("<I", head)[0] == _GLB_MAGIC:      # binary glTF container
            f.read(12)                                       # magic, version, length
            length = struct.unpack("<I", f.read(4))[0]
            ctype = struct.unpack("<I", f.read(4))[0]
            if ctype != _CHUNK_JSON:
                raise ValueError("first GLB chunk is not JSON")
            return json.loads(f.read(length))
        return json.loads(f.read())


def _node_world_translations(gltf: dict) -> dict[str, np.ndarray]:
    """World-space translation of every named node (full TRS down the tree)."""
    nodes = gltf["nodes"]
    out: dict[str, np.ndarray] = {}

    def local(node: dict) -> np.ndarray:
        if "matrix" in node:                                 # column-major 4x4
            return np.array(node["matrix"], np.float64).reshape(4, 4).T
        M = np.eye(4)
        if "rotation" in node:
            x, y, z, w = node["rotation"]                    # glTF order -> ours
            M[:3, :3] = quat_to_matrix(np.array([w, x, y, z]))
        if "scale" in node:
            M[:3, :3] = M[:3, :3] @ np.diag(node["scale"])
        if "translation" in node:
            M[:3, 3] = node["translation"]
        return M

    def walk(idx: int, parent: np.ndarray) -> None:
        node = nodes[idx]
        world = parent @ local(node)
        if node.get("name"):
            out[node["name"]] = world[:3, 3].copy()
        for child in node.get("children", []):
            walk(child, world)

    roots = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
    for r in roots:
        walk(r, np.eye(4))
    return out


def load_rig_skeleton(gltf_path: str,
                      joint_map: dict[str, str] = QUATERNIUS_JOINTS) -> Skeleton:
    """Build a retarget :class:`Skeleton` from a character rig's bind pose.

    Reads the rig's bind-pose world joint positions and constructs our 14-bone
    armature on the rig's proportions, so :func:`capmo.retarget.retarget_pose`
    drives this rig directly.
    """
    world = _node_world_translations(_load_gltf_json(gltf_path))
    missing = [n for n in joint_map.values() if n not in world]
    if missing:
        raise ValueError(f"rig is missing expected bones: {missing}")
    bind = {our: world[joint_map[our]] for our, _ in JOINTS}
    return default_skeleton(bind)


def _load_buffer(gltf: dict, gltf_path: str) -> bytes:
    """Read the rig's single external ``.bin`` buffer (no base64/GLB here)."""
    uri = gltf["buffers"][0].get("uri")
    if not uri or uri.startswith("data:"):
        raise ValueError("expected a single external .bin buffer")
    with open(os.path.join(os.path.dirname(gltf_path), uri), "rb") as f:
        return f.read()


def _accessor(gltf: dict, buf: bytes, idx: int) -> np.ndarray:
    """Read accessor *idx* into a contiguous (count, ncomp) array.

    Only the dense, non-strided layout the Quaternius mesh uses is handled --
    enough to pull POSITION (VEC3 float) and the triangle indices out.
    """
    acc = gltf["accessors"][idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = _COMPONENT_DTYPE[acc["componentType"]]
    ncomp = _TYPE_COUNT[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    flat = np.frombuffer(buf, dtype=dtype, count=acc["count"] * ncomp, offset=offset)
    return flat.reshape(acc["count"], ncomp)


def load_rig_mesh(gltf_path: str,
                  mesh_name: str = "Sphere.005_Retopology.004"
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Load a rig's body mesh geometry as ``(positions (N,3), tris (M,3))``.

    Returns the bind-pose vertex POSITIONs (already authored Y-up in world
    space, ~1.82 m tall) and triangle vertex indices for *mesh_name* (the body
    primitive; defaults to the Quaternius superhero's). No skinning is applied --
    the A-pose bind mesh is all a detector needs.
    """
    gltf = _load_gltf_json(gltf_path)
    buf = _load_buffer(gltf, gltf_path)
    meshes = gltf.get("meshes") or []
    mesh = next((m for m in meshes if m.get("name") == mesh_name), None)
    if mesh is None:                                   # fall back to the largest
        mesh = max(meshes, key=lambda m: gltf["accessors"][
            m["primitives"][0]["attributes"]["POSITION"]]["count"])
    prim = mesh["primitives"][0]
    verts = _accessor(gltf, buf, prim["attributes"]["POSITION"]).astype(np.float64)
    tris = _accessor(gltf, buf, prim["indices"]).reshape(-1, 3).astype(np.int64)
    return verts, tris


def _torso_length(skel: Skeleton) -> float:
    """Bind-pose pelvis->chest (spine) length -- a robust scale reference."""
    return float(skel.rest_len[skel.bone_names.index("spine")])


def _captured_torso_length(frames) -> float | None:
    """Mean measured pelvis->chest distance over the track (None if never seen)."""
    lengths = []
    pelvis_src, chest_src = (23, 24), (11, 12)      # MediaPipe hip / shoulder pairs
    for f in frames:
        pts, val = f.skeleton.points, f.skeleton.valid
        if all(val[s] for s in pelvis_src) and all(val[s] for s in chest_src):
            p = np.mean([pts[s] for s in pelvis_src], axis=0)
            c = np.mean([pts[s] for s in chest_src], axis=0)
            lengths.append(float(np.linalg.norm(c - p)))
    return float(np.mean(lengths)) if lengths else None


def root_scale_for(target_skel: Skeleton, frames) -> float:
    """Scale that maps the capture's root translation into the rig's proportions.

    The ratio of the rig's torso length to the captured person's mean torso
    length; ``1.0`` when the capture never shows a usable torso.
    """
    captured = _captured_torso_length(frames)
    if not captured or captured < 1e-9:
        return 1.0
    return _torso_length(target_skel) / captured
