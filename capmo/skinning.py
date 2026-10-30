"""
Linear blend skinning for the vendored character rig (glTF).

:mod:`capmo.character_rig` reads the rig's *bind-pose* geometry (the rest A-pose
mesh) and its bind skeleton. That is all a detector needs for the static tests,
but to validate the full capture loop against a *known* articulated pose we have
to actually **pose** the mesh: bend the elbows/knees, render, detect, reconstruct,
and measure the recovered skeleton against the pose we applied.

This module adds standard linear blend skinning (LBS) on top of the glTF skin:

* :func:`load_skin` pulls the per-vertex influences (``JOINTS_0`` / ``WEIGHTS_0``),
  the skin's ``joints`` node list, the ``inverseBindMatrices`` and the node
  hierarchy (parent + local TRS) out of the asset.
* :func:`pose_rig_mesh` deforms the body mesh by a set of extra *local* rotations
  applied at chosen rig bone nodes (identity elsewhere) -- ordinary LBS.
* :func:`posed_joint_positions` returns the world positions of OUR 17 joints under
  the same pose, i.e. the ground-truth posed skeleton to measure against.

Coordinate spaces (verified by the round-trip in ``test_skinning``):
the rig's mesh POSITION accessor and its bind node-global transforms live in the
*same* Y-up world frame (the ``root`` joint carries the -90 deg X Blender->glTF
rotation inside its local TRS, so no extra world transform is layered on top --
exactly the frame :func:`capmo.character_rig.load_rig_skeleton` uses). A posed
joint's animated global transform composes the bind local TRS with the requested
extra local rotation, ``skinMatrix_j = G_j @ inverseBindMatrix_j``, and
``v' = sum_k w_k (skinMatrix_{j_k} @ v_homogeneous)``.

Pure numpy + the stdlib glTF parsing reused from :mod:`capmo.character_rig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import character_rig as CR
from .character_rig import QUATERNIUS_JOINTS, _load_buffer, _load_gltf_json
from .retarget import JOINTS, JOINT_INDEX, quat_to_matrix

# glTF accessor type -> component count, including the MAT4 the skin needs (the
# character_rig accessor reader only handles up to VEC4).
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _accessor(gltf: dict, buf: bytes, idx: int) -> np.ndarray:
    """Read accessor *idx* into a contiguous ``(count, ncomp)`` array (MAT4-aware)."""
    acc = gltf["accessors"][idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = CR._COMPONENT_DTYPE[acc["componentType"]]
    ncomp = _TYPE_COUNT[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    flat = np.frombuffer(buf, dtype=dtype, count=acc["count"] * ncomp, offset=offset)
    return flat.reshape(acc["count"], ncomp)


def _node_local(node: dict) -> np.ndarray:
    """Local TRS of a glTF node as a 4x4 matrix (matches character_rig's walk)."""
    if "matrix" in node:                                      # column-major 4x4
        return np.array(node["matrix"], np.float64).reshape(4, 4).T
    M = np.eye(4)
    if node.get("rotation"):
        x, y, z, w = node["rotation"]                         # glTF order -> ours
        M[:3, :3] = quat_to_matrix(np.array([w, x, y, z]))
    if node.get("scale"):
        M[:3, :3] = M[:3, :3] @ np.diag(node["scale"])
    if node.get("translation"):
        M[:3, 3] = node["translation"]
    return M


@dataclass
class SkinData:
    """Everything LBS needs from one glTF skin + its body mesh primitive."""
    verts: np.ndarray            # (V, 3) bind-pose mesh positions (world frame)
    tris: np.ndarray             # (M, 3) triangle vertex indices
    joint_indices: np.ndarray    # (V, 4) per-vertex skin-joint influences
    weights: np.ndarray          # (V, 4) per-vertex influence weights
    skin_joints: list[int]       # skin-joint index -> glTF node index
    inverse_bind: np.ndarray     # (J, 4, 4) inverse bind matrices, row-major 4x4
    parent: dict[int, int]       # glTF node index -> parent node index
    local: dict[int, np.ndarray] # glTF node index -> local TRS (4x4)
    node_name: dict[str, int]    # node name -> glTF node index


def load_skin(gltf_path: str,
              mesh_name: str = "Sphere.005_Retopology.004") -> SkinData:
    """Load the skin influences, bind matrices and node hierarchy from a rig.

    *mesh_name* is the body primitive (defaults to the Quaternius superhero's
    ``SuperHero_Male`` mesh, ``Sphere.005_Retopology.004``); falls back to the
    largest skinned mesh if the name is absent.
    """
    gltf = _load_gltf_json(gltf_path)
    buf = _load_buffer(gltf, gltf_path)
    nodes = gltf["nodes"]

    meshes = gltf.get("meshes") or []
    mesh = next((m for m in meshes if m.get("name") == mesh_name), None)
    if mesh is None:                                          # largest skinned mesh
        skinned = [m for m in meshes
                   if "JOINTS_0" in m["primitives"][0]["attributes"]]
        mesh = max(skinned or meshes, key=lambda m: gltf["accessors"][
            m["primitives"][0]["attributes"]["POSITION"]]["count"])
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]
    verts = _accessor(gltf, buf, attrs["POSITION"]).astype(np.float64)
    tris = _accessor(gltf, buf, prim["indices"]).reshape(-1, 3).astype(np.int64)
    joint_indices = _accessor(gltf, buf, attrs["JOINTS_0"]).astype(np.int64)
    weights = _accessor(gltf, buf, attrs["WEIGHTS_0"]).astype(np.float64)

    skin = gltf["skins"][0]
    skin_joints = list(skin["joints"])
    ibm = _accessor(gltf, buf, skin["inverseBindMatrices"]).astype(np.float64)
    inverse_bind = np.array([m.reshape(4, 4).T for m in ibm])  # col-major -> 4x4

    parent: dict[int, int] = {}
    local: dict[int, np.ndarray] = {}
    node_name: dict[str, int] = {}
    for i, n in enumerate(nodes):
        local[i] = _node_local(n)
        if n.get("name"):
            node_name[n["name"]] = i
        for c in n.get("children", []):
            parent[c] = i

    return SkinData(verts=verts, tris=tris, joint_indices=joint_indices,
                    weights=weights, skin_joints=skin_joints,
                    inverse_bind=inverse_bind, parent=parent, local=local,
                    node_name=node_name)


def _animated_globals(skin: SkinData,
                      node_rotations: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Animated global transform of every node (cached recursion up the tree).

    *node_rotations* maps a glTF node index to a 3x3 (or 4x4) extra rotation that
    is post-multiplied onto the node's bind local TRS -- i.e. a rotation in the
    bone's own local frame, identity for any node not present.
    """
    cache: dict[int, np.ndarray] = {}

    def g_of(idx: int) -> np.ndarray:
        cached = cache.get(idx)
        if cached is not None:
            return cached
        M = skin.local[idx].copy()
        extra = node_rotations.get(idx)
        if extra is not None:
            R = np.eye(4)
            R[:3, :3] = np.asarray(extra, np.float64)[:3, :3]
            M = M @ R
        par = skin.parent.get(idx)
        out = M if par is None else g_of(par) @ M
        cache[idx] = out
        return out

    # Resolve every node that could matter (all skin joints + their ancestors).
    for j in skin.skin_joints:
        g_of(j)
    return cache


def _resolve_rotations(skin: SkinData,
                       bone_rotations: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Map ``{bone node name: 3x3 rotation}`` to ``{glTF node index: rotation}``."""
    out: dict[int, np.ndarray] = {}
    for name, R in bone_rotations.items():
        if name not in skin.node_name:
            raise ValueError(f"rig has no bone node named {name!r}")
        out[skin.node_name[name]] = np.asarray(R, np.float64)
    return out


def pose_rig_mesh(gltf_path_or_skin,
                  bone_rotations: dict[str, np.ndarray] | None = None,
                  *, mesh_name: str = "Sphere.005_Retopology.004") -> np.ndarray:
    """Linear-blend-skin the body mesh under *bone_rotations* -> ``(V, 3)`` verts.

    *gltf_path_or_skin* is either a path (loaded via :func:`load_skin`) or a
    pre-loaded :class:`SkinData`. *bone_rotations* maps a rig bone *node name*
    (e.g. ``"lowerarm_l"``) to a 3x3 rotation applied in that bone's local frame;
    omitted bones stay at the bind pose. With no rotations this reproduces the
    bind mesh (:func:`capmo.character_rig.load_rig_mesh`) to ~1e-6.
    """
    skin = (gltf_path_or_skin if isinstance(gltf_path_or_skin, SkinData)
            else load_skin(gltf_path_or_skin, mesh_name=mesh_name))
    node_rot = _resolve_rotations(skin, bone_rotations or {})
    globals_ = _animated_globals(skin, node_rot)

    G = np.array([globals_[j] for j in skin.skin_joints])     # (J, 4, 4)
    skin_mat = G @ skin.inverse_bind                          # (J, 4, 4)

    vh = np.concatenate([skin.verts, np.ones((len(skin.verts), 1))], axis=1)
    out = np.zeros((len(skin.verts), 3))
    for k in range(skin.joint_indices.shape[1]):
        w = skin.weights[:, k]
        M = skin_mat[skin.joint_indices[:, k]]                # (V, 4, 4)
        tv = np.einsum("vij,vj->vi", M, vh)[:, :3]
        out += w[:, None] * tv
    return out


def posed_joint_positions(gltf_path_or_skin,
                          bone_rotations: dict[str, np.ndarray] | None = None,
                          joint_map: dict[str, str] = QUATERNIUS_JOINTS,
                          *, mesh_name: str = "Sphere.005_Retopology.004"
                          ) -> np.ndarray:
    """World positions of OUR 17 joints under *bone_rotations* -> ``(17, 3)``.

    The ground-truth posed skeleton: each mapped rig bone node's animated global
    origin, ordered by :data:`capmo.retarget.JOINTS`. Uses the same hierarchy walk
    as :func:`pose_rig_mesh`, so mesh and skeleton stay in one frame.
    """
    skin = (gltf_path_or_skin if isinstance(gltf_path_or_skin, SkinData)
            else load_skin(gltf_path_or_skin, mesh_name=mesh_name))
    node_rot = _resolve_rotations(skin, bone_rotations or {})
    globals_ = _animated_globals(skin, node_rot)

    out = np.full((len(JOINTS), 3), np.nan)
    for name, _ in JOINTS:
        bone = joint_map[name]
        node = skin.node_name.get(bone)
        if node is None:
            raise ValueError(f"rig is missing expected bone {bone!r}")
        out[JOINT_INDEX[name]] = globals_[node][:3, 3]
    return out
