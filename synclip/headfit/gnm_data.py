"""GNM head model data access for head fitting (loaded once, cached).

Frame: metres, +Y up, +Z forward.  Bind pose is linear:
    verts = template + sum_k id[k] * vertex_identity_basis[k]
The identity PCs are sqrt-eigenvalue scaled (prior ~ N(0, 1)).
Expression adds ``sum_k expr[k] * expression_basis[k]`` (383 PCs: lower face,
left/right eye regions, tongue, pupils).  Joints are linear in identity
(``joint_identity_basis``); ``joint_regressor`` (J,V) maps vertices to joints.
"""
from __future__ import annotations

import functools
import pathlib

import numpy as np

NPZ_PATH = (pathlib.Path(__file__).resolve().parent.parent
            / "gnm" / "gnm" / "shape" / "data" / "versions" / "v3_0" / "gnm_head.npz")
LANDMARKS_68_PATH = (pathlib.Path(__file__).resolve().parent.parent
                     / "gnm" / "gnm" / "shape" / "data" / "landmarks" / "head_sparse_68.txt")

N_ID_FULL = 253
N_HEAD = 170
N_EXPR = 383

# Expression PCs fitted by default (tongue_* and pupils_* are excluded).
FIT_EXPRESSION_PREFIXES = ("lower_face_region", "left_eye_region", "right_eye_region")

# Face region groups (union) and exclusions.
FACE_REGION_GROUPS = (
    "hockey_mask", "forehead_region", "left_brow_region", "middle_brow_region",
    "right_brow_region", "left_temple_region", "right_temple_region",
    "left_orbital_region", "right_orbital_region", "left_zygomatic_region",
    "right_zygomatic_region", "nose_region", "left_infraorbital_region",
    "right_infraorbital_region", "left_cheek_region", "right_cheek_region",
    "upper_lip_region", "lower_lip_region", "chin_region",
)
EXCLUDE_GROUPS = ("eyes", "eye_interiors", "eye_sockets", "mouth_sock",
                  "teeth", "gums", "tongue")
# Groups whose 1-ring neighbourhood is additionally removed (eyelid / mouth opening).
MARGIN_GROUPS = ("eyes", "eye_interiors", "eye_sockets", "mouth_sock")


class GNMData:
    def __init__(self, path: pathlib.Path | str = NPZ_PATH):
        d = np.load(path, allow_pickle=True)
        self.path = pathlib.Path(path)
        self.template = d["template_vertex_positions"].astype(np.float64)       # (V,3)
        self.identity_names = [str(n) for n in d["identity_names"]]
        basis = d["vertex_identity_basis"]                                        # (253,V,3)
        self.head_idx = np.array([i for i, n in enumerate(self.identity_names)
                                  if n.startswith("head_")], dtype=np.int64)
        assert len(self.head_idx) == N_HEAD and basis.shape[0] == N_ID_FULL
        self.head_basis = basis[self.head_idx].astype(np.float64)                 # (170,V,3)
        self.joint_identity_basis = d["joint_identity_basis"].astype(np.float64)
        self.template_joints = d["template_joint_positions"].astype(np.float64)
        self.triangles = d["triangles"].astype(np.int64)
        self.quads = d["quads"].astype(np.int64)
        self.group_names = [str(n) for n in d["vertex_group_names"]]
        self.vertex_groups = d["vertex_groups"].astype(np.float32)                # (46,V)
        self.mirror_indices = d["mirror_indices"].astype(np.int64)
        self.joint_regressor = d["joint_regressor"].astype(np.float64)            # (J,V)
        self.expression_names = [str(n) for n in d["expression_names"]]
        self.n_verts = self.template.shape[0]

    @functools.cached_property
    def expression_basis(self) -> np.ndarray:
        """(383,V,3) float32, loaded on first use."""
        return np.load(self.path)["expression_basis"].astype(np.float32)

    def expression_indices(self, prefixes=FIT_EXPRESSION_PREFIXES) -> np.ndarray:
        return np.array([i for i, n in enumerate(self.expression_names) if n.startswith(tuple(prefixes))],
                        dtype=np.int64)

    # ----- groups -----
    def group(self, name: str, thresh: float = 0.5) -> np.ndarray:
        return self.vertex_groups[self.group_names.index(name)] > thresh

    @functools.cached_property
    def vertex_adjacency(self):
        import scipy.sparse as sp
        t = self.triangles
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        n = self.n_verts
        a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
        return ((a + a.T) > 0).astype(np.float32).tocsr()

    def dilate(self, mask: np.ndarray, rings: int = 1) -> np.ndarray:
        m = mask.astype(np.float32)
        for _ in range(rings):
            m = np.maximum(m, (self.vertex_adjacency @ m > 0).astype(np.float32))
        return m > 0

    @functools.lru_cache(maxsize=8)
    def face_mask(self, margin_rings: int = 1) -> np.ndarray:
        """Exterior face skin: region groups minus eyes/mouth interior + margin."""
        face = np.zeros(self.n_verts, bool)
        for g in FACE_REGION_GROUPS:
            face |= self.group(g)
        face &= self.group("skin_exterior")
        excl = np.zeros(self.n_verts, bool)
        for g in EXCLUDE_GROUPS:
            excl |= self.group(g)
        margin = np.zeros(self.n_verts, bool)
        for g in MARGIN_GROUPS:
            margin |= self.group(g)
        # vertices of skin that are not exterior (lid/lip inner rims) also widen the margin
        margin |= self.group("skin") & ~self.group("skin_exterior")
        excl |= self.dilate(margin, margin_rings)
        return face & ~excl

    def face_triangles(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Triangles whose three vertices are all in mask (default face_mask)."""
        m = self.face_mask() if mask is None else mask
        return np.nonzero(m[self.triangles].all(1))[0]

    # ----- shape -----
    def verts(self, id_head: np.ndarray | None = None) -> np.ndarray:
        """Bind-pose vertices (V,3) for head identity coefficients (<=170, zero-padded)."""
        v = self.template.copy()
        if id_head is None:
            return v
        c = np.zeros(N_HEAD)
        c[:len(id_head)] = id_head
        return v + np.tensordot(c, self.head_basis, axes=1)

    def joints(self, id_full253: np.ndarray, template_offset: np.ndarray | None = None) -> np.ndarray:
        """Bind-pose joint positions (J,3); a template offset D moves the template joints
        by ``joint_regressor @ D``."""
        j = self.template_joints + np.tensordot(np.asarray(id_full253, float), self.joint_identity_basis, axes=1)
        if template_offset is not None:
            j = j + self.joint_regressor @ template_offset
        return j

    def to_full(self, id_head: np.ndarray) -> np.ndarray:
        """253-vector with head_* coefficients filled in, eyes/teeth = 0."""
        full = np.zeros(N_ID_FULL)
        c = np.zeros(N_HEAD)
        c[:len(id_head)] = id_head
        full[self.head_idx] = c
        return full

    def landmarks_68(self):
        """GNM sparse 68 landmarks as (tri_idx (68,), bary (68,3)), from vertex triples."""
        rows = np.loadtxt(LANDMARKS_68_PATH)
        vid = rows[:, 0::2].astype(np.int64)
        w = rows[:, 1::2]
        return vid, w / w.sum(1, keepdims=True)


@functools.lru_cache(maxsize=1)
def load(path: str | None = None) -> GNMData:
    return GNMData(path or NPZ_PATH)


def face_mask() -> np.ndarray:
    return load().face_mask()


def verts(id_head: np.ndarray | None = None) -> np.ndarray:
    return load().verts(id_head)


def to_full(id_head: np.ndarray) -> np.ndarray:
    return load().to_full(id_head)
