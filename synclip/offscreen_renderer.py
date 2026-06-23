"""
Headless software rasterizer for the morph-target head.

The retarget pass (analysis-by-synthesis) needs to render the head from a set of
blendshape weights and hand the image to MediaPipe - but it must NOT depend on a
Qt OpenGL widget or any live GL context. This module draws the same head the GPU
``MeshRenderer`` draws, in pure NumPy, so retargeting runs fully headless and is
unit-testable.

Camera/projection match the live preview (ui/mesh_renderer.py) so the head is
framed the same way:
  * camera: neutral head pose, default orbit (yaw/pitch/zoom = 0), the head sat
    at z = -2.6 with a 180 deg yaw so it faces the camera;
  * projection: 45 deg vertical FOV perspective.

Shading, by contrast, is tuned for MediaPipe rather than for matching the GPU
preview: a two-light portrait rig (key + fill) over low ambient gives the face
a clear brightness gradient, which is what lets MediaPipe recover the landmarks.
Per-vertex Lambertian diffuse is computed from BASE-mesh normals (like the GPU
path, which uploads normals once and never re-derives them per morph), so the
shading is independent of the morph weights and precomputed a single time.

It is deliberately simple: a z-buffered barycentric triangle rasterizer. Each
triangle is filled with a vectorised bounding-box pass, so a typical head mesh
renders in tens of milliseconds - fine for an offline retarget.
"""

from __future__ import annotations

import numpy as np

def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# Two-light portrait rig (directions in camera space, +z toward the viewer) plus
# a low ambient floor.  A strong key from the upper-right models the face while a
# weak left fill stops the shadow side going black; keeping ambient low preserves
# the brightness gradient MediaPipe relies on to recover landmarks.
_LIGHTS = [
    (_norm(np.array([ 0.4,  0.6,  1.0], dtype=np.float32)), 0.90),  # key: upper-right, strong
    (_norm(np.array([-0.7, -0.1,  0.5], dtype=np.float32)), 0.25),  # fill: weak left
]
_SKIN    = np.array([0.91, 0.75, 0.62], dtype=np.float32)  # warm skin tone
_BG      = np.array([0.12, 0.12, 0.14], dtype=np.float32)  # dark, for face/background contrast
_AMBIENT = 0.10                                            # low floor so shadows stay dark

# Camera placement - the defaults MeshRenderer uses for a neutral, un-orbited
# view (see MeshRenderer._mvp with yaw=pitch=zoom=pan=0 and head pose None).
_BASE_Z = -2.6


# ---------------------------------------------------------------------------
# Pure matrix helpers (row-major)
# ---------------------------------------------------------------------------

def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def _rotate_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def _vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals (area-weighted face normals)."""
    normals = np.zeros_like(verts)
    if tris.size == 0:
        normals[:, 2] = 1.0
        return normals
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, tris[:, 0], fn)
    np.add.at(normals, tris[:, 1], fn)
    np.add.at(normals, tris[:, 2], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (normals / lengths).astype(np.float32)


class SoftwareRenderer:
    """Renders a HeadMesh deformed by 52 ARKit weights to an RGB image."""

    def __init__(self, mesh, width: int = 384, height: int = 384) -> None:
        self._mesh = mesh
        self._width = int(width)
        self._height = int(height)

        tris = (mesh.indices.reshape(-1, 3).astype(np.int64)
                if mesh.indices.size else np.zeros((0, 3), np.int64))
        self._tris = tris
        # Base-mesh normals, exactly like the GPU path (which uploads normals
        # once from the base geometry and never re-derives them per morph).
        self._base_normals = _vertex_normals(np.asarray(mesh.base, np.float32), tris)

        # Fixed retarget camera: neutral pose, default orbit.
        aspect = self._width / float(self._height)
        proj = _perspective(45.0, aspect, 0.1, 100.0)
        view = _translate(0.0, 0.0, _BASE_Z) @ _rotate_y(180.0)
        self._mvp = np.ascontiguousarray(proj @ view, dtype=np.float32)
        self._normal_mat = np.ascontiguousarray(view[:3, :3], dtype=np.float32)

        # Per-vertex diffuse intensity is independent of the morph weights
        # (normals come from the base mesh), so precompute it once.
        n_cam = self._base_normals @ self._normal_mat.T
        norms = np.linalg.norm(n_cam, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n_cam = n_cam / norms
        diffuse = np.zeros(n_cam.shape[0], dtype=np.float32)
        for light_dir, strength in _LIGHTS:
            diffuse += strength * np.clip(n_cam @ light_dir, 0.0, None)
        self._intensity = np.clip(_AMBIENT + diffuse, 0.0, 1.0).astype(np.float32)

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def render(self, weights) -> np.ndarray:
        """Render *weights* (52 floats) to an HxWx3 uint8 RGB image."""
        verts = np.asarray(self._mesh.blended(weights), dtype=np.float32)
        ndc, w_clip = self._project(verts)
        return self._rasterize(ndc, w_clip)

    # ------------------------------------------------------------------

    def _project(self, verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        homo = np.concatenate(
            [verts, np.ones((verts.shape[0], 1), np.float32)], axis=1
        )
        clip = homo @ self._mvp.T
        w = clip[:, 3].copy()
        safe = np.where(np.abs(w) < 1e-8, 1e-8, w)
        ndc = clip[:, :3] / safe[:, None]
        return ndc.astype(np.float32), w

    def _rasterize(self, ndc: np.ndarray, w_clip: np.ndarray) -> np.ndarray:
        width, height = self._width, self._height

        # NDC [-1,1] -> screen pixels (flip Y so +y is up in NDC, down in pixels).
        sx = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
        sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)
        sz = ndc[:, 2]

        tris = self._tris
        t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]
        x0a, y0a = sx[t0], sy[t0]
        x1a, y1a = sx[t1], sy[t1]
        x2a, y2a = sx[t2], sy[t2]

        # Signed area in screen space.  A CCW (front-facing) triangle in NDC
        # becomes CW after the Y-flip, so its signed area is NEGATIVE.
        area = (y1a - y2a) * (x0a - x2a) + (x2a - x1a) * (y0a - y2a)
        in_front = (w_clip[t0] > 0) & (w_clip[t1] > 0) & (w_clip[t2] > 0)
        bminx = np.maximum(np.floor(np.minimum(np.minimum(x0a, x1a), x2a)), 0).astype(np.int32)
        bmaxx = np.minimum(np.ceil(np.maximum(np.maximum(x0a, x1a), x2a)), width - 1).astype(np.int32)
        bminy = np.maximum(np.floor(np.minimum(np.minimum(y0a, y1a), y2a)), 0).astype(np.int32)
        bmaxy = np.minimum(np.ceil(np.maximum(np.maximum(y0a, y1a), y2a)), height - 1).astype(np.int32)
        visible = (area < -1e-9) & in_front & (bmaxx >= bminx) & (bmaxy >= bminy)

        vis = np.nonzero(visible)[0]
        img_flat = np.empty((height * width, 3), dtype=np.float32)
        img_flat[:] = _BG

        if len(vis) == 0:
            return (img_flat.reshape(height, width, 3) * 255.0).astype(np.uint8)

        # ---- Vectorized scatter rasterizer --------------------------------
        # Expand every visible triangle's bounding box into a flat candidate
        # list without any Python loop over triangles.

        vminx = bminx[vis]; vmaxx = bmaxx[vis]
        vminy = bminy[vis]; vmaxy = bmaxy[vis]
        bw = (vmaxx - vminx + 1).astype(np.int64)
        bh = (vmaxy - vminy + 1).astype(np.int64)
        counts = bw * bh                       # pixels per triangle bbox

        # Triangle index repeated for every candidate pixel in its bbox.
        tri_ids = np.repeat(vis, counts)       # shape: (total_candidates,)

        # Local (dx, dy) within each bbox via modular arithmetic on a flat index.
        flat_local = np.arange(counts.sum(), dtype=np.int64)
        starts = np.zeros(len(vis) + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        local = flat_local - np.repeat(starts[:-1], counts)
        bw_rep = np.repeat(bw, counts)
        dx = (local % bw_rep).astype(np.int32)
        dy = (local // bw_rep).astype(np.int32)

        px = (np.repeat(vminx, counts) + dx).astype(np.int32)
        py = (np.repeat(vminy, counts) + dy).astype(np.int32)

        # Barycentric coordinates for all candidates in one vectorised pass.
        inv = (1.0 / area[tri_ids]).astype(np.float32)
        px_f = px.astype(np.float32); py_f = py.astype(np.float32)
        x2r = x2a[tri_ids]; y2r = y2a[tri_ids]
        b0 = ((y1a[tri_ids] - y2r) * (px_f - x2r) + (x2a[tri_ids] - x1a[tri_ids]) * (py_f - y2r)) * inv
        b1 = ((y2r - y0a[tri_ids]) * (px_f - x2r) + (x0a[tri_ids] - x2r) * (py_f - y2r)) * inv
        b2 = 1.0 - b0 - b1
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)

        # Discard exterior candidates.
        b0 = b0[inside]; b1 = b1[inside]; b2 = b2[inside]
        px = px[inside]; py = py[inside]
        tri_f = tri_ids[inside]

        if len(tri_f) == 0:
            return (img_flat.reshape(height, width, 3) * 255.0).astype(np.uint8)

        # Interpolate depth and intensity at surviving fragments.
        sz0, sz1, sz2 = sz[t0], sz[t1], sz[t2]
        z = b0 * sz0[tri_f] + b1 * sz1[tri_f] + b2 * sz2[tri_f]

        intensity = self._intensity
        i_0, i_1, i_2 = intensity[t0], intensity[t1], intensity[t2]
        inten = b0 * i_0[tri_f] + b1 * i_1[tri_f] + b2 * i_2[tri_f]

        # Z-buffer: sort by (linear_pixel_idx ASC, z ASC) then take the first
        # (= nearest) fragment per pixel with np.unique.
        linear = (py.astype(np.int64) * width + px).astype(np.int64)
        order = np.lexsort((z, linear))
        linear_s = linear[order]
        _, first = np.unique(linear_s, return_index=True)

        sel = order[first]
        lin_win = linear[sel]
        inten_win = np.clip(inten[sel], 0.0, 1.0)
        img_flat[lin_win] = _SKIN[None, :] * inten_win[:, None]

        return (np.clip(img_flat.reshape(height, width, 3), 0.0, 1.0) * 255.0).astype(np.uint8)
