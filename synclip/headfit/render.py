"""Headless ray-cast renderer (no GPU / EGL) built on trimesh ray queries.

Camera convention (same as OpenGL / the GNM frame): the camera looks down its
own -Z axis, +X is image-right and +Y is image-up.  A world point ``p`` is
expressed in camera coordinates as ``R @ p`` where ``R`` is the world->camera
rotation passed to :func:`render`.  For a mesh already in the GNM frame
(+Y up, +Z forward, subject-left at +X) ``R = I`` gives a frontal view that is
not mirrored: the subject's left appears on the image right, as in a photo.

Image rows grow downwards, so pixel ``(u, v)`` (column, row; pixel centres at
``+0.5``) maps to camera ``x = cx + (u - W/2) * px`` and ``y = cy - (v - H/2) * px``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

_SKIN = np.array([0.86, 0.70, 0.60])
_BACKGROUND = np.array([38, 38, 42], dtype=np.uint8)


@dataclass
class Camera:
    """Orthographic camera.

    rotation: (3,3) world->camera rotation.
    center:   (3,) world point the view is centred on.
    extent:   world-space width/height covered by the (square) image.
    size:     image size in pixels (square).
    """

    rotation: np.ndarray
    center: np.ndarray
    extent: float
    size: int = 512

    @property
    def pixel_size(self) -> float:
        return self.extent / self.size

    def rays(self, uv: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray origins/directions through pixel coords ``uv`` (N,2)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        half = self.size / 2.0
        x = (uv[:, 0] - half) * self.pixel_size
        y = -(uv[:, 1] - half) * self.pixel_size
        cam = np.stack([x, y, np.full_like(x, depth)], axis=1)
        R = np.asarray(self.rotation, dtype=np.float64)
        origins = cam @ R + self.center  # R^T applied to row vectors
        direction = -R[2]  # camera -Z in world
        dirs = np.broadcast_to(direction, origins.shape).copy()
        return origins, dirs

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixel coords (N,2) (column, row)."""
        p = (np.asarray(points, dtype=np.float64) - self.center) @ np.asarray(self.rotation).T
        half = self.size / 2.0
        u = p[:, 0] / self.pixel_size + half
        v = -p[:, 1] / self.pixel_size + half
        return np.stack([u, v], axis=1)


@dataclass
class RenderResult:
    image: np.ndarray  # (H,W,3) uint8 RGB
    tri: np.ndarray  # (H,W) int, -1 where the ray missed
    points: np.ndarray  # (H,W,3) world-space hit points (nan where missed)
    camera: Camera

    @property
    def mask(self) -> np.ndarray:
        return self.tri >= 0


@dataclass
class Hits:
    """Result of casting rays through arbitrary pixel coordinates."""

    hit: np.ndarray  # (N,) bool
    points: np.ndarray  # (N,3) world, nan where missed
    tri: np.ndarray  # (N,) int, -1 where missed
    bary: np.ndarray  # (N,3) barycentrics, nan where missed


def frame_camera(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    size: int = 512,
    margin: float = 1.08,
    vertices: np.ndarray | None = None,
) -> Camera:
    """Camera that fits ``vertices`` (default: all mesh vertices) in view."""
    pts = mesh.vertices if vertices is None else vertices
    R = np.asarray(rotation, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64) @ R.T
    lo, hi = p.min(0), p.max(0)
    mid_cam = (lo + hi) / 2.0
    extent = float(max(hi[0] - lo[0], hi[1] - lo[1])) * margin
    center = mid_cam @ R  # back to world
    return Camera(rotation=R, center=center, extent=max(extent, 1e-9), size=size)


def _ray_depth(mesh: trimesh.Trimesh, cam: Camera) -> float:
    # Place ray origins in front of every vertex along the view axis.
    z = (mesh.vertices - cam.center) @ np.asarray(cam.rotation)[2]
    return float(z.max()) + 0.01 * float(np.ptp(mesh.vertices, axis=0).max() + 1e-9) + 1e-6


def cast(mesh: trimesh.Trimesh, cam: Camera, uv: np.ndarray) -> Hits:
    """Cast one ray per pixel coordinate (sub-pixel allowed); first hit only."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = len(uv)
    origins, dirs = cam.rays(uv, _ray_depth(mesh, cam))
    pts = np.full((n, 3), np.nan)
    tri = np.full(n, -1, dtype=np.int64)
    bary = np.full((n, 3), np.nan)
    if n == 0:
        return Hits(np.zeros(0, bool), pts, tri, bary)
    loc, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(idx_ray):
        # Defensive: keep the nearest hit per ray if several were returned.
        d = np.einsum("ij,ij->i", loc - origins[idx_ray], dirs[idx_ray])
        order = np.lexsort((d, idx_ray))
        idx_ray, idx_tri, loc = idx_ray[order], idx_tri[order], loc[order]
        first = np.r_[True, idx_ray[1:] != idx_ray[:-1]]
        idx_ray, idx_tri, loc = idx_ray[first], idx_tri[first], loc[first]
        pts[idx_ray] = loc
        tri[idx_ray] = idx_tri
        b = trimesh.triangles.points_to_barycentric(mesh.triangles[idx_tri], loc)
        bary[idx_ray] = b
    return Hits(tri >= 0, pts, tri, bary)


def render(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray | None = None,
    size: int = 512,
    camera: Camera | None = None,
    light_dir: np.ndarray | None = None,
) -> RenderResult:
    """Orthographic two-sided Lambert render of ``mesh``.

    Either pass ``camera`` or a world->camera ``rotation`` (auto-framed).
    """
    if camera is None:
        camera = frame_camera(mesh, np.eye(3) if rotation is None else rotation, size)
    S = camera.size
    vv, uu = np.mgrid[0:S, 0:S]
    uv = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], axis=1)
    hits = cast(mesh, camera, uv)

    img = np.empty((S * S, 3), dtype=np.uint8)
    img[:] = _BACKGROUND
    h = hits.hit
    if h.any():
        R = np.asarray(camera.rotation)
        faces = mesh.faces[hits.tri[h]]
        vn = mesh.vertex_normals
        n = np.einsum("ij,ijk->ik", hits.bary[h], vn[faces])
        fn = mesh.face_normals[hits.tri[h]]
        bad = ~np.isfinite(n).all(1) | (np.linalg.norm(n, axis=1) < 1e-8)
        n[bad] = fn[bad]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        view = R[2]  # world direction towards the camera
        # Two-sided: flip normals facing away from the viewer.
        n *= np.where((n @ view) < 0, -1.0, 1.0)[:, None]
        if light_dir is None:
            l_cam = np.array([-0.3, 0.35, 1.0])
        else:
            l_cam = np.asarray(light_dir, dtype=np.float64)
        l_cam = l_cam / np.linalg.norm(l_cam)
        l_world = l_cam @ R
        shade = 0.22 + 0.78 * np.clip(n @ l_world, 0.0, 1.0)
        img[h] = np.clip(_SKIN[None, :] * shade[:, None] * 255.0, 0, 255).astype(np.uint8)
    return RenderResult(
        image=img.reshape(S, S, 3),
        tri=hits.tri.reshape(S, S),
        points=hits.points.reshape(S, S, 3),
        camera=camera,
    )


def rotation_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
