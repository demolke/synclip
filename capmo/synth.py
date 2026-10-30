"""
Synthetic multi-camera renderer for tests/calibration validation.

synclip's ``SoftwareRenderer`` only draws a flat-shaded head (no textures), so
this is a small **generic** pinhole rasterizer in the same spirit: a vectorised
barycentric z-buffer, but with per-triangle flat colour and a real camera
``K [R|t]``. It renders arbitrary coloured-quad meshes.

We build the ChArUco board as geometry rather than a texture: OpenCV's own board
bitmap is run-length merged into a few thousand rectangles, each a quad coloured
in **grey** (light-grey for the board's white, dark-grey for its black) -- closer
to what a real camera sends than pure 0/255, and exactly what detection needs to
find the markers and corners.

Pure numpy + OpenCV (only for the board bitmap). No Qt, no GL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class Mesh:
    verts: np.ndarray        # (N,3) float
    tris: np.ndarray         # (M,3) int
    tri_color: np.ndarray    # (M,3) uint8


def pinhole_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV-convention world->camera (R, t): camera looks along +z at *target*."""
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, np.float64))
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])      # rows = camera x, y, z in world
    t = -R @ eye
    return R, t


def rasterize(mesh: Mesh, K: np.ndarray, R: np.ndarray, t: np.ndarray,
              width: int, height: int, bg: int = 128) -> np.ndarray:
    """Render *mesh* from camera (K, R, t) to an HxWx3 uint8 BGR image."""
    verts = np.asarray(mesh.verts, np.float64)
    cam = verts @ np.asarray(R, np.float64).T + np.asarray(t, np.float64)   # (N,3)
    depth = cam[:, 2]
    proj = cam @ np.asarray(K, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = proj[:, 0] / proj[:, 2]
        sy = proj[:, 1] / proj[:, 2]

    tris = np.asarray(mesh.tris, np.int64)
    t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]
    x0, y0 = sx[t0], sy[t0]
    x1, y1 = sx[t1], sy[t1]
    x2, y2 = sx[t2], sy[t2]
    d0, d1, d2 = depth[t0], depth[t1], depth[t2]

    area = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    in_front = (d0 > 1e-6) & (d1 > 1e-6) & (d2 > 1e-6)
    bminx = np.maximum(np.floor(np.minimum(np.minimum(x0, x1), x2)), 0).astype(np.int32)
    bmaxx = np.minimum(np.ceil(np.maximum(np.maximum(x0, x1), x2)), width - 1).astype(np.int32)
    bminy = np.maximum(np.floor(np.minimum(np.minimum(y0, y1), y2)), 0).astype(np.int32)
    bmaxy = np.minimum(np.ceil(np.maximum(np.maximum(y0, y1), y2)), height - 1).astype(np.int32)
    visible = (np.abs(area) > 1e-9) & in_front & (bmaxx >= bminx) & (bmaxy >= bminy)

    img = np.full((height * width, 3), bg, np.uint8)
    zbuf = np.full(height * width, np.inf, np.float64)
    vis = np.nonzero(visible)[0]
    if len(vis) == 0:
        return img.reshape(height, width, 3)

    vminx, vmaxx, vminy, vmaxy = bminx[vis], bmaxx[vis], bminy[vis], bmaxy[vis]
    bw = (vmaxx - vminx + 1).astype(np.int64)
    bh = (vmaxy - vminy + 1).astype(np.int64)
    counts = bw * bh
    tri_ids = np.repeat(vis, counts)

    flat = np.arange(int(counts.sum()), dtype=np.int64)
    starts = np.zeros(len(vis) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    local = flat - np.repeat(starts[:-1], counts)
    bw_rep = np.repeat(bw, counts)
    px = (np.repeat(vminx, counts) + (local % bw_rep)).astype(np.int32)
    py = (np.repeat(vminy, counts) + (local // bw_rep)).astype(np.int32)

    inv = (1.0 / area[tri_ids])
    pxf, pyf = px.astype(np.float64), py.astype(np.float64)
    x2r, y2r = x2[tri_ids], y2[tri_ids]
    b0 = ((y1[tri_ids] - y2r) * (pxf - x2r) + (x2[tri_ids] - x1[tri_ids]) * (pyf - y2r)) * inv
    b1 = ((y2r - y0[tri_ids]) * (pxf - x2r) + (x0[tri_ids] - x2r) * (pyf - y2r)) * inv
    b2 = 1.0 - b0 - b1
    # Double-sided: accept whichever winding (the board is a single plane).
    inside = ((b0 >= 0) & (b1 >= 0) & (b2 >= 0)) | ((b0 <= 0) & (b1 <= 0) & (b2 <= 0))

    b0, b1, b2 = b0[inside], b1[inside], b2[inside]
    px, py, tri_f = px[inside], py[inside], tri_ids[inside]
    z = b0 * d0[tri_f] + b1 * d1[tri_f] + b2 * d2[tri_f]

    # Nearest-fragment z-buffer: sort by (pixel, depth), keep first per pixel.
    linear = py.astype(np.int64) * width + px
    order = np.lexsort((z, linear))
    lin_s = linear[order]
    _, first = np.unique(lin_s, return_index=True)
    sel = order[first]
    img[linear[sel]] = mesh.tri_color[tri_f[sel]]
    return img.reshape(height, width, 3)


def _quad(verts, tris, colors, x_lo, x_hi, y_lo, y_hi, color) -> None:
    base = len(verts)
    verts.extend([(x_lo, y_lo, 0.0), (x_hi, y_lo, 0.0),
                  (x_hi, y_hi, 0.0), (x_lo, y_hi, 0.0)])
    tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    colors.extend([color, color])


def charuco_board_mesh(board_cfg, px_per_square: int = 30,
                       white: int = 210, black: int = 45) -> Mesh:
    """Build the ChArUco board as grey quads on the Z=0 plane, centred at origin.

    Uses OpenCV's board bitmap (so markers/corners are exact), run-length merged
    per row into rectangles. Greys (not 0/255) emulate a real camera.
    """
    sx, sy = board_cfg.squares_x, board_cfg.squares_y
    bw, bh = sx * px_per_square, sy * px_per_square
    bitmap = board_cfg.board().generateImage((bw, bh))   # grayscale, board fills it
    w_m = sx * board_cfg.square_len
    h_m = sy * board_cfg.square_len

    verts: list = []
    tris: list = []
    colors: list = []
    for row in range(bh):
        col = 0
        rowpix = bitmap[row]
        while col < bw:
            val = rowpix[col]
            start = col
            while col < bw and rowpix[col] == val:
                col += 1
            shade = white if val >= 128 else black
            # pixel rect [start,col) x [row,row+1) -> world XY (centred). Flip the
            # row axis so the board's front face (markers as drawn) points toward
            # +Z where the cameras sit -- otherwise we'd view its mirror image and
            # the (non-mirror-invariant) ArUco markers wouldn't decode.
            x_lo = (start / bw - 0.5) * w_m
            x_hi = (col / bw - 0.5) * w_m
            y_lo = (0.5 - row / bh) * h_m
            y_hi = (0.5 - (row + 1) / bh) * h_m
            _quad(verts, tris, colors, x_lo, x_hi, y_lo, y_hi, (shade, shade, shade))
    return Mesh(np.asarray(verts, np.float64),
                np.asarray(tris, np.int64),
                np.asarray(colors, np.uint8))
