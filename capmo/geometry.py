"""
Multi-view geometry helpers (pure numpy): linear triangulation.

Used to validate calibration (reconstruct known board corners from two solved
cameras) and, later, to fuse per-view 2D pose into a 3D skeleton. Generic enough
to share with synclip eventually.
"""

from __future__ import annotations

import numpy as np


def triangulate_point(projections: list[np.ndarray], points2d: list,
                      weights: list[float] | None = None) -> np.ndarray:
    """Triangulate one 3D point from >=2 views by the linear (DLT) method.

    *projections*: 3x4 camera matrices P_i = K_i [R_i | t_i] (world -> pixels).
    *points2d*: matching (u, v) pixel observations, one per projection.
    *weights*: optional per-view confidence (e.g. landmark visibility); each
    view's two equations are scaled by the weight so low-confidence views pull
    the solution less. Returns the 3D point (x, y, z) in the shared world frame.
    """
    if len(projections) < 2 or len(projections) != len(points2d):
        raise ValueError("need >=2 matched (projection, point) pairs")
    if weights is not None and len(weights) != len(projections):
        raise ValueError("weights must match the number of views")
    rows = []
    for i, (P, (u, v)) in enumerate(zip(projections, points2d)):
        P = np.asarray(P, np.float64)
        w = 1.0 if weights is None else float(weights[i])
        rows.append(w * (u * P[2] - P[0]))
        rows.append(w * (v * P[2] - P[1]))
    A = np.asarray(rows, np.float64)
    _u, _s, vt = np.linalg.svd(A)
    X = vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


def reprojection_error(P: np.ndarray, point3d: np.ndarray, point2d) -> float:
    """Pixel distance between *point2d* and the projection of *point3d* by *P*."""
    P = np.asarray(P, np.float64)
    X = np.append(np.asarray(point3d, np.float64), 1.0)
    proj = P @ X
    uv = proj[:2] / proj[2]
    return float(np.hypot(uv[0] - point2d[0], uv[1] - point2d[1]))
