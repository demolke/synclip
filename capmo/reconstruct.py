"""
Triangulate per-view 2D poses into a single 3D skeleton.

Given several calibrated cameras (their projection matrices live in
:class:`capmo.calibration.RigCalibration`) and each camera's 2D pose for the same
synchronized instant, every joint is reconstructed by triangulating the views
that saw it confidently. Visibility gates which views vote and weights the
linear solve; the per-joint reprojection residual is returned so callers can
flag joints the cameras disagree on.

Pure geometry (numpy) -- testable by projecting a known skeleton into synthetic
cameras and checking it comes back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import triangulate_point, reprojection_error


@dataclass
class View:
    """One camera's contribution to a frame: its projection + 2D pose."""
    cam_id: str
    P: np.ndarray              # (3, 4) world -> pixels
    points: np.ndarray         # (K, 2) pixel landmarks
    visibility: np.ndarray     # (K,) confidence in [0, 1]


@dataclass
class Skeleton3D:
    points: np.ndarray         # (K, 3) world coords (nan where unsolved)
    valid: np.ndarray          # (K,) bool
    residual_px: np.ndarray    # (K,) mean reprojection error (nan where unsolved)
    n_views: np.ndarray        # (K,) int, how many cameras voted for each joint

    @property
    def num_valid(self) -> int:
        return int(self.valid.sum())


def reconstruct(views: list[View], vis_thresh: float = 0.5,
                min_views: int = 2) -> Skeleton3D:
    """Reconstruct a 3D skeleton from calibrated 2D views.

    A joint is solved from the views whose visibility >= *vis_thresh*; it needs
    at least *min_views* such views (>=2 to triangulate at all).
    """
    if not views:
        raise ValueError("need at least one view")
    K = views[0].points.shape[0]
    for v in views:
        if v.points.shape[0] != K:
            raise ValueError("all views must share the same landmark count")

    points = np.full((K, 3), np.nan)
    valid = np.zeros(K, bool)
    residual = np.full(K, np.nan)
    n_views = np.zeros(K, int)

    for k in range(K):
        sel = [v for v in views if v.visibility[k] >= vis_thresh]
        n_views[k] = len(sel)
        if len(sel) < min_views:
            continue
        projs = [v.P for v in sel]
        pts = [tuple(v.points[k]) for v in sel]
        wts = [float(v.visibility[k]) for v in sel]
        X = triangulate_point(projs, pts, weights=wts)
        points[k] = X
        valid[k] = True
        residual[k] = float(np.mean([reprojection_error(P, X, p)
                                     for P, p in zip(projs, pts)]))
    return Skeleton3D(points=points, valid=valid, residual_px=residual, n_views=n_views)
