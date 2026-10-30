"""
Frame offsets from the **performer's body** (re-exported from
:mod:`capmo.selfcal`, where they share the take model and the epipolar /
reprojection machinery of the self-calibration):

* :func:`estimate_offsets` -- calibration-free: per camera pair, scan sub-frame
  shifts and keep the one with the lowest median Sampson (epipolar) error of
  the 2D keypoints; optional clock-based *prior* (capmo sessions) centres the
  scan and is the fallback for an unresolved camera.
* :func:`refine_offsets` -- calibrated: the shift that minimises the
  reprojection error against the triangulated track.
* :func:`shift_track` -- apply a (sub-)frame shift to one camera's keypoints.
"""
from ...selfcal import estimate_offsets, refine_offsets, shift_track  # noqa: F401
