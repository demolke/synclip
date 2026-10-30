"""
Offsets between per-camera **frame indices** of a take whose frames carry
times on one common clock (``extract_keypoints`` on a capmo session: ``t`` is
the host capture time since the session origin, ``frame_idx`` counts each
camera's own decodable frames, so equal ``frame_idx`` are *not* the same
instant when the cameras started at different times).
"""

from __future__ import annotations

import numpy as np


def frame_index_offsets(frame_idx: dict, t: dict, fps: float, ref: str | None = None) -> dict:
    """``{cam: offset}`` in the ``calibrate_board --offsets`` convention (camera
    X's frame ``k + offset`` shows the reference camera's frame ``k``).

    Per camera the frame clock ``a_c = median(frame_idx - fps * t)`` is fitted
    (robust to dropped frames); ``offset_c = a_c - a_ref`` (fractional: the
    cameras' frames interleave). Cameras without samples get 0. *ref* defaults
    to the first camera.
    """
    cams = list(frame_idx)
    if not cams:
        return {}
    ref = ref or cams[0]
    a = {}
    for c in cams:
        fi = np.asarray(frame_idx[c], np.float64)
        tt = np.asarray(t[c], np.float64)
        if len(fi):
            a[c] = float(np.median(fi - fps * tt))
    if ref not in a:
        return {c: 0.0 for c in cams}
    return {c: (a[c] - a[ref]) if c in a else 0.0 for c in cams}


def session_take_offsets(take_dir: str) -> dict:
    """:func:`frame_index_offsets` of the ``<cam>.calib.npz`` files of a take
    (``meta.json`` ``calib_videos``), for ``calibrate_board --offsets``."""
    import json
    import os
    with open(os.path.join(take_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    fi, tt, fps = {}, {}, None
    for e in meta.get("calib_videos", []):
        p = os.path.join(take_dir, e.get("npz") or f"{e['cam_id']}.calib.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=False)
        fi[e["cam_id"]], tt[e["cam_id"]] = d["frame_idx"], d["t"]
        fps = fps or float(d["fps"])
    return frame_index_offsets(fi, tt, fps or 30.0)
