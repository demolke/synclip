"""
Frame offsets between unsynchronised cameras from the **board motion** (the
calibration segment): each camera's PnP poses of the board give the 3D speed of
the board centre per frame, which is view-independent, so the lag that best
cross-correlates two cameras' speed signals is their frame offset.

Used by ``calibrate_board --estimate-offsets`` (moved here from
``capmo/tools/calibrate_board.py``; the old names are re-exported there). The
cameras are duck-typed: ``cam.cam_id`` and ``cam.views`` with ``frame_idx``,
``rvec``, ``tvec`` (``rvec is None`` = no pose).
"""

from __future__ import annotations

import numpy as np


def _rodrigues(rvec):
    from ...multiview import rodrigues
    return rodrigues(rvec)


def board_speed_signal(cam, board_center, max_gap=3):
    """Per-frame 3D speed (m/frame) of the board centre (nan where unknown)."""
    vs = [v for v in cam.views if v.rvec is not None]
    if len(vs) < 3:
        return None, 0
    fi = np.array([v.frame_idx for v in vs])
    C = np.stack([_rodrigues(v.rvec) @ board_center + v.tvec for v in vs])
    o = np.argsort(fi)
    fi, C = fi[o], C[o]
    k0 = int(fi[0])
    grid = np.arange(k0, int(fi[-1]) + 1)
    pos = np.full((len(grid), 3), np.nan)
    for a in range(len(fi) - 1):
        g = fi[a + 1] - fi[a]
        if g <= 0 or g > max_gap:
            continue
        for s in range(g + 1):
            pos[fi[a] - k0 + s] = C[a] + (C[a + 1] - C[a]) * s / g
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return speed, k0


def _xcorr_lag(si, ki, sj, kj, max_lag, min_overlap=20):
    """Best lag d (cam j frame k+d ~ cam i frame k) by normalised cross-correlation."""
    best = None
    curve = {}
    for d in range(-max_lag, max_lag + 1):
        # sample k of i at index k-ki; sample k+d of j at index k+d-kj
        lo = max(ki, kj - d)
        hi = min(ki + len(si), kj - d + len(sj))
        if hi - lo < min_overlap:
            continue
        a = si[lo - ki:hi - ki]
        b = sj[lo + d - kj:hi + d - kj]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_overlap:
            continue
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den <= 0:
            continue
        c = float((a * b).sum() / den)
        curve[d] = (c, int(m.sum()))
        if best is None or c > curve[best][0]:
            best = d
    return best, curve


def estimate_board_offsets(cams: list, board_center, ref: int, *, max_lag=10,
                           min_corr=0.5, log=print) -> dict:
    """Per-camera frame offsets (vs *ref*) from the board centre speed signals."""
    V = len(cams)
    sig = [board_speed_signal(c, board_center) for c in cams]
    pairs = {}
    for i in range(V):
        for j in range(i + 1, V):
            if sig[i][0] is None or sig[j][0] is None:
                continue
            d, curve = _xcorr_lag(sig[i][0], sig[i][1], sig[j][0], sig[j][1], max_lag)
            if d is None:
                continue
            c, n = curve[d]
            sub = float(d)
            if d - 1 in curve and d + 1 in curve:
                y0, y1, y2 = curve[d - 1][0], c, curve[d + 1][0]
                den = y0 - 2 * y1 + y2
                if den < 0:
                    sub = d + 0.5 * (y0 - y2) / den
            pairs[(i, j)] = {"lag": int(d), "lag_subframe": sub, "corr": c, "n": n,
                             "trusted": bool(c >= min_corr)}
    rows, rhs, wts = [], [], []
    for (i, j), p in pairs.items():
        if p["trusted"]:
            r = np.zeros(V)
            r[j], r[i] = 1, -1
            rows.append(r)
            rhs.append(p["lag"])
            wts.append(p["corr"] * np.sqrt(p["n"]))
    r0 = np.zeros(V)
    r0[ref] = 1
    rows.append(r0)
    rhs.append(0.0)
    wts.append(100.0)
    A = np.array(rows) * np.array(wts)[:, None]
    off = np.linalg.lstsq(A, np.array(rhs) * np.array(wts), rcond=None)[0]
    linked = {ref}
    for _ in range(V):
        for (i, j), p in pairs.items():
            if p["trusted"] and (i in linked or j in linked):
                linked |= {i, j}
    off = np.round(off).astype(int)
    unresolved = [v for v in range(V) if v not in linked]
    off[unresolved] = 0
    log("  board-motion offsets (frames): " + ", ".join(
        f"{c.cam_id} {o:+d}" for c, o in zip(cams, off))
        + (f"; unresolved {[cams[v].cam_id for v in unresolved]}" if unresolved else ""))
    return {"offsets": off, "pairs": pairs, "unresolved": unresolved,
            "signals": sig}
