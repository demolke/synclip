"""
Nearest-timestamp matching of frames across free-running cameras.

Frames stamped on one common clock (host capture times of a capmo session, or
the arrival times of the live preview) do not land at the same instants and the
rates differ, so "simultaneous" means "nearest frame within a tolerance".

* :func:`nearest` -- index of the closest timestamp in a sorted list (bisect);
  the live board tracker uses it to find views of the board that two cameras
  saw at nearly the same time.
* :func:`synchronized_frames` -- group whole streams into near-simultaneous
  sets anchored on a master stream (formerly ``capmo.sync``; the offline
  reconstruction now resamples per-camera timelines instead, see
  :func:`capmo.selfcal.load_take`).

Pure timestamp arithmetic (no video, no decode).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass
class FrameSet:
    """One synchronized set: a frame index per camera, near a common time."""
    t_ms: float                  # the master frame's timestamp
    frames: dict[str, int]       # cam_id -> frame index into that cam's stream
    spread_ms: float             # max - min timestamp across the matched frames


def nearest(sorted_ts, t: float) -> int:
    """Index of the timestamp in *sorted_ts* closest to *t*."""
    i = bisect.bisect_left(sorted_ts, t)
    if i == 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    return i if (sorted_ts[i] - t) < (t - sorted_ts[i - 1]) else i - 1


def synchronized_frames(timestamps: dict[str, list[float]], tol_ms: float = 20.0,
                        min_cams: int = 2, master: str | None = None) -> list[FrameSet]:
    """Group frames across cameras into near-simultaneous sets.

    *timestamps*: ``{cam_id: [server_ms, ...]}`` -- each list **ascending**.
    *tol_ms*: a camera joins a set only if its nearest frame is within this gap
    of the master frame.
    *min_cams*: drop sets seen by fewer than this many cameras (need >=2 to
    triangulate). *master*: stream to anchor on; defaults to the one with the
    most frames (the highest-rate camera makes the densest timeline).
    """
    streams = {c: ts for c, ts in timestamps.items() if ts}
    if len(streams) < min_cams:
        return []
    if master is None or master not in streams:
        master = max(streams, key=lambda c: len(streams[c]))

    others = [c for c in streams if c != master]
    out: list[FrameSet] = []
    seen: set[tuple] = set()
    for mi, t in enumerate(streams[master]):
        frames = {master: mi}
        matched_ts = [t]
        for c in others:
            j = nearest(streams[c], t)
            if abs(streams[c][j] - t) <= tol_ms:
                frames[c] = j
                matched_ts.append(streams[c][j])
        if len(frames) < min_cams:
            continue
        key = tuple(sorted(frames.items()))
        if key in seen:        # two master frames snapped to the same partners
            continue
        seen.add(key)
        out.append(FrameSet(t_ms=t, frames=frames,
                            spread_ms=max(matched_ts) - min(matched_ts)))
    return out


_nearest = nearest          # old private name (tests / callers of capmo.sync)
