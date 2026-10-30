"""
Per-frame capture times for a capmo recording, on the **host** clock.

What a recording knows per frame (``<slug>.index.jsonl``):

* ``recv_ms`` -- host arrival time of *every* frame. Includes the phone's
  encode delay plus Wi-Fi/TCP latency: biased late by tens of ms and jittery
  (bursts after a stall), so it is only a fallback.
* ``meta_ms`` -- the phone's capture time (phone wall clock, integer ms), but
  **only on keyframes** (the phone sends META before each IDR).
* the phone clock offset (``offset = phone - host``) from the PING/SYNC
  exchanges, sampled over time (``<slug>.clock.jsonl``) or, for old sessions,
  the ``meta_ms - server_ms`` of each keyframe / the final manifest value.

Chosen model (:func:`device_frame_times`): the keyframe META times are exact
anchors in the phone's capture timeline. Between two consecutive anchors the
frames are spaced evenly by frame index ``n``; if the anchor gap is longer than
``dn`` nominal periods (the encoder dropped ``d`` frames inside that GOP), the
``d`` missing slots are placed at the ``d`` largest ``recv_ms`` gaps of that
segment. Before the first / after the last anchor the frames are extrapolated
with the robust period (median of the anchor-pair periods). Anchors that
disagree with both neighbours (implausible period) are rejected first. The
phone times are then mapped to the host clock with a robust offset model
(:func:`offset_model`: low-rtt samples, constant or linear drift).

Fallbacks: one anchor -> the smoothed ``recv_ms`` shape pinned to that anchor;
no anchor (or no clock) -> :func:`recv_envelope`, the lower envelope of the
arrivals (minimum latency), which is smooth but biased late by the minimum
capture-to-arrival latency (unknown, typically 20-80 ms).

Accuracy on a synthetic stream (``test_timeline_accuracy_on_synthetic_stream``:
30 fps, 900 frames, 0.5 ms capture jitter, integer-ms META on every 30th frame,
3 encoder drops, 5..~100 ms arrival latency, clock samples with 1 ms noise):
anchor model **0.75 ms RMS / 2.3 ms max** error vs the true host capture time
(all 3 drops located); the recv envelope alone 8.3 ms RMS (57 ms max), raw
``recv_ms`` 24.8 ms RMS. On top of that comes the clock-offset error, bounded
by rtt_min / 2 per phone (NTP). Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameTimes:
    t_ms: np.ndarray                 # (N,) capture time per frame, host clock (ms)
    method: str                      # "meta_anchors" | "meta_single+recv" | "recv_envelope"
    period_ms: float                 # robust frame period
    n_anchors: int = 0
    n_anchors_rejected: int = 0
    dropped_slots: int = 0           # frames inferred missing between anchors
    anchor_resid_ms: float | None = None   # estimated META (anchor) jitter, RMS ms
    info: dict = field(default_factory=dict)


@dataclass
class OffsetModel:
    """``offset(host_ms) = a + b * (host_ms - t_ref)`` (phone - host, ms)."""
    a: float
    b: float = 0.0
    t_ref: float = 0.0
    method: str = "constant"         # "clock_history" | "keyframe_server_ms" | "manifest" | ...
    n_samples: int = 0
    rtt_min_ms: float | None = None
    spread_ms: float | None = None   # std of the kept offset samples
    uncertainty_ms: float | None = None  # NTP-style bound (rtt_min / 2), else a guess

    def at(self, host_ms):
        return self.a + self.b * (np.asarray(host_ms, np.float64) - self.t_ref)

    def to_host(self, phone_ms):
        """Phone ms -> host ms (one fixed-point step is plenty: |b| ~ 1e-5)."""
        p = np.asarray(phone_ms, np.float64)
        h = p - self.a
        return p - self.at(h)

    def to_dict(self) -> dict:
        return {"offset_ms": self.a, "drift_ppm": self.b * 1e6, "t_ref_ms": self.t_ref,
                "method": self.method, "n_samples": self.n_samples,
                "rtt_min_ms": self.rtt_min_ms, "spread_ms": self.spread_ms,
                "uncertainty_ms": self.uncertainty_ms}


# ----------------------------------------------------------------- helpers
def robust_slope(n: np.ndarray, y: np.ndarray, max_pairs: int = 4000) -> float:
    """Theil-Sen-like slope of y vs n (median of pairwise slopes, subsampled)."""
    n = np.asarray(n, np.float64)
    y = np.asarray(y, np.float64)
    if len(n) < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    m = len(n)
    k = min(max_pairs, m * (m - 1) // 2)
    if m * (m - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(m, 1)
    else:
        i = rng.integers(0, m, k)
        j = rng.integers(0, m, k)
    dn = n[j] - n[i]
    ok = dn != 0
    if not ok.any():
        return float("nan")
    return float(np.median((y[j] - y[i])[ok] / dn[ok]))


def recv_envelope(recv_ms: np.ndarray, period_ms: float | None = None,
                  window: int = 15) -> tuple[np.ndarray, float]:
    """Smoothed arrival-time timeline: ``P * n`` plus the running *minimum* of
    the residual (lowest latency in a +-window), then a moving average.

    Returns ``(t_ms, period_ms)``. Smooth, but biased late by the minimum
    latency.
    """
    r = np.asarray(recv_ms, np.float64)
    N = len(r)
    n = np.arange(N, dtype=np.float64)
    if N == 0:
        return r.copy(), float(period_ms or 0.0)
    if N == 1:
        return r.copy(), float(period_ms or 0.0)
    P = float(period_ms) if period_ms else robust_slope(n, r)
    if not np.isfinite(P) or P <= 0:
        P = float(np.median(np.diff(r))) if N > 1 else 0.0
    e = r - P * n
    W = max(1, int(window))
    pad = np.pad(e, W, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    env = sliding_window_view(pad, 2 * W + 1).min(axis=1)
    k = np.ones(2 * W + 1) / (2 * W + 1)
    env = np.convolve(np.pad(env, W, mode="edge"), k, mode="valid")
    return P * n + env, P


def _segment_times(n0, n1, m0, m1, P, recv):
    """Times of frames n0..n1 (inclusive) between anchors (n0,m0) and (n1,m1).

    Even spacing, except that ``d = round((m1-m0)/P) - (n1-n0)`` dropped slots
    are inserted where the arrival times say so: with the right drop position
    the latency ``recv - t`` is one stationary series, with a wrong one a run of
    frames is shifted by a whole period. Each slot is placed (greedily) at the
    step that minimises the L1 spread of that latency.
    """
    dn = n1 - n0
    dt = m1 - m0
    d = int(round(dt / P)) - dn if P > 0 else 0
    slots = np.arange(dn + 1, dtype=np.float64)
    placed = 0
    if d > 0 and recv is not None and dn >= 2:
        r = recv[n0:n1 + 1]
        for _ in range(min(d, dn)):
            best, best_cost = None, np.inf
            for j in range(1, dn + 1):
                cand = slots.copy()
                cand[j:] += 1.0
                lat = r - (m0 + P * cand)
                cost = float(np.abs(lat - np.median(lat)).sum())
                if cost < best_cost:
                    best, best_cost = j, cost
            slots[best:] += 1.0
            placed += 1
    total = slots[-1] if slots[-1] > 0 else 1.0
    return m0 + dt * slots / total, placed


def device_frame_times(n_frames: int, anchor_n, anchor_ms, recv_ms=None,
                       nominal_fps: float | None = None) -> FrameTimes:
    """Per-frame capture times in the **phone** clock from keyframe anchors.

    *anchor_n* / *anchor_ms*: frame indices and META times of the keyframes.
    Needs >= 2 anchors (callers use the recv fallbacks otherwise).
    """
    an = np.asarray(anchor_n, np.int64)
    am = np.asarray(anchor_ms, np.float64)
    order = np.argsort(an, kind="stable")
    an, am = an[order], am[order]
    _, uniq = np.unique(an, return_index=True)
    an, am = an[uniq], am[uniq]
    recv = None if recv_ms is None else np.asarray(recv_ms, np.float64)
    nominal_P = 1000.0 / nominal_fps if nominal_fps else None

    def periods(a_n, a_m):
        return np.diff(a_m) / np.maximum(np.diff(a_n), 1)

    P = float(np.median(periods(an, am))) if len(an) >= 2 else (nominal_P or 0.0)
    # Reject corrupt anchors. A segment between two anchors implies
    # d = round(dt / P) - dn missing frames: d < 0 is impossible (fewer slots
    # than frames) and a single bad anchor shows up as a +d / -d pair on its two
    # segments while the segment bridging over it is consistent.
    rejected = 0
    if len(an) >= 3 and P > 0:
        Pr = nominal_P if (nominal_P and abs(P / nominal_P - 1) < 0.06) else P

        def miss(i, j):
            return int(round((am[j] - am[i]) / Pr)) - int(an[j] - an[i])

        keep = np.ones(len(an), bool)
        for k in range(len(an)):
            dl = miss(k - 1, k) if k > 0 else None
            dr = miss(k, k + 1) if k < len(an) - 1 else None
            if dl is not None and dr is not None:
                if dl * dr < 0 and abs(miss(k - 1, k + 1)) < min(abs(dl), abs(dr)):
                    keep[k] = False
            elif (dl if dl is not None else dr) < 0:
                keep[k] = False            # end anchor that would need negative drops
        if keep.sum() >= 2:
            rejected = int((~keep).sum())
            an, am = an[keep], am[keep]
            P = float(np.median(periods(an, am)))
    # The anchor-pair median overestimates the period when most GOPs lost a
    # frame; the nominal period (when it agrees within 6%) is the better slot size.
    P_slot = nominal_P if (nominal_P and P > 0 and abs(P / nominal_P - 1) < 0.06) else P
    t = np.empty(n_frames, np.float64)
    dropped = 0
    for k in range(len(an) - 1):
        seg, d = _segment_times(int(an[k]), int(an[k + 1]), am[k], am[k + 1], P_slot, recv)
        t[an[k]:an[k + 1] + 1] = seg
        dropped += d
    if an[0] > 0:
        t[:an[0]] = am[0] - P * (an[0] - np.arange(an[0]))
    if an[-1] < n_frames - 1:
        t[an[-1] + 1:] = am[-1] + P * (np.arange(an[-1] + 1, n_frames) - an[-1])
    # META jitter: each anchor segment should span a whole number of slots;
    # the fractional remainder is the difference of two anchor errors.
    resid = None
    if len(an) >= 2 and P_slot > 0:
        q = np.diff(am) / P_slot
        resid = float(np.sqrt(np.mean(((q - np.round(q)) * P_slot) ** 2) / 2.0))
    return FrameTimes(t, "meta_anchors", P, int(len(an)), rejected, dropped, resid)


def offset_model(samples=None, keyframe_pairs=None, manifest_offset=None,
                 meta_recv_pairs=None, min_drift_span_ms: float = 60_000.0) -> OffsetModel | None:
    """Robust phone-minus-host clock offset.

    Preference order:

    1. *samples*: clock history ``[(host_ms, offset_ms, rtt_ms), ...]`` -- keep
       the lower-rtt half; constant (rtt-weighted mean), or a linear drift fit
       when they span more than *min_drift_span_ms*. Uncertainty = rtt_min/2.
    2. *keyframe_pairs*: ``[(server_ms, meta_ms), ...]`` from old index files
       (offset = meta - server, as estimated live) -> median.
    3. *manifest_offset*: the single final offset of an old manifest.
    4. *meta_recv_pairs*: ``[(recv_ms, meta_ms), ...]`` -> ``max(meta - recv)``
       (= offset minus the minimum latency; biased, last resort).
    """
    if samples:
        s = np.asarray([(h, o, r) for h, o, r in samples
                        if h is not None and o is not None and r is not None], np.float64)
        if len(s):
            rtt = s[:, 2]
            thr = np.quantile(rtt, 0.5)
            k = s[rtt <= thr + 1e-9]
            w = 1.0 / np.maximum(k[:, 2], 0.5)
            rmin = float(rtt.min())
            span = float(k[:, 0].max() - k[:, 0].min()) if len(k) > 1 else 0.0
            if len(k) >= 10 and span >= min_drift_span_ms:
                tref = float(np.average(k[:, 0], weights=w))
                A = np.stack([np.ones(len(k)), k[:, 0] - tref], 1) * np.sqrt(w)[:, None]
                a, b = np.linalg.lstsq(A, k[:, 1] * np.sqrt(w), rcond=None)[0]
                res = k[:, 1] - (a + b * (k[:, 0] - tref))
                return OffsetModel(float(a), float(b), tref, "clock_history_drift", len(s),
                                   rmin, float(np.std(res)), max(rmin / 2.0, 0.5))
            a = float(np.average(k[:, 1], weights=w))
            return OffsetModel(a, 0.0, float(k[:, 0].mean()), "clock_history", len(s), rmin,
                               float(np.std(k[:, 1])), max(rmin / 2.0, 0.5))
    if keyframe_pairs:
        d = np.asarray([m - sv for sv, m in keyframe_pairs], np.float64)
        return OffsetModel(float(np.median(d)), method="keyframe_server_ms", n_samples=len(d),
                           spread_ms=float(np.std(d)), uncertainty_ms=None)
    if manifest_offset is not None:
        return OffsetModel(float(manifest_offset), method="manifest", n_samples=1)
    if meta_recv_pairs:
        d = np.asarray([m - r for r, m in meta_recv_pairs], np.float64)
        return OffsetModel(float(d.max()), method="meta_recv_envelope", n_samples=len(d),
                           spread_ms=float(np.std(d)), uncertainty_ms=50.0)
    return None


def host_frame_times(recv_ms, meta_ms, offset: OffsetModel | None,
                     nominal_fps: float | None = None) -> FrameTimes:
    """Capture time of every frame on the host clock (ms).

    *recv_ms* (N,) arrival times; *meta_ms* (N,) phone capture time or nan
    (keyframes only). See the module docstring for the model.
    """
    recv = np.asarray(recv_ms, np.float64)
    meta = np.asarray(meta_ms, np.float64)
    N = len(recv)
    an = np.flatnonzero(np.isfinite(meta))
    if offset is not None and len(an) >= 2:
        ft = device_frame_times(N, an, meta[an], recv, nominal_fps)
        ft.t_ms = offset.to_host(ft.t_ms)
        return ft
    env, P = recv_envelope(recv, 1000.0 / nominal_fps if nominal_fps else None)
    if offset is not None and len(an) == 1:
        k = int(an[0])
        t = env + (offset.to_host(meta[k]) - env[k])
        return FrameTimes(t, "meta_single+recv", P, 1)
    return FrameTimes(env, "recv_envelope", P, int(len(an)))
