"""
Temporal cleanup of a triangulated 3D landmark track (pure numpy + scipy).

Steps, in the order :func:`clean_track` applies them:

1. :func:`fix_chirality` -- frames whose arms and legs disagree on left/right
   get the implausible part mirrored (:func:`fix_lr_glitches` is a temporal
   alternative, not used by default: noisy hips make neighbour votes flaky);
2. :func:`reject_bad_bones` (implausibly stretched/shrunk bones) and
   :func:`reject_outliers` (Hampel filter on positions) -> invalid;
3. :func:`fill_gaps` -- short gaps (<= ``max_gap`` frames) PCHIP-interpolated,
   unless the interpolated samples make implausible bones;
4. :func:`smooth` -- zero-phase Butterworth low-pass per contiguous segment
   (``filtfilt``: no lag), Savitzky-Golay for segments too short for it;
   per-landmark cutoffs (default 4 Hz head/torso, 6 Hz elbows/knees, 7 Hz
   wrists/hands/ankles/feet) so fast limb motion (kicks) is kept;
5. :func:`enforce_bone_lengths` -- per-frame constraint projection to constant,
   left/right-symmetric bone lengths (median of the take);
6. long gaps are bridged (interpolated / held) so the output is finite
   everywhere, but those samples stay ``valid = False``.
"""

from __future__ import annotations

import numpy as np

from .multiview import PART_PAIRS, quiet, swap_perm

# Bones (landmark index pairs) held at constant length.
BONES = [
    (11, 12), (23, 24), (11, 23), (12, 24),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),                 # arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (23, 25), (25, 27), (24, 26), (26, 28),                 # legs
    (27, 29), (27, 31), (29, 31), (28, 30), (28, 32), (30, 32),  # feet
    (7, 8), (0, 7), (0, 8), (2, 5), (0, 2), (0, 5), (9, 10),     # head (rigid-ish)
]
_MIRROR = swap_perm()


def _mirror_bone(a, b):
    return tuple(sorted((int(_MIRROR[a]), int(_MIRROR[b]))))


def bone_lengths(X: np.ndarray, bones=BONES) -> np.ndarray:
    """(T, nb) lengths (nan where an endpoint is missing)."""
    a = np.array([b[0] for b in bones])
    b = np.array([b[1] for b in bones])
    return np.linalg.norm(X[:, a] - X[:, b], axis=-1)


def fix_lr_glitches(X: np.ndarray, valid: np.ndarray, *, half_window: int = 4,
                    ratio: float = 0.6, passes: int = 2):
    """Relabel parts whose L/R assignment disagrees with most neighbours."""
    X = X.copy()
    valid = valid.copy()
    T = len(X)
    n_fixed = 0
    for _ in range(passes):
        for part, pairs in PART_PAIRS.items():
            J = np.array([j for p in pairs for j in p])
            perm = swap_perm([part])
            Jm = perm[J]
            changed = []
            for t in range(T):
                if not valid[t, J].any():
                    continue
                votes = 0
                total = 0
                for u in range(max(0, t - half_window), min(T, t + half_window + 1)):
                    if u == t:
                        continue
                    m = valid[t, J] & valid[u, J] & valid[t, Jm] & valid[u, Jm]
                    if m.sum() < 2:
                        continue
                    dk = np.linalg.norm(X[t, J[m]] - X[u, J[m]], axis=-1).sum()
                    ds = np.linalg.norm(X[t, Jm[m]] - X[u, J[m]], axis=-1).sum()
                    total += 1
                    votes += ds < ratio * dk
                if total >= 2 and votes > total / 2:
                    changed.append(t)
            for t in changed:
                X[t, J] = X[t, Jm]
                valid[t, J] = valid[t, Jm]
            n_fixed += len(changed)
    return X, valid, n_fixed


def _facing(X, l, r, up):
    f = np.cross(X[:, l] - X[:, r], up)
    with np.errstate(invalid="ignore", divide="ignore"):
        return f / np.linalg.norm(f, axis=-1, keepdims=True)


def fix_chirality(X: np.ndarray, valid: np.ndarray, up=(0.0, 0.0, 1.0)):
    """Make arms and legs agree on left/right in every frame.

    Shoulders and hips can't face opposite ways, so where the labelled
    shoulder and hip axes point opposite directions one part is mirrored.
    The part whose own cue is worse is flipped: the nose should be in front
    of the shoulders (arms), the toes in front of the heels (legs). When
    neither cue is clearly negative the torso joints are invalidated instead.
    Returns (X, valid, n_flipped, n_invalidated).
    """
    X = X.copy()
    valid = valid.copy()
    up = np.asarray(up, float)
    m = valid[:, [11, 12, 23, 24]].all(1)
    dot = np.sum((X[:, 11] - X[:, 12]) * (X[:, 23] - X[:, 24]), 1)
    bad = np.flatnonzero(m & (dot < 0))
    fa = _facing(X, 11, 12, up)
    fl = _facing(X, 23, 24, up)
    nose = X[:, 0] - 0.5 * (X[:, 11] + X[:, 12])
    toes = 0.5 * (X[:, 31] + X[:, 32] - X[:, 29] - X[:, 30])
    with np.errstate(invalid="ignore", divide="ignore"):
        s_arm = np.sum(fa * nose, 1) / np.linalg.norm(nose - (nose @ up)[:, None] * up, axis=1)
        s_leg = np.sum(fl * toes, 1) / np.linalg.norm(toes - (toes @ up)[:, None] * up, axis=1)
    s_arm = np.where(valid[:, 0], s_arm, np.nan)
    s_leg = np.where(valid[:, [29, 30, 31, 32]].all(1), s_leg, np.nan)
    pa, pl = swap_perm(["arms"]), swap_perm(["legs"])
    n_flip = n_inv = 0
    for t in bad:
        a, l = s_arm[t], s_leg[t]
        a = 0.0 if not np.isfinite(a) else a
        l = 0.0 if not np.isfinite(l) else l
        if min(a, l) < -0.2:
            perm = pa if a < l else pl
            X[t] = X[t, perm]
            valid[t] = valid[t, perm]
            n_flip += 1
        else:
            valid[t, [11, 12, 23, 24]] = False
            n_inv += 1
    return X, valid, n_flip, n_inv


def reject_bad_bones(X: np.ndarray, valid: np.ndarray, bones=BONES[:24], *,
                     lo: float = 0.65, hi: float = 1.45) -> np.ndarray:
    """Invalidate both endpoints of bones far from the take's median length."""
    a = np.array([b[0] for b in bones])
    b = np.array([b[1] for b in bones])
    L = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    L = np.where(valid[:, a] & valid[:, b], L, np.nan)
    with quiet():
        r = L / np.nanmedian(L, 0)
    bad = (r < lo) | (r > hi)
    out = valid.copy()
    t, k = np.nonzero(bad)
    out[t, a[k]] = False
    out[t, b[k]] = False
    return out


def reject_outliers(X: np.ndarray, valid: np.ndarray, *, half_window: int = 4,
                    k: float = 4.0, min_dev: float = 0.06, min_support: int = 2) -> np.ndarray:
    """Hampel filter: invalidate samples far from the windowed median, and
    samples with fewer than *min_support* valid neighbours in the window
    (isolated triangulations can't be verified and are often wrong)."""
    T = len(X)
    out = valid.copy()
    Xn = np.where(valid[..., None], X, np.nan)
    pad = np.pad(Xn, ((half_window, half_window), (0, 0), (0, 0)), constant_values=np.nan)
    win = np.stack([pad[i:i + T] for i in range(2 * half_window + 1)])   # (W,T,J,3)
    with quiet():
        med = np.nanmedian(win, 0)
        dev = np.linalg.norm(win - med[None], axis=-1)
        mad = np.nanmedian(dev, 0)
    d = np.linalg.norm(Xn - med, axis=-1)
    thr = np.maximum(k * 1.4826 * np.nan_to_num(mad, nan=np.inf), min_dev)
    out &= ~(d > thr)
    support = np.isfinite(win[..., 0]).sum(0) - np.isfinite(Xn[..., 0])
    out &= support >= min_support
    return out


def _runs(mask):
    """(start, end) half-open runs of True in a 1-D bool array."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_gaps(X: np.ndarray, valid: np.ndarray, max_gap: int = 10):
    """PCHIP-interpolate interior gaps of <= *max_gap* frames per joint."""
    from scipy.interpolate import PchipInterpolator
    X = X.copy()
    filled = np.zeros_like(valid)
    T, J = valid.shape
    t = np.arange(T)
    for j in range(J):
        v = valid[:, j]
        if v.sum() < 2:
            continue
        gaps = [(s, e) for s, e in _runs(~v) if s > 0 and e < T and e - s <= max_gap]
        if not gaps:
            continue
        f = PchipInterpolator(t[v], X[v, j], axis=0)
        for s, e in gaps:
            X[s:e, j] = f(t[s:e])
            filled[s:e, j] = True
    return X, valid | filled, filled


CORE_JOINTS = list(range(0, 13)) + [23, 24]          # head + shoulders + hips
MID_JOINTS = [13, 14, 25, 26]                        # elbows, knees


def joint_cutoffs(cutoff_hz: float = 7.0, core_hz: float = 4.0, mid_hz: float = 6.0) -> np.ndarray:
    """Per-landmark low-pass cutoffs: the torso/head move slowly (and their
    MediaPipe landmarks wobble most), the extremities carry the fast motion."""
    c = np.full(33, float(cutoff_hz))
    c[MID_JOINTS] = min(mid_hz, cutoff_hz)
    c[CORE_JOINTS] = min(core_hz, cutoff_hz)
    return c


def smooth(X: np.ndarray, valid: np.ndarray, fps: float, cutoff_hz=7.0,
           order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass on each contiguous valid run.

    *cutoff_hz* is a scalar or a per-landmark array (see :func:`joint_cutoffs`).
    """
    from scipy.signal import butter, filtfilt, savgol_filter
    cut = np.broadcast_to(np.asarray(cutoff_hz, float), (X.shape[1],))
    out = X.copy()
    for j in range(X.shape[1]):
        b, a = butter(order, min(cut[j] / (0.5 * fps), 0.99))
        padlen = 3 * max(len(a), len(b))
        for s, e in _runs(valid[:, j]):
            n = e - s
            seg = X[s:e, j]
            if n > padlen + 1:
                out[s:e, j] = filtfilt(b, a, seg, axis=0, padtype="odd", padlen=min(padlen, n - 1))
            elif n >= 5:
                out[s:e, j] = savgol_filter(seg, 5 if n < 7 else 7, 2, axis=0)
    return out


def target_lengths(X: np.ndarray, bones=BONES, symmetric: bool = True) -> np.ndarray:
    """Median length per bone; mirrored bones share their mean."""
    with quiet():
        L = np.nanmedian(bone_lengths(X, bones), 0)
    if symmetric:
        idx = {tuple(sorted(b)): i for i, b in enumerate(bones)}
        L2 = L.copy()
        for i, (a, b) in enumerate(bones):
            m = idx.get(_mirror_bone(a, b))
            if m is not None and np.isfinite(L[m]):
                L2[i] = 0.5 * (L[i] + L[m])
        L = L2
    return L


def enforce_bone_lengths(X: np.ndarray, valid: np.ndarray, lengths: np.ndarray,
                         bones=BONES, iters: int = 60) -> np.ndarray:
    """Project each frame onto constant bone lengths (Gauss-Seidel PBD).

    Both endpoints move equally along the bone, so the correction is spread
    over the skeleton rather than dumped on one joint. Only bones with both
    endpoints valid are constrained.
    """
    X = X.copy()
    for _ in range(iters):
        for (a, b), L0 in zip(bones, lengths):
            if not np.isfinite(L0):
                continue
            m = valid[:, a] & valid[:, b]
            d = X[m, b] - X[m, a]
            L = np.linalg.norm(d, axis=-1, keepdims=True)
            corr = 0.5 * (L - L0) / np.maximum(L, 1e-9) * d
            X[m, a] += corr
            X[m, b] -= corr
    return X


def bridge(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill every remaining nan by linear interpolation / nearest hold."""
    X = X.copy()
    T, J = valid.shape
    t = np.arange(T)
    for j in range(J):
        v = valid[:, j] & np.isfinite(X[:, j]).all(-1)
        if not v.any():
            X[:, j] = 0.0
            continue
        for d in range(3):
            X[~v, j, d] = np.interp(t[~v], t[v], X[v, j, d])
    return X


def clean_track(X: np.ndarray, valid: np.ndarray, fps: float, *, cutoff_hz: float = 7.0,
                core_hz: float = 4.0, mid_hz: float = 6.0,
                max_gap: int = 6, bones: bool = True, log=print) -> dict:
    """Run the full cleanup; returns a dict of intermediate + final arrays."""
    X = np.where(valid[..., None], X, np.nan)
    X1, v1, n_lr, n_inv = fix_chirality(X, valid)
    v1b = reject_bad_bones(X1, v1)
    n_bones = int(v1.sum() - v1b.sum())
    v2 = reject_outliers(X1, v1b)
    n_out = int(v1b.sum() - v2.sum())
    X2 = np.where(v2[..., None], X1, np.nan)
    X3, v3, filled = fill_gaps(X2, v2, max_gap=max_gap)
    # interpolated samples must still make plausible bones (independent
    # per-joint interpolation of a swinging limb, or across a relabel, can't)
    bad_fill = filled & ~reject_bad_bones(X3, v3, lo=0.8, hi=1.25)
    v3 = v3 & ~bad_fill
    filled = filled & ~bad_fill
    X3 = np.where(v3[..., None], X3, np.nan)
    X4 = smooth(X3, v3, fps, joint_cutoffs(cutoff_hz, core_hz, mid_hz))
    L = target_lengths(np.where(v3[..., None], X4, np.nan))
    X5 = enforce_bone_lengths(X4, v3, L) if bones else X4
    X6 = bridge(np.where(v3[..., None], X5, np.nan), v3)
    log(f"  cleanup: {n_lr} L/R part relabels ({n_inv} ambiguous torsos dropped), {n_bones} bad-bone samples, {n_out} outlier samples, "
        f"{int(filled.sum())} gap samples filled, low-pass {core_hz}/{mid_hz}/{cutoff_hz} Hz "
        "(core/elbow+knee/extremities)")
    return {"joints": X6, "valid": v3, "smoothed": X4, "filled": filled,
            "lengths": L, "n_lr_fixed": n_lr, "n_outliers": n_out, "n_bad_bone": n_bones}
