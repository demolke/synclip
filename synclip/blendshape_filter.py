"""
Post-processing of raw MediaPipe blendshape weights for nicer mouth shapes.

MediaPipe FaceLandmarker emits all 52 ARKit channels independently, so it
routinely produces combinations a real face can't make (e.g. a wide smile and
a tight pucker at the same time), and it systematically *under-drives* the
mouth-articulation channels used for speech. This module applies two optional,
independently-tunable passes, in order:

    1. Mouth-articulation gain    - boost the weak speech channels.
    2. Temporal smoothing (EMA)   - remove per-frame jitter.

Everything is driven by plain attributes so the UI thread can retune live; the
capture thread just calls process().
"""

from __future__ import annotations

from .arkit_names import BLENDSHAPE_NAMES

# Name -> index lookup so the constraint logic reads by channel name.
_IDX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}


def _i(name: str) -> int:
    return _IDX[name]


def _sample_lut(lut: list[float], v: float) -> float:
    """Linear-interpolated lookup of *v* (0..1) in *lut*."""
    n = len(lut)
    if n == 0:
        return v
    x = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
    f = x * (n - 1)
    i = int(f)
    if i >= n - 1:
        return lut[n - 1]
    frac = f - i
    return lut[i] * (1.0 - frac) + lut[i + 1] * frac


# Channels boosted by the mouth-articulation gain (the ones MediaPipe
# under-drives for speech). Emotional shapes (smile/frown) are left alone.
_GAIN_CHANNELS = [
    _i("jawOpen"), _i("mouthClose"), _i("mouthFunnel"), _i("mouthPucker"),
    _i("mouthLowerDownLeft"), _i("mouthLowerDownRight"),
    _i("mouthUpperUpLeft"), _i("mouthUpperUpRight"),
    _i("mouthShrugLower"), _i("mouthShrugUpper"),
    _i("mouthRollLower"), _i("mouthRollUpper"),
]

# Mouth-width channels. MediaPipe tends to over-fire these, making vowels look
# like a wide grimace instead of a round shape; the width curve tames them
# (drag the high end down to round the mouth out).
_WIDTH_CHANNELS = [_i("mouthStretchLeft"), _i("mouthStretchRight")]


class BlendshapeFilter:
    """Stateful per-stream filter. One instance per capture worker."""

    def __init__(self) -> None:
        # Live-tunable parameters (written from the UI thread).
        # Default 0.0: no temporal smoothing, so we add zero latency unless the
        # user explicitly dials it in. When 0, the EMA pass below is skipped
        # entirely (no blend against the previous frame).
        self.smoothing: float = 0.0           # EMA weight on the previous frame [0, 0.95]
        # Response curves (LUTs mapping input weight 0..1 -> output 0..1).
        # gain_lut  shapes the mouth-articulation channels; width_lut shapes the
        # mouth-width (stretch) channels. None = identity (no change).
        self.gain_lut: list[float] | None = None
        self.width_lut: list[float] | None = None

        self._prev: list[float] | None = None

    def reset(self) -> None:
        """Drop smoothing history (call when the source / take changes)."""
        self._prev = None

    # ------------------------------------------------------------------

    def process(self, raw: list[float]) -> list[float]:
        """Return a cleaned-up copy of *raw* (length 52)."""
        vals = list(raw)

        lut = self.gain_lut
        if lut:
            for idx in _GAIN_CHANNELS:
                vals[idx] = _sample_lut(lut, vals[idx])

        wlut = self.width_lut
        if wlut:
            for idx in _WIDTH_CHANNELS:
                vals[idx] = _sample_lut(wlut, vals[idx])

        if self.smoothing > 0.0 and self._prev is not None:
            a = self.smoothing
            vals = [(1.0 - a) * v + a * p for v, p in zip(vals, self._prev)]

        # Final clamp to the valid weight range.
        vals = [0.0 if v < 0.0 else 1.0 if v > 1.0 else v for v in vals]
        self._prev = vals
        return vals
