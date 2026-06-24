"""
Shared mouth-channel definitions and LUT sampling for the modifier pipeline.

MediaPipe FaceLandmarker emits all 52 ARKit channels independently and
systematically *under-drives* the mouth-articulation channels used for speech
while *over-firing* the mouth-width channels. The response-curve modifiers
(``CurvesModifier``) reshape those channels with a baked 0..1 -> 0..1 LUT; this
module just names which channels each curve targets and provides the LUT lookup,
so the engine and the UI share one definition.

(The former stateful ``BlendshapeFilter`` class lived here too; it was superseded
by the generic ``CurvesModifier`` + ``SmoothModifier`` in the modifier pipeline.)
"""

from __future__ import annotations

from .arkit_names import BLENDSHAPE_NAMES

# Name -> index lookup so the channel lists below read by channel name.
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


# Channels boosted by the mouth-articulation gain curve (the ones MediaPipe
# under-drives for speech). Emotional shapes (smile/frown) are left alone.
_GAIN_CHANNELS = [
    _i("jawOpen"), _i("mouthClose"), _i("mouthFunnel"), _i("mouthPucker"),
    _i("mouthLowerDownLeft"), _i("mouthLowerDownRight"),
    _i("mouthUpperUpLeft"), _i("mouthUpperUpRight"),
    _i("mouthShrugLower"), _i("mouthShrugUpper"),
    _i("mouthRollLower"), _i("mouthRollUpper"),
]

# Mouth-width channels. MediaPipe tends to over-fire these, making vowels look
# like a wide grimace instead of a round shape; the width curve tames them.
_WIDTH_CHANNELS = [_i("mouthStretchLeft"), _i("mouthStretchRight")]
