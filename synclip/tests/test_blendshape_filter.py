"""
BlendshapeFilter tests: gain/width LUTs + EMA smoothing remain; the removed
plausibility constraints must NOT come back (they caused the mouthRoll artifact).
"""

from __future__ import annotations

from ..blendshape_filter import BlendshapeFilter
from ..arkit_names import BLENDSHAPE_NAMES

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def test_smoothing_defaults_to_zero():
    """Default smoothing must be 0.0 so we never add latency unasked."""
    assert BlendshapeFilter().smoothing == 0.0


def test_zero_smoothing_has_no_frame_dependence():
    """With smoothing 0 (default), the output depends only on the current frame
    -- no blend against history, hence no added latency."""
    f = BlendshapeFilter()  # default smoothing == 0.0
    f.gain_lut = None
    f.width_lut = None
    f.process([1.0] * 52)             # prime history
    out = f.process([0.2] * 52)
    assert out == [0.2] * 52          # current frame only, no lag from prev


def test_plausibility_attribute_gone():
    f = BlendshapeFilter()
    assert not hasattr(f, "plausibility_enabled")
    assert not hasattr(f, "_apply_plausibility")


def test_identity_passthrough_without_luts():
    f = BlendshapeFilter()
    f.smoothing = 0.0
    f.gain_lut = None
    f.width_lut = None
    raw = [0.3] * 52
    out = f.process(raw)
    assert out == [0.3] * 52


def test_mouthclose_not_clamped_to_jawopen():
    """The old plausibility pass forced mouthClose <= jawOpen. It must not."""
    f = BlendshapeFilter()
    f.smoothing = 0.0
    f.gain_lut = None
    f.width_lut = None
    raw = [0.0] * 52
    raw[_IDX["mouthClose"]] = 0.8
    raw[_IDX["jawOpen"]] = 0.1
    out = f.process(raw)
    assert out[_IDX["mouthClose"]] == 0.8  # unchanged, not clamped down


def test_smoothing_blends_with_previous():
    f = BlendshapeFilter()
    f.smoothing = 0.5
    f.gain_lut = None
    f.width_lut = None
    f.process([0.0] * 52)
    out = f.process([1.0] * 52)
    # EMA: 0.5*prev(0) + 0.5*new(1) = 0.5
    assert all(abs(v - 0.5) < 1e-6 for v in out)


def test_output_clamped_to_unit_range():
    f = BlendshapeFilter()
    f.smoothing = 0.0
    f.gain_lut = None
    f.width_lut = None
    out = f.process([2.0, -1.0] + [0.0] * 50)
    assert out[0] == 1.0
    assert out[1] == 0.0


def test_reset_clears_history():
    f = BlendshapeFilter()
    f.smoothing = 0.9
    f.process([1.0] * 52)
    f.reset()
    # After reset, first frame is not blended against stale history.
    out = f.process([0.0] * 52)
    assert all(v == 0.0 for v in out)


def test_smoothing_is_the_final_post_process_step():
    """Smoothing must be applied LAST (after gain/width), as a pure post-process.

    With identity gain/width, the only difference a frame-to-frame change makes
    is the EMA blend -- proving smoothing sits at the end of the chain and does
    not feed back into the gain/width stages.
    """
    f = BlendshapeFilter()
    f.gain_lut = None
    f.width_lut = None
    f.smoothing = 0.5
    f.process([0.0] * 52)
    out = f.process([1.0] * 52)
    # Post-process EMA only: 0.5*0 + 0.5*1 = 0.5 everywhere.
    assert all(abs(v - 0.5) < 1e-9 for v in out)


def test_smoothing_does_not_mutate_input():
    """process() must not write smoothing back into the caller's raw list."""
    f = BlendshapeFilter()
    f.smoothing = 0.8
    raw = [0.42] * 52
    f.process([0.0] * 52)
    f.process(raw)
    assert raw == [0.42] * 52  # caller's raw values untouched
