"""
Tests for the generic modifier framework (modifiers module).

These exercise the registry, the self-describing param schema, the wet/dry
influence semantics and each concrete modifier -- all without Qt or GL.
"""

from __future__ import annotations

from .. import modifiers as M
from ..modifiers import ModifierConfig, ModifierContext
from .. import ai_blendshapes as ai
from ..arkit_names import BLENDSHAPE_NAMES

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def _ctx(streams, pos_ms=0.0):
    return ModifierContext(streams=streams, pos_ms=pos_ms)


def test_registry_lists_builtin_types():
    names = [t for t, _ in M.available_types()]
    for expected in ("curves", "smooth", "closure", "pose_filter"):
        assert expected in names


def test_create_unknown_type_is_passthrough():
    m = M.create(ModifierConfig(type="does_not_exist"))
    vals = [0.3] * 52
    out, pose = m.apply(vals, {"rot": [1, 2, 3]}, _ctx({}))
    assert out == vals


def test_default_config_fills_param_defaults():
    cfg = M.ClosureModifier.default_config()
    assert cfg.type == "closure"
    assert cfg.params["drop"] == 0.05
    assert cfg.params["detect_streams"] == ["mediapipe", "ai"]


def test_smooth_modifier_is_ema_of_influence():
    m = M.create(ModifierConfig("smooth", influence=0.5))
    out1, _ = m.apply([0.0] * 52, None, _ctx({}))
    out2, _ = m.apply([1.0] * 52, None, _ctx({}))
    assert abs(out2[0] - 0.5) < 1e-9          # halfway between prev (0) and cur (1)
    out3, _ = m.apply([1.0] * 52, None, _ctx({}))
    assert abs(out3[0] - 0.75) < 1e-9         # EMA converges toward 1


def test_smooth_modifier_reset_clears_state():
    m = M.create(ModifierConfig("smooth", influence=0.5))
    m.apply([1.0] * 52, None, _ctx({}))
    m.reset()
    out, _ = m.apply([1.0] * 52, None, _ctx({}))
    assert out[0] == 1.0                       # no previous -> identity


def test_pose_filter_zeroes_axes_and_emits_neck_anchor():
    m = M.create(ModifierConfig("pose_filter", params={
        "rot": [True, False, True], "pos": [False, True, True], "neck_anchor": 0.4}))
    vals = [0.2] * 52
    out, pose = m.apply(vals, {"rot": [10, 20, 30], "pos": [1, 2, 3]}, _ctx({}))
    assert out == vals                          # pose modifier leaves blendshapes
    assert pose["rot"] == [10, 0, 30]
    assert pose["pos"] == [0, 2, 3]
    assert pose["neck_anchor"] == 0.4


def test_pose_filter_has_no_influence_knob():
    assert M.PoseFilterModifier.has_influence is False


def test_disabled_modifier_via_pipeline_semantics():
    # The base apply() always runs; "enabled" is honoured by the pipeline, but
    # a muted config still round-trips its params.
    cfg = ModifierConfig("smooth", enabled=False, influence=0.9)
    assert ModifierConfig.from_dict(cfg.to_dict()) == cfg


def test_closure_prepare_detects_and_enforces():
    frames = []
    for i, jaw in enumerate([0.6, 0.6, 0.05, 0.6, 0.6]):
        bs = [0.0] * 52
        bs[_IDX["jawOpen"]] = jaw
        frames.append({"audio_position_ms": i * 16.0, "blendshapes": bs})
    positions = [f["audio_position_ms"] for f in frames]
    m = M.create(ModifierConfig("closure", influence=1.0, params={
        "detect_streams": ["mediapipe"], "drop": 0.1, "open_min": 0.15}))
    m.prepare({"mediapipe": (frames, positions)})
    assert m.event_count >= 1
    # At the valley centre an open jaw gets pulled closed.
    raw = [0.0] * 52
    raw[_IDX["jawOpen"]] = 0.8
    out, _ = m.apply(raw, None, _ctx({}, pos_ms=32.0))
    assert out[_IDX["jawOpen"]] < 0.2


def test_closure_status_text():
    frames = []
    for i, jaw in enumerate([0.6, 0.6, 0.05, 0.6, 0.6]):
        bs = [0.0] * 52
        bs[_IDX["jawOpen"]] = jaw
        frames.append({"audio_position_ms": i * 16.0, "blendshapes": bs})
    positions = [f["audio_position_ms"] for f in frames]
    m = M.create(ModifierConfig("closure", influence=1.0, params={
        "detect_streams": ["mediapipe"], "drop": 0.1, "open_min": 0.15}))
    assert m.status_text() == "Closures detected: 0"
    m.prepare({"mediapipe": (frames, positions)})
    assert m.status_text() == f"Closures detected: {m.event_count}"
    assert m.event_count >= 1


def test_base_modifier_status_text_empty():
    m = M.create(ModifierConfig("smooth", influence=0.5))
    assert m.status_text() == ""


# ---------------------------------------------------------------------------
# Animated (time-varying) influence
# ---------------------------------------------------------------------------

def test_sample_curve_is_smooth_and_unclamped():
    from ..curve_lut import sample_curve
    pts = [[0.0, -1.0], [0.5, 1.0], [1.0, 0.0]]
    assert sample_curve(pts, 0.0) == -1.0          # passes through control points
    assert sample_curve(pts, 0.5) == 1.0           # peak (non-monotone, no clamp)
    assert sample_curve(pts, 1.0) == 0.0
    assert sample_curve(pts, 2.0) == 0.0           # past the end holds last y
    assert sample_curve([], 0.5) == 0.0
    # Negative y is preserved (no clamp), unlike the response-curve LUT.
    assert sample_curve(pts, 0.1) < 0.0


def test_sample_curve_two_points_is_linear():
    from ..curve_lut import sample_curve
    pts = [[0.0, 0.0], [1.0, 1.0]]
    assert abs(sample_curve(pts, 0.25) - 0.25) < 1e-9   # spline reduces to a line
    assert abs(sample_curve(pts, 0.5) - 0.5) < 1e-9


def test_effective_influence_off_returns_static():
    m = M.create(ModifierConfig("input", influence=0.7))
    assert m.effective_influence(ModifierContext({}, pos_ms=500.0, duration_ms=1000.0)) == 0.7


def test_effective_influence_zero_duration_returns_static():
    cfg = ModifierConfig("input", influence=0.4, influence_anim="absolute",
                         influence_curve=[[0.0, 1.0], [1.0, 1.0]])
    m = M.create(cfg)
    # No clip length (e.g. LIVE) -> animation inert, base used.
    assert m.effective_influence(ModifierContext({}, pos_ms=0.0, duration_ms=0.0)) == 0.4


def test_effective_influence_relative_punches_up_and_down():
    # base 0.5; offset curve goes +0.5 at start (punch up to 1.0) and -0.5 at end.
    cfg = ModifierConfig("input", influence=0.5, influence_anim="relative",
                         influence_curve=[[0.0, 0.5], [1.0, -0.5]])
    m = M.create(cfg)
    assert abs(m.effective_influence(ModifierContext({}, 0.0, 1000.0)) - 1.0) < 1e-9
    assert abs(m.effective_influence(ModifierContext({}, 500.0, 1000.0)) - 0.5) < 1e-9
    assert abs(m.effective_influence(ModifierContext({}, 1000.0, 1000.0)) - 0.0) < 1e-9


def test_effective_influence_relative_clamps_to_unit():
    # base 0.8 + offset 0.8 would be 1.6 -> clamped to 1.0 (no overshoot).
    cfg = ModifierConfig("input", influence=0.8, influence_anim="relative",
                         influence_curve=[[0.0, 0.8], [1.0, 0.8]])
    m = M.create(cfg)
    assert m.effective_influence(ModifierContext({}, 0.0, 1000.0)) == 1.0


def test_effective_influence_absolute_ignores_base():
    cfg = ModifierConfig("input", influence=0.9, influence_anim="absolute",
                         influence_curve=[[0.0, 0.0], [1.0, 1.0]])
    m = M.create(cfg)
    assert abs(m.effective_influence(ModifierContext({}, 250.0, 1000.0)) - 0.25) < 1e-9


def test_animated_influence_drives_input_modifier_blend():
    # InputModifier mixing an 'ai' stream, absolute influence ramp 0 -> 1.
    cfg = ModifierConfig("input", influence=1.0, influence_anim="absolute",
                         influence_curve=[[0.0, 0.0], [1.0, 1.0]],
                         params={"stream": "ai", "scope": ai.SCOPE_ALL})
    m = M.create(cfg)
    streams = {"ai": [0.8] * 52}
    out0, _ = m.apply([0.2] * 52, None, ModifierContext(streams, 0.0, 1000.0))
    outm, _ = m.apply([0.2] * 52, None, ModifierContext(streams, 500.0, 1000.0))
    assert abs(out0[_IDX["jawOpen"]] - 0.2) < 1e-9   # influence 0 -> base
    assert abs(outm[_IDX["jawOpen"]] - 0.5) < 1e-9   # influence 0.5 -> halfway


def test_influence_curve_round_trips_through_dict():
    cfg = ModifierConfig("input", influence=0.5, influence_anim="relative",
                         influence_curve=[[0.0, 0.2], [1.0, -0.3]])
    assert ModifierConfig.from_dict(cfg.to_dict()) == cfg
