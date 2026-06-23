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
