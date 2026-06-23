"""
Per-view pipeline tests (view_pipeline + modifiers).

Each preview view starts from a named *source* stream and runs an ordered stack
of modifiers over it.  These cover config serialisation, legacy migration, and
that the runtime pipeline honours each modifier without a Qt or GL context.
"""

from __future__ import annotations

from ..view_pipeline import ViewConfig, ViewPipeline
from ..modifiers import ModifierConfig
from .. import ai_blendshapes
from ..arkit_names import BLENDSHAPE_NAMES

_IDX = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def _inp(stream="mediapipe"):
    return ModifierConfig("input", params={"stream": stream})


def _streams(raw, ai=None, retarget=None):
    return {"mediapipe": raw, "ai": ai, "retarget": retarget}


def test_default_view_passes_blendshapes_through():
    # A fresh view has only a pose_filter modifier, so blendshapes are unchanged.
    pipe = ViewPipeline(ViewConfig(label="Raw"))
    raw = [0.3] * 52
    out, _ = pipe.process(_streams(list(raw)))
    assert out == raw


def test_default_view_applies_head_pose_axes():
    pipe = ViewPipeline(ViewConfig(label="Raw"))
    _, pose = pipe.process(_streams([0.0] * 52), {"rot": [10, 20, 30], "pos": [1, 2, 3]})
    assert pose["rot"] == [10, 20, 30]
    assert pose["neck_anchor"] == 0.0


def test_config_roundtrip():
    cfg = ViewConfig(label="Out", mesh_path="/x/head.glb", is_output=True, source="ai")
    cfg.modifiers = [
        ModifierConfig("ai", influence=0.75, params={"scope": "mouth", "stream": "ai"}),
        ModifierConfig("smooth", influence=0.4),
        ModifierConfig("pose_filter", params={"rot": [True, False, True],
                                              "pos": [True, True, True],
                                              "neck_anchor": 0.5}),
    ]
    back = ViewConfig.from_dict(cfg.to_dict())
    assert back.to_dict() == cfg.to_dict()


def test_ai_modifier_replaces_scoped_channels():
    cfg = ViewConfig(label="AI")
    # Two input modifiers: first loads mediapipe (base), second overlays ai (mouth only keeps base elsewhere)
    cfg.modifiers = [
        _inp(),
        ModifierConfig("input", influence=1.0, params={"stream": "ai", "scope": ai_blendshapes.SCOPE_ALL}),
    ]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.2] * 52, ai=[0.9] * 52))
    assert abs(out[_IDX["jawOpen"]] - 0.9) < 1e-6
    assert abs(out[_IDX["_neutral"]] - 0.9) < 1e-6  # SCOPE_ALL = full copy


def test_ai_modifier_influence_blends():
    cfg = ViewConfig(label="AI")
    cfg.modifiers = [_inp(), ModifierConfig("ai", influence=0.5,
                                    params={"scope": ai_blendshapes.SCOPE_ALL, "stream": "ai"})]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.0] * 52, ai=[1.0] * 52))
    assert abs(out[_IDX["jawOpen"]] - 0.5) < 1e-6


def test_ai_modifier_noop_without_ai_stream():
    cfg = ViewConfig(label="AI")
    cfg.modifiers = [_inp(), ModifierConfig("ai", influence=1.0,
                                    params={"scope": ai_blendshapes.SCOPE_ALL, "stream": "ai"})]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.2] * 52, ai=None))
    assert out == [0.2] * 52


def test_curves_modifier_changes_mouth_channels():
    cfg = ViewConfig(label="Gain")
    cfg.modifiers = [_inp(), ModifierConfig("curves", params={
        "gain_curve": [[0.0, 0.0], [1.0, 0.0]],  # flatten gain channels to 0
        "width_curve": [[0.0, 0.0], [1.0, 1.0]],
    })]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.5] * 52))
    assert out[_IDX["jawOpen"]] < 0.05
    assert abs(out[_IDX["browInnerUp"]] - 0.5) < 1e-6


def test_smooth_modifier_blends_toward_previous():
    cfg = ViewConfig(label="Smooth")
    cfg.modifiers = [_inp(), ModifierConfig("smooth", influence=0.5)]
    pipe = ViewPipeline(cfg)
    pipe.process(_streams([0.0] * 52))
    out, _ = pipe.process(_streams([1.0] * 52))
    assert abs(out[_IDX["jawOpen"]] - 0.5) < 1e-6


def test_source_stream_selects_input():
    cfg = ViewConfig(label="rt")
    cfg.modifiers = [ModifierConfig("input", params={"stream": "retarget"})]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.0] * 52, retarget=[0.7] * 52))
    assert out == [0.7] * 52


def test_no_modifiers_outputs_zeros():
    cfg = ViewConfig(label="empty")
    cfg.modifiers = []
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.9] * 52))
    assert out == [0.0] * 52


def test_disabled_modifier_is_skipped():
    cfg = ViewConfig(label="x")
    cfg.modifiers = [_inp(), ModifierConfig("ai", enabled=False, influence=1.0,
                                    params={"scope": "all", "stream": "ai"})]
    pipe = ViewPipeline(cfg)
    out, _ = pipe.process(_streams([0.2] * 52, ai=[0.9] * 52))
    assert out == [0.2] * 52


def test_modifier_order_matters_and_apply_config_rebuilds():
    # Default view has InputModifier + PoseFilter -> mediapipe passthrough.
    pipe = ViewPipeline(ViewConfig(label="v"))
    out, _ = pipe.process(_streams([0.4] * 52))
    assert out == [0.4] * 52
    cfg = ViewConfig(label="v")
    cfg.modifiers = [_inp(), ModifierConfig("ai", influence=1.0,
                                    params={"scope": "all", "stream": "ai"})]
    pipe.apply_config(cfg)
    out, _ = pipe.process(_streams([0.4] * 52, ai=[0.1] * 52))
    assert abs(out[_IDX["jawOpen"]] - 0.1) < 1e-6


def test_legacy_config_migrates_to_modifiers():
    legacy = {
        "label": "Out",
        "smoothing": 0.4,
        "ai_scope": ai_blendshapes.SCOPE_MOUTH,
        "ai_mode": ai_blendshapes.MODE_MIX,
        "ai_influence": 0.6,
        "closure_enabled": True,
        "closure_amount": 0.8,
        "pose_axes": {"rot_x": False, "rot_y": True, "rot_z": True,
                      "pos_x": True, "pos_y": True, "pos_z": True},
        "neck_anchor": 0.3,
    }
    cfg = ViewConfig.from_dict(legacy)
    types = [m.type for m in cfg.modifiers]
    assert types == ["input", "input", "smooth", "closure", "pose_filter"]
    ai = cfg.modifiers[1]
    assert ai.influence == 0.6 and ai.params["scope"] == ai_blendshapes.SCOPE_MOUTH
    smooth = cfg.modifiers[2]
    assert smooth.influence == 0.4
    pose = cfg.modifiers[-1]
    assert pose.params["rot"] == [False, True, True]
    assert pose.params["neck_anchor"] == 0.3


def test_closure_modifier_prepare_detects_events():
    # Build a take stream with a sharp jawOpen valley (a closure).
    frames = []
    for i, jaw in enumerate([0.6, 0.6, 0.05, 0.6, 0.6]):
        bs = [0.0] * 52
        bs[_IDX["jawOpen"]] = jaw
        frames.append({"audio_position_ms": i * 16.0, "blendshapes": bs})
    positions = [f["audio_position_ms"] for f in frames]
    cfg = ViewConfig(label="c")
    cfg.modifiers = [ModifierConfig("closure", influence=1.0,
                                    params={"detect_streams": ["mediapipe"],
                                            "drop": 0.1, "open_min": 0.15})]
    pipe = ViewPipeline(cfg)
    pipe.set_streams({"mediapipe": (frames, positions)})
    assert pipe.modifier(0).event_count >= 1


# ---------------------------------------------------------------------------
# process_all: unified export path
# ---------------------------------------------------------------------------

def _make_frames(n: int, value: float = 0.5) -> list[dict]:
    bs = [0.0] * 52
    jaw = _IDX["jawOpen"]
    bs[jaw] = value
    return [{"audio_position_ms": i * 16.0, "blendshapes": list(bs)} for i in range(n)]


def test_process_all_returns_one_output_per_frame():
    from ..data import Stream
    frames = _make_frames(5)
    take_stream = Stream.from_frames(frames)
    pipe = ViewPipeline(ViewConfig(label="out"))
    pipe.config.modifiers = [_inp("mediapipe")]
    pipe.apply_config(pipe.config)
    result = pipe.process_all(take_stream, {})
    assert len(result) == 5
    assert all("audio_position_ms" in r and "blendshapes" in r for r in result)


def test_process_all_preserves_positions():
    from ..data import Stream
    frames = _make_frames(3, value=0.7)
    take_stream = Stream.from_frames(frames)
    pipe = ViewPipeline(ViewConfig(label="out"))
    pipe.config.modifiers = [_inp("mediapipe")]
    pipe.apply_config(pipe.config)
    result = pipe.process_all(take_stream, {})
    for i, r in enumerate(result):
        assert abs(r["audio_position_ms"] - i * 16.0) < 1e-9


def test_process_all_applies_modifier():
    from ..data import Stream
    jaw = _IDX["jawOpen"]
    frames = _make_frames(4, value=0.8)
    ai_bs = [0.0] * 52
    ai_bs[jaw] = 0.2
    ai_frames = [{"audio_position_ms": i * 16.0, "blendshapes": list(ai_bs)} for i in range(4)]
    take_stream = Stream.from_frames(frames)
    ai_stream = Stream.from_frames(ai_frames)

    # Replace mediapipe jawOpen entirely with AI value.
    cfg = ViewConfig(label="out")
    cfg.modifiers = [
        ModifierConfig("input", params={"stream": "mediapipe"}),
        ModifierConfig("input", influence=1.0, params={"stream": "ai",
                                                       "scope": ai_blendshapes.SCOPE_MOUTH}),
    ]
    pipe = ViewPipeline(cfg)
    result = pipe.process_all(take_stream, {"ai": ai_stream})
    # ai fully overrides jaw in mouth scope
    for r in result:
        assert abs(r["blendshapes"][jaw] - 0.2) < 1e-6


def test_process_all_with_empty_named_streams():
    from ..data import Stream
    frames = _make_frames(2, value=0.4)
    take_stream = Stream.from_frames(frames)
    pipe = ViewPipeline(ViewConfig(label="out"))
    pipe.config.modifiers = [_inp("mediapipe")]
    pipe.apply_config(pipe.config)
    result = pipe.process_all(take_stream, {})
    jaw = _IDX["jawOpen"]
    for r in result:
        assert abs(r["blendshapes"][jaw] - 0.4) < 1e-6
