"""
Quad preview + per-view pipeline integration tests (MainWindow level).

These confirm the window builds the 3 preview views, that clicking a view binds
the right-dock controls to *that* view's config, that each view's AI/curve
settings are independent, and that "Bake into Take" produces a new take.

The MeshRenderers can't get a GL context under the offscreen platform, but they
construct without crashing, so the wiring is fully exercisable here.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(tmp_path):
    from synclip.ui.main_window import MainWindow
    return MainWindow(root_dir=str(tmp_path), ipc_port=0)


def _seed_take(w):
    w._streams.set("mediapipe", [
        {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
        {"audio_position_ms": 1000.0, "blendshapes": [1.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
    ])
    w._current_take_id = "t1"
    audio = os.path.join(str(w._root_dir), "seed.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00")
    w._current_audio_path = audio


def test_window_has_three_views_with_one_output(qapp, tmp_path):
    from synclip.ui.main_window import OUTPUT_VIEW
    from synclip.ui.quad_view import MESH_VIEW_COUNT
    w = _make_window(tmp_path)
    try:
        assert len(w._views) == MESH_VIEW_COUNT == 3
        assert sum(1 for c in w._views if c.is_output) == 1
        assert w._views[OUTPUT_VIEW].is_output is True
        assert hasattr(w, "_quad")
    finally:
        w._worker.stop()


def test_selecting_view_binds_dock_widgets(qapp, tmp_path):
    from synclip.modifiers import ModifierConfig
    w = _make_window(tmp_path)
    try:
        # Give view 0 a distinctive modifier stack, then select it.
        w._views[0].modifiers = [ModifierConfig("smooth", influence=0.42)]
        w._pipelines[0].apply_config(w._views[0])
        w._on_view_selected(0)
        assert w._selected_view == 0
        # The modifier-stack dock is bound to view 0's config.
        assert w._modifier_stack._cfg is w._views[0]
        assert "Raw results" in w._editing_label.text()
    finally:
        w._worker.stop()


def test_per_view_modifiers_are_independent(qapp, tmp_path):
    from synclip import ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    RAW_VIEW = 0
    w = _make_window(tmp_path)
    try:
        # Output view replaces-all with AI; raw view stays clean.
        w._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("ai", influence=1.0, params={"scope": ai.SCOPE_ALL, "stream": "ai"})
        ]
        w._pipelines[OUTPUT_VIEW].apply_config(w._views[OUTPUT_VIEW])
        # Same raw + ai inputs -> different per-view results.
        streams = {"mediapipe": [0.2] * 52, "ai": [0.9] * 52}
        out_w, _ = w._pipelines[OUTPUT_VIEW].process(streams)
        raw_w, _ = w._pipelines[RAW_VIEW].process(streams)
        from synclip.arkit_names import BLENDSHAPE_NAMES
        jaw = BLENDSHAPE_NAMES.index("jawOpen")
        assert abs(out_w[jaw] - 0.9) < 1e-6
        assert abs(raw_w[jaw] - 0.2) < 1e-6
    finally:
        w._worker.stop()


def test_drive_previews_returns_output_weights(qapp, tmp_path):
    w = _make_window(tmp_path)
    try:
        out = w._drive_previews([0.33] * 52, None)
        assert out == [0.33] * 52  # default output view passes blendshapes through
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Bars source combo: raw / mix / AI switching
# ---------------------------------------------------------------------------

def test_bars_combo_has_mix_and_mediapipe_by_default(qapp, tmp_path):
    """The combo always has mix + mediapipe; AI option is absent without ai_frames."""
    w = _make_window(tmp_path)
    try:
        datas = [w._bars_source_combo.itemData(i)
                 for i in range(w._bars_source_combo.count())]
        assert "mix" in datas
        assert "mediapipe" in datas
        assert "raw" not in datas
        assert "ai" not in datas
    finally:
        w._worker.stop()


def test_ai_option_appears_when_ai_frames_present(qapp, tmp_path):
    """AI option is added dynamically and removed when ai_frames are cleared."""
    w = _make_window(tmp_path)
    try:
        w._streams.set("ai", [{"audio_position_ms": 0.0, "blendshapes": [0.5] * 52}])
        w._update_bars_source_combo()
        datas = [w._bars_source_combo.itemData(i)
                 for i in range(w._bars_source_combo.count())]
        assert "ai" in datas
        # Clear ai -> option disappears.
        w._streams.clear("ai")
        w._update_bars_source_combo()
        datas = [w._bars_source_combo.itemData(i)
                 for i in range(w._bars_source_combo.count())]
        assert "ai" not in datas
    finally:
        w._worker.stop()


def test_retarget_option_appears_when_retarget_frames_present(qapp, tmp_path):
    """Retarget option appears dynamically and is removed when frames are cleared."""
    w = _make_window(tmp_path)
    try:
        w._streams.set("retarget", [{"audio_position_ms": 0.0, "blendshapes": [0.3] * 52}])
        w._update_bars_source_combo()
        datas = [w._bars_source_combo.itemData(i)
                 for i in range(w._bars_source_combo.count())]
        assert "retarget" in datas
        # Clear -> option disappears.
        w._streams.clear("retarget")
        w._update_bars_source_combo()
        datas = [w._bars_source_combo.itemData(i)
                 for i in range(w._bars_source_combo.count())]
        assert "retarget" not in datas
    finally:
        w._worker.stop()


def test_mix_source_is_read_only_in_review(qapp, tmp_path):
    """Selecting 'Mix' locks the editor even in REVIEW mode."""
    w = _make_window(tmp_path)
    try:
        _seed_take(w)
        w._go_review(from_start=False)
        idx = w._bars_source_combo.findData("mix")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)
        assert w._value_editor._read_only is True
    finally:
        w._worker.stop()


def test_raw_source_is_editable_in_review(qapp, tmp_path):
    """Selecting 'Mediapipe' in REVIEW mode makes the editor writable."""
    w = _make_window(tmp_path)
    try:
        _seed_take(w)
        w._go_review(from_start=False)
        idx = w._bars_source_combo.findData("mediapipe")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)
        assert w._value_editor._read_only is False
    finally:
        w._worker.stop()


def test_ai_source_is_editable_in_review(qapp, tmp_path):
    """Selecting 'AI frames' in REVIEW mode makes the editor writable."""
    w = _make_window(tmp_path)
    try:
        _seed_take(w)
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [0.5] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [0.5] * 52},
        ])
        w._update_bars_source_combo()
        w._go_review(from_start=False)
        idx = w._bars_source_combo.findData("ai")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)
        assert w._value_editor._read_only is False
    finally:
        w._worker.stop()


def test_editor_read_only_in_live_mode(qapp, tmp_path):
    """Outside REVIEW the editor is always read-only regardless of combo."""
    w = _make_window(tmp_path)
    try:
        # LIVE is the default mode; switch combo to raw (normally editable in REVIEW).
        idx = w._bars_source_combo.findData("mediapipe")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)
        assert w._value_editor._read_only is True
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Value editing: mutations land in the right data store
# ---------------------------------------------------------------------------

def _seed_persisted_take(w):
    """Create a take on disk and wire the window to it."""
    from synclip import data as ld
    frames = [
        {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
        {"audio_position_ms": 1000.0, "blendshapes": [0.0] * 52,
         "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
    ]
    audio = os.path.join(str(w._root_dir), "edit.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00")
    ld.append_take(audio, frames, 1000.0)
    data = ld.load_synclip(audio)
    w._current_audio_path = audio
    w._current_synclip = data
    take = data["takes"][0]
    w._current_take_id = take["take_id"]
    w._streams.set("mediapipe", take["frames"])


def test_editing_raw_updates_take_frame_in_memory(qapp, tmp_path, monkeypatch):
    """_on_value_edited with src=raw mutates the take frame blendshapes in-memory."""
    w = _make_window(tmp_path)
    try:
        _seed_persisted_take(w)
        w._go_review(from_start=False)
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 0.0)
        idx = w._bars_source_combo.findData("mediapipe")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)

        w._on_value_edited(5, 0.77)

        assert abs(w._streams.frames("mediapipe")[0]["blendshapes"][5] - 0.77) < 1e-6
    finally:
        w._worker.stop()


def test_editing_raw_persists_to_disk(qapp, tmp_path, monkeypatch):
    """Editing a raw frame value saves the mutated blendshapes to the synclip file."""
    from synclip import data as ld
    w = _make_window(tmp_path)
    try:
        _seed_persisted_take(w)
        w._go_review(from_start=False)
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 0.0)
        idx = w._bars_source_combo.findData("mediapipe")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)

        w._on_value_edited(7, 0.99)
        w._save_current_take()  # flush the debounce timer immediately

        data = ld.load_synclip(w._current_audio_path)
        assert abs(data["takes"][0]["frames"][0]["blendshapes"][7] - 0.99) < 1e-6
    finally:
        w._worker.stop()


def test_editing_ai_frames_updates_in_memory(qapp, tmp_path, monkeypatch):
    """_on_value_edited with src=ai mutates the ai_frames array in-memory."""
    w = _make_window(tmp_path)
    try:
        _seed_persisted_take(w)
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [0.3] * 52},
            {"audio_position_ms": 1000.0, "blendshapes": [0.3] * 52},
        ])
        w._update_bars_source_combo()
        w._go_review(from_start=False)
        monkeypatch.setattr(w._audio, "get_position_ms", lambda: 0.0)
        idx = w._bars_source_combo.findData("ai")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)

        w._on_value_edited(10, 0.88)

        assert abs(w._streams.frames("ai")[0]["blendshapes"][10] - 0.88) < 1e-6
        # The raw take frames must NOT be touched.
        assert abs(w._streams.frames("mediapipe")[0]["blendshapes"][10] - 0.0) < 1e-6
    finally:
        w._worker.stop()


def test_editor_is_read_only_when_mix_selected(qapp, tmp_path):
    """With src=mix the ValueEditor widget is in read-only mode (sliders can't be dragged)."""
    w = _make_window(tmp_path)
    try:
        _seed_persisted_take(w)
        w._go_review(from_start=False)
        idx = w._bars_source_combo.findData("mix")
        w._bars_source_combo.setCurrentIndex(idx)
        w._on_bars_source_changed(idx)
        # Read-only flag set on the widget itself; user interaction cannot emit value_changed.
        assert w._value_editor._read_only is True
        # All individual sliders must also be marked read-only.
        for slider in w._value_editor._sliders.values():
            assert slider._read_only is True
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Bake



def test_export_glb_produces_valid_file(tmp_path):
    """export_glb writes a parseable GLB with one animation and reduced keyframes."""
    import json, struct
    from synclip.export_glb import export_glb, _read_glb, _reduce_keyframes
    from synclip.arkit_names import BLENDSHAPE_NAMES

    # Find head.glb
    here = os.path.dirname(__file__)
    glb_path = os.path.normpath(os.path.join(here, "../../godot/head.glb"))
    if not os.path.isfile(glb_path):
        import pytest; pytest.skip("head.glb not found")

    # Build synthetic frames: 10 frames, only jawOpen moves (all else 0)
    jaw = BLENDSHAPE_NAMES.index("jawOpen")
    frames = []
    for i in range(10):
        bs = [0.0] * 52
        bs[jaw] = i / 9.0
        frames.append({"audio_position_ms": i * 100.0, "blendshapes": bs})

    out = str(tmp_path / "out.glb")
    n_keys = export_glb(glb_path, frames, out)

    # File exists and is a valid GLB
    assert os.path.isfile(out)
    j, bin_data = _read_glb(out)

    # Has exactly one animation
    anims = j.get("animations", [])
    assert len(anims) == 1
    anim = anims[0]
    assert anim["name"] == "synclip"
    assert len(anim["channels"]) == 1
    assert anim["channels"][0]["target"]["path"] == "weights"

    # Keyframe count <= original frame count (some may be reduced)
    assert n_keys <= len(frames)
    assert n_keys >= 2  # first and last always kept

    # weights accessor has correct element count (n_keys * 52 morph targets)
    weights_acc_idx = anim["samplers"][0]["output"]
    weights_acc = j["accessors"][weights_acc_idx]
    n_targets = len(j["meshes"][0]["extras"]["targetNames"])
    assert weights_acc["count"] == n_keys * n_targets

    # The sampler input (time) accessor carries min/max; the weights output
    # accessor must NOT carry the time range as its bounds.
    time_acc = j["accessors"][anim["samplers"][0]["input"]]
    assert "min" in time_acc and "max" in time_acc
    assert "min" not in weights_acc and "max" not in weights_acc

    # Every appended accessor's bufferView must start on a 4-byte boundary.
    for bv in j["bufferViews"]:
        assert bv.get("byteOffset", 0) % 4 == 0


def test_reduce_keyframes_removes_linear_middle():
    from synclip.export_glb import _reduce_keyframes
    # Three frames: middle is exactly on the linear path -> should be removed
    times = [0.0, 0.5, 1.0]
    weights = [[0.0] * 4, [0.5] * 4, [1.0] * 4]
    t2, w2 = _reduce_keyframes(times, weights, tolerance=0.001)
    assert len(t2) == 2
    assert t2 == [0.0, 1.0]


def test_reduce_keyframes_keeps_nonlinear():
    from synclip.export_glb import _reduce_keyframes
    times = [0.0, 0.5, 1.0]
    # Middle frame deviates significantly on channel 0
    weights = [[0.0] * 4, [0.9, 0.5, 0.5, 0.5], [1.0] * 4]
    t2, w2 = _reduce_keyframes(times, weights, tolerance=0.001)
    assert len(t2) == 3


def _build_minimal_glb(target_names, unaligned=False):
    """Construct a minimal but structurally valid GLB with morph targetNames.

    When *unaligned*, the BIN chunk payload is sized so it is NOT a multiple of
    4 before padding, exercising the export alignment path.
    """
    import json, struct
    bin_payload = b"\x01\x02\x03" if unaligned else b""
    j = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{"attributes": {"POSITION": 0}}],
            "extras": {"targetNames": list(target_names)},
        }],
        "buffers": [{"byteLength": len(bin_payload)}],
        "bufferViews": [],
        "accessors": [],
    }
    json_bytes = json.dumps(j).encode()
    json_bytes += b"\x20" * ((4 - len(json_bytes) % 4) % 4)
    chunks = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    if bin_payload:
        padded = bin_payload + b"\x00" * ((4 - len(bin_payload) % 4) % 4)
        chunks += struct.pack("<II", len(padded), 0x004E4942) + padded
    header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunks))
    return header + chunks


def test_export_glb_minimal_synthetic(tmp_path):
    """Exercise export_glb end-to-end without needing the real head.glb."""
    from synclip.export_glb import export_glb, _read_glb
    from synclip.arkit_names import BLENDSHAPE_NAMES

    src = str(tmp_path / "src.glb")
    with open(src, "wb") as fh:
        fh.write(_build_minimal_glb(BLENDSHAPE_NAMES, unaligned=True))

    jaw = BLENDSHAPE_NAMES.index("jawOpen")
    frames = []
    for i in range(8):
        bs = [0.0] * 52
        bs[jaw] = i / 7.0
        frames.append({"audio_position_ms": i * 100.0, "blendshapes": bs})

    out = str(tmp_path / "out.glb")
    n_keys = export_glb(src, frames, out)

    j, _bin = _read_glb(out)
    anim = j["animations"][0]
    time_acc = j["accessors"][anim["samplers"][0]["input"]]
    weights_acc = j["accessors"][anim["samplers"][0]["output"]]

    # min/max only on the time accessor.
    assert "min" in time_acc and "max" in time_acc
    assert "min" not in weights_acc and "max" not in weights_acc
    # Weight element count matches.
    assert weights_acc["count"] == n_keys * 52
    # All appended bufferViews are 4-byte aligned even though the source BIN
    # chunk payload was deliberately unaligned.
    for bv in j["bufferViews"]:
        assert bv.get("byteOffset", 0) % 4 == 0
