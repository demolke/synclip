"""
Tests for the bottom-pane bars source selector.

Verifies that switching the source combo causes the ValueEditor to show:
  - "mediapipe" -> the raw MediaPipe blendshape values
  - "ai"        -> the AI-generated blendshape values
  - "mix"       -> the processed/mixed output of the active pipeline
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")
pytest.importorskip("mediapipe")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _select_source(w, key: str) -> None:
    combo = w._bars_source_combo
    idx = combo.findData(key)
    assert idx != -1, f"Source {key!r} not found in combo"
    combo.setCurrentIndex(idx)


def _bar_values(w) -> list[float]:
    # Index 0 is the neutral channel and is intentionally skipped by ValueEditor.
    return list(w._value_editor.values())[1:]


def _make_raw() -> list[float]:
    return [0.1 * (i % 10) for i in range(52)]


def _make_ai() -> list[float]:
    return [0.5] * 52


# ---------------------------------------------------------------------------


def test_combo_has_mediapipe_not_raw(qapp, tmp_path):
    """The combo should label the camera source 'Mediapipe', not 'Raw'."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        combo = w._bars_source_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        assert any("Mediapipe" in t for t in labels), f"Labels: {labels}"
        assert not any("Raw" in t for t in labels), f"Labels: {labels}"
        assert any("Mix" in t for t in labels), f"Labels: {labels}"
    finally:
        w._worker.stop()


def test_mediapipe_source_shows_raw_values(qapp, tmp_path):
    """Bars in Mediapipe mode must show the unprocessed camera values."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        raw = _make_raw()
        ai_vals = _make_ai()

        # Inject values through _drive_previews with Mediapipe selected.
        _select_source(w, "mediapipe")
        w._drive_previews(raw, ai_vals)

        bars = _bar_values(w)
        assert bars == pytest.approx(raw[1:], abs=1e-4), \
            "Mediapipe source must show raw camera values"
    finally:
        w._worker.stop()


def test_ai_source_shows_ai_values(qapp, tmp_path):
    """Bars in AI mode must show the AI-generated blendshape values."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Inject fake AI frames so the "AI frames" combo item is added.
        ai_frames = [{"audio_position_ms": 0.0, "blendshapes": _make_ai()}]
        w._streams.set("ai", ai_frames)
        w._update_bars_source_combo()

        raw = _make_raw()
        ai_vals = _make_ai()

        _select_source(w, "ai")
        w._drive_previews(raw, ai_vals)

        bars = _bar_values(w)
        assert bars == pytest.approx(ai_vals[1:], abs=1e-4), \
            "AI source must show AI-generated values"
    finally:
        w._worker.stop()


def test_mix_source_shows_pipeline_output(qapp, tmp_path):
    """Bars in Mix mode must show the processed pipeline output."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        raw = [0.3] * 52
        ai_vals = None

        _select_source(w, "mix")
        out = w._drive_previews(raw, ai_vals)

        bars = _bar_values(w)
        # bars must equal the pipeline output (whatever the pipeline produces)
        assert bars == pytest.approx(out[1:], abs=1e-4), \
            "Mix source must show the pipeline output values"
    finally:
        w._worker.stop()


def _write_synclip_with_ai_frames(audio_path: str, ai_values: list[float]) -> None:
    """Write a minimal .synclip.json next to *audio_path* with one AI frame."""
    import json, pathlib
    synclip = {
        "schema_version": "1.0",
        "takes": [],
        "default_take": None,
        "ai_frames": [{"audio_position_ms": 0.0, "blendshapes": ai_values}],
    }
    p = pathlib.Path(audio_path)
    out = p.parent / (p.stem + ".synclip.json")
    out.write_text(json.dumps(synclip))


def test_ai_frames_restored_from_json_shown_in_bars(qapp, tmp_path):
    """AI frames persisted in synclip JSON must appear in bars when source is 'ai'."""
    from synclip.ui.main_window import MainWindow

    # Write a fake audio file and a synclip JSON with AI frames.
    audio_path = str(tmp_path / "test.ogg")
    pathlib.Path(audio_path).touch()
    ai_values = [float(i % 10) / 10.0 for i in range(52)]
    _write_synclip_with_ai_frames(audio_path, ai_values)

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Simulate loading the audio file - this calls _try_restore_ai_frames.
        w._try_restore_ai_frames(audio_path)

        # AI frames combo item must have been added.
        combo = w._bars_source_combo
        datas = [combo.itemData(i) for i in range(combo.count())]
        assert "ai" in datas, f"AI source not added after restore; datas={datas}"

        _select_source(w, "ai")
        raw = _make_raw()
        # _streams.sample("ai", 0.0) should interpolate to ai_values.
        ai_vals = w._streams.sample("ai", 0.0)
        assert ai_vals is not None, "_streams.sample('ai') returned None after restore"
        w._drive_previews(raw, ai_vals)

        bars = _bar_values(w)
        assert bars == pytest.approx(ai_values[1:], abs=1e-4), \
            "Bars must show restored AI frame values when source='ai'"
    finally:
        w._worker.stop()


import pathlib


def test_ai_frames_are_editable_and_routed_to_ai_array(qapp, tmp_path):
    """Editing a slider with the AI source selected must write into the AI
    frames array (indexed on the AI timeline), not silently no-op."""
    from synclip.ui.main_window import MainWindow

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Two AI frames on their own timeline (distinct length from any take).
        w._streams.set("ai", [
            {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52},
            {"audio_position_ms": 100.0, "blendshapes": [0.0] * 52},
        ])
        w._current_audio_path = str(tmp_path / "clip.ogg")
        w._update_bars_source_combo()
        _select_source(w, "ai")

        # Edit jawOpen (index 25) at position 0 -> nearest AI frame is frame 0.
        w._on_value_edited(25, 0.8)

        assert w._streams.frames("ai")[0]["blendshapes"][25] == pytest.approx(0.8), \
            "Editing with AI source must update the AI frame value"
    finally:
        w._worker.stop()


def test_ai_edit_does_not_touch_take_frames(qapp, tmp_path):
    """An AI-source edit must not write into the take frames."""
    from synclip.ui.main_window import MainWindow

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        w._streams.set("mediapipe", [
            {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52,
             "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
        ])
        w._streams.set("ai", [{"audio_position_ms": 0.0, "blendshapes": [0.0] * 52}])
        w._current_audio_path = str(tmp_path / "clip.ogg")
        w._update_bars_source_combo()
        _select_source(w, "ai")

        w._on_value_edited(25, 0.7)

        assert w._streams.frames("ai")[0]["blendshapes"][25] == pytest.approx(0.7)
        assert w._streams.frames("mediapipe")[0]["blendshapes"][25] == pytest.approx(0.0), \
            "AI edit must not modify take frames"
    finally:
        w._worker.stop()


def test_closure_enforcement_closes_mouth_in_output(qapp, tmp_path):
    """With closure enforcement on, a detected rapid closure must pull jawOpen
    down in the mix output at the valley position."""
    from synclip.mouth_closure import JAW_OPEN

    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Take with a 1-frame jawOpen valley at frame 3 (100ms apart).
        jaw = [0.6, 0.6, 0.6, 0.0, 0.6, 0.6]
        frames = []
        for i, j in enumerate(jaw):
            bs = [0.0] * 52
            bs[JAW_OPEN] = j
            frames.append({"audio_position_ms": i * 100.0, "blendshapes": bs,
                           "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}})
        w._streams.set("mediapipe", frames)

        from synclip.modifiers import ModifierConfig
        view = w._selected_view
        cfg = w._views[view]
        cfg.modifiers = [
            ModifierConfig("input", params={"stream": "mediapipe"}),
            ModifierConfig("closure", influence=0.9, params={
                "detect_streams": ["mediapipe"], "drop": 0.1, "open_min": 0.15})]
        w._pipelines[view].apply_config(cfg)
        w._update_pipeline_streams()
        assert w._pipelines[view].modifier(1).event_count == 1, "closure should be detected"

        # Drive at a position just inside the valley window (frame 3 = 300ms)
        # where the take's own jawOpen is 0.0 but the *enforcement* is what we
        # assert: feed a raw frame that is wide open and confirm it's closed.
        raw = [0.0] * 52
        raw[JAW_OPEN] = 0.8
        w._output_index = view
        out = w._drive_previews(raw, None, None, pos_ms=300.0)
        assert out[JAW_OPEN] < 0.2, \
            f"closure enforcement should close the mouth (got {out[JAW_OPEN]})"

        # With enforcement muted, the same raw stays open.
        cfg.modifiers[1].enabled = False
        w._pipelines[view].apply_config(cfg)
        out2 = w._drive_previews(raw, None, None, pos_ms=300.0)
        assert out2[JAW_OPEN] == pytest.approx(0.8, abs=1e-4)
    finally:
        w._worker.stop()


def test_disabling_pose_axes_zeroes_preview_pose(qapp, tmp_path):
    """Disabling a view's rot/pos axes must zero those axes in the 3D preview
    (the pose handed to the quad renderer), not only in the broadcast."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        from synclip.modifiers import ModifierConfig
        view = w._selected_view
        cfg = w._views[view]
        # Disable all rotation axes; keep position (pose_filter modifier).
        cfg.modifiers = [ModifierConfig("pose_filter", params={
            "rot": [False, False, False], "pos": [True, True, True],
            "neck_anchor": 0.0})]
        w._pipelines[view].apply_config(cfg)

        captured = {}
        orig = w._quad.set_head_pose

        def spy(index, pose):
            if index == view:
                captured["pose"] = pose
            return orig(index, pose)

        w._quad.set_head_pose = spy
        pose = {"rot": [10.0, 20.0, 30.0], "pos": [1.0, 2.0, 3.0]}
        w._drive_previews([0.0] * 52, None, pose, pos_ms=0.0)

        assert captured["pose"]["rot"] == [0.0, 0.0, 0.0], \
            "disabled rotation axes must be zeroed in the preview pose"
        assert captured["pose"]["pos"] == [1.0, 2.0, 3.0], \
            "enabled position axes must pass through"
    finally:
        w._worker.stop()


def test_unchecking_pose_axis_resets_it_immediately_when_idle(qapp, tmp_path):
    """Unchecking a pose axis must zero it in the preview right away, even with
    no playback running (not just keep the last value)."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        from synclip.modifiers import ModifierConfig
        view = w._selected_view
        # Seed a visible head pose as the "last" pose; no playback active.
        w._last_head_pose = {"rot": [10.0, 20.0, 30.0], "pos": [1.0, 2.0, 3.0]}

        captured = {}
        orig = w._quad.set_head_pose

        def spy(index, pose):
            if index == view:
                captured["pose"] = pose
            return orig(index, pose)

        w._quad.set_head_pose = spy

        # Disable rot_y on the selected view's pose-filter modifier and apply.
        w._views[view].modifiers = [ModifierConfig("pose_filter", params={
            "rot": [True, False, True], "pos": [True, True, True],
            "neck_anchor": 0.0})]
        w._on_modifier_stack_changed()

        assert captured["pose"] is not None
        assert captured["pose"]["rot"][1] == 0.0, \
            "unchecked axis must be reset to zero immediately"
        # Other axes untouched.
        assert captured["pose"]["rot"][0] == 10.0
        assert captured["pose"]["pos"] == [1.0, 2.0, 3.0]
    finally:
        w._worker.stop()


def test_switching_source_updates_bars(qapp, tmp_path):
    """Switching the combo live refreshes the bars immediately."""
    from synclip.ui.main_window import MainWindow
    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        raw = [0.2] * 52
        ai_frames = [{"audio_position_ms": 0.0, "blendshapes": [0.8] * 52}]
        w._streams.set("ai", ai_frames)
        w._update_bars_source_combo()

        # Prime last-raw so _refresh_all_previews has something to show.
        import numpy as np
        w._last_raw_blendshapes = np.array(raw, dtype="float32")

        _select_source(w, "mediapipe")
        bars_mp = _bar_values(w)

        _select_source(w, "ai")
        bars_ai = _bar_values(w)

        assert bars_mp != pytest.approx(bars_ai, abs=1e-4), \
            "Switching source must change the displayed values"
        assert bars_mp == pytest.approx(raw[1:], abs=1e-4)
        assert bars_ai == pytest.approx([0.8] * 51, abs=1e-4)
    finally:
        w._worker.stop()


def test_retarget_edit_writes_retarget_frames(qapp, tmp_path):
    """Editing with retarget source selected must write into _retarget_frames."""
    from synclip.ui.main_window import MainWindow

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        w._streams.set("retarget", [
            {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52},
            {"audio_position_ms": 100.0, "blendshapes": [0.0] * 52},
        ])
        w._current_audio_path = str(tmp_path / "clip.ogg")
        w._update_bars_source_combo()
        _select_source(w, "retarget")

        w._on_value_edited(25, 0.6)

        assert w._streams.frames("retarget")[0]["blendshapes"][25] == pytest.approx(0.6), \
            "Editing with retarget source must update the retarget frame"
    finally:
        w._worker.stop()


def test_retarget_edit_does_not_touch_take_frames(qapp, tmp_path):
    """A retarget-source edit must not modify take frames."""
    from synclip.ui.main_window import MainWindow

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        w._streams.set("mediapipe", [
            {"audio_position_ms": 0.0, "blendshapes": [0.0] * 52,
             "head_pose": {"rot": [0.0] * 3, "pos": [0.0] * 3}},
        ])
        w._streams.set("retarget", [{"audio_position_ms": 0.0, "blendshapes": [0.0] * 52}])
        w._current_audio_path = str(tmp_path / "clip.ogg")
        w._update_bars_source_combo()
        _select_source(w, "retarget")

        w._on_value_edited(25, 0.5)

        assert w._streams.frames("retarget")[0]["blendshapes"][25] == pytest.approx(0.5)
        assert w._streams.frames("mediapipe")[0]["blendshapes"][25] == pytest.approx(0.0), \
            "Retarget edit must not modify take frames"
    finally:
        w._worker.stop()


def test_save_timer_flushed_on_close(qapp, tmp_path):
    """Pending take edits must be written to disk before the window closes."""
    import json
    from synclip.ui.main_window import MainWindow
    from synclip import data as ld

    audio_path = str(tmp_path / "clip.ogg")
    pathlib.Path(audio_path).touch()

    w = MainWindow(root_dir=str(tmp_path), ipc_port=0)
    try:
        # Seed a minimal synclip document with one take.
        bs = [0.0] * 52
        seed_frames = [{"audio_position_ms": 0.0, "blendshapes": list(bs)}]
        w._current_audio_path = audio_path
        from synclip.data import append_take
        append_take(audio_path, seed_frames, audio_duration_ms=1000.0)
        # Load via _set_current_take_by_id so _current_take_frames shares the
        # same list object as _current_synclip (the normal runtime invariant).
        w._current_synclip = ld.load_synclip(audio_path)
        take_id = w._current_synclip["takes"][0]["take_id"]
        w._set_current_take_by_id(take_id)
        assert w._streams.has("mediapipe"), "take frames should be set"

        # Dirty the take and start the debounce timer (but don't let it fire).
        w._streams.frames("mediapipe")[0]["blendshapes"][25] = 0.9
        w._schedule_take_save()
        assert w._save_timer.isActive(), "timer should be running after schedule"

        # closeEvent should flush the pending save.
        from PySide6.QtGui import QCloseEvent
        w.closeEvent(QCloseEvent())

        assert not w._save_timer.isActive(), "timer must be stopped after close"
        saved = ld.load_synclip(audio_path)
        assert saved is not None
        saved_frames = saved["takes"][0]["frames"]
        assert saved_frames[0]["blendshapes"][25] == pytest.approx(0.9), \
            "pending edit must be flushed to disk on close"
    finally:
        w._worker.stop()
