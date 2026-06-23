"""
Tests for ui.config persistence (save + restore of per-project UI settings).

Covers:
  - Saving and loading the raw ui_config module (unit level).
  - MainWindow writing smoothing / AI scope / view label into ui.config.
  - A second MainWindow in the same directory restoring those values.
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


def _make_window(tmp_path):
    from synclip.ui.main_window import MainWindow
    return MainWindow(root_dir=str(tmp_path), ipc_port=0)


# ---------------------------------------------------------------------------
# Unit-level: ui_config module
# ---------------------------------------------------------------------------

def test_ui_config_roundtrip(tmp_path):
    from synclip import ui_config as cfg
    data = {"video_source": "clip.mp4", "views": [{"smoothing": 0.42}]}
    cfg.save(str(tmp_path), data)
    loaded = cfg.load(str(tmp_path))
    assert loaded["video_source"] == "clip.mp4"
    assert loaded["views"][0]["smoothing"] == 0.42


def test_ui_config_returns_empty_dict_when_missing(tmp_path):
    from synclip import ui_config as cfg
    result = cfg.load(str(tmp_path / "nonexistent"))
    assert result == {}


def test_ui_config_returns_empty_dict_on_corrupt_file(tmp_path):
    from synclip import ui_config as cfg
    path = os.path.join(str(tmp_path), "ui.config")
    with open(path, "w") as f:
        f.write("not json {{{")
    result = cfg.load(str(tmp_path))
    assert result == {}


# ---------------------------------------------------------------------------
# Integration: MainWindow saves settings into ui.config
# ---------------------------------------------------------------------------

def test_save_ui_config_writes_modifier_stack(qapp, tmp_path):
    """A smooth modifier added to a view's stack is written to ui.config."""
    from synclip import ui_config as cfg
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        w._on_view_selected(OUTPUT_VIEW)
        w._views[OUTPUT_VIEW].modifiers = [ModifierConfig("smooth", influence=0.37)]
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        assert "views" in data
        mods = data["views"][OUTPUT_VIEW]["modifiers"]
        assert mods[0]["type"] == "smooth"
        assert abs(mods[0]["influence"] - 0.37) < 1e-3
    finally:
        w._worker.stop()


def test_save_ui_config_writes_ai_modifier_scope(qapp, tmp_path):
    """An AI modifier's scope param is written to ui.config."""
    from synclip import ui_config as cfg, ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        w._on_view_selected(OUTPUT_VIEW)
        w._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("input", influence=0.5, params={"stream": "ai", "scope": ai.SCOPE_MOUTH})
        ]
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        mods = data["views"][OUTPUT_VIEW]["modifiers"]
        assert mods[0]["type"] == "input"
        assert mods[0]["params"]["scope"] == ai.SCOPE_MOUTH
    finally:
        w._worker.stop()


def test_save_ui_config_writes_view_label(qapp, tmp_path):
    """Renaming a view persists its label to ui.config."""
    from synclip import ui_config as cfg
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        w._on_view_selected(OUTPUT_VIEW)
        w._views[OUTPUT_VIEW].label = "My Custom View"
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        view_data = data["views"][OUTPUT_VIEW]
        assert view_data.get("label") == "My Custom View"
    finally:
        w._worker.stop()


def test_save_ui_config_writes_audio_path(qapp, tmp_path):
    """Loading an audio file records its path in ui.config."""
    from synclip import ui_config as cfg
    audio = str(tmp_path / "track.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00")
    w = _make_window(tmp_path)
    try:
        w._load_audio_file(audio)
        data = cfg.load(str(tmp_path))
        assert data.get("audio_path") == audio
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Integration: second MainWindow in same dir restores settings
# ---------------------------------------------------------------------------

def test_restore_ui_config_modifier_stack(qapp, tmp_path):
    """A second window restores the saved modifier stack (smooth influence)."""
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    w1 = _make_window(tmp_path)
    try:
        w1._views[OUTPUT_VIEW].modifiers = [ModifierConfig("smooth", influence=0.55)]
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        mods = w2._views[OUTPUT_VIEW].modifiers
        assert mods[0].type == "smooth"
        assert abs(mods[0].influence - 0.55) < 1e-3
    finally:
        w2._worker.stop()


def test_restore_ui_config_ai_modifier_scope(qapp, tmp_path):
    """A second window restores the AI modifier scope that was saved."""
    from synclip import ai_blendshapes as ai
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    w1 = _make_window(tmp_path)
    try:
        w1._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("input", influence=1.0, params={"stream": "ai", "scope": ai.SCOPE_ALL})
        ]
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        mods = w2._views[OUTPUT_VIEW].modifiers
        assert mods[0].type == "input"
        assert mods[0].params["scope"] == ai.SCOPE_ALL
    finally:
        w2._worker.stop()


def test_restore_ui_config_view_label(qapp, tmp_path):
    """A second window restores the renamed view label."""
    from synclip.ui.main_window import OUTPUT_VIEW
    w1 = _make_window(tmp_path)
    try:
        w1._views[OUTPUT_VIEW].label = "Hero Cam"
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        assert w2._views[OUTPUT_VIEW].label == "Hero Cam"
    finally:
        w2._worker.stop()


def test_restore_ui_config_pose_axes(qapp, tmp_path):
    """Head-pose axis states (pose_filter modifier) are saved and restored."""
    from synclip.modifiers import ModifierConfig
    from synclip.ui.main_window import OUTPUT_VIEW
    w1 = _make_window(tmp_path)
    try:
        w1._views[OUTPUT_VIEW].modifiers = [
            ModifierConfig("pose_filter", params={"rot": [True, False, True],
                                                  "pos": [True, True, False],
                                                  "neck_anchor": 0.0})
        ]
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        pose = w2._views[OUTPUT_VIEW].modifiers[0]
        assert pose.type == "pose_filter"
        assert pose.params["rot"] == [True, False, True]
        assert pose.params["pos"] == [True, True, False]
    finally:
        w2._worker.stop()


def test_missing_ui_config_does_not_crash(qapp, tmp_path):
    """Starting with no ui.config file is silent - defaults are used."""
    assert not os.path.exists(str(tmp_path / "ui.config"))
    w = _make_window(tmp_path)
    try:
        # A fresh view defaults to a single pose_filter modifier.
        mods = w._views[0].modifiers
        assert [m.type for m in mods] == ["input", "pose_filter"]
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# New: broadcast, camera_mode, selected_file, splitter sizes
# ---------------------------------------------------------------------------

def test_save_ui_config_writes_broadcast(qapp, tmp_path):
    """Broadcast toggle state is written to ui.config."""
    from synclip import ui_config as cfg
    w = _make_window(tmp_path)
    try:
        w._broadcast_check.setChecked(False)
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        assert data.get("broadcast") is False
    finally:
        w._worker.stop()


def test_restore_ui_config_broadcast(qapp, tmp_path):
    """A second window restores the saved broadcast toggle state."""
    w1 = _make_window(tmp_path)
    try:
        w1._broadcast_check.setChecked(False)
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        assert w2._broadcast_check.isChecked() is False
    finally:
        w2._worker.stop()


def test_save_ui_config_writes_camera_mode(qapp, tmp_path):
    """Camera mode (w, h, fps) is written to ui.config."""
    from synclip import ui_config as cfg
    w = _make_window(tmp_path)
    try:
        w._current_mode = (1280, 720, 30)
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        assert data.get("camera_mode") == [1280, 720, 30]
    finally:
        w._worker.stop()


def test_restore_ui_config_camera_mode(qapp, tmp_path):
    """A second window restores the saved camera mode."""
    w1 = _make_window(tmp_path)
    try:
        w1._current_mode = (1920, 1080, 60)
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        assert w2._current_mode == (1920, 1080, 60)
    finally:
        w2._worker.stop()


def test_save_ui_config_writes_selected_file(qapp, tmp_path):
    """File browser selection is written to ui.config."""
    from synclip import ui_config as cfg
    audio = str(tmp_path / "demo.wav")
    open(audio, "wb").close()
    w = _make_window(tmp_path)
    try:
        w._file_browser.select_file(audio)
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        assert data.get("selected_file") == audio
    finally:
        w._worker.stop()


def test_restore_ui_config_selected_file(qapp, tmp_path):
    """A second window restores the file browser highlight."""
    audio = str(tmp_path / "demo.wav")
    open(audio, "wb").close()
    w1 = _make_window(tmp_path)
    try:
        w1._file_browser.select_file(audio)
        w1._save_ui_config()
    finally:
        w1._worker.stop()

    w2 = _make_window(tmp_path)
    try:
        # FileBrowser.current_media_path() returns what is currently highlighted.
        assert w2._file_browser.current_media_path() == audio
    finally:
        w2._worker.stop()


def test_save_ui_config_writes_splitter_sizes(qapp, tmp_path):
    """Splitter sizes are written to ui.config (whatever the current sizes are)."""
    from synclip import ui_config as cfg
    w = _make_window(tmp_path)
    try:
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        # Sizes are present and are lists of integers - exact values depend on
        # the display environment so we only check structure.
        h = data.get("h_splitter_sizes")
        v = data.get("v_splitter_sizes")
        assert isinstance(h, list) and len(h) == 2
        assert isinstance(v, list) and len(v) == 2
    finally:
        w._worker.stop()


def test_camera_mode_none_saves_as_null(qapp, tmp_path):
    """When no camera mode has been set, camera_mode is null in ui.config."""
    from synclip import ui_config as cfg
    w = _make_window(tmp_path)
    try:
        w._current_mode = None
        w._save_ui_config()
        data = cfg.load(str(tmp_path))
        assert data.get("camera_mode") is None
    finally:
        w._worker.stop()


def test_file_browser_select_file_highlights(qapp, tmp_path):
    """FileBrowser.select_file() updates current_media_path() without emitting file_selected."""
    audio = str(tmp_path / "song.ogg")
    open(audio, "wb").close()
    w = _make_window(tmp_path)
    try:
        emitted = []
        w._file_browser.file_selected.connect(emitted.append)
        w._file_browser.select_file(audio)
        assert w._file_browser.current_media_path() == audio
        assert emitted == []  # must not emit the signal
    finally:
        w._worker.stop()


# ---------------------------------------------------------------------------
# Per-take capture settings (all views)
# ---------------------------------------------------------------------------

def test_capture_settings_roundtrip_all_views(qapp, tmp_path):
    """gather/apply round-trips the modifier stack of every view, not just output."""
    from synclip.modifiers import ModifierConfig
    w = _make_window(tmp_path)
    try:
        # Give each view a distinct stack.
        for i in range(len(w._views)):
            w._views[i].modifiers = [
                ModifierConfig("smooth", influence=0.1 * (i + 1))
            ]
        settings = w._gather_capture_settings()
        assert len(settings["view_modifiers"]) == len(w._views)

        # Mutate, then restore from the captured settings.
        for cfg in w._views:
            cfg.modifiers = []
        w._apply_capture_settings(settings)
        for i in range(len(w._views)):
            mods = w._views[i].modifiers
            assert len(mods) == 1 and mods[0].type == "smooth"
            assert abs(mods[0].influence - 0.1 * (i + 1)) < 1e-6
    finally:
        w._worker.stop()


def test_capture_settings_legacy_output_only(qapp, tmp_path):
    """The old output-only {'modifiers': [...]} format still restores."""
    from synclip.ui.main_window import OUTPUT_VIEW
    w = _make_window(tmp_path)
    try:
        w._views[OUTPUT_VIEW].modifiers = []
        w._apply_capture_settings(
            {"modifiers": [{"type": "smooth", "influence": 0.42, "params": {}}]}
        )
        mods = w._views[OUTPUT_VIEW].modifiers
        assert len(mods) == 1 and mods[0].type == "smooth"
        assert abs(mods[0].influence - 0.42) < 1e-6
    finally:
        w._worker.stop()


def test_select_take_does_not_emit(qapp, tmp_path):
    """TakesPanel.select_take highlights a row without emitting take_selected."""
    w = _make_window(tmp_path)
    try:
        panel = w._takes_panel
        panel.load_takes({
            "takes": [{"take_id": "t1", "name": "One"},
                      {"take_id": "t2", "name": "Two"}],
            "default_take": "t1",
        })
        emitted = []
        panel.take_selected.connect(emitted.append)
        panel.select_take("t2")
        assert panel.current_take_id() == "t2"
        assert emitted == []
    finally:
        w._worker.stop()
