"""``python -m capmo`` argument handling: the stream settings reach the UI."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ..core.presets import DEFAULT_PRESET, preset
from ..main import _build_parser, _resolve_stream, _window_kwargs


def _args(*argv):
    return _build_parser().parse_args(list(argv))


def test_defaults_are_the_default_preset():
    args = _args()
    assert _resolve_stream(args) == preset(DEFAULT_PRESET)
    assert (args.width, args.height, args.fps) == preset(DEFAULT_PRESET)


def test_explicit_values_and_preset_win():
    assert _resolve_stream(_args("--fps", "30")) == (1280, 720, 30)
    assert _resolve_stream(_args("--width", "640", "--height", "480")) == (640, 480, preset(DEFAULT_PRESET)[2])
    assert _resolve_stream(_args("--preset", "1080p30", "--fps", "60")) == (1920, 1080, 30)


def test_ui_window_gets_the_stream_settings(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ..ui.main_window import CapmoMainWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    kw = _window_kwargs(_args("--port", "0", "--host", "127.0.0.1", "--record", str(tmp_path),
                              "--preset", "1080p30"))
    assert (kw["width"], kw["height"], kw["fps"]) == (1920, 1080, 30)
    win = CapmoMainWindow(**kw)
    try:
        assert win._stream == (1920, 1080, 30)
        assert win._quality.currentText() == "1920×1080 · 30 fps"
    finally:
        win.close()
    kw = _window_kwargs(_args("--port", "0", "--host", "127.0.0.1", "--fps", "30"))
    win = CapmoMainWindow(**kw)
    try:
        assert win._stream == (1280, 720, 30)
    finally:
        win.close()
