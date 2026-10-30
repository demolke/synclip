"""
Headless Qt smoke tests for the character playback view (QT_QPA_PLATFORM=offscreen).

Builds the real :class:`CharacterView`, loads a 2-frame rig-pose track derived
from two known swing poses, advances the timer once and asserts a pixmap was
produced with no exception. Also a smoke test for the main window's View button
with no recording -- it must degrade gracefully (status updated, no crash); the
View-after-processing path is covered in ``test_ui_calibration.py``. No GL, no
MediaPipe; rendering is the pure-numpy rasterizer.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication

from .. import animate, skinning
from ..ui.character_view import CharacterView
from ..ui.main_window import CapmoMainWindow

_RIG = os.path.join(os.path.dirname(__file__), "..", "data", "characters",
                    "superhero_male", "superhero_male.gltf")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def _axis_angle(axis, deg):
    a = math.radians(deg)
    x, y, z = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1.0 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def test_character_view_set_track_and_advance(app):
    skin = skinning.load_skin(_RIG)
    track = [{}, {"upperarm_l": _axis_angle((1, 0, 0), 40.0),
                  "lowerarm_l": _axis_angle((1, 0, 0), 60.0)}]
    view = CharacterView(width=200, height=260)
    view.set_track(skin, track)
    # First frame rendered on load (by the render thread; flush delivers it).
    assert view.flush()
    assert not view.current_pixmap().isNull()
    view.advance()                                   # step the timer's callback once
    assert view.flush()
    assert not view.current_pixmap().isNull()
    view.shutdown()


def test_character_view_empty_track_is_graceful(app):
    view = CharacterView()
    view.set_track(None, None)
    view.advance()                                   # must not raise
    assert view.current_pixmap().isNull()


def test_main_window_view_button_no_data_is_graceful(app, tmp_path):
    win = CapmoMainWindow(host="127.0.0.1", port=0, record_dir=str(tmp_path))
    try:
        win._view_btn.setChecked(True)               # nothing recorded / processed yet
        # Graceful: button pops back off, status says what is missing, no crash.
        assert not win._view_btn.isChecked()
        assert "no recorded session" in win._status.text().lower()
        assert not win.processing
    finally:
        win.close()


def test_playback_renders_off_the_gui_thread_and_skips_frames(app, monkeypatch):
    """A render slower than the play interval must not block the GUI thread:
    the timer callback returns at once, the render runs in the worker, and the
    frames it could not keep up with are skipped."""
    import threading
    import time
    skin = skinning.load_skin(_RIG)
    rendered = []
    orig = animate.render_character

    def slow_render(*a, **k):
        rendered.append(threading.get_ident())
        time.sleep(0.15)
        return orig(*a, **k)

    monkeypatch.setattr(animate, "render_character", slow_render)
    view = CharacterView(width=120, height=160)
    view.set_track(skin, [{}, {}, {}, {}, {}, {}])
    t0 = time.perf_counter()
    for _ in range(5):
        view.advance()
    assert time.perf_counter() - t0 < 0.1            # the GUI thread only requested renders
    assert view.flush()
    assert rendered and threading.get_ident() not in rendered
    assert len(rendered) < 6                         # superseded frames were dropped
    assert not view.current_pixmap().isNull()
    view.shutdown()


def test_set_track_resets_the_cached_camera_for_a_new_rig(app):
    import dataclasses
    skin = skinning.load_skin(_RIG)
    big = dataclasses.replace(skin, verts=skin.verts * 2.0)
    view = CharacterView(width=100, height=130)
    view.set_track(skin, [{}])
    h1 = view._base_camera()[1]
    view.set_track(big, [{}])
    assert view._base_camera()[1] == pytest.approx(2.0 * h1)   # not the stale cached rig
    view.set_track(None, None)
    assert view._cam is None
    view.shutdown()
