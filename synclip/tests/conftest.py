"""
Shared pytest fixtures for the synclip test suite.

Patches hardware-probing code so tests never try to open cameras.
"""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest


class _FakeCapture:
    """cv2.VideoCapture stand-in that never touches OS camera/video backends."""

    def __init__(self, *args, **kwargs):
        pass

    def isOpened(self) -> bool:  # noqa: N802
        return False

    def read(self):
        return False, None

    def get(self, prop):
        return 0.0

    def set(self, prop, value):
        pass

    def release(self):
        pass


@pytest.fixture(autouse=True)
def _no_camera_probe(monkeypatch):
    """Prevent any cv2.VideoCapture call from touching hardware during tests."""
    try:
        import cv2
        monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    except ImportError:
        pass
    try:
        from synclip.capture_worker import CaptureWorker
        monkeypatch.setattr(CaptureWorker, "list_cameras", staticmethod(lambda: []))
    except ImportError:
        pass
