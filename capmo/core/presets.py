"""Capture quality presets shared by the UI and the CLI recorder."""

from __future__ import annotations

# (name, label, width, height, fps) -- same choices as the UI's dropdown.
PRESETS = [
    ("480p30", "640×480 · 30 fps", 640, 480, 30),
    ("720p30", "1280×720 · 30 fps", 1280, 720, 30),
    ("720p60", "1280×720 · 60 fps", 1280, 720, 60),
    ("1080p30", "1920×1080 · 30 fps", 1920, 1080, 30),
    ("1080p60", "1920×1080 · 60 fps", 1920, 1080, 60),
]
DEFAULT_PRESET = "720p60"


def preset(name: str) -> tuple[int, int, int]:
    """``(width, height, fps)`` of a preset name (e.g. ``720p60``)."""
    for n, _, w, h, f in PRESETS:
        if n == name.lower():
            return w, h, f
    raise KeyError(f"unknown preset {name!r}; choose from {', '.join(p[0] for p in PRESETS)}")
