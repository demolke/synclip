"""
Persistent per-project UI configuration.

Saved as ``ui.config`` (JSON) in the working directory so each project
directory remembers the video/audio inputs, the current take, and all
per-view mix settings (model paths, AI options, curves, smoothing, head
pose axes).

The file is read on startup and written every time a relevant setting
changes.  Nothing is fatal if it is missing or corrupt -- defaults are
used silently.
"""

from __future__ import annotations

import json
import os


_FILENAME = "ui.config"


def config_path(root_dir: str) -> str:
    return os.path.join(root_dir, _FILENAME)


def load(root_dir: str) -> dict:
    path = config_path(root_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save(root_dir: str, data: dict) -> None:
    path = config_path(root_dir)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"[ui_config] save failed: {exc}")
