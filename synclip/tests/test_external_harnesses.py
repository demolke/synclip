"""
T-10/T-11 (Godot) and T-12/T-13 (Blender) cross-implementation harnesses.

These launch the real Godot / Blender binaries on their headless test scripts,
feeding a Python-packed frame and checking the other implementation decodes the
exact same wire format. Skipped (not failed) when the binary isn't on PATH, so
the suite still runs on a machine without them.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.abspath(os.path.join(_HERE, "..", ".."))

_FRAME = struct.Struct("<I d 52f 3f 3f")
MODE_LIVE = 0xAF0002


def _godot_bin() -> str | None:
    # Prefer an explicit GODOT env var (set by run_tests.bat on Windows), then
    # fall back to PATH.
    env = os.environ.get("GODOT")
    if env and (os.path.isfile(env) or shutil.which(env)):
        return env
    return shutil.which("godot") or shutil.which("godot4")


def _blender_bin() -> str | None:
    env = os.environ.get("BLENDER")
    if env and (os.path.isfile(env) or shutil.which(env)):
        return env
    return shutil.which("blender")


# ---------------------------------------------------------------------------
# Godot (T-10): decode a Python-packed frame with the GDScript offsets
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_godot_bin() is None, reason="godot binary not on PATH")
def test_godot_frame_parse():
    godot = _godot_bin()
    script = os.path.join(_TOOLS, "godot", "tests", "frame_parse_test.gd")
    assert os.path.exists(script), script

    raw = [i / 52.0 for i in range(52)]
    rot = [1.0, 2.0, 3.0]
    pos = [4.0, 5.0, 6.0]
    data = _FRAME.pack(MODE_LIVE, 987.6, *raw, *rot, *pos)

    in_path = tempfile.mktemp(suffix=".bin")
    out_path = tempfile.mktemp(suffix=".json")
    with open(in_path, "wb") as f:
        f.write(data)

    env = dict(os.environ, SYNCLIP_FRAME_FILE=in_path, SYNCLIP_OUT_FILE=out_path)
    proc = subprocess.run(
        [godot, "--headless", "--script", script],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"godot rc={proc.returncode}\n{proc.stderr[-800:]}"
    assert os.path.exists(out_path), "godot wrote no output"

    decoded = json.load(open(out_path))
    assert decoded["magic"] == MODE_LIVE
    assert abs(decoded["audio_pos"] - 987.6) < 1e-3
    assert all(abs(a - b) < 1e-5 for a, b in zip(decoded["blendshapes"], raw))
    assert all(abs(a - b) < 1e-4 for a, b in zip(decoded["rot"], rot))
    assert all(abs(a - b) < 1e-4 for a, b in zip(decoded["pos"], pos))


# ---------------------------------------------------------------------------
# Blender (T-12/T-13): build 52 ARKit shape keys, map + parse a frame
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_blender_bin() is None, reason="blender binary not on PATH")
def test_blender_mapping_and_parse():
    blender = _blender_bin()
    script = os.path.join(_TOOLS, "blender", "tests", "headless_test.py")
    assert os.path.exists(script), script

    out_path = tempfile.mktemp(suffix=".json")
    env = dict(os.environ, SYNCLIP_OUT=out_path)
    proc = subprocess.run(
        [blender, "--background", "--python", script],
        env=env, capture_output=True, text=True, timeout=240,
    )
    assert os.path.exists(out_path), (
        f"blender wrote no output (rc={proc.returncode})\n{proc.stdout[-800:]}"
    )
    result = json.load(open(out_path))
    assert result["ok"], result.get("errors")
    assert result["mapped_count"] == 52
