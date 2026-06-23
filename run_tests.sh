#!/usr/bin/env bash
# Run the synclip test suite.
#
#   - The pytest suite always runs (state machine, worker, filter, IPC, e2e,
#     headless MainWindow smoke/bridge).
#   - The Godot and Blender cross-implementation harnesses run automatically iff
#     those binaries are on PATH; otherwise pytest skips them with a notice.
#
# Usage:  tools/run_tests.sh [extra pytest args]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Headless Qt + silent audio so the GUI smoke tests need no display/speakers.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"

cd "$HERE"

# The harnesses honour the GODOT / BLENDER env vars first, then fall back to PATH.
if [ -n "${GODOT:-}" ] || command -v godot >/dev/null 2>&1 || command -v godot4 >/dev/null 2>&1; then
    echo "[run_tests] Godot available - viewer harness will run."
else
    echo "[run_tests] Godot not found (set GODOT) - viewer harness will be skipped."
fi
if [ -n "${BLENDER:-}" ] || command -v blender >/dev/null 2>&1; then
    echo "[run_tests] Blender available - addon harness will run."
else
    echo "[run_tests] Blender not found (set BLENDER) - addon harness will be skipped."
fi

exec python -m pytest synclip/tests/ "$@"
