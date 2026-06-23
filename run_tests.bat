@echo off
REM Run the synclip test suite on Windows.
REM
REM   - The pytest suite always runs (state machine, worker, filter, IPC, e2e,
REM     headless MainWindow smoke/bridge).
REM   - The Godot and Blender cross-implementation harnesses run iff their
REM     binaries are available. Point at them with the GODOT and BLENDER
REM     environment variables, e.g.:
REM         set GODOT=C:\tools\Godot.exe
REM         set BLENDER=C:\Program Files\Blender Foundation\Blender 5.1\blender.exe
REM         tools\run_tests.bat
REM     If unset, the harnesses fall back to PATH, and skip if not found.
REM
REM Usage:  tools\run_tests.bat [extra pytest args]
setlocal

set "HERE=%~dp0"

REM Headless Qt + silent audio so the GUI smoke tests need no display/speakers.
if not defined QT_QPA_PLATFORM set "QT_QPA_PLATFORM=offscreen"
if not defined SDL_AUDIODRIVER set "SDL_AUDIODRIVER=dummy"

cd /d "%HERE%"

if defined GODOT (
    echo [run_tests] Using GODOT=%GODOT%
) else (
    where godot >nul 2>nul && (
        echo [run_tests] Godot found on PATH - viewer harness will run.
    ) || (
        echo [run_tests] Godot not found ^(set GODOT^) - viewer harness will be skipped.
    )
)

if defined BLENDER (
    echo [run_tests] Using BLENDER=%BLENDER%
) else (
    where blender >nul 2>nul && (
        echo [run_tests] Blender found on PATH - addon harness will run.
    ) || (
        echo [run_tests] Blender not found ^(set BLENDER^) - addon harness will be skipped.
    )
)

python -m pytest synclip\tests\ %*
endlocal
