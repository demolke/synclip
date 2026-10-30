#!/usr/bin/env python3
"""
capmo entry point.

Default launches the Qt UI (a dynamic camera grid). ``--headless`` runs just the
capture server + recorder with a periodic status print, which is enough to
exercise real phones (negotiate clocks, record raw streams) without a display.

    python -m capmo --headless --record ./captures --start
    python -m capmo.tools.record --out ./captures --start   # same, + start/stop control
    python -m capmo --record ./captures  # Qt UI: live grid, record, board check, process, view
"""

from __future__ import annotations

import argparse
import os
import sys

# Support both `python -m capmo` (run as a module) and `python main.py` /
# `python capmo/main.py` (run as a script). As a script there is no parent
# package, so the relative imports below would fail; put the project root on
# sys.path and set __package__ *before* they resolve.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "capmo"


from .core.presets import DEFAULT_PRESET, PRESETS, preset as _preset

_DEFAULT_SIZE = _preset(DEFAULT_PRESET)           # (width, height, fps)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="capmo multi-phone capture server")
    p.add_argument("--host", default="0.0.0.0",
                   help="Interface to bind (default: 0.0.0.0, all interfaces)")
    p.add_argument("--port", type=int, default=12345,
                   help="TCP port the phones dial (default: 12345)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose (DEBUG) logging: per-frame and raw-byte traces")
    p.add_argument("--record", metavar="DIR", default=None,
                   help="Record sessions under DIR (armed immediately in --headless)")
    p.add_argument("--start", action="store_true",
                   help="Send START/FRAMERATE to each phone as it connects "
                        "(--headless only: the UI always starts the phones)")
    p.add_argument("--preset", choices=[x[0] for x in PRESETS], default=None,
                   help=f"resolution@fps preset (overrides --width/--height/--fps; "
                        f"default {DEFAULT_PRESET})")
    p.add_argument("--width", type=int, default=None,
                   help=f"capture width (default {_DEFAULT_SIZE[0]})")
    p.add_argument("--height", type=int, default=None,
                   help=f"capture height (default {_DEFAULT_SIZE[1]})")
    p.add_argument("--fps", type=int, default=None,
                   help=f"capture frame rate (default {_DEFAULT_SIZE[2]})")
    p.add_argument("--headless", action="store_true",
                   help="No UI; run the server + status print only")
    # ChArUco calibration board (must match the displayed/printed target).
    p.add_argument("--board-squares", metavar="XxY", default=None,
                   help="ChArUco board size, e.g. 5x8 (default: 5x8)")
    p.add_argument("--board-square-len", type=float, default=None,
                   help="ChArUco square side length in metres (default: 0.04)")
    p.add_argument("--board-marker-len", type=float, default=None,
                   help="ChArUco marker side length in metres (default: 0.02)")
    return p


def _resolve_stream(args) -> tuple[int, int, int]:
    """``(width, height, fps)``: --preset, else the explicit --width/--height/--fps,
    each falling back to :data:`DEFAULT_PRESET`. Stored back on *args* (the
    headless recorder reads them there)."""
    if args.preset:
        args.width, args.height, args.fps = _preset(args.preset)
    else:
        dw, dh, df = _DEFAULT_SIZE
        args.width = args.width or dw
        args.height = args.height or dh
        args.fps = args.fps or df
    return args.width, args.height, args.fps


def _window_kwargs(args) -> dict:
    """Constructor arguments of the UI window for the parsed *args*."""
    w, h, fps = _resolve_stream(args)
    return dict(host=args.host, port=args.port,
                record_dir=os.path.abspath(args.record or os.getcwd()),
                board_cfg=_board_from_args(args), width=w, height=h, fps=fps)


def _board_from_args(args):
    """Build a BoardConfig from argparse, falling back to defaults."""
    from .calibration import BoardConfig
    cfg = BoardConfig()
    if args.board_squares:
        sx, sy = args.board_squares.lower().split("x")
        cfg.squares_x, cfg.squares_y = int(sx), int(sy)
    if args.board_square_len is not None:
        cfg.square_len = args.board_square_len
    if args.board_marker_len is not None:
        cfg.marker_len = args.board_marker_len
    return cfg


def _run_headless(args) -> None:
    """Server + (optional) recorder + status print; delegates to
    :mod:`capmo.tools.record` (which adds start/stop controls)."""
    from .tools.record import run_from_args
    run_from_args(args, record_dir=os.path.abspath(args.record) if args.record else None,
                  arm=True)


def main() -> None:
    import logging
    args = _build_parser().parse_args()
    _resolve_stream(args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    # PyAV decodes the phones' H.264 -- without it there is no video, so fail
    # fast and clearly instead of running with blank, "waiting for video" tiles.
    from .decode import AVAILABLE as _AV_AVAILABLE
    if not _AV_AVAILABLE:
        sys.exit("[capmo] FATAL: PyAV (the H.264 decoder) is not installed, so "
                 "no video can be shown or processed.\n"
                 "        Install it with:  pip install av\n"
                 "        (see capmo/requirements.txt)")
    if args.headless or not _qt_available():
        if not args.headless:
            print("[capmo] PySide6 not available; falling back to --headless.",
                  file=sys.stderr)
        _run_headless(args)
        return
    # Qt UI path.
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import CapmoMainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("capmo")
    win = CapmoMainWindow(**_window_kwargs(args))
    # A close can be deferred (a running Process worker is cancelled first), so
    # quit when the window has really closed, not on its first close request.
    app.setQuitOnLastWindowClosed(False)
    win.closed.connect(app.quit)
    win.show()
    sys.exit(app.exec())


def _qt_available() -> bool:
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
