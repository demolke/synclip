#!/usr/bin/env python3
"""SynClip capture tool"""

from __future__ import annotations

import argparse
import os
import sys


def _build_dark_palette():
    """Return a dark QPalette for the application."""
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    dark   = QColor(30, 30, 30)
    mid    = QColor(50, 50, 50)
    light  = QColor(70, 70, 70)
    text   = QColor(210, 210, 210)
    bright = QColor(255, 255, 255)
    accent = QColor(60, 140, 220)

    palette.setColor(QPalette.ColorRole.Window,          dark)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            mid)
    palette.setColor(QPalette.ColorRole.AlternateBase,   dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     dark)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          mid)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.BrightText,      bright)
    palette.setColor(QPalette.ColorRole.Link,            accent)
    palette.setColor(QPalette.ColorRole.Highlight,       accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, bright)
    return palette


def _list_cameras() -> None:
    """Print available camera indices to stdout and exit."""
    import cv2

    print("Enumerating cameras...")
    found: list[int] = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    if found:
        for idx in found:
            print(f"  Camera {idx}")
    else:
        print("  No cameras found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SynClip capture tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.getcwd(),
        help="Root directory to browse for audio files (default: cwd)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices and exit (no UI)",
    )
    args = parser.parse_args()

    if args.list_cameras:
        _list_cameras()
        sys.exit(0)

    root_dir = os.path.abspath(args.directory)
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("SynClip Capture")
    app.setOrganizationName("minigltf")

    # Apply dark palette
    app.setPalette(_build_dark_palette())
    app.setStyle("Fusion")  # Fusion respects QPalette on all platforms

    from .ui.main_window import MainWindow

    window = MainWindow(root_dir=root_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    # Allow running this file directly (``python main.py``) as well as via
    # ``python -m synclip``. When run as a plain script there is no
    # parent package, so the relative imports inside main() would fail. We
    # bootstrap the package context by putting the package's parent directory
    # on sys.path and setting __package__ to the package (folder) name.
    if __package__ in (None, ""):
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(_pkg_dir))
        __package__ = os.path.basename(_pkg_dir)

    main()
