"""
WebcamView - displays webcam frame with optional face landmark overlay.
"""

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

# State indicator colours (BGR for OpenCV)
_STATE_COLOURS = {
    "LIVE":   (0,   200,  0),
    "REC":    (0,   0,   220),
    "REVIEW": (200, 0,    0),
}
_STATE_LABELS = {
    "LIVE":   "LIVE",
    "REC":    "REC",
    "REVIEW": "REVIEW",
}


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    """Convert a BGR numpy frame to QPixmap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class WebcamView(QWidget):
    """Displays a webcam frame with an optional landmark overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setScaledContents(True)
        self._label.setMinimumSize(320, 240)
        self._label.setStyleSheet("background: #111;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self.setLayout(layout)

        self._show_overlay: bool = True
        self._state_key: str = "LIVE"  # "LIVE", "REC", "REVIEW"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state_indicator(self, state_key: str) -> None:
        """Set the state label drawn on frames.  state_key: 'LIVE' | 'REC' | 'REVIEW'"""
        self._state_key = state_key

    def set_overlay_visible(self, visible: bool) -> None:
        self._show_overlay = visible

    def update_frame(
        self,
        bgr_frame: np.ndarray,
        landmarks_raw=None,
        fps: float | None = None,
    ) -> None:
        """Update the display with a live frame, optionally drawing landmarks."""
        frame = bgr_frame.copy()

        if self._show_overlay and landmarks_raw is not None:
            self._draw_landmarks(frame, landmarks_raw)

        self._draw_state_indicator(frame)
        if fps is not None:
            self._draw_fps(frame, fps)
        self._label.setPixmap(_bgr_to_pixmap(frame))

    def set_playback_frame(self, bgr_frame: np.ndarray) -> None:
        """Show a static frame (for REVIEW state)."""
        frame = bgr_frame.copy()
        self._draw_state_indicator(frame)
        self._label.setPixmap(_bgr_to_pixmap(frame))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_landmarks(self, frame: np.ndarray, landmarks_raw) -> None:
        """Draw face landmarks onto *frame* in-place.

        The dot radius scales with the frame size so the mesh stays visible at
        any resolution -- a fixed 1px dot is invisible on a 1080p/4K webcam
        frame once it's scaled down into the widget.
        """
        h, w = frame.shape[:2]
        try:
            face_landmarks = landmarks_raw.face_landmarks
            if not face_landmarks:
                return
            # ~1px at 360p, growing with resolution (2px at 720p, 3px at 1080p).
            radius = max(1, int(round(min(h, w) / 360.0)))
            overlay = frame.copy()
            for lm in face_landmarks[0]:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(overlay, (x, y), radius, (0, 230, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        except (AttributeError, IndexError):
            pass

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw the current capture FPS in the top-right corner."""
        h, w = frame.shape[:2]
        text = f"{fps:4.1f} FPS"
        text_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness
        )
        pad = 4
        x0 = w - tw - pad * 2 - 8
        y0 = 8
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + tw + pad * 2, y0 + th + pad * 2 + baseline),
            (40, 40, 40),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x0 + pad, y0 + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 230, 230),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_state_indicator(self, frame: np.ndarray) -> None:
        """Draw a coloured state badge in the top-left corner."""
        colour = _STATE_COLOURS.get(self._state_key, (200, 200, 200))
        label_text = _STATE_LABELS.get(self._state_key, self._state_key)

        # Draw filled rectangle background
        text_scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness
        )
        pad = 4
        x0, y0 = 8, 8
        cv2.rectangle(
            frame,
            (x0, y0),
            (x0 + tw + pad * 2, y0 + th + pad * 2 + baseline),
            colour,
            -1,
        )
        # Draw dark text on the badge
        cv2.putText(
            frame,
            label_text,
            (x0 + pad, y0 + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (10, 10, 10),
            thickness,
            cv2.LINE_AA,
        )
